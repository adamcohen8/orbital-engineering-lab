"""Global Earth conical-sensor coverage from deterministic OEL state evidence."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from sim.analysis.event_refinement import (
    availability_intervals,
    refine_availability_transitions,
)
from sim.analysis.healpix import (
    HEALPIX_GRID_ID,
    WGS84_AUTHALIC_RADIUS_KM,
    WGS84_SURFACE_AREA_KM2,
    cached_healpix_wgs84_centers,
    healpix_npix,
)
from sim.analysis.observer_target_geometry import evaluate_surface_targets_ecef
from sim.dynamics.orbit.frames import FrameContext, eci_to_ecef_rotation_context
from sim.numeric_backend import normalize_numeric_backend
from sim.utils.geodesy import WGS84_A_KM, WGS84_B_KM
from sim.utils.quaternion import quaternion_to_dcm_bn

GLOBAL_COVERAGE_CONTRACT_VERSION = "oel.global-earth-coverage-analysis.v0.2"
_SUPPORTED_ORDERS = frozenset(range(5, 9))
_ATTITUDE_SOURCE_KINDS = frozenset({"achieved", "replay", "analytic_ideal"})
_QUATERNION_NORM_TOLERANCE = 1.0e-10
_ANGULAR_TOLERANCE_RAD = 1.0e-12
_RANGE_TOLERANCE_KM = 1.0e-9


@dataclass(frozen=True)
class GlobalCoverageConfig:
    analysis_id: str
    source_asset_id: str
    state_provider_id: str
    attitude_source_kind: str
    attitude_provider_id: str
    sensor_id: str
    order: int
    half_angle_rad: float
    quat_body_from_sensor: tuple[float, float, float, float]
    max_range_km: float | None = None
    chunk_size: int = 8192
    max_working_memory_bytes: int = 512 * 1024 * 1024
    max_cell_time_comparisons: int = 300_000_000
    max_transition_refinement_evaluations: int = 5_000_000
    transition_time_tolerance_s: float | None = None
    transition_max_iterations: int | None = None
    numeric_backend: str = "rust"

    def __post_init__(self) -> None:
        for field_name in (
            "analysis_id",
            "source_asset_id",
            "state_provider_id",
            "attitude_provider_id",
            "sensor_id",
        ):
            if not str(getattr(self, field_name) or "").strip():
                raise ValueError(f"{field_name} must be a non-empty string.")
            object.__setattr__(self, field_name, str(getattr(self, field_name)).strip())
        source_kind = str(self.attitude_source_kind or "").strip().lower()
        if source_kind not in _ATTITUDE_SOURCE_KINDS:
            choices = ", ".join(sorted(_ATTITUDE_SOURCE_KINDS))
            raise ValueError(f"attitude_source_kind must be one of: {choices}.")
        object.__setattr__(self, "attitude_source_kind", source_kind)
        if isinstance(self.order, (bool, np.bool_)) or int(self.order) != self.order:
            raise ValueError("Global coverage order must be an integer.")
        if int(self.order) not in _SUPPORTED_ORDERS:
            raise ValueError("Global coverage v0.1 supports HEALPix orders 5 through 8.")
        object.__setattr__(self, "order", int(self.order))
        if not np.isfinite(float(self.half_angle_rad)) or not 0.0 < float(self.half_angle_rad) < 0.5 * np.pi:
            raise ValueError("half_angle_rad must be finite and strictly within (0, pi/2).")
        object.__setattr__(self, "half_angle_rad", float(self.half_angle_rad))
        mounting = _validated_quaternion(self.quat_body_from_sensor, "quat_body_from_sensor")
        object.__setattr__(self, "quat_body_from_sensor", tuple(float(value) for value in mounting))
        if self.max_range_km is not None and (
            not np.isfinite(float(self.max_range_km)) or float(self.max_range_km) <= 0.0
        ):
            raise ValueError("max_range_km must be positive and finite when provided.")
        if self.max_range_km is not None:
            object.__setattr__(self, "max_range_km", float(self.max_range_km))
        for field_name in (
            "chunk_size",
            "max_working_memory_bytes",
            "max_cell_time_comparisons",
            "max_transition_refinement_evaluations",
        ):
            value = getattr(self, field_name)
            try:
                numeric = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{field_name} must be a positive integer.") from exc
            if (
                isinstance(value, (bool, np.bool_))
                or not np.isfinite(numeric)
                or not numeric.is_integer()
                or numeric <= 0.0
            ):
                raise ValueError(f"{field_name} must be a positive integer.")
        object.__setattr__(self, "chunk_size", int(self.chunk_size))
        object.__setattr__(self, "max_working_memory_bytes", int(self.max_working_memory_bytes))
        object.__setattr__(self, "max_cell_time_comparisons", int(self.max_cell_time_comparisons))
        object.__setattr__(
            self,
            "max_transition_refinement_evaluations",
            int(self.max_transition_refinement_evaluations),
        )
        if (self.transition_time_tolerance_s is None) != (self.transition_max_iterations is None):
            raise ValueError("Coverage transition refinement tolerance and iteration limit must be declared together.")
        if self.transition_time_tolerance_s is not None:
            tolerance = float(self.transition_time_tolerance_s)
            iterations = self.transition_max_iterations
            if not np.isfinite(tolerance) or tolerance <= 0.0:
                raise ValueError("transition_time_tolerance_s must be positive and finite.")
            if isinstance(iterations, (bool, np.bool_)) or int(iterations) != iterations or int(iterations) <= 0:
                raise ValueError("transition_max_iterations must be a positive integer.")
            object.__setattr__(self, "transition_time_tolerance_s", tolerance)
            object.__setattr__(self, "transition_max_iterations", int(iterations))
        backend = normalize_numeric_backend(self.numeric_backend, error_message="numeric_backend must be python or rust.")
        object.__setattr__(self, "numeric_backend", backend)


@dataclass(frozen=True)
class SparseCoverageIntervals:
    cell_index: np.ndarray
    interval_offset: np.ndarray
    start_sample_index: np.ndarray
    end_sample_index_exclusive: np.ndarray

    @property
    def interval_count(self) -> int:
        return int(self.start_sample_index.size)


@dataclass(frozen=True)
class CoverageCellMetrics:
    cell_index: np.ndarray
    dwell_s: np.ndarray
    interval_count: np.ndarray
    observed_acquisition_count: np.ndarray
    max_complete_revisit_gap_s: np.ndarray
    prefix_boundary_gap_s: np.ndarray
    suffix_boundary_gap_s: np.ndarray
    start_censored: np.ndarray
    end_censored: np.ndarray
    intervals: SparseCoverageIntervals


@dataclass(frozen=True)
class GlobalCoverageResult:
    config: GlobalCoverageConfig
    frame_metadata: dict[str, Any]
    times_s: np.ndarray
    covered_cell_count: np.ndarray
    instantaneous_covered_fraction: np.ndarray
    cell_geodetic_latitude_deg: np.ndarray
    cell_longitude_deg: np.ndarray
    cell_metrics: CoverageCellMetrics
    summary: dict[str, Any]
    resource_estimate: dict[str, int]
    input_evidence_sha256: str
    interval_semantic_sha256: str
    refined_intervals: tuple[CoverageAvailabilityInterval, ...] = ()
    refined_transitions: tuple[CoverageTransitionEvidence, ...] = ()
    refinement_provider_id: str | None = None


@dataclass(frozen=True)
class CoverageAvailabilityInterval:
    cell_index: int
    interval_index: int
    start_s: float
    end_s: float
    duration_s: float
    start_censored: bool
    end_censored: bool
    acquisition_disposition: str
    loss_disposition: str
    acquisition_reason: str
    loss_reason: str


@dataclass(frozen=True)
class CoverageTransitionEvidence:
    cell_index: int
    transition_kind: str
    time_s: float
    bracket_start_s: float
    bracket_end_s: float
    disposition: str
    iterations: int
    reason_before: str
    reason_after: str


CoverageAvailabilityEvaluator = Callable[[float, np.ndarray], tuple[np.ndarray, tuple[str, ...]]]


@dataclass(frozen=True)
class GlobalCoverageArtifacts:
    output_dir: Path
    manifest_json: Path
    summary_json: Path
    samples_csv: Path
    cells_csv: Path | None
    intervals_npz: Path
    fraction_plot_png: Path | None
    fraction_plot_quality_json: Path | None


def _validated_quaternion(values: Any, field_name: str) -> np.ndarray:
    quaternion = np.asarray(values, dtype=float).reshape(-1)
    if quaternion.size != 4 or not np.all(np.isfinite(quaternion)):
        raise ValueError(f"{field_name} must contain four finite scalar-first values.")
    norm = float(np.linalg.norm(quaternion))
    if abs(norm - 1.0) > _QUATERNION_NORM_TOLERANCE:
        raise ValueError(
            f"{field_name} must be normalized within {_QUATERNION_NORM_TOLERANCE:.1e}; "
            f"received norm {norm:.17g}."
        )
    return quaternion


def _validated_evidence(
    times_s: np.ndarray,
    positions_eci_km: np.ndarray,
    attitudes_quat_bn: np.ndarray,
    frame_context: FrameContext,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    times = np.asarray(times_s, dtype=float)
    positions = np.asarray(positions_eci_km, dtype=float)
    attitudes = np.asarray(attitudes_quat_bn, dtype=float)
    if times.ndim != 1 or times.size < 2 or not np.all(np.isfinite(times)):
        raise ValueError("times_s must contain at least two finite epochs.")
    if np.any(np.diff(times) <= 0.0):
        raise ValueError("times_s must be strictly increasing.")
    if positions.shape != (times.size, 3) or not np.all(np.isfinite(positions)):
        raise ValueError("positions_eci_km must be finite with shape (samples, 3).")
    if attitudes.shape != (times.size, 4) or not np.all(np.isfinite(attitudes)):
        raise ValueError("attitudes_quat_bn must be finite with shape (samples, 4).")
    norms = np.linalg.norm(attitudes, axis=1)
    invalid = np.flatnonzero(np.abs(norms - 1.0) > _QUATERNION_NORM_TOLERANCE)
    if invalid.size:
        index = int(invalid[0])
        raise ValueError(
            "attitudes_quat_bn must contain achieved or explicitly assumed normalized quaternions; "
            f"sample {index} has norm {float(norms[index]):.17g}."
        )
    if frame_context.jd_utc_start is None:
        raise ValueError("Global coverage v0.1 requires an absolute UTC epoch in FrameContext.jd_utc_start.")
    return times, positions, attitudes


def estimate_global_coverage_resources(
    config: GlobalCoverageConfig,
    sample_count: int,
) -> dict[str, int]:
    samples = int(sample_count)
    if samples < 2:
        raise ValueError("sample_count must be at least two.")
    npix = healpix_npix(config.order)
    chunk_cells = min(int(config.chunk_size), npix)
    dense_comparisons = npix * samples
    chunk_boolean_bytes = chunk_cells * samples
    chunk_geometry_bytes = chunk_cells * 11 * 8
    full_metric_bytes = npix * (8 * 8 + 2)
    input_state_bytes = samples * (3 + 3 + 4) * 8
    worst_case_interval_count = npix * ((samples + 1) // 2)
    # Chunk interval arrays and their final concatenation coexist briefly.
    worst_case_interval_working_bytes = worst_case_interval_count * 16 * 2
    estimated_peak_bytes = (
        chunk_boolean_bytes
        + chunk_geometry_bytes
        + full_metric_bytes
        + input_state_bytes
        + worst_case_interval_working_bytes
    )
    return {
        "sample_count": samples,
        "cell_count": npix,
        "cell_time_comparisons": dense_comparisons,
        "chunk_size": chunk_cells,
        "worst_case_interval_count": worst_case_interval_count,
        "worst_case_interval_working_bytes": worst_case_interval_working_bytes,
        "estimated_peak_working_bytes": estimated_peak_bytes,
        "max_working_memory_bytes": int(config.max_working_memory_bytes),
        "max_cell_time_comparisons": int(config.max_cell_time_comparisons),
        "max_transition_refinement_evaluations": int(
            config.max_transition_refinement_evaluations
        ),
    }


def _validated_sparse_coverage_inputs(covered_by_sample, times_s, cell_indices):
    mask = np.asarray(covered_by_sample, dtype=bool)
    times = np.asarray(times_s, dtype=float)
    if mask.ndim != 2:
        raise ValueError("covered_by_sample must have shape (samples, cells).")
    if times.ndim != 1 or times.size != mask.shape[0] or times.size < 2:
        raise ValueError("times_s must match the sample dimension and contain at least two epochs.")
    if not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0.0):
        raise ValueError("times_s must be finite and strictly increasing.")
    cell_count = mask.shape[1]
    if cell_indices is None:
        cells = np.arange(cell_count, dtype=np.int64)
    else:
        cells = np.asarray(cell_indices)
        if cells.shape != (cell_count,) or cells.dtype.kind not in {"i", "u"}:
            raise ValueError("cell_indices must be an integer vector matching the cell dimension.")
        cells = cells.astype(np.int64, copy=False)
        if cells.size > 1 and np.any(cells[1:] <= cells[:-1]):
            raise ValueError("cell_indices must be unique and strictly increasing.")

    return mask, times, cells


def summarize_sampled_coverage_mask(
    covered_by_sample: np.ndarray,
    times_s: np.ndarray,
    *,
    cell_indices: np.ndarray | None = None,
    numeric_backend: str = "rust",
) -> CoverageCellMetrics:
    """Convert a sampled boolean mask into frozen sparse interval semantics."""

    backend = normalize_numeric_backend(numeric_backend)
    mask, times, cells = _validated_sparse_coverage_inputs(covered_by_sample, times_s, cell_indices)
    if backend == "rust":
        from sim.rust_coverage_backend import summarize_sampled_mask

        result = summarize_sampled_mask(
            mask,
            times,
            cell_indices=cells,
        )
        return CoverageCellMetrics(
            cell_index=cells,
            dwell_s=result.dwell_s,
            interval_count=result.interval_count,
            observed_acquisition_count=result.observed_acquisition_count,
            max_complete_revisit_gap_s=result.max_complete_revisit_gap_s,
            prefix_boundary_gap_s=result.prefix_boundary_gap_s,
            suffix_boundary_gap_s=result.suffix_boundary_gap_s,
            start_censored=result.start_censored,
            end_censored=result.end_censored,
            intervals=SparseCoverageIntervals(
                cell_index=result.intervals.cell_index,
                interval_offset=result.intervals.interval_offset,
                start_sample_index=result.intervals.start_sample_index,
                end_sample_index_exclusive=result.intervals.end_sample_index_exclusive,
            ),
        )

    cell_count = mask.shape[1]
    padded = np.vstack((np.zeros((1, cell_count), dtype=bool), mask, np.zeros((1, cell_count), dtype=bool)))
    starts_at, starts_cell = np.nonzero((~padded[:-1]) & padded[1:])
    ends_at, ends_cell = np.nonzero(padded[:-1] & (~padded[1:]))
    start_order = np.lexsort((starts_at, starts_cell))
    end_order = np.lexsort((ends_at, ends_cell))
    starts_at = starts_at[start_order].astype(np.int64, copy=False)
    starts_cell = starts_cell[start_order].astype(np.int64, copy=False)
    ends_at = ends_at[end_order].astype(np.int64, copy=False)
    ends_cell = ends_cell[end_order].astype(np.int64, copy=False)
    if not np.array_equal(starts_cell, ends_cell):
        raise RuntimeError("Sampled coverage interval extraction lost cell alignment.")

    counts = np.bincount(starts_cell, minlength=cell_count).astype(np.int64, copy=False)
    covered_local = np.flatnonzero(counts).astype(np.int64, copy=False)
    offsets = np.zeros(covered_local.size + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(counts[covered_local], dtype=np.int64)
    intervals = SparseCoverageIntervals(
        cell_index=cells[covered_local].astype("<i8", copy=False),
        interval_offset=offsets.astype("<i8", copy=False),
        start_sample_index=starts_at.astype("<i8", copy=False),
        end_sample_index_exclusive=ends_at.astype("<i8", copy=False),
    )

    delta_t = np.diff(times)
    dwell = np.einsum("tc,t->c", mask[:-1].astype(float), delta_t, optimize=True)
    observed_acquisitions = counts - mask[0].astype(np.int64)
    start_censored = mask[0].copy()
    end_censored = mask[-1].copy()
    maximum_gap = np.full(cell_count, np.nan, dtype=float)
    if starts_at.size > 1:
        same_cell = starts_cell[1:] == starts_cell[:-1]
        if np.any(same_cell):
            gap_cells = starts_cell[1:][same_cell]
            gap_start = starts_at[1:][same_cell]
            gap_end = ends_at[:-1][same_cell]
            gaps = times[gap_start] - times[gap_end]
            maximum_gap[gap_cells] = -np.inf
            np.maximum.at(maximum_gap, gap_cells, gaps)

    duration = float(times[-1] - times[0])
    prefix_gap = np.full(cell_count, np.nan, dtype=float)
    suffix_gap = np.full(cell_count, np.nan, dtype=float)
    if covered_local.size:
        first_starts = starts_at[offsets[:-1]]
        last_ends = ends_at[offsets[1:] - 1]
        has_prefix = first_starts > 0
        has_suffix = last_ends < times.size
        prefix_gap[covered_local[has_prefix]] = times[first_starts[has_prefix]] - times[0]
        suffix_gap[covered_local[has_suffix]] = times[-1] - times[last_ends[has_suffix]]
    never_covered = counts == 0
    prefix_gap[never_covered] = duration
    suffix_gap[never_covered] = duration

    return CoverageCellMetrics(
        cell_index=cells,
        dwell_s=dwell,
        interval_count=counts,
        observed_acquisition_count=observed_acquisitions,
        max_complete_revisit_gap_s=maximum_gap,
        prefix_boundary_gap_s=prefix_gap,
        suffix_boundary_gap_s=suffix_gap,
        start_censored=start_censored,
        end_censored=end_censored,
        intervals=intervals,
    )


def _prepare_ecef_state(
    times: np.ndarray,
    positions_eci: np.ndarray,
    attitudes_quat_bn: np.ndarray,
    mounting_quaternion: np.ndarray,
    frame_context: FrameContext,
) -> tuple[np.ndarray, np.ndarray]:
    positions_ecef = np.empty_like(positions_eci)
    boresight_ecef = np.empty_like(positions_eci)
    sensor_to_body = quaternion_to_dcm_bn(mounting_quaternion)
    sensor_boresight = np.array([0.0, 0.0, 1.0], dtype=float)
    boresight_body = sensor_to_body @ sensor_boresight
    for sample_index, time_s in enumerate(times):
        rotation = eci_to_ecef_rotation_context(float(time_s), frame_context)
        body_from_eci = quaternion_to_dcm_bn(attitudes_quat_bn[sample_index])
        boresight_eci = body_from_eci.T @ boresight_body
        positions_ecef[sample_index] = rotation @ positions_eci[sample_index]
        boresight_ecef[sample_index] = rotation @ boresight_eci
    ellipsoid_level = (
        (positions_ecef[:, 0] / WGS84_A_KM) ** 2
        + (positions_ecef[:, 1] / WGS84_A_KM) ** 2
        + (positions_ecef[:, 2] / WGS84_B_KM) ** 2
    )
    invalid = np.flatnonzero(ellipsoid_level <= 1.0)
    if invalid.size:
        index = int(invalid[0])
        raise ValueError(f"Source spacecraft must be outside WGS84; sample {index} is on or inside the ellipsoid.")
    return positions_ecef, boresight_ecef


def _combine_chunk_metrics(
    chunks: list[CoverageCellMetrics],
) -> CoverageCellMetrics:
    cells = np.concatenate([chunk.cell_index for chunk in chunks])
    counts = np.concatenate([chunk.interval_count for chunk in chunks])
    sparse_cells = np.concatenate([chunk.intervals.cell_index for chunk in chunks])
    sparse_counts = np.concatenate(
        [np.diff(chunk.intervals.interval_offset) for chunk in chunks]
    ).astype(np.int64, copy=False)
    offsets = np.zeros(sparse_cells.size + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(sparse_counts, dtype=np.int64)
    intervals = SparseCoverageIntervals(
        cell_index=sparse_cells.astype("<i8", copy=False),
        interval_offset=offsets.astype("<i8", copy=False),
        start_sample_index=np.concatenate(
            [chunk.intervals.start_sample_index for chunk in chunks]
        ).astype("<i8", copy=False),
        end_sample_index_exclusive=np.concatenate(
            [chunk.intervals.end_sample_index_exclusive for chunk in chunks]
        ).astype("<i8", copy=False),
    )
    return CoverageCellMetrics(
        cell_index=cells,
        dwell_s=np.concatenate([chunk.dwell_s for chunk in chunks]),
        interval_count=counts,
        observed_acquisition_count=np.concatenate(
            [chunk.observed_acquisition_count for chunk in chunks]
        ),
        max_complete_revisit_gap_s=np.concatenate(
            [chunk.max_complete_revisit_gap_s for chunk in chunks]
        ),
        prefix_boundary_gap_s=np.concatenate(
            [chunk.prefix_boundary_gap_s for chunk in chunks]
        ),
        suffix_boundary_gap_s=np.concatenate(
            [chunk.suffix_boundary_gap_s for chunk in chunks]
        ),
        start_censored=np.concatenate([chunk.start_censored for chunk in chunks]),
        end_censored=np.concatenate([chunk.end_censored for chunk in chunks]),
        intervals=intervals,
    )


def _semantic_interval_hash(
    config: GlobalCoverageConfig,
    input_evidence_sha256: str,
    times_s: np.ndarray,
    intervals: SparseCoverageIntervals,
    refined_intervals: tuple[CoverageAvailabilityInterval, ...] = (),
    refinement_provider_id: str | None = None,
    refined_transitions: tuple[CoverageTransitionEvidence, ...] = (),
) -> str:
    identity = {
        "analysis_id": config.analysis_id,
        "attitude_provider_id": config.attitude_provider_id,
        "attitude_source_kind": config.attitude_source_kind,
        "contract_version": GLOBAL_COVERAGE_CONTRACT_VERSION,
        "grid_identity": HEALPIX_GRID_ID,
        "half_angle_rad": float(config.half_angle_rad),
        "input_evidence_sha256": input_evidence_sha256,
        "max_range_km": None if config.max_range_km is None else float(config.max_range_km),
        "order": int(config.order),
        "quat_body_from_sensor": [float(value) for value in config.quat_body_from_sensor],
        "sensor_id": config.sensor_id,
        "source_asset_id": config.source_asset_id,
        "state_provider_id": config.state_provider_id,
        "transition_time_tolerance_s": config.transition_time_tolerance_s,
        "transition_max_iterations": config.transition_max_iterations,
        "refinement_provider_id": refinement_provider_id,
    }
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    for name, values, dtype in (
        ("times_s", times_s, "<f8"),
        ("cell_index", intervals.cell_index, "<i8"),
        ("interval_offset", intervals.interval_offset, "<i8"),
        ("start_sample_index", intervals.start_sample_index, "<i8"),
        ("end_sample_index_exclusive", intervals.end_sample_index_exclusive, "<i8"),
    ):
        array = np.ascontiguousarray(np.asarray(values, dtype=dtype))
        digest.update(name.encode("utf-8"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    refined_rows = np.asarray(
        [[row.cell_index, row.interval_index, row.start_s, row.end_s] for row in refined_intervals],
        dtype="<f8",
    ).reshape(-1, 4)
    digest.update(b"refined_interval_rows")
    digest.update(refined_rows.dtype.str.encode("ascii"))
    digest.update(json.dumps(list(refined_rows.shape), separators=(",", ":")).encode("ascii"))
    digest.update(refined_rows.tobytes(order="C"))
    digest.update(b"refined_interval_evidence")
    digest.update(
        json.dumps(
            [asdict(row) for row in refined_intervals],
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    digest.update(b"refined_transition_evidence")
    digest.update(
        json.dumps(
            [asdict(row) for row in refined_transitions],
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    return digest.hexdigest()


def _input_evidence_hash(
    times_s: np.ndarray,
    positions_eci_km: np.ndarray,
    attitudes_quat_bn: np.ndarray,
    frame_metadata: dict[str, Any],
) -> str:
    digest = hashlib.sha256(
        json.dumps(
            {
                "frame": frame_metadata,
                "schema": "oel.global-coverage-input-evidence.v1",
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    )
    for name, values in (
        ("times_s", times_s),
        ("positions_eci_km", positions_eci_km),
        ("attitudes_quat_bn", attitudes_quat_bn),
    ):
        array = np.ascontiguousarray(np.asarray(values, dtype="<f8"))
        digest.update(name.encode("utf-8"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _summary(
    config: GlobalCoverageConfig,
    times: np.ndarray,
    covered_count: np.ndarray,
    metrics: CoverageCellMetrics,
) -> dict[str, Any]:
    npix = healpix_npix(config.order)
    cell_area = WGS84_SURFACE_AREA_KM2 / npix
    fraction = covered_count.astype(float) / npix
    duration = float(times[-1] - times[0])
    time_weighted_mean = float(np.dot(fraction[:-1], np.diff(times)) / duration)
    ever_covered = metrics.interval_count > 0
    finite_revisit = np.isfinite(metrics.max_complete_revisit_gap_s)
    revisit_values = metrics.max_complete_revisit_gap_s[finite_revisit]
    return {
        "contract_version": GLOBAL_COVERAGE_CONTRACT_VERSION,
        "analysis_id": config.analysis_id,
        "status": "complete",
        "domain_disposition": "global_earth",
        "source_asset_id": config.source_asset_id,
        "sensor_id": config.sensor_id,
        "attitude_source_kind": config.attitude_source_kind,
        "grid_identity": HEALPIX_GRID_ID,
        "order": int(config.order),
        "cell_count": npix,
        "cell_area_km2": float(cell_area),
        "wgs84_surface_area_km2": WGS84_SURFACE_AREA_KM2,
        "authalic_radius_km": WGS84_AUTHALIC_RADIUS_KM,
        "sample_count": int(times.size),
        "horizon_start_s": float(times[0]),
        "horizon_end_s": float(times[-1]),
        "horizon_duration_s": duration,
        "instantaneous_covered_fraction_min": float(np.min(fraction)),
        "instantaneous_covered_fraction_max": float(np.max(fraction)),
        "time_weighted_mean_covered_fraction": time_weighted_mean,
        "ever_covered_cell_count": int(np.count_nonzero(ever_covered)),
        "never_covered_cell_count": int(np.count_nonzero(~ever_covered)),
        "ever_covered_fraction": float(np.count_nonzero(ever_covered) / npix),
        "never_covered_fraction": float(np.count_nonzero(~ever_covered) / npix),
        "dwell_s": {
            "minimum": float(np.min(metrics.dwell_s)),
            "mean": float(np.mean(metrics.dwell_s)),
            "maximum": float(np.max(metrics.dwell_s)),
            "included_cell_count": npix,
        },
        "max_complete_revisit_gap_s": {
            "minimum": None if not revisit_values.size else float(np.min(revisit_values)),
            "mean": None if not revisit_values.size else float(np.mean(revisit_values)),
            "maximum": None if not revisit_values.size else float(np.max(revisit_values)),
            "evaluated_cell_count": int(revisit_values.size),
            "not_evaluated_cell_count": int(npix - revisit_values.size),
            "disposition": "not_evaluated" if not revisit_values.size else "evaluated",
        },
        "start_censored_cell_count": int(np.count_nonzero(metrics.start_censored)),
        "end_censored_cell_count": int(np.count_nonzero(metrics.end_censored)),
        "sampling_semantics": "instantaneous_cell_center_left_closed_right_open",
        "footprint_semantics": "healpix_cell_center",
        "claim_limits": [
            "No swept-footprint or between-sample access inference.",
            "No exact footprint-to-cell overlap-area claim.",
            "No terrain, atmosphere, weather, illumination, tasking, or payload-performance model.",
            "No communications, constellation, repeating-orbit, or steady-state claim.",
            "Decision-grade use requires a finer-cadence and next-order sensitivity comparison.",
        ],
    }


def evaluate_global_coverage(
    config: GlobalCoverageConfig,
    *,
    times_s: np.ndarray,
    positions_eci_km: np.ndarray,
    attitudes_quat_bn: np.ndarray,
    frame_context: FrameContext,
    evaluator_at_time: CoverageAvailabilityEvaluator | None = None,
    refinement_provider_id: str | None = None,
) -> GlobalCoverageResult:
    """Evaluate Phase 1 global conical-sensor coverage from supplied evidence."""

    times, positions, attitudes = _validated_evidence(
        times_s,
        positions_eci_km,
        attitudes_quat_bn,
        frame_context,
    )
    resources = estimate_global_coverage_resources(config, times.size)
    if resources["estimated_peak_working_bytes"] > resources["max_working_memory_bytes"]:
        raise ValueError(
            "Estimated global coverage working memory exceeds the configured limit: "
            f"{resources['estimated_peak_working_bytes']} > {resources['max_working_memory_bytes']} bytes."
        )
    if resources["cell_time_comparisons"] > resources["max_cell_time_comparisons"]:
        raise ValueError(
            "Global coverage cell-time comparisons exceed the configured limit: "
            f"{resources['cell_time_comparisons']} > {resources['max_cell_time_comparisons']}."
        )

    mounting = _validated_quaternion(config.quat_body_from_sensor, "quat_body_from_sensor")
    positions_ecef, boresight_ecef = _prepare_ecef_state(
        times,
        positions,
        attitudes,
        mounting,
        frame_context,
    )
    npix = healpix_npix(config.order)
    covered_count = np.zeros(times.size, dtype=np.int64)
    metric_chunks: list[CoverageCellMetrics] = []
    latitude_chunks: list[np.ndarray] = []
    longitude_chunks: list[np.ndarray] = []
    refined_intervals: list[CoverageAvailabilityInterval] = []
    refined_transitions: list[CoverageTransitionEvidence] = []
    refinement_evaluation_count = 0
    if evaluator_at_time is not None and config.transition_time_tolerance_s is None:
        raise ValueError("evaluator_at_time requires coverage transition refinement configuration.")
    if evaluator_at_time is None and refinement_provider_id is not None:
        raise ValueError("refinement_provider_id requires evaluator_at_time.")
    normalized_refinement_provider_id = (
        None
        if evaluator_at_time is None
        else str(refinement_provider_id or "caller_supplied_evaluator").strip()
    )
    for start in range(0, npix, int(config.chunk_size)):
        stop = min(start + int(config.chunk_size), npix)
        cells = np.arange(start, stop, dtype=np.int64)
        centers = cached_healpix_wgs84_centers(config.order, cells)
        latitude_chunks.append(np.rad2deg(centers.geodetic_latitude_rad))
        longitude_chunks.append(np.rad2deg(centers.longitude_rad))
        mask = None
        if config.numeric_backend == "rust":
            from sim.rust_coverage_backend import try_surface_targets_tile

            mask = try_surface_targets_tile(
                positions_ecef, centers.ecef_km, centers.outward_normal_ecef,
                boresight_ecef, config.half_angle_rad, config.max_range_km,
                _ANGULAR_TOLERANCE_RAD, _RANGE_TOLERANCE_KM,
            )
        if mask is None:
            mask = np.zeros((times.size, cells.size), dtype=bool)
            for sample_index in range(times.size):
                geometry = evaluate_surface_targets_ecef(
                    observer_ecef_km=positions_ecef[sample_index],
                    target_ecef_km=centers.ecef_km,
                    target_outward_normal_ecef=centers.outward_normal_ecef,
                    boresight_ecef=boresight_ecef[sample_index],
                    half_angle_rad=config.half_angle_rad,
                    max_range_km=config.max_range_km,
                    angular_tolerance_rad=_ANGULAR_TOLERANCE_RAD,
                    range_tolerance_km=_RANGE_TOLERANCE_KM,
                    numeric_backend=config.numeric_backend,
                )
                mask[sample_index] = geometry.available
        covered_count += np.count_nonzero(mask, axis=1)
        metric_chunks.append(
            summarize_sampled_coverage_mask(
                mask,
                times,
                cell_indices=cells,
                numeric_backend=config.numeric_backend,
            )
        )
        if evaluator_at_time is not None:
            sampled_transition_count = int(np.count_nonzero(mask[1:] != mask[:-1]))
            maximum_bracket_s = float(np.max(np.diff(times)))
            iterations_per_transition = min(
                int(config.transition_max_iterations),
                max(
                    0,
                    int(np.ceil(np.log2(maximum_bracket_s / float(config.transition_time_tolerance_s)))),
                ),
            )
            projected = refinement_evaluation_count + sampled_transition_count * iterations_per_transition
            if projected > int(config.max_transition_refinement_evaluations):
                raise ValueError(
                    "Coverage transition refinement exceeds the configured evaluation limit: "
                    f"{projected} > {config.max_transition_refinement_evaluations}."
                )
            for local_index in np.flatnonzero(np.any(mask, axis=0)):
                cell_index = int(cells[int(local_index)])
                reasons = tuple("available" if value else "not_covered" for value in mask[:, int(local_index)])

                def scalar_evaluator(time_s: float, *, selected_cell: int = cell_index) -> tuple[bool, str]:
                    available_at_time, reason_at_time = evaluator_at_time(
                        float(time_s), np.asarray([selected_cell], dtype=np.int64)
                    )
                    available_array = np.asarray(available_at_time, dtype=bool).reshape(-1)
                    reason_tuple = tuple(str(value) for value in reason_at_time)
                    if available_array.size != 1 or len(reason_tuple) != 1 or not reason_tuple[0]:
                        raise ValueError("Coverage refinement evaluator returned malformed evidence.")
                    return bool(available_array[0]), reason_tuple[0]

                transitions = refine_availability_transitions(
                    times,
                    mask[:, int(local_index)],
                    reasons,
                    evaluator_at_time=scalar_evaluator,
                    time_tolerance_s=config.transition_time_tolerance_s,
                    max_iterations=config.transition_max_iterations,
                    numeric_backend=config.numeric_backend,
                )
                refinement_evaluation_count += sum(value.iterations for value in transitions)
                refined_transitions.extend(
                    CoverageTransitionEvidence(cell_index=cell_index, **asdict(value))
                    for value in transitions
                )
                intervals = availability_intervals(
                    times, mask[:, int(local_index)], reasons, transitions=transitions
                )
                refined_intervals.extend(
                    CoverageAvailabilityInterval(cell_index=cell_index, **asdict(interval))
                    for interval in intervals
                )

    metrics = _combine_chunk_metrics(metric_chunks)
    summary = _summary(config, times, covered_count, metrics)
    summary["transition_refinement"] = {
        "enabled": evaluator_at_time is not None,
        "method": "provider_bisection" if evaluator_at_time is not None else "sample_bounded",
        "time_tolerance_s": config.transition_time_tolerance_s,
        "max_iterations": config.transition_max_iterations,
        "refined_interval_count": len(refined_intervals),
        "transition_count": len(refined_transitions),
        "provider_id": normalized_refinement_provider_id,
        "evaluator_call_count": refinement_evaluation_count,
    }
    frame_metadata = frame_context.metadata(sample_t_s=float(times[0]))
    input_hash = _input_evidence_hash(times, positions, attitudes, frame_metadata)
    semantic_hash = _semantic_interval_hash(
        config, input_hash, times, metrics.intervals, tuple(refined_intervals),
        normalized_refinement_provider_id,
        tuple(refined_transitions),
    )
    return GlobalCoverageResult(
        config=config,
        frame_metadata=frame_metadata,
        times_s=times,
        covered_cell_count=covered_count,
        instantaneous_covered_fraction=covered_count.astype(float) / npix,
        cell_geodetic_latitude_deg=np.concatenate(latitude_chunks),
        cell_longitude_deg=np.concatenate(longitude_chunks),
        cell_metrics=metrics,
        summary=summary,
        resource_estimate=resources,
        input_evidence_sha256=input_hash,
        interval_semantic_sha256=semantic_hash,
        refined_intervals=tuple(refined_intervals),
        refined_transitions=tuple(refined_transitions),
        refinement_provider_id=normalized_refinement_provider_id,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_global_coverage_artifacts(
    result: GlobalCoverageResult,
    output_dir: str | Path,
    *,
    include_cell_csv: bool = True,
    include_fraction_plot: bool = False,
    plot_scenario_name: str = "",
) -> GlobalCoverageArtifacts:
    """Write the frozen Phase 1 summary, sparse intervals, and review tables."""

    destination = Path(output_dir).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"Coverage output directory already exists: {destination}")
    destination.mkdir(parents=True)
    summary_path = destination / "coverage_summary.json"
    samples_path = destination / "coverage_samples.csv"
    cells_path = destination / "coverage_cells.csv" if include_cell_csv else None
    intervals_path = destination / "coverage_intervals.npz"
    plot_path = destination / "coverage_fraction.png" if include_fraction_plot else None
    plot_quality_path = destination / "coverage_fraction.quality.json" if include_fraction_plot else None
    manifest_path = destination / "coverage_analysis_manifest.json"

    _write_json(summary_path, result.summary)
    with samples_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "sample_index",
                "time_s",
                "covered_cell_count",
                "instantaneous_covered_fraction",
                "covered_area_km2",
            )
        )
        cell_area = WGS84_SURFACE_AREA_KM2 / healpix_npix(result.config.order)
        for sample_index, time_s in enumerate(result.times_s):
            writer.writerow(
                (
                    sample_index,
                    f"{float(time_s):.17g}",
                    int(result.covered_cell_count[sample_index]),
                    f"{float(result.instantaneous_covered_fraction[sample_index]):.17g}",
                    f"{float(result.covered_cell_count[sample_index] * cell_area):.17g}",
                )
            )

    if cells_path is not None:
        metrics = result.cell_metrics
        with cells_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(
                (
                    "cell_index",
                    "geodetic_latitude_deg",
                    "longitude_deg",
                    "sampled_dwell_s",
                    "interval_count",
                    "observed_acquisition_count",
                    "max_complete_revisit_gap_s",
                    "prefix_boundary_gap_s",
                    "suffix_boundary_gap_s",
                    "start_censored",
                    "end_censored",
                )
            )
            for index in range(metrics.cell_index.size):
                max_gap = metrics.max_complete_revisit_gap_s[index]
                prefix = metrics.prefix_boundary_gap_s[index]
                suffix = metrics.suffix_boundary_gap_s[index]
                writer.writerow(
                    (
                        int(metrics.cell_index[index]),
                        f"{float(result.cell_geodetic_latitude_deg[index]):.17g}",
                        f"{float(result.cell_longitude_deg[index]):.17g}",
                        f"{float(metrics.dwell_s[index]):.17g}",
                        int(metrics.interval_count[index]),
                        int(metrics.observed_acquisition_count[index]),
                        "" if not np.isfinite(max_gap) else f"{float(max_gap):.17g}",
                        "" if not np.isfinite(prefix) else f"{float(prefix):.17g}",
                        "" if not np.isfinite(suffix) else f"{float(suffix):.17g}",
                        str(bool(metrics.start_censored[index])).lower(),
                        str(bool(metrics.end_censored[index])).lower(),
                    )
                )

    intervals = result.cell_metrics.intervals
    refined = result.refined_intervals
    transitions = result.refined_transitions
    np.savez_compressed(
        intervals_path,
        cell_index=np.asarray(intervals.cell_index, dtype="<i8"),
        interval_offset=np.asarray(intervals.interval_offset, dtype="<i8"),
        start_sample_index=np.asarray(intervals.start_sample_index, dtype="<i8"),
        end_sample_index_exclusive=np.asarray(intervals.end_sample_index_exclusive, dtype="<i8"),
        refined_cell_index=np.asarray([row.cell_index for row in refined], dtype="<i8"),
        refined_interval_index=np.asarray([row.interval_index for row in refined], dtype="<i8"),
        refined_start_s=np.asarray([row.start_s for row in refined], dtype="<f8"),
        refined_end_s=np.asarray([row.end_s for row in refined], dtype="<f8"),
        refined_start_censored=np.asarray([row.start_censored for row in refined], dtype="|u1"),
        refined_end_censored=np.asarray([row.end_censored for row in refined], dtype="|u1"),
        refined_acquisition_disposition=np.asarray([row.acquisition_disposition for row in refined], dtype="U32"),
        refined_loss_disposition=np.asarray([row.loss_disposition for row in refined], dtype="U32"),
        refined_acquisition_reason=np.asarray([row.acquisition_reason for row in refined], dtype="U32"),
        refined_loss_reason=np.asarray([row.loss_reason for row in refined], dtype="U32"),
        transition_cell_index=np.asarray([row.cell_index for row in transitions], dtype="<i8"),
        transition_kind=np.asarray([row.transition_kind for row in transitions], dtype="U16"),
        transition_time_s=np.asarray([row.time_s for row in transitions], dtype="<f8"),
        transition_bracket_start_s=np.asarray([row.bracket_start_s for row in transitions], dtype="<f8"),
        transition_bracket_end_s=np.asarray([row.bracket_end_s for row in transitions], dtype="<f8"),
        transition_disposition=np.asarray([row.disposition for row in transitions], dtype="U32"),
        transition_iterations=np.asarray([row.iterations for row in transitions], dtype="<i8"),
        transition_reason_before=np.asarray([row.reason_before for row in transitions], dtype="U32"),
        transition_reason_after=np.asarray([row.reason_after for row in transitions], dtype="U32"),
    )

    config_record = asdict(result.config)
    config_record["quat_body_from_sensor"] = list(result.config.quat_body_from_sensor)
    artifacts: dict[str, dict[str, Any]] = {
        "coverage_summary.json": {"sha256": _sha256_file(summary_path)},
        "coverage_samples.csv": {"sha256": _sha256_file(samples_path)},
        "coverage_intervals.npz": {
            "container_sha256": _sha256_file(intervals_path),
            "semantic_sha256": result.interval_semantic_sha256,
        },
    }
    if cells_path is not None:
        artifacts["coverage_cells.csv"] = {"sha256": _sha256_file(cells_path)}
    if plot_path is not None:
        from sim.analysis.coverage_plotting import write_global_coverage_fraction_plot

        write_global_coverage_fraction_plot(
            result,
            plot_path,
            scenario_name=plot_scenario_name,
        )
        artifacts[plot_path.name] = {"sha256": _sha256_file(plot_path)}
    if plot_quality_path is not None:
        artifacts[plot_quality_path.name] = {"sha256": _sha256_file(plot_quality_path)}
    manifest = {
        "contract_version": GLOBAL_COVERAGE_CONTRACT_VERSION,
        "analysis_id": result.config.analysis_id,
        "status": "complete",
        "normalized_config": config_record,
        "frame": result.frame_metadata,
        "input_evidence_sha256": result.input_evidence_sha256,
        "resource_estimate": result.resource_estimate,
        "artifacts": artifacts,
        "scientific_semantics": {
            "earth_model": "wgs84_ellipsoid_v1",
            "grid_identity": HEALPIX_GRID_ID,
            "footprint": "axisymmetric_hard_cone",
            "cell_membership": "representative_center",
            "time": "instantaneous_left_closed_right_open",
            "attitude_source_kind": result.config.attitude_source_kind,
            "refinement_provider_id": result.refinement_provider_id,
        },
        "claim_limits": result.summary["claim_limits"],
    }
    _write_json(manifest_path, manifest)
    return GlobalCoverageArtifacts(
        output_dir=destination,
        manifest_json=manifest_path,
        summary_json=summary_path,
        samples_csv=samples_path,
        cells_csv=cells_path,
        intervals_npz=intervals_path,
        fraction_plot_png=plot_path,
        fraction_plot_quality_json=plot_quality_path,
    )


__all__ = [
    "GLOBAL_COVERAGE_CONTRACT_VERSION",
    "CoverageCellMetrics",
    "CoverageAvailabilityInterval",
    "CoverageTransitionEvidence",
    "GlobalCoverageArtifacts",
    "GlobalCoverageConfig",
    "GlobalCoverageResult",
    "SparseCoverageIntervals",
    "estimate_global_coverage_resources",
    "evaluate_global_coverage",
    "summarize_sampled_coverage_mask",
    "write_global_coverage_artifacts",
]
