"""RF-qualified HEALPix coverage using the authoritative Pro RF closure kernel."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from sim.analysis.global_coverage import (
    CoverageCellMetrics,
    _combine_chunk_metrics,
    _validated_evidence,
    _validated_quaternion,
    summarize_sampled_coverage_mask,
)
from sim.analysis.healpix import HEALPIX_GRID_ID, cached_healpix_wgs84_centers, healpix_npix
from sim.dynamics.orbit.frames import FrameContext, eci_to_ecef_rotation_context
from sim.numeric_backend import normalize_numeric_backend
from sim.pro_communications.contracts import LinkBudgetProfile, ResourceLimits
from sim.pro_communications.link_budget import evaluate_pro_rf_closure
from sim.utils.geodesy import WGS84_A_KM, WGS84_B_KM
from sim.utils.quaternion import quaternion_to_dcm_bn

PRO_COVERAGE_CONTRACT_VERSION = "oel.pro-communications-coverage.v0.1"
PRO_COVERAGE_REASON_NAMES = (
    "available",
    "earth_occulted",
    "terrain_blocked",
    "below_elevation_mask",
    "beyond_max_range",
    "tx_outside_pattern",
    "rx_outside_pattern",
    "atmospheric_model_outside_applicability",
    "negative_margin",
    "no_supported_waveform_mode",
)


def _required_text(value: Any, field_name: str) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        raise ValueError(f"{field_name} must be a non-empty string.")
    return normalized


@dataclass(frozen=True)
class ProCommunicationsCoverageConfig:
    analysis_id: str
    service_id: str
    source_asset_id: str
    state_provider_id: str
    attitude_source_kind: str
    attitude_provider_id: str | None
    quat_body_from_source_terminal: tuple[float, float, float, float]
    direction: str
    order: int
    link_profile: LinkBudgetProfile
    minimum_elevation_rad: float = np.deg2rad(5.0)
    maximum_range_km: float | None = None
    chunk_size: int = 8192
    numeric_backend: str = "rust"

    def __post_init__(self) -> None:
        for field_name in ("analysis_id", "service_id", "source_asset_id", "state_provider_id"):
            object.__setattr__(self, field_name, _required_text(getattr(self, field_name), field_name))
        if not isinstance(self.link_profile, LinkBudgetProfile):
            raise ValueError("link_profile must be a LinkBudgetProfile.")
        direction = str(self.direction or "").strip().lower()
        if direction not in {"spacecraft_to_earth", "earth_to_spacecraft"}:
            raise ValueError("direction must be spacecraft_to_earth or earth_to_spacecraft.")
        object.__setattr__(self, "direction", direction)
        source_antenna = (
            self.link_profile.transmitter.antenna
            if direction == "spacecraft_to_earth"
            else self.link_profile.receiver.antenna
        )
        source_kind = str(self.attitude_source_kind or "").strip().lower()
        if source_kind not in {"achieved", "replay", "analytic_ideal", "not_required"}:
            raise ValueError("attitude_source_kind must be achieved, replay, analytic_ideal, or not_required.")
        if source_kind == "not_required":
            if source_antenna.kind != "constant":
                raise ValueError("not_required attitude is valid only for a constant source antenna pattern.")
            if self.attitude_provider_id is not None:
                raise ValueError("attitude_provider_id must be absent when attitude is not required.")
        else:
            object.__setattr__(
                self,
                "attitude_provider_id",
                _required_text(self.attitude_provider_id, "attitude_provider_id"),
            )
        object.__setattr__(self, "attitude_source_kind", source_kind)
        mounting = _validated_quaternion(
            self.quat_body_from_source_terminal,
            "quat_body_from_source_terminal",
        )
        object.__setattr__(
            self,
            "quat_body_from_source_terminal",
            tuple(float(value) for value in mounting),
        )
        if (
            isinstance(self.order, (bool, np.bool_))
            or int(self.order) != self.order
            or int(self.order) not in range(5, 9)
        ):
            raise ValueError("Pro communications coverage supports integer HEALPix orders 5 through 8.")
        object.__setattr__(self, "order", int(self.order))
        minimum = float(self.minimum_elevation_rad)
        if not np.isfinite(minimum) or not -0.5 * np.pi <= minimum <= 0.5 * np.pi:
            raise ValueError("minimum_elevation_rad must be finite within [-pi/2, pi/2].")
        object.__setattr__(self, "minimum_elevation_rad", minimum)
        if self.maximum_range_km is not None:
            maximum = float(self.maximum_range_km)
            if not np.isfinite(maximum) or maximum <= 0.0:
                raise ValueError("maximum_range_km must be positive and finite.")
            object.__setattr__(self, "maximum_range_km", maximum)
        if (
            isinstance(self.chunk_size, (bool, np.bool_))
            or int(self.chunk_size) != self.chunk_size
            or int(self.chunk_size) <= 0
        ):
            raise ValueError("chunk_size must be a positive integer.")
        object.__setattr__(self, "chunk_size", int(self.chunk_size))
        backend = normalize_numeric_backend(self.numeric_backend, error_message="numeric_backend must be python or rust.")
        object.__setattr__(self, "numeric_backend", backend)


@dataclass(frozen=True)
class ProCommunicationsCoverageResult:
    config: ProCommunicationsCoverageConfig
    frame_metadata: dict[str, Any]
    times_s: np.ndarray
    covered_cell_count: np.ndarray
    instantaneous_covered_fraction: np.ndarray
    cell_geodetic_latitude_deg: np.ndarray
    cell_longitude_deg: np.ndarray
    cell_metrics: CoverageCellMetrics
    primary_reason_count: np.ndarray
    sample_margin_min_db: np.ndarray
    sample_margin_max_db: np.ndarray
    cell_best_margin_db: np.ndarray
    cell_worst_margin_db: np.ndarray
    summary: dict[str, Any]
    semantic_sha256: str

    @property
    def interval_semantic_sha256(self) -> str:
        """Expose the stable public coverage-product structural contract."""

        return self.semantic_sha256


def _prepare_source(
    config: ProCommunicationsCoverageConfig,
    times: np.ndarray,
    positions_eci: np.ndarray,
    attitudes: np.ndarray | None,
    frame_context: FrameContext,
) -> tuple[np.ndarray, np.ndarray]:
    positions_ecef = np.empty_like(positions_eci)
    terminal_from_ecef = np.empty((times.size, 3, 3), dtype=float)
    body_from_terminal = quaternion_to_dcm_bn(np.asarray(config.quat_body_from_source_terminal))
    terminal_from_body = body_from_terminal.T
    for index, time_s in enumerate(times):
        ecef_from_eci = eci_to_ecef_rotation_context(float(time_s), frame_context)
        positions_ecef[index] = ecef_from_eci @ positions_eci[index]
        if attitudes is None:
            terminal_from_ecef[index] = np.eye(3)
        else:
            body_from_eci = quaternion_to_dcm_bn(attitudes[index])
            terminal_from_ecef[index] = terminal_from_body @ body_from_eci @ ecef_from_eci.T
    return positions_ecef, terminal_from_ecef


def _scientific_config(config: ProCommunicationsCoverageConfig) -> dict[str, Any]:
    record = asdict(config)
    record.pop("chunk_size")
    record.pop("numeric_backend", None)
    record["link_profile_sha256"] = config.link_profile.semantic_sha256
    record.pop("link_profile")
    return record


def evaluate_pro_communications_coverage(
    config: ProCommunicationsCoverageConfig,
    *,
    times_s: np.ndarray,
    positions_eci_km: np.ndarray,
    attitudes_quat_bn: np.ndarray | None,
    frame_context: FrameContext,
    resource_limits: ResourceLimits | None = None,
) -> ProCommunicationsCoverageResult:
    """Evaluate service-qualified global cell-center coverage from propagated evidence."""

    limits = resource_limits or ResourceLimits()
    if config.attitude_source_kind == "not_required":
        if attitudes_quat_bn is not None:
            raise ValueError("attitudes_quat_bn must be absent when attitude is not required.")
        identity = np.tile((1.0, 0.0, 0.0, 0.0), (np.asarray(times_s).size, 1))
        times, positions, _ = _validated_evidence(times_s, positions_eci_km, identity, frame_context)
        attitudes = None
    else:
        if attitudes_quat_bn is None:
            raise ValueError("attitudes_quat_bn is required for the selected source antenna.")
        times, positions, attitudes = _validated_evidence(
            times_s,
            positions_eci_km,
            attitudes_quat_bn,
            frame_context,
        )
    npix = healpix_npix(config.order)
    limits.enforce(time_samples=times.size, cells=npix, links=1, terminals=2)
    positions_ecef, terminal_from_ecef = _prepare_source(config, times, positions, attitudes, frame_context)
    ellipsoid_level = (
        (positions_ecef[:, 0] / WGS84_A_KM) ** 2
        + (positions_ecef[:, 1] / WGS84_A_KM) ** 2
        + (positions_ecef[:, 2] / WGS84_B_KM) ** 2
    )
    invalid = np.flatnonzero(ellipsoid_level <= 1.0)
    if invalid.size:
        raise ValueError(f"Coverage source sample {int(invalid[0])} is on or inside WGS84.")

    reason_lookup = {reason: index for index, reason in enumerate(PRO_COVERAGE_REASON_NAMES)}
    covered_count = np.zeros(times.size, dtype=np.int64)
    reason_count = np.zeros((times.size, len(PRO_COVERAGE_REASON_NAMES)), dtype=np.int64)
    sample_margin_min = np.full(times.size, np.inf)
    sample_margin_max = np.full(times.size, -np.inf)
    metric_chunks: list[CoverageCellMetrics] = []
    latitude_chunks: list[np.ndarray] = []
    longitude_chunks: list[np.ndarray] = []
    best_chunks: list[np.ndarray] = []
    worst_chunks: list[np.ndarray] = []

    for start in range(0, npix, config.chunk_size):
        stop = min(start + config.chunk_size, npix)
        cells = np.arange(start, stop, dtype=np.int64)
        centers = cached_healpix_wgs84_centers(config.order, cells)
        latitude_chunks.append(np.rad2deg(centers.geodetic_latitude_rad))
        longitude_chunks.append(np.rad2deg(centers.longitude_rad))
        mask = np.zeros((times.size, cells.size), dtype=bool)
        best_margin = np.full(cells.size, -np.inf)
        worst_margin = np.full(cells.size, np.inf)
        for sample_index in range(times.size):
            delta = centers.ecef_km - positions_ecef[sample_index]
            ranges = np.linalg.norm(delta, axis=1)
            source_to_cell = delta / ranges[:, None]
            cell_to_source = -source_to_cell
            visible = (
                np.einsum(
                    "ij,ij->i",
                    centers.outward_normal_ecef,
                    positions_ecef[sample_index] - centers.ecef_km,
                )
                > 1.0e-9
            )
            elevation = np.arcsin(
                np.clip(np.einsum("ij,ij->i", centers.outward_normal_ecef, cell_to_source), -1.0, 1.0)
            )
            source_direction_terminal = source_to_cell @ terminal_from_ecef[sample_index].T
            source_off_axis = np.arccos(np.clip(source_direction_terminal[:, 2], -1.0, 1.0))
            earth_off_axis = np.arccos(
                np.clip(np.einsum("ij,ij->i", centers.outward_normal_ecef, cell_to_source), -1.0, 1.0)
            )
            if config.direction == "spacecraft_to_earth":
                tx_off_axis, rx_off_axis = source_off_axis, earth_off_axis
            else:
                tx_off_axis, rx_off_axis = earth_off_axis, source_off_axis
            rf = evaluate_pro_rf_closure(
                config.link_profile,
                range_km=ranges,
                elevation_rad=elevation,
                tx_off_axis_rad=tx_off_axis,
                rx_off_axis_rad=rx_off_axis,
                earth_clear=visible,
                minimum_elevation_rad=config.minimum_elevation_rad,
                maximum_range_km=config.maximum_range_km,
                numeric_backend=config.numeric_backend,
            )
            mask[sample_index] = rf.available
            codes = np.fromiter(
                (reason_lookup[str(value)] for value in rf.primary_reason),
                dtype=np.int64,
                count=cells.size,
            )
            reason_count[sample_index] += np.bincount(codes, minlength=len(PRO_COVERAGE_REASON_NAMES))
            finite_margin = rf.margin_db[np.isfinite(rf.margin_db)]
            if finite_margin.size:
                sample_margin_min[sample_index] = min(
                    sample_margin_min[sample_index],
                    float(np.min(finite_margin)),
                )
                sample_margin_max[sample_index] = max(
                    sample_margin_max[sample_index],
                    float(np.max(finite_margin)),
                )
            best_margin = np.fmax(best_margin, rf.margin_db)
            worst_margin = np.fmin(worst_margin, rf.margin_db)
        covered_count += np.count_nonzero(mask, axis=1)
        metric_chunks.append(
            summarize_sampled_coverage_mask(
                mask,
                times,
                cell_indices=cells,
                numeric_backend=config.numeric_backend,
            )
        )
        best_chunks.append(best_margin)
        worst_chunks.append(worst_margin)

    metrics = _combine_chunk_metrics(metric_chunks)
    best_margin = np.concatenate(best_chunks)
    worst_margin = np.concatenate(worst_chunks)
    frame_metadata = frame_context.metadata(sample_t_s=float(times[0]))
    digest = hashlib.sha256(
        json.dumps(
            {
                "contract_version": PRO_COVERAGE_CONTRACT_VERSION,
                "grid_identity": HEALPIX_GRID_ID,
                "config": _scientific_config(config),
                "frame": frame_metadata,
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    )
    for name, values, dtype in (
        ("times_s", times, "<f8"),
        ("covered_cell_count", covered_count, "<i8"),
        ("reason_count", reason_count, "<i8"),
        ("cell_best_margin_db", np.round(best_margin, 10), "<f8"),
        ("cell_worst_margin_db", np.round(worst_margin, 10), "<f8"),
        ("interval_offset", metrics.intervals.interval_offset, "<i8"),
        ("interval_start", metrics.intervals.start_sample_index, "<i8"),
        ("interval_end", metrics.intervals.end_sample_index_exclusive, "<i8"),
    ):
        digest.update(name.encode())
        digest.update(np.ascontiguousarray(values, dtype=dtype).tobytes())
    semantic_sha256 = digest.hexdigest()
    summary = {
        "contract_version": PRO_COVERAGE_CONTRACT_VERSION,
        "status": "complete",
        "domain_disposition": "global_earth_communications_service",
        "analysis_id": config.analysis_id,
        "service_id": config.service_id,
        "link_profile_id": config.link_profile.profile_id,
        "link_profile_sha256": config.link_profile.semantic_sha256,
        "grid_identity": HEALPIX_GRID_ID,
        "order": config.order,
        "membership_semantics": "WGS84 surface HEALPix NESTED cell center",
        "area_semantics": "equal authalic-sphere represented area per cell",
        "interval_semantics": "sample-bounded half-open intervals with left and right censoring retained",
        "covered_fraction_semantics": "fraction of represented HEALPix cells",
        "sample_count": int(times.size),
        "horizon_start_s": float(times[0]),
        "horizon_end_s": float(times[-1]),
        "cell_count": int(npix),
        "maximum_instantaneous_covered_fraction": float(np.max(covered_count / npix)),
        "ever_covered_cell_count": int(np.count_nonzero(metrics.interval_count)),
        "reason_names": list(PRO_COVERAGE_REASON_NAMES),
        "atmospheric_model_editions": list(
            evaluate_pro_rf_closure(
                config.link_profile,
                range_km=np.asarray([1000.0]),
                elevation_rad=np.asarray([np.pi / 2.0]),
            ).atmospheric_evidence.model_editions
        ),
        "claims": [
            "Deterministic sample-bounded RF-qualified cell-center coverage under the declared service profile.",
            "Per-cell best and worst engineering margin and primary limiting-reason counts.",
        ],
        "non_claims": [
            "No area-intersection or terrain-blockage claim.",
            "No packet delivery, operational availability, or weather-truth assurance.",
        ],
        "semantic_sha256": semantic_sha256,
    }
    return ProCommunicationsCoverageResult(
        config=config,
        frame_metadata=frame_metadata,
        times_s=times,
        covered_cell_count=covered_count,
        instantaneous_covered_fraction=covered_count / npix,
        cell_geodetic_latitude_deg=np.concatenate(latitude_chunks),
        cell_longitude_deg=np.concatenate(longitude_chunks),
        cell_metrics=metrics,
        primary_reason_count=reason_count,
        sample_margin_min_db=sample_margin_min,
        sample_margin_max_db=sample_margin_max,
        cell_best_margin_db=best_margin,
        cell_worst_margin_db=worst_margin,
        summary=summary,
        semantic_sha256=semantic_sha256,
    )


__all__ = [
    "PRO_COVERAGE_CONTRACT_VERSION",
    "PRO_COVERAGE_REASON_NAMES",
    "ProCommunicationsCoverageConfig",
    "ProCommunicationsCoverageResult",
    "evaluate_pro_communications_coverage",
]
