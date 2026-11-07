"""Rust coverage, access, aggregation, and RF kernel bindings.

The adapter deliberately contains no workflow policy. Rust is the primary
numerical backend; callers select ``numeric_backend="python"`` for the
reference path.  Arrays are copied at the boundary so callers never observe a
borrowed view into a native return value.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module

import numpy as np


def _extension():
    try:
        return import_module("oel_rust_orbit")
    except ImportError as exc:  # pragma: no cover - exercised by install smoke
        raise RuntimeError(
            "Rust coverage/link backend requested but oel_rust_orbit is unavailable; "
            "install the optional oel_rust_orbit wheel first"
        ) from exc


def _required(native, name: str):
    function = getattr(native, name, None)
    if function is None:
        raise RuntimeError(
            f"Rust coverage/link kernel {name} is unavailable; "
            "install a wheel that includes coverage/link kernels"
        )
    return function


def _pack_f64(values: np.ndarray) -> bytes:
    """Pack a numeric array once in the native kernels' little-endian ABI."""
    return np.asarray(values, dtype="<f8").tobytes(order="C")


def _pack_i64(values: np.ndarray) -> bytes:
    return np.ascontiguousarray(np.asarray(values, dtype="<i8")).tobytes(order="C")


def _pack_u8(values: np.ndarray) -> bytes:
    return np.ascontiguousarray(np.asarray(values, dtype=np.uint8)).tobytes(order="C")


def _unpack_f64(values: bytes, shape: tuple[int, ...] | None = None) -> np.ndarray:
    result = np.frombuffer(values, dtype="<f8").copy()
    return result if shape is None else result.reshape(shape)


def _unpack_i64(values: bytes, shape: tuple[int, ...] | None = None) -> np.ndarray:
    result = np.frombuffer(values, dtype="<i8").copy()
    return result if shape is None else result.reshape(shape)


def _unpack_flags(values: bytes, shape: tuple[int, ...] | None = None) -> np.ndarray:
    result = np.frombuffer(values, dtype=bool).copy()
    return result if shape is None else result.reshape(shape)


def _tile_rows(value: np.ndarray, label: str) -> np.ndarray:
    rows = np.asarray(value, dtype=np.float64)
    if rows.ndim != 2 or rows.shape[1] != 3:
        raise ValueError(f"{label} must have shape (samples, 3)")
    if not np.all(np.isfinite(rows)):
        raise ValueError(f"{label} must contain only finite values")
    return np.ascontiguousarray(rows)


def try_surface_targets_tile(
    observers: np.ndarray, targets: np.ndarray, normals: np.ndarray,
    boresights: np.ndarray, half_angle_rad: float, max_range_km: float | None,
    angular_tolerance_rad: float, range_tolerance_km: float,
) -> np.ndarray | None:
    observer_rows = _tile_rows(observers, "observers")
    target_rows = _tile_rows(targets, "targets")
    normal_rows = _tile_rows(normals, "normals")
    boresight_rows = _tile_rows(boresights, "boresights")
    if normal_rows.shape != target_rows.shape:
        raise ValueError("normals must match targets with shape (cells, 3)")
    if boresight_rows.shape != observer_rows.shape:
        raise ValueError("boresights must match observers with shape (samples, 3)")
    native = getattr(_extension(), "coverage_surface_targets_tile_bytes", None)
    if native is None:
        return None
    samples, cells = observer_rows.shape[0], target_rows.shape[0]
    raw = native(
        _pack_f64(observer_rows), _pack_f64(target_rows), _pack_f64(normal_rows),
        _pack_f64(boresight_rows), float(half_angle_rad), max_range_km,
        float(angular_tolerance_rad), float(range_tolerance_km),
    )
    if len(raw) != samples * cells:
        raise RuntimeError("Rust surface tile returned an unexpected cell count")
    return _unpack_flags(raw, (samples, cells))


def try_rich_reasons_tile(
    observers: np.ndarray, targets: np.ndarray, normals: np.ndarray,
    rotations: np.ndarray, suns: np.ndarray | None, pattern_kind: str,
    x_angle: float, y_angle: float, max_range: float | None,
    max_off_nadir: float | None, max_incidence: float | None,
    min_sun: float | None, max_sun: float | None,
    angular_tol: float, range_tol: float,
) -> np.ndarray | None:
    kinds = {"axisymmetric_hard_cone": 0, "rectangular_hard_fov": 1, "pushbroom_hard_fov": 2}
    observer_rows = _tile_rows(observers, "observers")
    target_rows = _tile_rows(targets, "targets")
    normal_rows = _tile_rows(normals, "normals")
    rotation_rows = np.asarray(rotations, dtype=np.float64)
    if rotation_rows.shape != (observer_rows.shape[0], 3, 3) or not np.all(np.isfinite(rotation_rows)):
        raise ValueError("rotations must be finite with shape (samples, 3, 3)")
    if normal_rows.shape != target_rows.shape:
        raise ValueError("normals must match targets with shape (cells, 3)")
    sun_rows = None if suns is None else _tile_rows(suns, "suns")
    if sun_rows is not None and sun_rows.shape != observer_rows.shape:
        raise ValueError("suns must match observers with shape (samples, 3)")
    if pattern_kind not in kinds:
        raise ValueError(f"unsupported coverage pattern kind {pattern_kind!r}")
    native = getattr(_extension(), "coverage_rich_cell_reasons_tile_bytes", None)
    if native is None:
        return None
    samples, cells = observer_rows.shape[0], target_rows.shape[0]
    raw = native(
        _pack_f64(observer_rows), _pack_f64(target_rows), _pack_f64(normal_rows),
        _pack_f64(rotation_rows), None if sun_rows is None else _pack_f64(sun_rows),
        kinds[pattern_kind], float(x_angle), float(y_angle), max_range,
        max_off_nadir, max_incidence, min_sun, max_sun,
        float(angular_tol), float(range_tol),
    )
    if len(raw) != samples * cells:
        raise RuntimeError("Rust rich tile returned an unexpected cell count")
    return np.frombuffer(raw, dtype=np.uint8).copy().reshape(samples, cells)


def try_tasking_exact_search(
    ids: tuple[str, ...], rows: list[float], suffix: np.ndarray,
    transitions: list[float], initial_storage: float, capacity: float,
    budget: float, duty: float, horizon: float,
) -> tuple[tuple[int, ...], float, int] | None:
    native = getattr(_extension(), "coverage_tasking_exact_search", None)
    if native is None:
        return None
    indices, value, evaluated = native(
        list(ids), rows, np.asarray(suffix, dtype=float).tolist(), transitions,
        initial_storage, capacity, budget, duty, horizon,
    )
    return tuple(int(index) for index in indices), float(value), int(evaluated)


def try_communications_geometry_tile(
    sources_ecef: np.ndarray,
    terminal_from_ecef: np.ndarray,
    centers_ecef: np.ndarray,
    normals_ecef: np.ndarray,
    *,
    minimum_elevation_rad: float,
    max_range_km: float | None,
    source_half_angle_rad: float | None,
    earth_half_angle_rad: float | None,
    direct_cosine: bool,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return source-to-cell ranges and the first five gate flags by tile."""
    native = getattr(_extension(), "coverage_communications_geometry_tile_bytes", None)
    if native is None:
        return None
    sources = np.ascontiguousarray(np.asarray(sources_ecef, dtype=np.float64))
    rotations = np.ascontiguousarray(np.asarray(terminal_from_ecef, dtype=np.float64))
    centers = np.ascontiguousarray(np.asarray(centers_ecef, dtype=np.float64))
    normals = np.ascontiguousarray(np.asarray(normals_ecef, dtype=np.float64))
    if sources.ndim != 2 or sources.shape[1] != 3 or rotations.shape != (sources.shape[0], 3, 3):
        raise ValueError("communications source and terminal shapes are inconsistent")
    if centers.ndim != 2 or centers.shape[1] != 3 or normals.shape != centers.shape:
        raise ValueError("communications center and normal shapes are inconsistent")
    raw_range, raw_flags = native(
        _pack_f64(sources), _pack_f64(rotations), _pack_f64(centers), _pack_f64(normals),
        float(minimum_elevation_rad), None if max_range_km is None else float(max_range_km),
        None if source_half_angle_rad is None else float(source_half_angle_rad),
        None if earth_half_angle_rad is None else float(earth_half_angle_rad), bool(direct_cosine),
    )
    shape = (sources.shape[0], centers.shape[0])
    return _unpack_f64(raw_range, shape), np.frombuffer(raw_flags, dtype=np.uint8).copy().reshape(shape)


@dataclass(frozen=True)
class SurfaceTargetResult:
    range_km: np.ndarray
    cosine_off_axis: np.ndarray
    horizon_clearance_km: np.ndarray
    visible: np.ndarray
    inside_pattern: np.ndarray
    inside_range: np.ndarray
    available: np.ndarray


def access_batch(
    observer_eci_km: np.ndarray,
    target_eci_km: np.ndarray,
    *,
    boresight_eci: np.ndarray | None = None,
    max_range_km: float | None = None,
    fov_half_angle_rad: float | None = None,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    observer = np.asarray(observer_eci_km, dtype=np.float64)
    target = np.asarray(target_eci_km, dtype=np.float64)
    if observer.ndim != 2 or observer.shape[1] != 3 or target.shape != observer.shape:
        raise ValueError("observer and target must have shape (samples, 3)")
    boresight = None if boresight_eci is None else np.asarray(boresight_eci, dtype=np.float64)
    if boresight is not None and boresight.shape != observer.shape:
        raise ValueError("boresight_eci must match observer shape")
    raw = _required(_extension(), "coverage_access_batch")(
        np.ascontiguousarray(observer).ravel().tolist(),
        np.ascontiguousarray(target).ravel().tolist(),
        None if boresight is None else np.ascontiguousarray(boresight).ravel().tolist(),
        None if max_range_km is None else float(max_range_km),
        None if fov_half_angle_rad is None else float(fov_half_angle_rad),
    )
    return (
        np.asarray(raw[0], dtype=np.float64),
        np.asarray(raw[1], dtype=bool),
        tuple(str(value) for value in raw[2]),
    )


def surface_targets_ecef(
    observer_ecef_km: np.ndarray,
    target_ecef_km: np.ndarray,
    target_outward_normal_ecef: np.ndarray,
    boresight_ecef: np.ndarray,
    half_angle_rad: float,
    max_range_km: float | None = None,
    *,
    angular_tolerance_rad: float = 1.0e-12,
    range_tolerance_km: float = 1.0e-9,
) -> SurfaceTargetResult:
    observer = np.asarray(observer_ecef_km, dtype=np.float64).reshape(3)
    targets = np.asarray(target_ecef_km, dtype=np.float64)
    normals = np.asarray(target_outward_normal_ecef, dtype=np.float64)
    boresight = np.asarray(boresight_ecef, dtype=np.float64).reshape(3)
    if targets.ndim != 2 or targets.shape[1] != 3:
        raise ValueError("target_ecef_km must have shape (targets, 3)")
    if normals.shape != targets.shape:
        raise ValueError("target_outward_normal_ecef must match target_ecef_km")
    if not np.all(np.isfinite(targets)):
        raise ValueError("target_ecef_km must be finite with shape (targets, 3).")
    if not np.all(np.isfinite(normals)):
        raise ValueError(
            "target_outward_normal_ecef must be finite and match target_ecef_km."
        )
    if not np.all(np.isfinite(observer)) or not np.all(np.isfinite(boresight)):
        raise ValueError("Observer and boresight vectors must be finite.")
    boresight_norm = float(np.linalg.norm(boresight))
    if abs(boresight_norm - 1.0) > 1.0e-10:
        raise ValueError("boresight_ecef must be a unit vector within 1e-10.")
    normal_norms = np.linalg.norm(normals, axis=1)
    if np.any(np.abs(normal_norms - 1.0) > 1.0e-10):
        raise ValueError("Target outward normals must be unit vectors within 1e-10.")
    # Keep direct adapter callers on the same documented physical boundary as
    # evaluate_surface_targets_ecef before loading or dispatching to native code.
    from sim.analysis.observer_target_geometry import _validate_wgs84_surface_targets

    _validate_wgs84_surface_targets(targets, normals)
    native = _extension()
    byte_function = getattr(native, "coverage_surface_targets_ecef_bytes", None)
    if byte_function is not None:
        values = byte_function(
            _pack_f64(observer),
            _pack_f64(targets),
            _pack_f64(normals),
            _pack_f64(boresight),
            float(half_angle_rad),
            None if max_range_km is None else float(max_range_km),
            float(angular_tolerance_rad),
            float(range_tolerance_km),
        )
        target_shape = (targets.shape[0],)
        return SurfaceTargetResult(
            range_km=_unpack_f64(values[0], target_shape),
            cosine_off_axis=_unpack_f64(values[1], target_shape),
            horizon_clearance_km=_unpack_f64(values[2], target_shape),
            visible=_unpack_flags(values[3], target_shape),
            inside_pattern=_unpack_flags(values[4], target_shape),
            inside_range=_unpack_flags(values[5], target_shape),
            available=_unpack_flags(values[6], target_shape),
        )
    values = _required(native, "coverage_surface_targets_ecef")(
        observer.tolist(),
        np.ascontiguousarray(targets).ravel().tolist(),
        np.ascontiguousarray(normals).ravel().tolist(),
        boresight.tolist(),
        float(half_angle_rad),
        None if max_range_km is None else float(max_range_km),
        float(angular_tolerance_rad),
        float(range_tolerance_km),
    )
    return SurfaceTargetResult(
        range_km=np.asarray(values[0], dtype=np.float64),
        cosine_off_axis=np.asarray(values[1], dtype=np.float64),
        horizon_clearance_km=np.asarray(values[2], dtype=np.float64),
        visible=np.asarray(values[3], dtype=bool),
        inside_pattern=np.asarray(values[4], dtype=bool),
        inside_range=np.asarray(values[5], dtype=bool),
        available=np.asarray(values[6], dtype=bool),
    )


def rich_cell_reasons(
    observer: np.ndarray,
    targets: np.ndarray,
    normals: np.ndarray,
    rotation: np.ndarray,
    sun: np.ndarray | None,
    pattern_kind: str,
    x_angle: float,
    y_angle: float,
    max_range: float | None,
    max_off_nadir: float | None,
    max_incidence: float | None,
    min_sun: float | None,
    max_sun: float | None,
    angular_tol: float,
    range_tol: float,
) -> np.ndarray:
    """Return ordered rich-coverage reason codes from the native cell gate."""
    kinds = {"axisymmetric_hard_cone": 0, "rectangular_hard_fov": 1, "pushbroom_hard_fov": 2}
    raw = _required(_extension(), "coverage_rich_cell_reasons_bytes")(
        _pack_f64(observer), _pack_f64(targets), _pack_f64(normals), _pack_f64(rotation),
        None if sun is None else _pack_f64(sun), kinds[pattern_kind], float(x_angle), float(y_angle),
        max_range, max_off_nadir, max_incidence, min_sun, max_sun,
        float(angular_tol), float(range_tol),
    )
    result = np.frombuffer(raw, dtype=np.uint8).copy()
    if result.size != np.asarray(targets).shape[0]:
        raise RuntimeError("Rust rich coverage returned an unexpected cell count")
    return result


@dataclass(frozen=True)
class WGS84RayIntersections:
    hit: np.ndarray
    distance_km: np.ndarray
    point_ecef_km: np.ndarray


def intersect_rays_wgs84(
    observer_ecef_km: np.ndarray,
    direction_ecef: np.ndarray,
    *,
    discriminant_tolerance: float = 1.0e-12,
    distance_tolerance_km: float = 1.0e-9,
) -> WGS84RayIntersections:
    for name, value in (("discriminant_tolerance", discriminant_tolerance),
                        ("distance_tolerance_km", distance_tolerance_km)):
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")
    observer = np.asarray(observer_ecef_km, dtype=np.float64).reshape(3)
    directions = np.asarray(direction_ecef, dtype=np.float64)
    if directions.ndim != 2 or directions.shape[1] != 3:
        raise ValueError("direction_ecef must have shape (items, 3)")
    native = _extension()
    byte_function = getattr(native, "coverage_intersect_rays_wgs84_bytes", None)
    if byte_function is not None:
        values = byte_function(
            _pack_f64(observer),
            _pack_f64(directions),
            float(discriminant_tolerance),
            float(distance_tolerance_km),
        )
        direction_shape = directions.shape
        return WGS84RayIntersections(
            hit=_unpack_flags(values[0], (directions.shape[0],)),
            distance_km=_unpack_f64(values[1], (directions.shape[0],)),
            point_ecef_km=_unpack_f64(values[2], direction_shape),
        )
    values = _required(native, "coverage_intersect_rays_wgs84")(
        observer.tolist(),
        np.ascontiguousarray(directions).ravel().tolist(),
        float(discriminant_tolerance),
        float(distance_tolerance_km),
    )
    return WGS84RayIntersections(
        hit=np.asarray(values[0], dtype=bool),
        distance_km=np.asarray(values[1], dtype=np.float64),
        point_ecef_km=np.asarray(values[2], dtype=np.float64).reshape(directions.shape),
    )


@dataclass(frozen=True)
class CoverageMetrics:
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
class SparseCoverageIntervals:
    cell_index: np.ndarray
    interval_offset: np.ndarray
    start_sample_index: np.ndarray
    end_sample_index_exclusive: np.ndarray


def summarize_sampled_mask(
    covered_by_sample: np.ndarray,
    times_s: np.ndarray,
    *,
    cell_indices: np.ndarray | None = None,
) -> CoverageMetrics:
    from sim.analysis.global_coverage import _validated_sparse_coverage_inputs

    mask, times, cells = _validated_sparse_coverage_inputs(covered_by_sample, times_s, cell_indices)
    native = _extension()
    byte_function = getattr(native, "coverage_summarize_mask_bytes", None)
    if byte_function is not None:
        raw = byte_function(
            _pack_f64(times),
            _pack_u8(mask),
            int(mask.shape[0]),
            int(mask.shape[1]),
            None if cells is None else _pack_i64(cells),
        )
        return CoverageMetrics(
            dwell_s=_unpack_f64(raw[0]),
            interval_count=_unpack_i64(raw[1]),
            observed_acquisition_count=_unpack_i64(raw[2]),
            max_complete_revisit_gap_s=_unpack_f64(raw[3]),
            prefix_boundary_gap_s=_unpack_f64(raw[4]),
            suffix_boundary_gap_s=_unpack_f64(raw[5]),
            start_censored=_unpack_flags(raw[6]),
            end_censored=_unpack_flags(raw[7]),
            intervals=SparseCoverageIntervals(
                cell_index=_unpack_i64(raw[8]),
                interval_offset=_unpack_i64(raw[9]),
                start_sample_index=_unpack_i64(raw[10]),
                end_sample_index_exclusive=_unpack_i64(raw[11]),
            ),
        )
    raw = _required(native, "coverage_summarize_mask")(
        times.tolist(),
        np.ascontiguousarray(mask, dtype=np.uint8).ravel().tolist(),
        int(mask.shape[0]),
        int(mask.shape[1]),
        None if cells is None else cells.tolist(),
    )
    return CoverageMetrics(
        dwell_s=np.asarray(raw[0], dtype=np.float64),
        interval_count=np.asarray(raw[1], dtype=np.int64),
        observed_acquisition_count=np.asarray(raw[2], dtype=np.int64),
        max_complete_revisit_gap_s=np.asarray(raw[3], dtype=np.float64),
        prefix_boundary_gap_s=np.asarray(raw[4], dtype=np.float64),
        suffix_boundary_gap_s=np.asarray(raw[5], dtype=np.float64),
        start_censored=np.asarray(raw[6], dtype=bool),
        end_censored=np.asarray(raw[7], dtype=bool),
        intervals=SparseCoverageIntervals(
            cell_index=np.asarray(raw[8], dtype=np.int64),
            interval_offset=np.asarray(raw[9], dtype=np.int64),
            start_sample_index=np.asarray(raw[10], dtype=np.int64),
            end_sample_index_exclusive=np.asarray(raw[11], dtype=np.int64),
        ),
    )


def aggregate_multiplicity(
    member_masks: np.ndarray,
    *,
    required_multiplicity: int,
) -> dict[str, np.ndarray]:
    masks = np.asarray(member_masks, dtype=bool)
    if masks.ndim != 3:
        raise ValueError("member_masks must have shape (members, samples, cells)")
    native = _extension()
    byte_function = getattr(native, "coverage_aggregate_multiplicity_bytes", None)
    if byte_function is not None:
        raw = byte_function(_pack_u8(masks), int(masks.shape[0]), int(masks.shape[1]),
                            int(masks.shape[2]), int(required_multiplicity))
        def unpack_u16(value):
            return np.frombuffer(value, dtype="<u2").copy()
        return {
            "multiplicity": unpack_u16(raw[0]).reshape(masks.shape[1:]),
            "qualified": _unpack_flags(raw[1]).reshape(masks.shape[1:]),
            "active_asset_count_by_sample": _unpack_i64(raw[2]),
            "maximum_multiplicity_by_sample": unpack_u16(raw[3]),
            "mean_multiplicity_per_cell": _unpack_f64(raw[4]),
            "max_multiplicity_per_cell": unpack_u16(raw[5]),
            "multiplicity_histogram": _unpack_i64(raw[6]),
        }
    raw = _required(native, "coverage_aggregate_multiplicity")(
        np.ascontiguousarray(masks, dtype=np.uint8).ravel().tolist(),
        int(masks.shape[0]),
        int(masks.shape[1]),
        int(masks.shape[2]),
        int(required_multiplicity),
    )
    return {
        "multiplicity": np.asarray(raw[0], dtype=np.uint16).reshape(masks.shape[1:]),
        "qualified": np.asarray(raw[1], dtype=bool).reshape(masks.shape[1:]),
        "active_asset_count_by_sample": np.asarray(raw[2], dtype=np.int64),
        "maximum_multiplicity_by_sample": np.asarray(raw[3], dtype=np.uint16),
        "mean_multiplicity_per_cell": np.asarray(raw[4], dtype=np.float64),
        "max_multiplicity_per_cell": np.asarray(raw[5], dtype=np.uint16),
        "multiplicity_histogram": np.asarray(raw[6], dtype=np.int64),
    }


def sampled_windows(
    times_s: np.ndarray,
    available: np.ndarray,
    reasons: tuple[str, ...] | list[str] | np.ndarray,
) -> tuple[tuple[tuple, ...], tuple[tuple, ...]]:
    times = np.asarray(times_s, dtype=np.float64)
    mask = np.asarray(available, dtype=bool)
    if times.ndim != 1 or mask.ndim != 1:
        raise ValueError("times_s and available must be one-dimensional")
    raw = _required(_extension(), "coverage_sampled_windows")(
        times.tolist(),
        np.ascontiguousarray(mask, dtype=np.uint8).tolist(),
        tuple(str(value) for value in reasons),
    )
    return tuple(tuple(value) for value in raw[0]), tuple(tuple(value) for value in raw[1])


def refine_availability_transitions(
    times_s: np.ndarray,
    available: np.ndarray,
    reasons: tuple[str, ...] | list[str] | np.ndarray,
    *,
    evaluator_at_time=None,
    time_tolerance_s: float | None = None,
    max_iterations: int | None = None,
) -> tuple[tuple, ...]:
    times = np.asarray(times_s, dtype=np.float64)
    mask = np.asarray(available, dtype=bool)
    if times.ndim != 1 or mask.ndim != 1:
        raise ValueError("times_s and available must be one-dimensional")
    raw = _required(_extension(), "coverage_refine_transitions")(
        times.tolist(),
        np.ascontiguousarray(mask, dtype=np.uint8).tolist(),
        tuple(str(value) for value in reasons),
        evaluator_at_time,
        None if time_tolerance_s is None else float(time_tolerance_s),
        None if max_iterations is None else int(max_iterations),
    )
    return tuple(tuple(value) for value in raw)


def earth_occulted(
    tx_position_ecef_km: np.ndarray,
    rx_position_ecef_km: np.ndarray,
) -> np.ndarray:
    tx = np.asarray(tx_position_ecef_km, dtype=np.float64)
    rx = np.asarray(rx_position_ecef_km, dtype=np.float64)
    if tx.ndim != 2 or tx.shape[1] != 3 or rx.shape != tx.shape:
        raise ValueError("endpoint ECEF positions must both have shape (samples, 3)")
    native = _extension()
    byte_function = getattr(native, "link_earth_occulted_bytes", None)
    if byte_function is not None:
        return _unpack_flags(byte_function(_pack_f64(tx), _pack_f64(rx)))
    result = _required(native, "link_earth_occulted")(
        np.ascontiguousarray(tx).ravel().tolist(),
        np.ascontiguousarray(rx).ravel().tolist(),
    )
    return np.asarray(result, dtype=bool)


def endpoint_kinematics(
    tx_position_eci_km: np.ndarray,
    rx_position_eci_km: np.ndarray,
    tx_velocity_eci_km_s: np.ndarray,
    rx_velocity_eci_km_s: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    arrays = [
        np.asarray(value, dtype=np.float64) for value in (
            tx_position_eci_km,
            rx_position_eci_km,
            tx_velocity_eci_km_s,
            rx_velocity_eci_km_s,
        )
    ]
    if any(value.ndim != 2 or value.shape[1] != 3 for value in arrays):
        raise ValueError("endpoint kinematics arrays must have shape (samples, 3)")
    if any(value.shape != arrays[0].shape for value in arrays[1:]):
        raise ValueError("endpoint kinematics arrays must have equal shapes")
    native = _extension()
    byte_function = getattr(native, "link_endpoint_kinematics_bytes", None)
    if byte_function is not None:
        raw = byte_function(*(_pack_f64(value) for value in arrays))
        return _unpack_f64(raw[0]), _unpack_f64(raw[1])
    raw = _required(native, "link_endpoint_kinematics")(
        *(np.ascontiguousarray(value).ravel().tolist() for value in arrays)
    )
    return np.asarray(raw[0], dtype=np.float64), np.asarray(raw[1], dtype=np.float64)


def terminal_pattern(
    peer_direction_eci: np.ndarray,
    dcm_parent_from_eci: np.ndarray,
    quat_parent_from_terminal: np.ndarray,
    half_angle_rad: float,
) -> tuple[np.ndarray, np.ndarray]:
    directions = np.asarray(peer_direction_eci, dtype=np.float64)
    dcm = np.asarray(dcm_parent_from_eci, dtype=np.float64)
    if directions.ndim != 2 or directions.shape[1] != 3:
        raise ValueError("peer_direction_eci must have shape (samples, 3)")
    if dcm.shape != (directions.shape[0], 3, 3):
        raise ValueError("dcm_parent_from_eci must have shape (samples, 3, 3)")
    raw = _required(_extension(), "link_terminal_pattern")(
        np.ascontiguousarray(directions).ravel().tolist(),
        np.ascontiguousarray(dcm).ravel().tolist(),
        np.asarray(quat_parent_from_terminal, dtype=np.float64).reshape(4).tolist(),
        float(half_angle_rad),
    )
    return np.asarray(raw[0], dtype=np.float64), np.asarray(raw[1], dtype=bool)


@dataclass(frozen=True)
class FreeSpaceLedger:
    tx_power_dbw: np.ndarray
    tx_gain_dbi: np.ndarray
    rx_gain_dbi: np.ndarray
    free_space_path_loss_db: np.ndarray
    eirp_dbw: np.ndarray
    received_power_dbw: np.ndarray
    noise_density_dbw_hz: np.ndarray
    cn0_db_hz: np.ndarray
    eb_n0_db: np.ndarray
    margin_db: np.ndarray
    margin_pass: np.ndarray


def free_space_link_ledger(
    range_km: np.ndarray,
    *,
    carrier_frequency_hz: float,
    tx_power_w: float,
    tx_gain_dbi: np.ndarray | float,
    rx_gain_dbi: np.ndarray | float,
    data_rate_bps: float,
    system_noise_temperature_k: float,
    required_eb_n0_db: float,
    tx_line_loss_db: float = 0.0,
    rx_line_loss_db: float = 0.0,
    misc_loss_db: float = 0.0,
) -> FreeSpaceLedger:
    range_array = np.asarray(range_km, dtype=np.float64)
    output_shape = range_array.shape
    ranges = range_array.reshape(-1)
    tx_gain = np.broadcast_to(np.asarray(tx_gain_dbi, dtype=np.float64), output_shape).reshape(-1)
    rx_gain = np.broadcast_to(np.asarray(rx_gain_dbi, dtype=np.float64), output_shape).reshape(-1)
    native = _extension()
    byte_function = getattr(native, "link_free_space_ledger_bytes", None)
    if byte_function is not None:
        raw = byte_function(
            _pack_f64(ranges),
            _pack_f64(tx_gain),
            _pack_f64(rx_gain),
            float(carrier_frequency_hz),
            float(tx_power_w),
            float(data_rate_bps),
            float(system_noise_temperature_k),
            float(required_eb_n0_db),
            float(tx_line_loss_db),
            float(rx_line_loss_db),
            float(misc_loss_db),
        )
        return FreeSpaceLedger(
            *(
                _unpack_f64(value, output_shape)
                if index < 10
                else _unpack_flags(value, output_shape)
                for index, value in enumerate(raw)
            )
        )
    raw = _required(native, "link_free_space_ledger")(
        ranges.tolist(),
        np.ascontiguousarray(tx_gain).tolist(),
        np.ascontiguousarray(rx_gain).tolist(),
        float(carrier_frequency_hz),
        float(tx_power_w),
        float(data_rate_bps),
        float(system_noise_temperature_k),
        float(required_eb_n0_db),
        float(tx_line_loss_db),
        float(rx_line_loss_db),
        float(misc_loss_db),
    )
    return FreeSpaceLedger(
        *(
            np.asarray(value, dtype=np.float64 if index < 10 else bool).reshape(output_shape)
            for index, value in enumerate(raw)
        )
    )


def p838_3_specific_rain_attenuation(
    frequency_ghz: np.ndarray,
    *,
    rain_rate_mm_h: np.ndarray,
    elevation_deg: np.ndarray,
    polarization_tilt_deg: np.ndarray,
) -> np.ndarray:
    arrays = np.broadcast_arrays(
        np.asarray(frequency_ghz, dtype=np.float64),
        np.asarray(rain_rate_mm_h, dtype=np.float64),
        np.asarray(elevation_deg, dtype=np.float64),
        np.asarray(polarization_tilt_deg, dtype=np.float64),
    )
    output_shape = arrays[0].shape
    arrays = tuple(np.ascontiguousarray(value, dtype=np.float64).reshape(-1) for value in arrays)
    raw = _required(_extension(), "link_p838_rain_specific")(*(value.tolist() for value in arrays))
    return np.asarray(raw, dtype=np.float64).reshape(output_shape)


def p840_9_cloud_attenuation(
    frequency_ghz: np.ndarray,
    *,
    elevation_deg: np.ndarray,
    integrated_cloud_liquid_water_kg_m2: np.ndarray,
) -> np.ndarray:
    arrays = np.broadcast_arrays(
        np.asarray(frequency_ghz, dtype=np.float64),
        np.asarray(elevation_deg, dtype=np.float64),
        np.asarray(integrated_cloud_liquid_water_kg_m2, dtype=np.float64),
    )
    output_shape = arrays[0].shape
    arrays = tuple(np.ascontiguousarray(value, dtype=np.float64).reshape(-1) for value in arrays)
    raw = _required(_extension(), "link_p840_cloud_attenuation")(*(value.tolist() for value in arrays))
    return np.asarray(raw, dtype=np.float64).reshape(output_shape)


def p618_14_total_attenuation(
    *,
    gaseous_attenuation_db: np.ndarray,
    cloud_attenuation_db: np.ndarray,
    rain_attenuation_db: np.ndarray,
    scintillation_fade_depth_db: np.ndarray,
) -> np.ndarray:
    arrays = np.broadcast_arrays(
        np.asarray(gaseous_attenuation_db, dtype=np.float64),
        np.asarray(cloud_attenuation_db, dtype=np.float64),
        np.asarray(rain_attenuation_db, dtype=np.float64),
        np.asarray(scintillation_fade_depth_db, dtype=np.float64),
    )
    output_shape = arrays[0].shape
    arrays = tuple(np.ascontiguousarray(value, dtype=np.float64).reshape(-1) for value in arrays)
    raw = _required(_extension(), "link_p618_total_attenuation")(*(value.tolist() for value in arrays))
    return np.asarray(raw, dtype=np.float64).reshape(output_shape)


__all__ = [
    "CoverageMetrics",
    "FreeSpaceLedger",
    "SparseCoverageIntervals",
    "SurfaceTargetResult",
    "WGS84RayIntersections",
    "aggregate_multiplicity",
    "access_batch",
    "earth_occulted",
    "endpoint_kinematics",
    "free_space_link_ledger",
    "intersect_rays_wgs84",
    "p618_14_total_attenuation",
    "p838_3_specific_rain_attenuation",
    "p840_9_cloud_attenuation",
    "refine_availability_transitions",
    "sampled_windows",
    "summarize_sampled_mask",
    "surface_targets_ecef",
    "terminal_pattern",
]
