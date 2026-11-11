"""Rust numerical kernels for private Scale workflows.

The Scale owners continue to construct trials, preserve IDs, write stores, and
apply policy in Python.  This adapter only marshals dense numeric arrays into
the ``oel_rust_orbit`` extension. Rust is the primary numerical backend;
explicit Python selection retains the reference implementation.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from sim.rust_orbit_backend import _extension


def _native_function(name: str):
    native = _extension()
    function = getattr(native, name, None)
    if function is None:
        raise RuntimeError(
            f"Rust Scale backend requires an oel_rust_orbit wheel exposing {name}; "
            "install a current optional wheel first"
        )
    return function


def _f64_bytes(value: Any) -> bytes:
    """Pack an array as explicit little endian float64 bytes."""

    return np.asarray(value, dtype="<f8").tobytes(order="C")


def _f64_from_bytes(value: bytes, shape: tuple[int, ...]) -> np.ndarray:
    """Retain owned writable output storage or copy immutable legacy bytes."""

    values = np.frombuffer(value, dtype="<f8")
    if not values.flags.writeable:
        values = values.copy()
    return values.reshape(shape)



def closest_sampled_approach(
    left_states: np.ndarray,
    right_states: np.ndarray,
    sample_jd_utc: np.ndarray,
) -> tuple[int, float, float, float] | None:
    """Return the first minimum sampled pair row.

    Histories have shape ``(samples, 6)`` in position/velocity order.  The
    returned tuple is ``(sample_index, distance_km, jd_utc, speed_km_s)``.
    """

    left = np.asarray(left_states, dtype=np.float64)
    right = np.asarray(right_states, dtype=np.float64)
    times = np.asarray(sample_jd_utc, dtype=np.float64).reshape(-1)
    if left.ndim != 2 or left.shape[1:] != (6,):
        raise ValueError("left pair history must have shape (samples, 6)")
    if right.shape != left.shape or times.shape != (left.shape[0],):
        raise ValueError("pair histories and sample times have inconsistent shapes")
    native = _extension()
    byte_function = getattr(native, "scale_closest_sampled_approach_bytes", None)
    if byte_function is not None:
        result = byte_function(_f64_bytes(left), _f64_bytes(right), _f64_bytes(times))
    else:
        result = _native_function("scale_closest_sampled_approach")(
            np.ascontiguousarray(left, dtype="<f8").ravel().tolist(),
            np.ascontiguousarray(right, dtype="<f8").ravel().tolist(),
            np.ascontiguousarray(times, dtype="<f8").tolist(),
        )
    if result is None:
        return None
    return int(result[0]), float(result[1]), float(result[2]), float(result[3])


def relative_metrics(left_states: np.ndarray, right_states: np.ndarray) -> np.ndarray:
    """Compute synchronized distance and relative-speed rows in Rust."""

    left = np.asarray(left_states, dtype=np.float64)
    right = np.asarray(right_states, dtype=np.float64)
    if left.ndim != 2 or left.shape[1:] != (6,) or right.shape != left.shape:
        raise ValueError("pair histories must both have shape (samples, 6)")
    native = _extension()
    byte_function = getattr(native, "scale_relative_metrics_bytes", None)
    if byte_function is not None:
        values = _f64_from_bytes(byte_function(_f64_bytes(left), _f64_bytes(right)), (left.shape[0], 2))
    else:
        values = _native_function("scale_relative_metrics")(
            np.ascontiguousarray(left, dtype="<f8").ravel().tolist(),
            np.ascontiguousarray(right, dtype="<f8").ravel().tolist(),
        )
        values = np.asarray(values, dtype=np.float64).reshape(left.shape[0], 2)
    return values


def propagate_pair_grid(elements: list[Any], time_grid_min: np.ndarray):
    """Propagate a pair or small object batch through the existing OGP batch.

    The OGP adapter already owns the mean-element conversion and sparse error
    contract.  This convenience function keeps Scale refinement independent of
    that adapter's private representation while avoiding one Python call per
    sample.
    """

    if len(elements) == 0:
        raise ValueError("at least one OGP element row is required")
    grid = np.asarray(time_grid_min, dtype=np.float64)
    if grid.ndim != 2 or grid.shape[0] != len(elements):
        raise ValueError("OGP pair time grid must have shape (objects, samples)")
    from sim.rust_ogp_backend import propagate_batch

    return propagate_batch(elements, np.ascontiguousarray(grid, dtype=np.float64))


def pso_update(
    positions: np.ndarray,
    velocities: np.ndarray,
    personal_best_positions: np.ndarray,
    personal_best_costs: np.ndarray,
    global_best_position: np.ndarray,
    random_cognitive: np.ndarray,
    random_social: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    inertia: float,
    cognitive: float,
    social: float,
    costs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Run one bounded PSO arithmetic/update step in Rust.

    Objective evaluation remains in Python.  ``costs`` must therefore be the
    values for the newly proposed particle positions.
    """

    values = [
        np.asarray(item, dtype=np.float64) for item in (
            positions,
            velocities,
            personal_best_positions,
            personal_best_costs,
            global_best_position,
            random_cognitive,
            random_social,
            lower,
            upper,
            costs,
        )
    ]
    native = _extension()
    byte_function = getattr(native, "campaign_pso_update_bytes", None)
    if byte_function is not None:
        position, velocity, pbest_position, pbest_cost, best_index = byte_function(
            *(_f64_bytes(item) for item in values[:9]),
            float(inertia),
            float(cognitive),
            float(social),
            _f64_bytes(values[9]),
        )
        return (
            _f64_from_bytes(position, np.asarray(positions).shape),
            _f64_from_bytes(velocity, np.asarray(velocities).shape),
            _f64_from_bytes(pbest_position, np.asarray(personal_best_positions).shape),
            _f64_from_bytes(pbest_cost, np.asarray(personal_best_costs).shape),
            int(best_index),
        )
    position, velocity, pbest_position, pbest_cost, best_index = _native_function("campaign_pso_update")(
        *(np.ascontiguousarray(item, dtype="<f8").ravel().tolist() for item in values[:3]),
        np.ascontiguousarray(values[3], dtype="<f8").ravel().tolist(),
        np.ascontiguousarray(values[4], dtype="<f8").ravel().tolist(),
        np.ascontiguousarray(values[5], dtype="<f8").ravel().tolist(),
        np.ascontiguousarray(values[6], dtype="<f8").ravel().tolist(),
        np.ascontiguousarray(values[7], dtype="<f8").ravel().tolist(),
        np.ascontiguousarray(values[8], dtype="<f8").ravel().tolist(),
        float(inertia),
        float(cognitive),
        float(social),
        np.ascontiguousarray(values[9], dtype="<f8").ravel().tolist(),
    )
    return (
        np.asarray(position, dtype=np.float64).reshape(np.asarray(positions).shape),
        np.asarray(velocity, dtype=np.float64).reshape(np.asarray(velocities).shape),
        np.asarray(pbest_position, dtype=np.float64).reshape(np.asarray(personal_best_positions).shape),
        np.asarray(pbest_cost, dtype=np.float64).reshape(np.asarray(personal_best_costs).shape),
        int(best_index),
    )


def pso_propose(
    positions: np.ndarray,
    velocities: np.ndarray,
    personal_best_positions: np.ndarray,
    global_best_position: np.ndarray,
    random_cognitive: np.ndarray,
    random_social: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    inertia: float,
    cognitive: float,
    social: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Run only the bounded PSO arithmetic, leaving objective evaluation to Python."""

    arrays = [
        np.asarray(item, dtype=np.float64)
        for item in (
            positions,
            velocities,
            personal_best_positions,
            global_best_position,
            random_cognitive,
            random_social,
            lower,
            upper,
        )
    ]
    native = _extension()
    byte_function = getattr(native, "campaign_pso_propose_bytes", None)
    if byte_function is None:
        byte_function = getattr(native, "campaign_pso_propose_buffers", None)
    if byte_function is not None:
        next_positions, next_velocities = byte_function(
            *(_f64_bytes(item) for item in arrays),
            float(inertia),
            float(cognitive),
            float(social),
        )
        return (
            _f64_from_bytes(next_positions, np.asarray(positions).shape),
            _f64_from_bytes(next_velocities, np.asarray(velocities).shape),
        )
    next_positions, next_velocities = _native_function("campaign_pso_propose")(
        *(np.ascontiguousarray(item, dtype="<f8").ravel().tolist() for item in arrays),
        float(inertia),
        float(cognitive),
        float(social),
    )
    return (
        np.asarray(next_positions, dtype=np.float64).reshape(np.asarray(positions).shape),
        np.asarray(next_velocities, dtype=np.float64).reshape(np.asarray(velocities).shape),
    )


def aggregate_metric(values: list[float] | np.ndarray, *, attempted_count: int) -> dict[str, float | int | None]:
    """Return the metric statistics used by the Scale campaign summary."""

    numeric = np.asarray(values, dtype=np.float64).reshape(-1)
    if np.any(~np.isfinite(numeric)):
        raise ValueError("campaign metric aggregation expects finite values")
    native = _extension()
    byte_function = getattr(native, "campaign_aggregate_metric_bytes", None)
    if byte_function is not None:
        raw = byte_function(_f64_bytes(numeric), int(attempted_count))
    else:
        raw = _native_function("campaign_aggregate_metric")(
            np.ascontiguousarray(numeric, dtype="<f8").tolist(), int(attempted_count),
        )
    keys = ("count", "missing_count", "min", "p05", "mean", "median", "p95", "max", "std")
    return dict(zip(keys, raw, strict=True))


__all__ = [
    "aggregate_metric",
    "closest_sampled_approach",
    "pso_propose",
    "pso_update",
    "propagate_pair_grid",
    "relative_metrics",
]
