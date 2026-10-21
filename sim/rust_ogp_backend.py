"""Optional Rust OGP-SGP4/SDP4 binding on OEL's numeric mean-element contract."""

from __future__ import annotations

import os
from numbers import Integral

import numpy as np

from sim.dynamics.orbit.tle import TLEElements
from sim.rust_orbit_backend import _extension


def _numeric_elements(elements: TLEElements) -> list[float]:
    return [
        float(elements.epoch_jd_utc),
        float(elements.mean_motion_rev_per_day),
        float(elements.eccentricity),
        float(elements.inclination_deg),
        float(elements.raan_deg),
        float(elements.argp_deg),
        float(elements.mean_anomaly_deg),
        float(elements.bstar),
    ]


class RustOGPContext:
    """One initialized Rust OGP record; accepts arbitrary time-query order."""

    def __init__(self, elements: TLEElements) -> None:
        self._native = _extension().OGPContext(_numeric_elements(elements))

    def propagate(self, tsince_min: float) -> tuple[np.ndarray, np.ndarray]:
        if hasattr(self._native, "propagate_bytes"):
            values = np.frombuffer(self._native.propagate_bytes(float(tsince_min)), dtype="<f8").copy()
            return values[:3], values[3:]
        position, velocity = self._native.propagate(float(tsince_min))
        return np.asarray(position, dtype=np.float64), np.asarray(velocity, dtype=np.float64)

    def propagate_many(self, tsince_min: np.ndarray | list[float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        times = np.asarray(tsince_min, dtype=np.float64).reshape(-1)
        if hasattr(self._native, "propagate_many_bytes"):
            positions, velocities, sparse_errors = self._native.propagate_many_bytes(np.asarray(times, dtype="<f8").tobytes())
            errors = np.full(times.shape, "", dtype=object)
            for index, error in sparse_errors:
                errors[index] = error
            return (np.frombuffer(positions, dtype="<f8").reshape(-1, 3).copy(),
                    np.frombuffer(velocities, dtype="<f8").reshape(-1, 3).copy(), errors)
        positions, velocities, errors = self._native.propagate_many(times.tolist())
        return (
            np.asarray(positions, dtype=np.float64).reshape(-1, 3),
            np.asarray(velocities, dtype=np.float64).reshape(-1, 3),
            np.asarray(errors, dtype=object),
        )


def resolve_ogp_batch_workers(requested: int | None = None) -> int:
    """Allocate native workers within the current campaign branch's budget."""
    from sim.execution.worker_budget import (
        CAMPAIGN_WORKER_COUNT_ENV,
        OBJECT_WORKER_BUDGET_ENV,
        TOTAL_PROCESS_BUDGET_ENV,
        worker_budget_value,
    )

    if requested is not None and (isinstance(requested, bool) or not isinstance(requested, Integral) or requested < 0):
        raise ValueError("Rust OGP workers must be a nonnegative integer or None.")
    cpu_cap = max(1, int(os.cpu_count() or 1) - 1)
    workers = min(int(requested or min(4, cpu_cap)), cpu_cap)
    budgets = {}
    for name in (OBJECT_WORKER_BUDGET_ENV, CAMPAIGN_WORKER_COUNT_ENV, TOTAL_PROCESS_BUDGET_ENV):
        raw = worker_budget_value(name)
        if raw in (None, ""):
            continue
        try:
            value = int(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be an integer when set.") from exc
        minimum = 0 if name == OBJECT_WORKER_BUDGET_ENV else 1
        if value < minimum:
            raise ValueError(f"{name} must be at least {minimum} when set.")
        budgets[name] = value
    object_cap = budgets.get(OBJECT_WORKER_BUDGET_ENV)
    campaigns = budgets.get(CAMPAIGN_WORKER_COUNT_ENV, 1)
    if object_cap is not None:
        workers = min(workers, max(1, object_cap))
    elif campaigns > 1:
        # No inner allocation: the outer campaign owns all parallel slots.
        workers = 1
    total = budgets.get(TOTAL_PROCESS_BUDGET_ENV)
    if total is not None:
        # Campaign owners retain their slots while native child threads run.
        cap = max(1, total // campaigns - 1) if campaigns > 1 else total
        workers = min(workers, cap)
    return max(1, workers)


def propagate_batch(
    elements: list[TLEElements], time_grid: np.ndarray, *, workers: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One binary batch boundary, with the older wheel's row API as fallback."""

    positions, velocities, errors, _ = propagate_batch_with_telemetry(elements, time_grid, workers=workers)
    return positions, velocities, errors


def propagate_batch_with_telemetry(
    elements: list[TLEElements], time_grid: np.ndarray, *, workers: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    allocated = resolve_ogp_batch_workers(workers)
    native = _extension()
    shape = (*time_grid.shape, 3)
    errors = np.full(time_grid.shape, "", dtype=object)
    telemetry: dict[str, object] = {
        "workers_requested": int(workers or 0),
        "workers_allocated": allocated,
        "workers_used": 1,
        "execution_backend": "serial",
        "reason": "compatible wheel uses serial propagation",
    }
    if hasattr(native, "ogp_propagate_batch_parallel_bytes"):
        numeric = np.asarray([_numeric_elements(element) for element in elements], dtype="<f8")
        positions, velocities, sparse_errors, used = native.ogp_propagate_batch_parallel_bytes(
            numeric.tobytes(), np.asarray(time_grid, dtype="<f8").tobytes(), time_grid.shape[1], allocated,
        )
        telemetry.update(
            workers_used=used,
            execution_backend="native_threads" if used > 1 else "serial",
            reason=("bounded object workers" if used > 1 else
                    "worker allocation is serial" if allocated == 1 else
                    "small batch, one object, or native CPU limit"),
        )
        for index, error in sparse_errors:
            errors.flat[index] = error
        return (
            np.frombuffer(positions, dtype="<f8").reshape(shape).copy(),
            np.frombuffer(velocities, dtype="<f8").reshape(shape).copy(), errors, telemetry,
        )
    if hasattr(native, "ogp_propagate_batch_bytes"):
        numeric = np.asarray([_numeric_elements(element) for element in elements], dtype="<f8")
        positions, velocities, sparse_errors = native.ogp_propagate_batch_bytes(
            numeric.tobytes(), np.asarray(time_grid, dtype="<f8").tobytes(), time_grid.shape[1],
        )
        for index, error in sparse_errors:
            errors.flat[index] = error
        return (
            np.frombuffer(positions, dtype="<f8").reshape(shape).copy(),
            np.frombuffer(velocities, dtype="<f8").reshape(shape).copy(),
            errors,
            telemetry,
        )
    positions = np.zeros(shape, dtype=float)
    velocities = np.zeros(shape, dtype=float)
    for index, element in enumerate(elements):
        try:
            row_pos, row_vel, row_errors = RustOGPContext(element).propagate_many(time_grid[index])
        except (RuntimeError, ValueError) as exc:
            errors[index, :] = str(exc)
            continue
        positions[index], velocities[index], errors[index] = row_pos, row_vel, row_errors
    return positions, velocities, errors, telemetry


class RustOGPObservationBatch:
    """Immutable observation and whitening arrays held natively for one fit."""

    def __init__(self, times_jd_utc, positions, velocities, factors) -> None:
        native = _extension()
        if not hasattr(native, "OGPObservationBatch"):
            raise RuntimeError("Fused Rust OGP residuals require an oel_rust_orbit wheel with OGPObservationBatch.")
        observations = np.column_stack((positions, velocities))
        if hasattr(native.OGPObservationBatch, "from_bytes"):
            self._native = native.OGPObservationBatch.from_bytes(
                np.asarray(times_jd_utc, dtype="<f8").tobytes(),
                np.asarray(observations, dtype="<f8").tobytes(),
                np.asarray(factors, dtype="<f8").tobytes(),
            )
            return
        self._native = native.OGPObservationBatch(
            np.asarray(times_jd_utc, dtype=float).tolist(),
            observations.ravel().tolist(),
            np.asarray(factors, dtype=float).ravel().tolist(),
        )

    def residual(self, elements: TLEElements) -> np.ndarray:
        values = self._native.residual_bytes(_numeric_elements(elements))
        return np.frombuffer(values, dtype="<f8").copy()
