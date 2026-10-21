from __future__ import annotations

import numpy as np

from sim.dynamics.orbit.sdp4 import sdp4_initialize, sdp4_propagate_teme, sdp4_propagate_teme_from_context
from sim.dynamics.orbit.sgp4 import (
    SGP4_DEEP_SPACE_PERIOD_THRESHOLD_MIN,
    SGP4BatchResult,
    SGP4State,
    sgp4_orbital_period_min,
    sgp4_propagate_teme,
    sgp4_propagate_teme_batch_numba,
    sgp4_propagate_teme_batch_reference,
)
from sim.dynamics.orbit.tle import TLEElements

OGP_DEEP_SPACE_PERIOD_THRESHOLD_MIN = SGP4_DEEP_SPACE_PERIOD_THRESHOLD_MIN


def ogp_regime_for_elements(elements: TLEElements) -> str:
    """Return the OGP regime for a TLE/mean-element product."""

    return "sdp4" if sgp4_orbital_period_min(elements) >= OGP_DEEP_SPACE_PERIOD_THRESHOLD_MIN else "sgp4"


def ogp_propagator_name_for_elements(elements: TLEElements) -> str:
    regime = ogp_regime_for_elements(elements)
    return "OGP-SDP4" if regime == "sdp4" else "OGP-SGP4"


def ogp_propagate_teme(elements: TLEElements, tsince_min: float, *, backend: str = "rust") -> SGP4State:
    """Dispatch OGP propagation to the supported near/deep-space regime path."""

    if backend == "rust":
        from sim.rust_ogp_backend import RustOGPContext

        try:
            position, velocity = RustOGPContext(elements).propagate(tsince_min)
        except (RuntimeError, ValueError) as exc:
            return SGP4State(np.zeros(3), np.zeros(3), str(exc))
        return SGP4State(position, velocity)
    if backend != "python":
        raise ValueError("OGP backend must be python or rust.")
    if ogp_regime_for_elements(elements) == "sdp4":
        return sdp4_propagate_teme(elements, tsince_min)
    return sgp4_propagate_teme(elements, tsince_min)


def ogp_propagate_teme_batch_reference(
    elements: list[TLEElements] | tuple[TLEElements, ...],
    tsince_min: np.ndarray | list[float] | tuple[float, ...],
) -> SGP4BatchResult:
    """Propagate many TLEs through the scalar OGP dispatcher.

    This is the correctness reference contract for mixed near-Earth/deep-space
    OGP batches. Optimized backends can target this shape once the scalar OGP
    path is the trusted baseline.
    """

    element_list = list(elements)
    object_count = len(element_list)
    if object_count <= 0:
        raise ValueError("Batched OGP propagation requires at least one element set.")
    time_grid = _coerce_ogp_batch_times(tsince_min, object_count=object_count)
    sample_count = int(time_grid.shape[1])
    positions = np.zeros((object_count, sample_count, 3), dtype=float)
    velocities = np.zeros((object_count, sample_count, 3), dtype=float)
    errors = np.full((object_count, sample_count), "", dtype=object)
    sdp4_contexts = {}
    for object_index, element in enumerate(element_list):
        if ogp_regime_for_elements(element) == "sdp4":
            try:
                sdp4_contexts[object_index] = sdp4_initialize(element)
            except Exception as exc:
                errors[object_index, :] = str(exc)
    for object_index, element in enumerate(element_list):
        for sample_index, offset_min in enumerate(time_grid[object_index]):
            if errors[object_index, sample_index]:
                continue
            if object_index in sdp4_contexts:
                state = sdp4_propagate_teme_from_context(sdp4_contexts[object_index], float(offset_min))
            else:
                state = sgp4_propagate_teme(element, float(offset_min))
            positions[object_index, sample_index, :] = np.asarray(state.position_teme_km, dtype=float)
            velocities[object_index, sample_index, :] = np.asarray(state.velocity_teme_km_s, dtype=float)
            errors[object_index, sample_index] = "" if state.error is None else str(state.error)
    return SGP4BatchResult(
        backend="ogp_scalar_reference",
        tsince_min=time_grid,
        position_teme_km=positions,
        velocity_teme_km_s=velocities,
        errors=errors,
    )


def ogp_propagate_teme_batch_accelerated(
    elements: list[TLEElements] | tuple[TLEElements, ...],
    tsince_min: np.ndarray | list[float] | tuple[float, ...],
    *,
    backend: str = "auto",
    workers: int | None = None,
) -> SGP4BatchResult:
    """Propagate a mixed OGP batch with safe available acceleration.

    Near-Earth OGP-SGP4 rows use the existing Numba batch backend when available;
    deep-space OGP-SDP4 rows reuse one initialized scalar context per object.
    This deliberately avoids a risky vector rewrite of the stateful resonance
    branch while still removing repeated SDP4 setup from batch workloads.
    """

    if backend == "rust":
        return ogp_propagate_teme_batch_rust(elements, tsince_min, workers=workers)
    if workers not in (None, 0):
        raise ValueError("OGP batch workers apply only to the rust backend.")
    if backend != "auto":
        raise ValueError("Accelerated OGP backend must be auto or rust.")
    element_list = list(elements)
    object_count = len(element_list)
    if object_count <= 0:
        raise ValueError("Batched OGP propagation requires at least one element set.")
    time_grid = _coerce_ogp_batch_times(tsince_min, object_count=object_count)
    sample_count = int(time_grid.shape[1])
    positions = np.zeros((object_count, sample_count, 3), dtype=float)
    velocities = np.zeros((object_count, sample_count, 3), dtype=float)
    errors = np.full((object_count, sample_count), "", dtype=object)

    sgp4_indices = [idx for idx, element in enumerate(element_list) if ogp_regime_for_elements(element) == "sgp4"]
    sgp4_index_set = set(sgp4_indices)
    sdp4_indices = [idx for idx in range(object_count) if idx not in sgp4_index_set]

    sgp4_backend = ""
    if sgp4_indices:
        sgp4_elements = [element_list[idx] for idx in sgp4_indices]
        sgp4_times = time_grid[sgp4_indices, :]
        try:
            sgp4_batch = sgp4_propagate_teme_batch_numba(sgp4_elements, sgp4_times)
        except RuntimeError:
            sgp4_batch = sgp4_propagate_teme_batch_reference(sgp4_elements, sgp4_times)
        sgp4_backend = sgp4_batch.backend
        for source_row, target_index in enumerate(sgp4_indices):
            positions[target_index, :, :] = sgp4_batch.position_teme_km[source_row, :, :]
            velocities[target_index, :, :] = sgp4_batch.velocity_teme_km_s[source_row, :, :]
            errors[target_index, :] = sgp4_batch.errors[source_row, :]

    for object_index in sdp4_indices:
        try:
            context = sdp4_initialize(element_list[object_index])
        except Exception as exc:
            errors[object_index, :] = str(exc)
            continue
        for sample_index, offset_min in enumerate(time_grid[object_index]):
            state = sdp4_propagate_teme_from_context(context, float(offset_min))
            positions[object_index, sample_index, :] = np.asarray(state.position_teme_km, dtype=float)
            velocities[object_index, sample_index, :] = np.asarray(state.velocity_teme_km_s, dtype=float)
            errors[object_index, sample_index] = "" if state.error is None else str(state.error)

    backend_parts = ["ogp_mixed"]
    if sgp4_backend:
        backend_parts.append(f"sgp4_{sgp4_backend}")
    if sdp4_indices:
        backend_parts.append("sdp4_context")
    return SGP4BatchResult(
        backend="+".join(backend_parts),
        tsince_min=time_grid,
        position_teme_km=positions,
        velocity_teme_km_s=velocities,
        errors=errors,
    )


def ogp_propagate_teme_batch_rust(
    elements: list[TLEElements] | tuple[TLEElements, ...],
    tsince_min: np.ndarray | list[float] | tuple[float, ...],
    *,
    workers: int | None = None,
) -> SGP4BatchResult:
    """Propagate mixed OGP regimes with one Rust initialization per object."""

    from sim.rust_ogp_backend import propagate_batch_with_telemetry

    element_list = list(elements)
    if not element_list:
        raise ValueError("Batched OGP propagation requires at least one element set.")
    time_grid = _coerce_ogp_batch_times(tsince_min, object_count=len(element_list))
    positions, velocities, errors, execution = propagate_batch_with_telemetry(element_list, time_grid, workers=workers)
    return SGP4BatchResult(
        backend="ogp_rust",
        tsince_min=time_grid,
        position_teme_km=positions,
        velocity_teme_km_s=velocities,
        errors=errors,
        execution_metadata=execution,
    )


def _coerce_ogp_batch_times(
    tsince_min: np.ndarray | list[float] | tuple[float, ...],
    *,
    object_count: int,
) -> np.ndarray:
    times = np.asarray(tsince_min, dtype=float)
    if times.ndim == 0:
        times = times.reshape(1)
    if times.ndim == 1:
        if times.size <= 0:
            raise ValueError("Batched OGP propagation requires at least one time sample.")
        if not np.all(np.isfinite(times)):
            raise ValueError("Batched OGP time offsets must be finite.")
        return np.broadcast_to(times.reshape(1, -1), (object_count, times.size)).copy()
    if times.ndim == 2:
        if times.shape[0] != object_count:
            raise ValueError(
                "Batched OGP per-object time grid must have shape "
                f"(object_count, sample_count); got {times.shape} for {object_count} objects."
            )
        if times.shape[1] <= 0:
            raise ValueError("Batched OGP propagation requires at least one time sample.")
        if not np.all(np.isfinite(times)):
            raise ValueError("Batched OGP time offsets must be finite.")
        return np.array(times, dtype=float, copy=True)
    raise ValueError("Batched OGP time offsets must be a 1-D or 2-D array.")
