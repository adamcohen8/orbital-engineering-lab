from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from sim.dynamics.orbit.accelerations import OrbitContext
from sim.dynamics.orbit.propagator import OrbitPropagator
from sim.numeric_backend import normalize_numeric_backend
from sim.utils.frames import eci_relative_to_ric_rect, ric_rect_state_to_eci


@dataclass(frozen=True)
class ONPRelativeVariationalResult:
    """Nonlinear ONP relative state and local initial-state transition matrix."""

    relative_state_ric: np.ndarray
    transition_matrix: np.ndarray
    chief_state_eci_km_s: np.ndarray
    deputy_state_eci_km_s: np.ndarray
    metadata: Mapping[str, Any]


def onp_relative_variational_propagate(
    initial_relative_state_ric: np.ndarray,
    dt_s: float,
    chief_initial_state_eci_km_s: np.ndarray,
    *,
    propagator: OrbitPropagator,
    context: OrbitContext,
    deputy_context: OrbitContext | None = None,
    environment: Mapping[str, Any] | None = None,
    epoch_t_s: float = 0.0,
    integration_substep_s: float = 10.0,
    perturbation_steps: np.ndarray | None = None,
    numeric_backend: str = "rust",
) -> ONPRelativeVariationalResult:
    """Propagate two ONP objects and central-difference the relative flow.

    This is the nonlinear companion to HCW/SS/TH/YA. It differentiates the
    checked-in ONP flow map in rectangular RIC coordinates; it does not create
    or approximate a separate force model. Every configured ONP acceleration
    plugin, including J2, therefore contributes to both the reference state and
    local STM.
    """

    relative0 = np.asarray(initial_relative_state_ric, dtype=float).reshape(6)
    chief0 = np.asarray(chief_initial_state_eci_km_s, dtype=float).reshape(6)
    dt = float(dt_s)
    if not np.isfinite(dt) or dt < 0.0:
        raise ValueError("dt_s must be finite and nonnegative.")
    if not np.isfinite(integration_substep_s) or integration_substep_s <= 0.0:
        raise ValueError("integration_substep_s must be finite and positive.")
    backend = normalize_numeric_backend(numeric_backend, error_message="numeric_backend must be 'python' or 'rust'.")
    steps = (
        np.array([1.0e-5] * 3 + [1.0e-8] * 3, dtype=float)
        if perturbation_steps is None
        else np.asarray(perturbation_steps, dtype=float).reshape(6)
    )
    if np.any(~np.isfinite(steps)) or np.any(steps <= 0.0):
        raise ValueError("perturbation_steps must contain six finite positive values.")
    deputy_ctx = deputy_context or context
    env = dict(environment or {})

    chief_accel0 = _chief_acceleration_or_none(propagator, chief0, epoch_t_s, context, env)
    chief = _propagate(propagator, chief0, epoch_t_s, dt, context, env, integration_substep_s)
    chief_accel_final = _chief_acceleration_or_none(
        propagator,
        chief,
        float(epoch_t_s) + dt,
        context,
        env,
    )

    trial_relatives = [relative0]
    for axis, step in enumerate(steps):
        plus, minus = relative0.copy(), relative0.copy()
        plus[axis] += step
        minus[axis] -= step
        trial_relatives.extend((plus, minus))
    trial_deputies = None
    if backend == "rust":
        from sim.rust_relative_backend import try_orbit_trial_batch

        trial_deputies = try_orbit_trial_batch(
            propagator,
            np.asarray([ric_rect_state_to_eci(
                value, chief0[:3], chief0[3:], chief_accel_eci_km_s2=chief_accel0,
            ) for value in trial_relatives]),
            _interval_widths(dt, integration_substep_s), deputy_ctx,
        )

    def relative_flow(relative_state: np.ndarray, index: int) -> tuple[np.ndarray, np.ndarray]:
        deputy0 = ric_rect_state_to_eci(
            relative_state,
            chief0[:3],
            chief0[3:],
            chief_accel_eci_km_s2=chief_accel0,
        )
        deputy = (trial_deputies[index] if trial_deputies is not None else
                  _propagate(propagator, deputy0, epoch_t_s, dt, deputy_ctx, env, integration_substep_s))
        return eci_relative_to_ric_rect(
            deputy,
            chief,
            chief_accel_eci_km_s2=chief_accel_final,
        ), deputy

    relative, deputy = relative_flow(relative0, 0)
    transition = np.zeros((6, 6), dtype=float)
    plus_minus = np.zeros((6, 12), dtype=float)
    for axis, step in enumerate(steps):
        plus = relative0.copy()
        minus = relative0.copy()
        plus[axis] += step
        minus[axis] -= step
        plus_value = relative_flow(plus, 1 + 2 * axis)[0]
        minus_value = relative_flow(minus, 2 + 2 * axis)[0]
        if backend == "python":
            transition[:, axis] = (plus_value - minus_value) / (2.0 * step)
        else:
            plus_minus[axis, :6] = plus_value
            plus_minus[axis, 6:] = minus_value
    if backend == "rust":
        # Eligible native zonal flows batch above. Other force/plugin owners
        # retain their scalar propagation, including custom callbacks.
        from sim.rust_relative_backend import finite_difference_stm

        transition = finite_difference_stm(plus_minus, steps)

    metadata = {
        "model": "onp_relative_variational",
        "state_basis": "rectangular_RIC_deputy_relative_to_chief",
        "differencing": "centered",
        "perturbation_steps": steps.tolist(),
        "force_model_source": "supplied_OEL_Numerical_Propagator",
        "configured_plugin_count": len(propagator.plugins),
        "integration_substep_s": float(integration_substep_s),
    }
    if backend == "rust":
        metadata["numeric_backend"] = "rust"
    return ONPRelativeVariationalResult(
        relative_state_ric=relative,
        transition_matrix=transition,
        chief_state_eci_km_s=chief,
        deputy_state_eci_km_s=deputy,
        metadata=metadata,
    )


def _chief_acceleration_or_none(
    propagator: OrbitPropagator,
    chief_state: np.ndarray,
    t_s: float,
    context: OrbitContext,
    environment: Mapping[str, Any],
) -> np.ndarray | None:
    """Resolve the supplied ONP force model for an ECI chief state."""

    if propagator.state_frame != "eci":
        return None
    return np.asarray(
        propagator.acceleration_at(
            t_s=float(t_s),
            x_eci=np.asarray(chief_state, dtype=float).reshape(6),
            command_accel_eci_km_s2=np.zeros(3, dtype=float),
            env=dict(environment),
            ctx=context,
        ),
        dtype=float,
    ).reshape(3)


def _interval_widths(dt_s: float, integration_substep_s: float) -> list[float]:
    widths = []
    elapsed = 0.0
    while elapsed < float(dt_s) - 1.0e-12:
        step = min(float(integration_substep_s), float(dt_s) - elapsed)
        widths.append(step)
        elapsed += step
    return widths


def _propagate(
    propagator: OrbitPropagator,
    state: np.ndarray,
    epoch_t_s: float,
    dt_s: float,
    context: OrbitContext,
    environment: Mapping[str, Any],
    integration_substep_s: float,
) -> np.ndarray:
    current = np.asarray(state, dtype=float).copy()
    elapsed = 0.0
    while elapsed < float(dt_s) - 1.0e-12:
        step = min(float(integration_substep_s), float(dt_s) - elapsed)
        current = propagator.propagate(
            current,
            step,
            float(epoch_t_s) + elapsed,
            np.zeros(3),
            dict(environment),
            context,
        )
        elapsed += step
    return current


def _chief_acceleration_or_none(
    propagator: OrbitPropagator,
    chief_state: np.ndarray,
    t_s: float,
    context: OrbitContext,
    environment: Mapping[str, Any],
) -> np.ndarray | None:
    """Resolve the supplied ONP force model for an ECI chief state."""

    if propagator.state_frame != "eci":
        return None
    return np.asarray(
        propagator.acceleration_at(
            t_s=float(t_s),
            x_eci=np.asarray(chief_state, dtype=float).reshape(6),
            command_accel_eci_km_s2=np.zeros(3, dtype=float),
            env=dict(environment),
            ctx=context,
        ),
        dtype=float,
    ).reshape(3)
