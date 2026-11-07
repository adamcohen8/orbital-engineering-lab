"""Deterministic two-object closest-approach and encounter geometry.

The implementation refines closest approach inside each supplied trajectory
interval with cubic Hermite interpolation.  It therefore uses the position and
velocity states already produced by OEL instead of substituting a second force
model during refinement.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


class ConjunctionGeometryError(ValueError):
    """Raised when closest-approach inputs cannot define valid geometry."""


@dataclass(frozen=True)
class StateHistory:
    """One ECI Cartesian state history in km and km/s."""

    times_s: tuple[float, ...]
    states_eci_km_km_s: tuple[tuple[float, float, float, float, float, float], ...]
    incoming_velocities_eci_km_s: tuple[tuple[float, float, float], ...]

    @classmethod
    def from_arrays(
        cls,
        times_s: Sequence[float],
        states: Sequence[Sequence[float]],
        *,
        incoming_velocities_eci_km_s: Sequence[Sequence[float]] | None = None,
    ) -> StateHistory:
        times = np.asarray(times_s, dtype=float)
        values = np.asarray(states, dtype=float)
        if times.ndim != 1 or times.size < 2:
            raise ConjunctionGeometryError("A state history requires at least two time samples.")
        if values.shape != (times.size, 6):
            raise ConjunctionGeometryError("State history values must have shape (sample_count, 6).")
        if not np.all(np.isfinite(times)) or not np.all(np.isfinite(values)):
            raise ConjunctionGeometryError("State histories must contain only finite values.")
        if np.any(np.diff(times) <= 0.0):
            raise ConjunctionGeometryError("State-history times must be strictly increasing.")
        incoming = (
            values[:, 3:]
            if incoming_velocities_eci_km_s is None
            else np.asarray(incoming_velocities_eci_km_s, dtype=float)
        )
        if incoming.shape != (times.size, 3) or not np.all(np.isfinite(incoming)):
            raise ConjunctionGeometryError(
                "Incoming state-history velocities must have shape (sample_count, 3) and contain only finite values."
            )
        return cls(
            times_s=tuple(float(value) for value in times),
            states_eci_km_km_s=tuple(tuple(float(value) for value in row) for row in values),
            incoming_velocities_eci_km_s=tuple(tuple(float(value) for value in row) for row in incoming),
        )

    def arrays(self) -> tuple[np.ndarray, np.ndarray]:
        return np.asarray(self.times_s, dtype=float), np.asarray(self.states_eci_km_km_s, dtype=float)


def _hermite_state(left: np.ndarray, right: np.ndarray, dt_s: float, fraction: float) -> np.ndarray:
    u = float(fraction)
    u2 = u * u
    u3 = u2 * u
    h00 = 2.0 * u3 - 3.0 * u2 + 1.0
    h10 = u3 - 2.0 * u2 + u
    h01 = -2.0 * u3 + 3.0 * u2
    h11 = u3 - u2
    position = h00 * left[:3] + h10 * dt_s * left[3:] + h01 * right[:3] + h11 * dt_s * right[3:]
    dh00 = 6.0 * u2 - 6.0 * u
    dh10 = 3.0 * u2 - 4.0 * u + 1.0
    dh01 = -6.0 * u2 + 6.0 * u
    dh11 = 3.0 * u2 - 2.0 * u
    velocity = (dh00 * left[:3] + dh01 * right[:3]) / dt_s + dh10 * left[3:] + dh11 * right[3:]
    return np.hstack((position, velocity))


def _interpolate_interval(history: StateHistory, states: np.ndarray, index: int, fraction: float) -> np.ndarray:
    left = states[index].copy()
    right = states[index + 1].copy()
    right[3:] = np.asarray(history.incoming_velocities_eci_km_s[index + 1], dtype=float)
    dt = float(history.times_s[index + 1] - history.times_s[index])
    return _hermite_state(left, right, dt, fraction)


def interpolate_history(history: StateHistory, time_s: float, *, side: str = "right") -> np.ndarray:
    """Interpolate one continuous history interval.

    At an impulsive discontinuity, ``side='left'`` returns the incoming state
    and the default ``side='right'`` returns the post-impulse state.
    """

    times, states = history.arrays()
    return _interpolate_prepared(history, times, states, time_s, side=side)


def _interpolate_prepared(history: StateHistory, times: np.ndarray, states: np.ndarray,
                          time_s: float, *, side: str = "right") -> np.ndarray:
    """Use operation-local snapshots without changing public arrays() copy semantics."""
    query = float(time_s)
    if query < times[0] or query > times[-1]:
        raise ConjunctionGeometryError("Interpolation time lies outside the supplied history.")
    if side not in {"left", "right"}:
        raise ConjunctionGeometryError("Interpolation side must be 'left' or 'right'.")
    exact = np.flatnonzero(times == query)
    if exact.size and side == "right":
        return states[int(exact[0])].copy()
    if exact.size and int(exact[0]) == 0:
        return states[0].copy()
    search_side = "left" if side == "left" else "right"
    index = min(int(np.searchsorted(times, query, side=search_side)) - 1, times.size - 2)
    index = max(index, 0)
    dt = float(times[index + 1] - times[index])
    fraction = (query - float(times[index])) / dt
    return _interpolate_interval(history, states, index, fraction)


def _stationary_fractions(left_relative: np.ndarray, right_relative: np.ndarray, dt_s: float) -> list[float]:
    """Return every real stationary point of squared separation in [0, 1]."""

    position_left = left_relative[:3]
    velocity_left = left_relative[3:]
    position_right = right_relative[:3]
    velocity_right = right_relative[3:]
    coefficients = np.vstack(
        (
            position_left,
            dt_s * velocity_left,
            -3.0 * position_left - 2.0 * dt_s * velocity_left + 3.0 * position_right - dt_s * velocity_right,
            2.0 * position_left + dt_s * velocity_left - 2.0 * position_right + dt_s * velocity_right,
        )
    )
    derivative = np.vstack((coefficients[1], 2.0 * coefficients[2], 3.0 * coefficients[3]))
    distance_derivative = np.zeros(6, dtype=float)
    for axis in range(3):
        distance_derivative += np.convolve(coefficients[:, axis], derivative[:, axis])
    scale = max(1.0, float(np.max(np.abs(distance_derivative))))
    while distance_derivative.size > 1 and abs(float(distance_derivative[-1])) <= 1.0e-14 * scale:
        distance_derivative = distance_derivative[:-1]
    roots = np.polynomial.polynomial.polyroots(distance_derivative)
    fractions = [0.0, 1.0]
    for root in roots:
        if abs(float(root.imag)) <= 1.0e-9 and -1.0e-12 <= float(root.real) <= 1.0 + 1.0e-12:
            fractions.append(min(max(float(root.real), 0.0), 1.0))
    return sorted({round(value, 15) for value in fractions})


def refine_time_of_closest_approach(primary: StateHistory, secondary: StateHistory, *,
                                    numeric_backend: str = "rust") -> dict[str, Any]:
    """Return the global minimum separation over the histories' common span."""

    backend = str(numeric_backend).strip().lower()
    if backend not in {"python", "rust"}:
        raise ConjunctionGeometryError("numeric_backend must be 'python' or 'rust'.")
    if backend == "rust":
        from sim.rust_conjunction_backend import _kernels

        _kernels()
    primary_times, primary_states = primary.arrays()
    secondary_times, secondary_states = secondary.arrays()
    start = max(float(primary_times[0]), float(secondary_times[0]))
    stop = min(float(primary_times[-1]), float(secondary_times[-1]))
    if not stop > start:
        raise ConjunctionGeometryError("The two histories do not overlap in time.")
    breakpoints = np.unique(
        np.concatenate(
            (
                primary_times[(primary_times >= start) & (primary_times <= stop)],
                secondary_times[(secondary_times >= start) & (secondary_times <= stop)],
                np.array([start, stop]),
            )
        )
    )
    if backend == "rust":
        from sim.rust_conjunction_backend import hermite_batch, history_context, interpolate

        primary_context = history_context(primary_times, primary_states, primary.incoming_velocities_eci_km_s)
        secondary_context = history_context(secondary_times, secondary_states, secondary.incoming_velocities_eci_km_s)
        relative_lefts = (interpolate(primary_context, breakpoints[:-1], side="right")
                          - interpolate(secondary_context, breakpoints[:-1], side="right"))
        relative_rights = (interpolate(primary_context, breakpoints[1:], side="left")
                           - interpolate(secondary_context, breakpoints[1:], side="left"))
    pending = []
    cases = []
    candidates: list[tuple[float, float, bool, int, int, str]] = []
    evaluations = 0
    for index, (left, right) in enumerate(zip(breakpoints[:-1], breakpoints[1:], strict=True)):
        duration = float(right - left)
        if backend == "rust":
            relative_left, relative_right = relative_lefts[index], relative_rights[index]
        else:
            primary_left = _interpolate_prepared(primary, primary_times, primary_states, float(left), side="right")
            primary_right = _interpolate_prepared(primary, primary_times, primary_states, float(right), side="left")
            secondary_left = _interpolate_prepared(secondary, secondary_times, secondary_states, float(left), side="right")
            secondary_right = _interpolate_prepared(secondary, secondary_times, secondary_states, float(right), side="left")
            relative_left = primary_left - secondary_left
            relative_right = primary_right - secondary_right
        fractions = _stationary_fractions(relative_left, relative_right, duration)
        for fraction in fractions:
            time_s = float(left) + fraction * duration
            evaluations += 1
            # A breakpoint has two physically meaningful velocity states.  The
            # left endpoint is post-impulse (the right side of the preceding
            # interval), while the right endpoint is pre-impulse (the left
            # side of this interval).  Keep that side with the candidate so
            # the selected metrics and returned states describe one state.
            candidate_side = "left" if fraction == 1.0 else "right"
            descriptor = (time_s, fraction in {0.0, 1.0}, index, len(fractions) - 2, candidate_side)
            if backend == "rust":
                pending.append(descriptor)
                cases.append(np.concatenate((relative_left, relative_right, [duration, fraction])))
            else:
                relative = _hermite_state(relative_left, relative_right, duration, fraction)
                distance_squared = float(relative[:3] @ relative[:3])
                candidates.append((distance_squared, *descriptor))
    if backend == "rust":
        for relative, descriptor in zip(hermite_batch(cases), pending, strict=True):
            candidates.append((float(relative[:3] @ relative[:3]), *descriptor))
    distance2, time_s, boundary, interval_index, stationary_roots, winner_side = min(
        candidates, key=lambda item: item[0]
    )
    primary_state = _interpolate_prepared(
        primary, primary_times, primary_states, time_s, side=winner_side
    )
    secondary_state = _interpolate_prepared(
        secondary, secondary_times, secondary_states, time_s, side=winner_side
    )
    relative = primary_state - secondary_state
    # Recompute the metric from the same one-sided state returned to callers.
    # This keeps range-rate and miss-distance evidence internally consistent
    # even when an impulse lies at the winning interval boundary.
    distance2 = float(relative[:3] @ relative[:3])
    r_dot_v = float(relative[:3] @ relative[3:])
    result = {
        "time_s": time_s,
        "miss_distance_km": math.sqrt(max(distance2, 0.0)),
        "relative_speed_km_s": float(np.linalg.norm(relative[3:])),
        "relative_position_eci_km": relative[:3].tolist(),
        "relative_velocity_eci_km_s": relative[3:].tolist(),
        "relative_position_dot_velocity_km2_s": r_dot_v,
        "relative_range_rate_km_s": None
        if distance2 <= 0.0
        else r_dot_v / math.sqrt(max(distance2, 0.0)),
        "primary_state_eci_km_km_s": primary_state.tolist(),
        "secondary_state_eci_km_km_s": secondary_state.tolist(),
        "interval_start_s": float(breakpoints[interval_index]),
        "interval_end_s": float(breakpoints[interval_index + 1]),
        "at_search_boundary": bool(boundary and (time_s == start or time_s == stop)),
        "method": "piecewise_cubic_hermite_stationary_roots",
        "resources": {
            "intervals": int(breakpoints.size - 1),
            "distance_evaluations": evaluations,
            "winning_interval_stationary_roots": stationary_roots,
        },
    }
    if backend == "rust":
        result["numeric_backend"] = "rust"
    return result


def encounter_frame(
    relative_position_eci_km: Sequence[float], relative_velocity_eci_km_s: Sequence[float]
) -> dict[str, Any]:
    """Construct an orthonormal encounter frame with z along relative velocity."""

    position = np.asarray(relative_position_eci_km, dtype=float)
    velocity = np.asarray(relative_velocity_eci_km_s, dtype=float)
    if position.shape != (3,) or velocity.shape != (3,) or not np.all(np.isfinite(np.hstack((position, velocity)))):
        raise ConjunctionGeometryError("Encounter position and velocity must be finite three-vectors.")
    speed = float(np.linalg.norm(velocity))
    if speed <= 1.0e-12:
        raise ConjunctionGeometryError("Encounter geometry requires nonzero relative velocity.")
    z_hat = velocity / speed
    projected = position - float(position @ z_hat) * z_hat
    projected_norm = float(np.linalg.norm(projected))
    if projected_norm <= 1.0e-12:
        seed = np.array([1.0, 0.0, 0.0])
        if abs(float(seed @ z_hat)) > 0.9:
            seed = np.array([0.0, 1.0, 0.0])
        projected = seed - float(seed @ z_hat) * z_hat
        projected_norm = float(np.linalg.norm(projected))
    x_hat = projected / projected_norm
    y_hat = np.cross(z_hat, x_hat)
    basis = np.vstack((x_hat, y_hat, z_hat))
    coordinates = basis @ position
    return {
        "basis_rows_eci": basis.tolist(),
        "relative_position_encounter_km": coordinates.tolist(),
        "plane_miss_km": coordinates[:2].tolist(),
        "along_relative_velocity_km": float(coordinates[2]),
        "orthonormality_max_abs_error": float(np.max(np.abs(basis @ basis.T - np.eye(3)))),
        "convention": "x=projected_miss,y=z_cross_x,z=relative_velocity",
    }


__all__ = [
    "ConjunctionGeometryError",
    "StateHistory",
    "encounter_frame",
    "interpolate_history",
    "refine_time_of_closest_approach",
]
