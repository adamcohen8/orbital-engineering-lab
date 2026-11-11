"""One prograde impulse at periapsis, targeted to the first apogee event.

This is a bounded numerical targeter, not a general maneuver optimizer. Its
trajectory evaluations use OEL's checked-in two-body propagator.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from sim.dynamics.orbit.elements import rv_to_coe_eci
from sim.dynamics.orbit.environment import EARTH_MU_KM3_S2, EARTH_RADIUS_KM
from sim.dynamics.orbit.two_body import propagate_two_body_rk4
from sim.trajectory_targeting.scalar import EventObservation, ScalarTargetProblem, solve_scalar_target


@dataclass(frozen=True)
class ApogeeTargetSpec:
    position_eci_km: tuple[float, float, float]
    velocity_eci_km_s: tuple[float, float, float]
    target_apogee_radius_km: float
    max_delta_v_m_s: float
    tolerance_km: float = 0.01
    coast_step_s: float = 30.0
    max_coast_s: float = 20_000.0
    max_iterations: int = 24

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> ApogeeTargetSpec:
        allowed = {
            "schema_version",
            "position_eci_km",
            "velocity_eci_km_s",
            "target_apogee_radius_km",
            "max_delta_v_m_s",
            "tolerance_km",
            "coast_step_s",
            "max_coast_s",
            "max_iterations",
        }
        unknown = sorted(set(data) - allowed)
        if unknown:
            raise ValueError(f"Unknown apogee-target fields: {', '.join(unknown)}")
        if data.get("schema_version") != "oel.trajectory_target.apogee.v1":
            raise ValueError("schema_version must be oel.trajectory_target.apogee.v1")
        try:
            position = tuple(float(value) for value in data["position_eci_km"])
            velocity = tuple(float(value) for value in data["velocity_eci_km_s"])
            kwargs = {
                name: float(data[name])
                for name in (
                    "target_apogee_radius_km",
                    "max_delta_v_m_s",
                    "tolerance_km",
                    "coast_step_s",
                    "max_coast_s",
                )
                if name in data
            }
            if "max_iterations" in data:
                iterations = float(data["max_iterations"])
                if not iterations.is_integer():
                    raise ValueError("max_iterations must be an integer")
                kwargs["max_iterations"] = int(iterations)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Apogee target requires numeric position and velocity vectors") from exc
        return cls(position_eci_km=position, velocity_eci_km_s=velocity, **kwargs)


@dataclass(frozen=True)
class ApogeeTargetResult:
    status: str
    reason: str
    delta_v_m_s: float | None
    delta_v_eci_m_s: tuple[float, float, float] | None
    first_apogee_time_s: float | None
    first_apogee_radius_km: float | None
    target_residual_km: float | None
    verification_residual_km: float | None
    post_burn_state_eci: tuple[float, ...] | None
    apogee_state_eci: tuple[float, ...] | None
    iterations: tuple[dict[str, float], ...]
    assumptions: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _validated_initial_state(spec: ApogeeTargetSpec) -> np.ndarray:
    state = np.array((*spec.position_eci_km, *spec.velocity_eci_km_s), dtype=float)
    if state.shape != (6,) or not np.isfinite(state).all():
        raise ValueError("Initial ECI position and velocity must each have three finite values")
    scalars = (
        spec.target_apogee_radius_km,
        spec.max_delta_v_m_s,
        spec.tolerance_km,
        spec.coast_step_s,
        spec.max_coast_s,
    )
    if not all(math.isfinite(float(value)) for value in scalars):
        raise ValueError("Target and solver bounds must be finite")
    if (
        spec.target_apogee_radius_km <= EARTH_RADIUS_KM
        or spec.max_delta_v_m_s <= 0.0
        or spec.tolerance_km <= 0.0
        or not 0.1 <= spec.coast_step_s <= 3600.0
        or spec.max_coast_s <= spec.coast_step_s
        or spec.max_coast_s > 86_400.0
        or spec.max_coast_s / spec.coast_step_s > 5000.0
        or not 1 <= spec.max_iterations <= 100
    ):
        raise ValueError("Target radius and solver bounds must be positive and within resource limits")
    elements = rv_to_coe_eci(state[:3], state[3:])
    radius = float(np.linalg.norm(state[:3]))
    radial_speed = float(np.dot(state[:3], state[3:]) / radius)
    periapsis_radius = elements.a_km * (1.0 - elements.ecc)
    if radius <= EARTH_RADIUS_KM or abs(radial_speed) > 1.0e-6 or abs(radius - periapsis_radius) > 0.01:
        raise ValueError("Initial state must be Earth-centered and at periapsis (or circular)")
    if spec.target_apogee_radius_km <= elements.a_km * (1.0 + elements.ecc) + spec.tolerance_km:
        raise ValueError("This prograde targeter requires an apogee increase beyond tolerance")
    return state


def _first_apsis(state: np.ndarray, *, kind: str, step_s: float, max_coast_s: float) -> tuple[float, np.ndarray]:
    """Locate the first named radial-velocity crossing by bisection."""
    if kind not in {"apogee", "perigee"}:
        raise ValueError("kind must be apogee or perigee")
    sign = 1.0 if kind == "apogee" else -1.0
    t_left = 0.0
    left = state.copy()
    radial_left = sign * float(np.dot(left[:3], left[3:]))
    while t_left < max_coast_s:
        dt = min(step_s, max_coast_s - t_left)
        right = propagate_two_body_rk4(left, dt, EARTH_MU_KM3_S2, np.zeros(3))
        radial_right = sign * float(np.dot(right[:3], right[3:]))
        if radial_left > 0.0 and radial_right <= 0.0:
            lo, hi = 0.0, dt
            # Keep the left state fixed; each midpoint is propagated by the
            # same OEL integrator, so the event search never changes physics.
            for _ in range(36):
                mid = 0.5 * (lo + hi)
                at_mid = propagate_two_body_rk4(left, mid, EARTH_MU_KM3_S2, np.zeros(3))
                if sign * float(np.dot(at_mid[:3], at_mid[3:])) > 0.0:
                    lo = mid
                else:
                    hi = mid
            event_dt = 0.5 * (lo + hi)
            event = propagate_two_body_rk4(left, event_dt, EARTH_MU_KM3_S2, np.zeros(3))
            return t_left + event_dt, event
        left = right
        t_left += dt
        radial_left = radial_right
    raise ValueError(f"No first {kind} event occurred within max_coast_s")


def _first_apogee(state: np.ndarray, *, step_s: float, max_coast_s: float) -> tuple[float, np.ndarray]:
    return _first_apsis(state, kind="apogee", step_s=step_s, max_coast_s=max_coast_s)


def target_apogee(spec: ApogeeTargetSpec) -> ApogeeTargetResult:
    """Solve one tangential impulse and verify its first apogee by repropagation."""
    initial = _validated_initial_state(spec)
    vhat = initial[3:] / np.linalg.norm(initial[3:])
    assumptions: dict[str, Any] = {
        "frame": "Earth-centered inertial Cartesian (ECI)",
        "units": {"position": "km", "velocity": "km/s", "delta_v": "m/s", "time": "s"},
        "force_model": "OEL two-body RK4; Earth mu from sim.dynamics.orbit.environment",
        "earth_mu_km3_s2": EARTH_MU_KM3_S2,
        "burn": "instantaneous prograde impulse at initial periapsis",
        "goal": "radius at first apogee event after the burn",
        "non_claims": [
            "finite burns",
            "perturbed trajectories",
            "global optimization",
            "operational maneuver authority",
        ],
    }
    cache: dict[float, tuple[float, float | None, np.ndarray | None]] = {}

    def evaluate(dv_m_s: float) -> tuple[float, float | None, np.ndarray | None]:
        if dv_m_s in cache:
            return cache[dv_m_s]
        trial = initial.copy()
        trial[3:] += vhat * (dv_m_s / 1000.0)
        elements = rv_to_coe_eci(trial[:3], trial[3:])
        if dv_m_s == 0.0 and elements.ecc < 1.0e-9:
            result = (float(np.linalg.norm(trial[:3])), None, None)
        else:
            event_time, event = _first_apogee(trial, step_s=spec.coast_step_s, max_coast_s=spec.max_coast_s)
            result = (float(np.linalg.norm(event[:3])), event_time, event)
        cache[dv_m_s] = result
        return result

    def observe(dv_m_s: float) -> EventObservation:
        radius, time_s, event = evaluate(dv_m_s)
        return EventObservation(radius, time_s, None if event is None else tuple(float(x) for x in event))

    correction = solve_scalar_target(
        ScalarTargetProblem(
            variable_name="prograde_delta_v_m_s",
            event_name="first_apogee",
            metric_name="radius_km",
            target_value=spec.target_apogee_radius_km,
            lower_bound=0.0,
            upper_bound=float(spec.max_delta_v_m_s),
            tolerance=spec.tolerance_km,
            max_iterations=spec.max_iterations,
        ),
        observe,
    )
    rows = tuple(
        {
            "iteration": row["iteration"],
            "delta_v_m_s": row["variable_value"],
            "apogee_radius_km": row["event_value"],
            "residual_km": row["residual"],
        }
        for row in correction.iterations
    )
    if correction.status != "converged":
        return ApogeeTargetResult(
            correction.status,
            correction.reason,
            None,
            None,
            None,
            None,
            correction.residual,
            None,
            None,
            None,
            rows,
            assumptions,
        )

    dv = correction.variable_value
    assert dv is not None

    post_burn = initial.copy()
    post_burn[3:] += vhat * (dv / 1000.0)
    radius, event_time, event = evaluate(dv)
    assert event_time is not None and event is not None
    verification_time, verification = (event_time, event)
    if event_time > 0.0:
        verification_time, verification = _first_apogee(
            post_burn,
            step_s=spec.coast_step_s / 2.0,
            max_coast_s=spec.max_coast_s,
        )
    verification_residual = float(np.linalg.norm(verification[:3]) - spec.target_apogee_radius_km)
    if abs(verification_residual) > spec.tolerance_km or abs(verification_time - event_time) > spec.coast_step_s:
        return ApogeeTargetResult(
            "verification_failed",
            "Independent-step repropagation missed target tolerance",
            float(dv),
            tuple(float(x) for x in vhat * dv),
            float(event_time),
            radius,
            float(radius - spec.target_apogee_radius_km),
            verification_residual,
            tuple(float(x) for x in post_burn),
            tuple(float(x) for x in event),
            rows,
            assumptions,
        )
    return ApogeeTargetResult(
        "converged",
        "Target and repropagation satisfy tolerance",
        float(dv),
        tuple(float(x) for x in vhat * dv),
        float(event_time),
        radius,
        float(radius - spec.target_apogee_radius_km),
        verification_residual,
        tuple(float(x) for x in post_burn),
        tuple(float(x) for x in event),
        rows,
        assumptions,
    )
