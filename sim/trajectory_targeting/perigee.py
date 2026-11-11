"""One retrograde impulse at apogee, targeted to the first perigee event."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from sim.dynamics.orbit.elements import rv_to_coe_eci
from sim.dynamics.orbit.environment import EARTH_MU_KM3_S2, EARTH_RADIUS_KM
from sim.trajectory_targeting.apogee import _first_apsis
from sim.trajectory_targeting.scalar import EventObservation, ScalarTargetProblem, solve_scalar_target


@dataclass(frozen=True)
class PerigeeTargetSpec:
    position_eci_km: tuple[float, float, float]
    velocity_eci_km_s: tuple[float, float, float]
    target_perigee_radius_km: float
    max_delta_v_m_s: float
    tolerance_km: float = 0.01
    coast_step_s: float = 30.0
    max_coast_s: float = 20_000.0
    max_iterations: int = 24

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> PerigeeTargetSpec:
        allowed = {
            "schema_version",
            "position_eci_km",
            "velocity_eci_km_s",
            "target_perigee_radius_km",
            "max_delta_v_m_s",
            "tolerance_km",
            "coast_step_s",
            "max_coast_s",
            "max_iterations",
        }
        unknown = sorted(set(data) - allowed)
        if unknown:
            raise ValueError(f"Unknown perigee-target fields: {', '.join(unknown)}")
        if data.get("schema_version") != "oel.trajectory_target.perigee.v1":
            raise ValueError("schema_version must be oel.trajectory_target.perigee.v1")
        try:
            position = tuple(float(value) for value in data["position_eci_km"])
            velocity = tuple(float(value) for value in data["velocity_eci_km_s"])
            kwargs = {
                name: float(data[name])
                for name in (
                    "target_perigee_radius_km",
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
            raise ValueError("Perigee target requires numeric position and velocity vectors") from exc
        return cls(position_eci_km=position, velocity_eci_km_s=velocity, **kwargs)


@dataclass(frozen=True)
class PerigeeTargetResult:
    status: str
    reason: str
    delta_v_m_s: float | None
    delta_v_eci_m_s: tuple[float, float, float] | None
    first_perigee_time_s: float | None
    first_perigee_radius_km: float | None
    target_residual_km: float | None
    verification_residual_km: float | None
    post_burn_state_eci: tuple[float, ...] | None
    perigee_state_eci: tuple[float, ...] | None
    iterations: tuple[dict[str, float], ...]
    assumptions: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def target_perigee(spec: PerigeeTargetSpec) -> PerigeeTargetResult:
    """Solve one retrograde impulse using OEL propagation to first perigee."""
    initial = np.array((*spec.position_eci_km, *spec.velocity_eci_km_s), dtype=float)
    if initial.shape != (6,) or not np.isfinite(initial).all():
        raise ValueError("Initial ECI position and velocity must each have three finite values")
    scalars = (
        spec.target_perigee_radius_km,
        spec.max_delta_v_m_s,
        spec.tolerance_km,
        spec.coast_step_s,
        spec.max_coast_s,
    )
    if not all(math.isfinite(float(value)) for value in scalars):
        raise ValueError("Target and solver bounds must be finite")
    if (
        spec.target_perigee_radius_km <= EARTH_RADIUS_KM
        or spec.max_delta_v_m_s <= 0.0
        or spec.tolerance_km <= 0.0
        or not 0.1 <= spec.coast_step_s <= 3600.0
        or spec.max_coast_s <= spec.coast_step_s
        or spec.max_coast_s > 86_400.0
        or spec.max_coast_s / spec.coast_step_s > 5000.0
        or not 1 <= spec.max_iterations <= 100
    ):
        raise ValueError("Target radius and solver bounds must be positive and within resource limits")
    elements = rv_to_coe_eci(initial[:3], initial[3:])
    radius = float(np.linalg.norm(initial[:3]))
    radial_speed = float(np.dot(initial[:3], initial[3:]) / radius)
    if (
        radius <= EARTH_RADIUS_KM
        or abs(radial_speed) > 1.0e-6
        or abs(radius - elements.a_km * (1.0 + elements.ecc)) > 0.01
    ):
        raise ValueError("Initial state must be Earth-centered and at apogee (or circular)")
    if spec.target_perigee_radius_km >= elements.a_km * (1.0 - elements.ecc) - spec.tolerance_km:
        raise ValueError("This retrograde targeter requires a perigee decrease beyond tolerance")
    vhat = initial[3:] / np.linalg.norm(initial[3:])
    assumptions = {
        "frame": "Earth-centered inertial Cartesian (ECI)",
        "units": {"position": "km", "velocity": "km/s", "delta_v": "m/s", "time": "s"},
        "force_model": "OEL two-body RK4; Earth mu from sim.dynamics.orbit.environment",
        "earth_mu_km3_s2": EARTH_MU_KM3_S2,
        "burn": "instantaneous retrograde impulse at initial apogee",
        "goal": "radius at first perigee event after the burn",
        "non_claims": [
            "finite burns",
            "perturbed trajectories",
            "global optimization",
            "operational maneuver authority",
        ],
    }

    def observe(dv_m_s: float) -> EventObservation:
        trial = initial.copy()
        trial[3:] -= vhat * (dv_m_s / 1000.0)
        if float(np.dot(trial[3:], vhat)) <= 0.0:
            raise ValueError("Retrograde bound reverses the velocity direction")
        trial_elements = rv_to_coe_eci(trial[:3], trial[3:])
        if trial_elements.a_km * (1.0 - trial_elements.ecc) <= EARTH_RADIUS_KM:
            raise ValueError("Retrograde bound produces an Earth-intersecting trial orbit")
        if dv_m_s == 0.0 and elements.ecc < 1.0e-9:
            return EventObservation(radius, None, None)
        time_s, event = _first_apsis(trial, kind="perigee", step_s=spec.coast_step_s, max_coast_s=spec.max_coast_s)
        return EventObservation(float(np.linalg.norm(event[:3])), time_s, tuple(float(x) for x in event))

    correction = solve_scalar_target(
        ScalarTargetProblem(
            variable_name="retrograde_delta_v_m_s",
            event_name="first_perigee",
            metric_name="radius_km",
            target_value=spec.target_perigee_radius_km,
            lower_bound=0.0,
            upper_bound=spec.max_delta_v_m_s,
            tolerance=spec.tolerance_km,
            max_iterations=spec.max_iterations,
        ),
        observe,
    )
    rows = tuple(
        {
            "iteration": row["iteration"],
            "delta_v_m_s": row["variable_value"],
            "perigee_radius_km": row["event_value"],
            "residual_km": row["residual"],
        }
        for row in correction.iterations
    )
    if correction.status != "converged":
        return PerigeeTargetResult(
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
    observation = correction.observation
    assert dv is not None and observation is not None and observation.time_s is not None
    post_burn = initial.copy()
    post_burn[3:] -= vhat * (dv / 1000.0)
    verification_time, verification = _first_apsis(
        post_burn,
        kind="perigee",
        step_s=spec.coast_step_s / 2.0,
        max_coast_s=spec.max_coast_s,
    )
    verification_residual = float(np.linalg.norm(verification[:3]) - spec.target_perigee_radius_km)
    passed = (
        abs(verification_residual) <= spec.tolerance_km
        and abs(verification_time - observation.time_s) <= spec.coast_step_s
    )
    return PerigeeTargetResult(
        "converged" if passed else "verification_failed",
        "Target and repropagation satisfy tolerance"
        if passed
        else "Independent-step repropagation missed target tolerance",
        float(dv),
        tuple(float(x) for x in -vhat * dv),
        observation.time_s,
        observation.value,
        correction.residual,
        verification_residual,
        tuple(float(x) for x in post_burn),
        observation.state_eci,
        rows,
        assumptions,
    )
