"""First-contact classification against an Earth-fixed reference surface."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import TYPE_CHECKING, Callable

import numpy as np

from sim.dynamics.orbit.environment import EARTH_RADIUS_KM
from sim.dynamics.orbit.frames import FrameContext, transform_position, transform_state
from sim.utils.geodesy import ecef_to_enu_rotation, ecef_to_geodetic_deg_km
from sim.utils.quaternion import quaternion_to_dcm_bn

if TYPE_CHECKING:
    from sim.rocket.models import RocketState


@dataclass(frozen=True)
class RocketTouchdownLimits:
    max_normal_energy_j: float
    surface_altitude_m: float = 0.0
    contact_height_m: float = 0.0
    max_lateral_speed_m_s: float | None = None
    max_tilt_deg: float | None = None
    max_angular_rate_deg_s: float | None = None

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            if value is None and name.startswith("max_") and name != "max_normal_energy_j":
                continue
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not isfinite(value):
                raise ValueError(f"{name} must be a finite number.")
            if name != "surface_altitude_m" and value < 0.0:
                raise ValueError(f"{name} must be non-negative.")
            if name == "max_tilt_deg" and value > 180.0:
                raise ValueError("max_tilt_deg must be at most 180.")


@dataclass(frozen=True)
class RocketTouchdown:
    state: RocketState
    outcome: str
    evidence: dict

    @property
    def termination_reason(self) -> str:
        return f"rocket_{self.outcome}"


def locate_touchdown(
    start: RocketState,
    end: RocketState,
    propagate: Callable[[float], RocketState],
    *,
    limits: RocketTouchdownLimits,
    context: FrameContext,
    use_wgs84: bool,
    radius_km: float,
    upright_axis_body: np.ndarray,
) -> RocketTouchdown | None:
    """Refine a bracketed contact with the same command and physics as the trial step."""
    trigger_alt_km = (limits.surface_altitude_m + limits.contact_height_m) / 1000.0
    use_ellipsoid = use_wgs84 and radius_km == EARTH_RADIUS_KM

    def clearance_km(state: RocketState) -> float:
        if use_ellipsoid:
            position = transform_position(state.position_eci_km, "eci", "ecef", t_s=state.t_s, context=context)
            altitude = ecef_to_geodetic_deg_km(position)[2]
        else:
            altitude = float(np.linalg.norm(state.position_eci_km)) - radius_km
        return altitude - trigger_alt_km

    # One micrometre of clearance avoids classifying roundoff on the pad.
    tolerance_km = 1.0e-9
    start_clearance = clearance_km(start)
    if start_clearance < -tolerance_km:
        raise ValueError("Rocket touchdown step starts below the configured contact surface.")
    end_clearance = clearance_km(end)
    if end_clearance > tolerance_km:
        return None
    if start_clearance <= tolerance_km and end_clearance >= -tolerance_km:
        return None

    contact = end
    lo_s = 0.0
    hi_s = float(end.t_s - start.t_s)
    # Using a positive interval also covers descent from a state exactly at
    # contact without appending a duplicate timestamp to the history.
    while hi_s - lo_s > 1.0e-7:
        mid_s = (lo_s + hi_s) / 2.0
        trial = propagate(mid_s)
        if clearance_km(trial) > tolerance_km:
            lo_s = mid_s
        else:
            hi_s = mid_s
            contact = trial

    return classify_touchdown(
        contact, limits=limits, context=context, use_ellipsoid=use_ellipsoid,
        upright_axis_body=upright_axis_body, bracket_start_s=start.t_s + lo_s,
    )


def classify_touchdown(
    contact: RocketState,
    *,
    limits: RocketTouchdownLimits,
    context: FrameContext,
    use_ellipsoid: bool,
    upright_axis_body: np.ndarray,
    bracket_start_s: float,
    point=None,
    failure: str | None = None,
) -> RocketTouchdown:
    position_ecef, velocity_ecef = transform_state(
        contact.position_eci_km, contact.velocity_eci_km_s, "eci", "ecef",
        t_s=contact.t_s, context=context,
    )
    latitude_deg, longitude_deg, _ = ecef_to_geodetic_deg_km(position_ecef)
    up_ecef = (
        ecef_to_enu_rotation(latitude_deg, longitude_deg)[2]
        if use_ellipsoid else position_ecef / np.linalg.norm(position_ecef)
    )
    if point is not None:
        up_ecef = point.normal_ecef
    velocity_m_s = 1000.0 * velocity_ecef
    signed_normal_speed = float(velocity_m_s @ up_ecef)
    normal_speed = max(0.0, -signed_normal_speed)
    lateral_speed = float(np.linalg.norm(velocity_m_s - signed_normal_speed * up_ecef))
    upright_eci = quaternion_to_dcm_bn(contact.attitude_quat_bn).T @ upright_axis_body
    upright_ecef = transform_position(upright_eci, "eci", "ecef", t_s=contact.t_s, context=context)
    cos_tilt = float(upright_ecef @ up_ecef) / float(np.linalg.norm(upright_ecef))
    tilt = float(np.rad2deg(np.arccos(np.clip(cos_tilt, -1.0, 1.0))))
    angular_rate = float(np.rad2deg(np.linalg.norm(contact.angular_rate_body_rad_s)))
    energy = 0.5 * contact.mass_kg * normal_speed**2
    measurements = {
        "normal_energy_j": energy,
        "lateral_speed_m_s": lateral_speed,
        "tilt_deg": tilt,
        "angular_rate_deg_s": angular_rate,
    }
    checks = {
        name: {"value": value, "limit": limit, "passed": bool(value <= limit)}
        for name, value in measurements.items()
        if (limit := getattr(limits, f"max_{name}")) is not None
    }
    failed_limits = [name for name, check in checks.items() if not check["passed"]]
    if failure is not None:
        failed_limits.append(failure)
    outcome = "landing" if not failed_limits else "crash"
    return RocketTouchdown(contact, outcome, {
        "outcome": outcome,
        "time_s": contact.t_s,
        "bracket_start_s": bracket_start_s,
        "bracket_end_s": contact.t_s,
        "surface_altitude_m": limits.surface_altitude_m,
        "contact_height_m": limits.contact_height_m,
        "latitude_deg": latitude_deg,
        "longitude_deg": longitude_deg,
        "mass_kg": contact.mass_kg,
        "position_eci_km": contact.position_eci_km.tolist(),
        "velocity_eci_km_s": contact.velocity_eci_km_s.tolist(),
        "velocity_ecef_m_s": velocity_m_s.tolist(),
        "attitude_quat_bn": contact.attitude_quat_bn.tolist(),
        "angular_rate_body_rad_s": contact.angular_rate_body_rad_s.tolist(),
        "normal_closing_speed_m_s": normal_speed,
        **measurements,
        "checks": checks,
        "failed_limits": failed_limits,
    })
