from __future__ import annotations

from dataclasses import dataclass
from threading import RLock

import numpy as np

from sim.dynamics.orbit.atmosphere import density_exponential
from sim.dynamics.orbit.eclipse import resolve_srp_geometry, srp_shadow_factor
from sim.dynamics.orbit.environment import (
    EARTH_J2,
    EARTH_J3,
    EARTH_J4,
    EARTH_RADIUS_KM,
    EARTH_ROT_RATE_RAD_S,
    srp_pressure_n_m2,
)

_ATMOSPHERE_RELATIVE_VELOCITY_ECI_KM_S = None
_ATMOSPHERE_RELATIVE_VELOCITY_LOCK = RLock()


def _atmosphere_relative_velocity_eci_km_s(*args, **kwargs) -> np.ndarray:
    global _ATMOSPHERE_RELATIVE_VELOCITY_ECI_KM_S
    resolver = _ATMOSPHERE_RELATIVE_VELOCITY_ECI_KM_S
    if resolver is None:
        with _ATMOSPHERE_RELATIVE_VELOCITY_LOCK:
            resolver = _ATMOSPHERE_RELATIVE_VELOCITY_ECI_KM_S
            if resolver is None:
                from sim.aero.core import atmosphere_relative_velocity_eci_km_s

                resolver = atmosphere_relative_velocity_eci_km_s
                _ATMOSPHERE_RELATIVE_VELOCITY_ECI_KM_S = resolver
    return resolver(*args, **kwargs)


@dataclass(frozen=True)
class OrbitContext:
    mu_km3_s2: float
    mass_kg: float
    area_m2: float = 1.0
    cd: float = 2.2
    cr: float = 1.2


def accel_two_body(r_eci_km: np.ndarray, mu_km3_s2: float) -> np.ndarray:
    r2 = float(np.dot(r_eci_km, r_eci_km))
    if r2 == 0.0:
        return np.zeros(3)
    r = float(np.sqrt(r2))
    return (-mu_km3_s2 / (r * r2)) * r_eci_km


def accel_j2(
    r_eci_km: np.ndarray, mu_km3_s2: float, j2: float = EARTH_J2, re_km: float = EARTH_RADIUS_KM
) -> np.ndarray:
    x, y, z = r_eci_km
    r2 = float(np.dot(r_eci_km, r_eci_km))
    r = np.sqrt(r2)
    if r == 0.0:
        return np.zeros(3)
    z2 = z * z
    f = 1.5 * j2 * mu_km3_s2 * (re_km**2) / (r**5)
    g = 5.0 * z2 / r2
    return np.array(
        [
            f * x * (g - 1.0),
            f * y * (g - 1.0),
            f * z * (g - 3.0),
        ]
    )


def accel_j3(
    r_eci_km: np.ndarray, mu_km3_s2: float, j3: float = EARTH_J3, re_km: float = EARTH_RADIUS_KM
) -> np.ndarray:
    """
    Zonal J3 perturbation acceleration in ECI (km/s^2).

    Uses the standard spherical-harmonic zonal expansion for n=3.
    """
    x, y, z = r_eci_km
    r2 = float(np.dot(r_eci_km, r_eci_km))
    r = np.sqrt(r2)
    if r == 0.0:
        return np.zeros(3)
    s = z / r
    s2 = s * s
    s4 = s2 * s2

    # a_xy = mu*J3*Re^3 * x(y) / r^6 * [ (5/2) s (7 s^2 - 3) ]
    axy_scale = mu_km3_s2 * j3 * (re_km**3) / (r**6)
    axy_factor = 2.5 * s * (7.0 * s2 - 3.0)

    # a_z = mu*J3*Re^3 / r^5 * [ (1/2) (35 s^4 - 30 s^2 + 3) ]
    az_scale = mu_km3_s2 * j3 * (re_km**3) / (r**5)
    az_factor = 0.5 * (35.0 * s4 - 30.0 * s2 + 3.0)

    return np.array(
        [
            axy_scale * x * axy_factor,
            axy_scale * y * axy_factor,
            az_scale * az_factor,
        ]
    )


def accel_j4(
    r_eci_km: np.ndarray, mu_km3_s2: float, j4: float = EARTH_J4, re_km: float = EARTH_RADIUS_KM
) -> np.ndarray:
    """
    Zonal J4 perturbation acceleration in ECI (km/s^2).

    Uses the standard spherical-harmonic zonal expansion for n=4.
    """
    x, y, z = r_eci_km
    r2 = float(np.dot(r_eci_km, r_eci_km))
    r = np.sqrt(r2)
    if r == 0.0:
        return np.zeros(3)
    s = z / r
    s2 = s * s
    s4 = s2 * s2

    # a_xy = mu*J4*Re^4 * x(y) / r^7 * [ (5/8) (63 s^4 - 42 s^2 + 3) ]
    axy_scale = mu_km3_s2 * j4 * (re_km**4) / (r**7)
    axy_factor = 0.625 * (63.0 * s4 - 42.0 * s2 + 3.0)

    # a_z = mu*J4*Re^4 / r^6 * [ (5/8) s (63 s^4 - 70 s^2 + 15) ]
    az_scale = mu_km3_s2 * j4 * (re_km**4) / (r**6)
    az_factor = 0.625 * s * (63.0 * s4 - 70.0 * s2 + 15.0)

    return np.array(
        [
            axy_scale * x * axy_factor,
            axy_scale * y * axy_factor,
            az_scale * az_factor,
        ]
    )


def accel_drag(
    r_eci_km: np.ndarray,
    v_eci_km_s: np.ndarray,
    t_s: float,
    mass_kg: float,
    area_m2: float,
    cd: float,
    env: dict,
) -> np.ndarray:
    rho = float(env.get("density_kg_m3", 0.0))
    area_eff_m2 = float(env.get("drag_area_m2", area_m2))
    drag_frame_model = str(env.get("drag_frame_model", "simple")).strip().lower()
    jd_utc_start = env.get("jd_utc_start")
    drag_eop_path = env.get("drag_eop_path")
    omega_raw = env.get("drag_earth_rotation_rad_s", EARTH_ROT_RATE_RAD_S)
    return accel_drag_resolved(
        r_eci_km=r_eci_km,
        v_eci_km_s=v_eci_km_s,
        t_s=t_s,
        mass_kg=mass_kg,
        cd=cd,
        density_kg_m3=rho,
        area_eff_m2=area_eff_m2,
        drag_frame_model=drag_frame_model,
        jd_utc_start=None if jd_utc_start is None else float(jd_utc_start),
        drag_eop_path=None if drag_eop_path is None else str(drag_eop_path),
        omega_earth_rad_s=float(EARTH_ROT_RATE_RAD_S if omega_raw is None else omega_raw),
        dut1_s=None if env.get("dut1_s") is None else float(env["dut1_s"]),
        xp_arcsec=None if env.get("xp_arcsec") is None else float(env["xp_arcsec"]),
        yp_arcsec=None if env.get("yp_arcsec") is None else float(env["yp_arcsec"]),
        dat_s=None if env.get("dat_s") is None else float(env["dat_s"]),
        tt_minus_utc_s=None if env.get("tt_minus_utc_s") is None else float(env["tt_minus_utc_s"]),
        ddpsi_rad=float(env.get("ddpsi_rad", 0.0) or 0.0),
        ddeps_rad=float(env.get("ddeps_rad", 0.0) or 0.0),
        eop_extrapolation=str(env.get("eop_extrapolation", "error") or "error"),
    )


def accel_drag_resolved(
    *,
    r_eci_km: np.ndarray,
    v_eci_km_s: np.ndarray,
    t_s: float,
    mass_kg: float,
    cd: float,
    density_kg_m3: float,
    area_eff_m2: float,
    drag_frame_model: str,
    jd_utc_start: float | None,
    drag_eop_path: str | None,
    omega_earth_rad_s: float,
    dut1_s: float | None = None,
    xp_arcsec: float | None = None,
    yp_arcsec: float | None = None,
    dat_s: float | None = None,
    tt_minus_utc_s: float | None = None,
    ddpsi_rad: float = 0.0,
    ddeps_rad: float = 0.0,
    eop_extrapolation: str = "error",
) -> np.ndarray:
    rho = float(density_kg_m3)
    if rho <= 0.0 or mass_kg <= 0.0:
        return np.zeros(3)
    area_eff_m2 = float(area_eff_m2)
    if area_eff_m2 <= 0.0:
        return np.zeros(3)
    v_rel_eci_km_s = _atmosphere_relative_velocity_eci_km_s(
        r_eci_km,
        v_eci_km_s,
        t_s=float(t_s),
        earth_rotation_rad_s=float(omega_earth_rad_s),
        frame_model=str(drag_frame_model).strip().lower(),
        jd_utc_start=jd_utc_start,
        eop_path=drag_eop_path,
        dut1_s=dut1_s,
        xp_arcsec=xp_arcsec,
        yp_arcsec=yp_arcsec,
        dat_s=dat_s,
        tt_minus_utc_s=tt_minus_utc_s,
        ddpsi_rad=ddpsi_rad,
        ddeps_rad=ddeps_rad,
        eop_extrapolation=eop_extrapolation,
    )
    v_rel_m_s = v_rel_eci_km_s * 1e3
    v_norm2 = float(np.dot(v_rel_m_s, v_rel_m_s))
    if v_norm2 == 0.0:
        return np.zeros(3)
    v_norm = float(np.sqrt(v_norm2))
    a_m_s2 = -0.5 * rho * cd * area_eff_m2 / mass_kg * v_norm * v_rel_m_s
    return a_m_s2 / 1e3


def accel_lift(
    r_eci_km: np.ndarray,
    v_eci_km_s: np.ndarray,
    t_s: float,
    mass_kg: float,
    area_m2: float,
    cl: float,
    lift_direction_eci: np.ndarray,
    env: dict,
) -> np.ndarray:
    rho = float(env.get("density_kg_m3", 0.0))
    if rho <= 0.0 or mass_kg <= 0.0:
        return np.zeros(3)
    area_eff_m2 = float(env.get("lift_area_m2", area_m2))
    if area_eff_m2 <= 0.0 or float(cl) == 0.0:
        return np.zeros(3)
    drag_frame_model = str(env.get("drag_frame_model", "inertial_z")).strip().lower()
    jd_utc_start = env.get("jd_utc_start")
    drag_eop_path = env.get("drag_eop_path")
    omega_raw = env.get("drag_earth_rotation_rad_s", EARTH_ROT_RATE_RAD_S)
    omega_earth_rad_s = float(EARTH_ROT_RATE_RAD_S if omega_raw is None else omega_raw)
    v_rel_eci_km_s = _atmosphere_relative_velocity_eci_km_s(
        r_eci_km,
        v_eci_km_s,
        t_s=float(t_s),
        earth_rotation_rad_s=omega_earth_rad_s,
        frame_model=drag_frame_model,
        jd_utc_start=None if jd_utc_start is None else float(jd_utc_start),
        eop_path=None if drag_eop_path is None else str(drag_eop_path),
        dut1_s=None if env.get("dut1_s") is None else float(env["dut1_s"]),
        xp_arcsec=None if env.get("xp_arcsec") is None else float(env["xp_arcsec"]),
        yp_arcsec=None if env.get("yp_arcsec") is None else float(env["yp_arcsec"]),
        dat_s=None if env.get("dat_s") is None else float(env["dat_s"]),
        tt_minus_utc_s=None if env.get("tt_minus_utc_s") is None else float(env["tt_minus_utc_s"]),
        ddpsi_rad=float(env.get("ddpsi_rad", 0.0) or 0.0),
        ddeps_rad=float(env.get("ddeps_rad", 0.0) or 0.0),
        eop_extrapolation=str(env.get("eop_extrapolation", "error") or "error"),
    )
    v_rel_m_s = v_rel_eci_km_s * 1e3
    speed_m_s = float(np.linalg.norm(v_rel_m_s))
    if speed_m_s <= 0.0:
        return np.zeros(3)
    v_hat = v_rel_m_s / speed_m_s
    desired = np.array(lift_direction_eci, dtype=float).reshape(3)
    desired_norm = float(np.linalg.norm(desired))
    if desired_norm <= 0.0:
        return np.zeros(3)
    desired = desired / desired_norm
    lift_dir = desired - float(np.dot(desired, v_hat)) * v_hat
    lift_norm = float(np.linalg.norm(lift_dir))
    if lift_norm <= 1e-12:
        return np.zeros(3)
    lift_dir = lift_dir / lift_norm
    q_dyn_pa = 0.5 * rho * speed_m_s * speed_m_s
    a_m_s2 = q_dyn_pa * area_eff_m2 * float(cl) / float(mass_kg)
    return (a_m_s2 / 1e3) * lift_dir


def accel_srp(
    r_eci_km: np.ndarray,
    mass_kg: float,
    area_m2: float,
    cr: float,
    t_s: float,
    env: dict,
) -> np.ndarray:
    if mass_kg <= 0.0:
        return np.zeros(3)
    area_eff_m2 = float(env.get("srp_area_m2", area_m2))
    if area_eff_m2 <= 0.0:
        return np.zeros(3)
    srp_geometry = env.get("srp_geometry")
    if not isinstance(srp_geometry, dict):
        srp_geometry = resolve_srp_geometry(r_eci_km, t_s, env)

    sun_dir_eci = env.get("srp_sun_dir_eci")
    if sun_dir_eci is None:
        sun_dir_eci = srp_geometry["sun_dir_sc_eci"]
    sun_dir_eci = np.asarray(sun_dir_eci, dtype=float).reshape(3)

    shadow = env.get("srp_shadow_factor")
    if shadow is None:
        shadow = srp_shadow_factor(r_sc_eci_km=r_eci_km, t_s=t_s, env=env, srp_geometry=srp_geometry)

    distance_scale = float(env.get("srp_distance_scale", srp_geometry.get("distance_scale", 1.0)))
    return accel_srp_resolved(
        sun_dir_eci=sun_dir_eci,
        mass_kg=mass_kg,
        area_eff_m2=area_eff_m2,
        cr=cr,
        distance_scale=distance_scale,
        shadow_factor=float(shadow),
        pressure_n_m2=srp_pressure_n_m2(env),
    )


def accel_srp_resolved(
    *,
    sun_dir_eci: np.ndarray,
    mass_kg: float,
    area_eff_m2: float,
    cr: float,
    distance_scale: float,
    shadow_factor: float,
    pressure_n_m2: float,
) -> np.ndarray:
    if mass_kg <= 0.0:
        return np.zeros(3)
    area_eff_m2 = float(area_eff_m2)
    if area_eff_m2 <= 0.0:
        return np.zeros(3)
    shadow = float(shadow_factor)
    if shadow <= 0.0:
        return np.zeros(3)

    sun_dir_eci = np.asarray(sun_dir_eci, dtype=float).reshape(3)
    n2 = float(np.dot(sun_dir_eci, sun_dir_eci))
    if n2 <= 0.0:
        return np.zeros(3)
    if abs(n2 - 1.0) > 1e-12:
        sun_dir_eci = sun_dir_eci / float(np.sqrt(n2))

    force_n = float(pressure_n_m2) * float(distance_scale) * cr * area_eff_m2
    a_m_s2 = force_n / mass_kg
    return -(a_m_s2 / 1e3) * shadow * sun_dir_eci


def accel_third_body(r_eci_km: np.ndarray, body_pos_eci_km: np.ndarray, body_mu_km3_s2: float) -> np.ndarray:
    rb = body_pos_eci_km - r_eci_km
    rb_norm2 = float(np.dot(rb, rb))
    b_norm2 = float(np.dot(body_pos_eci_km, body_pos_eci_km))
    rb_norm = float(np.sqrt(rb_norm2)) if rb_norm2 > 0.0 else 0.0
    b_norm = float(np.sqrt(b_norm2)) if b_norm2 > 0.0 else 0.0
    if rb_norm == 0.0 or b_norm == 0.0:
        return np.zeros(3)
    return body_mu_km3_s2 * (rb / (rb_norm**3) - body_pos_eci_km / (b_norm**3))


def default_density_model(r_eci_km: np.ndarray, t_s: float) -> float:
    return density_exponential(r_eci_km, t_s)
