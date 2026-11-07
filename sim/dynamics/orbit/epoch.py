from __future__ import annotations

from datetime import datetime, timezone
from functools import lru_cache

import numpy as np

AU_KM = 149597870.7
TIME_DEPENDENT_ENV_CACHE_KEY = "_time_dependent_env_cache"


def _validated_position_vector(value: object, label: str) -> np.ndarray:
    """Return one finite ECI position vector without silently reshaping it."""

    try:
        vector = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must contain exactly three finite values.") from exc
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{label} must contain exactly three finite values.")
    return np.array(vector, dtype=float, copy=True)


def datetime_to_julian_date(dt: datetime) -> float:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    dt_utc = dt.astimezone(timezone.utc)
    y = dt_utc.year
    m = dt_utc.month
    d = dt_utc.day
    frac_day = dt_utc.hour / 24.0 + dt_utc.minute / 1440.0 + (dt_utc.second + dt_utc.microsecond * 1e-6) / 86400.0
    if m <= 2:
        y -= 1
        m += 12
    a = y // 100
    b = 2 - a + (a // 4)
    jd = int(365.25 * (y + 4716)) + int(30.6001 * (m + 1)) + d + frac_day + b - 1524.5
    return float(jd)


def julian_date_to_datetime(jd_utc: float) -> datetime:
    jd = float(jd_utc) + 0.5
    z = int(np.floor(jd))
    f = jd - z
    if z < 2299161:
        a = z
    else:
        alpha = int((z - 1867216.25) / 36524.25)
        a = z + 1 + alpha - alpha // 4
    b = a + 1524
    c = int((b - 122.1) / 365.25)
    d = int(365.25 * c)
    e = int((b - d) / 30.6001)
    day = b - d - int(30.6001 * e) + f
    month = e - 1 if e < 14 else e - 13
    year = c - 4716 if month > 2 else c - 4715

    day_int = int(np.floor(day))
    frac = day - day_int
    sec_total = frac * 86400.0
    hour = int(sec_total // 3600.0)
    sec_total -= hour * 3600.0
    minute = int(sec_total // 60.0)
    sec_total -= minute * 60.0
    second = int(np.floor(sec_total))
    usec = int(np.round((sec_total - second) * 1e6))
    if usec >= 1_000_000:
        usec -= 1_000_000
        second += 1
    if second >= 60:
        second -= 60
        minute += 1
    if minute >= 60:
        minute -= 60
        hour += 1
    if hour >= 24:
        hour -= 24
        day_int += 1
    return datetime(year, month, day_int, hour, minute, second, usec, tzinfo=timezone.utc)


def gmst_angle_rad_from_jd(jd_utc: float) -> float:
    jd = float(jd_utc)
    t = (jd - 2451545.0) / 36525.0
    theta_deg = 280.46061837 + 360.98564736629 * (jd - 2451545.0) + 0.000387933 * (t**2) - (t**3) / 38710000.0
    return float(np.deg2rad(np.mod(theta_deg, 360.0)))


def sun_position_eci_km_simple(jd_utc: float) -> np.ndarray:
    n = float(jd_utc) - 2451545.0
    l_deg = np.mod(280.460 + 0.9856474 * n, 360.0)
    g_rad = np.deg2rad(np.mod(357.528 + 0.9856003 * n, 360.0))
    lam_deg = l_deg + 1.915 * np.sin(g_rad) + 0.020 * np.sin(2.0 * g_rad)
    lam_rad = np.deg2rad(lam_deg)
    eps_rad = np.deg2rad(23.439 - 0.0000004 * n)
    r_au = 1.00014 - 0.01671 * np.cos(g_rad) - 0.00014 * np.cos(2.0 * g_rad)
    r_km = r_au * AU_KM
    x = r_km * np.cos(lam_rad)
    y = r_km * np.cos(eps_rad) * np.sin(lam_rad)
    z = r_km * np.sin(eps_rad) * np.sin(lam_rad)
    return _mean_equator_of_date_to_j2000(np.array([x, y, z], dtype=float), jd_utc)


def moon_position_eci_km_simple(jd_utc: float) -> np.ndarray:
    n = float(jd_utc) - 2451545.0
    l0_rad = np.deg2rad(np.mod(218.316 + 13.176396 * n, 360.0))
    m_moon_rad = np.deg2rad(np.mod(134.963 + 13.064993 * n, 360.0))
    f_rad = np.deg2rad(np.mod(93.272 + 13.229350 * n, 360.0))
    lon_rad = l0_rad + np.deg2rad(6.289) * np.sin(m_moon_rad)
    lat_rad = np.deg2rad(5.128) * np.sin(f_rad)
    r_km = 385001.0 - 20905.0 * np.cos(m_moon_rad)
    x_ecl = r_km * np.cos(lat_rad) * np.cos(lon_rad)
    y_ecl = r_km * np.cos(lat_rad) * np.sin(lon_rad)
    z_ecl = r_km * np.sin(lat_rad)
    eps_rad = np.deg2rad(23.439 - 0.0000004 * n)
    x = x_ecl
    y = y_ecl * np.cos(eps_rad) - z_ecl * np.sin(eps_rad)
    z = y_ecl * np.sin(eps_rad) + z_ecl * np.cos(eps_rad)
    return _mean_equator_of_date_to_j2000(np.array([x, y, z], dtype=float), jd_utc)


def _wrap_deg(x: float) -> float:
    return float(np.mod(x, 360.0))


def _mean_equator_of_date_to_j2000(vector: np.ndarray, jd_utc: float) -> np.ndarray:
    """Rotate a date-equator vector into OEL's mean-equator J2000 frame."""

    t = (float(jd_utc) - 2451545.0) / 36525.0
    arcsec_to_rad = np.deg2rad(1.0 / 3600.0)
    zeta = (2306.2181 * t + 0.30188 * t**2 + 0.017998 * t**3) * arcsec_to_rad
    theta = (2004.3109 * t - 0.42665 * t**2 - 0.041833 * t**3) * arcsec_to_rad
    z = (2306.2181 * t + 1.09468 * t**2 + 0.018203 * t**3) * arcsec_to_rad
    rz_zeta = np.array(
        [[np.cos(-zeta), np.sin(-zeta), 0.0], [-np.sin(-zeta), np.cos(-zeta), 0.0], [0.0, 0.0, 1.0]],
        dtype=float,
    )
    ry_theta = np.array(
        [[np.cos(theta), 0.0, -np.sin(theta)], [0.0, 1.0, 0.0], [np.sin(theta), 0.0, np.cos(theta)]],
        dtype=float,
    )
    rz_z = np.array(
        [[np.cos(-z), np.sin(-z), 0.0], [-np.sin(-z), np.cos(-z), 0.0], [0.0, 0.0, 1.0]],
        dtype=float,
    )
    date_from_j2000 = rz_z @ ry_theta @ rz_zeta
    return date_from_j2000.T @ np.asarray(vector, dtype=float).reshape(3)


def _sun_position_true_equator_of_date_km(jd_utc: float) -> np.ndarray:
    t = (float(jd_utc) - 2451545.0) / 36525.0
    l0 = _wrap_deg(280.46646 + 36000.76983 * t + 0.0003032 * (t**2))
    m = _wrap_deg(357.52911 + 35999.05029 * t - 0.0001537 * (t**2))
    m_rad = np.deg2rad(m)
    e = 0.016708634 - 0.000042037 * t - 0.0000001267 * (t**2)
    c = (
        (1.914602 - 0.004817 * t - 0.000014 * (t**2)) * np.sin(m_rad)
        + (0.019993 - 0.000101 * t) * np.sin(2.0 * m_rad)
        + 0.000289 * np.sin(3.0 * m_rad)
    )
    true_long = _wrap_deg(l0 + c)
    true_anom_deg = _wrap_deg(m + c)
    v_rad = np.deg2rad(true_anom_deg)
    r_au = (1.000001018 * (1.0 - e * e)) / (1.0 + e * np.cos(v_rad))

    omega = _wrap_deg(125.04 - 1934.136 * t)
    lam_app_deg = true_long - 0.00569 - 0.00478 * np.sin(np.deg2rad(omega))
    lam = np.deg2rad(lam_app_deg)

    eps0 = 23.0 + 26.0 / 60.0 + 21.448 / 3600.0 - (46.8150 * t + 0.00059 * (t**2) - 0.001813 * (t**3)) / 3600.0
    eps = np.deg2rad(eps0 + 0.00256 * np.cos(np.deg2rad(omega)))

    r_km = r_au * AU_KM
    x = r_km * np.cos(lam)
    y = r_km * np.cos(eps) * np.sin(lam)
    z = r_km * np.sin(eps) * np.sin(lam)
    return np.array([x, y, z], dtype=float)


def _true_equator_of_date_to_j2000(vector: np.ndarray, jd_utc: float) -> np.ndarray:
    """Rotate a true-equator/equinox-of-date vector into OEL/ECI/J2000."""

    # Keep epoch and frame modules acyclic while sharing their authoritative
    # IAU-76/80 matrices at call time.
    from sim.dynamics.orbit.frames import (
        _DEFAULT_TT_MINUS_UTC_S,
        _nutation_iau1980_vallado_matrix,
        _precession_iau1976_matrix,
    )

    jd_tt = float(jd_utc) + float(_DEFAULT_TT_MINUS_UTC_S) / 86400.0
    nutation = _nutation_iau1980_vallado_matrix(jd_tt)[4]
    precession = _precession_iau1976_matrix(jd_tt)
    return precession.T @ nutation @ np.asarray(vector, dtype=float).reshape(3)


def sun_position_eci_km_enhanced(jd_utc: float) -> np.ndarray:
    """
    Enhanced low-cost Sun ephemeris (Meeus-style) in OEL/ECI/J2000.
    """
    return _true_equator_of_date_to_j2000(_sun_position_true_equator_of_date_km(jd_utc), jd_utc)


def moon_position_eci_km_enhanced(jd_utc: float) -> np.ndarray:
    """
    Enhanced low-cost Moon ephemeris with dominant periodic terms in
    OEL/ECI/J2000.
    """
    t = (float(jd_utc) - 2451545.0) / 36525.0
    l_prime = _wrap_deg(
        218.3164477 + 481267.88123421 * t - 0.0015786 * (t**2) + (t**3) / 538841.0 - (t**4) / 65194000.0
    )
    d = _wrap_deg(297.8501921 + 445267.1114034 * t - 0.0018819 * (t**2) + (t**3) / 545868.0 - (t**4) / 113065000.0)
    m = _wrap_deg(357.5291092 + 35999.0502909 * t - 0.0001536 * (t**2) + (t**3) / 24490000.0)
    m_prime = _wrap_deg(134.9633964 + 477198.8675055 * t + 0.0087414 * (t**2) + (t**3) / 69699.0 - (t**4) / 14712000.0)
    f = _wrap_deg(93.2720950 + 483202.0175233 * t - 0.0036539 * (t**2) - (t**3) / 3526000.0 + (t**4) / 863310000.0)

    d_rad = np.deg2rad(d)
    m_rad = np.deg2rad(m)
    mp_rad = np.deg2rad(m_prime)
    f_rad = np.deg2rad(f)

    lon_deg = (
        l_prime
        + 6.289 * np.sin(mp_rad)
        + 1.274 * np.sin(2.0 * d_rad - mp_rad)
        + 0.658 * np.sin(2.0 * d_rad)
        + 0.214 * np.sin(2.0 * mp_rad)
        + 0.11 * np.sin(d_rad)
    )
    lat_deg = (
        5.128 * np.sin(f_rad)
        + 0.280 * np.sin(mp_rad + f_rad)
        + 0.277 * np.sin(mp_rad - f_rad)
        + 0.173 * np.sin(2.0 * d_rad - f_rad)
        + 0.055 * np.sin(2.0 * d_rad + f_rad - mp_rad)
        + 0.046 * np.sin(2.0 * d_rad - f_rad - mp_rad)
        + 0.033 * np.sin(2.0 * d_rad + f_rad)
        + 0.017 * np.sin(2.0 * mp_rad + f_rad)
    )
    r_km = (
        385000.56
        - 20905.0 * np.cos(mp_rad)
        - 3699.0 * np.cos(2.0 * d_rad - mp_rad)
        - 2956.0 * np.cos(2.0 * d_rad)
        - 570.0 * np.cos(2.0 * mp_rad)
        + 246.0 * np.cos(2.0 * mp_rad - 2.0 * d_rad)
        - 205.0 * np.cos(m_rad - 2.0 * d_rad)
        - 171.0 * np.cos(mp_rad + 2.0 * d_rad)
    )

    lon = np.deg2rad(_wrap_deg(lon_deg))
    lat = np.deg2rad(lat_deg)
    eps0 = 23.439291 - 0.0130042 * t
    eps = np.deg2rad(eps0)

    x_ecl = r_km * np.cos(lat) * np.cos(lon)
    y_ecl = r_km * np.cos(lat) * np.sin(lon)
    z_ecl = r_km * np.sin(lat)
    x = x_ecl
    y = y_ecl * np.cos(eps) - z_ecl * np.sin(eps)
    z = y_ecl * np.sin(eps) + z_ecl * np.cos(eps)
    return _mean_equator_of_date_to_j2000(np.array([x, y, z], dtype=float), jd_utc)



@lru_cache(maxsize=128)
def _analytic_sun_moon_positions(jd_utc: float, simple: bool) -> tuple[np.ndarray, np.ndarray]:
    """Reuse pure analytic ephemerides at exact epochs, without time rounding.

    A short bounded cache shares repeated RK stages and satellite epochs.
    Immutable byte-backed storage prevents accidental cache poisoning; the
    resolver returns fresh writable copies to preserve its ownership contract.
    """
    if simple:
        sun, moon = sun_position_eci_km_simple(jd_utc), moon_position_eci_km_simple(jd_utc)
    else:
        sun, moon = sun_position_eci_km_enhanced(jd_utc), moon_position_eci_km_enhanced(jd_utc)
    return (
        np.frombuffer(sun.tobytes(), dtype=np.float64),
        np.frombuffer(moon.tobytes(), dtype=np.float64),
    )


def _sampled_ephemeris_position(body: str, env: dict, t_s: float) -> np.ndarray | None:
    """Interpolate one sampled body history with strict shape and coverage checks."""
    time_key = f"{body}_ephemeris_time_s"
    state_key = f"{body}_ephemeris_eci_km"
    if time_key not in env or state_key not in env:
        return None
    tt = np.asarray(env[time_key], dtype=float).reshape(-1)
    rr = np.asarray(env[state_key], dtype=float)
    if tt.size < 1 or rr.ndim != 2 or rr.shape != (tt.size, 3):
        raise ValueError(
            f"{body} ephemeris history must have matching (N,) times and (N, 3) states."
        )
    if not np.all(np.isfinite(tt)) or np.any(np.diff(tt) <= 0.0):
        raise ValueError(f"{body} ephemeris times must be finite and strictly increasing.")
    if not np.all(np.isfinite(rr)):
        raise ValueError(f"{body} ephemeris states must contain only finite values.")
    if float(t_s) < float(tt[0]) or float(t_s) > float(tt[-1]):
        raise ValueError(
            f"{body} ephemeris time {float(t_s):.9g} s is outside supplied coverage "
            f"[{float(tt[0]):.9g}, {float(tt[-1]):.9g}] s."
        )
    return np.array([np.interp(float(t_s), tt, rr[:, j]) for j in range(3)], dtype=float)


def resolve_sun_position_eci_km(env: dict, t_s: float) -> np.ndarray:
    """Resolve only the Sun vector, honoring sampled, explicit, then provider inputs.

    Unlike the paired resolver, this path does not inspect Moon histories or
    require a Moon result. It uses the same deterministic ephemeris providers
    for any missing Sun value.
    """
    sun = _sampled_ephemeris_position("sun", env, t_s)
    if sun is None and "sun_pos_eci_km" in env:
        sun = _validated_position_vector(env["sun_pos_eci_km"], "sun_pos_eci_km")
    if sun is not None:
        if sun.shape != (3,) or not np.all(np.isfinite(sun)):
            raise ValueError("sun_pos_eci_km must contain exactly three finite values.")
        return sun

    jd = resolved_jd_utc(env=env, t_s=t_s)
    if jd is None:
        return np.array([AU_KM, 0.0, 0.0], dtype=float)

    eph_callable = env.get("ephemeris_callable", None)
    if callable(eph_callable):
        out = eph_callable(float(jd), env)
        if isinstance(out, dict) and "sun_pos_eci_km" in out:
            return _validated_position_vector(out["sun_pos_eci_km"], "sun_pos_eci_km")

    mode = str(env.get("ephemeris_mode", "analytic_enhanced")).lower()
    if mode in ("de440_hpop", "hpop_de440", "de440"):
        from sim.acceleration.settings import acceleration_enabled_from_mode
        from sim.dynamics.orbit.de440_hpop import (
            hpop_de440_positions_km,
            hpop_de440_sun_moon_positions_km,
        )

        if acceleration_enabled_from_mode():
            sun, _moon = hpop_de440_sun_moon_positions_km(jd, env)
        else:
            sun = hpop_de440_positions_km(jd, env)["sun"]
        sun = np.array(sun, dtype=float)
    elif mode in ("spice", "spiceypy"):
        spice_pair_callable = env.get("spice_ephemeris_callable", None)
        spice_body_callable = env.get("spice_body_ephemeris_callable", None)
        if callable(spice_body_callable) or not callable(spice_pair_callable):
            from sim.dynamics.orbit.spice import spice_body_position_eci_km

            sun = spice_body_position_eci_km("sun", jd, env)
        else:
            out = spice_pair_callable(float(jd), env)
            if not isinstance(out, dict) or "sun_pos_eci_km" not in out:
                raise RuntimeError(
                    "spice_ephemeris_callable must return a dict containing 'sun_pos_eci_km'."
                )
            sun = np.array(out["sun_pos_eci_km"], dtype=float)
    elif mode in ("analytic_simple", "simple"):
        sun = sun_position_eci_km_simple(float(jd))
    elif mode in ("analytic_enhanced", "enhanced", ""):
        sun = sun_position_eci_km_enhanced(float(jd))
    else:
        raise ValueError(
            "ephemeris_mode must be one of: analytic_enhanced, analytic_simple, de440, hpop_de440, de440_hpop, spice, spiceypy."
        )

    return _validated_position_vector(sun, "sun_pos_eci_km")


def resolve_sun_moon_positions(env: dict, t_s: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Resolve Sun and Moon inertial position vectors (km) using explicit env values,
    optional callable hook, then configured analytic mode.
    """
    sun_explicit = _sampled_ephemeris_position("sun", env, t_s)
    moon_explicit = _sampled_ephemeris_position("moon", env, t_s)
    if sun_explicit is None and "sun_pos_eci_km" in env:
        sun_explicit = _validated_position_vector(env["sun_pos_eci_km"], "sun_pos_eci_km")
    if moon_explicit is None and "moon_pos_eci_km" in env:
        moon_explicit = _validated_position_vector(env["moon_pos_eci_km"], "moon_pos_eci_km")
    if sun_explicit is not None and moon_explicit is not None:
        return sun_explicit, moon_explicit

    jd = resolved_jd_utc(env=env, t_s=t_s)
    if jd is None:
        sun = sun_explicit if sun_explicit is not None else np.array([AU_KM, 0.0, 0.0], dtype=float)
        moon = moon_explicit if moon_explicit is not None else np.array([384400.0, 0.0, 0.0], dtype=float)
        return sun, moon

    eph_callable = env.get("ephemeris_callable", None)
    if callable(eph_callable):
        out = eph_callable(float(jd), env)
        if isinstance(out, dict):
            if "sun_pos_eci_km" in out and "moon_pos_eci_km" in out:
                sun = _validated_position_vector(out["sun_pos_eci_km"], "sun_pos_eci_km")
                moon = _validated_position_vector(out["moon_pos_eci_km"], "moon_pos_eci_km")
                return (
                    sun_explicit if sun_explicit is not None else sun,
                    moon_explicit if moon_explicit is not None else moon,
                )

    mode = str(env.get("ephemeris_mode", "analytic_enhanced")).lower()
    if mode in ("de440_hpop", "hpop_de440", "de440"):
        from sim.acceleration.settings import acceleration_enabled_from_mode
        from sim.dynamics.orbit.de440_hpop import (
            hpop_de440_positions_km,
            hpop_de440_sun_moon_positions_km,
        )

        if acceleration_enabled_from_mode():
            sun, moon = hpop_de440_sun_moon_positions_km(jd, env)
            sun = _validated_position_vector(sun, "sun_pos_eci_km")
            moon = _validated_position_vector(moon, "moon_pos_eci_km")
            return sun_explicit if sun_explicit is not None else sun, moon_explicit if moon_explicit is not None else moon
        pos = hpop_de440_positions_km(jd, env)
        sun = _validated_position_vector(pos["sun"], "sun_pos_eci_km")
        moon = _validated_position_vector(pos["moon"], "moon_pos_eci_km")
        return sun_explicit if sun_explicit is not None else sun, moon_explicit if moon_explicit is not None else moon
    if mode in ("spice", "spiceypy"):
        from sim.dynamics.orbit.spice import spice_sun_moon_positions_eci_km

        sun, moon = spice_sun_moon_positions_eci_km(jd, env)
        sun = _validated_position_vector(sun, "sun_pos_eci_km")
        moon = _validated_position_vector(moon, "moon_pos_eci_km")
        return sun_explicit if sun_explicit is not None else sun, moon_explicit if moon_explicit is not None else moon
    if mode in ("analytic_simple", "simple"):
        sun, moon = _analytic_sun_moon_positions(float(jd), True)
        return (
            sun_explicit if sun_explicit is not None else sun.copy(),
            moon_explicit if moon_explicit is not None else moon.copy(),
        )
    if mode in ("analytic_enhanced", "enhanced", ""):
        sun, moon = _analytic_sun_moon_positions(float(jd), False)
        return (
            sun_explicit if sun_explicit is not None else sun.copy(),
            moon_explicit if moon_explicit is not None else moon.copy(),
        )
    raise ValueError(
        "ephemeris_mode must be one of: analytic_enhanced, analytic_simple, de440, hpop_de440, de440_hpop, spice, spiceypy."
    )


def resolve_body_position_eci_km(body_name: str, env: dict, t_s: float) -> np.ndarray:
    name = str(body_name).strip().lower()
    key = f"{name}_pos_eci_km"
    if key in env:
        return _validated_position_vector(env[key], key)

    if name in ("sun", "moon"):
        sun, moon = resolve_sun_moon_positions(env, t_s)
        return sun if name == "sun" else moon

    jd = resolved_jd_utc(env=env, t_s=t_s)
    if jd is None:
        raise RuntimeError(
            f"Body '{body_name}' requested but no position provided in env['{key}'] and no epoch is available."
        )

    mode = str(env.get("ephemeris_mode", "analytic_enhanced")).lower()
    if mode in ("de440_hpop", "hpop_de440", "de440"):
        from sim.dynamics.orbit.de440_hpop import hpop_de440_positions_km

        pos = hpop_de440_positions_km(jd, env)
        if name not in pos:
            raise RuntimeError(f"Body '{body_name}' is not supported by de440_hpop ephemeris mode.")
        return _validated_position_vector(pos[name], f"{name}_pos_eci_km")
    if mode in ("spice", "spiceypy"):
        from sim.dynamics.orbit.spice import spice_body_position_eci_km

        return _validated_position_vector(spice_body_position_eci_km(name, jd, env), f"{name}_pos_eci_km")

    cb = env.get("ephemeris_body_callable", None)
    if callable(cb):
        out = cb(name, float(jd), env)
        return _validated_position_vector(out, f"{name}_pos_eci_km")

    raise RuntimeError(
        f"Body '{body_name}' requested but ephemeris_mode='{mode}' cannot resolve it. "
        "Use SPICE mode or provide env position / callable."
    )


def resolved_jd_utc(env: dict, t_s: float) -> float | None:
    if "jd_utc" in env:
        return float(env["jd_utc"])
    if "jd_utc_start" in env:
        from sim.dynamics.orbit.frames import (
            FRAME_MODEL_IAU76_80_EOP,
            _validate_eop_elapsed_interval,
            normalize_frame_model,
        )

        active_paths = []
        for prefix in ("", "spherical_harmonics_", "drag_", "density_"):
            model = env.get(prefix + "frame_model", env.get("drag_frame_model", "simple"))
            if normalize_frame_model(model) == FRAME_MODEL_IAU76_80_EOP:
                path = env.get(prefix + "eop_path", env.get("drag_eop_path") if prefix == "density_" else env.get("eop_path"))
                if path not in (None, ""):
                    active_paths.append(str(path))
        if str(env.get("ephemeris_mode", "")).lower() in {"de440_hpop", "hpop_de440", "de440"}:
            path = env.get("de440_eop_path") or env.get("spherical_harmonics_eop_path") or env.get("drag_eop_path")
            if path:
                active_paths.append(str(path))
        for path in set(active_paths):
            _validate_eop_elapsed_interval(float(env["jd_utc_start"]), t_s, path)
        return float(env["jd_utc_start"]) + float(t_s) / 86400.0
    return None


def resolve_time_dependent_env(
    env: dict,
    t_s: float,
    *,
    cache_override: dict[tuple, dict[str, np.ndarray]] | None = None,
) -> dict:
    out = dict(env)
    out["sim_t_s"] = float(t_s)
    jd = resolved_jd_utc(env=out, t_s=t_s)
    if jd is None:
        return out
    out["jd_utc"] = float(jd)

    mode = str(out.get("ephemeris_mode", "analytic_enhanced")).lower()
    cache = cache_override if cache_override is not None else out.get(TIME_DEPENDENT_ENV_CACHE_KEY)
    cacheable_ephemeris = (
        isinstance(cache, dict)
        and mode
        in (
            "analytic",
            "analytic_simple",
            "analytic_enhanced",
            "enhanced",
            "simple",
            "",
            "de440_hpop",
            "hpop_de440",
            "de440",
        )
        and "ephemeris_callable" not in out
        and "sun_ephemeris_time_s" not in out
        and "moon_ephemeris_time_s" not in out
        and "sun_pos_eci_km" not in out
        and "moon_pos_eci_km" not in out
        and "sun_dir_eci" not in out
    )
    cache_key = None
    additions = None
    if cacheable_ephemeris:
        if mode in ("de440_hpop", "hpop_de440", "de440"):
            from sim.dynamics.orbit.de440_hpop import _resource_signature, default_de440_coeff_path

            coeff_path = out.get("de440_coeff_path")
            coeff_path = default_de440_coeff_path() if coeff_path in (None, "") else coeff_path
            eop_resource = out.get("de440_eop_path") or out.get("spherical_harmonics_eop_path") or out.get("drag_eop_path")
            cache_key = (
                mode,
                float(jd),
                out.get("de440_coeff_path"),
                _resource_signature(coeff_path),
                eop_resource,
                None if eop_resource in (None, "") else _resource_signature(eop_resource),
                out.get("de440_tai_utc_s"),
            )
        else:
            cache_key = (mode, float(jd))
        additions = cache.get(cache_key)
        if isinstance(additions, dict):
            if "sun_pos_eci_km" not in out and "sun_pos_eci_km" in additions:
                out["sun_pos_eci_km"] = additions["sun_pos_eci_km"]
            if "moon_pos_eci_km" not in out and "moon_pos_eci_km" in additions:
                out["moon_pos_eci_km"] = additions["moon_pos_eci_km"]
            if "sun_dir_eci" not in out and "sun_dir_eci" in additions:
                out["sun_dir_eci"] = additions["sun_dir_eci"]
            return out

    if mode in (
        "analytic",
        "analytic_simple",
        "analytic_enhanced",
        "enhanced",
        "simple",
        "spice",
        "spiceypy",
        "de440_hpop",
        "hpop_de440",
        "de440",
    ):
        sun, moon = resolve_sun_moon_positions(out, t_s)
        computed_additions: dict[str, np.ndarray] = {}
        if "sun_pos_eci_km" not in out:
            out["sun_pos_eci_km"] = sun
            computed_additions["sun_pos_eci_km"] = sun
        if "moon_pos_eci_km" not in out:
            out["moon_pos_eci_km"] = moon
            computed_additions["moon_pos_eci_km"] = moon
        s_norm = float(np.linalg.norm(sun))
        if s_norm > 0.0 and "sun_dir_eci" not in out:
            out["sun_dir_eci"] = sun / s_norm
            computed_additions["sun_dir_eci"] = out["sun_dir_eci"]
        if cacheable_ephemeris and cache_key is not None:
            cache[cache_key] = computed_additions
    elif mode in ("external", "callable"):
        sun, moon = resolve_sun_moon_positions(out, t_s)
        if "sun_pos_eci_km" not in out:
            out["sun_pos_eci_km"] = sun
        if "moon_pos_eci_km" not in out:
            out["moon_pos_eci_km"] = moon
        s_norm = float(np.linalg.norm(sun))
        if s_norm > 0.0 and "sun_dir_eci" not in out:
            out["sun_dir_eci"] = sun / s_norm
    return out
