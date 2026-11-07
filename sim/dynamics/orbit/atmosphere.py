from __future__ import annotations

import math
from bisect import bisect_right
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from typing import Literal

import numpy as np

from sim.acceleration.settings import acceleration_enabled_from_mode
from sim.dynamics.orbit.environment import EARTH_RADIUS_KM
from sim.dynamics.orbit.epoch import (
    datetime_to_julian_date,
    julian_date_to_datetime,
    resolve_sun_position_eci_km,
    sun_position_eci_km_enhanced,
)
from sim.dynamics.orbit.frames import (
    FRAME_MODEL_IAU76_80_EOP,
    apparent_sidereal_time_hpop_like,
    eci_to_ecef_harmonic,
    normalize_frame_model,
)
from sim.utils.geodesy import ecef_to_geodetic_altitude_km, ecef_to_geodetic_deg_km
from sim.utils.vector_math import dot3

AtmosphereModelName = Literal[
    "exponential", "ussa1976", "msis86", "nrlmsise00", "jacchia70", "jb2006", "jb2008", "harris_priester"
]

_USSA1976_G0_M_S2 = 9.80665
_USSA1976_R_AIR_J_KG_K = 287.05287
_USSA1976_HB_M = np.array([0.0, 11.0, 20.0, 32.0, 47.0, 51.0, 71.0, 86.0], dtype=float) * 1e3
_USSA1976_LB_K_M = np.array([-0.0065, 0.0, 0.0010, 0.0028, 0.0, -0.0028, -0.0020], dtype=float)
_USSA1976_TB_K = np.array([288.15, 216.65, 216.65, 228.65, 270.65, 270.65, 214.65], dtype=float)
_USSA1976_PB_PA = np.empty(_USSA1976_LB_K_M.size + 1, dtype=float)
_USSA1976_PB_PA[0] = 101325.0
for _USSA1976_LAYER_INDEX in range(_USSA1976_LB_K_M.size):
    _h0 = float(_USSA1976_HB_M[_USSA1976_LAYER_INDEX])
    _h1 = float(_USSA1976_HB_M[_USSA1976_LAYER_INDEX + 1])
    _lapse = float(_USSA1976_LB_K_M[_USSA1976_LAYER_INDEX])
    _t0 = float(_USSA1976_TB_K[_USSA1976_LAYER_INDEX])
    _p0 = float(_USSA1976_PB_PA[_USSA1976_LAYER_INDEX])
    if abs(_lapse) < 1e-12:
        _p1 = _p0 * np.exp(-_USSA1976_G0_M_S2 * (_h1 - _h0) / (_USSA1976_R_AIR_J_KG_K * _t0))
    else:
        _t1 = _t0 + _lapse * (_h1 - _h0)
        _p1 = _p0 * (_t1 / _t0) ** (-_USSA1976_G0_M_S2 / (_USSA1976_R_AIR_J_KG_K * _lapse))
    _USSA1976_PB_PA[_USSA1976_LAYER_INDEX + 1] = _p1
del _USSA1976_LAYER_INDEX, _h0, _h1, _lapse, _t0, _p0, _p1

_USSA1976_HIGH_ALT_KM = np.array(
    [
        86.0,
        100.0,
        110.0,
        120.0,
        130.0,
        140.0,
        150.0,
        180.0,
        200.0,
        250.0,
        300.0,
        350.0,
        400.0,
        450.0,
        500.0,
        600.0,
        700.0,
        800.0,
        900.0,
        1000.0,
    ],
    dtype=float,
)
_USSA1976_HIGH_LOG_RHO = np.log(
    np.array(
        [
            6.958e-6,
            5.604e-7,
            9.708e-8,
            2.222e-8,
            8.152e-9,
            3.831e-9,
            2.076e-9,
            5.464e-10,
            2.789e-10,
            7.248e-11,
            2.418e-11,
            9.518e-12,
            3.725e-12,
            1.585e-12,
            6.967e-13,
            1.454e-13,
            3.614e-14,
            1.170e-14,
            5.245e-15,
            3.019e-15,
        ],
        dtype=float,
    )
)
_USSA1976_HIGH_ALT_KM_SCALAR = tuple(float(value) for value in _USSA1976_HIGH_ALT_KM)
_USSA1976_HIGH_LOG_RHO_SCALAR = tuple(float(value) for value in _USSA1976_HIGH_LOG_RHO)
_USSA1976_HIGH_LOG_RHO_SLOPE = tuple(
    (_USSA1976_HIGH_LOG_RHO_SCALAR[index + 1] - _USSA1976_HIGH_LOG_RHO_SCALAR[index])
    / (_USSA1976_HIGH_ALT_KM_SCALAR[index + 1] - _USSA1976_HIGH_ALT_KM_SCALAR[index])
    for index in range(len(_USSA1976_HIGH_ALT_KM_SCALAR) - 1)
)

_EXPONENTIAL_ENV_KEYS = frozenset(
    {
        "exponential_ceiling_altitude_km",
        "exponential_reference_density_kg_m3",
        "exponential_reference_altitude_km",
        "exponential_scale_height_km",
    }
)


@lru_cache(maxsize=1)
def _nrlmsise00_backend():
    from sim.dynamics.orbit.nrlmsise00_backend import nrlmsise00_density

    return nrlmsise00_density


@lru_cache(maxsize=1)
def _msis86_backend():
    from sim.dynamics.orbit.msis86_backend import msis86_density

    return msis86_density


@lru_cache(maxsize=1)
def _jb_backends():
    from sim.dynamics.orbit.jb2008_backend import jb2006_density, jb2008_density

    return jb2006_density, jb2008_density


@lru_cache(maxsize=1)
def _jacchia70_backend():
    from sim.dynamics.orbit.jacchia70_backend import jacchia70_density

    return jacchia70_density


@lru_cache(maxsize=1)
def _harris_priester_backend():
    from sim.dynamics.orbit.harris_priester_backend import harris_priester_density

    return harris_priester_density


@lru_cache(maxsize=1)
def _compiled_ecef_to_geodetic_deg_km():
    from sim.acceleration.kernels.geodesy import ecef_to_geodetic_deg_km_kernel

    return ecef_to_geodetic_deg_km_kernel


def _validated_eci_position(r_eci_km: np.ndarray) -> np.ndarray:
    """Normalize one ECI position and reject non-finite coordinates early."""

    try:
        position = np.asarray(r_eci_km, dtype=float).reshape(3)
    except (TypeError, ValueError) as exc:
        raise ValueError("r_eci_km must contain exactly three finite values") from exc
    if not np.all(np.isfinite(position)):
        raise ValueError("r_eci_km must contain exactly three finite values")
    return position


def _radial_altitude_km_from_eci(r_eci_km: np.ndarray) -> float:
    r_vec = _validated_eci_position(r_eci_km)
    return _radial_altitude_from_validated_position(r_vec)


def _radial_altitude_from_validated_position(r_vec: np.ndarray) -> float:
    r2 = dot3(r_vec, r_vec)
    if r2 <= 0.0:
        return 0.0
    return max(0.0, math.sqrt(r2) - EARTH_RADIUS_KM)


def _ecef_from_eci_for_atmosphere(r_eci_km: np.ndarray, t_s: float, env: dict) -> np.ndarray:
    r_vec = _validated_eci_position(r_eci_km)
    prepared = env.get("_prepared_density_frame")
    if prepared is not None:
        return prepared.rotation(float(t_s)) @ r_vec
    frame_model = str(env.get("density_frame_model", env.get("drag_frame_model", "simple"))).strip().lower()
    eop_path = env.get("density_eop_path", env.get("drag_eop_path"))
    if env.get("_rust_numeric_backend") == "rust" and not _is_eop_frame_model(frame_model):
        from sim.rust_environment_backend import try_rotation

        rotation = try_rotation(
            float(t_s),
            jd_utc_start=None if env.get("jd_utc_start") is None else float(env["jd_utc_start"]),
            earth_rotation_rad_s=7.2921159e-5,
        )
        if rotation is not None:
            return rotation @ r_vec
    return eci_to_ecef_harmonic(
        r_vec,
        float(t_s),
        jd_utc_start=env.get("jd_utc_start"),
        frame_model=frame_model,
        eop_path=None if eop_path is None else str(eop_path),
        dut1_s=None if env.get("dut1_s") is None else float(env["dut1_s"]),
        xp_arcsec=None if env.get("xp_arcsec") is None else float(env["xp_arcsec"]),
        yp_arcsec=None if env.get("yp_arcsec") is None else float(env["yp_arcsec"]),
        dat_s=None if env.get("dat_s") is None else float(env["dat_s"]),
        tt_minus_utc_s=None if env.get("tt_minus_utc_s") is None else float(env["tt_minus_utc_s"]),
        ddpsi_rad=float(env.get("ddpsi_rad", 0.0) or 0.0),
        ddeps_rad=float(env.get("ddeps_rad", 0.0) or 0.0),
        eop_extrapolation=str(env.get("eop_extrapolation", "error") or "error"),
    )


@lru_cache(maxsize=16)
def _is_eop_frame_model(frame_model: str) -> bool:
    try:
        return normalize_frame_model(frame_model) == FRAME_MODEL_IAU76_80_EOP
    except ValueError:
        return False


def _altitude_km_from_eci(r_eci_km: np.ndarray, t_s: float, env: dict | None = None) -> float:
    env = {} if env is None else env
    coordinates = env.get("_native_density_coordinates")
    if coordinates is not None:
        return float(coordinates[0])
    r_ecef_km = _ecef_from_eci_for_atmosphere(r_eci_km, t_s, env)
    if str(env.get("geodetic_model", "")).lower() == "wgs84":
        if env.get("_rust_numeric_backend") == "rust":
            from sim.rust_environment_backend import try_geodetic

            geodetic = try_geodetic(r_ecef_km)
            if geodetic is not None:
                return float(max(float(geodetic[2]), 0.0))
        alt_km = ecef_to_geodetic_altitude_km(r_ecef_km)
        return float(max(alt_km, 0.0))
    return float(max(0.0, np.linalg.norm(r_ecef_km) - EARTH_RADIUS_KM))


def _spherical_lat_lon_deg_from_eci(r_eci_km: np.ndarray, t_s: float, env: dict | None = None) -> tuple[float, float]:
    env = {} if env is None else env
    coordinates = env.get("_native_density_coordinates")
    if coordinates is not None:
        return (float(coordinates[1]), float(coordinates[2]))
    r_ecef_km = _ecef_from_eci_for_atmosphere(r_eci_km, t_s, env)
    if str(env.get("geodetic_model", "")).lower() == "wgs84":
        if env.get("_rust_numeric_backend") == "rust":
            from sim.rust_environment_backend import try_geodetic

            geodetic = try_geodetic(r_ecef_km)
            if geodetic is not None:
                return float(geodetic[0]), float(geodetic[1])
        lat, lon, _ = ecef_to_geodetic_deg_km(r_ecef_km)
        return float(lat), float(lon)
    r = float(np.linalg.norm(r_ecef_km))
    if r <= 0.0:
        return 0.0, 0.0
    x, y, z = r_ecef_km
    lat = np.degrees(np.arcsin(np.clip(z / r, -1.0, 1.0)))
    lon = np.degrees(np.arctan2(y, x))
    return float(lat), float(lon)


def _altitude_lat_lon_deg_from_eci(
    r_eci_km: np.ndarray,
    t_s: float,
    env: dict | None = None,
) -> tuple[float, float, float]:
    """Return atmosphere position coordinates from one ECI-to-ECEF conversion."""
    env = {} if env is None else env
    coordinates = env.get("_native_density_coordinates")
    if coordinates is not None:
        return tuple(float(value) for value in coordinates)
    r_ecef_km = _ecef_from_eci_for_atmosphere(r_eci_km, t_s, env)
    if str(env.get("geodetic_model", "")).lower() == "wgs84":
        if env.get("_rust_numeric_backend") == "rust":
            from sim.rust_environment_backend import try_geodetic

            geodetic = try_geodetic(r_ecef_km)
            if geodetic is not None:
                return float(max(float(geodetic[2]), 0.0)), float(geodetic[0]), float(geodetic[1])
        if acceleration_enabled_from_mode():
            lat_deg, lon_deg, alt_km = _compiled_ecef_to_geodetic_deg_km()(r_ecef_km)
        else:
            lat_deg, lon_deg, alt_km = ecef_to_geodetic_deg_km(r_ecef_km)
        return float(max(alt_km, 0.0)), float(lat_deg), float(lon_deg)
    radius_km = float(np.linalg.norm(r_ecef_km))
    alt_km = float(max(0.0, radius_km - EARTH_RADIUS_KM))
    if radius_km <= 0.0:
        return alt_km, 0.0, 0.0
    x, y, z = r_ecef_km
    lat_deg = np.degrees(np.arcsin(np.clip(z / radius_km, -1.0, 1.0)))
    lon_deg = np.degrees(np.arctan2(y, x))
    return alt_km, float(lat_deg), float(lon_deg)


def density_exponential(r_eci_km: np.ndarray, t_s: float, env: dict | None = None) -> float:
    r_vec = _validated_eci_position(r_eci_km)
    alt_km = _radial_altitude_from_validated_position(r_vec)
    if env is None or _EXPONENTIAL_ENV_KEYS.isdisjoint(env):
        ceiling_km = 1000.0
        rho_ref = 1.225
        reference_altitude_km = 0.0
        scale_height_km = 8.5
    else:
        ceiling_km = float(env.get("exponential_ceiling_altitude_km", 1000.0))
        rho_ref = float(env.get("exponential_reference_density_kg_m3", 1.225))
        reference_altitude_km = float(env.get("exponential_reference_altitude_km", 0.0))
        scale_height_km = float(env.get("exponential_scale_height_km", 8.5))
    if not math.isfinite(ceiling_km) or ceiling_km < 0.0:
        raise ValueError("exponential_ceiling_altitude_km must be finite and nonnegative.")
    if not math.isfinite(rho_ref) or rho_ref < 0.0:
        raise ValueError("exponential_reference_density_kg_m3 must be finite and nonnegative.")
    if not math.isfinite(reference_altitude_km):
        raise ValueError("exponential_reference_altitude_km must be finite.")
    if not math.isfinite(scale_height_km) or scale_height_km <= 0.0:
        raise ValueError("exponential_scale_height_km must be finite and positive.")
    if env is not None and env.get("_rust_numeric_backend") == "rust":
        from sim.rust_environment_backend import try_exponential_density, try_radial_exponential_density
        native = try_radial_exponential_density(
            r_vec,
            radius_km=EARTH_RADIUS_KM,
            reference_density_kg_m3=rho_ref,
            reference_altitude_km=reference_altitude_km,
            scale_height_km=scale_height_km,
            ceiling_km=ceiling_km,
        )
        if native is not None:
            return native
        native = try_exponential_density(
            alt_km,
            reference_density_kg_m3=rho_ref,
            reference_altitude_km=reference_altitude_km,
            scale_height_km=scale_height_km,
            ceiling_km=ceiling_km,
        )
        if native is not None:
            return native
    if alt_km > ceiling_km:
        return 0.0
    return rho_ref * math.exp(-(alt_km - reference_altitude_km) / scale_height_km)


def altitude_km_from_eci(r_eci_km: np.ndarray, t_s: float, env: dict | None = None) -> float:
    return _altitude_km_from_eci(r_eci_km, t_s, env=env)


def _datetime_from_env_t_s(env: dict, t_s: float) -> datetime:
    jd_utc = env.get("jd_utc")
    if jd_utc is not None:
        return julian_date_to_datetime(float(jd_utc))
    jd_utc_start = env.get("jd_utc_start")
    if jd_utc_start is not None:
        return julian_date_to_datetime(float(jd_utc_start) + float(t_s) / 86400.0)
    base_epoch = env.get("atmo_epoch_utc", datetime(2020, 1, 1, tzinfo=timezone.utc))
    if isinstance(base_epoch, datetime):
        if base_epoch.tzinfo is None:
            base_epoch = base_epoch.replace(tzinfo=timezone.utc)
        return base_epoch + timedelta(seconds=float(t_s))
    return datetime(2020, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=float(t_s))


@lru_cache(maxsize=32768)
def _local_solar_time_epoch_terms(
    jd: float,
    eop_path: str | None,
    dut1_s: float | None,
    dat_s: float | None,
    tt_minus_utc_s: float | None,
    ddpsi_rad: float,
    ddeps_rad: float,
    eop_extrapolation: str = "error",
) -> tuple[float, float]:
    sidereal = apparent_sidereal_time_hpop_like(
        jd,
        eop_path,
        dut1_s=dut1_s,
        dat_s=dat_s,
        tt_minus_utc_s=tt_minus_utc_s,
        ddpsi_rad=ddpsi_rad,
        ddeps_rad=ddeps_rad,
        eop_extrapolation=eop_extrapolation,
    )
    sun_eci = sun_position_eci_km_enhanced(jd)
    sun_ra = math.atan2(float(sun_eci[1]), float(sun_eci[0]))
    return float(sidereal), float(sun_ra)


def _local_solar_time_hr(lon_deg: float, dt_utc: datetime, env: dict) -> float:
    """Legacy MSIS-86 solar-time convention; NRLMSISE uses a common frame."""
    jd = datetime_to_julian_date(dt_utc)
    frame_model = str(env.get("density_frame_model", env.get("drag_frame_model", ""))).strip().lower()
    eop_path_raw = env.get("density_eop_path", env.get("drag_eop_path")) if _is_eop_frame_model(frame_model) else None
    sidereal, sun_ra = _local_solar_time_epoch_terms(
        float(jd),
        None if eop_path_raw is None else str(eop_path_raw),
        None if env.get("dut1_s") is None else float(env["dut1_s"]),
        None if env.get("dat_s") is None else float(env["dat_s"]),
        None if env.get("tt_minus_utc_s") is None else float(env["tt_minus_utc_s"]),
        float(env.get("ddpsi_rad", 0.0) or 0.0),
        float(env.get("ddeps_rad", 0.0) or 0.0),
        str(env.get("eop_extrapolation", "error") or "error"),
    )
    hour_angle = (sidereal + math.radians(float(lon_deg)) - sun_ra + math.pi) % (2.0 * math.pi) - math.pi
    return float((12.0 + hour_angle * 12.0 / math.pi) % 24.0)


def _nrlmsise00_sun_longitude_rad(
    dt_utc: datetime,
    t_s: float,
    env: dict,
    *,
    rotation: np.ndarray | None = None,
) -> float:
    """Resolve the Sun in the same Earth-fixed frame as density coordinates.

    The ephemeris owner supplies J2000 vectors, including explicit/sampled
    inputs and their coverage guards. Sidereal time alone cannot transform
    their right ascension into the full IAU-76/80 density frame.
    """
    solar_env = env
    if env.get("jd_utc") is None and env.get("jd_utc_start") is None:
        solar_env = {**env, "jd_utc": datetime_to_julian_date(dt_utc)}
    sun = resolve_sun_position_eci_km(solar_env, float(t_s))
    sun_fixed = (
        _ecef_from_eci_for_atmosphere(sun, t_s, env)
        if rotation is None
        else rotation @ sun
    )
    return math.atan2(float(sun_fixed[1]), float(sun_fixed[0]))


def _nrlmsise00_local_solar_time_hr(lon_deg: float, dt_utc: datetime, t_s: float, env: dict) -> float:
    # Internal native stage input shares the exact resource-bound matrix.
    # The ephemeris resolver and NumPy Sun projection retain their semantics.
    rotation = env.get("_native_density_rotation")
    sun_lon = (
        _nrlmsise00_sun_longitude_rad(dt_utc, t_s, env) if rotation is None
        else _nrlmsise00_sun_longitude_rad(dt_utc, t_s, env, rotation=rotation)
    )
    hour_angle = (math.radians(float(lon_deg)) - sun_lon + math.pi) % (2.0 * math.pi) - math.pi
    return float((12.0 + hour_angle * 12.0 / math.pi) % 24.0)


def density_ussa1976(r_eci_km: np.ndarray, t_s: float, env: dict | None = None) -> float:
    """
    Approximate US Standard Atmosphere 1976 density profile.

    - 0..86 km: standard lapse-rate layers via geopotential-altitude equations.
    - 86..1000 km: log-linear interpolation on tabulated USSA-1976 reference densities.
    """
    if env is not None and str(env.get("geodetic_model", "")).lower() == "wgs84":
        alt_km = _altitude_km_from_eci(r_eci_km, t_s, env=env)
    else:
        alt_km = _radial_altitude_km_from_eci(r_eci_km)

    if alt_km <= 86.0:
        h = alt_km * 1e3
        i = int(np.searchsorted(_USSA1976_HB_M, h, side="right") - 1)
        i = max(0, min(i, _USSA1976_LB_K_M.size - 1))
        h0 = float(_USSA1976_HB_M[i])
        lapse = float(_USSA1976_LB_K_M[i])
        t0 = float(_USSA1976_TB_K[i])
        p0 = float(_USSA1976_PB_PA[i])
        if abs(lapse) < 1e-12:
            t = t0
            p = p0 * math.exp(-_USSA1976_G0_M_S2 * (h - h0) / (_USSA1976_R_AIR_J_KG_K * t))
        else:
            t = t0 + lapse * (h - h0)
            p = p0 * (t / t0) ** (-_USSA1976_G0_M_S2 / (_USSA1976_R_AIR_J_KG_K * lapse))
        rho = p / (_USSA1976_R_AIR_J_KG_K * t)
        return float(max(rho, 0.0))

    if alt_km > 1000.0:
        return 0.0

    if alt_km <= _USSA1976_HIGH_ALT_KM_SCALAR[0]:
        lrho = _USSA1976_HIGH_LOG_RHO_SCALAR[0]
    elif alt_km >= _USSA1976_HIGH_ALT_KM_SCALAR[-1]:
        lrho = _USSA1976_HIGH_LOG_RHO_SCALAR[-1]
    else:
        index = bisect_right(_USSA1976_HIGH_ALT_KM_SCALAR, alt_km) - 1
        lrho = (
            _USSA1976_HIGH_LOG_RHO_SLOPE[index] * (alt_km - _USSA1976_HIGH_ALT_KM_SCALAR[index])
            + _USSA1976_HIGH_LOG_RHO_SCALAR[index]
        )
    return math.exp(lrho)


def _temperature_from_altitude_k_approx(alt_km: float) -> float:
    h = float(max(alt_km, 0.0))
    if h < 11.0:
        return 288.15 - 6.5 * h
    if h < 20.0:
        return 216.65
    if h < 32.0:
        return 216.65 + (h - 20.0) * 1.0
    if h < 47.0:
        return 228.65 + (h - 32.0) * 2.8
    if h < 51.0:
        return 270.65
    if h < 71.0:
        return 270.65 - (h - 51.0) * 2.8
    if h < 86.0:
        return 214.65 - (h - 71.0) * 2.0
    return 186.87


def density_nrlmsise00(r_eci_km: np.ndarray, t_s: float, env: dict | None = None) -> float:
    """
    Local NRLMSISE-00 density model copied from the MATLAB HPOP implementation.

    Supported backends:
    - env-provided callable: env["nrlmsise00_density_callable"](alt_km, lat_deg, lon_deg, dt_utc, env) -> kg/m^3
    - source-local OEL backend with direct env f107/f107a/ap inputs, or optional HPOP-style SW-All table input.
    """
    env_source = {} if env is None else env
    alt_km, lat_deg, lon_deg = _altitude_lat_lon_deg_from_eci(r_eci_km, t_s, env=env_source)

    dt_utc = _datetime_from_env_t_s(env_source, t_s)

    custom_fn = env_source.get("nrlmsise00_density_callable", None)
    if callable(custom_fn):
        env_local = dict(env_source)
        return float(max(0.0, custom_fn(alt_km, lat_deg, lon_deg, dt_utc, env_local)))
    lst_hr = env_source.get("nrlmsise00_lst_hr")
    if lst_hr is None:
        lst_hr = _nrlmsise00_local_solar_time_hr(lon_deg, dt_utc, t_s, env_source)

    return float(
        max(
            0.0,
            _nrlmsise00_backend()(
                alt_km,
                lat_deg,
                lon_deg,
                dt_utc,
                env_source,
                lst_hr=None if lst_hr is None else float(lst_hr),
            ),
        )
    )


def density_msis86(r_eci_km: np.ndarray, t_s: float, env: dict | None = None) -> float:
    env = {} if env is None else dict(env)
    alt_km, lat_deg, lon_deg = _altitude_lat_lon_deg_from_eci(r_eci_km, t_s, env=env)

    dt_utc = _datetime_from_env_t_s(env, t_s)

    custom_fn = env.get("msis86_density_callable", None)
    if callable(custom_fn):
        return float(max(0.0, custom_fn(alt_km, lat_deg, lon_deg, dt_utc, env)))
    if bool(env.get("msis86_hpop_angle_compat", True)):
        if env.get("msis86_lst_hr") is None:
            env["msis86_lst_hr"] = _local_solar_time_hr(lon_deg, dt_utc, env)
        lat_deg = math.radians(lat_deg)
        lon_deg = math.radians(lon_deg)
    return float(max(0.0, _msis86_backend()(alt_km, lat_deg, lon_deg, dt_utc, env)))


def density_jb2008(r_eci_km: np.ndarray, t_s: float, env: dict | None = None) -> float:
    """
    JB2008 density model via externally supplied backend callable.

    Required:
    - env["jb2008_density_callable"](alt_km, lat_deg, lon_deg, dt_utc, env) -> kg/m^3
    """
    env = {} if env is None else dict(env)
    alt_km, lat_deg, lon_deg = _altitude_lat_lon_deg_from_eci(r_eci_km, t_s, env=env)

    dt_utc = _datetime_from_env_t_s(env, t_s)

    custom_fn = env.get("jb2008_density_callable", None)
    if callable(custom_fn):
        return float(max(0.0, custom_fn(alt_km, lat_deg, lon_deg, dt_utc, env)))
    return float(max(0.0, _jb_backends()[1](alt_km, lat_deg, lon_deg, dt_utc, env)))


def density_jb2006(r_eci_km: np.ndarray, t_s: float, env: dict | None = None) -> float:
    env = {} if env is None else dict(env)
    alt_km, lat_deg, lon_deg = _altitude_lat_lon_deg_from_eci(r_eci_km, t_s, env=env)

    dt_utc = _datetime_from_env_t_s(env, t_s)

    custom_fn = env.get("jb2006_density_callable", None)
    if callable(custom_fn):
        return float(max(0.0, custom_fn(alt_km, lat_deg, lon_deg, dt_utc, env)))
    return float(max(0.0, _jb_backends()[0](alt_km, lat_deg, lon_deg, dt_utc, env)))


def density_jacchia70(r_eci_km: np.ndarray, t_s: float, env: dict | None = None) -> float:
    env = {} if env is None else dict(env)
    alt_km, lat_deg, lon_deg = _altitude_lat_lon_deg_from_eci(r_eci_km, t_s, env=env)

    dt_utc = _datetime_from_env_t_s(env, t_s)

    custom_fn = env.get("jacchia70_density_callable", None)
    if callable(custom_fn):
        return float(max(0.0, custom_fn(alt_km, lat_deg, lon_deg, dt_utc, env)))
    return float(max(0.0, _jacchia70_backend()(alt_km, lat_deg, lon_deg, dt_utc, env)))


def density_harris_priester(r_eci_km: np.ndarray, t_s: float, env: dict | None = None) -> float:
    return _harris_priester_backend()(r_eci_km, t_s, env=env)


def atmosphere_state_from_model(
    model: AtmosphereModelName,
    r_eci_km: np.ndarray,
    t_s: float,
    env: dict | None = None,
) -> dict[str, float]:
    """
    Return atmosphere state dictionary with at least:
    - density_kg_m3
    - temperature_k
    - pressure_pa
    - sound_speed_m_s
    """
    env_local = {} if env is None else dict(env)
    alt_km = _altitude_km_from_eci(r_eci_km, t_s, env=env_local)
    lat_deg, lon_deg = _spherical_lat_lon_deg_from_eci(r_eci_km, t_s, env=env_local)
    r_air = float(env_local.get("air_gas_constant_j_kg_k", 287.05287))
    gamma = float(env_local.get("air_gamma", 1.4))

    dt_utc = _datetime_from_env_t_s(env_local, t_s)

    m = str(model).lower()
    cb = None
    if m == "nrlmsise00":
        cb = env_local.get("nrlmsise00_density_callable", None)
    elif m in {"msis86", "msis-86", "hpop_msis86"}:
        cb = env_local.get("msis86_density_callable", None)
    elif m in {"jacchia70", "jacchia-70", "hpop_jacchia70"}:
        cb = env_local.get("jacchia70_density_callable", None)
    elif m == "jb2006":
        cb = env_local.get("jb2006_density_callable", None)
    elif m == "jb2008":
        cb = env_local.get("jb2008_density_callable", None)

    if callable(cb):
        out = cb(alt_km, lat_deg, lon_deg, dt_utc, env_local)
        if isinstance(out, dict):
            rho = float(out.get("density_kg_m3", out.get("rho_kg_m3", 0.0)))
            t_k = float(out.get("temperature_k", _temperature_from_altitude_k_approx(alt_km)))
            p_pa = float(out.get("pressure_pa", max(rho, 0.0) * r_air * max(t_k, 1.0)))
            a_m_s = float(out.get("sound_speed_m_s", np.sqrt(max(gamma * r_air * max(t_k, 1.0), 1e-9))))
            return {
                "density_kg_m3": max(rho, 0.0),
                "temperature_k": max(t_k, 1.0),
                "pressure_pa": max(p_pa, 0.0),
                "sound_speed_m_s": max(a_m_s, 1e-3),
            }
        rho = float(out)
        t_k = _temperature_from_altitude_k_approx(alt_km)
        p_pa = max(rho, 0.0) * r_air * t_k
        a_m_s = float(np.sqrt(max(gamma * r_air * t_k, 1e-9)))
        return {
            "density_kg_m3": max(rho, 0.0),
            "temperature_k": t_k,
            "pressure_pa": max(p_pa, 0.0),
            "sound_speed_m_s": max(a_m_s, 1e-3),
        }

    rho = density_from_model(model, r_eci_km, t_s, env=env_local)
    t_k = _temperature_from_altitude_k_approx(alt_km)
    p_pa = max(rho, 0.0) * r_air * t_k
    a_m_s = float(np.sqrt(max(gamma * r_air * t_k, 1e-9)))
    return {
        "density_kg_m3": max(float(rho), 0.0),
        "temperature_k": max(t_k, 1.0),
        "pressure_pa": max(float(p_pa), 0.0),
        "sound_speed_m_s": max(a_m_s, 1e-3),
    }


def density_from_model(
    model: AtmosphereModelName,
    r_eci_km: np.ndarray,
    t_s: float,
    env: dict | None = None,
) -> float:
    m = str(model).lower()
    if m == "exponential":
        return density_exponential(r_eci_km, t_s, env=env)
    if m == "ussa1976":
        return density_ussa1976(r_eci_km, t_s, env=env)
    if m == "nrlmsise00":
        return density_nrlmsise00(r_eci_km, t_s, env=env)
    if m in {"msis86", "msis-86", "hpop_msis86"}:
        return density_msis86(r_eci_km, t_s, env=env)
    if m in {"jacchia70", "jacchia-70", "hpop_jacchia70"}:
        return density_jacchia70(r_eci_km, t_s, env=env)
    if m == "jb2006":
        return density_jb2006(r_eci_km, t_s, env=env)
    if m == "jb2008":
        return density_jb2008(r_eci_km, t_s, env=env)
    if m in {"harris_priester", "harris-priester", "hp", "hpop_harris_priester"}:
        return density_harris_priester(r_eci_km, t_s, env=env)
    raise ValueError(f"Unknown atmosphere model '{model}'.")
