from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np

from sim.acceleration.settings import acceleration_enabled_from_mode
from sim.dynamics.orbit.epoch import (
    datetime_to_julian_date,
    sun_position_eci_km_enhanced,
    sun_position_eci_km_simple,
)
from sim.dynamics.orbit.frames import eci_to_ecef_rotation_hpop_like
from sim.utils.geodesy import WGS84_E2, geodetic_to_ecef_km

_XAMBAR_COEFFICIENTS = (28.15204, -8.5586e-2, 1.2840e-4, -1.0056e-5, -1.0210e-5, 1.5044e-6, 9.9826e-8)
_JB_ALPHA = (0.0, 0.0, 0.0, 0.0, -0.38)
_JB_AMW = (28.0134, 31.9988, 15.9994, 39.9480, 4.0026, 1.00797)
_JB_FRAC = (0.78110, 0.20955, 9.3400e-3, 1.2890e-5)
_JB_WT = (0.311111111111111, 1.422222222222222, 0.533333333333333, 1.422222222222222, 0.311111111111111)
_JB_WT_TAIL = _JB_WT[1:]
_JB_CHT = (0.22, -0.20e-2, 0.115e-2, -0.211e-5)
_AL10 = math.log(10.0)
_TWOPI = 2.0 * math.pi
_PIOV2 = 0.5 * math.pi
_AVOGAD = 6.02257e26
_RSTAR = 8314.32


@lru_cache(maxsize=1)
def _default_jb2008_sol_path() -> Path:
    return (
        Path(__file__).resolve().parents[3]
        / "validation/High Precision Orbit Propagator_4-2/High Precision Orbit Propagator_4.2.2/SOLFSMY.txt"
    )


@lru_cache(maxsize=1)
def _default_jb2008_dtc_path() -> Path:
    return (
        Path(__file__).resolve().parents[3]
        / "validation/High Precision Orbit Propagator_4-2/High Precision Orbit Propagator_4.2.2/DTCFILE.txt"
    )


@lru_cache(maxsize=1)
def _default_jb2006_ap_path() -> Path:
    return (
        Path(__file__).resolve().parents[3]
        / "validation/High Precision Orbit Propagator_4-2/High Precision Orbit Propagator_4.2.2/SOLRESAP.txt"
    )


@lru_cache(maxsize=4)
def _load_soldata(path: str) -> np.ndarray:
    arr = np.loadtxt(str(Path(path).expanduser().resolve()), dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 11:
        raise ValueError(f"Unexpected SOLFSMY shape {arr.shape} from {path}")
    return arr.T


@lru_cache(maxsize=4)
def _load_dtcdata(path: str) -> np.ndarray:
    arr = np.loadtxt(str(Path(path).expanduser().resolve()), dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 26:
        raise ValueError(f"Unexpected DTCFILE shape {arr.shape} from {path}")
    return arr.T


@lru_cache(maxsize=4)
def _load_apdata(path: str) -> np.ndarray:
    arr = np.loadtxt(str(Path(path).expanduser().resolve()), dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 12:
        raise ValueError(f"Unexpected SOLRESAP shape {arr.shape} from {path}")
    return arr.T


@lru_cache(maxsize=4096)
def _sol_day_index(path: str, julian_day: int) -> int:
    indices = np.where(_load_soldata(path)[2, :] == julian_day)[0]
    return -1 if indices.size == 0 else int(indices[0])


@lru_cache(maxsize=4096)
def _dtc_day_index(path: str, year: int, day_of_year: int) -> int:
    dtc = _load_dtcdata(path)
    indices = np.where((dtc[0, :] == year) & (dtc[1, :] == day_of_year))[0]
    return -1 if indices.size == 0 else int(indices[0])


@lru_cache(maxsize=4096)
def _ap_day_index(path: str, year: int, day_of_year: int) -> int:
    apdata = _load_apdata(path)
    indices = np.where((apdata[0, :] == year) & (apdata[1, :] == day_of_year))[0]
    return -1 if indices.size == 0 else int(indices[0])


@lru_cache(maxsize=8)
def _resolve_table_path(raw: str | None, default: Path) -> str:
    p = default if raw in (None, "") else Path(str(raw)).expanduser()
    if not p.is_absolute():
        p = Path(__file__).resolve().parents[3] / p
    return str(p.resolve())


def _day_of_year(dt_utc: datetime) -> float:
    start = datetime(dt_utc.year, 1, 1, tzinfo=timezone.utc)
    return 1.0 + (dt_utc - start).total_seconds() / 86400.0


def _gd2gc_rad(lat_gd_rad: float) -> float:
    return float(math.atan((1.0 - WGS84_E2) * math.tan(float(lat_gd_rad))))


def _sun_position_from_env(jd_utc: float, env: dict) -> np.ndarray:
    r_sun = env.get("sun_pos_eci_km")
    if r_sun is not None:
        return np.asarray(r_sun, dtype=float).reshape(3)
    sun_model = str(env.get("atmosphere_sun_model", env.get("sun_model", ""))).strip().lower()
    if sun_model in {"hpop_simple", "hpop_validation_simple", "validation_simple", "simple"}:
        return sun_position_eci_km_simple(float(jd_utc))
    return sun_position_eci_km_enhanced(float(jd_utc))


def _position_eci_from_geodetic(lat_deg: float, lon_deg: float, alt_km: float, jd_utc: float, env: dict) -> np.ndarray:
    r_ecef_km = geodetic_to_ecef_km(lat_deg, lon_deg, alt_km)
    eop_path = env.get("drag_eop_path") or env.get("spherical_harmonics_eop_path")
    rot = eci_to_ecef_rotation_hpop_like(
        0.0, jd_utc_start=float(jd_utc), eop_path=None if eop_path is None else str(eop_path)
    )
    return rot.T @ r_ecef_km


def _xambar(z_km: float) -> float:
    c = np.array([28.15204, -8.5586e-2, 1.2840e-4, -1.0056e-5, -1.0210e-5, 1.5044e-6, 9.9826e-8], dtype=float)
    dz = float(z_km) - 100.0
    amb = float(c[-1])
    for coeff in c[-2::-1]:
        amb = dz * amb + float(coeff)
    return amb


def _xgrav(z_km: float) -> float:
    return 9.80665 / (1.0 + float(z_km) / 6356.766) ** 2


def _xlocal(z_km: float, tc: np.ndarray) -> float:
    dz = float(z_km) - 125.0
    if dz > 0.0:
        return float(tc[0] + tc[2] * math.atan(tc[3] * dz * (1.0 + 4.5e-6 * dz**2.5)))
    return float((((-9.8204695e-6 * dz - 7.3039742e-4) * dz * dz + 1.0) * dz * tc[1]) + tc[0])


def _validate_jb_altitude(alt_km: float, model_name: str) -> float:
    altitude = float(alt_km)
    if not math.isfinite(altitude):
        raise ValueError(f"{model_name} requires a finite altitude in km.")
    if altitude < 90.0:
        raise ValueError(f"{model_name} is defined for altitudes at or above 90 km; got {altitude:g} km.")
    return altitude


def _integrate_lower_atmosphere_python(
    zht: float, tc: tuple[float, float, float, float],
) -> tuple[float, float, float, float, float, float, float, float, float]:
    """Authoritative JB2006/JB2008 90–105 km quadrature, unchanged ordering."""
    zht = _validate_jb_altitude(zht, "JB2006/JB2008")
    tc0, tc1, tc2, tc3 = tc
    xambar_c0, xambar_c1, xambar_c2, xambar_c3, xambar_c4, xambar_c5, xambar_c6 = _XAMBAR_COEFFICIENTS
    r1 = 0.010
    wt, wt_tail, atan = _JB_WT, _JB_WT_TAIL, math.atan
    z1 = 90.0
    z2 = min(zht, 105.0)
    al = math.log(z2 / z1)
    n = int(math.floor(al / r1) + 1)
    zr = math.exp(al / n)
    ambar1 = _xambar(z1)
    tloc1 = _xlocal(z1, tc)
    zend = z1
    sum2 = 0.0
    ain = ambar1 * _xgrav(z1) / tloc1
    ambar2 = ambar1
    tloc2 = tloc1
    gravl = _xgrav(z1)
    for _ in range(n):
        z = zend
        zend = zr * z
        dz = 0.25 * (zend - z)
        sum1 = wt[0] * ain
        for weight in wt_tail:
            z = z + dz
            z_minus_100 = z - 100.0
            ambar2 = z_minus_100 * xambar_c6 + xambar_c5
            ambar2 = z_minus_100 * ambar2 + xambar_c4
            ambar2 = z_minus_100 * ambar2 + xambar_c3
            ambar2 = z_minus_100 * ambar2 + xambar_c2
            ambar2 = z_minus_100 * ambar2 + xambar_c1
            ambar2 = z_minus_100 * ambar2 + xambar_c0
            z_minus_125 = z - 125.0
            if z_minus_125 > 0.0:
                tloc2 = tc0 + tc2 * atan(tc3 * z_minus_125 * (1.0 + 4.5e-6 * z_minus_125**2.5))
            else:
                tloc2 = (
                    (((-9.8204695e-6 * z_minus_125 - 7.3039742e-4) * z_minus_125 * z_minus_125 + 1.0)
                    * z_minus_125
                    * tc1)
                    + tc0
                )
            gravl = 9.80665 / (1.0 + z / 6356.766) ** 2
            ain = ambar2 * gravl / tloc2
            sum1 += weight * ain
        sum2 += dz * sum1

    return zend, z, tloc1, tloc2, ambar1, ambar2, gravl, ain, sum2


def _integrate_lower_atmosphere(zht: float, tc: tuple, env: dict) -> tuple:
    if env.get("_rust_numeric_backend") == "rust" and zht >= 90.0:
        from sim.rust_environment_backend import try_jb_lower_atmosphere

        native = try_jb_lower_atmosphere(zht, tc)
        if native is not None:
            return native
    return _integrate_lower_atmosphere_python(zht, tc)


def _integrate_upper_atmosphere_python(
    zend: float,
    z_current: float,
    z_target: float,
    r_step: float,
    ain: float,
    tc: tuple[float, float, float, float],
) -> tuple[float, float, float, float, float, float]:
    """Integrate one fixed four-node upper-atmosphere quadrature segment."""
    al = math.log(z_target / z_current)
    n = int(math.floor(al / r_step) + 1)
    zr = math.exp(al / n)
    tc0, tc1, tc2, tc3 = tc
    weight0, weight1, weight2, weight3, weight4 = _JB_WT
    atan = math.atan
    total = 0.0
    z = zend
    tloc = tc0
    gravl = 0.0

    for _ in range(n):
        z = zend
        zend = zr * z
        dz = 0.25 * (zend - z)
        sum1 = weight0 * ain

        z = z + dz
        z_minus_125 = z - 125.0
        if z_minus_125 > 0.0:
            tloc = tc0 + tc2 * atan(tc3 * z_minus_125 * (1.0 + 4.5e-6 * z_minus_125**2.5))
        else:
            tloc = (
                (((-9.8204695e-6 * z_minus_125 - 7.3039742e-4) * z_minus_125 * z_minus_125 + 1.0)
                * z_minus_125
                * tc1)
                + tc0
            )
        gravl = 9.80665 / (1.0 + z / 6356.766) ** 2
        ain = gravl / tloc
        sum1 += weight1 * ain

        z = z + dz
        z_minus_125 = z - 125.0
        if z_minus_125 > 0.0:
            tloc = tc0 + tc2 * atan(tc3 * z_minus_125 * (1.0 + 4.5e-6 * z_minus_125**2.5))
        else:
            tloc = (
                (((-9.8204695e-6 * z_minus_125 - 7.3039742e-4) * z_minus_125 * z_minus_125 + 1.0)
                * z_minus_125
                * tc1)
                + tc0
            )
        gravl = 9.80665 / (1.0 + z / 6356.766) ** 2
        ain = gravl / tloc
        sum1 += weight2 * ain

        z = z + dz
        z_minus_125 = z - 125.0
        if z_minus_125 > 0.0:
            tloc = tc0 + tc2 * atan(tc3 * z_minus_125 * (1.0 + 4.5e-6 * z_minus_125**2.5))
        else:
            tloc = (
                (((-9.8204695e-6 * z_minus_125 - 7.3039742e-4) * z_minus_125 * z_minus_125 + 1.0)
                * z_minus_125
                * tc1)
                + tc0
            )
        gravl = 9.80665 / (1.0 + z / 6356.766) ** 2
        ain = gravl / tloc
        sum1 += weight3 * ain

        z = z + dz
        z_minus_125 = z - 125.0
        if z_minus_125 > 0.0:
            tloc = tc0 + tc2 * atan(tc3 * z_minus_125 * (1.0 + 4.5e-6 * z_minus_125**2.5))
        else:
            tloc = (
                (((-9.8204695e-6 * z_minus_125 - 7.3039742e-4) * z_minus_125 * z_minus_125 + 1.0)
                * z_minus_125
                * tc1)
                + tc0
            )
        gravl = 9.80665 / (1.0 + z / 6356.766) ** 2
        ain = gravl / tloc
        sum1 += weight4 * ain
        total += dz * sum1

    return zend, z, tloc, gravl, ain, total


@lru_cache(maxsize=1)
def _compiled_integrate_upper_atmosphere():
    from numba import njit

    return njit(cache=True, fastmath=False)(_integrate_upper_atmosphere_python)


def _integrate_upper_atmosphere(
    zend: float,
    z_current: float,
    z_target: float,
    r_step: float,
    ain: float,
    tc: tuple[float, float, float, float],
    env: dict,
) -> tuple[float, float, float, float, float, float]:
    if env.get("_rust_numeric_backend") == "rust":
        from sim.rust_environment_backend import try_jb_upper_atmosphere

        native = try_jb_upper_atmosphere(zend, z_current, z_target, r_step, ain, tc)
        if native is not None:
            return native
    if acceleration_enabled_from_mode():
        result = _compiled_integrate_upper_atmosphere()(zend, z_current, z_target, r_step, ain, tc)
        return tuple(float(value) for value in result)
    return _integrate_upper_atmosphere_python(zend, z_current, z_target, r_step, ain, tc)


@lru_cache(maxsize=1)
def _dtsub_coefficients() -> tuple[np.ndarray, np.ndarray]:
    b = np.array(
        [
            -0.457512297e1,
            -0.512114909e1,
            -0.693003609e2,
            0.203716701e3,
            0.703316291e3,
            -0.194349234e4,
            0.110651308e4,
            -0.174378996e3,
            0.188594601e4,
            -0.709371517e4,
            0.922454523e4,
            -0.384508073e4,
            -0.645841789e1,
            0.409703319e2,
            -0.482006560e3,
            0.181870931e4,
            -0.237389204e4,
            0.996703815e3,
            0.361416936e2,
        ],
        dtype=float,
    )
    c = np.array(
        [
            -0.155986211e2,
            -0.512114909e1,
            -0.693003609e2,
            0.203716701e3,
            0.703316291e3,
            -0.194349234e4,
            0.110651308e4,
            -0.220835117e3,
            0.143256989e4,
            -0.318481844e4,
            0.328981513e4,
            -0.135332119e4,
            0.199956489e2,
            -0.127093998e2,
            0.212825156e2,
            -0.275555432e1,
            0.110234982e2,
            0.148881951e3,
            -0.751640284e3,
            0.637876542e3,
            0.127093998e2,
            -0.212825156e2,
            0.275555432e1,
        ],
        dtype=float,
    )
    return b, c


def _dtsub(f10: float, xlst_hr: float, xlat_rad: float, zht_km: float) -> float:
    b, c = _dtsub_coefficients()
    dtc = 0.0
    tx = float(xlst_hr) / 24.0
    ycs = math.cos(float(xlat_rad))
    f = (float(f10) - 100.0) / 100.0

    if 120.0 <= zht_km <= 200.0:
        dtc200 = (
            c[16]
            + c[17] * tx * ycs
            + c[18] * tx**2 * ycs
            + c[19] * tx**3 * ycs
            + c[20] * f * ycs
            + c[21] * tx * f * ycs
            + c[22] * tx**2 * f * ycs
        )
        s = (
            c[0]
            + b[1] * f
            + c[2] * tx * f
            + c[3] * tx**2 * f
            + c[4] * tx**3 * f
            + c[5] * tx**4 * f
            + c[6] * tx**5 * f
            + c[7] * tx * ycs
            + c[8] * tx**2 * ycs
            + c[9] * tx**3 * ycs
            + c[10] * tx**4 * ycs
            + c[11] * tx**5 * ycs
            + c[12] * ycs
            + c[13] * f * ycs
            + c[14] * tx * f * ycs
            + c[15] * tx**2 * f * ycs
        )
        cc = 3.0 * dtc200 - s
        dd = dtc200 - cc
        zp = (zht_km - 120.0) / 80.0
        dtc = cc * zp**2 + dd * zp**3
    elif 200.0 < zht_km <= 240.0:
        h = (zht_km - 200.0) / 50.0
        dtc = (
            c[0] * h
            + b[1] * f * h
            + c[2] * tx * f * h
            + c[3] * tx**2 * f * h
            + c[4] * tx**3 * f * h
            + c[5] * tx**4 * f * h
            + c[6] * tx**5 * f * h
            + c[7] * tx * ycs * h
            + c[8] * tx**2 * ycs * h
            + c[9] * tx**3 * ycs * h
            + c[10] * tx**4 * ycs * h
            + c[11] * tx**5 * ycs * h
            + c[12] * ycs * h
            + c[13] * f * ycs * h
            + c[14] * tx * f * ycs * h
            + c[15] * tx**2 * f * ycs * h
            + c[16]
            + c[17] * tx * ycs
            + c[18] * tx**2 * ycs
            + c[19] * tx**3 * ycs
            + c[20] * f * ycs
            + c[21] * tx * f * ycs
            + c[22] * tx**2 * f * ycs
        )
    elif 240.0 < zht_km <= 300.0:
        h = 40.0 / 50.0
        aa = (
            c[0] * h
            + b[1] * f * h
            + c[2] * tx * f * h
            + c[3] * tx**2 * f * h
            + c[4] * tx**3 * f * h
            + c[5] * tx**4 * f * h
            + c[6] * tx**5 * f * h
            + c[7] * tx * ycs * h
            + c[8] * tx**2 * ycs * h
            + c[9] * tx**3 * ycs * h
            + c[10] * tx**4 * ycs * h
            + c[11] * tx**5 * ycs * h
            + c[12] * ycs * h
            + c[13] * f * ycs * h
            + c[14] * tx * f * ycs * h
            + c[15] * tx**2 * f * ycs * h
            + c[16]
            + c[17] * tx * ycs
            + c[18] * tx**2 * ycs
            + c[19] * tx**3 * ycs
            + c[20] * f * ycs
            + c[21] * tx * f * ycs
            + c[22] * tx**2 * f * ycs
        )
        bb = (
            c[0]
            + b[1] * f
            + c[2] * tx * f
            + c[3] * tx**2 * f
            + c[4] * tx**3 * f
            + c[5] * tx**4 * f
            + c[6] * tx**5 * f
            + c[7] * tx * ycs
            + c[8] * tx**2 * ycs
            + c[9] * tx**3 * ycs
            + c[10] * tx**4 * ycs
            + c[11] * tx**5 * ycs
            + c[12] * ycs
            + c[13] * f * ycs
            + c[14] * tx * f * ycs
            + c[15] * tx**2 * f * ycs
        )
        h = 3.0
        dtc300 = (
            b[0]
            + b[1] * f
            + b[2] * tx * f
            + b[3] * tx**2 * f
            + b[4] * tx**3 * f
            + b[5] * tx**4 * f
            + b[6] * tx**5 * f
            + b[7] * tx * ycs
            + b[8] * tx**2 * ycs
            + b[9] * tx**3 * ycs
            + b[10] * tx**4 * ycs
            + b[11] * tx**5 * ycs
            + b[12] * h * ycs
            + b[13] * tx * h * ycs
            + b[14] * tx**2 * h * ycs
            + b[15] * tx**3 * h * ycs
            + b[16] * tx**4 * h * ycs
            + b[17] * tx**5 * h * ycs
            + b[18] * ycs
        )
        dtc300dz = (
            b[12] * ycs
            + b[13] * tx * ycs
            + b[14] * tx**2 * ycs
            + b[15] * tx**3 * ycs
            + b[16] * tx**4 * ycs
            + b[17] * tx**5 * ycs
        )
        cc = 3.0 * dtc300 - dtc300dz - 3.0 * aa - 2.0 * bb
        dd = dtc300 - aa - bb - cc
        zp = (zht_km - 240.0) / 60.0
        dtc = aa + bb * zp + cc * zp**2 + dd * zp**3
    elif 300.0 < zht_km <= 600.0:
        h = zht_km / 100.0
        dtc = (
            b[0]
            + b[1] * f
            + b[2] * tx * f
            + b[3] * tx**2 * f
            + b[4] * tx**3 * f
            + b[5] * tx**4 * f
            + b[6] * tx**5 * f
            + b[7] * tx * ycs
            + b[8] * tx**2 * ycs
            + b[9] * tx**3 * ycs
            + b[10] * tx**4 * ycs
            + b[11] * tx**5 * ycs
            + b[12] * h * ycs
            + b[13] * tx * h * ycs
            + b[14] * tx**2 * h * ycs
            + b[15] * tx**3 * h * ycs
            + b[16] * tx**4 * h * ycs
            + b[17] * tx**5 * h * ycs
            + b[18] * ycs
        )
    elif 600.0 < zht_km <= 800.0:
        hp = 6.0
        aa = (
            b[0]
            + b[1] * f
            + b[2] * tx * f
            + b[3] * tx**2 * f
            + b[4] * tx**3 * f
            + b[5] * tx**4 * f
            + b[6] * tx**5 * f
            + b[7] * tx * ycs
            + b[8] * tx**2 * ycs
            + b[9] * tx**3 * ycs
            + b[10] * tx**4 * ycs
            + b[11] * tx**5 * ycs
            + b[12] * hp * ycs
            + b[13] * tx * hp * ycs
            + b[14] * tx**2 * hp * ycs
            + b[15] * tx**3 * hp * ycs
            + b[16] * tx**4 * hp * ycs
            + b[17] * tx**5 * hp * ycs
            + b[18] * ycs
        )
        bb = (
            b[12] * ycs
            + b[13] * tx * ycs
            + b[14] * tx**2 * ycs
            + b[15] * tx**3 * ycs
            + b[16] * tx**4 * ycs
            + b[17] * tx**5 * ycs
        )
        cc = -(3.0 * aa + 4.0 * bb) / 4.0
        dd = (aa + bb) / 4.0
        zp = (zht_km - 600.0) / 100.0
        dtc = aa + bb * zp + cc * zp**2 + dd * zp**3
    return float(dtc)


@lru_cache(maxsize=1)
def _semian08_coefficients() -> tuple[np.ndarray, np.ndarray]:
    fzm = np.array([0.2689, -0.1176e-1, 0.2782e-1, -0.2782e-1, 0.3470e-3], dtype=float)
    gtm = np.array(
        [-0.3633, 0.8506e-1, 0.2401, -0.1897, -0.2554, -0.1790e-1, 0.5650e-3, -0.6407e-3, -0.3418e-2, -0.1252e-2],
        dtype=float,
    )
    return fzm, gtm


def _semian08(day: float, ht_km: float, f10b: float, s10b: float, xm10b: float) -> tuple[float, float, float]:
    twopi = 2.0 * math.pi
    fzm, gtm = _semian08_coefficients()
    fsmb = float(f10b) - 0.70 * float(s10b) - 0.04 * float(xm10b)
    htz = float(ht_km) / 1000.0
    fzz = fzm[0] + fzm[1] * fsmb + fzm[2] * fsmb * htz + fzm[3] * fsmb * htz**2 + fzm[4] * fsmb**2 * htz
    fsmb = float(f10b) - 0.75 * float(s10b) - 0.37 * float(xm10b)
    tau = (float(day) - 1.0) / 365.0
    sin1p = math.sin(twopi * tau)
    cos1p = math.cos(twopi * tau)
    sin2p = math.sin(2.0 * twopi * tau)
    cos2p = math.cos(2.0 * twopi * tau)
    gtz = (
        gtm[0]
        + gtm[1] * sin1p
        + gtm[2] * cos1p
        + gtm[3] * sin2p
        + gtm[4] * cos2p
        + gtm[5] * fsmb
        + gtm[6] * fsmb * sin1p
        + gtm[7] * fsmb * cos1p
        + gtm[8] * fsmb * sin2p
        + gtm[9] * fsmb * cos2p
    )
    if fzz < 1e-6:
        fzz = 1e-6
    return float(fzz), float(gtz), float(fzz * gtz)


@lru_cache(maxsize=1)
def _semian06_coefficients() -> tuple[np.ndarray, np.ndarray]:
    fzm = np.array([0.111613, -0.159000e-2, 0.126190e-1, -0.100064e-1, -0.237509e-4, 0.260759e-4])
    gtm = np.array(
        [
            -0.833646,
            -0.265450,
            0.467603,
            -0.299906,
            -0.105451,
            -0.165537e-1,
            -0.380037e-1,
            -0.150991e-1,
            -0.541280e-1,
            0.119554e-1,
            0.437544e-2,
            -0.369016e-2,
            0.206763e-2,
            -0.142888e-2,
            -0.867124e-5,
            0.189032e-4,
            0.156988e-3,
            0.491286e-3,
            -0.391484e-4,
            -0.126854e-4,
            0.134078e-4,
            -0.614176e-5,
            0.343423e-5,
        ],
        dtype=float,
    )
    return fzm, gtm


def _semian06(day: float, ht_km: float, f10bar: float) -> tuple[float, float, float]:
    twopi = 2.0 * math.pi
    fzm, gtm = _semian06_coefficients()
    f10b = float(f10bar)
    fb2 = f10b * f10b
    htz = float(ht_km) / 1000.0
    fzz = fzm[0] + fzm[1] * f10b + fzm[2] * f10b * htz + fzm[3] * f10b * htz**2
    fzz += fzm[4] * f10b**2 * htz + fzm[5] * f10b**2 * htz**2

    tau = (float(day) - 1.0) / 365.0
    sin1p = math.sin(twopi * tau)
    cos1p = math.cos(twopi * tau)
    sin2p = math.sin(2.0 * twopi * tau)
    cos2p = math.cos(2.0 * twopi * tau)
    sin3p = math.sin(3.0 * twopi * tau)
    cos3p = math.cos(3.0 * twopi * tau)
    sin4p = math.sin(4.0 * twopi * tau)
    cos4p = math.cos(4.0 * twopi * tau)

    gtz = (
        gtm[0]
        + gtm[1] * sin1p
        + gtm[2] * cos1p
        + gtm[3] * sin2p
        + gtm[4] * cos2p
        + gtm[5] * sin3p
        + gtm[6] * cos3p
        + gtm[7] * sin4p
        + gtm[8] * cos4p
        + gtm[9] * f10b
        + gtm[10] * f10b * sin1p
        + gtm[11] * f10b * cos1p
        + gtm[12] * f10b * sin2p
        + gtm[13] * f10b * cos2p
        + gtm[14] * f10b * sin3p
        + gtm[15] * f10b * cos3p
        + gtm[16] * f10b * sin4p
        + gtm[17] * f10b * cos4p
        + gtm[18] * fb2
        + gtm[19] * fb2 * sin1p
        + gtm[20] * fb2 * cos1p
        + gtm[21] * fb2 * sin2p
        + gtm[22] * fb2 * cos2p
    )
    if fzz < 1e-6:
        fzz = 1e-6
    return float(fzz), float(gtz), float(fzz * gtz)


def jb2006_density(alt_km: float, lat_deg: float, lon_deg: float, dt_utc: datetime, env: dict | None = None) -> float:
    env = {} if env is None else env
    zht = _validate_jb_altitude(alt_km, "JB2006")
    if dt_utc.tzinfo is None:
        dt_utc = dt_utc.replace(tzinfo=timezone.utc)
    dt_utc = dt_utc.astimezone(timezone.utc)
    jd_utc = datetime_to_julian_date(dt_utc)
    mjd_utc = jd_utc - 2400000.5

    sol_path = _resolve_table_path(
        env.get("jb2006_sol_path", env.get("jb2008_sol_path")), _default_jb2008_sol_path()
    )
    ap_path = _resolve_table_path(env.get("jb2006_ap_path"), _default_jb2006_ap_path())
    sol = _load_soldata(sol_path)
    apdata = _load_apdata(ap_path)

    jd_floor = math.floor(mjd_utc + 2400000.5)
    i = _sol_day_index(sol_path, jd_floor)
    if i < 0:
        raise RuntimeError(f"JB2006 SOL data missing day {jd_floor}")
    f10b = float(sol[4, i])
    s10b = float(sol[6, i])
    xm10 = float(sol[7, i])
    f10 = float(sol[3, max(i - 1, 0)])
    s10 = float(sol[5, max(i - 1, 0)])
    xm10b = float(sol[8, max(i - 5, 0)])

    ap_dt = dt_utc - timedelta(hours=6.7)
    ap_doy = _day_of_year(ap_dt)
    ap_doy_floor = math.floor(ap_doy)
    i_ap = _ap_day_index(ap_path, ap_dt.year, ap_doy_floor)
    if i_ap < 0:
        raise RuntimeError(f"JB2006 AP data missing day {ap_dt.year}-{ap_doy_floor}")
    ap_slot = max(0, min(int(math.floor(ap_dt.hour / 3.0)), 7))
    ap = float(apdata[4 + ap_slot, i_ap])

    r_sat_eci_km = _position_eci_from_geodetic(float(lat_deg), float(lon_deg), float(alt_km), jd_utc, env)
    ra_sat = math.atan2(float(r_sat_eci_km[1]), float(r_sat_eci_km[0])) % (2.0 * math.pi)
    latgc = _gd2gc_rad(math.radians(float(lat_deg)))

    r_sun = _sun_position_from_env(jd_utc, env)
    ra_sun = math.atan2(float(r_sun[1]), float(r_sun[0])) % (2.0 * math.pi)
    dec_sun = math.atan2(float(r_sun[2]), math.hypot(float(r_sun[0]), float(r_sun[1])))
    alpha = _JB_ALPHA
    al10 = _AL10
    amw = _JB_AMW
    avogad = _AVOGAD
    twopi = _TWOPI
    piov2 = _PIOV2
    frac = _JB_FRAC
    rstar = _RSTAR
    r2, r3 = 0.025, 0.075
    cht = _JB_CHT

    tsubc = 379.0 + 3.353 * f10b + 0.358 * (f10 - f10b) + 2.094 * (s10 - s10b) + 0.343 * (xm10 - xm10b)

    eta = 0.5 * abs(latgc - dec_sun)
    theta = 0.5 * abs(latgc + dec_sun)
    h = ra_sat - ra_sun
    tau = h - 0.64577182 + 0.10471976 * math.sin(h + 0.75049158)
    glat = latgc
    glst = h + math.pi
    glsthr = (glst * 24.0 / twopi) % 24.0
    c = math.cos(eta) ** 2.5
    s = math.sin(theta) ** 2.5
    df = s + (c - s) * abs(math.cos(0.5 * tau)) ** 3
    tsubl = tsubc * (1.0 + 0.31 * df)
    dtg = ap + 100.0 * (1.0 - math.exp(-0.08 * ap))
    dtclst = _dtsub(f10, glsthr, glat, zht)
    tinf = tsubl + dtg + dtclst
    tsubx = 444.3807 + 0.02385 * tinf - 392.8292 * math.exp(-0.0021357 * tinf)
    gsubx = 0.054285714 * (tsubx - 183.0)
    tc2 = (tinf - tsubx) / piov2
    tc = (tsubx, gsubx, tc2, gsubx / tc2)

    zend, z, tloc1, tloc2, ambar1, ambar2, gravl, ain, sum2 = _integrate_lower_atmosphere(zht, tc, env)

    fact1 = 1000.0 / rstar
    rho = 3.46e-6 * ambar2 * tloc1 * math.exp(-fact1 * sum2) / ambar1 / tloc2
    anm = avogad * rho
    an = anm / ambar2
    fact2 = anm / 28.960
    aln = [0.0] * 6
    aln[0] = math.log(frac[0] * fact2)
    aln[3] = math.log(frac[2] * fact2)
    aln[4] = math.log(frac[3] * fact2)
    aln[1] = math.log(fact2 * (1.0 + frac[1]) - an)
    aln[2] = math.log(2.0 * (an - fact2))

    doy = _day_of_year(dt_utc)

    def _apply_corrections(
        rho_in: float, zht_in: float, sat_lat_rad: float, aln_in: list[float], temp2: float, z_for_check: float
    ) -> float:
        trash = (mjd_utc - 36204.0) / 365.2422
        capphi = trash % 1.0
        dlrsl = (
            0.02
            * (zht_in - 90.0)
            * math.exp(-0.045 * (zht_in - 90.0))
            * math.copysign(1.0, sat_lat_rad)
            * math.sin(twopi * capphi + 1.72)
            * math.sin(sat_lat_rad) ** 2
        )
        dlrsa = 0.0
        if z_for_check < 2000.0:
            fzz, _, dlrsa = _semian06(doy, zht_in, f10b)
            if fzz < 0.0:
                dlrsa = 0.0
        dlr = al10 * (dlrsl + dlrsa)
        sumnm = 0.0
        for idx in range(6):
            an_local = math.exp(aln_in[idx] + dlr)
            sumnm += an_local * amw[idx]
        rho_out = sumnm / avogad
        fex = 1.0
        if 1000.0 <= zht_in < 1500.0:
            zeta = (zht_in - 1000.0) * 0.002
            zeta2 = zeta * zeta
            zeta3 = zeta * zeta2
            f15c = cht[0] + cht[1] * f10b + cht[2] * 1500.0 + cht[3] * f10b * 1500.0
            f15c_zeta = (cht[2] + cht[3] * f10b) * 500.0
            fex2 = 3.0 * f15c - f15c_zeta - 3.0
            fex3 = f15c_zeta - 2.0 * f15c + 2.0
            fex = 1.0 + fex2 * zeta2 + fex3 * zeta3
        elif zht_in >= 1500.0:
            fex = cht[0] + cht[1] * f10b + cht[2] * zht_in + cht[3] * f10b * zht_in
        return float(fex * rho_out)

    if zht <= 105.0:
        aln[5] = aln[4] - 25.0
        return _apply_corrections(rho, zht, latgc, aln, tloc2, zht)

    z3 = min(zht, 500.0)
    ain = gravl / tloc2
    zend, z, tloc3, gravl, ain, sum2b = _integrate_upper_atmosphere(zend, z, z3, r2, ain, tc, env)

    z4 = max(zht, 500.0)
    r_step = r3 if zht > 500.0 else r2
    zend, z, tloc4, gravl, ain, sum3 = _integrate_upper_atmosphere(zend, z, z4, r_step, ain, tc, env)

    if zht > 500.0:
        temp2 = tloc4
        altr = math.log(tloc4 / tloc2)
        fact2 = fact1 * (sum2b + sum3)
        hsign = -1.0
    else:
        temp2 = tloc3
        altr = math.log(tloc3 / tloc2)
        fact2 = fact1 * sum2b
        hsign = 1.0

    for idx in range(5):
        aln[idx] = aln[idx] - (1.0 + alpha[idx]) * altr - fact2 * amw[idx]
    al10t5 = math.log10(tinf)
    alnh5 = (5.5 * al10t5 - 39.40) * al10t5 + 73.13
    aln[5] = al10 * (alnh5 + 6.0) + hsign * (math.log(tloc4 / tloc3) + fact1 * sum3 * amw[5])
    return _apply_corrections(rho, zht, latgc, aln, temp2, z)


def jb2008_density(alt_km: float, lat_deg: float, lon_deg: float, dt_utc: datetime, env: dict | None = None) -> float:
    env = {} if env is None else env
    zht = _validate_jb_altitude(alt_km, "JB2008")
    if dt_utc.tzinfo is None:
        dt_utc = dt_utc.replace(tzinfo=timezone.utc)
    dt_utc = dt_utc.astimezone(timezone.utc)
    jd_utc = datetime_to_julian_date(dt_utc)
    mjd_utc = jd_utc - 2400000.5

    sol_path = _resolve_table_path(env.get("jb2008_sol_path"), _default_jb2008_sol_path())
    dtc_path = _resolve_table_path(env.get("jb2008_dtc_path"), _default_jb2008_dtc_path())
    sol = _load_soldata(sol_path)
    dtc = _load_dtcdata(dtc_path)

    jd_floor = math.floor(mjd_utc - 1.0 + 2400000.5)
    i = _sol_day_index(sol_path, jd_floor)
    if i < 0:
        raise RuntimeError(f"JB2008 SOL data missing day {jd_floor}")
    f10, f10b, s10, s10b = [float(v) for v in sol[3:7, i]]
    xm10, xm10b = [float(v) for v in sol[7:9, max(i - 1, 0)]]
    y10, y10b = [float(v) for v in sol[9:11, max(i - 4, 0)]]

    doy = _day_of_year(dt_utc)
    doy_floor = math.floor(doy)
    i_dtc = _dtc_day_index(dtc_path, dt_utc.year, doy_floor)
    if i_dtc < 0:
        raise RuntimeError(f"JB2008 DTC data missing day {dt_utc.year}-{doy_floor}")
    hour_slot = int(math.floor(dt_utc.hour)) + 3
    hour_slot = max(2, min(hour_slot, dtc.shape[0] - 1))
    dstdtc = float(dtc[hour_slot, i_dtc])

    r_sat_eci_km = _position_eci_from_geodetic(float(lat_deg), float(lon_deg), float(alt_km), jd_utc, env)
    ra_sat = math.atan2(float(r_sat_eci_km[1]), float(r_sat_eci_km[0])) % (2.0 * math.pi)
    latgc = _gd2gc_rad(math.radians(float(lat_deg)))

    r_sun = _sun_position_from_env(jd_utc, env)
    ra_sun = math.atan2(float(r_sun[1]), float(r_sun[0])) % (2.0 * math.pi)
    dec_sun = math.atan2(float(r_sun[2]), math.hypot(float(r_sun[0]), float(r_sun[1])))
    alpha = _JB_ALPHA
    al10 = _AL10
    amw = _JB_AMW
    avogad = _AVOGAD
    twopi = _TWOPI
    piov2 = _PIOV2
    frac = _JB_FRAC
    rstar = _RSTAR
    r2, r3 = 0.025, 0.075
    cht = _JB_CHT

    fn = min((f10b / 240.0) ** 0.25, 1.0)
    fsb = f10b * fn + s10b * (1.0 - fn)
    tsubc = (
        392.4
        + 3.227 * fsb
        + 0.298 * (f10 - f10b)
        + 2.259 * (s10 - s10b)
        + 0.312 * (xm10 - xm10b)
        + 0.178 * (y10 - y10b)
    )

    eta = 0.5 * abs(latgc - dec_sun)
    theta = 0.5 * abs(latgc + dec_sun)
    h = ra_sat - ra_sun
    tau = h - 0.64577182 + 0.10471976 * math.sin(h + 0.75049158)
    glat = latgc
    glst = h + math.pi
    glsthr = (glst * 24.0 / twopi) % 24.0
    c = math.cos(eta) ** 2.5
    s = math.sin(theta) ** 2.5
    df = s + (c - s) * abs(math.cos(0.5 * tau)) ** 3
    tsubl = tsubc * (1.0 + 0.31 * df)
    dtclst = _dtsub(f10, glsthr, glat, zht)
    tinf = tsubl + dstdtc + dtclst
    tsubx = 444.3807 + 0.02385 * tinf - 392.8292 * math.exp(-0.0021357 * tinf)
    gsubx = 0.054285714 * (tsubx - 183.0)
    tc2 = (tinf - tsubx) / piov2
    tc = (tsubx, gsubx, tc2, gsubx / tc2)

    zend, z, tloc1, tloc2, ambar1, ambar2, gravl, ain, sum2 = _integrate_lower_atmosphere(zht, tc, env)

    fact1 = 1000.0 / rstar
    rho = 3.46e-6 * ambar2 * tloc1 * math.exp(-fact1 * sum2) / ambar1 / tloc2
    anm = avogad * rho
    an = anm / ambar2
    fact2 = anm / 28.960
    aln = [0.0] * 6
    aln[0] = math.log(frac[0] * fact2)
    aln[3] = math.log(frac[2] * fact2)
    aln[4] = math.log(frac[3] * fact2)
    aln[1] = math.log(fact2 * (1.0 + frac[1]) - an)
    aln[2] = math.log(2.0 * (an - fact2))

    def _apply_corrections(
        rho_in: float, zht_in: float, sat_lat_rad: float, aln_in: list[float], temp2: float, z_for_check: float
    ) -> float:
        trash = (mjd_utc - 36204.0) / 365.2422
        capphi = trash % 1.0
        dlrsl = (
            0.02
            * (zht_in - 90.0)
            * math.exp(-0.045 * (zht_in - 90.0))
            * math.copysign(1.0, sat_lat_rad)
            * math.sin(twopi * capphi + 1.72)
            * math.sin(sat_lat_rad) ** 2
        )
        dlrsa = 0.0
        if z_for_check < 2000.0:
            fzz, _, dlrsa = _semian08(doy, zht_in, f10b, s10b, xm10b)
            if fzz < 0.0:
                dlrsa = 0.0
        dlr = al10 * (dlrsl + dlrsa)
        sumnm = 0.0
        for idx in range(6):
            an_local = math.exp(aln_in[idx] + dlr)
            sumnm += an_local * amw[idx]
        rho_out = sumnm / avogad
        fex = 1.0
        if 1000.0 <= zht_in < 1500.0:
            zeta = (zht_in - 1000.0) * 0.002
            zeta2 = zeta * zeta
            zeta3 = zeta * zeta2
            f15c = cht[0] + cht[1] * f10b + cht[2] * 1500.0 + cht[3] * f10b * 1500.0
            f15c_zeta = (cht[2] + cht[3] * f10b) * 500.0
            fex2 = 3.0 * f15c - f15c_zeta - 3.0
            fex3 = f15c_zeta - 2.0 * f15c + 2.0
            fex = 1.0 + fex2 * zeta2 + fex3 * zeta3
        elif zht_in >= 1500.0:
            fex = cht[0] + cht[1] * f10b + cht[2] * zht_in + cht[3] * f10b * zht_in
        return float(fex * rho_out)

    if zht <= 105.0:
        aln[5] = aln[4] - 25.0
        return _apply_corrections(rho, zht, latgc, aln, tloc2, zht)

    z3 = min(zht, 500.0)
    ain = gravl / tloc2
    zend, z, tloc3, gravl, ain, sum2b = _integrate_upper_atmosphere(zend, z, z3, r2, ain, tc, env)

    z4 = max(zht, 500.0)
    r_step = r3 if zht > 500.0 else r2
    zend, z, tloc4, gravl, ain, sum3 = _integrate_upper_atmosphere(zend, z, z4, r_step, ain, tc, env)

    if zht > 500.0:
        temp2 = tloc4
        altr = math.log(tloc4 / tloc2)
        fact2 = fact1 * (sum2b + sum3)
        hsign = -1.0
    else:
        temp2 = tloc3
        altr = math.log(tloc3 / tloc2)
        fact2 = fact1 * sum2b
        hsign = 1.0

    for idx in range(5):
        aln[idx] = aln[idx] - (1.0 + alpha[idx]) * altr - fact2 * amw[idx]
    al10t5 = math.log10(tinf)
    alnh5 = (5.5 * al10t5 - 39.40) * al10t5 + 73.13
    aln[5] = al10 * (alnh5 + 6.0) + hsign * (math.log(tloc4 / tloc3) + fact1 * sum3 * amw[5])
    return _apply_corrections(rho, zht, latgc, aln, temp2, z)
