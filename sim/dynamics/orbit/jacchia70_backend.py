from __future__ import annotations

import math
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np

from sim.acceleration.settings import acceleration_enabled_from_mode
from sim.dynamics.orbit.epoch import datetime_to_julian_date, sun_position_eci_km_enhanced, sun_position_eci_km_simple
from sim.dynamics.orbit.frames import (
    FRAME_MODEL_IAU76_80_EOP,
    apparent_sidereal_time_hpop_like,
    eci_to_ecef_rotation_hpop_like,
    normalize_frame_model,
)
from sim.utils.geodesy import geodetic_to_ecef_km

_BDATA = np.array([28.15204, -0.085586, 0.0001284, -0.000010056, -0.00001021, 0.0000015044, 0.000000099826])
_ALPHA = np.array([0.0, 0.0, 0.0, 0.0, -0.38, 0.0])
_EI = np.array([28.0134, 31.9988, 15.9994, 39.948, 4.0026, 1.00797])
_ALTMIN = np.array([90.0, 105.0, 125.0, 160.0, 200.0, 300.0, 500.0, 1500.0, 2500.0])
_NG = np.array([4, 5, 6, 6, 6, 6, 6, 6])
_CZ = np.array([1.0, 0.9045085, 0.6545085, 0.3454915, 0.0954915, 0.0])
_GAUSS_W = {
    3: np.array([0.5555556, 0.8888889, 0.5555556]),
    4: np.array([0.3478548, 0.6521452, 0.6521452, 0.3478548]),
    5: np.array([0.2369269, 0.4786287, 0.5688889, 0.4786287, 0.2369269]),
    6: np.array([0.1713245, 0.3607616, 0.4679139, 0.4679139, 0.3607616, 0.1713245]),
    7: np.array([0.1294850, 0.2797054, 0.3818301, 0.4179592, 0.3818301, 0.2797054, 0.1294850]),
    8: np.array([0.1012285, 0.2223810, 0.3137067, 0.3626838, 0.3626838, 0.3137067, 0.2223810, 0.1012285]),
}
_GAUSS_X = {
    3: np.array([-0.7745967, 0.0, 0.7745967]),
    4: np.array([-0.8611363, -0.3399810, 0.3399810, 0.8611363]),
    5: np.array([-0.9061798, -0.5384693, 0.0, 0.5384693, 0.9061798]),
    6: np.array([-0.9324695, -0.6612094, -0.2386192, 0.2386192, 0.6612094, 0.9324695]),
    7: np.array([-0.9491079, -0.7415312, -0.4058452, 0.0, 0.4058452, 0.7415312, 0.9491079]),
    8: np.array([-0.9602899, -0.7966665, -0.5255324, -0.1834346, 0.1834346, 0.5255324, 0.7966665, 0.9602899]),
}
_GAUSS_PAIRS = {
    order: tuple((float(weight), float(abscissa)) for weight, abscissa in zip(_GAUSS_W[order], _GAUSS_X[order]))
    for order in _GAUSS_W
}
_GAUSS_W_TABLE = np.zeros((9, 8), dtype=float)
_GAUSS_X_TABLE = np.zeros((9, 8), dtype=float)
for _order in _GAUSS_W:
    _GAUSS_W_TABLE[_order, :_order] = _GAUSS_W[_order]
    _GAUSS_X_TABLE[_order, :_order] = _GAUSS_X[_order]
_BDATA_FLOATS = tuple(float(coeff) for coeff in _BDATA)


@lru_cache(maxsize=4)
def _load_swdata(path: str) -> np.ndarray:
    arr = np.loadtxt(str(Path(path).expanduser().resolve()), dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 30:
        raise ValueError(f"Unexpected SW-All shape {arr.shape} from {path}")
    arr.setflags(write=False)
    return arr


@lru_cache(maxsize=4096)
def _sw_day_index(path: str, year: int, month: int, day: int) -> int:
    sw = _load_swdata(path)
    matches = np.where((sw[:, 0] == year) & (sw[:, 1] == month) & (sw[:, 2] == day))[0]
    return -1 if matches.size == 0 else int(matches[0])


@lru_cache(maxsize=8)
def _resolve_table_path(raw: str) -> str:
    p = Path(str(raw)).expanduser()
    if not p.is_absolute():
        p = Path(__file__).resolve().parents[3] / p
    return str(p.resolve())


def _day_of_year(dt_utc: datetime) -> float:
    start = datetime(dt_utc.year, 1, 1, tzinfo=timezone.utc)
    return 1.0 + (dt_utc - start).total_seconds() / 86400.0


@lru_cache(maxsize=4096)
def _time_inputs(dt_utc: datetime) -> tuple[float, float]:
    return datetime_to_julian_date(dt_utc), _day_of_year(dt_utc)


def _solar_geomagnetic_inputs(dt_utc: datetime, env: dict) -> tuple[float, float, float, int]:
    i1 = int(env.get("jacchia70_geomagnetic_index_type", 2))
    f10_direct = env.get("jacchia70_f10", env.get("f107"))
    f10b_direct = env.get("jacchia70_f10b", env.get("f107a"))
    gi_direct = env.get("jacchia70_ap" if i1 == 2 else "jacchia70_kp", env.get("ap" if i1 == 2 else "kp"))

    sw_path = env.get("jacchia70_sw_path")
    if sw_path not in (None, "") and (f10_direct is None or f10b_direct is None or gi_direct is None):
        resolved_sw_path = _resolve_table_path(str(sw_path))
        sw = _load_swdata(resolved_sw_path)
        i = _sw_day_index(resolved_sw_path, dt_utc.year, dt_utc.month, dt_utc.day)
        if i < 0:
            raise RuntimeError(f"Jacchia-70 SW data missing day {dt_utc.year}-{dt_utc.month}-{dt_utc.day}")
        slot = max(0, min(int(math.floor(dt_utc.hour / 3.0)), 7))
        if gi_direct is None:
            gi_direct = sw[i, 14 + slot] if i1 == 2 else sw[i, 5 + slot]
        if f10_direct is None:
            f10_direct = sw[max(i - 1, 0), 29]
        if f10b_direct is None:
            f10b_direct = 0.5 * (sw[i, 27] + sw[max(i - 81, 0), 27])

    f10 = float(150.0 if f10_direct is None else f10_direct)
    f10b = float(f10 if f10b_direct is None else f10b_direct)
    gi = float(4.0 if gi_direct is None else gi_direct)
    return f10, f10b, gi, i1


def _jmweight(alt_km: float) -> float:
    if alt_km > 105.0:
        return 1.0
    u = float(alt_km) - 100.0
    return float(sum(float(coeff) * u**idx for idx, coeff in enumerate(_BDATA)))


def _jtemp(alt_km: float, tx: float, t1: float, t3: float, t4: float, a2: float) -> float:
    u = float(alt_km) - 125.0
    if u > 0.0:
        return float(tx + a2 * math.atan(t1 * u * (1.0 + 4.5e-6 * u**2.5) / a2))
    return float(tx + t1 * u + t3 * u**3 + t4 * u**4)


def _jgauss_python(z1: float, z2: float, nmin: int, tx: float, t1: float, t3: float, t4: float, a2: float) -> float:
    if z2 <= z1:
        return 0.0
    total = 0.0
    for k in range(max(0, int(nmin) - 1), 8):
        ngauss = int(_NG[k])
        a = float(_ALTMIN[k])
        d = min(float(z2), float(_ALTMIN[k + 1]))
        if d <= a:
            continue
        half_width = 0.5 * (d - a)
        rr = 0.0
        for weight, abscissa in _GAUSS_PAIRS[ngauss]:
            z = half_width * (abscissa + 1.0) + a
            if z > 105.0:
                molecular_weight = 1.0
            else:
                u = z - 100.0
                molecular_weight = 0.0
                for idx, coeff in enumerate(_BDATA_FLOATS):
                    molecular_weight += coeff * u**idx
            gravity = 9.80665 / (1.0 + z / 6356.766) ** 2
            u = z - 125.0
            if u > 0.0:
                temperature = tx + a2 * math.atan(t1 * u * (1.0 + 4.5e-6 * u**2.5) / a2)
            else:
                temperature = tx + t1 * u + t3 * u**3 + t4 * u**4
            rr += weight * molecular_weight * gravity / temperature
        total += half_width * rr
        if d == z2:
            break
    return float(total)


def _jgauss_kernel(
    z1: float, z2: float, nmin: int, tx: float, t1: float, t3: float, t4: float, a2: float
) -> float:
    if z2 <= z1:
        return 0.0
    total = 0.0
    for k in range(max(0, nmin - 1), 8):
        ngauss = _NG[k]
        a = _ALTMIN[k]
        d = min(z2, _ALTMIN[k + 1])
        if d <= a:
            continue
        half_width = 0.5 * (d - a)
        rr = 0.0
        for node in range(ngauss):
            weight = _GAUSS_W_TABLE[ngauss, node]
            abscissa = _GAUSS_X_TABLE[ngauss, node]
            z = half_width * (abscissa + 1.0) + a
            if z > 105.0:
                molecular_weight = 1.0
            else:
                u = z - 100.0
                molecular_weight = 0.0
                for idx in range(_BDATA.size):
                    molecular_weight += _BDATA[idx] * math.pow(u, float(idx))
            gravity = 9.80665 / math.pow(1.0 + z / 6356.766, 2.0)
            u = z - 125.0
            if u > 0.0:
                temperature = tx + a2 * math.atan(t1 * u * (1.0 + 4.5e-6 * math.pow(u, 2.5)) / a2)
            else:
                temperature = tx + t1 * u + t3 * math.pow(u, 3.0) + t4 * math.pow(u, 4.0)
            rr += weight * molecular_weight * gravity / temperature
        total += half_width * rr
        if d == z2:
            break
    return total


_JGAUSS_MID_ALTITUDE = _jgauss_kernel


@lru_cache(maxsize=1)
def _compiled_jgauss():
    from numba import njit

    return njit(cache=True, fastmath=False)(_jgauss_kernel)


def _jacchia_python(z_km: float, tinf_k: float) -> tuple[float, float, float, float, np.ndarray]:
    av = 6.02257e23
    qn = 0.7811
    qo2 = 0.20955
    qa = 9.343e-3
    qhe = 0.00001289
    rgas = 8.31432
    t0 = 183.0

    tx = 444.3807 + 0.02385 * tinf_k - 392.8292 * math.exp(-0.0021357 * tinf_k)
    a2 = 2.0 * (tinf_k - tx) / math.pi
    txt0 = tx - t0
    t1 = 1.9 * txt0 / 35.0
    t3 = -1.7 * txt0 / 35.0**3
    t4 = -0.8 * txt0 / 35.0**4
    tz = _jtemp(z_km, tx, t1, t3, t4, a2)

    d = min(z_km, 105.0)
    r = _jgauss_python(90.0, d, 1, tx, t1, t3, t4, a2)
    em = _jmweight(d)
    td = _jtemp(d, tx, t1, t3, t4, a2)
    dens = 2.1926e-8 * em * math.exp(-r / rgas) / td
    factor = av * dens
    par = factor / em
    factor = factor / 28.96

    if z_km <= 105.0:
        a = np.zeros(6, dtype=float)
        a[0] = math.log10(max(qn * factor, 1e-300))
        a[3] = math.log10(max(qa * factor, 1e-300))
        a[4] = math.log10(max(qhe * factor, 1e-300))
        a[2] = math.log10(max(2.0 * par * (1.0 - em / 28.96), 1e-300))
        a[1] = math.log10(max(par * (em * (1.0 + qo2) / 28.96 - 1.0), 1e-300))
        a[5] = 0.0
        return float(dens), math.log10(max(dens, 1e-300)), float(em), float(tz), a

    di = np.zeros(6, dtype=float)
    di[0] = qn * factor
    di[1] = par * (em * (1.0 + qo2) / 28.96 - 1.0)
    di[2] = 2.0 * par * (1.0 - em / 28.96)
    di[3] = qa * factor
    di[4] = qhe * factor

    r = _jgauss_python(d, z_km, 2, tx, t1, t3, t4, a2)
    dit = np.ones(6, dtype=float)
    for idx in range(5):
        dit[idx] = di[idx] * (td / tz) ** (1.0 + _ALPHA[idx]) * math.exp(-_EI[idx] * r / rgas)
        if dit[idx] <= 0.0:
            dit[idx] = 1e-6

    if z_km > 500.0:
        s = _jtemp(500.0, tx, t1, t3, t4, a2)
        di[5] = 10.0 ** (73.13 - 39.4 * math.log10(s) + 5.5 * math.log10(s) ** 2)
        r = _jgauss_python(500.0, z_km, 7, tx, t1, t3, t4, a2)
        dit[5] = di[5] * (s / tz) * math.exp(-_EI[5] * r / rgas)

    dens = float(np.sum(_EI * dit) / av)
    em = float(dens * av / np.sum(dit))
    a = np.log10(np.maximum(dit, 1e-300))
    return dens, math.log10(max(dens, 1e-300)), em, float(tz), a


def _jacchia_mid_altitude_log_kernel(z_km: float, tinf_k: float) -> float:
    """Return log10 density for the 105-440 km branch without array outputs."""
    av = 6.02257e23
    rgas = 8.31432
    tx = 444.3807 + 0.02385 * tinf_k - 392.8292 * math.exp(-0.0021357 * tinf_k)
    a2 = 2.0 * (tinf_k - tx) / math.pi
    txt0 = tx - 183.0
    t1 = 1.9 * txt0 / 35.0
    t3 = -1.7 * txt0 / math.pow(35.0, 3.0)
    t4 = -0.8 * txt0 / math.pow(35.0, 4.0)

    u = z_km - 125.0
    if u > 0.0:
        tz = tx + a2 * math.atan(t1 * u * (1.0 + 4.5e-6 * math.pow(u, 2.5)) / a2)
    else:
        tz = tx + t1 * u + t3 * math.pow(u, 3.0) + t4 * math.pow(u, 4.0)

    r = _JGAUSS_MID_ALTITUDE(90.0, 105.0, 1, tx, t1, t3, t4, a2)
    em = 0.0
    for idx in range(_BDATA.size):
        em += _BDATA[idx] * math.pow(5.0, float(idx))
    ud = -20.0
    td = tx + t1 * ud + t3 * math.pow(ud, 3.0) + t4 * math.pow(ud, 4.0)
    dens = 2.1926e-8 * em * math.exp(-r / rgas) / td
    factor = av * dens
    par = factor / em
    factor = factor / 28.96

    di0 = 0.7811 * factor
    di1 = par * (em * (1.0 + 0.20955) / 28.96 - 1.0)
    di2 = 2.0 * par * (1.0 - em / 28.96)
    di3 = 9.343e-3 * factor
    di4 = 0.00001289 * factor
    r = _JGAUSS_MID_ALTITUDE(105.0, z_km, 2, tx, t1, t3, t4, a2)
    td_tz = td / tz
    dit0 = di0 * td_tz * math.exp(-_EI[0] * r / rgas)
    dit1 = di1 * td_tz * math.exp(-_EI[1] * r / rgas)
    dit2 = di2 * td_tz * math.exp(-_EI[2] * r / rgas)
    dit3 = di3 * td_tz * math.exp(-_EI[3] * r / rgas)
    dit4 = di4 * math.pow(td_tz, 1.0 + _ALPHA[4]) * math.exp(-_EI[4] * r / rgas)
    if dit0 <= 0.0:
        dit0 = 1e-6
    if dit1 <= 0.0:
        dit1 = 1e-6
    if dit2 <= 0.0:
        dit2 = 1e-6
    if dit3 <= 0.0:
        dit3 = 1e-6
    if dit4 <= 0.0:
        dit4 = 1e-6

    mass_density = 0.0
    mass_density += _EI[0] * dit0
    mass_density += _EI[1] * dit1
    mass_density += _EI[2] * dit2
    mass_density += _EI[3] * dit3
    mass_density += _EI[4] * dit4
    mass_density += _EI[5]
    mass_density /= av
    return math.log10(max(mass_density, 1e-300))


@lru_cache(maxsize=1)
def _compiled_jacchia_mid_altitude_log():
    global _JGAUSS_MID_ALTITUDE
    from numba import njit

    _JGAUSS_MID_ALTITUDE = _compiled_jgauss()
    return njit(cache=True, fastmath=False)(_jacchia_mid_altitude_log_kernel)


def _accelerated_jacchia_mid_altitude_log(z_km: float, tinf_k: float) -> float | None:
    if not acceleration_enabled_from_mode():
        return None
    return float(_compiled_jacchia_mid_altitude_log()(z_km, tinf_k))


def _jtinf(f10: float, f10b: float, gi: float, xlat: float, sda: float, sha: float, dy: float, i1: int) -> float:
    tc = 383.0 + 3.32 * f10b + 1.8 * (f10 - f10b)
    eta = 0.5 * abs(xlat - sda)
    theta = 0.5 * abs(xlat + sda)
    tau = sha - 0.6457718 + 0.1047198 * math.sin(sha + 0.7504916)
    if tau > math.pi:
        tau -= 2.0 * math.pi
    if tau < -math.pi:
        tau += 2.0 * math.pi

    a1 = math.sin(theta) ** 2.5
    a2 = math.cos(eta) ** 2.5
    a3 = math.cos(tau / 2.0) ** 3
    b1 = 1.0 + 0.31 * a1
    tv = b1 * (1.0 + 0.31 * ((a2 - a1) / b1) * a3)
    tl = tc * tv

    tg = 28.0 * gi + 0.03 * math.exp(gi) if i1 == 1 else gi + 100.0 * (1.0 - math.exp(-0.08 * gi))
    g3 = 0.5 * (1.0 + math.sin(2.0 * math.pi * dy + 5.974262))
    tau1 = dy + 0.1145 * (g3**2.16 - 0.5)
    g1 = 0.349 + 0.206 * math.sin(2.0 * math.pi * tau1 + 3.9531708)
    g2 = math.sin(4.0 * math.pi * tau1 + 4.3214352)
    ts = 2.41 + f10b * g1 * g2
    return float(tl + tg + ts)


def _jslv(alt_km: float, xlat_rad: float, day: float) -> float:
    if alt_km > 170.0:
        return 0.0
    z = alt_km - 90.0
    d = 0.014 * z * math.exp(-0.0013 * z * z) * math.sin(0.0172 * day + 1.72) * math.sin(xlat_rad) ** 2
    return float(-d if xlat_rad < 0.0 else d)


def _jslvh(dl: float, denhe: float, xlat_rad: float, sda: float) -> float:
    ezero = 10.0**denhe
    a = abs(0.65 * (sda / 0.40909079))
    b = 0.5 * xlat_rad
    if sda < 0.0:
        b = -b
    denhe = denhe + a * (math.sin(0.7854 - b) ** 3 - 0.35356)
    rho = 10.0**dl + 6.646e-24 * (10.0**denhe - ezero)
    return math.log10(max(rho, 1e-300))


def _jfair(dhel1: float, dhel2: float, dlg1: float, dlg2: float, alt_km: float) -> tuple[float, float]:
    idx = max(0, min(int(math.trunc((alt_km - 440.0) / 10.0)), _CZ.size - 1))
    czi = float(_CZ[idx])
    szi = 1.0 - czi
    return dhel1 * czi + dhel2 * szi, dlg1 * czi + dlg2 * szi


@lru_cache(maxsize=4096)
def _default_sun_angles(jd_utc: float, sun_model: str) -> tuple[float, float]:
    if sun_model in {"hpop_simple", "hpop_validation_simple", "validation_simple", "simple"}:
        r_sun = sun_position_eci_km_simple(jd_utc)
    else:
        r_sun = sun_position_eci_km_enhanced(jd_utc)
    ra_sun = math.atan2(float(r_sun[1]), float(r_sun[0])) % (2.0 * math.pi)
    sda = math.atan2(float(r_sun[2]), math.hypot(float(r_sun[0]), float(r_sun[1])))
    return ra_sun, sda


@lru_cache(maxsize=4096)
def _position_rotation(jd_utc: float, eop_path: str | None) -> np.ndarray:
    rotation = eci_to_ecef_rotation_hpop_like(0.0, jd_utc_start=jd_utc, eop_path=eop_path)
    rotation.setflags(write=False)
    return rotation


def clear_trajectory_epoch_caches() -> None:
    """Clear per-epoch values so a benchmark can measure a fresh trajectory."""
    _time_inputs.cache_clear()
    _default_sun_angles.cache_clear()
    _position_rotation.cache_clear()


def _position_eci_from_geodetic(
    lat_deg: float, lon_deg: float, alt_km: float, jd_utc: float, env: dict
) -> np.ndarray:
    r_ecef_km = geodetic_to_ecef_km(lat_deg, lon_deg, alt_km)
    raw_eop_path = (
        env.get("density_eop_path")
        or env.get("drag_eop_path")
        or env.get("spherical_harmonics_eop_path")
    )
    eop_path = None if raw_eop_path is None else str(raw_eop_path)
    return _position_rotation(jd_utc, eop_path).T @ r_ecef_km


def jacchia70_density(
    alt_km: float, lat_deg: float, lon_deg: float, dt_utc: datetime, env: dict | None = None
) -> float:
    env = {} if env is None else env
    if dt_utc.tzinfo is None:
        dt_utc = dt_utc.replace(tzinfo=timezone.utc)
    dt_utc = dt_utc.astimezone(timezone.utc)

    z = max(90.0, min(float(alt_km), 2500.0))
    jd_utc, day = _time_inputs(dt_utc)
    r_sun = env.get("sun_pos_eci_km")
    if r_sun is None:
        sun_model = str(env.get("atmosphere_sun_model", env.get("sun_model", ""))).strip().lower()
        ra_sun, sda = _default_sun_angles(jd_utc, sun_model)
    else:
        r_sun = np.asarray(r_sun, dtype=float).reshape(3)
        ra_sun = math.atan2(float(r_sun[1]), float(r_sun[0])) % (2.0 * math.pi)
        sda = math.atan2(float(r_sun[2]), math.hypot(float(r_sun[0]), float(r_sun[1])))
    frame_model = str(env.get("density_frame_model", env.get("drag_frame_model", ""))).strip().lower()
    if normalize_frame_model(frame_model) == FRAME_MODEL_IAU76_80_EOP:
        eop_path = env.get("density_eop_path", env.get("drag_eop_path"))
        gast = env.get("jacchia70_gast_rad")
        if gast is None:
            gast = apparent_sidereal_time_hpop_like(
                jd_utc,
                None if eop_path is None else str(eop_path),
                eop_extrapolation=str(env.get("eop_extrapolation", "error") or "error"),
            )
        ra_sat = (float(gast) + math.radians(float(lon_deg))) % (2.0 * math.pi)
    else:
        r_sat_eci_km = _position_eci_from_geodetic(float(lat_deg), float(lon_deg), z, jd_utc, env)
        ra_sat = math.atan2(float(r_sat_eci_km[1]), float(r_sat_eci_km[0])) % (2.0 * math.pi)
    sha = (ra_sat - ra_sun) % (2.0 * math.pi)
    if sha > math.pi:
        sha -= 2.0 * math.pi
    if sha < -math.pi:
        sha += 2.0 * math.pi

    f10, f10b, gi, i1 = _solar_geomagnetic_inputs(dt_utc, env)
    xlat = math.radians(float(lat_deg))
    dy = day / 365.2422
    te = _jtinf(f10, f10b, gi, xlat, sda, sha, dy, i1)
    accelerated_dl = _accelerated_jacchia_mid_altitude_log(z, te) if 105.0 < z <= 440.0 else None
    if accelerated_dl is not None:
        dl = accelerated_dl
        helium_log10 = 0.0
    else:
        _, dl, _, _, species_log10 = _jacchia_python(z, te)
        helium_log10 = float(species_log10[4])

    denlg = _jslv(z, xlat, day) if z <= 170.0 else 0.0
    if z >= 500.0:
        dl = _jslvh(dl, helium_log10, xlat, sda)
    elif z > 440.0:
        dlg2 = _jslvh(dl, helium_log10, xlat, sda)
        _, dl = _jfair(helium_log10, helium_log10, dl, dlg2, z)

    return float(max(0.0, 1000.0 * 10.0 ** (dl + denlg)))
