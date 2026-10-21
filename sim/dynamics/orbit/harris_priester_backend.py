from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

from sim.dynamics.orbit.epoch import sun_position_eci_km_enhanced, sun_position_eci_km_simple
from sim.dynamics.orbit.frames import (
    FRAME_MODEL_IAU76_80_EOP,
    normalize_frame_model,
    precession_nutation_rotation_hpop_like,
)
from sim.utils.geodesy import ecef_to_geodetic_deg_km

_BACKEND_PATH = Path(__file__).resolve()
_DEFAULT_COEFF_PATH = str((_BACKEND_PATH.parent / "data" / "harris_priester_hpop.csv").resolve())
_PROJECT_ROOT = _BACKEND_PATH.parents[3]


def _default_coeff_path() -> Path:
    return Path(_DEFAULT_COEFF_PATH)


@lru_cache(maxsize=4)
def _load_coefficients(path: str) -> dict[int, np.ndarray]:
    arr = np.loadtxt(path, delimiter=",", comments="#", skiprows=3)
    if arr.ndim != 2 or arr.shape[1] != 4:
        raise ValueError(f"Unexpected Harris-Priester coefficient shape {arr.shape} from {path}")
    tables: dict[int, np.ndarray] = {}
    for f107 in sorted({int(v) for v in arr[:, 0]}):
        table = arr[arr[:, 0] == float(f107), 1:4]
        if table.shape[0] < 2:
            raise ValueError(f"Harris-Priester F10.7 table {f107} from {path} has too few rows.")
        table.setflags(write=False)
        tables[f107] = table
    return tables


@lru_cache(maxsize=8)
def _resolved_coeff_path(raw: str | None) -> str:
    if raw in (None, ""):
        return _DEFAULT_COEFF_PATH
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = _PROJECT_ROOT / p
    return str(p.resolve())


def _resolve_coeff_path(env: dict) -> str:
    raw = env.get("harris_priester_coeff_path") or env.get("hp_coeff_path")
    return _resolved_coeff_path(None if raw in (None, "") else str(raw))


@lru_cache(maxsize=64)
def _selected_f107_table(path: str, f107: float) -> tuple[int, np.ndarray]:
    tables = _load_coefficients(path)
    key = min(tables, key=lambda candidate: abs(float(candidate) - f107))
    return key, tables[key]


@lru_cache(maxsize=4096)
def _default_sun_position(jd_utc: float, sun_model: str) -> np.ndarray:
    if sun_model in {"hpop_simple", "hpop_validation_simple", "validation_simple", "simple"}:
        position = sun_position_eci_km_simple(jd_utc)
    else:
        position = sun_position_eci_km_enhanced(jd_utc)
    position = np.asarray(position, dtype=float).reshape(3)
    position.setflags(write=False)
    return position


def clear_trajectory_epoch_caches() -> None:
    """Clear per-epoch values so a benchmark can measure a fresh trajectory."""
    _default_sun_position.cache_clear()


def _sun_position_from_env(jd_utc: float | None, env: dict) -> np.ndarray:
    r_sun = env.get("sun_pos_eci_km")
    if r_sun is not None:
        return np.asarray(r_sun, dtype=float).reshape(3)
    if jd_utc is None:
        return np.array([1.0, 0.0, 0.0], dtype=float)
    sun_model = str(env.get("atmosphere_sun_model", env.get("sun_model", ""))).strip().lower()
    return _default_sun_position(float(jd_utc), sun_model)


def harris_priester_density(
    r_eci_km: np.ndarray,
    t_s: float,
    env: dict | None = None,
) -> float:
    """
    Modified Harris-Priester density model using the HPOP HP coefficient table.

    The model is intentionally local and deterministic. It follows the HPOP
    Density_HP formulation: altitude-indexed min/max density interpolation plus
    a solar-apex diurnal bulge term.
    """
    env_local = {} if env is None else env
    coeff_path = _resolve_coeff_path(env_local)
    raw_f107 = (
        env_local.get("harris_priester_f107")
        or env_local.get("hp_f107")
        or env_local.get("solar_flux_f107")
        or env_local.get("f107")
        or 175.0
    )
    _, table = _selected_f107_table(coeff_path, float(raw_f107))

    r_sat = np.asarray(r_eci_km, dtype=float).reshape(3)
    r_eval = r_sat
    frame_model = str(env_local.get("density_frame_model", env_local.get("drag_frame_model", ""))).strip().lower()
    eop_path = env_local.get("density_eop_path", env_local.get("drag_eop_path"))
    if normalize_frame_model(frame_model) == FRAME_MODEL_IAU76_80_EOP:
        rbpn = precession_nutation_rotation_hpop_like(
            float(t_s),
            jd_utc_start=env_local.get("jd_utc_start"),
            eop_path=None if eop_path is None else str(eop_path),
            eop_extrapolation=str(env_local.get("eop_extrapolation", "error") or "error"),
        )
        r_eval = rbpn @ r_sat
    r_norm = float(np.linalg.norm(r_eval))
    if r_norm <= 0.0:
        return 0.0
    if str(env_local.get("geodetic_model", "")).lower() == "wgs84":
        _, _, altitude_km = ecef_to_geodetic_deg_km(r_eval)
        altitude_km = max(float(altitude_km), 0.0)
    else:
        altitude_km = max(0.0, r_norm - float(env_local.get("earth_radius_km", 6378.137)))

    lower_limit_km = float(env_local.get("harris_priester_lower_limit_km", 110.0))
    upper_limit_km = float(env_local.get("harris_priester_upper_limit_km", 2000.0))
    if altitude_km <= lower_limit_km or altitude_km >= upper_limit_km:
        return 0.0

    altitudes = table[:, 0]
    idx = int(np.searchsorted(altitudes, altitude_km, side="right") - 1)
    idx = max(0, min(idx, altitudes.size - 2))
    h0 = float(altitudes[idx])
    h1 = float(altitudes[idx + 1])
    rho_min0 = float(table[idx, 1])
    rho_min1 = float(table[idx + 1, 1])
    rho_max0 = float(table[idx, 2])
    rho_max1 = float(table[idx + 1, 2])

    h_min = (h0 - h1) / np.log(rho_min1 / rho_min0)
    h_max = (h0 - h1) / np.log(rho_max1 / rho_max0)
    d_min = rho_min0 * np.exp((h0 - altitude_km) / h_min)
    d_max = rho_max0 * np.exp((h0 - altitude_km) / h_max)

    jd_utc = env_local.get("jd_utc")
    if jd_utc is None:
        jd_start = env_local.get("jd_utc_start")
        jd_utc = None if jd_start is None else float(jd_start) + float(t_s) / 86400.0
    r_sun = _sun_position_from_env(None if jd_utc is None else float(jd_utc), env_local)
    sun_norm = float(np.linalg.norm(r_sun))
    if sun_norm <= 0.0:
        return float(max(d_min, 0.0) * 1.0e-9)

    ra_lag_rad = float(env_local.get("harris_priester_ra_lag_rad", 0.523599))
    n_prm = float(env_local.get("harris_priester_n", 4.0))
    sun_unit = r_sun / sun_norm
    ra_sun = float(np.arctan2(sun_unit[1], sun_unit[0]))
    dec_sun = float(np.arcsin(np.clip(sun_unit[2], -1.0, 1.0)))
    cos_dec = float(np.cos(dec_sun))
    bulge_unit = np.array(
        [
            cos_dec * np.cos(ra_sun + ra_lag_rad),
            cos_dec * np.sin(ra_sun + ra_lag_rad),
            np.sin(dec_sun),
        ],
        dtype=float,
    )
    c_psi2 = 0.5 + 0.5 * float(np.dot(r_eval, bulge_unit)) / r_norm
    c_psi2 = float(np.clip(c_psi2, 0.0, 1.0))
    density_g_km3 = d_min + (d_max - d_min) * c_psi2**n_prm
    return float(max(density_g_km3, 0.0) * 1.0e-9)
