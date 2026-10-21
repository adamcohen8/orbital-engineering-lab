from __future__ import annotations

import hashlib
import math
import os
import tempfile
import urllib.request
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from threading import RLock

import numpy as np

from sim.dynamics.orbit.environment import EARTH_MU_KM3_S2, EARTH_RADIUS_KM
from sim.dynamics.orbit.frames import (
    FRAME_MODEL_IAU76_80_EOP,
    ecef_to_eci_harmonic,
    eci_to_ecef_harmonic,
    eci_to_ecef_rotation,
    eci_to_ecef_rotation_hpop_like,
    normalize_frame_model,
)

_NORMALIZED_SPHERICAL_HARMONIC_ACCEL_ECI_KERNEL = None
_SPHERICAL_HARMONIC_KERNEL_LOCK = RLock()


def _load_normalized_spherical_harmonic_kernel():
    global _NORMALIZED_SPHERICAL_HARMONIC_ACCEL_ECI_KERNEL
    kernel = _NORMALIZED_SPHERICAL_HARMONIC_ACCEL_ECI_KERNEL
    if kernel is None:
        with _SPHERICAL_HARMONIC_KERNEL_LOCK:
            kernel = _NORMALIZED_SPHERICAL_HARMONIC_ACCEL_ECI_KERNEL
            if kernel is None:
                from sim.acceleration.kernels.spherical_harmonics import (
                    normalized_spherical_harmonic_accel_eci_kernel,
                )

                kernel = normalized_spherical_harmonic_accel_eci_kernel
                _NORMALIZED_SPHERICAL_HARMONIC_ACCEL_ECI_KERNEL = kernel
    return kernel


@dataclass(frozen=True)
class SphericalHarmonicTerm:
    """
    Single (n, m) spherical harmonic term.

    Notes:
    - Uses unnormalized associated Legendre polynomials P_n^m.
    - Coefficients C_nm and S_nm are interpreted in the same (unnormalized) convention.
    """

    n: int
    m: int
    c_nm: float
    s_nm: float = 0.0
    normalized: bool = False

    def __post_init__(self) -> None:
        if self.n < 2:
            raise ValueError("n must be >= 2 for perturbation terms.")
        if self.m < 0 or self.m > self.n:
            raise ValueError("m must satisfy 0 <= m <= n.")


@dataclass(frozen=True)
class GravityModelDownload:
    urls: tuple[str, ...]
    sha256: str
    min_size_bytes: int = 1
    size_bytes: int | None = None


@dataclass(frozen=True)
class _ICGEMGravityModel:
    terms: tuple[SphericalHarmonicTerm, ...]
    reference_radius_km: float | None
    gravity_constant_km3_s2: float | None
    normalization: str


@dataclass(frozen=True)
class CompiledSphericalHarmonics:
    terms: tuple[SphericalHarmonicTerm, ...]
    n_max: int
    m_max: int
    c_nm: np.ndarray
    s_nm: np.ndarray
    all_normalized: bool
    legendre_diag_scale: np.ndarray
    legendre_subdiag_scale: np.ndarray
    legendre_recur_a: np.ndarray
    legendre_recur_b: np.ndarray
    legendre_recur_c: np.ndarray
    legendre_recurrence: tuple[tuple[int, int, float, float, float], ...]
    legendre_pnm_scratch: np.ndarray
    legendre_dpnm_scratch: np.ndarray


def compile_spherical_harmonic_terms(terms: list[SphericalHarmonicTerm]) -> CompiledSphericalHarmonics | None:
    if not terms:
        return None
    term_tuple = tuple(terms)
    n_max = max(term.n for term in term_tuple)
    m_max = max(term.m for term in term_tuple)
    c_nm = np.zeros((n_max + 1, m_max + 1), dtype=float)
    s_nm = np.zeros((n_max + 1, m_max + 1), dtype=float)
    c_nm[0, 0] = 1.0
    for term in term_tuple:
        c_nm[term.n, term.m] = float(term.c_nm)
        s_nm[term.n, term.m] = float(term.s_nm)
    diag_scale = np.zeros(n_max + 1, dtype=float)
    for i in range(2, n_max + 1):
        diag_scale[i] = math.sqrt((2.0 * i + 1.0) / (2.0 * i))
    subdiag_scale = np.zeros(n_max + 1, dtype=float)
    for i in range(1, n_max + 1):
        subdiag_scale[i] = math.sqrt(2.0 * i + 1.0)
    recur_a = np.zeros((n_max + 1, m_max + 1), dtype=float)
    recur_b = np.zeros((n_max + 1, m_max + 1), dtype=float)
    recur_c = np.zeros((n_max + 1, m_max + 1), dtype=float)
    recurrence: list[tuple[int, int, float, float, float]] = []
    j = 0
    k = 2
    while j <= m_max:
        for i in range(k, n_max + 1):
            recur_a[i, j] = math.sqrt((2.0 * i + 1.0) / ((i - j) * (i + j)))
            recur_b[i, j] = math.sqrt(2.0 * i - 1.0)
            recur_c[i, j] = math.sqrt(((i + j - 1.0) * (i - j - 1.0)) / (2.0 * i - 3.0))
            recurrence.append((i, j, recur_a[i, j], recur_b[i, j], recur_c[i, j]))
        j += 1
        k += 1
    return CompiledSphericalHarmonics(
        terms=term_tuple,
        n_max=n_max,
        m_max=m_max,
        c_nm=c_nm,
        s_nm=s_nm,
        all_normalized=all(bool(term.normalized) for term in term_tuple),
        legendre_diag_scale=diag_scale,
        legendre_subdiag_scale=subdiag_scale,
        legendre_recur_a=recur_a,
        legendre_recur_b=recur_b,
        legendre_recur_c=recur_c,
        legendre_recurrence=tuple(recurrence),
        legendre_pnm_scratch=np.empty((n_max + 1, m_max + 1), dtype=float),
        legendre_dpnm_scratch=np.empty((n_max + 1, m_max + 1), dtype=float),
    )


def _double_factorial(k: int) -> float:
    if k <= 0:
        return 1.0
    out = 1.0
    for i in range(k, 0, -2):
        out *= float(i)
    return out


def _associated_legendre_unnormalized(n: int, m: int, x: float) -> float:
    if m < 0 or m > n:
        return 0.0
    x = float(np.clip(x, -1.0, 1.0))
    # P_m^m
    p_mm = ((-1.0) ** m) * _double_factorial(2 * m - 1) * (max(0.0, 1.0 - x * x) ** (0.5 * m))
    if n == m:
        return p_mm
    # P_{m+1}^m
    p_m1m = x * (2 * m + 1) * p_mm
    if n == m + 1:
        return p_m1m
    p_nm2 = p_mm
    p_nm1 = p_m1m
    p_nm = p_nm1
    for ell in range(m + 2, n + 1):
        p_nm = ((2 * ell - 1) * x * p_nm1 - (ell + m - 1) * p_nm2) / (ell - m)
        p_nm2 = p_nm1
        p_nm1 = p_nm
    return p_nm


def _term_potential_ecef_km2_s2(
    r_ecef_km: np.ndarray,
    mu_km3_s2: float,
    re_km: float,
    term: SphericalHarmonicTerm,
) -> float:
    x, y, z = np.array(r_ecef_km, dtype=float)
    r = float(np.linalg.norm(r_ecef_km))
    if r <= 0.0:
        return 0.0
    sin_phi = z / r
    lon = float(np.arctan2(y, x))
    p_nm = _associated_legendre_unnormalized(term.n, term.m, sin_phi)
    if term.normalized:
        p_nm *= _fully_normalized_legendre_scale(term.n, term.m)
    amp = term.c_nm * np.cos(term.m * lon) + term.s_nm * np.sin(term.m * lon)
    return float(mu_km3_s2 / r * (re_km / r) ** term.n * p_nm * amp)


def _legendre_normalized_hpop(
    n_max: int,
    m_max: int,
    lat_gc_rad: float,
    compiled: CompiledSphericalHarmonics | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    # Compiled coefficient/recurrence data are immutable and shareable.  The
    # Legendre work arrays are evaluation-local so concurrent evaluations of a
    # single compiled environment cannot overwrite one another.
    pnm = np.zeros((n_max + 1, m_max + 1), dtype=float)
    dpnm = np.zeros((n_max + 1, m_max + 1), dtype=float)

    sin_f = float(np.sin(lat_gc_rad))
    cos_f = float(np.cos(lat_gc_rad))
    pnm[0, 0] = 1.0
    dpnm[0, 0] = 0.0
    if n_max >= 1 and m_max >= 1:
        pnm[1, 1] = math.sqrt(3.0) * cos_f
        dpnm[1, 1] = -math.sqrt(3.0) * sin_f

    for i in range(2, n_max + 1):
        if i <= m_max:
            scale = (
                float(compiled.legendre_diag_scale[i])
                if compiled is not None
                else math.sqrt((2.0 * i + 1.0) / (2.0 * i))
            )
            pnm[i, i] = scale * cos_f * pnm[i - 1, i - 1]
            dpnm[i, i] = scale * (cos_f * dpnm[i - 1, i - 1] - sin_f * pnm[i - 1, i - 1])

    for i in range(1, n_max + 1):
        m = i - 1
        if m <= m_max:
            scale = float(compiled.legendre_subdiag_scale[i]) if compiled is not None else math.sqrt(2.0 * i + 1.0)
            pnm[i, m] = scale * sin_f * pnm[i - 1, m]
            dpnm[i, m] = scale * (cos_f * pnm[i - 1, m] + sin_f * dpnm[i - 1, m])

    if compiled is not None:
        recurrence = compiled.legendre_recurrence
    else:
        recurrence_rows: list[tuple[int, int, float, float, float]] = []
        j = 0
        k = 2
        while j <= m_max:
            for i in range(k, n_max + 1):
                a = math.sqrt((2.0 * i + 1.0) / ((i - j) * (i + j)))
                b_scale = math.sqrt(2.0 * i - 1.0)
                c_scale = math.sqrt(((i + j - 1.0) * (i - j - 1.0)) / (2.0 * i - 3.0))
                recurrence_rows.append((i, j, a, b_scale, c_scale))
            j += 1
            k += 1
        recurrence = tuple(recurrence_rows)
    for i, j, a, b_scale, c_scale in recurrence:
        b = b_scale * sin_f * pnm[i - 1, j]
        c = c_scale * pnm[i - 2, j]
        pnm[i, j] = a * (b - c)
        db = b_scale * sin_f * dpnm[i - 1, j]
        dc = b_scale * cos_f * pnm[i - 1, j]
        dd = c_scale * dpnm[i - 2, j]
        dpnm[i, j] = a * (db + dc - dd)
    return pnm, dpnm


def _two_body_accel_km_s2(r_eci_km: np.ndarray, mu_km3_s2: float) -> np.ndarray:
    r2 = float(np.dot(r_eci_km, r_eci_km))
    if r2 <= 0.0:
        return np.zeros(3, dtype=float)
    r = math.sqrt(r2)
    return (-mu_km3_s2 / (r * r2)) * np.array(r_eci_km, dtype=float)


def _harmonic_rotation_matrix(
    t_s: float,
    jd_utc_start: float | None,
    frame_model: str,
    eop_path: str | None,
    dut1_s: float | None = None,
    xp_arcsec: float | None = None,
    yp_arcsec: float | None = None,
    dat_s: float | None = None,
    tt_minus_utc_s: float | None = None,
    ddpsi_rad: float = 0.0,
    ddeps_rad: float = 0.0,
    eop_extrapolation: str = "error",
) -> np.ndarray:
    model = normalize_frame_model(frame_model)
    if model == FRAME_MODEL_IAU76_80_EOP:
        return eci_to_ecef_rotation_hpop_like(
            t_s,
            jd_utc_start=jd_utc_start,
            eop_path=eop_path,
            dut1_s=dut1_s,
            xp_arcsec=xp_arcsec,
            yp_arcsec=yp_arcsec,
            dat_s=dat_s,
            tt_minus_utc_s=tt_minus_utc_s,
            ddpsi_rad=ddpsi_rad,
            ddeps_rad=ddeps_rad,
            eop_extrapolation=eop_extrapolation,
        )
    return eci_to_ecef_rotation(t_s, jd_utc_start=jd_utc_start)


def _analytic_harmonic_accel_hpop_eci_km_s2(
    *,
    r_eci_km: np.ndarray,
    t_s: float,
    terms: list[SphericalHarmonicTerm],
    mu_km3_s2: float,
    re_km: float,
    jd_utc_start: float | None,
    frame_model: str,
    eop_path: str | None,
    dut1_s: float | None = None,
    xp_arcsec: float | None = None,
    yp_arcsec: float | None = None,
    dat_s: float | None = None,
    tt_minus_utc_s: float | None = None,
    ddpsi_rad: float = 0.0,
    ddeps_rad: float = 0.0,
    eop_extrapolation: str = "error",
    compiled: CompiledSphericalHarmonics | None = None,
) -> np.ndarray:
    compiled_terms = compiled if compiled is not None else compile_spherical_harmonic_terms(terms)
    if compiled_terms is None:
        return np.zeros(3, dtype=float)
    n_max = compiled_terms.n_max
    m_max = compiled_terms.m_max
    c_nm = compiled_terms.c_nm
    s_nm = compiled_terms.s_nm

    e_mat = _harmonic_rotation_matrix(
        t_s=float(t_s),
        jd_utc_start=jd_utc_start,
        frame_model=frame_model,
        eop_path=eop_path,
        dut1_s=dut1_s,
        xp_arcsec=xp_arcsec,
        yp_arcsec=yp_arcsec,
        dat_s=dat_s,
        tt_minus_utc_s=tt_minus_utc_s,
        ddpsi_rad=ddpsi_rad,
        ddeps_rad=ddeps_rad,
        eop_extrapolation=eop_extrapolation,
    )
    r_eci = np.array(r_eci_km, dtype=float).reshape(3)
    r_bf = e_mat @ r_eci
    d = float(np.linalg.norm(r_bf))
    if d <= 0.0:
        return np.zeros(3, dtype=float)
    lat_gc = math.asin(float(r_bf[2]) / d)
    lon = math.atan2(float(r_bf[1]), float(r_bf[0]))
    pnm, dpnm = _legendre_normalized_hpop(
        n_max=n_max,
        m_max=m_max,
        lat_gc_rad=lat_gc,
        compiled=compiled_terms,
    )
    cos_mlon = tuple(math.cos(m * lon) for m in range(m_max + 1))
    sin_mlon = tuple(math.sin(m * lon) for m in range(m_max + 1))

    dUdr = 0.0
    dUdlatgc = 0.0
    dUdlon = 0.0
    for n in range(0, n_max + 1):
        b1 = (-mu_km3_s2 / (d * d)) * (re_km / d) ** n * (n + 1.0)
        b2 = (mu_km3_s2 / d) * (re_km / d) ** n
        b3 = b2
        q1 = 0.0
        q2 = 0.0
        q3 = 0.0
        pnm_n = pnm[n]
        dpnm_n = dpnm[n]
        c_nm_n = c_nm[n]
        s_nm_n = s_nm[n]
        for m in range(0, min(m_max, n) + 1):
            cos_ml = cos_mlon[m]
            sin_ml = sin_mlon[m]
            amp = c_nm_n[m] * cos_ml + s_nm_n[m] * sin_ml
            q1 += pnm_n[m] * amp
            q2 += dpnm_n[m] * amp
            q3 += m * pnm_n[m] * (s_nm_n[m] * cos_ml - c_nm_n[m] * sin_ml)
        dUdr += q1 * b1
        dUdlatgc += q2 * b2
        dUdlon += q3 * b3

    r2xy = float(r_bf[0] * r_bf[0] + r_bf[1] * r_bf[1])
    if r2xy <= 0.0:
        r2xy = np.finfo(float).eps
    sqrt_r2xy = math.sqrt(r2xy)
    common = (1.0 / d) * dUdr - (float(r_bf[2]) / (d * d * sqrt_r2xy)) * dUdlatgc
    ax = common * float(r_bf[0]) - (1.0 / r2xy) * dUdlon * float(r_bf[1])
    ay = common * float(r_bf[1]) + (1.0 / r2xy) * dUdlon * float(r_bf[0])
    az = (1.0 / d) * dUdr * float(r_bf[2]) + (sqrt_r2xy / (d * d)) * dUdlatgc
    a_bf = np.array([ax, ay, az], dtype=float)
    a_eci_full = e_mat.T @ a_bf
    return a_eci_full - _two_body_accel_km_s2(r_eci, mu_km3_s2)


def _accelerated_harmonic_accel_hpop_eci_km_s2(
    *,
    r_eci_km: np.ndarray,
    t_s: float,
    mu_km3_s2: float,
    re_km: float,
    jd_utc_start: float | None,
    frame_model: str,
    eop_path: str | None,
    dut1_s: float | None,
    xp_arcsec: float | None,
    yp_arcsec: float | None,
    dat_s: float | None,
    tt_minus_utc_s: float | None,
    ddpsi_rad: float,
    ddeps_rad: float,
    eop_extrapolation: str,
    compiled: CompiledSphericalHarmonics,
) -> np.ndarray:
    """Run normalized harmonic mathematics through the optional backend."""

    kernel = _load_normalized_spherical_harmonic_kernel()

    rotation = _harmonic_rotation_matrix(
        t_s=float(t_s),
        jd_utc_start=jd_utc_start,
        frame_model=frame_model,
        eop_path=eop_path,
        dut1_s=dut1_s,
        xp_arcsec=xp_arcsec,
        yp_arcsec=yp_arcsec,
        dat_s=dat_s,
        tt_minus_utc_s=tt_minus_utc_s,
        ddpsi_rad=ddpsi_rad,
        ddeps_rad=ddeps_rad,
        eop_extrapolation=eop_extrapolation,
    )
    return kernel(
        np.asarray(r_eci_km, dtype=float).reshape(3),
        rotation,
        float(mu_km3_s2),
        float(re_km),
        compiled.c_nm,
        compiled.s_nm,
        compiled.legendre_diag_scale,
        compiled.legendre_subdiag_scale,
        compiled.legendre_recur_a,
        compiled.legendre_recur_b,
        compiled.legendre_recur_c,
        int(compiled.n_max),
        int(compiled.m_max),
    )


def accel_spherical_harmonics_terms(
    r_eci_km: np.ndarray,
    t_s: float,
    terms: list[SphericalHarmonicTerm],
    mu_km3_s2: float = EARTH_MU_KM3_S2,
    re_km: float = EARTH_RADIUS_KM,
    fd_step_km: float = 1e-3,
    jd_utc_start: float | None = None,
    frame_model: str = "simple",
    eop_path: str | None = None,
    dut1_s: float | None = None,
    xp_arcsec: float | None = None,
    yp_arcsec: float | None = None,
    dat_s: float | None = None,
    tt_minus_utc_s: float | None = None,
    ddpsi_rad: float = 0.0,
    ddeps_rad: float = 0.0,
    eop_extrapolation: str = "error",
    compiled: CompiledSphericalHarmonics | None = None,
    use_acceleration: bool = False,
) -> np.ndarray:
    """
    Acceleration in ECI from arbitrary spherical-harmonic terms (n,m).

    Uses finite-difference gradient of the perturbing potential in ECEF.
    """
    if not terms:
        return np.zeros(3)
    compiled_terms = compiled if compiled is not None else compile_spherical_harmonic_terms(terms)
    if compiled_terms is not None and compiled_terms.all_normalized:
        if use_acceleration:
            return _accelerated_harmonic_accel_hpop_eci_km_s2(
                r_eci_km=r_eci_km,
                t_s=t_s,
                mu_km3_s2=mu_km3_s2,
                re_km=re_km,
                jd_utc_start=jd_utc_start,
                frame_model=frame_model,
                eop_path=eop_path,
                dut1_s=dut1_s,
                xp_arcsec=xp_arcsec,
                yp_arcsec=yp_arcsec,
                dat_s=dat_s,
                tt_minus_utc_s=tt_minus_utc_s,
                ddpsi_rad=ddpsi_rad,
                ddeps_rad=ddeps_rad,
                eop_extrapolation=eop_extrapolation,
                compiled=compiled_terms,
            )
        return _analytic_harmonic_accel_hpop_eci_km_s2(
            r_eci_km=r_eci_km,
            t_s=t_s,
            terms=terms,
            mu_km3_s2=mu_km3_s2,
            re_km=re_km,
            jd_utc_start=jd_utc_start,
            frame_model=frame_model,
            eop_path=eop_path,
            dut1_s=dut1_s,
            xp_arcsec=xp_arcsec,
            yp_arcsec=yp_arcsec,
            dat_s=dat_s,
            tt_minus_utc_s=tt_minus_utc_s,
            ddpsi_rad=ddpsi_rad,
            ddeps_rad=ddeps_rad,
            eop_extrapolation=eop_extrapolation,
            compiled=compiled_terms,
        )
    r_ecef = eci_to_ecef_harmonic(
        np.array(r_eci_km, dtype=float),
        float(t_s),
        jd_utc_start=jd_utc_start,
        frame_model=frame_model,
        eop_path=eop_path,
        dut1_s=dut1_s,
        xp_arcsec=xp_arcsec,
        yp_arcsec=yp_arcsec,
        dat_s=dat_s,
        tt_minus_utc_s=tt_minus_utc_s,
        ddpsi_rad=ddpsi_rad,
        ddeps_rad=ddeps_rad,
        eop_extrapolation=eop_extrapolation,
    )

    if fd_step_km <= 0.0:
        raise ValueError("fd_step_km must be positive.")

    def _u_at(pos_ecef_km: np.ndarray) -> float:
        u = 0.0
        for term in terms:
            u += _term_potential_ecef_km2_s2(pos_ecef_km, mu_km3_s2, re_km, term)
        return float(u)

    h = float(fd_step_km)
    grad_ecef = np.zeros(3)
    for i in range(3):
        d = np.zeros(3)
        d[i] = h
        up = _u_at(r_ecef + d)
        um = _u_at(r_ecef - d)
        grad_ecef[i] = (up - um) / (2.0 * h)

    # Gravity acceleration equals gradient of potential.
    return ecef_to_eci_harmonic(
        grad_ecef,
        float(t_s),
        jd_utc_start=jd_utc_start,
        frame_model=frame_model,
        eop_path=eop_path,
        dut1_s=dut1_s,
        xp_arcsec=xp_arcsec,
        yp_arcsec=yp_arcsec,
        dat_s=dat_s,
        tt_minus_utc_s=tt_minus_utc_s,
        ddpsi_rad=ddpsi_rad,
        ddeps_rad=ddeps_rad,
        eop_extrapolation=eop_extrapolation,
    )


def default_hpop_ggm03_coeff_path() -> Path:
    return Path(__file__).resolve().parents[3] / "validation" / "data" / "GGM03C.txt"


def configure_spherical_harmonics_env(base_env: dict | None, orbit_cfg: dict | None) -> dict:
    env = dict(base_env or {})
    orbit = dict(orbit_cfg or {})
    sh = dict(orbit.get("spherical_harmonics", {}) or {})
    drag_frame_model = orbit.get("drag_frame_model")
    if drag_frame_model is not None:
        env["drag_frame_model"] = str(drag_frame_model)
    drag_eop_path_raw = orbit.get("drag_eop_path")
    if drag_eop_path_raw is not None:
        drag_eop_path = Path(str(drag_eop_path_raw)).expanduser()
        if not drag_eop_path.is_absolute():
            drag_eop_path = Path(__file__).resolve().parents[3] / drag_eop_path
        env["drag_eop_path"] = str(drag_eop_path.resolve())
    if orbit.get("drag_earth_rotation_rad_s") is not None:
        env["drag_earth_rotation_rad_s"] = float(orbit["drag_earth_rotation_rad_s"])
    drag_uses_eop = normalize_frame_model(env.get("drag_frame_model", "simple")) == FRAME_MODEL_IAU76_80_EOP
    if not bool(sh.get("enabled", False)):
        if drag_uses_eop and env.get("drag_eop_path") is None:
            default_eop = (
                Path(__file__).resolve().parents[3]
                / "validation/High Precision Orbit Propagator_4-2/High Precision Orbit Propagator_4.2.2/EOP-All.txt"
            )
            env["drag_eop_path"] = str(default_eop.resolve())
        return env

    raw_terms = sh.get("terms")
    parsed_terms = parse_spherical_harmonic_terms(raw_terms) if raw_terms not in (None, "") else []
    inferred_degree = max((term.n for term in parsed_terms), default=0)
    inferred_order = max((term.m for term in parsed_terms), default=0)
    degree = int(max(int(sh.get("degree", inferred_degree) or 0), 0))
    if degree < 2:
        raise ValueError(
            "Enabled spherical harmonics require degree >= 2 or inline terms from which degree can be inferred."
        )
    order_raw = sh.get("order", inferred_order if parsed_terms else degree)
    order = int(max(min(int(degree if order_raw is None else order_raw), degree), 0))
    source = str(sh.get("source", sh.get("model", "")) or "").strip().lower()
    coeff_path_raw = sh.get("coeff_path") or sh.get("source_path")
    if not source and coeff_path_raw not in (None, ""):
        source = "icgem"

    if source in {"hpop", "hpop_ggm03", "ggm03"}:
        coeff_path = (
            default_hpop_ggm03_coeff_path() if coeff_path_raw in (None, "") else Path(str(coeff_path_raw)).expanduser()
        )
        if not coeff_path.is_absolute():
            coeff_path = Path(__file__).resolve().parents[3] / coeff_path
        if coeff_path_raw in (None, "") and not coeff_path.exists():
            raise FileNotFoundError(
                "HPOP GGM03 coefficients are not bundled with this distribution. "
                "Provide spherical_harmonics.coeff_path/source_path, or use explicit "
                "spherical_harmonics.terms for public-core scenarios."
            )
        terms = load_hpop_ggm03_terms(
            coeff_path=coeff_path,
            max_degree=degree,
            max_order=order,
            normalized=bool(sh.get("normalized", True)),
        )
        env["spherical_harmonics_terms"] = terms
        env["_parsed_spherical_harmonics_terms"] = terms
        compiled = compile_spherical_harmonic_terms(terms)
        if compiled is not None:
            env["_compiled_spherical_harmonics_terms"] = compiled
        env["spherical_harmonics_source"] = str(coeff_path.resolve())
        env["spherical_harmonics_reference_radius_km"] = float(sh.get("reference_radius_km", 6378.1363))
        env["spherical_harmonics_frame_model"] = str(sh.get("frame_model", "hpop_like"))
        eop_path_raw = (
            sh.get("eop_path")
            or "validation/High Precision Orbit Propagator_4-2/High Precision Orbit Propagator_4.2.2/EOP-All.txt"
        )
        eop_path = Path(str(eop_path_raw)).expanduser()
        if not eop_path.is_absolute():
            eop_path = Path(__file__).resolve().parents[3] / eop_path
        env["spherical_harmonics_eop_path"] = str(eop_path.resolve())
    elif parsed_terms:
        terms = [term for term in parsed_terms if term.n <= degree and term.m <= order]
        if not terms:
            raise ValueError(f"No inline spherical harmonic terms remain within degree {degree} and order {order}.")
        env["spherical_harmonics_terms"] = terms
        env["_parsed_spherical_harmonics_terms"] = terms
        compiled = compile_spherical_harmonic_terms(terms)
        if compiled is not None:
            env["_compiled_spherical_harmonics_terms"] = compiled
        if sh.get("reference_radius_km") is not None:
            env["spherical_harmonics_reference_radius_km"] = float(sh["reference_radius_km"])
        if sh.get("frame_model") is not None:
            env["spherical_harmonics_frame_model"] = str(sh["frame_model"])
        if sh.get("eop_path") is not None:
            eop_path = Path(str(sh["eop_path"])).expanduser()
            if not eop_path.is_absolute():
                eop_path = Path(__file__).resolve().parents[3] / eop_path
            env["spherical_harmonics_eop_path"] = str(eop_path.resolve())
    elif source in {"icgem", "gfc"}:
        if coeff_path_raw in (None, ""):
            raise ValueError("ICGEM spherical harmonics require coeff_path or source_path.")
        coeff_path = Path(str(coeff_path_raw)).expanduser()
        if not coeff_path.is_absolute():
            coeff_path = Path(__file__).resolve().parents[3] / coeff_path
        model = _load_icgem_gfc_model(coeff_path, max_degree=degree, max_order=order)
        terms = list(model.terms)
        env["spherical_harmonics_terms"] = terms
        env["_parsed_spherical_harmonics_terms"] = terms
        compiled = compile_spherical_harmonic_terms(terms)
        if compiled is not None:
            env["_compiled_spherical_harmonics_terms"] = compiled
        env["spherical_harmonics_source"] = str(coeff_path.resolve())
        reference_radius_km = sh.get("reference_radius_km", model.reference_radius_km)
        env["spherical_harmonics_reference_radius_km"] = float(
            EARTH_RADIUS_KM if reference_radius_km is None else reference_radius_km
        )
        if model.gravity_constant_km3_s2 is not None:
            env["spherical_harmonics_mu_km3_s2"] = float(model.gravity_constant_km3_s2)
        env["spherical_harmonics_frame_model"] = str(sh.get("frame_model", "simple"))
        if sh.get("eop_path") is not None:
            eop_path = Path(str(sh["eop_path"])).expanduser()
            if not eop_path.is_absolute():
                eop_path = Path(__file__).resolve().parents[3] / eop_path
            env["spherical_harmonics_eop_path"] = str(eop_path.resolve())
    elif source == "egm96":
        env["spherical_harmonics_use_real_coefficients"] = True
        env["spherical_harmonics_model"] = "EGM96"
        env["spherical_harmonics_max_degree"] = degree
        env["spherical_harmonics_max_order"] = order
        env["spherical_harmonics_allow_download"] = bool(sh.get("allow_download", True))
        model_reference_radius_km = 6378.1363
        model_gravity_constant_km3_s2 = EARTH_MU_KM3_S2
        if coeff_path_raw not in (None, ""):
            coeff_path = Path(str(coeff_path_raw)).expanduser()
            if not coeff_path.is_absolute():
                coeff_path = Path(__file__).resolve().parents[3] / coeff_path
            model = _load_icgem_gfc_model(coeff_path, max_degree=degree, max_order=order)
            if model.reference_radius_km is not None:
                model_reference_radius_km = model.reference_radius_km
            if model.gravity_constant_km3_s2 is not None:
                model_gravity_constant_km3_s2 = model.gravity_constant_km3_s2
            env["spherical_harmonics_coeff_path"] = str(coeff_path.resolve())
            env["spherical_harmonics_source"] = str(coeff_path.resolve())
        else:
            env["spherical_harmonics_source"] = "EGM96"
        env["spherical_harmonics_reference_radius_km"] = float(
            sh.get("reference_radius_km", model_reference_radius_km)
        )
        env["spherical_harmonics_mu_km3_s2"] = float(model_gravity_constant_km3_s2)
        env["spherical_harmonics_frame_model"] = str(sh.get("frame_model", "simple"))
        if sh.get("eop_path") is not None:
            eop_path = Path(str(sh["eop_path"])).expanduser()
            if not eop_path.is_absolute():
                eop_path = Path(__file__).resolve().parents[3] / eop_path
            env["spherical_harmonics_eop_path"] = str(eop_path.resolve())
    else:
        raise ValueError(
            "Enabled spherical harmonics did not select inline terms or a supported coefficient source."
        )

    if sh.get("fd_step_km") is not None:
        env["spherical_harmonics_fd_step_km"] = float(sh["fd_step_km"])
    if drag_uses_eop and env.get("drag_eop_path") is None:
        if env.get("spherical_harmonics_eop_path") is not None:
            env["drag_eop_path"] = str(env["spherical_harmonics_eop_path"])
        else:
            default_eop = (
                Path(__file__).resolve().parents[3]
                / "validation/High Precision Orbit Propagator_4-2/High Precision Orbit Propagator_4.2.2/EOP-All.txt"
            )
            env["drag_eop_path"] = str(default_eop.resolve())
    return env


def _fully_normalized_legendre_scale(n: int, m: int) -> float:
    # sqrt((2-delta_0m)*(2n+1)*(n-m)!/(n+m)!)
    delta = 1.0 if m == 0 else 0.0
    log_scale = math.log((2.0 - delta) * (2.0 * n + 1.0)) + math.lgamma(n - m + 1.0) - math.lgamma(n + m + 1.0)
    return float(math.exp(0.5 * log_scale))


def parse_spherical_harmonic_terms(raw_terms: list[dict | SphericalHarmonicTerm] | None) -> list[SphericalHarmonicTerm]:
    if not raw_terms:
        return []
    out: list[SphericalHarmonicTerm] = []
    seen: set[tuple[int, int]] = set()
    for i, item in enumerate(raw_terms):
        if isinstance(item, SphericalHarmonicTerm):
            term = item
        else:
            if not isinstance(item, dict):
                raise ValueError(f"spherical harmonic term index {i} must be a dict or SphericalHarmonicTerm.")
            n = int(item.get("n", -1))
            m = int(item.get("m", -1))
            c_nm = float(item.get("c_nm", item.get("c", 0.0)))
            s_nm = float(item.get("s_nm", item.get("s", 0.0)))
            normalized = bool(item.get("normalized", False))
            term = SphericalHarmonicTerm(n=n, m=m, c_nm=c_nm, s_nm=s_nm, normalized=normalized)
        key = (int(term.n), int(term.m))
        if key in seen:
            raise ValueError(f"Duplicate spherical harmonic coefficient for (n, m)={key} at term index {i}.")
        seen.add(key)
        out.append(term)
    return out


def _load_icgem_gfc_model(
    gfc_path: str | Path,
    max_degree: int,
    max_order: int | None = None,
) -> _ICGEMGravityModel:
    path = Path(gfc_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Gravity coefficient file not found: {path}")

    n_max = int(max_degree)
    if n_max < 2:
        raise ValueError("max_degree must be >= 2.")
    m_max = n_max if max_order is None else int(max_order)
    if m_max < 0:
        raise ValueError("max_order must be >= 0.")

    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    norm_kind = "unknown"
    reference_radius_km = None
    gravity_constant_km3_s2 = None
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        key = parts[0].lower()
        if key == "norm":
            norm_kind = parts[1].strip().lower()
        elif key == "radius":
            reference_radius_km = float(parts[1].replace("D", "E").replace("d", "e")) / 1000.0
        elif key == "earth_gravity_constant":
            gravity_constant_km3_s2 = float(parts[1].replace("D", "E").replace("d", "e")) / 1.0e9

    for label, value in (
        ("radius", reference_radius_km),
        ("earth_gravity_constant", gravity_constant_km3_s2),
    ):
        if value is not None and (not math.isfinite(value) or value <= 0.0):
            raise ValueError(f"ICGEM {label} must be finite and positive in {path}.")

    out: list[SphericalHarmonicTerm] = []
    for line in lines:
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        parts = s.split()
        if parts[0].lower() != "gfc" or len(parts) < 5:
            continue
        n = int(parts[1])
        m = int(parts[2])
        if n < 2 or n > n_max or m < 0 or m > min(n, m_max):
            continue
        c_nm = float(parts[3].replace("D", "E").replace("d", "e"))
        s_nm = float(parts[4].replace("D", "E").replace("d", "e"))
        out.append(
            SphericalHarmonicTerm(
                n=n,
                m=m,
                c_nm=c_nm,
                s_nm=s_nm,
                normalized=("unnormalized" not in norm_kind),
            )
        )

    if not out:
        raise ValueError(f"No usable spherical harmonic terms found in {path} for n<= {n_max}, m<= {m_max}.")
    return _ICGEMGravityModel(
        terms=tuple(out),
        reference_radius_km=reference_radius_km,
        gravity_constant_km3_s2=gravity_constant_km3_s2,
        normalization=norm_kind,
    )


def load_icgem_gfc_terms(
    gfc_path: str | Path,
    max_degree: int,
    max_order: int | None = None,
) -> list[SphericalHarmonicTerm]:
    """
    Load real gravity coefficients from an ICGEM-style .gfc file.

    The parser supports coefficients on `gfc` lines:
      gfc n m Cnm Snm ...
    and respects the `norm` header when present. Configuration-backed loading
    additionally applies the file's reference radius and gravity constant.
    """

    return list(_load_icgem_gfc_model(gfc_path, max_degree=max_degree, max_order=max_order).terms)


def load_hpop_ggm03_terms(
    coeff_path: str | Path,
    max_degree: int,
    max_order: int | None = None,
    normalized: bool = True,
) -> list[SphericalHarmonicTerm]:
    """
    Load HPOP-style GGM03 coefficient table (e.g., GGM03C.txt).

    Expected row format:
      n  m  Cnm  Snm  sigmaC  sigmaS
    """
    path = Path(coeff_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"HPOP gravity coefficient file not found: {path}")

    n_max = int(max_degree)
    if n_max < 2:
        raise ValueError("max_degree must be >= 2.")
    m_max = n_max if max_order is None else int(max_order)
    if m_max < 0:
        raise ValueError("max_order must be >= 0.")

    out: list[SphericalHarmonicTerm] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            parts = s.split()
            if len(parts) < 4:
                continue
            try:
                n = int(parts[0])
                m = int(parts[1])
                c_nm = float(parts[2])
                s_nm = float(parts[3])
            except ValueError:
                continue
            if n < 2 or n > n_max or m < 0 or m > min(n, m_max):
                continue
            out.append(
                SphericalHarmonicTerm(
                    n=n,
                    m=m,
                    c_nm=c_nm,
                    s_nm=s_nm,
                    normalized=bool(normalized),
                )
            )

    if not out:
        raise ValueError(f"No usable GGM03 terms found in {path} for n<= {n_max}, m<= {m_max}.")
    return out


_REAL_MODEL_DOWNLOADS = {
    "EGM96": GravityModelDownload(
        urls=(
            # SatelliteToolboxGravityModels.jl documented direct ICGEM link.
            "https://icgem.gfz-potsdam.de/getmodel/gfc/"
            "971b0a3b49a497910aad23cd85e066d4cd9af0aeafe7ce6301a696bed8570be3/EGM96.gfc",
        ),
        sha256="5247a9e9c316dd2c8f8fd491d53be0e163cb5cb3676b021754240cc9e44cb43b",
        min_size_bytes=5_000_000,
        size_bytes=5_620_000,
    )
}


def _gravity_model_download_allowed(requested: bool) -> bool:
    if not requested:
        return False
    sealed = os.environ.get("OEL_SEALED_MODE", "").strip().lower() in {"1", "true", "yes", "on"}
    approved = os.environ.get("OEL_ALLOW_GRAVITY_MODEL_DOWNLOADS", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    return not sealed or approved


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_downloaded_model_file(model: str, path: Path) -> None:
    spec = _REAL_MODEL_DOWNLOADS.get(model.upper())
    if spec is None:
        raise RuntimeError(f"No download metadata configured for gravity model '{model}'.")
    if not path.exists():
        raise FileNotFoundError(path)
    size = path.stat().st_size
    if size < spec.min_size_bytes:
        raise RuntimeError(
            f"Downloaded gravity model '{model}' at {path} is too small: "
            f"{size} bytes < {spec.min_size_bytes} bytes."
        )
    if spec.size_bytes is not None and size != spec.size_bytes:
        raise RuntimeError(
            f"Downloaded gravity model '{model}' at {path} has unexpected size: "
            f"{size} bytes != {spec.size_bytes} bytes."
        )
    actual = _sha256_file(path)
    expected = spec.sha256.lower()
    if actual.lower() != expected:
        raise RuntimeError(
            f"Gravity model '{model}' integrity check failed for {path}: "
            f"sha256={actual}, expected {expected}."
        )


def _download_model_file(model: str, outpath: Path) -> None:
    spec = _REAL_MODEL_DOWNLOADS.get(model.upper())
    urls = () if spec is None else spec.urls
    last_err: Exception | None = None
    for url in urls:
        tmp_path: Path | None = None
        try:
            outpath.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                prefix=f".{outpath.name}.",
                suffix=".tmp",
                dir=outpath.parent,
                delete=False,
            ) as tmp:
                tmp_path = Path(tmp.name)
                with urllib.request.urlopen(url, timeout=120.0) as response:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        tmp.write(chunk)
                tmp.flush()
                os.fsync(tmp.fileno())
            _verify_downloaded_model_file(model=model, path=tmp_path)
            os.replace(tmp_path, outpath)
            return
        except Exception as exc:
            last_err = exc
            if tmp_path is not None:
                tmp_path.unlink(missing_ok=True)
    if last_err is not None:
        raise RuntimeError(f"Failed downloading gravity model '{model}' to {outpath}: {last_err}") from last_err
    raise RuntimeError(f"No download URL configured for gravity model '{model}'.")


@lru_cache(maxsize=16)
def _cached_real_terms(
    model: str,
    coeff_path: str | None,
    max_degree: int,
    max_order: int | None,
    allow_download: bool,
) -> tuple[SphericalHarmonicTerm, ...]:
    if coeff_path:
        path = Path(coeff_path).expanduser().resolve()
    else:
        preferred_cache_dir = Path.home() / ".orbital_engineering_lab" / "gravity_models"
        previous_cache_dir = Path.home() / ".orbital_engagement_lab" / "gravity_models"
        legacy_cache_dir = Path.home() / ".noncooprpo" / "gravity_models"
        preferred_path = preferred_cache_dir / f"{model.upper()}.gfc"
        previous_path = previous_cache_dir / f"{model.upper()}.gfc"
        legacy_path = legacy_cache_dir / f"{model.upper()}.gfc"
        # Reuse verified downloads from either earlier project namespace.
        path = next(
            (candidate for candidate in (preferred_path, previous_path, legacy_path) if candidate.exists()),
            preferred_path,
        )
        if path.exists():
            try:
                _verify_downloaded_model_file(model=model, path=path)
            except RuntimeError:
                if not allow_download:
                    raise
                path = preferred_path
                _download_model_file(model=model, outpath=path)
        else:
            if not allow_download:
                raise FileNotFoundError(
                    f"Real gravity coefficients requested but file not found: {path}. "
                    "Provide spherical_harmonics_coeff_path or enable download."
                )
            _download_model_file(model=model, outpath=path)

    terms = load_icgem_gfc_terms(path, max_degree=max_degree, max_order=max_order)
    return tuple(terms)


def load_real_earth_gravity_terms(
    max_degree: int,
    max_order: int | None = None,
    model: str = "EGM96",
    coeff_path: str | None = None,
    allow_download: bool = True,
) -> list[SphericalHarmonicTerm]:
    """
    Load real Earth gravity terms from an external coefficient file/model.

    Returns real (not synthetic) coefficients suitable for spherical harmonics propagation.
    """
    return list(
        _cached_real_terms(
            model=str(model).upper(),
            coeff_path=None if coeff_path is None else str(Path(coeff_path).expanduser().resolve()),
            max_degree=int(max_degree),
            max_order=None if max_order is None else int(max_order),
            allow_download=_gravity_model_download_allowed(bool(allow_download)),
        )
    )
