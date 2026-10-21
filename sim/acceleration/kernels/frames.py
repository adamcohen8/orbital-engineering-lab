from __future__ import annotations

import math

import numpy as np

from sim.acceleration.optional import njit_or_identity

_ARCSEC_TO_RAD = math.pi / (180.0 * 3600.0)
_J2000 = 2451545.0
_JULIAN_CENTURY_DAYS = 36525.0


@njit_or_identity(cache=True, fastmath=False)
def _earth_frame_rx(angle_rad: float) -> np.ndarray:
    sine = math.sin(angle_rad)
    cosine = math.cos(angle_rad)
    return np.array(
        [[1.0, 0.0, 0.0], [0.0, cosine, sine], [0.0, -sine, cosine]],
        dtype=np.float64,
    )


@njit_or_identity(cache=True, fastmath=False)
def _earth_frame_ry(angle_rad: float) -> np.ndarray:
    sine = math.sin(angle_rad)
    cosine = math.cos(angle_rad)
    return np.array(
        [[cosine, 0.0, -sine], [0.0, 1.0, 0.0], [sine, 0.0, cosine]],
        dtype=np.float64,
    )


@njit_or_identity(cache=True, fastmath=False)
def _earth_frame_rz(angle_rad: float) -> np.ndarray:
    sine = math.sin(angle_rad)
    cosine = math.cos(angle_rad)
    return np.array(
        [[cosine, sine, 0.0], [-sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


@njit_or_identity(cache=True, fastmath=False)
def _iau80_nutation_angles_kernel(
    jd_tt: float,
    ddpsi_rad: float,
    ddeps_rad: float,
    nutation_coefficients: np.ndarray,
    nutation_terms: np.ndarray,
) -> tuple[float, float, float, float]:
    centuries_tt = (jd_tt - _J2000) / _JULIAN_CENTURY_DAYS
    centuries_tt2 = centuries_tt * centuries_tt
    centuries_tt3 = centuries_tt2 * centuries_tt

    fundamental_degrees = np.empty(5, dtype=np.float64)
    fundamental_degrees[0] = (
        ((0.064 * centuries_tt + 31.310) * centuries_tt + 1717915922.6330)
        * centuries_tt
        / 3600.0
        + 134.96298139
    )
    fundamental_degrees[1] = (
        ((-0.012 * centuries_tt - 0.577) * centuries_tt + 129596581.2240)
        * centuries_tt
        / 3600.0
        + 357.52772333
    )
    fundamental_degrees[2] = (
        ((0.011 * centuries_tt - 13.257) * centuries_tt + 1739527263.1370)
        * centuries_tt
        / 3600.0
        + 93.27191028
    )
    fundamental_degrees[3] = (
        ((0.019 * centuries_tt - 6.891) * centuries_tt + 1602961601.3280)
        * centuries_tt
        / 3600.0
        + 297.85036306
    )
    fundamental_degrees[4] = (
        ((0.008 * centuries_tt + 7.455) * centuries_tt - 6962890.5390)
        * centuries_tt
        / 3600.0
        + 125.04452222
    )
    fundamental_arguments = np.empty(5, dtype=np.float64)
    for index in range(5):
        fundamental_arguments[index] = math.radians(fundamental_degrees[index] % 360.0)

    delta_psi = 0.0
    delta_epsilon = 0.0
    for row in range(nutation_coefficients.shape[0]):
        phase = 0.0
        for column in range(5):
            phase += nutation_coefficients[row, column] * fundamental_arguments[column]
        delta_psi += (nutation_terms[row, 0] + nutation_terms[row, 1] * centuries_tt) * math.sin(phase)
        delta_epsilon += (nutation_terms[row, 2] + nutation_terms[row, 3] * centuries_tt) * math.cos(phase)
    delta_psi = np.fmod(delta_psi + ddpsi_rad, 2.0 * math.pi)
    delta_epsilon = np.fmod(delta_epsilon + ddeps_rad, 2.0 * math.pi)

    mean_epsilon_degrees = (
        -46.8150 * centuries_tt
        - 0.00059 * centuries_tt2
        + 0.001813 * centuries_tt3
        + 84381.448
    ) / 3600.0
    mean_epsilon = math.radians(mean_epsilon_degrees % 360.0)
    true_epsilon = mean_epsilon + delta_epsilon
    return centuries_tt, delta_psi, true_epsilon, mean_epsilon


@njit_or_identity(cache=True, fastmath=False)
def _gmst_iau1982_split_kernel(jd_utc_start: float, elapsed_s: float, dut1_s: float) -> float:
    # IERS 1996 GMST-UT1; mirror the reference and Rust two-part time policy.
    offset_days = (elapsed_s + dut1_s) / 86400.0
    centuries = ((jd_utc_start - _J2000) + offset_days) / _JULIAN_CENTURY_DAYS
    fractional_seconds = 86400.0 * (np.fmod(jd_utc_start, 1.0) + np.fmod(offset_days, 1.0))
    seconds = (24110.54841 - 43200.0
               + (8640184.812866 + (0.093104 - 6.2e-6 * centuries) * centuries) * centuries
               + fractional_seconds)
    return (seconds * (2.0 * math.pi / 86400.0)) % (2.0 * math.pi)


@njit_or_identity(cache=True, fastmath=False)
def _equation_of_equinoxes_iau1994_kernel(jd_tt: float, dpsi_rad: float, mean_epsilon: float) -> float:
    centuries = (jd_tt - _J2000) / _JULIAN_CENTURY_DAYS
    complementary = 0.0
    if jd_tt >= 2450506.5 + 62.184 / 86400.0:
        omega = math.radians(((((0.008 * centuries + 7.455) * centuries - 6962890.5390)
                              * centuries) / 3600.0 + 125.04452222) % 360.0)
        complementary = (0.00264 * math.sin(omega) + 0.000063 * math.sin(2.0 * omega)) * _ARCSEC_TO_RAD
    return dpsi_rad * math.cos(mean_epsilon) + complementary


@njit_or_identity(cache=True, fastmath=False)
def apparent_sidereal_time_iau76_80_kernel(
    jd_utc: float,
    dut1_s: float,
    dat_s: float,
    ddpsi_rad: float,
    ddeps_rad: float,
    nutation_coefficients: np.ndarray,
    nutation_terms: np.ndarray,
) -> float:
    """Evaluate HPOP-style apparent sidereal time at one UTC epoch."""

    jd_tt = jd_utc + (dat_s + 32.184) / 86400.0
    _centuries_tt, delta_psi, _true_epsilon, mean_epsilon = _iau80_nutation_angles_kernel(
        jd_tt,
        ddpsi_rad,
        ddeps_rad,
        nutation_coefficients,
        nutation_terms,
    )
    gmst = _gmst_iau1982_split_kernel(jd_utc, 0.0, dut1_s)
    return (gmst + _equation_of_equinoxes_iau1994_kernel(jd_tt, delta_psi, mean_epsilon)) % (2.0 * math.pi)


@njit_or_identity(cache=True, fastmath=False)
def eci_to_ecef_iau76_80_kernel(
    t_s: float,
    jd_utc_start: float,
    xp_arcsec: float,
    yp_arcsec: float,
    dut1_s: float,
    dat_s: float,
    ddpsi_rad: float,
    ddeps_rad: float,
    nutation_coefficients: np.ndarray,
    nutation_terms: np.ndarray,
) -> np.ndarray:
    """Evaluate the HPOP-style IAU-76/80 ECI-to-ECEF rotation."""

    jd_utc = jd_utc_start + t_s / 86400.0
    jd_tt = jd_utc + (dat_s + 32.184) / 86400.0
    centuries_tt, delta_psi, true_epsilon, mean_epsilon = _iau80_nutation_angles_kernel(
        jd_tt,
        ddpsi_rad,
        ddeps_rad,
        nutation_coefficients,
        nutation_terms,
    )
    centuries_tt2 = centuries_tt * centuries_tt
    centuries_tt3 = centuries_tt2 * centuries_tt

    zeta = (
        2306.2181 * centuries_tt + 0.30188 * centuries_tt2 + 0.017998 * centuries_tt3
    ) * _ARCSEC_TO_RAD
    theta = (
        2004.3109 * centuries_tt - 0.42665 * centuries_tt2 - 0.041833 * centuries_tt3
    ) * _ARCSEC_TO_RAD
    z = (
        2306.2181 * centuries_tt + 1.09468 * centuries_tt2 + 0.018203 * centuries_tt3
    ) * _ARCSEC_TO_RAD
    precession = _earth_frame_rz(-z) @ _earth_frame_ry(theta) @ _earth_frame_rz(-zeta)
    nutation = (
        _earth_frame_rx(-true_epsilon)
        @ _earth_frame_rz(-delta_psi)
        @ _earth_frame_rx(mean_epsilon)
    )
    # This explicit Vallado rotation is MOD -> TOD (the transpose of the
    # matrix returned by the public TOD -> MOD nutation helper).  Compose it
    # with J2000 -> MOD precession for the J2000 -> TOD celestial factor.
    rbpn = nutation @ precession

    gmst = _gmst_iau1982_split_kernel(jd_utc_start, t_s, dut1_s)
    gast = (gmst + _equation_of_equinoxes_iau1994_kernel(jd_tt, delta_psi, mean_epsilon)) % (2.0 * math.pi)
    sp = -47.0e-6 * ((jd_tt - _J2000) / _JULIAN_CENTURY_DAYS) * _ARCSEC_TO_RAD
    polar_motion = (
        _earth_frame_rz(sp)
        @ _earth_frame_ry(-xp_arcsec * _ARCSEC_TO_RAD)
        @ _earth_frame_rx(-yp_arcsec * _ARCSEC_TO_RAD)
    )
    return polar_motion @ _earth_frame_rz(gast) @ rbpn


@njit_or_identity(cache=True)
def ric_dcm_ir_from_rv_kernel(r_eci_km: np.ndarray, v_eci_km_s: np.ndarray) -> np.ndarray:
    r_norm = np.sqrt(np.dot(r_eci_km, r_eci_km))
    if not np.isfinite(r_norm) or r_norm <= 1e-12:
        raise ValueError("RIC frame is undefined for a zero or non-finite position vector.")
    r_hat = r_eci_km / r_norm
    h = np.empty(3, dtype=np.float64)
    h[0] = r_eci_km[1] * v_eci_km_s[2] - r_eci_km[2] * v_eci_km_s[1]
    h[1] = r_eci_km[2] * v_eci_km_s[0] - r_eci_km[0] * v_eci_km_s[2]
    h[2] = r_eci_km[0] * v_eci_km_s[1] - r_eci_km[1] * v_eci_km_s[0]
    h_norm = np.sqrt(np.dot(h, h))
    if not np.isfinite(h_norm) or h_norm <= 1e-12:
        raise ValueError("RIC frame is undefined for zero angular momentum.")
    c_hat = h / h_norm
    i_hat = np.empty(3, dtype=np.float64)
    i_hat[0] = c_hat[1] * r_hat[2] - c_hat[2] * r_hat[1]
    i_hat[1] = c_hat[2] * r_hat[0] - c_hat[0] * r_hat[2]
    i_hat[2] = c_hat[0] * r_hat[1] - c_hat[1] * r_hat[0]
    i_norm = np.sqrt(np.dot(i_hat, i_hat))
    if not np.isfinite(i_norm) or i_norm <= 1e-12:
        raise ValueError("RIC frame is undefined for a degenerate basis.")
    i_hat = i_hat / i_norm
    out = np.empty((3, 3), dtype=np.float64)
    out[:, 0] = r_hat
    out[:, 1] = i_hat
    out[:, 2] = c_hat
    return out


@njit_or_identity(cache=True)
def ric_angular_rate_eci_from_rv_kernel(r_eci_km: np.ndarray, v_eci_km_s: np.ndarray) -> np.ndarray:
    r2 = np.dot(r_eci_km, r_eci_km)
    out = np.zeros(3, dtype=np.float64)
    if r2 <= 1e-12:
        return out
    out[0] = r_eci_km[1] * v_eci_km_s[2] - r_eci_km[2] * v_eci_km_s[1]
    out[1] = r_eci_km[2] * v_eci_km_s[0] - r_eci_km[0] * v_eci_km_s[2]
    out[2] = r_eci_km[0] * v_eci_km_s[1] - r_eci_km[1] * v_eci_km_s[0]
    return out / r2


@njit_or_identity(cache=True)
def ric_angular_rate_eci_from_rva_kernel(
    r_eci_km: np.ndarray,
    v_eci_km_s: np.ndarray,
    a_eci_km_s2: np.ndarray,
) -> np.ndarray:
    """Return the full instantaneous RIC angular rate when acceleration is known."""

    r_norm = np.sqrt(np.dot(r_eci_km, r_eci_km))
    if not np.isfinite(r_norm) or r_norm <= 1e-12:
        raise ValueError("RIC frame is undefined for a zero or non-finite position vector.")
    h = np.empty(3, dtype=np.float64)
    h[0] = r_eci_km[1] * v_eci_km_s[2] - r_eci_km[2] * v_eci_km_s[1]
    h[1] = r_eci_km[2] * v_eci_km_s[0] - r_eci_km[0] * v_eci_km_s[2]
    h[2] = r_eci_km[0] * v_eci_km_s[1] - r_eci_km[1] * v_eci_km_s[0]
    h_norm = np.sqrt(np.dot(h, h))
    if not np.isfinite(h_norm) or h_norm <= 1e-12:
        raise ValueError("RIC frame is undefined for zero angular momentum.")
    acceleration = np.asarray(a_eci_km_s2, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(acceleration)):
        raise ValueError("RIC angular rate requires a finite acceleration vector.")
    transverse_speed = h_norm / r_norm
    if not np.isfinite(transverse_speed) or transverse_speed <= 1e-12:
        raise ValueError("RIC angular rate is undefined for negligible transverse speed.")
    r_hat = r_eci_km / r_norm
    c_hat = h / h_norm
    out = h / (r_norm * r_norm)
    out += (np.dot(acceleration, c_hat) / transverse_speed) * r_hat
    return out


@njit_or_identity(cache=True)
def ric_rect_state_to_eci_kernel(
    x_rel_ric_rect: np.ndarray,
    r_chief_eci_km: np.ndarray,
    v_chief_eci_km_s: np.ndarray,
) -> np.ndarray:
    c_ir = ric_dcm_ir_from_rv_kernel(r_chief_eci_km, v_chief_eci_km_s)
    omega = ric_angular_rate_eci_from_rv_kernel(r_chief_eci_km, v_chief_eci_km_s)
    dr_eci = c_ir @ x_rel_ric_rect[:3]
    omega_cross_dr = np.empty(3, dtype=np.float64)
    omega_cross_dr[0] = omega[1] * dr_eci[2] - omega[2] * dr_eci[1]
    omega_cross_dr[1] = omega[2] * dr_eci[0] - omega[0] * dr_eci[2]
    omega_cross_dr[2] = omega[0] * dr_eci[1] - omega[1] * dr_eci[0]
    dv_eci = c_ir @ x_rel_ric_rect[3:] + omega_cross_dr
    out = np.empty(6, dtype=np.float64)
    out[:3] = r_chief_eci_km + dr_eci
    out[3:] = v_chief_eci_km_s + dv_eci
    return out


@njit_or_identity(cache=True)
def ric_rect_state_to_eci_rva_kernel(
    x_rel_ric_rect: np.ndarray,
    r_chief_eci_km: np.ndarray,
    v_chief_eci_km_s: np.ndarray,
    a_chief_eci_km_s2: np.ndarray,
) -> np.ndarray:
    c_ir = ric_dcm_ir_from_rv_kernel(r_chief_eci_km, v_chief_eci_km_s)
    omega = ric_angular_rate_eci_from_rva_kernel(
        r_chief_eci_km,
        v_chief_eci_km_s,
        a_chief_eci_km_s2,
    )
    dr_eci = c_ir @ x_rel_ric_rect[:3]
    omega_cross_dr = np.empty(3, dtype=np.float64)
    omega_cross_dr[0] = omega[1] * dr_eci[2] - omega[2] * dr_eci[1]
    omega_cross_dr[1] = omega[2] * dr_eci[0] - omega[0] * dr_eci[2]
    omega_cross_dr[2] = omega[0] * dr_eci[1] - omega[1] * dr_eci[0]
    dv_eci = c_ir @ x_rel_ric_rect[3:] + omega_cross_dr
    out = np.empty(6, dtype=np.float64)
    out[:3] = r_chief_eci_km + dr_eci
    out[3:] = v_chief_eci_km_s + dv_eci
    return out


@njit_or_identity(cache=True)
def eci_relative_to_ric_rect_kernel(x_dep_eci: np.ndarray, x_chief_eci: np.ndarray) -> np.ndarray:
    r_chief = x_chief_eci[:3]
    v_chief = x_chief_eci[3:]
    c_ir = ric_dcm_ir_from_rv_kernel(r_chief, v_chief)
    omega = ric_angular_rate_eci_from_rv_kernel(r_chief, v_chief)
    dr_eci = x_dep_eci[:3] - r_chief
    dv_eci = x_dep_eci[3:] - v_chief
    dr_ric = c_ir.T @ dr_eci
    omega_cross_dr = np.empty(3, dtype=np.float64)
    omega_cross_dr[0] = omega[1] * dr_eci[2] - omega[2] * dr_eci[1]
    omega_cross_dr[1] = omega[2] * dr_eci[0] - omega[0] * dr_eci[2]
    omega_cross_dr[2] = omega[0] * dr_eci[1] - omega[1] * dr_eci[0]
    dv_ric = c_ir.T @ (dv_eci - omega_cross_dr)
    out = np.empty(6, dtype=np.float64)
    out[:3] = dr_ric
    out[3:] = dv_ric
    return out


@njit_or_identity(cache=True)
def eci_relative_to_ric_rect_rva_kernel(
    x_dep_eci: np.ndarray,
    x_chief_eci: np.ndarray,
    a_chief_eci_km_s2: np.ndarray,
) -> np.ndarray:
    r_chief = x_chief_eci[:3]
    v_chief = x_chief_eci[3:]
    c_ir = ric_dcm_ir_from_rv_kernel(r_chief, v_chief)
    omega = ric_angular_rate_eci_from_rva_kernel(r_chief, v_chief, a_chief_eci_km_s2)
    dr_eci = x_dep_eci[:3] - r_chief
    dv_eci = x_dep_eci[3:] - v_chief
    dr_ric = c_ir.T @ dr_eci
    omega_cross_dr = np.empty(3, dtype=np.float64)
    omega_cross_dr[0] = omega[1] * dr_eci[2] - omega[2] * dr_eci[1]
    omega_cross_dr[1] = omega[2] * dr_eci[0] - omega[0] * dr_eci[2]
    omega_cross_dr[2] = omega[0] * dr_eci[1] - omega[1] * dr_eci[0]
    dv_ric = c_ir.T @ (dv_eci - omega_cross_dr)
    out = np.empty(6, dtype=np.float64)
    out[:3] = dr_ric
    out[3:] = dv_ric
    return out


@njit_or_identity(cache=True)
def ric_curv_to_rect_kernel(x_ric_curv: np.ndarray, r0_km: float, eps: float = 1e-12) -> np.ndarray:
    x_r_curv = x_ric_curv[0]
    x_i_curv = x_ric_curv[1]
    x_c_curv = x_ric_curv[2]
    x_r_curv_dot = x_ric_curv[3]
    x_i_curv_dot = x_ric_curv[4]
    x_c_curv_dot = x_ric_curv[5]
    r0 = max(r0_km, eps)
    r = max(r0 + x_r_curv, eps)
    theta_i = x_i_curv / r0
    theta_c = x_c_curv / r0
    c_i = np.cos(theta_i)
    s_i = np.sin(theta_i)
    c_c = np.cos(theta_c)
    s_c = np.sin(theta_c)
    x = r * c_c * c_i
    y = r * c_c * s_i
    z = r * s_c
    r_dot = x_r_curv_dot
    theta_i_dot = x_i_curv_dot / r0
    theta_c_dot = x_c_curv_dot / r0
    out = np.empty(6, dtype=np.float64)
    out[0] = x - r0
    out[1] = y
    out[2] = z
    out[3] = r_dot * c_c * c_i - r * s_c * theta_c_dot * c_i - r * c_c * s_i * theta_i_dot
    out[4] = r_dot * c_c * s_i - r * s_c * theta_c_dot * s_i + r * c_c * c_i * theta_i_dot
    out[5] = r_dot * s_c + r * c_c * theta_c_dot
    return out


@njit_or_identity(cache=True)
def ric_rect_to_curv_kernel(x_ric_rect: np.ndarray, r0_km: float, eps: float = 1e-12) -> np.ndarray:
    x_r = x_ric_rect[0]
    x_i = x_ric_rect[1]
    x_c = x_ric_rect[2]
    x_rdot = x_ric_rect[3]
    x_idot = x_ric_rect[4]
    x_cdot = x_ric_rect[5]
    r0 = max(r0_km, eps)
    x = r0 + x_r
    y = x_i
    z = x_c
    r = max(np.sqrt(x * x + y * y + z * z), eps)
    p2 = x * x + y * y
    p = np.sqrt(max(p2, eps))
    theta_i = np.arctan2(y, x)
    theta_c = np.arctan2(z, p)
    r_dot = (x * x_rdot + y * x_idot + z * x_cdot) / r
    theta_i_dot = (x * x_idot - y * x_rdot) / max(p2, eps)
    p_dot = (x * x_rdot + y * x_idot) / p
    theta_c_dot = (p * x_cdot - z * p_dot) / (r * r)
    out = np.empty(6, dtype=np.float64)
    out[0] = r - r0
    out[1] = r0 * theta_i
    out[2] = r0 * theta_c
    out[3] = r_dot
    out[4] = r0 * theta_i_dot
    out[5] = r0 * theta_c_dot
    return out
