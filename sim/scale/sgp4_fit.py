from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

import numpy as np

from sim.dynamics.orbit.elements import rv_to_coe_eci
from sim.dynamics.orbit.environment import EARTH_MU_KM3_S2
from sim.dynamics.orbit.sgp4 import sgp4_propagate_teme, sgp4_unsupported_reason
from sim.dynamics.orbit.tle import TLEElements
from sim.estimation.batch_least_squares import solve_batch_least_squares
from sim.estimation.parameters import EstimatedParameter, ParameterSet


def fit_state_to_sgp4_mean_elements(
    *,
    position_teme_km: np.ndarray,
    velocity_teme_km_s: np.ndarray,
    epoch_jd_utc: float,
    template_elements: TLEElements,
    arc_duration_s: float = 1800.0,
    step_s: float = 120.0,
    position_sigma_km: float = 0.001,
    velocity_sigma_km_s: float = 1.0e-6,
    max_nfev: int = 80,
    acceptance_position_rms_m: float = 1000.0,
    acceptance_velocity_rms_mm_s: float = 3000.0,
) -> dict[str, Any]:
    """Fit fixed-epoch near-Earth SGP4 mean elements to a short TEME state arc.

    This produces a synthetic SGP4-compatible mean-element product. It is not a
    catalog TLE and should remain labeled as analyst/trial generated.
    """

    r0 = np.array(position_teme_km, dtype=float).reshape(3)
    v0 = np.array(velocity_teme_km_s, dtype=float).reshape(3)
    epoch = float(epoch_jd_utc)
    times_s = _fit_times(float(arc_duration_s), float(step_s))
    truth_pos, truth_vel = _two_body_reference_arc(r0, v0, times_s)
    seed_elements = _initial_elements_from_state(
        r0,
        v0,
        epoch_jd_utc=epoch,
        template_elements=template_elements,
    )
    unsupported = sgp4_unsupported_reason(seed_elements)
    if unsupported:
        raise ValueError(unsupported)
    seed_vector = _element_vector(seed_elements)
    parameters = ParameterSet(
        [
            EstimatedParameter("inclination_deg", seed_vector[0], scale=1.0, lower=0.0, upper=180.0),
            EstimatedParameter("raan_deg", seed_vector[1], scale=1.0, lower=0.0, upper=360.0),
            EstimatedParameter("eccentricity", seed_vector[2], scale=0.01, lower=1.0e-8, upper=0.999),
            EstimatedParameter("argp_deg", seed_vector[3], scale=1.0, lower=0.0, upper=360.0),
            EstimatedParameter("mean_anomaly_deg", seed_vector[4], scale=1.0, lower=0.0, upper=360.0),
            EstimatedParameter(
                "mean_motion_rev_per_day",
                seed_vector[5],
                scale=0.01,
                lower=1440.0 / 225.0,
                upper=18.0,
            ),
        ]
    )

    def residual(vector: np.ndarray) -> np.ndarray:
        elements = _elements_from_vector(seed_elements, vector)
        try:
            pos_err, vel_err = _state_errors(elements, times_s, epoch, truth_pos, truth_vel)
        except ValueError:
            return np.full(times_s.size * 6, 1.0e9, dtype=float)
        return np.hstack(
            (
                pos_err / float(position_sigma_km),
                vel_err / float(velocity_sigma_km_s),
            )
        ).reshape(-1)

    seed_pos_err, seed_vel_err = _state_errors(seed_elements, times_s, epoch, truth_pos, truth_vel)
    solve = solve_batch_least_squares(
        parameters,
        residual,
        max_nfev=int(max_nfev),
        xtol=1.0e-12,
        ftol=1.0e-12,
        gtol=1.0e-12,
    )
    estimated_vector = np.array(solve.x_native, dtype=float)
    estimated_elements = _elements_from_vector(seed_elements, estimated_vector)
    final_pos_err, final_vel_err = _state_errors(estimated_elements, times_s, epoch, truth_pos, truth_vel)
    initial_metrics = _state_error_metrics(seed_pos_err, seed_vel_err)
    final_metrics = _state_error_metrics(final_pos_err, final_vel_err)
    accepted_by_residual_gate = _fit_residual_gate_passed(
        initial_metrics=initial_metrics,
        final_metrics=final_metrics,
        acceptance_position_rms_m=float(acceptance_position_rms_m),
        acceptance_velocity_rms_mm_s=float(acceptance_velocity_rms_mm_s),
    )
    fit_completed = bool(solve.success) or accepted_by_residual_gate
    return {
        "method": "synthetic_state_to_sgp4_mean_elements_fit",
        "status": "completed" if fit_completed else "fit_not_converged",
        "reference_arc": {
            "model": "two_body_teme_assumed_inertial",
            "duration_s": float(arc_duration_s),
            "step_s": float(step_s),
            "sample_count": int(times_s.size),
        },
        "fixed_parameters": [
            "epoch_jd_utc",
            "mean_motion_derivative_rev_per_day2",
            "mean_motion_second_derivative_rev_per_day3",
            "bstar",
        ],
        "fit_parameters": parameters.names,
        "epoch_jd_utc": epoch,
        "position_sigma_km": float(position_sigma_km),
        "velocity_sigma_km_s": float(velocity_sigma_km_s),
        "seed_elements": _elements_payload(seed_elements),
        "estimated_elements": _elements_payload(estimated_elements),
        "initial_state_error": initial_metrics,
        "final_state_error": final_metrics,
        "final_normalized_residual_rms": float(solve.rms_residual),
        "solver": {
            "success": bool(solve.success),
            "accepted_by_residual_gate": bool(accepted_by_residual_gate),
            "message": solve.message,
            "nfev": int(solve.nfev),
            "initial_cost": float(solve.initial_cost),
            "final_cost": float(solve.cost),
        },
        "non_claims": [
            "This is a synthetic SGP4-compatible mean-element fit, not an observed catalog TLE.",
            "The reference arc is deterministic two-body TEME-treated-as-inertial for initial product plumbing.",
            "B*, epoch, and mean-motion derivative terms are inherited/fixed, not estimated.",
        ],
    }


def _initial_elements_from_state(
    position_teme_km: np.ndarray,
    velocity_teme_km_s: np.ndarray,
    *,
    epoch_jd_utc: float,
    template_elements: TLEElements,
) -> TLEElements:
    coe = rv_to_coe_eci(position_teme_km, velocity_teme_km_s)
    mean_anomaly_deg = _true_to_mean_anomaly_deg(coe.true_anomaly_deg, coe.ecc)
    mean_motion_rev_day = _mean_motion_rev_day(coe.a_km)
    return replace(
        template_elements,
        epoch_jd_utc=float(epoch_jd_utc),
        inclination_deg=float(coe.inc_deg),
        raan_deg=float(coe.raan_deg % 360.0),
        eccentricity=float(np.clip(coe.ecc, 1.0e-8, 0.999)),
        argp_deg=float(coe.argp_deg % 360.0),
        mean_anomaly_deg=float(mean_anomaly_deg % 360.0),
        mean_motion_rev_per_day=float(mean_motion_rev_day),
    )


def _elements_from_vector(template: TLEElements, vector: np.ndarray) -> TLEElements:
    x = np.array(vector, dtype=float).reshape(6)
    return replace(
        template,
        inclination_deg=float(x[0]),
        raan_deg=float(x[1] % 360.0),
        eccentricity=float(np.clip(x[2], 1.0e-8, 0.999)),
        argp_deg=float(x[3] % 360.0),
        mean_anomaly_deg=float(x[4] % 360.0),
        mean_motion_rev_per_day=float(x[5]),
    )


def _element_vector(elements: TLEElements) -> np.ndarray:
    return np.array(
        [
            float(elements.inclination_deg),
            float(elements.raan_deg),
            float(elements.eccentricity),
            float(elements.argp_deg),
            float(elements.mean_anomaly_deg),
            float(elements.mean_motion_rev_per_day),
        ],
        dtype=float,
    )


def _state_errors(
    elements: TLEElements,
    times_s: np.ndarray,
    epoch_jd_utc: float,
    truth_pos_km: np.ndarray,
    truth_vel_km_s: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    pos_err = []
    vel_err = []
    for t_s, truth_pos, truth_vel in zip(times_s, truth_pos_km, truth_vel_km_s, strict=True):
        jd = float(epoch_jd_utc) + float(t_s) / 86400.0
        tsince_min = (jd - float(elements.epoch_jd_utc)) * 1440.0
        state = sgp4_propagate_teme(elements, tsince_min)
        if state.error:
            raise ValueError(state.error)
        pos_err.append(np.array(state.position_teme_km, dtype=float) - truth_pos)
        vel_err.append(np.array(state.velocity_teme_km_s, dtype=float) - truth_vel)
    return np.vstack(pos_err), np.vstack(vel_err)


def _two_body_reference_arc(r0_km: np.ndarray, v0_km_s: np.ndarray, times_s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r = np.array(r0_km, dtype=float).reshape(3)
    v = np.array(v0_km_s, dtype=float).reshape(3)
    positions = [r.copy()]
    velocities = [v.copy()]
    previous_t = 0.0
    for t_s in np.array(times_s[1:], dtype=float):
        dt_total = float(t_s - previous_t)
        substeps = max(1, int(math.ceil(abs(dt_total) / 30.0)))
        dt = dt_total / float(substeps)
        for _ in range(substeps):
            r, v = _rk4_two_body_step(r, v, dt)
        positions.append(r.copy())
        velocities.append(v.copy())
        previous_t = float(t_s)
    return np.vstack(positions), np.vstack(velocities)


def _rk4_two_body_step(r: np.ndarray, v: np.ndarray, dt_s: float) -> tuple[np.ndarray, np.ndarray]:
    y = np.hstack((r, v))

    def f(state: np.ndarray) -> np.ndarray:
        rr = state[:3]
        vv = state[3:]
        rn = max(float(np.linalg.norm(rr)), 1.0e-12)
        acc = -EARTH_MU_KM3_S2 * rr / (rn**3)
        return np.hstack((vv, acc))

    dt = float(dt_s)
    k1 = f(y)
    k2 = f(y + 0.5 * dt * k1)
    k3 = f(y + 0.5 * dt * k2)
    k4 = f(y + dt * k3)
    out = y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return out[:3], out[3:]


def _fit_times(duration_s: float, step_s: float) -> np.ndarray:
    duration = max(float(duration_s), 0.0)
    step = max(float(step_s), 1.0)
    count = int(math.floor(duration / step)) + 1
    times = np.arange(count, dtype=float) * step
    if times[-1] < duration - 1.0e-9:
        times = np.append(times, duration)
    return times


def _true_to_mean_anomaly_deg(true_anomaly_deg: float, eccentricity: float) -> float:
    e = float(eccentricity)
    nu = math.radians(float(true_anomaly_deg))
    sin_e = math.sqrt(max(1.0 - e * e, 0.0)) * math.sin(nu) / (1.0 + e * math.cos(nu))
    cos_e = (e + math.cos(nu)) / (1.0 + e * math.cos(nu))
    ecc_anomaly = math.atan2(sin_e, cos_e)
    mean_anomaly = ecc_anomaly - e * math.sin(ecc_anomaly)
    return float(math.degrees(mean_anomaly) % 360.0)


def _mean_motion_rev_day(a_km: float) -> float:
    n_rad_s = math.sqrt(EARTH_MU_KM3_S2 / (float(a_km) ** 3))
    return float(n_rad_s * 86400.0 / (2.0 * math.pi))


def _state_error_metrics(pos_err_km: np.ndarray, vel_err_km_s: np.ndarray) -> dict[str, float]:
    pos_norm_m = np.linalg.norm(np.array(pos_err_km, dtype=float), axis=1) * 1000.0
    vel_norm_mm_s = np.linalg.norm(np.array(vel_err_km_s, dtype=float), axis=1) * 1.0e6
    return {
        "position_rms_m": float(np.sqrt(np.mean(pos_norm_m * pos_norm_m))),
        "position_max_m": float(np.max(pos_norm_m)),
        "velocity_rms_mm_s": float(np.sqrt(np.mean(vel_norm_mm_s * vel_norm_mm_s))),
        "velocity_max_mm_s": float(np.max(vel_norm_mm_s)),
    }


def _fit_residual_gate_passed(
    *,
    initial_metrics: dict[str, float],
    final_metrics: dict[str, float],
    acceptance_position_rms_m: float,
    acceptance_velocity_rms_mm_s: float,
) -> bool:
    position_rms = float(final_metrics["position_rms_m"])
    velocity_rms = float(final_metrics["velocity_rms_mm_s"])
    if not (np.isfinite(position_rms) and np.isfinite(velocity_rms)):
        return False
    if position_rms > float(acceptance_position_rms_m):
        return False
    if velocity_rms > float(acceptance_velocity_rms_mm_s):
        return False
    return (
        position_rms < float(initial_metrics["position_rms_m"])
        and velocity_rms < float(initial_metrics["velocity_rms_mm_s"])
    )


def _elements_payload(elements: TLEElements) -> dict[str, float]:
    return {
        "epoch_jd_utc": float(elements.epoch_jd_utc),
        "inclination_deg": float(elements.inclination_deg),
        "raan_deg": float(elements.raan_deg),
        "eccentricity": float(elements.eccentricity),
        "argp_deg": float(elements.argp_deg),
        "mean_anomaly_deg": float(elements.mean_anomaly_deg),
        "mean_motion_rev_per_day": float(elements.mean_motion_rev_per_day),
        "mean_motion_derivative_rev_per_day2": float(elements.mean_motion_derivative_rev_per_day2),
        "mean_motion_second_derivative_rev_per_day3": float(elements.mean_motion_second_derivative_rev_per_day3),
        "bstar": float(elements.bstar),
    }
