"""Joint state/force and consider-parameter forecast uncertainty.

Propagation callbacks must use OEL's maintained deterministic engine. This
owner differentiates that engine; it contains no alternative dynamics model.
All covariance and sensitivity coordinates use the caller's explicit native
parameter order; output states are Cartesian ECI km and km/s.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from sim.estimation.covariance import validate_covariance


def propagate_joint_covariance(
    evaluate: Callable[[np.ndarray, np.ndarray], np.ndarray],
    *,
    parameter_values: Sequence[float],
    parameter_covariance: np.ndarray,
    parameter_steps: Sequence[float],
    times_s: Sequence[float],
    parameter_names: Sequence[str],
    consider_values: Sequence[float] = (),
    consider_covariance: np.ndarray | None = None,
    consider_steps: Sequence[float] = (),
    consider_names: Sequence[str] = (),
    parameter_consider_cross_covariance: np.ndarray | None = None,
    process_noise_covariances: np.ndarray | None = None,
) -> dict[str, Any]:
    """Push a complete joint covariance through engine sensitivities.

    Consider variables are uncertain but not fitted. Their cross-covariance
    with fitted parameters must be included when estimation couples them.
    ``process_noise_covariances`` are additive state covariances already mapped
    to each requested epoch; they must not duplicate parameter/consider terms.
    No process noise or calibration is silently inferred from residuals.
    """
    values = np.asarray(parameter_values, dtype=float).reshape(-1)
    considers = np.asarray(consider_values, dtype=float).reshape(-1)
    times = np.asarray(times_s, dtype=float).reshape(-1)
    n, c = len(values), len(considers)
    if not n or len(parameter_names) != n or len(set(parameter_names)) != n:
        raise ValueError("parameter_names must uniquely identify every parameter")
    if len(consider_names) != c or len(set(consider_names)) != c or set(parameter_names) & set(consider_names):
        raise ValueError("consider names must be unique and disjoint from fitted parameters")
    if (
        not len(times)
        or not np.all(np.isfinite(times))
        or np.any(times < 0)
        or np.any(np.diff(times) < 0)
        or not np.all(np.isfinite(values))
        or not np.all(np.isfinite(considers))
    ):
        raise ValueError("values and ordered nonnegative forecast epochs must be finite")
    steps = np.asarray(parameter_steps, dtype=float).reshape(-1)
    csteps = np.asarray(consider_steps, dtype=float).reshape(-1)
    if (
        steps.size != n
        or csteps.size != c
        or np.any(~np.isfinite(np.r_[steps, csteps]))
        or np.any(np.r_[steps, csteps] <= 0)
    ):
        raise ValueError("one positive finite sensitivity step is required per joint parameter")
    p = validate_covariance(parameter_covariance, n)
    if not c and (consider_covariance is not None or parameter_consider_cross_covariance is not None):
        raise ValueError("consider covariance requires consider variables")
    cc = validate_covariance(consider_covariance, c, name="consider covariance") if c else np.empty((0, 0))
    cross = (
        np.zeros((n, c))
        if parameter_consider_cross_covariance is None
        else np.asarray(parameter_consider_cross_covariance, dtype=float)
    )
    if cross.shape != (n, c):
        raise ValueError("parameter/consider cross covariance has the wrong shape")
    joint = validate_covariance(np.block([[p, cross], [cross.T, cc]]), n + c, name="joint covariance")

    def state(x, z):
        result = np.asarray(evaluate(x, z), dtype=float)
        if result.shape != (len(times), 6) or not np.all(np.isfinite(result)):
            raise ValueError("engine evaluation must return finite (epochs, 6) Cartesian states")
        return result

    nominal = state(values, considers)
    sensitivity = np.empty((len(times), 6, n + c))
    for index, step in enumerate(np.r_[steps, csteps]):
        xp, xm, cp, cm = values.copy(), values.copy(), considers.copy(), considers.copy()
        if index < n:
            xp[index] += step
            xm[index] -= step
        else:
            cp[index - n] += step
            cm[index - n] -= step
        sensitivity[:, :, index] = (state(xp, cp) - state(xm, cm)) / (2 * step)
    q = (
        np.zeros((len(times), 6, 6))
        if process_noise_covariances is None
        else np.asarray(process_noise_covariances, dtype=float)
    )
    if q.shape != (len(times), 6, 6):
        raise ValueError("process covariance must have shape (epochs, 6, 6)")
    records = []
    for i, (t, g, noise) in enumerate(zip(times, sensitivity, q, strict=True)):
        validate_covariance(noise, 6, name="process covariance")
        covariance = g @ joint @ g.T + noise
        covariance = 0.5 * (covariance + covariance.T)
        records.append(
            {
                "time_s": float(t),
                "state_eci_km_km_s": nominal[i].tolist(),
                "covariance_eci_km_km_s": covariance.tolist(),
                "sensitivity_native": g.tolist(),
            }
        )
    return {
        "schema_version": 1,
        "status": "formal_linearized_forecast",
        "frame": "ECI",
        "state_units": ["km"] * 3 + ["km/s"] * 3,
        "parameter_names": list(parameter_names),
        "consider_names": list(consider_names),
        "parameter_covariance": p.tolist(),
        "consider_covariance": cc.tolist(),
        "parameter_consider_cross_covariance": cross.tolist(),
        "parameter_cross_correlations_included": True,
        "process_noise_supplied": process_noise_covariances is not None,
        "empirically_calibrated": False,
        "records": records,
    }


def consider_parameter_covariance(
    data_jacobian: np.ndarray, nuisance_residual_jacobian: np.ndarray, nuisance_covariance: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Estimator response to uncertain fixed inputs, using a whitened data Jacobian.

    Returns additional fitted-parameter covariance and parameter/consider cross
    covariance. Add the former to the conditional fitted covariance before
    joint propagation. Residual derivatives include the direct nuisance effect;
    the minus sign captures the compensating fitted-parameter response.
    """
    a = np.asarray(data_jacobian, dtype=float)
    b = np.asarray(nuisance_residual_jacobian, dtype=float)
    if (
        a.ndim != 2
        or b.ndim != 2
        or a.shape[0] != b.shape[0]
        or not np.all(np.isfinite(a))
        or not np.all(np.isfinite(b))
    ):
        raise ValueError("consider residual Jacobians must be finite with equal row counts")
    u, s, vt = np.linalg.svd(a, full_matrices=False)
    if len(s) < a.shape[1] or np.any(s <= max(a.shape) * np.finfo(float).eps * s[0]):
        raise ValueError("consider response requires an identifiable fitted parameter set")
    c = validate_covariance(nuisance_covariance, b.shape[1], name="consider covariance")
    response = -(vt.T / s) @ u.T @ b
    cross = response @ c
    return cross @ response.T, cross


def transport_process_covariance(times_s, state_transitions, *, start_time_s, local_covariance):
    """Transport discrete process increments with the configured ONP sensitivities.

    F(t) maps initial Cartesian km/km-s perturbations to each epoch. Increments
    use the declared local discrete covariance at each forecast interval end.
    This is an explicit discretization policy, not an exact continuous-noise
    integral for arbitrary nonlinear orbital dynamics.
    """
    times = np.asarray(times_s, dtype=float).reshape(-1)
    transitions = np.asarray(state_transitions, dtype=float)
    if (
        not len(times)
        or transitions.shape != (len(times), 6, 6)
        or not np.all(np.isfinite(times))
        or np.any(np.diff(times) < 0)
        or not np.all(np.isfinite(transitions))
        or not np.isfinite(start_time_s)
    ):
        raise ValueError("process transport requires ordered epochs and finite (epochs,6,6) transitions")
    indices = np.flatnonzero(times == float(start_time_s))
    if not len(indices):
        raise ValueError("process transport must start at an explicit last-assimilated epoch")
    previous = int(indices[-1])
    covariance = np.zeros((6, 6))
    records = np.zeros((len(times), 6, 6))
    for i in range(previous + 1, len(times)):
        transition = np.linalg.solve(transitions[previous].T, transitions[i].T).T
        local = validate_covariance(
            local_covariance(float(times[i] - times[previous])), 6, name="local process covariance"
        )
        covariance = transition @ covariance @ transition.T + local
        covariance = 0.5 * (covariance + covariance.T)
        validate_covariance(covariance, 6, name="transported process covariance")
        records[i] = covariance
        previous = i
    return records
