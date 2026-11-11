"""Explicit joint process conditioning for formal precision OD uncertainty.

Callers construct sensitivities and process kernels through maintained dynamics
and measurement owners. This module supplies Gaussian algebra, not dynamics.
Training and forecast process errors are coupled; adding an independent Q to a
deterministic fitted covariance would lose that coupling.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import solve_triangular
from scipy.optimize import minimize_scalar

from sim.estimation.covariance import validate_covariance
from sim.estimation.sequential_od import SequentialProcessNoise


def mapped_acceleration_process_kernel(*, times_s, state_transitions,
        measurement_indices, measurement_jacobians, spectral_density):
    """Transport declared white-acceleration increments through native STMs.

    The grid starts at the fit epoch (zero) and includes every measurement.
    Local increments use the maintained sequential OD discretization at each
    interval end. Refine the grid before use: this is not an exact nonlinear
    continuous-noise integral. Output rows retain their shared process errors.
    """
    t, f = np.asarray(times_s, dtype=float), np.asarray(state_transitions, dtype=float)
    indices = np.asarray(measurement_indices)
    h = np.asarray(measurement_jacobians, dtype=float)
    if (t.ndim != 1 or len(t) < 2 or t[0] != 0 or np.any(np.diff(t) <= 0)
            or not np.all(np.isfinite(t)) or len(t) > 20000
            or f.shape != (len(t), 6, 6) or not np.all(np.isfinite(f))
            or indices.ndim != 1 or not 1 <= len(indices) <= 4096
            or np.any(~np.isfinite(indices)) or np.any(indices != np.floor(indices))
            or np.any(indices < 0) or np.any(indices >= len(t))
            or h.shape != (len(indices), 6) or not np.all(np.isfinite(h))):
        raise ValueError("Mapped process kernel requires an ordered native STM grid and explicit measurement rows")
    policy = SequentialProcessNoise(spectral_density=tuple(spectral_density))
    indices = indices.astype(int)
    wanted = set(indices)
    integrals, cumulative = {}, np.zeros((6, 6))
    if 0 in wanted:
        integrals[0] = cumulative.copy()
    for i in range(1, len(t)):
        inverse = np.linalg.solve(f[i], np.eye(6))
        cumulative += inverse@policy.covariance(t[i]-t[i-1])@inverse.T
        if i in wanted:
            integrals[i] = cumulative.copy()
    rows = np.einsum("ni,nij->nj", h, f[indices])
    # A bilinear form is symmetric analytically, but swapping its operands
    # changes floating-point rounding. Choose their order by measurement
    # content so permuting observations only permutes the resulting kernel.
    keys = [(int(index), tuple(row)) for index, row in zip(indices, rows, strict=True)]
    result = np.empty((len(indices), len(indices)))
    for i in range(len(indices)):
        for j in range(i, len(indices)):
            left, right = (i, j) if keys[i] <= keys[j] else (j, i)
            value = rows[left]@integrals[min(indices[i], indices[j])]@rows[right]
            result[i, j] = result[j, i] = value
    validate_covariance(result, len(indices), name="mapped acceleration process kernel")
    return result


def mapped_acceleration_process_factor(*, times_s, state_transitions,
        measurement_indices, measurement_jacobians, spectral_density):
    """Factor the same transported increments without cumulative cancellation.

    Columns are independent six-coordinate interval-end increments. This is
    the same discretization as mapped_acceleration_process_kernel, with no
    eigenvalue clipping, diagonal loading or inferred process scale.
    """
    # Preserve the existing input and physical-kernel validation contract.
    mapped_acceleration_process_kernel(times_s=times_s, state_transitions=state_transitions,
        measurement_indices=measurement_indices, measurement_jacobians=measurement_jacobians,
        spectral_density=spectral_density)
    t = np.asarray(times_s, dtype=float)
    f = np.asarray(state_transitions, dtype=float)
    indices = np.asarray(measurement_indices, dtype=int)
    h = np.asarray(measurement_jacobians, dtype=float)
    columns = 6*(len(t)-1)
    if len(indices)*columns > 64_000_000:
        raise ValueError('Mapped process factor exceeds the explicit 512 MB matrix budget')
    result = np.zeros((len(indices), columns))
    rows = np.einsum('ni,nij->nj', h, f[indices])
    policy = SequentialProcessNoise(spectral_density=tuple(spectral_density))
    for i in range(1, len(t)):
        local = policy.covariance(t[i]-t[i-1])
        positive = np.diag(local) > 0
        root = np.zeros((6, 6))
        if np.any(positive):
            scale = np.sqrt(np.diag(local)[positive])
            normalized = local[np.ix_(positive, positive)]/np.outer(scale, scale)
            root[np.ix_(positive, positive)] = scale[:, None]*np.linalg.cholesky(normalized)
        selected = indices >= i
        result[selected, 6*(i-1):6*i] = rows[selected]@np.linalg.solve(f[i], root)
    return result


def _training(design, residual, observation_covariance):
    x, y = np.asarray(design, dtype=float), np.asarray(residual, dtype=float)
    if (x.ndim != 2 or y.shape != (len(x),) or x.shape[1] < 1
            or len(x) < x.shape[1]+8 or len(x) > 4096
            or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y))):
        raise ValueError("Process conditioning requires finite observable training data and eight residual degrees of freedom")
    r = validate_covariance(observation_covariance, len(x), name="training observation covariance")
    scales = np.linalg.norm(x, axis=0)
    if np.any(scales == 0) or np.linalg.matrix_rank(x/scales) != x.shape[1]:
        raise ValueError("Process conditioning requires identifiable fixed effects")
    return x, y, r, scales


def _gls(x, y, covariance, scales):
    try:
        factor = np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError as exc:
        raise ValueError("Training process plus observation covariance must be positive definite without jitter") from exc
    a = solve_triangular(factor, x/scales, lower=True)
    b = solve_triangular(factor, y, lower=True)
    u, s, vt = np.linalg.svd(a, full_matrices=False)
    if np.any(s <= max(a.shape)*np.finfo(float).eps*s[0]):
        raise ValueError("Whitened process design has a fixed-effect nullspace")
    normalized_covariance = (vt.T/s**2)@vt
    covariance_parameters = normalized_covariance/np.outer(scales, scales)
    delta = (vt.T@((u.T@b)/s))/scales
    innovation = y-x@delta
    z = solve_triangular(factor, innovation, lower=True)
    restricted = float(2*np.log(np.diag(factor)).sum()+z@z+2*np.log(s).sum())
    return factor, delta, covariance_parameters, innovation, restricted


def condition_precision_process(*, training_design, training_residual,
        forecast_design, training_observation_covariance,
        joint_process_covariance, forecast_observation_covariance,
        training_forecast_observation_cross_covariance, joint_process_factor=None):
    """Universal Gaussian conditioning with an uninformative fixed-effect prior.

    Designs differentiate predicted coordinates, residuals are observed minus
    nominal, and the process matrix orders training then forecast coordinates.
    All matrices use the same native units as their coordinate rows. The caller
    must explicitly supply observation cross covariance, including an explicit
    zero matrix when independence is supported. Returns an additive forecast
    mean correction and full joint uncertainty, not an empirical calibration.
    """
    x, y, r, scales = _training(training_design, training_residual, training_observation_covariance)
    f = np.asarray(forecast_design, dtype=float)
    if f.ndim != 2 or f.shape[1] != x.shape[1] or not len(f) or not np.all(np.isfinite(f)):
        raise ValueError("Forecast design must match every fitted coordinate")
    n, m = len(x), len(f)
    if n+m > 4096:
        raise ValueError("Joint process conditioning is bounded to 4096 coordinates")
    q = validate_covariance(joint_process_covariance, n+m, name="joint training/forecast process covariance")
    rf = validate_covariance(forecast_observation_covariance, m, name="forecast observation covariance")
    cross = np.asarray(training_forecast_observation_cross_covariance, dtype=float)
    if cross.shape != (n, m) or not np.all(np.isfinite(cross)):
        raise ValueError("Explicit observation cross covariance must have training by forecast shape")
    validate_covariance(np.block([[r, cross], [cross.T, rf]]), n+m, name="joint observation covariance")
    v = r+q[:n, :n]
    factor, delta, p, innovation, objective = _gls(x, y, v, scales)
    c = q[n:, :n]+cross.T
    # W = C V^-1 via triangular solves; neither normals nor V are inverted.
    w = solve_triangular(factor.T, solve_triangular(factor, c.T, lower=True), lower=False).T
    a = f-w@x
    if joint_process_factor is None:
        covariance = q[n:, n:]+rf-w@c.T+a@p@a.T
    else:
        root = np.asarray(joint_process_factor, dtype=float)
        if (root.ndim != 2 or root.shape[0] != n+m or not root.shape[1]
                or root.size > 64_000_000 or not np.all(np.isfinite(root))):
            raise ValueError('Explicit joint process factor has invalid shape or exceeds matrix budget')
        if np.any(cross):
            raise ValueError('Factored conditioning requires explicitly independent sensor noise')
        reconstruction = root@root.T
        if np.linalg.norm(reconstruction-q) > 1e-7*np.linalg.norm(q):
            raise ValueError('Process factor does not reconstruct the declared physical kernel')
        # Joseph/square-root form: each contribution is a positive Gram matrix.
        # Keep the same gain, fixed effects and mean. No covariance is repaired.
        process_root = root[n:]-w@root[:n]
        observation_root = w@np.linalg.cholesky(r)
        whitened = solve_triangular(factor, x/scales, lower=True)
        _, singular, vt = np.linalg.svd(whitened, full_matrices=False)
        parameter_root = a@((vt.T/singular)/scales[:, None])
        terms = [process_root, observation_root, parameter_root]
        if np.any(rf):
            terms.append(np.linalg.cholesky(rf))
        combined = np.hstack(terms)
        covariance = combined@combined.T
    covariance = (covariance+covariance.T)*.5
    validate_covariance(covariance, m, name="conditioned forecast covariance")
    return dict(schema="oel.precision_process_conditioning.v1",
        fixed_effect_delta=delta.tolist(), fixed_effect_covariance=p.tolist(),
        forecast_mean_correction=(f@delta+w@innovation).tolist(),
        forecast_covariance=covariance.tolist(), restricted_objective=objective,
        covariance_evaluation="subtractive_schur" if joint_process_factor is None else "factored_joseph",
        training_process_cross_terms_retained=True, observation_cross_terms_explicit=True,
        hyperparameter_uncertainty_marginalized=False, empirically_calibrated=False)


def estimate_precision_process_scale(*, training_design, training_residual,
        training_observation_covariance, unit_process_covariance, scale_bounds):
    """Development-only REML/BIC for one source-declared process kernel.

    Both bounds and kernel units are caller declarations, frozen before future
    data. An explicit zero-process candidate competes against the fitted scale.
    No observation-noise bound, residual rejection or covariance rescaling is
    introduced. The selected scale still needs an independent frozen test.
    """
    x, y, r, scales = _training(training_design, training_residual, training_observation_covariance)
    k = validate_covariance(unit_process_covariance, len(x), name="unit process kernel")
    bounds = np.asarray(scale_bounds, dtype=float)
    if bounds.shape != (2,) or not np.all(np.isfinite(bounds)) or not 0 < bounds[0] < bounds[1]:
        raise ValueError("Explicit positive finite ordered process-scale bounds are required")
    if not np.any(k):
        raise ValueError("Process-scale fitting requires a nonzero source-declared kernel")

    def score(log_scale):
        return _gls(x, y, r+np.exp(log_scale)*k, scales)[-1]

    fit = minimize_scalar(score, method="bounded", bounds=np.log(bounds), options={"xatol": 1e-9})
    if not fit.success:
        raise ValueError("Development process-scale optimization did not converge")
    candidates = [(float(np.exp(fit.x)), float(fit.fun)),
                  (float(bounds[0]), score(np.log(bounds[0]))),
                  (float(bounds[1]), score(np.log(bounds[1])))]
    scale, objective = min(candidates, key=lambda c: c[1])
    zero = _gls(x, y, r, scales)[-1]
    selected = scale if objective+np.log(len(x)-x.shape[1]) < zero else 0.
    return dict(method="development_only_process_reml_bic.v1", selected_scale=selected,
        scale_bounds=bounds.tolist(), optimum_at_bound=scale in bounds,
        zero_restricted_bic=zero, process_restricted_bic=objective+float(np.log(len(x)-x.shape[1])),
        training_count=len(x), fixed_effect_rank=x.shape[1],
        hyperparameter_uncertainty_marginalized=False, empirically_calibrated=False)
