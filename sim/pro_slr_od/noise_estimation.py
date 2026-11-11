"""Fit-only restricted-likelihood selection of SLR session covariance.

The caller supplies residuals and sensitivities from the maintained physical
model on its declared training partition. This local linearization selects a
noise policy; a subsequent nonlinear refit and independent test are required.
"""

from __future__ import annotations

import hashlib

import numpy as np
from scipy.linalg import solve_triangular
from scipy.optimize import minimize

from sim.pro_slr_od.measurement_noise import SlrSessionNoise


def estimate_session_noise(fit_rows, residual_km, jacobian, *, include_session_drift=False):
    residual = np.asarray(residual_km, dtype=float)
    design = np.asarray(jacobian, dtype=float)
    if (design.ndim != 2 or residual.shape != (len(fit_rows),)
            or len(design) != len(fit_rows) or design.shape[1] < 1
            or len(fit_rows) < design.shape[1]+8 or not np.all(np.isfinite(design))
            or not np.all(np.isfinite(residual))):
        raise ValueError("Invalid training-only SLR noise design.")
    scales = np.linalg.norm(design, axis=0)
    if np.any(scales == 0):
        raise ValueError("SLR noise fitting requires observable fixed effects.")
    normalized = design/scales
    if np.linalg.matrix_rank(normalized) != design.shape[1]:
        raise ValueError("SLR noise fitting requires observable fixed effects.")
    # Validate the CRD metadata before numerical optimization.
    SlrSessionNoise(.004, .004).covariance_km2(fit_rows)
    bounds = [(np.log(1.e-5), np.log(.05))]  # meters: .01 mm through 50 mm

    def evaluate(theta, common):
        model = SlrSessionNoise(float(np.exp(theta[0])), float(np.exp(theta[1])) if common else 0.,
                                float(np.exp(theta[2])) if common == 2 else 0.)
        factor = np.linalg.cholesky(model.covariance_km2(fit_rows))
        Q = solve_triangular(factor, normalized, lower=True)
        y = solve_triangular(factor, residual, lower=True)
        x = np.linalg.lstsq(Q, y, rcond=1.e-13)[0]
        z = y-Q@x
        logdet_information = 2*np.sum(np.log(np.linalg.svd(Q, compute_uv=False)))
        score = 2*np.sum(np.log(np.diag(factor))) + z@z + logdet_information
        return float(score), model, x/scales

    candidates = []
    if not isinstance(include_session_drift, bool):
        raise ValueError("SLR session drift selection must be explicit boolean.")
    if include_session_drift:
        SlrSessionNoise(.004, .004, .004).covariance_km2(fit_rows)
    for common in range(3 if include_session_drift else 2):
        count = common+1
        result = minimize(lambda x, mode=common: evaluate(x, mode)[0], np.log([.004]*count),
                          method="L-BFGS-B", bounds=bounds*count)
        score, model, delta = evaluate(result.x, common)
        if not result.success:
            raise ValueError("SLR training-noise optimization did not converge.")
        candidates.append(dict(policy=model.policy(), restricted_objective=score,
            restricted_bic=float(score+count*np.log(len(design)-design.shape[1])),
            fixed_effect_delta_native=delta.tolist(), hyperparameter_count=count))
    selected = min(candidates, key=lambda x: x["restricted_bic"])
    evidence = dict(method="training_only_linearized_reml_bic.v1", candidates=candidates,
        selected=selected, fit_count=len(fit_rows), fixed_effect_rank=design.shape[1],
        white_floor_bounds_m=[1.e-5, .05], session_sigma_bounds_m=[1.e-5, .05],
        input_array_sha256=hashlib.sha256(design.tobytes()+residual.tobytes()).hexdigest(),
        fit_measurement_ids=[str(row["measurement_id"]) for row in fit_rows],
        calibrated_covariance=False,
        limitation="Hyperparameter uncertainty is not marginalized; independent arc validation remains required.")
    return SlrSessionNoise.from_policy(selected["policy"]), evidence
