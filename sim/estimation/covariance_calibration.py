"""Evidence-bound covariance calibration, coverage and correlated-noise tools.

Independent statistical units are whole trials/arcs, never adjacent samples.
Profiles are fitted on development units and evaluated on disjoint units.
No function turns statistical consistency into an operational qualification.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from scipy.stats import binom, chi2, norm

from sim.estimation.covariance import validate_covariance


def _ids(values, expected=None):
    ids = [str(value) for value in values]
    if (
        not ids
        or any(not value for value in ids)
        or len(set(ids)) != len(ids)
        or (expected is not None and len(ids) != expected)
    ):
        raise ValueError("one distinct nonempty independent trial/arc ID is required per member")
    return ids


def _squared_errors(errors, covariances):
    error = np.asarray(errors, dtype=float)
    cov = np.asarray(covariances, dtype=float)
    if error.ndim != 2 or not error.shape[0] or not error.shape[1] or not np.all(np.isfinite(error)):
        raise ValueError("errors must be a finite nonempty (independent members, dimension) array")
    if cov.shape != (len(error), error.shape[1], error.shape[1]):
        raise ValueError("one covariance per independent member is required")
    white, logdets = [], []
    for e, p in zip(error, cov, strict=True):
        validate_covariance(p, error.shape[1])
        try:
            factor = np.linalg.cholesky(p)
        except np.linalg.LinAlgError as exc:
            raise ValueError("calibration requires nonsingular covariance in the evaluated coordinates") from exc
        white.append(np.linalg.solve(factor, e))
        logdets.append(2 * np.log(np.diag(factor)).sum())
    w = np.asarray(white)
    return error, cov, w, np.sum(w * w, axis=1), np.asarray(logdets)


def assess_covariance(
    errors,
    covariances,
    *,
    trial_ids: Sequence[str],
    development_ids: Sequence[str] = (),
    reference_covariances=None,
    estimate_reference_cross_covariances=None,
    probabilities=(0.5, 0.9, 0.95, 0.99),
    alpha=0.01,
    family_test_count=1,
    minimum_members=100,
) -> dict[str, Any]:
    """Distribution, containment, bias and sharpness at one fixed horizon.

    Reference uncertainty contributes P_est + P_ref - C - C.T. Cross-covariance
    is only accepted with a PSD joint estimate/reference covariance. Chi-square
    reference distributions assume unbiased Gaussian errors. Bonferroni adjusts
    the declared family, including coverage and component-mean checks.
    """
    if not 0 < alpha < 1 or family_test_count < 1 or minimum_members < 2:
        raise ValueError("invalid statistical policy")
    error = np.asarray(errors, dtype=float)
    covariance = np.asarray(covariances, dtype=float)
    ids = _ids(trial_ids, len(error))
    if set(ids) & set(map(str, development_ids)):
        raise ValueError("qualification and development trial IDs overlap")
    if estimate_reference_cross_covariances is not None and reference_covariances is None:
        raise ValueError("reference cross covariance requires reference covariance")
    if reference_covariances is not None:
        reference = np.asarray(reference_covariances, dtype=float)
        cross = (
            np.zeros_like(covariance)
            if estimate_reference_cross_covariances is None
            else np.asarray(estimate_reference_cross_covariances, dtype=float)
        )
        if reference.shape != covariance.shape or cross.shape != covariance.shape:
            raise ValueError("reference and cross covariance shapes must match estimates")
        for p, r, c in zip(covariance, reference, cross, strict=True):
            validate_covariance(np.block([[p, c], [c.T, r]]), 2 * p.shape[0], name="joint reference covariance")
        covariance = covariance + reference - cross - np.swapaxes(cross, -1, -2)
    error, covariance, white, squared, logdet = _squared_errors(error, covariance)
    n, dimension = error.shape
    levels = tuple(float(p) for p in probabilities)
    if not levels or len(set(levels)) != len(levels) or any(not 0 < p < 1 for p in levels):
        raise ValueError("containment probabilities must be unique and inside (0, 1)")
    test_alpha = alpha / (family_test_count * (1 + len(levels) + dimension))
    band = chi2.ppf([test_alpha / 2, 1 - test_alpha / 2], n * dimension) / n
    coverage = []
    for probability in levels:
        count = int(np.count_nonzero(squared <= chi2.ppf(probability, dimension)))
        accepted = binom.ppf([test_alpha / 2, 1 - test_alpha / 2], n, probability)
        coverage.append(
            {
                "probability": probability,
                "count": count,
                "fraction": count / n,
                "acceptance_counts": accepted.tolist(),
                "passed": bool(accepted[0] <= count <= accepted[1]),
            }
        )
    # Unbiasedness is assessed in physical coordinates using each member's
    # predicted uncertainty of the ensemble mean; no time samples are pooled.
    mean_error = error.mean(axis=0)
    mean_covariance = covariance.sum(axis=0) / n**2
    standardized_mean = mean_error / np.sqrt(np.diag(mean_covariance))
    bias_limit = float(norm.ppf(1 - test_alpha / 2))
    joint_bias = assess_mean_bias(error, covariance, alpha=alpha, family_test_count=family_test_count)
    checks = {
        "joint_mean_bias": joint_bias["passed"],
        "sufficient_independent_members": n >= minimum_members,
        "mean_nees": bool(band[0] <= squared.mean() <= band[1]),
        "containment": all(row["passed"] for row in coverage),
        "unbiased": bool(np.max(np.abs(standardized_mean)) <= bias_limit),
    }
    return {
        "schema_version": 1,
        "checks": checks,
        "passed": all(checks.values()),
        "member_count": n,
        "dimension": dimension,
        "trial_ids": ids,
        "reference_uncertainty_included": reference_covariances is not None,
        "mean_nees": float(squared.mean()),
        "nees_acceptance": band.tolist(),
        "coverage": coverage,
        "mean_error": mean_error.tolist(),
        "standardized_mean": standardized_mean.tolist(),
        "bias_limit": bias_limit,
        "nees_quantiles": {str(p): float(np.quantile(squared, p)) for p in (0.5, 0.9, 0.95, 0.99)},
        "mean_negative_log_likelihood": float(np.mean(0.5 * (squared + logdet + dimension * np.log(2 * np.pi)))),
        "mean_covariance_trace": float(np.trace(covariance, axis1=1, axis2=2).mean()),
        "joint_mean_bias": joint_bias,
        "supplemental_joint_bias_alpha_family": alpha,
        "all_families_alpha_bound": min(2 * alpha, 1.0),
        "alpha_family": alpha,
        "alpha_per_test": test_alpha,
        "scope": "Gaussian consistency at this horizon and declared envelope; not operational qualification",
    }


def innovation_diagnostics(
    innovations, innovation_covariances, *, trial_ids, development_ids=(), alpha=0.01, minimum_members=100
):
    """Terminal independent-member NIS plus within-run whitened lag-one checks."""
    values = np.asarray(innovations, dtype=float)
    covariances = np.asarray(innovation_covariances, dtype=float)
    if values.ndim != 3 or values.shape[1] < 3 or covariances.shape != (*values.shape, values.shape[-1]):
        raise ValueError("innovations require (members, epochs>=3, components) and matching covariances")
    report = assess_covariance(
        values[:, -1],
        covariances[:, -1],
        trial_ids=trial_ids,
        development_ids=development_ids,
        alpha=alpha,
        minimum_members=minimum_members,
    )
    whitened = np.stack(
        [np.linalg.solve(np.linalg.cholesky(c), v[..., None])[..., 0] for v, c in zip(values, covariances, strict=True)]
    )
    # Pairs never bridge separate runs. Record per-run ACF so one persistently
    # correlated arc is not disguised by cancellation across unrelated arcs.
    acfs = []
    for member in whitened:
        acfs.append([float(np.corrcoef(member[:-1, j], member[1:, j])[0, 1]) for j in range(values.shape[-1])])
    pooled = [
        float(np.corrcoef(whitened[:, :-1, j].ravel(), whitened[:, 1:, j].ravel())[0, 1])
        for j in range(values.shape[-1])
    ]
    limit = float(norm.ppf(1 - alpha / (2 * values.shape[-1])) / np.sqrt(len(values) * (values.shape[1] - 1)))
    report.update(statistic="NIS", per_member_lag1=acfs, pooled_lag1=pooled, lag1_limit=limit)
    report["checks"]["innovation_whiteness"] = bool(np.all(np.isfinite(pooled)) and max(map(abs, pooled)) <= limit)
    report["passed"] = all(report["checks"].values())
    return report


def fit_scale_profile(errors, covariances, *, development_ids, envelope: Mapping[str, Any]):
    """Fit a scalar multiplier for one explicitly bounded envelope, not a global cure."""
    e, _, _, squared, _ = _squared_errors(errors, covariances)
    ids = _ids(development_ids, len(e))
    if len(ids) < 20 or not envelope:
        raise ValueError("a declared envelope and at least 20 independent development units are required")
    scale = float(squared.mean() / e.shape[1])
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("a positive identifiable covariance scale is required")
    profile = {
        "schema_version": 1,
        "method": "development_mean_nees_scale",
        "scale": scale,
        "dimension": e.shape[1],
        "development_ids": ids,
        "envelope": dict(envelope),
        "qualification_status": "unqualified_until_disjoint_evaluation",
    }
    profile["sha256"] = _profile_sha(profile)
    return profile


def _profile_sha(profile):
    payload = {k: v for k, v in profile.items() if k != "sha256"}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def apply_scale_profile(profile, covariances, *, trial_ids, envelope):
    if _profile_sha(profile) != profile.get("sha256"):
        raise ValueError("calibration profile checksum mismatch")
    ids = _ids(trial_ids)
    if set(ids) & set(profile["development_ids"]):
        raise ValueError("development units cannot qualify a fitted calibration profile")
    if dict(envelope) != profile["envelope"]:
        raise ValueError("calibration profile envelope mismatch")
    p = np.asarray(covariances, dtype=float)
    if p.shape != (len(ids), profile["dimension"], profile["dimension"]):
        raise ValueError("profile dimension or member count mismatch")
    for value in p:
        validate_covariance(value, profile["dimension"])
    return p * profile["scale"]


def fit_noise_profile(residual_arcs, *, development_ids, envelope):
    """Development-only bias, component covariance and lag-one AR model.

    Arcs must have the same cadence and measurement component order. The fitted
    bias is reported separately: callers must explicitly correct the mean or
    model it as a nuisance parameter. No residual rejection or silent centering
    of qualification observations occurs here.
    """
    ids = _ids(development_ids, len(residual_arcs))
    arcs = [np.asarray(a, dtype=float) for a in residual_arcs]
    if len(arcs) < 20 or any(
        a.ndim != 2 or len(a) < 4 or a.shape[1] != arcs[0].shape[1] or not np.all(np.isfinite(a)) for a in arcs
    ):
        raise ValueError("noise calibration needs >=20 independent finite arcs with >=4 samples")
    bias = np.concatenate(arcs).mean(axis=0)
    centered = [a - bias for a in arcs]
    component_covariance = np.cov(np.concatenate(centered), rowvar=False)
    component_covariance = np.atleast_2d(component_covariance)
    np.linalg.cholesky(component_covariance)
    white = [np.linalg.solve(np.linalg.cholesky(component_covariance), a.T).T for a in centered]
    left = np.concatenate([a[:-1] for a in white])
    right = np.concatenate([a[1:] for a in white])
    rho = float(np.sum(left * right) / np.sum(left * left))
    if not abs(rho) < 1:
        raise ValueError("development noise is not a stationary AR(1) process")
    profile = {
        "schema_version": 1,
        "method": "stationary_ar1_common_rho",
        "bias": bias.tolist(),
        "component_covariance": component_covariance.tolist(),
        "rho": rho,
        "development_ids": ids,
        "envelope": dict(envelope),
    }
    profile["sha256"] = _profile_sha(profile)
    return profile


def noise_covariance(profile, sample_indices, *, envelope, arc_id):
    if _profile_sha(profile) != profile.get("sha256") or profile["envelope"] != dict(envelope):
        raise ValueError("noise profile identity or envelope mismatch")
    if str(arc_id) in profile["development_ids"]:
        raise ValueError("noise profile evaluation must use a disjoint arc")
    indices = np.asarray(sample_indices)
    if (
        indices.ndim != 1
        or not len(indices)
        or not np.all(np.isfinite(indices))
        or np.any(indices != np.floor(indices))
        or np.any(np.diff(indices) <= 0)
    ):
        raise ValueError("AR sample indices must be strictly increasing finite integers")
    lag = np.abs(indices[:, None] - indices[None, :]).astype(int)
    return np.kron(profile["rho"] ** lag, np.asarray(profile["component_covariance"]))


def fit_process_noise_profile(evaluate, candidates, *, development_ids, envelope):
    """Select a declared process-noise policy using development predictive score.

    ``evaluate(policy)`` must run the maintained filter/propagator on development
    arcs and return one prediction error and covariance per independent arc.
    Selection uses Gaussian negative log likelihood, penalizing both excessive
    dispersion and overconfidence. Qualification must use disjoint arcs.
    """
    from sim.estimation.sequential_od import SequentialProcessNoise

    ids = _ids(development_ids)
    if len(ids) < 20 or not candidates or not envelope:
        raise ValueError("process-noise fitting requires candidates, an envelope and >=20 development arcs")
    scores = []
    for candidate in candidates:
        policy = SequentialProcessNoise(**dict(candidate))
        errors, covariances = evaluate(policy)
        errors, _, _, squared, logdet = _squared_errors(errors, covariances)
        if len(errors) != len(ids):
            raise ValueError("process-noise objective must preserve independent development arc identities")
        scores.append(
            {
                "policy": {"model": policy.model, "spectral_density": list(policy.spectral_density)},
                "score": float(np.mean(0.5 * (squared + logdet + errors.shape[1] * np.log(2 * np.pi)))),
            }
        )
    best = min(scores, key=lambda row: row["score"])
    profile = {
        "schema_version": 1,
        "method": "development_predictive_log_score_grid",
        "process_noise": best["policy"],
        "candidate_scores": scores,
        "development_ids": ids,
        "envelope": dict(envelope),
        "qualification_status": "unqualified_until_disjoint_evaluation",
    }
    profile["sha256"] = _profile_sha(profile)
    return profile


def process_noise_from_profile(profile, *, trial_ids, envelope):
    from sim.estimation.sequential_od import SequentialProcessNoise

    if _profile_sha(profile) != profile.get("sha256") or profile["envelope"] != dict(envelope):
        raise ValueError("process-noise profile identity or envelope mismatch")
    if set(_ids(trial_ids)) & set(profile["development_ids"]):
        raise ValueError("process-noise qualification arcs overlap development")
    return SequentialProcessNoise(**profile["process_noise"])


def assess_mean_bias(errors, covariances, *, alpha=0.01, family_test_count=1):
    """Joint mean-error test including correlated state/measurement directions.

    This supplements the original consistency family; its separately declared
    family alpha does not change any existing NEES or containment threshold.
    The joint covariance of the mean assumes independent statistical units.
    """
    if not 0 < alpha < 1 or family_test_count < 1:
        raise ValueError("invalid joint-mean bias statistical policy")
    error, covariance, _, _, _ = _squared_errors(errors, covariances)
    mean = error.mean(axis=0)
    mean_covariance = covariance.sum(axis=0) / len(error) ** 2
    statistic = float(mean @ np.linalg.solve(mean_covariance, mean))
    bound = float(chi2.ppf(1 - alpha / family_test_count, error.shape[1]))
    return {
        "joint_mean_nees": statistic,
        "acceptance_max": bound,
        "passed": bool(statistic <= bound),
        "alpha_family": alpha,
        "family_test_count": family_test_count,
        "member_count": len(error),
    }
