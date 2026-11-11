"""Prospective consistency tests for a complete Gaussian prediction.

The caller fixes chronological measurement ordering, covariance, mean and
protocol before opening observations. Raw forecast errors need not be white.
These checks test one conditional prediction, not population calibration.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import solve_triangular
from scipy.stats import chi2

from sim.estimation.covariance import validate_covariance


def _portmanteau(values, lags):
    centered = values - np.mean(values, axis=-1, keepdims=True)
    denominator = np.sum(centered**2, axis=-1)
    if np.any(denominator <= 0):
        raise ValueError("Innovation correlation requires nonconstant values")
    n = values.shape[-1]
    return n * (n + 2) * sum(
        (np.sum(centered[..., k:] * centered[..., :-k], axis=-1) / denominator)**2 / (n-k)
        for k in range(1, lags+1))


def assess_precision_innovations(residual, covariance, *, alpha_family=.01,
        null_draws=100000, seed=300352):
    """Bonferroni NIS and finite-sample whiteness tests at fixed family alpha.

    Cholesky innovations use the full cross-time covariance in the supplied
    order. A Gaussian Monte Carlo rank p-value calibrates a Ljung-Box statistic
    at lags 1..min(5,n//4). No parameters, means or scales are estimated here.
    At least eight observations are required. Seeds and draw counts are part
    of the protocol, never selected using forecast errors.
    """
    e = np.asarray(residual, dtype=float)
    c = np.asarray(covariance, dtype=float)
    if (e.ndim != 1 or not 8 <= len(e) <= 4096 or not np.all(np.isfinite(e))
            or not np.isscalar(alpha_family) or not 0 < alpha_family < 1
            or isinstance(null_draws, bool) or not isinstance(null_draws, int)
            or not 999 <= null_draws <= 1000000
            or isinstance(seed, bool) or not isinstance(seed, int) or seed < 0):
        raise ValueError("Invalid prospective innovation test inputs or insufficient observations")
    validate_covariance(c, len(e), name="complete forecast covariance")
    z = solve_triangular(np.linalg.cholesky(c), e, lower=True)
    n, alpha = len(e), alpha_family / 2
    lags = min(5, n//4)
    statistic = float(_portmanteau(z, lags))
    rng = np.random.default_rng(seed)
    exceed = 0
    # Bound memory independently of observation count and Monte Carlo draws.
    for start in range(0, null_draws, 1024):
        draws = rng.normal(size=(min(1024, null_draws-start), n))
        exceed += int(np.count_nonzero(_portmanteau(draws, lags) >= statistic))
    p = (exceed + 1) / (null_draws + 1)
    nis = float(z @ z)
    bounds = chi2.ppf([alpha/2, 1-alpha/2], n)
    nis_pass = bool(bounds[0] <= nis <= bounds[1])
    white_pass = bool(p > alpha)
    return dict(observations=n, alpha_family=alpha_family, alpha_per_test=alpha,
        joint_nis=nis, joint_nis_bounds=bounds.tolist(), joint_nis_passed=nis_pass,
        whiteness_statistic=statistic, whiteness_lags=lags, whiteness_p_value=p,
        whiteness_passed=white_pass, all_consistency_tests_passed=nis_pass and white_pass,
        null_draws=null_draws, seed=seed, no_future_estimation=True,
        independent_arc_count=1, empirical_state_covariance_qualified=False)
