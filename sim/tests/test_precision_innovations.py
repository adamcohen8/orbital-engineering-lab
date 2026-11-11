import numpy as np
import pytest
from scipy.stats import chi2

from sim.pro_tracking_od.precision_innovations import _portmanteau, assess_precision_innovations


def test_full_covariance_whitening_and_family_allocation():
    rng = np.random.default_rng(42)
    n = 100
    c = .95**abs(np.arange(n)[:, None]-np.arange(n))
    z = rng.normal(size=n)
    result = assess_precision_innovations(np.linalg.cholesky(c)@z, c, null_draws=9999)
    assert result['joint_nis'] == pytest.approx(z@z)
    assert result['joint_nis_bounds'] == pytest.approx(chi2.ppf([.0025, .9975], n))
    assert result['whiteness_passed']
    # This fixed draw has unusually little energy: the lower NIS gate matters.
    assert not result['joint_nis_passed']
    assert not result['empirical_state_covariance_qualified']


def test_null_calibration_and_correlated_alternative_power():
    rng = np.random.default_rng(43)
    draws = rng.normal(size=(20000, 100))
    threshold = np.quantile(_portmanteau(draws[:10000], 5), .995)
    rate = np.mean(_portmanteau(draws[10000:], 5) > threshold)
    assert .002 < rate < .009
    correlated = draws[10000:]@np.linalg.cholesky(.9**abs(np.arange(100)[:, None]-np.arange(100))).T
    assert np.mean(_portmanteau(correlated, 5) > threshold) > .99


def test_wrong_variance_and_seed_repeatability():
    z = np.random.default_rng(42).normal(size=100)
    a = assess_precision_innovations(z, np.eye(100), null_draws=999)
    assert a == assess_precision_innovations(z, np.eye(100), null_draws=999)
    for scale in [.1, 10]:
        assert not assess_precision_innovations(z, scale*np.eye(100), null_draws=999)['joint_nis_passed']


@pytest.mark.parametrize('residual,covariance', [(np.ones(7), np.eye(7)),
    (np.arange(8.), np.zeros((8, 8))), (np.full(8, np.nan), np.eye(8))])
def test_invalid_or_insufficient_forecasts_fail_closed(residual, covariance):
    with pytest.raises((ValueError, np.linalg.LinAlgError)):
        assess_precision_innovations(residual, covariance, null_draws=999)
