"""Independent joint Gaussian sampling for process-conditioned forecasts."""
import numpy as np
import pytest

from sim.pro_tracking_od.precision_process import (
    condition_precision_process,
    estimate_precision_process_scale,
    mapped_acceleration_process_kernel,
)


def problem():
    t = np.linspace(0., 2., 20)
    x = np.column_stack([np.ones(14), t[:14]])
    f = np.column_stack([np.ones(6), t[14:]])
    # A random walk shared across fit and forecast, plus correlated sensor noise.
    q = .2*np.minimum.outer(t+.1, t+.1)
    noise = .03*.4**np.abs(np.subtract.outer(np.arange(20), np.arange(20)))
    return x, f, q, noise


def call(y, *, x=None):
    base, f, q, noise = problem()
    return condition_precision_process(training_design=base if x is None else x,
        training_residual=y, forecast_design=f if x is None else f/np.array([1e10, 1e-8]),
        training_observation_covariance=noise[:14, :14], joint_process_covariance=q,
        forecast_observation_covariance=noise[14:, 14:],
        training_forecast_observation_cross_covariance=noise[:14, 14:])


def test_joint_sampling_contains_forecast_errors_and_detects_bad_covariance():
    x, f, q, noise = problem()
    # Build the estimator response, then independently sample complete joint errors.
    response = np.array([call(row)["forecast_mean_correction"] for row in np.eye(14)]).T
    p = np.array(call(np.zeros(14))["forecast_covariance"])
    draws = np.random.default_rng(701114).multivariate_normal(np.zeros(20), q+noise, 10000)
    errors = draws[:, 14:]-draws[:, :14]@response.T
    z = np.linalg.solve(np.linalg.cholesky(p), errors.T)
    mean = float(np.mean(np.sum(z*z, axis=0)))
    assert 5.85 < mean < 6.15
    assert mean/.1 > 6.15 and mean/10 < 5.85
    # Omitting fit/forecast process coupling gives a materially different predictor.
    naive = f@np.linalg.lstsq(x, np.eye(14), rcond=None)[0]
    assert np.linalg.norm(response-naive) > .1


def test_native_fixed_effect_units_do_not_change_forecast():
    x, _, _, _ = problem()
    y = np.sin(np.arange(14))
    expected = call(y)
    actual = call(y, x=x/np.array([1e10, 1e-8]))
    np.testing.assert_allclose(actual["forecast_mean_correction"], expected["forecast_mean_correction"], atol=1e-12)
    np.testing.assert_allclose(actual["forecast_covariance"], expected["forecast_covariance"], atol=1e-12)


def test_reml_selects_zero_for_zero_development_residual_and_rejects_null_design():
    x, _, q, noise = problem()
    result = estimate_precision_process_scale(training_design=x, training_residual=np.zeros(14),
        training_observation_covariance=noise[:14, :14], unit_process_covariance=q[:14, :14],
        scale_bounds=[1e-5, 10.])
    assert result["selected_scale"] == 0.
    assert not result["empirically_calibrated"]
    with pytest.raises(ValueError, match="identifiable"):
        call(np.zeros(14), x=np.ones((14, 2)))


def test_process_kernel_is_required_and_joint_noise_must_be_physical():
    x, f, q, noise = problem()
    args = dict(training_design=x, training_residual=np.zeros(14), forecast_design=f,
        training_observation_covariance=noise[:14, :14], forecast_observation_covariance=noise[14:, 14:],
        training_forecast_observation_cross_covariance=np.ones((14, 6)))
    with pytest.raises(ValueError):
        condition_precision_process(**args, joint_process_covariance=q)
    args["training_forecast_observation_cross_covariance"] = np.zeros((14, 6))
    with pytest.raises(ValueError):
        condition_precision_process(**args, joint_process_covariance=None)


def test_mapped_kernel_matches_exact_integrated_white_acceleration():
    times = np.arange(5.)
    transitions = np.tile(np.eye(6), (5, 1, 1))
    transitions[:, :3, 3:] = times[:, None, None]*np.eye(3)
    indices = np.array([4, 2, 4, 1])
    h = np.zeros((4, 6))
    h[:, 0] = 1.
    actual = mapped_acceleration_process_kernel(times_s=times, state_transitions=transitions,
        measurement_indices=indices, measurement_jacobians=h, spectral_density=[.7, .2, .4])
    expected = np.empty((4, 4))
    for i in range(4):
        for j in range(4):
            a, b = sorted([float(indices[i]), float(indices[j])])
            expected[i, j] = .7*a*a*(3*b-a)/6.
    np.testing.assert_allclose(actual, expected, rtol=2e-15, atol=1e-14)
    # Repeated observations share the entire force history.
    np.testing.assert_allclose(actual[0], actual[2], rtol=0., atol=1e-14)
    with pytest.raises(ValueError, match="ordered native"):
        mapped_acceleration_process_kernel(times_s=times+1., state_transitions=transitions,
            measurement_indices=indices, measurement_jacobians=h, spectral_density=[.7, .2, .4])
