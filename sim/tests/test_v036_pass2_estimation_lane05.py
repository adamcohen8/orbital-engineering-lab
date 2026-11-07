from __future__ import annotations

import numpy as np
import pytest

from sim.core.models import StateBelief
from sim.estimation.attitude_ekf import AttitudeEKFEstimator
from sim.estimation.orbit_ekf import OrbitEKFEstimator
from sim.estimation.relative_hcw_ekf import HCWRelativeEKFEstimator
from sim.estimation.relative_th_ekf import THRelativeEKFEstimator


def _orbit_estimator() -> OrbitEKFEstimator:
    return OrbitEKFEstimator(
        mu_km3_s2=0.0,
        dt_s=1.0,
        process_noise_diag=np.zeros(6),
        meas_noise_diag=np.ones(6),
        numeric_backend="python",
    )


def _attitude_estimator() -> AttitudeEKFEstimator:
    return AttitudeEKFEstimator(
        dt_s=0.1,
        inertia_kg_m2=np.diag([10.0, 12.0, 8.0]),
        process_noise_diag=np.zeros(7),
        meas_noise_diag=np.ones(7),
        numeric_backend="python",
    )


def _hcw_estimator() -> HCWRelativeEKFEstimator:
    return HCWRelativeEKFEstimator(
        mean_motion_rad_s=0.0011,
        dt_s=10.0,
        process_noise_diag=np.zeros(6),
        meas_noise_diag=np.ones(6),
        numeric_backend="python",
    )


def _th_estimator() -> THRelativeEKFEstimator:
    return THRelativeEKFEstimator(
        chief_state_eci_km_s=np.array([7000.0, 0.0, 0.0, 0.0, 7.546, 0.0]),
        chief_epoch_t_s=0.0,
        dt_s=10.0,
        process_noise_diag=np.zeros(6),
        meas_noise_diag=np.ones(6),
        numeric_backend="python",
    )


@pytest.mark.parametrize(
    ("factory", "state"),
    [
        (_orbit_estimator, np.zeros(6)),
        (_attitude_estimator, np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])),
        (_hcw_estimator, np.zeros(6)),
        (_th_estimator, np.zeros(6)),
    ],
)
def test_ekf_rejects_non_psd_belief_covariance(factory, state: np.ndarray) -> None:
    estimator = factory()
    covariance = np.eye(state.size)
    covariance[0, 0] = -1.0

    with pytest.raises(ValueError, match="belief covariance"):
        estimator.update(StateBelief(state=state, covariance=covariance, last_update_t_s=0.0), None, 0.0)


@pytest.mark.parametrize(
    ("factory", "state"),
    [
        (_orbit_estimator, np.zeros(6)),
        (_attitude_estimator, np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])),
        (_hcw_estimator, np.zeros(6)),
        (_th_estimator, np.zeros(6)),
    ],
)
def test_ekf_accepts_semidefinite_zero_belief_covariance(factory, state: np.ndarray) -> None:
    estimator = factory()

    result = estimator.update(
        StateBelief(state=state, covariance=np.zeros((state.size, state.size)), last_update_t_s=0.0),
        None,
        0.0,
    )

    assert np.all(np.isfinite(result.covariance))
    assert np.allclose(result.covariance, 0.0)


@pytest.mark.parametrize("epoch", [float("nan"), float("inf"), float("-inf")])
def test_th_reference_epoch_validation_is_atomic(epoch: float) -> None:
    estimator = _th_estimator()
    old_state = estimator.chief_state_eci_km_s.copy()
    old_epoch = estimator.chief_epoch_t_s

    with pytest.raises(ValueError, match="reference epoch must be finite"):
        estimator.set_reference_state(np.array([7001.0, 0.0, 0.0, 0.0, 7.545, 0.0]), epoch)

    np.testing.assert_array_equal(estimator.chief_state_eci_km_s, old_state)
    assert estimator.chief_epoch_t_s == old_epoch


@pytest.mark.parametrize(
    ("inertia", "message"),
    [
        (np.diag([1.0, -1.0, 1.0]), "positive definite"),
        (np.zeros((3, 3)), "positive definite"),
        (np.array([[1.0, 2.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]), "symmetric"),
    ],
)
def test_attitude_ekf_rejects_nonphysical_inertia(inertia: np.ndarray, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        AttitudeEKFEstimator(
            dt_s=0.1,
            inertia_kg_m2=inertia,
            process_noise_diag=np.zeros(7),
            meas_noise_diag=np.ones(7),
            numeric_backend="python",
        )
