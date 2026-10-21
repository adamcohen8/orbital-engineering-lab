from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from sim.core.models import Measurement, StateBelief
from sim.estimation.relative_hcw_ekf import (
    HCWRelativeEKFEstimator,
    MorseGlintGateConfig,
    hcw_measurement_dimension,
    hcw_measurement_jacobian,
    hcw_measurement_vector,
    hcw_state_transition_matrix,
    local_l1_medoid,
    morse_glint_gate,
)


def _estimator(
    *,
    measurement_model: str = "relative_state",
    measurement_origin: str = "chief",
) -> HCWRelativeEKFEstimator:
    return HCWRelativeEKFEstimator(
        mean_motion_rad_s=0.0011,
        dt_s=10.0,
        process_noise_diag=np.ones(6) * 1e-12,
        meas_noise_diag=np.ones(hcw_measurement_dimension(measurement_model)) * 1e-8,
        measurement_model=measurement_model,
        measurement_origin=measurement_origin,
    )


def test_hcw_state_transition_matches_closed_form_components() -> None:
    n = 0.0011
    dt_s = 120.0
    x0 = np.array([0.2, -0.4, 0.1, 0.0001, -0.0002, 0.00005], dtype=float)

    propagated = hcw_state_transition_matrix(n, dt_s) @ x0

    nt = n * dt_s
    c = np.cos(nt)
    s = np.sin(nt)
    expected = np.array(
        [
            (4.0 - 3.0 * c) * x0[0] + s / n * x0[3] + 2.0 * (1.0 - c) / n * x0[4],
            6.0 * (s - nt) * x0[0] + x0[1] - 2.0 * (1.0 - c) / n * x0[3] + (4.0 * s - 3.0 * nt) / n * x0[4],
            c * x0[2] + s / n * x0[5],
            3.0 * n * s * x0[0] + c * x0[3] + 2.0 * s * x0[4],
            6.0 * n * (c - 1.0) * x0[0] - 2.0 * s * x0[3] + (4.0 * c - 3.0) * x0[4],
            -n * s * x0[2] + c * x0[5],
        ],
        dtype=float,
    )

    assert np.allclose(propagated, expected)


def test_hcw_prediction_reuses_only_the_latest_transition_matrix() -> None:
    estimator = _estimator()
    estimator.numeric_backend = "python"
    belief = StateBelief(
        state=np.array([0.2, -0.4, 0.1, 0.0001, -0.0002, 0.00005], dtype=float),
        covariance=np.eye(6) * 1.0e-3,
        last_update_t_s=0.0,
    )

    with patch(
        "sim.estimation.relative_hcw_ekf.hcw_state_transition_matrix",
        wraps=hcw_state_transition_matrix,
    ) as transition:
        first = estimator.update(belief, None, 10.0)
        second = estimator.update(belief, None, 10.0)
        estimator.mean_motion_rad_s += 1.0e-6
        estimator.update(belief, None, 10.0)

    assert transition.call_count == 2
    assert np.array_equal(first.state, second.state)
    assert np.array_equal(first.covariance, second.covariance)


@pytest.mark.parametrize("origin", ["chief", "deputy"])
@pytest.mark.parametrize(
    "model",
    ["relative_range", "relative_range_rate", "relative_angles", "relative_angles_range_rate"],
)
def test_relative_measurement_analytic_jacobian_matches_central_difference(model: str, origin: str) -> None:
    state = np.array([0.8, -0.6, 0.3, 2.0e-4, -3.0e-4, 1.0e-4])
    analytic = hcw_measurement_jacobian(model, state, measurement_origin=origin)
    numerical = np.zeros_like(analytic)
    steps = np.array([1.0e-6] * 3 + [1.0e-9] * 3)
    for axis, step in enumerate(steps):
        plus = state.copy()
        minus = state.copy()
        plus[axis] += step
        minus[axis] -= step
        delta = hcw_measurement_vector(model, plus, measurement_origin=origin) - hcw_measurement_vector(
            model, minus, measurement_origin=origin
        )
        if model.startswith("relative_angles"):
            delta[:2] = (delta[:2] + np.pi) % (2.0 * np.pi) - np.pi
        numerical[:, axis] = delta / (2.0 * step)

    assert analytic == pytest.approx(numerical, rel=2.0e-6, abs=2.0e-8)


def test_angular_filter_rejects_dimensionally_invalid_cartesian_noise_fallback() -> None:
    with pytest.raises(ValueError, match=r"rad\^2"):
        HCWRelativeEKFEstimator(
            mean_motion_rad_s=0.0011,
            dt_s=10.0,
            process_noise_diag=np.ones(6) * 1e-12,
            meas_noise_diag=np.ones(6) * 1e-8,
            measurement_model="relative_angles_range_rate",
        )


def test_hcw_relative_ekf_direct_state_update_moves_toward_truth() -> None:
    estimator = _estimator()
    truth0 = np.array([0.5, -1.2, 0.3, 0.0002, -0.0001, 0.00005], dtype=float)
    truth = hcw_state_transition_matrix(estimator.mean_motion_rad_s, 10.0) @ truth0
    belief = StateBelief(
        state=truth0 + np.array([0.1, -0.2, 0.05, 5e-5, -3e-5, 2e-5], dtype=float),
        covariance=np.eye(6) * 1e-2,
        last_update_t_s=0.0,
    )
    measurement = Measurement(vector=truth, t_s=10.0)

    with patch("numpy.linalg.inv", side_effect=AssertionError("np.linalg.inv should not be used")):
        updated = estimator.update(belief, measurement, 10.0)
    assert estimator.last_update_diagnostics is not None
    assert estimator.last_update_diagnostics.update_applied is True

    predicted_only = estimator.update(belief, None, 10.0)
    assert np.linalg.norm(updated.state - truth) < np.linalg.norm(predicted_only.state - truth)
    assert np.allclose(updated.covariance, updated.covariance.T, atol=1e-12)
    assert np.all(np.linalg.eigvalsh(updated.covariance) >= -1e-12)


def test_hcw_relative_ekf_range_rate_update_reduces_measurement_residual() -> None:
    estimator = _estimator(measurement_model="relative_range_rate")
    truth = np.array([0.8, 0.6, -0.1, -0.0003, 0.0002, 0.00005], dtype=float)
    belief = StateBelief(
        state=truth + np.array([0.25, -0.15, 0.0, 0.0002, -0.0001, 0.0], dtype=float),
        covariance=np.diag([0.5, 0.5, 0.5, 1e-5, 1e-5, 1e-5]),
        last_update_t_s=0.0,
    )
    measurement = Measurement(vector=hcw_measurement_vector("relative_range_rate", truth), t_s=0.0)

    before = hcw_measurement_vector("relative_range_rate", belief.state) - measurement.vector
    updated = estimator.update(belief, measurement, 0.0)
    after = hcw_measurement_vector("relative_range_rate", updated.state) - measurement.vector

    assert np.linalg.norm(after) < np.linalg.norm(before)


def test_hcw_relative_ekf_deputy_origin_angles_wrap_across_pi() -> None:
    estimator = _estimator(measurement_model="relative_angles", measurement_origin="deputy")
    truth = np.array([1.0, 1e-6, 0.0, 0.0, 0.0, 0.0], dtype=float)
    belief = StateBelief(
        state=np.array([1.0, -1e-6, 0.0, 0.0, 0.0, 0.0], dtype=float),
        covariance=np.eye(6) * 1e-3,
        last_update_t_s=0.0,
    )
    measurement = Measurement(vector=hcw_measurement_vector("relative_angles", truth, measurement_origin="deputy"), t_s=0.0)

    updated = estimator.update(belief, measurement, 0.0)

    assert np.all(np.isfinite(updated.state))
    assert estimator.last_update_diagnostics is not None
    assert abs(float(estimator.last_update_diagnostics.innovation[0])) < 1e-3


def test_hcw_relative_ekf_partial_measurement_is_ignored() -> None:
    estimator = _estimator(measurement_model="relative_angles_range_rate")
    belief = StateBelief(state=np.array([1.0, 0.2, 0.1, 0.0, 0.0, 0.0]), covariance=np.eye(6), last_update_t_s=0.0)

    predicted = estimator.update(belief, None, 1.0)
    partial = estimator.update(belief, Measurement(vector=np.array([0.1, 0.2]), t_s=1.0), 1.0)

    assert np.allclose(partial.state, predicted.state)
    assert np.allclose(partial.covariance, predicted.covariance)
    assert estimator.last_update_diagnostics is not None
    assert estimator.last_update_diagnostics.measurement_available is True
    assert estimator.last_update_diagnostics.update_applied is False


def test_hcw_relative_ekf_updates_at_measurement_epoch_then_propagates_to_output_time() -> None:
    estimator = _estimator()
    belief = StateBelief(
        state=np.array([0.1, -0.2, 0.05, 0.0, 0.0, 0.0], dtype=float),
        covariance=np.eye(6) * 1e-3,
        last_update_t_s=0.0,
    )
    measurement_state = np.array([0.2, -0.1, 0.03, 0.0001, -0.0002, 0.00005], dtype=float)
    measurement = Measurement(vector=measurement_state, t_s=1.0)

    updated = estimator.update(belief, measurement, 3.0)
    at_measurement = estimator.update(belief, measurement, 1.0)
    expected = hcw_state_transition_matrix(estimator.mean_motion_rad_s, 2.0) @ at_measurement.state

    assert updated.last_update_t_s == 3.0
    assert np.allclose(updated.state, expected)


def test_hcw_relative_ekf_uses_full_epoch_measurement_covariance() -> None:
    estimator = _estimator()
    covariance = np.diag([1.0e-8] * 3 + [1.0e-12] * 3)
    covariance[0, 1] = covariance[1, 0] = 0.7e-8
    estimator.set_measurement_covariance(covariance)
    state = np.array([0.2, -0.1, 0.05, 1.0e-5, -2.0e-5, 3.0e-6])
    belief = StateBelief(state=state, covariance=np.zeros((6, 6)), last_update_t_s=0.0)

    estimator.update(belief, Measurement(vector=state, t_s=0.0), 0.0)

    assert estimator.last_update_diagnostics is not None
    np.testing.assert_allclose(
        estimator.last_update_diagnostics.innovation_covariance,
        covariance,
    )


def test_morse_glint_gate_rejects_single_sample_optical_glint_and_pins_l1_medoid() -> None:
    n = 0.0011
    dt_s = 10.0
    phi = hcw_state_transition_matrix(n, dt_s)
    truth0 = np.array([0.4, -0.8, 0.15, 0.0001, -0.0002, 0.00005], dtype=float)
    truth_arc = [truth0]
    for _ in range(6):
        truth_arc.append(phi @ truth_arc[-1])

    glint_spike = np.array([0.12, -0.08, 0.05, 0.004, -0.003, 0.002], dtype=float)
    measurements: list[Measurement] = []
    for idx in range(1, len(truth_arc)):
        vec = truth_arc[idx].copy()
        if idx == 3:
            vec += glint_spike
        measurements.append(Measurement(vector=vec, t_s=idx * dt_s))

    initial_belief = StateBelief(
        state=truth0.copy(),
        covariance=np.diag([1.0e-6] * 3 + [1.0e-10] * 3),
        last_update_t_s=0.0,
    )
    ungated = HCWRelativeEKFEstimator(
        mean_motion_rad_s=n,
        dt_s=dt_s,
        process_noise_diag=np.array([1.0e-8] * 3 + [1.0e-12] * 3),
        meas_noise_diag=np.array([1.0e-6] * 3 + [1.0e-10] * 3),
    )
    gated = HCWRelativeEKFEstimator(
        mean_motion_rad_s=n,
        dt_s=dt_s,
        process_noise_diag=np.array([1.0e-8] * 3 + [1.0e-12] * 3),
        meas_noise_diag=np.array([1.0e-6] * 3 + [1.0e-10] * 3),
        morse_glint_gate=MorseGlintGateConfig(enabled=True, nis_threshold=9.21, glint_weight=0.0),
    )

    ungated_belief = initial_belief
    gated_belief = initial_belief
    for idx, meas in enumerate(measurements):
        next_meas = measurements[idx + 1] if idx + 1 < len(measurements) else None
        ungated_belief = ungated.update(ungated_belief, meas, meas.t_s)
        gated_belief = gated.update(gated_belief, meas, meas.t_s, next_measurement=next_meas)
        if idx == 2:
            assert gated.last_update_diagnostics is not None
            gate_diag = gated.last_update_diagnostics.gate_diagnostics
            assert gate_diag is not None
            assert gate_diag.evaluated is True
            assert gate_diag.classification == "glint_rejected"
            assert gate_diag.delta_dot < 0.0
            assert gate_diag.exterior_curl == pytest.approx(1.0, abs=1.0e-12)
            assert gate_diag.return_ratio == pytest.approx(0.0, abs=1.0e-12)
            assert gate_diag.medoid_pinned is True

            ungated_glint_err = float(np.linalg.norm(ungated_belief.state - truth_arc[3]))
            gated_glint_err = float(np.linalg.norm(gated_belief.state - truth_arc[3]))
            rejection_ratio = 1.0 - gated_glint_err / ungated_glint_err
            assert rejection_ratio > 0.95
            assert gated_glint_err == pytest.approx(0.0, abs=1.0e-12)

    medoid = local_l1_medoid((truth_arc[3], truth_arc[3] + glint_spike, truth_arc[3] + 1.0e-15))
    np.testing.assert_allclose(medoid, truth_arc[3], atol=1.0e-14)


def test_morse_glint_gate_preserves_full_step_response_on_impulsive_target_delta_v() -> None:
    n = 0.0011
    dt_s = 10.0
    phi = hcw_state_transition_matrix(n, dt_s)
    truth0 = np.array([0.25, -0.5, 0.1, 0.0001, -0.0001, 0.0], dtype=float)
    delta_v = np.array([0.0, 0.0, 0.0, 0.0015, -0.0025, 0.0010], dtype=float)

    truth_arc = [truth0]
    for step in range(1, 7):
        propagated = phi @ truth_arc[-1]
        if step == 3:
            propagated = propagated + delta_v
        truth_arc.append(propagated)

    measurements = [Measurement(vector=truth_arc[idx], t_s=idx * dt_s) for idx in range(1, len(truth_arc))]
    initial_belief = StateBelief(
        state=truth0.copy(),
        covariance=np.diag([1.0e-10] * 3 + [1.0e-14] * 3),
        last_update_t_s=0.0,
    )
    ungated = HCWRelativeEKFEstimator(
        mean_motion_rad_s=n,
        dt_s=dt_s,
        process_noise_diag=np.array([1.0e-12] * 3 + [1.0e-16] * 3),
        meas_noise_diag=np.array([1.0e-6] * 3 + [1.0e-10] * 3),
    )
    gated = HCWRelativeEKFEstimator(
        mean_motion_rad_s=n,
        dt_s=dt_s,
        process_noise_diag=np.array([1.0e-12] * 3 + [1.0e-16] * 3),
        meas_noise_diag=np.array([1.0e-6] * 3 + [1.0e-10] * 3),
        morse_glint_gate=MorseGlintGateConfig(enabled=True, nis_threshold=9.21, step_gain_floor=1.0),
    )

    ungated_belief = initial_belief
    gated_belief = initial_belief
    for idx, meas in enumerate(measurements):
        next_meas = measurements[idx + 1] if idx + 1 < len(measurements) else None
        ungated_belief = ungated.update(ungated_belief, meas, meas.t_s)
        gated_belief = gated.update(gated_belief, meas, meas.t_s, next_measurement=next_meas)
        if idx == 2:
            assert gated.last_update_diagnostics is not None
            gate_diag = gated.last_update_diagnostics.gate_diagnostics
            assert gate_diag is not None
            assert gate_diag.evaluated is True
            assert gate_diag.classification == "step_maneuver"
            assert gate_diag.delta_dot >= -1.0e-15
            assert gate_diag.return_ratio == pytest.approx(1.0, abs=1.0e-12)
            assert gate_diag.step_preserved is True

            np.testing.assert_allclose(gated_belief.state[3:], truth_arc[3][3:], atol=1.0e-12)
            ungated_vel_err = float(np.linalg.norm(ungated_belief.state[3:] - truth_arc[3][3:]))
            assert ungated_vel_err > 0.95 * float(np.linalg.norm(delta_v[3:]))

    np.testing.assert_allclose(gated_belief.state, truth_arc[-1], atol=1.0e-12)


def test_hcw_relative_ekf_streaming_and_window_modes_handle_glint_and_delta_v_arc() -> None:
    n = 0.0011
    dt_s = 10.0
    phi = hcw_state_transition_matrix(n, dt_s)
    truth0 = np.array([0.3, -0.6, 0.2, 0.00015, -0.0001, 0.00005], dtype=float)
    glint_spike = np.array([0.08, -0.05, 0.04, 0.003, -0.002, 0.001], dtype=float)
    delta_v = np.array([0.0, 0.0, 0.0, -0.0012, 0.0020, -0.0008], dtype=float)

    truth_arc = [truth0]
    for step in range(1, 9):
        propagated = phi @ truth_arc[-1]
        if step == 5:
            propagated = propagated + delta_v
        truth_arc.append(propagated)

    measurements: list[Measurement] = []
    for step in range(1, len(truth_arc)):
        vec = truth_arc[step].copy()
        if step == 2:
            vec = vec + glint_spike
        measurements.append(Measurement(vector=vec, t_s=step * dt_s))

    initial_belief = StateBelief(
        state=truth0.copy(),
        covariance=np.diag([1.0e-8] * 3 + [1.0e-12] * 3),
        last_update_t_s=0.0,
    )
    estimator = HCWRelativeEKFEstimator(
        mean_motion_rad_s=n,
        dt_s=dt_s,
        process_noise_diag=np.array([1.0e-10] * 3 + [1.0e-14] * 3),
        meas_noise_diag=np.array([1.0e-6] * 3 + [1.0e-10] * 3),
        morse_glint_gate=MorseGlintGateConfig(enabled=True, nis_threshold=9.21),
    )

    window_beliefs = estimator.filter_measurements(initial_belief, measurements)
    np.testing.assert_allclose(window_beliefs[1].state, truth_arc[2], atol=1.0e-12)
    np.testing.assert_allclose(window_beliefs[4].state, truth_arc[5], atol=1.0e-12)
    np.testing.assert_allclose(window_beliefs[-1].state, truth_arc[-1], atol=1.0e-12)

    estimator.reset_gate_state()
    streaming_belief = initial_belief
    streaming_states: list[np.ndarray] = []
    for meas in measurements:
        streaming_belief = estimator.update(streaming_belief, meas, meas.t_s)
        streaming_states.append(streaming_belief.state.copy())

    np.testing.assert_allclose(streaming_states[1], truth_arc[2], atol=1.0e-12)
    np.testing.assert_allclose(streaming_states[2], truth_arc[3], atol=1.0e-12)
    np.testing.assert_allclose(streaming_states[5], truth_arc[6], atol=1.0e-12)
    np.testing.assert_allclose(streaming_states[-1], truth_arc[-1], atol=1.0e-12)

    direct_glint = morse_glint_gate(np.zeros(3), np.array([1.0, -2.0, 0.5]), np.zeros(3), nis=50.0)
    assert direct_glint.classification == "glint_rejected"
    direct_step = morse_glint_gate(
        np.zeros(3),
        np.array([0.0, 2.0, 0.0]),
        np.array([0.0, 2.0, 0.0]),
        nis=50.0,
    )
    assert direct_step.classification == "step_maneuver"

