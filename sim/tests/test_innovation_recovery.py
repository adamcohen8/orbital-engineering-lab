from __future__ import annotations

import sqlite3
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from sim.config.plugin_validation import _validate_object_knowledge
from sim.core.models import Measurement, StateBelief
from sim.dynamics.orbit.elements import coe_to_rv_eci
from sim.estimation.innovation_recovery import InnovationRecoveryConfig
from sim.estimation.maneuver_detection import EKFManeuverDetectionConfig, EKFManeuverDetector
from sim.estimation.orbit_ekf import OrbitEKFEstimator
from sim.estimation.relative_hcw_ekf import HCWRelativeEKFEstimator, SSJ2RelativeEKFEstimator
from sim.estimation.relative_th_ekf import THRelativeEKFEstimator, YARelativeEKFEstimator
from sim.runtime.knowledge_factory import _build_knowledge_base

MODELS = ["hcw", "ss_j2", "th", "ya", "absolute"]
R_DIAG = np.array([1e-6] * 3 + [1e-10] * 3)


def make_estimator(model, *, backend="python", policy=None, origin="chief"):
    r, v = coe_to_rv_eci(a_km=7600.0, ecc=0.12, inc_deg=34.0, raan_deg=18.0, argp_deg=42.0, true_anomaly_deg=25.0)
    chief = np.hstack((r, v))
    kwargs = dict(
        dt_s=10.0,
        process_noise_diag=R_DIAG * 1e-4,
        meas_noise_diag=R_DIAG.copy(),
        numeric_backend=backend,
        innovation_recovery=policy,
    )
    if model == "absolute":
        return OrbitEKFEstimator(mu_km3_s2=398600.4418, **kwargs), chief
    kwargs["measurement_origin"] = origin
    if model == "hcw":
        estimator = HCWRelativeEKFEstimator(mean_motion_rad_s=0.0011, **kwargs)
    elif model == "ss_j2":
        estimator = SSJ2RelativeEKFEstimator(
            mean_motion_rad_s=0.0011, reference_radius_km=6900.0, reference_inclination_rad=0.9, **kwargs
        )
    else:
        cls = THRelativeEKFEstimator if model == "th" else YARelativeEKFEstimator
        estimator = cls(chief_state_eci_km_s=chief, chief_epoch_t_s=0.0, **kwargs)
    return estimator, np.array([0.3, -0.6, 0.2, 0.00015, -0.0001, 0.00005])


def arc(estimator, initial, *, impulse=0, outliers=(), samples=12):
    state = initial.copy()
    truth, measurements = [], []
    sign = -1 if getattr(estimator, "measurement_origin", "chief") == "deputy" else 1
    for step in range(1, samples + 1):
        state, _ = estimator._predict(state, np.diag(R_DIAG), from_t_s=(step - 1) * 10.0, to_t_s=step * 10.0)
        if step == impulse:
            state[3:] += [0.0012, -0.002, 0.0008]
        observed = state.copy()
        if step in outliers:
            observed += (-1 if step % 2 else 1) * np.array([0.08, -0.05, 0.04, 0.003, -0.002, 0.001])
        truth.append(state.copy())
        measurements.append(Measurement(sign * observed, step * 10.0))
    return truth, measurements


def initial_belief(state):
    return StateBelief(state.copy(), np.diag(R_DIAG * 0.01), 0.0)


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("backend", ["python", "rust"])
def test_disabled_policy_is_bit_exact_to_original_update(model, backend):
    estimator, state = make_estimator(model, backend=backend, policy=InnovationRecoveryConfig())
    original, _ = make_estimator(model, backend=backend)
    _, measurements = arc(original, state, impulse=7, outliers=(3,))
    left, right = initial_belief(state), initial_belief(state)
    for i, measurement in enumerate(measurements):
        measurement = None if i == 4 else measurement
        left = estimator.update(left, measurement, (i + 1) * 10.0)
        right = original._update_standard(right, measurement, (i + 1) * 10.0)
        np.testing.assert_array_equal(left.state, right.state)
        np.testing.assert_array_equal(left.covariance, right.covariance)
        assert estimator.last_update_diagnostics.recovery_diagnostics is None


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("backend", ["python", "rust"])
@pytest.mark.parametrize("origin", ["chief", "deputy"])
def test_isolated_outlier_is_held_and_impulse_recovers_after_one_sample(model, backend, origin):
    estimator, state = make_estimator(
        model, backend=backend, origin=origin, policy=InnovationRecoveryConfig(enabled=True)
    )
    truth, measurements = arc(estimator, state, impulse=7, outliers=(3,))
    belief, diagnostics = initial_belief(state), []
    for index, measurement in enumerate(measurements):
        belief = estimator.update(belief, measurement, measurement.t_s)
        diagnostics.append(estimator.last_update_diagnostics.recovery_diagnostics)
        if index != 6:
            np.testing.assert_allclose(belief.state, truth[index], rtol=0, atol=2e-9)
        assert np.linalg.eigvalsh(belief.covariance).min() > 0
    assert diagnostics[2].action == "held"
    assert diagnostics[3].classification == "transient_discrepancy"
    assert diagnostics[6].classification == "pending_discrepancy"
    assert diagnostics[7].classification == "persistent_discrepancy"
    assert diagnostics[7].event_time_s == 70.0
    assert diagnostics[7].decision_time_s == 80.0
    assert diagnostics[7].action == "reinitialized"


@pytest.mark.parametrize("model", MODELS)
def test_shape_check_avoids_persistence_false_reset_on_unrelated_consecutive_outliers(model):
    robust, state = make_estimator(model, policy=InnovationRecoveryConfig(enabled=True))
    simple, _ = make_estimator(model, policy=InnovationRecoveryConfig(enabled=True, strategy="persistence"))
    truth, measurements = arc(robust, state, outliers=(3, 4))
    a, b = initial_belief(state), initial_belief(state)
    for measurement in measurements[:4]:
        a = robust.update(a, measurement, measurement.t_s)
        b = simple.update(b, measurement, measurement.t_s)
    assert robust.last_update_diagnostics.recovery_diagnostics.classification == "unresolved_discrepancy"
    assert simple.last_update_diagnostics.recovery_diagnostics.action == "reinitialized"
    assert np.linalg.norm(a.state[:3] - truth[3][:3]) < 1e-8
    assert np.linalg.norm(b.state[:3] - truth[3][:3]) > 0.01


@pytest.mark.parametrize("model", MODELS)
def test_dropouts_and_output_latency_keep_candidate_and_held_track_at_same_epoch(model):
    estimator, state = make_estimator(model, policy=InnovationRecoveryConfig(enabled=True))
    truth, measurements = arc(estimator, state, impulse=3)
    belief = initial_belief(state)
    for measurement in measurements[:3]:
        belief = estimator.update(belief, measurement, measurement.t_s + 1.0)
    assert estimator.last_update_diagnostics.recovery_diagnostics.action == "held"
    belief = estimator.update(belief, None, 35.0)
    belief = estimator.update(belief, measurements[3], 41.0)
    expected, _ = estimator._predict(truth[3], np.diag(R_DIAG), from_t_s=40.0, to_t_s=41.0)
    np.testing.assert_allclose(belief.state, expected, atol=2e-9, rtol=0)
    assert estimator.last_update_diagnostics.recovery_diagnostics.action == "reinitialized"


@pytest.mark.parametrize("model", MODELS)
def test_external_covariance_edit_invalidates_pending_hypothesis(model):
    estimator, state = make_estimator(model, policy=InnovationRecoveryConfig(enabled=True))
    _, measurements = arc(estimator, state, impulse=3)
    belief = initial_belief(state)
    for measurement in measurements[:3]:
        belief = estimator.update(belief, measurement, measurement.t_s)
    belief = StateBelief(belief.state.copy(), belief.covariance * 1.01, belief.last_update_t_s)
    estimator.update(belief, measurements[3], measurements[3].t_s)
    assert estimator.last_update_diagnostics.recovery_diagnostics.action == "held"
    estimator.reset_innovation_recovery()
    assert estimator._innovation_recovery.pending is None


def test_full_correlated_measurement_covariance_is_propagated_and_used():
    estimator, state = make_estimator("ya", policy=InnovationRecoveryConfig(enabled=True))
    covariance = np.diag(R_DIAG)
    covariance[0, 1] = covariance[1, 0] = 0.8e-6
    estimator.set_measurement_covariance(covariance)
    _, measurements = arc(estimator, state, impulse=3)
    belief = initial_belief(state)
    for measurement in measurements[:4]:
        belief = estimator.update(belief, measurement, measurement.t_s)
    assert estimator.last_update_diagnostics.recovery_diagnostics.action == "reinitialized"
    np.testing.assert_allclose(belief.covariance, 4 * covariance, rtol=1e-12, atol=1e-20)


@pytest.mark.parametrize("model", ["th", "ya"])
def test_eccentric_chief_reference_refresh_preserves_pending_recovery(model):
    fixed, state = make_estimator(model, policy=InnovationRecoveryConfig(enabled=True))
    refreshed, _ = make_estimator(model, policy=InnovationRecoveryConfig(enabled=True))
    _, measurements = arc(fixed, state, impulse=3)
    a, b = initial_belief(state), initial_belief(state)
    for measurement in measurements:
        a = fixed.update(a, measurement, measurement.t_s)
        b = refreshed.update(b, measurement, measurement.t_s)
        np.testing.assert_allclose(a.state, b.state, rtol=0, atol=1e-8)
        chief = refreshed._chief_state_at(measurement.t_s)
        refreshed.set_reference_state(chief, measurement.t_s)
    assert refreshed.last_update_diagnostics.recovery_diagnostics.classification == "nominal"


def test_maneuver_evidence_keeps_held_nis_and_recovery_epochs():
    estimator, state = make_estimator("hcw", policy=InnovationRecoveryConfig(enabled=True))
    detector = EKFManeuverDetector(
        EKFManeuverDetectionConfig(enabled=True, min_updates=1, window_size=3, warning_count=2, detection_count=2)
    )
    _, measurements = arc(estimator, state, impulse=3)
    belief = initial_belief(state)
    for measurement in measurements[:4]:
        belief = estimator.update(belief, measurement, measurement.t_s)
        detector.update(estimator.last_update_diagnostics, t_s=measurement.t_s)
    summary = detector.summary()
    assert summary["sample_count"] == 4
    assert summary["recovery_hold_count"] == 1
    assert summary["recovery_reinitialization_count"] == 1
    assert summary["last_recovery_event_t_s"] == 30.0
    assert summary["last_recovery_decision_t_s"] == 40.0
    assert summary["confirmed_event_count"] == 1


@pytest.mark.parametrize("model", MODELS)
def test_runtime_factory_wires_opt_in_policy_to_each_filter(model):
    type_name = "ekf" if model == "absolute" else f"relative_{'ss_j2' if model == 'ss_j2' else model}_ekf"
    agent = SimpleNamespace(
        knowledge={
            "targets": ["target"],
            "estimation": {
                "type": type_name,
                "measurement_model": "state" if model == "absolute" else "relative_state",
                "numeric_backend": "python",
                "ekf": {"innovation_recovery": {"enabled": True}},
            },
        }
    )
    knowledge = _build_knowledge_base("observer", agent, 10.0, np.random.default_rng(1))
    track = knowledge._tracks["target"]
    if model != "absolute":
        state = np.array([7000.0, 0.0, 0.0, 0.0, np.sqrt(398600.4418 / 7000.0), 0.0])
        track._ensure_relative_estimator(state, 0.0)
    assert track.estimator.innovation_recovery.enabled


@pytest.mark.parametrize(
    "raw",
    [
        {"enabled": "false"},
        {"nis_probability": float("nan")},
        {"strategy": "unknown"},
        {"confirmation_count": 1},
        {"covariance_inflation": 0.5},
        {"max_gap_s": 0},
        {"typo": 1},
    ],
)
def test_policy_contract_is_validated_before_execution(raw):
    errors = _validate_object_knowledge(
        {"estimation": {"ekf": {"innovation_recovery": raw}}}, "objects.observer.knowledge"
    )
    assert any("innovation_recovery" in error for error in errors)


def test_partial_measurements_are_rejected_only_when_recovery_is_enabled():
    kwargs = dict(
        mean_motion_rad_s=0.0011,
        dt_s=10.0,
        process_noise_diag=R_DIAG,
        meas_noise_diag=np.ones(2),
        measurement_model="relative_angles",
        numeric_backend="python",
    )
    HCWRelativeEKFEstimator(**kwargs, innovation_recovery=InnovationRecoveryConfig())
    with pytest.raises(ValueError, match="full-state"):
        HCWRelativeEKFEstimator(**kwargs, innovation_recovery=InnovationRecoveryConfig(enabled=True))


def test_timeout_and_policy_disable_discard_pending_state():
    estimator, state = make_estimator("hcw", policy=InnovationRecoveryConfig(enabled=True, max_gap_s=5.0))
    _, measurements = arc(estimator, state, impulse=3)
    belief = initial_belief(state)
    for measurement in measurements[:4]:
        belief = estimator.update(belief, measurement, measurement.t_s)
    assert estimator.last_update_diagnostics.recovery_diagnostics.action == "held"
    estimator.innovation_recovery = replace(estimator.innovation_recovery, enabled=False)
    estimator.update(belief, measurements[4], measurements[4].t_s)
    assert estimator._innovation_recovery.pending is None


def test_nonfinite_or_nonchronological_measurement_cannot_resolve_pending_state():
    estimator, state = make_estimator("hcw", policy=InnovationRecoveryConfig(enabled=True))
    _, measurements = arc(estimator, state, impulse=3)
    belief = initial_belief(state)
    for measurement in measurements[:3]:
        belief = estimator.update(belief, measurement, measurement.t_s)
    with pytest.raises(ValueError, match="finite"):
        estimator.update(belief, Measurement(np.full(6, np.nan), 40.0), 40.0)
    with pytest.raises(ValueError, match="strictly increasing"):
        estimator.update(belief, measurements[2], 30.0)
    estimator.update(belief, measurements[3], 40.0)
    assert estimator.last_update_diagnostics.recovery_diagnostics.action == "reinitialized"


def test_review_metrics_retain_recovery_counts_and_distinct_epochs():
    from sim.reporting.review_store import _insert_metrics

    connection = sqlite3.connect(":memory:")
    connection.execute(
        "CREATE TABLE metrics (metric_id, metric_name, object_id, deputy_id, chief_id, value, units, source)"
    )
    _insert_metrics(
        connection,
        summary={
            "knowledge_consistency_by_observer": {
                "observer": {
                    "target": {
                        "recovery_hold_count": 2,
                        "recovery_reinitialization_count": 1,
                        "last_recovery_event_t_s": 30.0,
                        "last_recovery_decision_t_s": 50.0,
                    }
                }
            }
        },
    )
    rows = dict(connection.execute("SELECT metric_name, value FROM metrics WHERE source='innovation_recovery'"))
    assert rows == {
        "recovery_hold_count": 2.0,
        "recovery_reinitialization_count": 1.0,
        "last_recovery_event_t_s": 30.0,
        "last_recovery_decision_t_s": 50.0,
    }
    connection.close()


@pytest.mark.parametrize("strategy", ["predictive_consistency", "persistence"])
def test_column_vector_measurements_keep_standard_measurement_shape_contract(strategy):
    estimator, state = make_estimator("hcw", policy=InnovationRecoveryConfig(enabled=True, strategy=strategy))
    truth, measurements = arc(estimator, state, impulse=3)
    belief = initial_belief(state)
    for measurement in measurements[:4]:
        belief = estimator.update(
            belief, Measurement(measurement.vector.reshape(6, 1), measurement.t_s), measurement.t_s
        )
    np.testing.assert_allclose(belief.state, truth[3], atol=1e-12)
    assert estimator.last_update_diagnostics.recovery_diagnostics.action == "reinitialized"
