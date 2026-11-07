from __future__ import annotations

import numpy as np
import pytest

from sim.core.models import StateTruth
from sim.knowledge.object_tracking import (
    KnowledgeConditionConfig,
    KnowledgeEKFConfig,
    KnowledgeNoiseConfig,
    ObjectKnowledgeBase,
    TrackedObjectConfig,
    _relative_measurement_and_jacobian,
    _relative_measurement_jacobian,
    _relative_measurement_vector,
)
from sim.rust_tracking_backend import (
    relative_measurement,
    relative_measurement_and_jacobian,
    relative_measurement_and_jacobian_batch,
)

native = pytest.importorskip("oel_rust_orbit")
_required = {
    "tracking_measurement",
    "tracking_measurement_and_jacobian",
    "tracking_measurement_batch",
    "tracking_measurement_and_jacobian_batch",
}
if not _required.issubset(set(dir(native))):
    pytest.skip("installed wheel predates the Rust onboard tracking kernels", allow_module_level=True)


MODELS = (
    "relative_range",
    "relative_range_rate",
    "relative_angles",
    "relative_angles_range",
    "relative_angles_range_rate",
)


def _states() -> tuple[np.ndarray, np.ndarray]:
    target = np.array([7001.25, 2.5, -0.75, 0.012, 7.48, -0.018], dtype=float)
    observer = np.array([7000.0, -1.5, 0.25, 0.002, 7.5, -0.01], dtype=float)
    return target, observer


def _truth(position: np.ndarray, velocity: np.ndarray) -> StateTruth:
    return StateTruth(
        position_eci_km=np.asarray(position, dtype=float),
        velocity_eci_km_s=np.asarray(velocity, dtype=float),
        attitude_quat_bn=np.array([1.0, 0.0, 0.0, 0.0]),
        angular_rate_body_rad_s=np.zeros(3),
        mass_kg=100.0,
        t_s=0.0,
    )


@pytest.mark.parametrize("model", MODELS)
def test_native_measurement_and_fused_fd_jacobian_match_python(model: str) -> None:
    target, observer = _states()
    python_measurement = _relative_measurement_vector(model, target, observer, numeric_backend="python")
    native_measurement = relative_measurement(target, observer, model)
    np.testing.assert_allclose(native_measurement, python_measurement, rtol=0.0, atol=2.0e-14)

    python_jacobian = _relative_measurement_jacobian(model, target, observer, numeric_backend="python")
    native_pair_measurement, native_jacobian = relative_measurement_and_jacobian(target, observer, model)
    np.testing.assert_allclose(native_pair_measurement, python_measurement, rtol=0.0, atol=2.0e-14)
    np.testing.assert_allclose(native_jacobian, python_jacobian, rtol=0.0, atol=2.0e-12)


def test_native_zero_range_preserves_reference_degenerate_geometry() -> None:
    target = np.array([7000.0, 0.0, 0.0, 0.1, 7.5, -0.2])
    observer = target.copy()
    for model in MODELS:
        np.testing.assert_array_equal(
            relative_measurement(target, observer, model),
            _relative_measurement_vector(model, target, observer, numeric_backend="python"),
        )


def test_native_zero_range_fused_jacobian_matches_reference() -> None:
    target = np.array([7000.0, 0.0, 0.0, 0.1, 7.5, -0.2])
    observer = target.copy()
    for model in MODELS:
        python_jacobian = _relative_measurement_jacobian(model, target, observer, numeric_backend="python")
        native_measurement, native_jacobian = relative_measurement_and_jacobian(target, observer, model)
        np.testing.assert_array_equal(
            native_measurement,
            _relative_measurement_vector(model, target, observer, numeric_backend="python"),
        )
        np.testing.assert_allclose(native_jacobian, python_jacobian, rtol=0.0, atol=2.0e-12)


def test_native_mixed_batch_matches_individual_fused_calls_and_angle_wrap() -> None:
    target, observer = _states()
    targets = np.vstack((target, target + np.array([0.25, -0.1, 0.3, 1.0e-4, -2.0e-4, 3.0e-4])))
    observers = np.vstack((observer, observer + np.array([-0.05, 0.04, 0.02, 2.0e-5, -3.0e-5, 1.0e-5])))
    models = ["relative_angles", "relative_angles_range_rate"]
    batch = relative_measurement_and_jacobian_batch(targets, observers, models)
    assert len(batch) == len(models)
    for index, model in enumerate(models):
        expected = relative_measurement_and_jacobian(targets[index], observers[index], model)
        np.testing.assert_allclose(batch[index][0], expected[0], rtol=0.0, atol=2.0e-14)
        np.testing.assert_allclose(batch[index][1], expected[1], rtol=0.0, atol=2.0e-12)

    # Crossing the azimuth branch must use the wrapped innovation in the FD
    # column, matching the Python forward-difference contract.
    branch_target = np.array([6999.0, -1.0e-10, 0.0, 0.0, 7.5, 0.0])
    branch_observer = np.array([7000.0, 0.0, 0.0, 0.0, 7.5, 0.0])
    rust_pair = relative_measurement_and_jacobian(branch_target, branch_observer, "relative_angles")
    python_pair = _relative_measurement_and_jacobian(
        "relative_angles", branch_target, branch_observer, numeric_backend="python"
    )
    np.testing.assert_allclose(rust_pair[1], python_pair[1], rtol=0.0, atol=2.0e-12)


def test_native_mixed_batch_accepts_per_row_finite_difference_steps() -> None:
    target, observer = _states()
    targets = np.vstack((target, target + np.array([0.25, -0.1, 0.3, 1.0e-4, -2.0e-4, 3.0e-4])))
    observers = np.vstack((observer, observer + np.array([-0.05, 0.04, 0.02, 2.0e-5, -3.0e-5, 1.0e-5])))
    models = ["relative_angles", "relative_angles_range_rate"]
    steps = np.vstack((
        np.array([1.0e-3, 1.0e-3, 1.0e-3, 1.0e-6, 1.0e-6, 1.0e-6]),
        np.array([2.0e-3, 1.5e-3, 2.5e-3, 2.0e-6, 1.5e-6, 2.5e-6]),
    ))

    batch = relative_measurement_and_jacobian_batch(targets, observers, models, steps=steps)
    for index, model in enumerate(models):
        expected = relative_measurement_and_jacobian(
            targets[index], observers[index], model, steps=steps[index]
        )
        np.testing.assert_allclose(batch[index][0], expected[0], rtol=0.0, atol=2.0e-14)
        np.testing.assert_allclose(batch[index][1], expected[1], rtol=0.0, atol=2.0e-12)


def test_native_seeded_random_mixed_batch_matches_reference() -> None:
    rng = np.random.default_rng(20260927)
    rows = 32
    observers = np.zeros((rows, 6), dtype=float)
    observers[:, :3] = np.array([7000.0, 0.0, 0.0]) + rng.normal(0.0, 40.0, size=(rows, 3))
    observers[:, 3:] = np.array([0.0, 7.5, 0.0]) + rng.normal(0.0, 0.02, size=(rows, 3))
    relative_positions = rng.normal(0.0, 1.0, size=(rows, 3)) * np.array([0.8, 1.4, 2.1])
    relative_positions += np.array([0.25, -0.35, 0.45])
    relative_velocities = rng.normal(0.0, 1.0, size=(rows, 3)) * np.array([0.002, 0.004, 0.006])
    targets = observers.copy()
    targets[:, :3] += relative_positions
    targets[:, 3:] += relative_velocities
    models = [MODELS[index % len(MODELS)] for index in range(rows)]
    native_pairs = relative_measurement_and_jacobian_batch(targets, observers, models)

    # The velocity FD epsilon is 1e-6.  A conservative 5e-10 absolute bound
    # allows roughly 5e-16 of f64 measurement roundoff after that division,
    # while remaining far below the tracking covariance scales.
    for index, model in enumerate(models):
        python_measurement, python_jacobian = _relative_measurement_and_jacobian(
            model, targets[index], observers[index], numeric_backend="python"
        )
        native_measurement, native_jacobian = native_pairs[index]
        np.testing.assert_allclose(native_measurement, python_measurement, rtol=0.0, atol=3.0e-14)
        np.testing.assert_allclose(native_jacobian, python_jacobian, rtol=2.0e-12, atol=5.0e-10)


def _knowledge(backend: str) -> ObjectKnowledgeBase:
    return ObjectKnowledgeBase(
        observer_id="chaser",
        tracked_objects=[
            TrackedObjectConfig(
                target_id="target",
                numeric_backend=backend,
                conditions=KnowledgeConditionConfig(refresh_rate_s=1.0, dropout_prob=0.25),
                sensor_noise=KnowledgeNoiseConfig(
                    angle_sigma_rad=2.0e-5,
                    range_sigma_km=2.0e-4,
                    range_rate_sigma_km_s=3.0e-6,
                    az_bias_rad=1.0e-6,
                    el_bias_rad=-2.0e-6,
                    range_bias_km=3.0e-5,
                    range_rate_bias_km_s=-4.0e-7,
                ),
                measurement_model="relative_angles_range_rate",
                ekf=KnowledgeEKFConfig(
                    process_noise_diag=np.ones(6) * 1.0e-12,
                    init_cov_diag=np.array([1.0e-2, 1.0e-2, 1.0e-2, 1.0e-6, 1.0e-6, 1.0e-6]),
                    initial_state_eci_km_s=np.array([7001.0, 2.0, -0.5, 0.01, 7.49, -0.015]),
                ),
            )
        ],
        dt_s=1.0,
        rng=np.random.default_rng(9127),
    )


@pytest.mark.parametrize("backend", ["python", "rust"])
@pytest.mark.parametrize("estimator", ["ekf", "relative_hcw_ekf", "relative_ss_j2_ekf", "relative_th_ekf", "relative_ya_ekf"])
def test_knowledge_forwards_configured_backend_to_estimator(backend: str, estimator: str) -> None:
    knowledge = ObjectKnowledgeBase(
        observer_id="chaser",
        tracked_objects=[TrackedObjectConfig(target_id="target", estimator=estimator, numeric_backend=backend)],
        dt_s=1.0,
        rng=np.random.default_rng(9127),
    )
    track = knowledge._tracks["target"]
    if estimator != "ekf":
        track._ensure_relative_estimator(np.array([7000.0, 0.0, 0.0, 0.0, 7.546053290107542, 0.0]), 0.0)
    assert track.sensor.numeric_backend == backend
    assert track.estimator is not None
    assert track.estimator.numeric_backend == backend


def test_rust_owner_tracking_preserves_seeded_detection_noise_and_belief() -> None:
    observer = _truth(np.array([7000.0, 0.0, 0.0]), np.array([0.0, 7.5, 0.0]))
    target = _truth(np.array([7001.25, 2.5, -0.75]), np.array([0.012, 7.48, -0.018]))
    python_knowledge = _knowledge("python")
    rust_knowledge = _knowledge("rust")
    python_measurements: list[np.ndarray | None] = []
    rust_measurements: list[np.ndarray | None] = []
    python_beliefs: list[np.ndarray | None] = []
    rust_beliefs: list[np.ndarray | None] = []
    for t_s in range(8):
        python_result = python_knowledge.update(observer, {"target": target}, float(t_s)).get("target")
        rust_result = rust_knowledge.update(observer, {"target": target}, float(t_s)).get("target")
        python_snapshot = python_knowledge.measurement_snapshot()
        rust_snapshot = rust_knowledge.measurement_snapshot()
        python_measurements.append(None if "target" not in python_snapshot else python_snapshot["target"].copy())
        rust_measurements.append(None if "target" not in rust_snapshot else rust_snapshot["target"].copy())
        python_beliefs.append(None if python_result is None else python_result.state.copy())
        rust_beliefs.append(None if rust_result is None else rust_result.state.copy())

    assert python_knowledge.detection_summary() == rust_knowledge.detection_summary()
    assert python_knowledge.consistency_summary()["target"]["measurement_count"] == rust_knowledge.consistency_summary()["target"]["measurement_count"]
    for python_value, rust_value in zip(python_measurements, rust_measurements):
        if python_value is None:
            assert rust_value is None
        else:
            assert rust_value is not None
            np.testing.assert_allclose(rust_value, python_value, rtol=0.0, atol=3.0e-12)
    for python_value, rust_value in zip(python_beliefs, rust_beliefs):
        if python_value is None:
            assert rust_value is None
        else:
            assert rust_value is not None
            np.testing.assert_allclose(rust_value, python_value, rtol=1.0e-11, atol=3.0e-12)


def test_well_conditioned_relative_ekf_uses_native_joseph_update(monkeypatch) -> None:
    import sim.rust_estimation_backend as estimation_backend

    if not hasattr(native, "estimation_ekf_update_innovation_bytes"):
        pytest.skip("requires native nonlinear-innovation EKF binding")
    observer = _truth(np.array([7000.0, 0.0, 0.0]), np.array([0.0, 7.5, 0.0]))
    target = _truth(np.array([7001.25, 2.5, -0.75]), np.array([0.012, 7.48, -0.018]))

    def knowledge(backend: str) -> ObjectKnowledgeBase:
        return ObjectKnowledgeBase(
            observer_id="chaser",
            tracked_objects=[TrackedObjectConfig(
                target_id="target", measurement_model="relative_range", numeric_backend=backend,
                conditions=KnowledgeConditionConfig(refresh_rate_s=1.0),
                sensor_noise=KnowledgeNoiseConfig(range_sigma_km=10.0),
                ekf=KnowledgeEKFConfig(
                    initial_state_eci_km_s=np.array([7001.0, 2.0, -0.5, 0.01, 7.49, -0.015]),
                ),
            )],
            dt_s=1.0, rng=np.random.default_rng(1),
        )

    python, rust = knowledge("python"), knowledge("rust")
    original = estimation_backend.ekf_update_innovation
    calls = []

    def counted(*args):
        calls.append(True)
        return original(*args)

    monkeypatch.setattr(estimation_backend, "ekf_update_innovation", counted)
    for index in range(50):
        expected = python.update(observer, {"target": target}, float(index))["target"]
        actual = rust.update(observer, {"target": target}, float(index))["target"]
        if index < 8:
            np.testing.assert_allclose(actual.state, expected.state, rtol=0.0, atol=3.0e-12)
            np.testing.assert_allclose(actual.covariance, expected.covariance, rtol=0.0, atol=3.0e-12)
    sigma = np.sqrt(np.diag(expected.covariance))
    assert np.max(np.abs(actual.state - expected.state) / sigma) < 1.0e-4
    assert np.max(np.abs(actual.covariance - expected.covariance)) / np.max(np.diag(expected.covariance)) < 1.0e-4
    assert calls

    def singular(*_args):
        raise ValueError("matrix must be positive definite")

    monkeypatch.setattr(estimation_backend, "ekf_update_innovation", singular)
    python, rust = knowledge("python"), knowledge("rust")
    for index in range(2):
        expected = python.update(observer, {"target": target}, float(index))["target"]
        actual = rust.update(observer, {"target": target}, float(index))["target"]
    np.testing.assert_array_equal(actual.state, expected.state)
