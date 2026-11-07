from __future__ import annotations

import numpy as np
import pytest

from sim.core.models import Measurement, StateBelief, StateTruth
from sim.dynamics.orbit.frames import FrameContext, eci_to_ecef_rotation_context
from sim.knowledge.object_tracking import (
    KnowledgeConditionConfig,
    KnowledgeEKFConfig,
    KnowledgeNoiseConfig,
    ObjectKnowledgeBase,
    TrackedObjectConfig,
)
from sim.sensors.access import AccessConfig, AccessModel, GroundSite
from sim.sensors.composite import CompositeSensorModel
from sim.sensors.joint_state import JointStateSensor
from sim.sensors.models import OwnStateSensor, RelativeSensor, SensorNoiseConfig
from sim.sensors.noisy_own_state import NoisyOwnStateSensor
from sim.utils.frames import ric_rect_state_to_eci
from sim.utils.geodesy import geodetic_to_ecef_km


def _truth(
    *,
    position: np.ndarray | None = None,
    velocity: np.ndarray | None = None,
    angular_rate: np.ndarray | None = None,
    t_s: float = 0.0,
) -> StateTruth:
    return StateTruth(
        position_eci_km=np.array(position if position is not None else [7000.0, 0.0, 0.0], dtype=float),
        velocity_eci_km_s=np.array(velocity if velocity is not None else [0.0, 7.5, 0.0], dtype=float),
        attitude_quat_bn=np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
        angular_rate_body_rad_s=np.array(
            angular_rate if angular_rate is not None else [0.01, -0.02, 0.03],
            dtype=float,
        ),
        mass_kg=100.0,
        t_s=float(t_s),
    )


def test_own_state_sensor_zero_noise_returns_truth_state() -> None:
    truth = _truth()
    sensor = OwnStateSensor(
        noise=SensorNoiseConfig(sigma=np.zeros(6), bias=np.zeros(6)),
        rng=np.random.default_rng(1),
    )

    meas = sensor.measure(truth, env={}, t_s=truth.t_s)

    assert meas is not None
    assert np.allclose(meas.vector, np.hstack((truth.position_eci_km, truth.velocity_eci_km_s)))
    assert meas.t_s == truth.t_s


def test_own_state_sensor_latency_preserves_acquisition_time() -> None:
    sensor = OwnStateSensor(
        noise=SensorNoiseConfig(sigma=np.zeros(6), bias=np.zeros(6), latency_s=2.0),
        rng=np.random.default_rng(11),
    )

    first = sensor.measure(_truth(t_s=0.0), env={}, t_s=0.0)
    second = sensor.measure(_truth(position=np.array([7002.0, 0.0, 0.0]), t_s=2.0), env={}, t_s=2.0)

    assert first is None
    assert second is not None
    assert np.isclose(second.t_s, 0.0)
    assert np.allclose(second.vector[:3], np.array([7000.0, 0.0, 0.0]))


def test_sensor_models_reject_invalid_noise_and_timing_domains() -> None:
    with pytest.raises(ValueError, match="sigma"):
        SensorNoiseConfig(sigma=np.array([]))
    with pytest.raises(ValueError, match="sigma"):
        SensorNoiseConfig(sigma=np.array([-1.0]))
    with pytest.raises(ValueError, match="dropout_prob"):
        SensorNoiseConfig(sigma=np.ones(1), dropout_prob=1.01)
    with pytest.raises(ValueError, match="latency_s"):
        SensorNoiseConfig(sigma=np.ones(1), latency_s=-1.0)
    with pytest.raises(ValueError, match="pos_sigma_km"):
        NoisyOwnStateSensor(-1.0, 0.0, np.random.default_rng(1))
    with pytest.raises(ValueError, match="dropout_prob"):
        JointStateSensor(dropout_prob=-0.1)


class _FixedSensor:
    def __init__(self, vector: list[float], sample_t_s: float) -> None:
        self.vector = np.array(vector, dtype=float)
        self.sample_t_s = float(sample_t_s)

    def measure(self, truth: StateTruth, env: dict, t_s: float) -> Measurement:
        return Measurement(vector=self.vector, t_s=self.sample_t_s)


def test_composite_sensor_rejects_negative_time_tolerance() -> None:
    with pytest.raises(ValueError, match="time_tolerance_s"):
        CompositeSensorModel(sensors=[], time_tolerance_s=-1.0)


def test_composite_sensor_rejects_mixed_acquisition_times() -> None:
    sensor = CompositeSensorModel(
        sensors=[
            _FixedSensor([1.0, 2.0, 3.0], sample_t_s=8.0),
            _FixedSensor([4.0, 5.0, 6.0], sample_t_s=10.0),
        ]
    )

    meas = sensor.measure(_truth(t_s=10.0), env={}, t_s=10.0)

    assert meas is None


def test_composite_sensor_concatenates_synchronized_measurements() -> None:
    sensor = CompositeSensorModel(
        sensors=[
            _FixedSensor([1.0, 2.0, 3.0], sample_t_s=8.0),
            _FixedSensor([4.0, 5.0, 6.0], sample_t_s=8.0),
        ]
    )

    meas = sensor.measure(_truth(t_s=10.0), env={}, t_s=10.0)

    assert meas is not None
    assert np.isclose(meas.t_s, 8.0)
    assert np.allclose(meas.vector, np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))


class _DropoutSensor:
    def measure(self, truth: StateTruth, env: dict, t_s: float) -> None:
        return None


def test_composite_sensor_dropout_preserves_fixed_layout_by_dropping_sample() -> None:
    sensor = CompositeSensorModel(
        sensors=[
            _FixedSensor([1.0, 2.0, 3.0], sample_t_s=8.0),
            _DropoutSensor(),
        ]
    )

    assert sensor.measure(_truth(t_s=10.0), env={}, t_s=10.0) is None


def test_relative_sensor_range_rate_matches_truth_geometry() -> None:
    observer = _truth(position=np.array([7000.0, 0.0, 0.0]), velocity=np.array([0.0, 7.5, 0.0]))
    target = _truth(position=np.array([7003.0, 4.0, 0.0]), velocity=np.array([0.0, 7.4, 0.0]))
    sensor = RelativeSensor(
        target_id="target",
        mode="range_rate",
        noise=SensorNoiseConfig(sigma=np.zeros(2), bias=np.zeros(2)),
        rng=np.random.default_rng(2),
    )

    meas = sensor.measure(observer, env={"world_truth": {"target": target}}, t_s=0.0)

    rel_r = target.position_eci_km - observer.position_eci_km
    rel_v = target.velocity_eci_km_s - observer.velocity_eci_km_s
    expected_range = np.linalg.norm(rel_r)
    expected_range_rate = float(np.dot(rel_v, rel_r / expected_range))
    assert meas is not None
    assert np.allclose(meas.vector, np.array([expected_range, expected_range_rate]))


def test_access_model_enforces_cadence_range_and_fov() -> None:
    access = AccessModel(
        AccessConfig(
            update_cadence_s=10.0,
            max_range_km=5.0,
            fov_half_angle_rad=np.deg2rad(20.0),
        )
    )
    observer = np.array([7000.0, 0.0, 0.0], dtype=float)

    assert access.can_update(observer, observer + np.array([3.0, 0.0, 0.0]), 0.0, boresight_eci=np.array([1.0, 0.0, 0.0]))
    assert not access.can_update(
        observer,
        observer + np.array([3.0, 0.0, 0.0]),
        5.0,
        boresight_eci=np.array([1.0, 0.0, 0.0]),
    )
    assert not access.can_update(
        observer,
        observer + np.array([6.0, 0.0, 0.0]),
        11.0,
        boresight_eci=np.array([1.0, 0.0, 0.0]),
    )
    assert not access.can_update(
        observer,
        observer + np.array([0.0, 3.0, 0.0]),
        12.0,
        boresight_eci=np.array([1.0, 0.0, 0.0]),
    )


def test_access_model_uses_configured_ground_site_and_elevation_mask() -> None:
    context = FrameContext(jd_utc_start=2451545.0)
    ecef_from_eci = eci_to_ecef_rotation_context(0.0, context)
    target = ecef_from_eci.T @ (
        geodetic_to_ecef_km(0.0, 0.0, 0.0) + np.array([500.0, 0.0, 0.0])
    )
    overhead = AccessModel(
        AccessConfig(
            require_ground_visibility=True,
            ground_site=GroundSite(lat_rad=0.0, lon_rad=0.0, min_elevation_rad=np.deg2rad(10.0)),
            frame_context=context,
        )
    )
    far_side = AccessModel(
        AccessConfig(
            require_ground_visibility=True,
            ground_site=GroundSite(lat_rad=0.0, lon_rad=np.pi, min_elevation_rad=np.deg2rad(10.0)),
            frame_context=context,
        )
    )

    assert overhead.evaluate(np.zeros(3), target, 0.0) == (True, "ok")
    assert far_side.evaluate(np.zeros(3), target, 0.0) == (False, "ground_elevation")


def test_joint_state_sensor_zero_noise_preserves_normalized_attitude_and_cadence() -> None:
    truth = _truth()
    sensor = JointStateSensor(
        pos_sigma_km=0.0,
        vel_sigma_km_s=0.0,
        quat_sigma=0.0,
        omega_sigma_rad_s=0.0,
        update_cadence_s=2.0,
        rng=np.random.default_rng(3),
    )

    first = sensor.measure(truth, env={}, t_s=0.0)
    skipped = sensor.measure(truth, env={}, t_s=1.0)
    second = sensor.measure(truth, env={}, t_s=2.0)

    assert first is not None
    assert skipped is None
    assert second is not None
    assert np.allclose(first.vector[:6], np.hstack((truth.position_eci_km, truth.velocity_eci_km_s)))
    assert np.allclose(first.vector[6:10], truth.attitude_quat_bn)
    assert np.isclose(np.linalg.norm(first.vector[6:10]), 1.0)
    assert np.allclose(first.vector[10:13], truth.angular_rate_body_rad_s)


def test_noisy_own_state_sensor_noise_statistics_are_centered() -> None:
    truth = _truth()
    sensor = NoisyOwnStateSensor(pos_sigma_km=0.01, vel_sigma_km_s=0.001, rng=np.random.default_rng(4))

    samples = np.array([sensor.measure(truth, env={}, t_s=float(i)).vector for i in range(512)], dtype=float)
    errors = samples - np.hstack((truth.position_eci_km, truth.velocity_eci_km_s))

    assert np.all(np.abs(np.mean(errors[:, :3], axis=0)) < 0.002)
    assert np.all(np.abs(np.mean(errors[:, 3:], axis=0)) < 0.0002)
    assert np.all(np.std(errors[:, :3], axis=0) > 0.005)
    assert np.all(np.std(errors[:, 3:], axis=0) > 0.0005)


def test_object_knowledge_base_exposes_raw_state_measurement_snapshot() -> None:
    observer = _truth(position=np.array([7000.0, 0.0, 0.0]), velocity=np.array([0.0, 7.5, 0.0]))
    target = _truth(position=np.array([7001.0, 2.0, 3.0]), velocity=np.array([0.01, 7.49, -0.02]))
    target_state = np.hstack((target.position_eci_km, target.velocity_eci_km_s))
    knowledge = ObjectKnowledgeBase(
        observer_id="chaser",
        tracked_objects=[
            TrackedObjectConfig(
                target_id="target",
                conditions=KnowledgeConditionConfig(refresh_rate_s=1.0),
                sensor_noise=KnowledgeNoiseConfig(
                    pos_sigma_km=np.zeros(3),
                    vel_sigma_km_s=np.zeros(3),
                ),
                ekf=KnowledgeEKFConfig(
                    process_noise_diag=np.ones(6) * 1e-12,
                    meas_noise_diag=np.ones(6) * 1e-12,
                    init_cov_diag=np.ones(6) * 1e-6,
                ),
            )
        ],
        dt_s=1.0,
        rng=np.random.default_rng(5),
    )

    beliefs = knowledge.update(observer, {"target": target}, t_s=0.0)
    measurements = knowledge.measurement_snapshot()

    assert "target" in beliefs
    assert "target" in measurements
    assert np.allclose(measurements["target"], target_state)


def test_knowledge_sensor_offset_velocity_includes_body_rotation() -> None:
    observer = _truth(
        position=np.array([7000.0, 0.0, 0.0]),
        velocity=np.zeros(3),
        angular_rate=np.array([0.0, 0.0, 1.0]),
    )
    target = _truth(position=np.array([7001.0, 2.0, 0.0]), velocity=np.zeros(3))
    knowledge = ObjectKnowledgeBase(
        observer_id="chaser",
        tracked_objects=[
            TrackedObjectConfig(
                target_id="target",
                conditions=KnowledgeConditionConfig(
                    refresh_rate_s=1.0,
                    sensor_position_body_m=np.array([1000.0, 0.0, 0.0]),
                ),
                sensor_noise=KnowledgeNoiseConfig(range_sigma_km=0.0, range_rate_sigma_km_s=0.0),
                measurement_model="relative_range_rate",
                ekf=KnowledgeEKFConfig(initial_state_eci_km_s=np.hstack((target.position_eci_km, target.velocity_eci_km_s))),
            )
        ],
        dt_s=1.0,
        rng=np.random.default_rng(5),
    )

    knowledge.update(observer, {"target": target}, t_s=0.0)

    np.testing.assert_allclose(knowledge.measurement_snapshot()["target"], [2.0, -1.0], atol=1.0e-12)


def test_knowledge_dropout_consumes_refresh_slot() -> None:
    observer = _truth()
    target = _truth(position=np.array([7001.0, 0.0, 0.0]))
    knowledge = ObjectKnowledgeBase(
        observer_id="chaser",
        tracked_objects=[
            TrackedObjectConfig(
                target_id="target",
                conditions=KnowledgeConditionConfig(refresh_rate_s=10.0, dropout_prob=1.0),
            )
        ],
        dt_s=1.0,
        rng=np.random.default_rng(5),
    )

    knowledge.update(observer, {"target": target}, t_s=0.0)
    knowledge.update(observer, {"target": target}, t_s=1.0)

    counts = knowledge.detection_summary()["target"]["status_counts"]
    assert counts == {"cadence": 1, "dropout": 1}


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"pos_sigma_km": np.array([-1.0])}, "pos_sigma_km"),
        ({"vel_sigma_km_s": np.array([np.nan])}, "vel_sigma_km_s"),
        ({"range_sigma_km": np.inf}, "range_sigma_km"),
        ({"az_bias_rad": np.nan}, "az_bias_rad"),
    ],
)
def test_knowledge_noise_rejects_invalid_domains(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        KnowledgeNoiseConfig(**kwargs)


def test_measured_state_estimator_trusts_latest_state_measurement() -> None:
    observer = _truth(position=np.array([7000.0, 0.0, 0.0]), velocity=np.array([0.0, 7.5, 0.0]))
    target0 = _truth(position=np.array([7001.0, 2.0, 3.0]), velocity=np.array([0.01, 7.49, -0.02]))
    target1 = _truth(position=np.array([7002.0, 2.5, 3.5]), velocity=np.array([0.02, 7.48, -0.01]))
    knowledge = ObjectKnowledgeBase(
        observer_id="chaser",
        tracked_objects=[
            TrackedObjectConfig(
                target_id="target",
                conditions=KnowledgeConditionConfig(refresh_rate_s=1.0),
                sensor_noise=KnowledgeNoiseConfig(
                    pos_sigma_km=np.zeros(3),
                    vel_sigma_km_s=np.zeros(3),
                ),
                estimator="measured_state",
            )
        ],
        dt_s=1.0,
        rng=np.random.default_rng(5),
    )

    first = knowledge.update(observer, {"target": target0}, t_s=0.0)["target"]
    second = knowledge.update(observer, {"target": target1}, t_s=1.0)["target"]

    np.testing.assert_allclose(first.state, np.hstack((target0.position_eci_km, target0.velocity_eci_km_s)))
    np.testing.assert_allclose(second.state, np.hstack((target1.position_eci_km, target1.velocity_eci_km_s)))
    assert np.all(np.diag(second.covariance) > 0.0)
    assert knowledge.consistency_summary()["target"]["update_count"] == 2


def test_measured_state_estimator_requires_state_measurement_model() -> None:
    with pytest.raises(ValueError, match="measured_state requires measurement_model='state'"):
        ObjectKnowledgeBase(
            observer_id="chaser",
            tracked_objects=[
                TrackedObjectConfig(
                    target_id="target",
                    estimator="measured_state",
                    measurement_model="relative_range_rate",
                )
            ],
            dt_s=1.0,
            rng=np.random.default_rng(5),
        )


def test_relative_knowledge_requires_explicit_initial_state_prior() -> None:
    observer = _truth(position=np.array([7000.0, 0.0, 0.0]), velocity=np.array([0.0, 7.5, 0.0]))
    target = _truth(position=np.array([7001.0, 2.0, 3.0]), velocity=np.array([0.01, 7.49, -0.02]))
    knowledge = ObjectKnowledgeBase(
        observer_id="chaser",
        tracked_objects=[
            TrackedObjectConfig(
                target_id="target",
                conditions=KnowledgeConditionConfig(refresh_rate_s=1.0),
                sensor_noise=KnowledgeNoiseConfig(range_sigma_km=0.0, range_rate_sigma_km_s=0.0),
                measurement_model="relative_range_rate",
                ekf=KnowledgeEKFConfig(init_cov_diag=np.ones(6)),
            )
        ],
        dt_s=1.0,
        rng=np.random.default_rng(5),
    )

    with pytest.raises(ValueError, match="initial_state_eci_km_s"):
        knowledge.update(observer, {"target": target}, t_s=0.0)


def test_relative_hcw_knowledge_estimator_tracks_target_from_relative_state_measurement() -> None:
    target = _truth(position=np.array([7000.0, 0.0, 0.0]), velocity=np.array([0.0, 7.546, 0.0]))
    chaser_rel_ric = np.array([0.1, -0.2, 0.05, 0.00001, -0.00002, 0.00003], dtype=float)
    chaser_state = ric_rect_state_to_eci(
        chaser_rel_ric,
        target.position_eci_km,
        target.velocity_eci_km_s,
    )
    observer = _truth(position=chaser_state[:3], velocity=chaser_state[3:])
    knowledge = ObjectKnowledgeBase(
        observer_id="chaser",
        tracked_objects=[
            TrackedObjectConfig(
                target_id="target",
                conditions=KnowledgeConditionConfig(refresh_rate_s=1.0),
                sensor_noise=KnowledgeNoiseConfig(
                    pos_sigma_km=np.zeros(3),
                    vel_sigma_km_s=np.zeros(3),
                ),
                estimator="relative_hcw_ekf",
                measurement_model="relative_state",
                ekf=KnowledgeEKFConfig(
                    process_noise_diag=np.ones(6) * 1e-12,
                    meas_noise_diag=np.ones(6) * 1e-12,
                    init_cov_diag=np.ones(6) * 1e-6,
                ),
            )
        ],
        dt_s=1.0,
        rng=np.random.default_rng(5),
    )

    belief = knowledge.update(observer, {"target": target}, t_s=0.0)["target"]

    assert np.allclose(knowledge.measurement_snapshot()["target"], -chaser_rel_ric, atol=1e-10)
    assert np.linalg.norm(belief.state[:3] - target.position_eci_km) < 1e-3
    assert np.linalg.norm(belief.state[3:6] - target.velocity_eci_km_s) < 1e-5
    assert np.allclose(belief.covariance, belief.covariance.T, atol=1e-14)
    assert np.max(np.abs(belief.covariance - np.diag(np.diag(belief.covariance)))) > 1e-12
    assert knowledge.consistency_summary()["target"]["initialization_count"] == 1


def test_relative_target_publication_uses_observer_belief_and_covariance() -> None:
    target = _truth(position=np.array([7000.0, 0.0, 0.0]), velocity=np.array([0.0, 7.546, 0.0]))
    observer = _truth(position=np.array([7000.1, -0.2, 0.05]), velocity=np.array([0.0, 7.546, 0.0]))
    observer_state = np.hstack((observer.position_eci_km, observer.velocity_eci_km_s))
    observer_belief = StateBelief(
        state=observer_state + np.array([1.0, -2.0, 0.5, 0.001, -0.002, 0.0005]),
        covariance=np.eye(6) * 0.25,
        last_update_t_s=0.0,
    )
    knowledge = ObjectKnowledgeBase(
        observer_id="chaser",
        tracked_objects=[
            TrackedObjectConfig(
                target_id="target",
                conditions=KnowledgeConditionConfig(refresh_rate_s=1.0),
                sensor_noise=KnowledgeNoiseConfig(pos_sigma_km=np.zeros(3), vel_sigma_km_s=np.zeros(3)),
                estimator="relative_hcw_ekf",
                measurement_model="relative_state",
                ekf=KnowledgeEKFConfig(init_cov_diag=np.ones(6) * 1.0e-9),
            )
        ],
        dt_s=1.0,
        rng=np.random.default_rng(5),
    )

    published = knowledge.update(
        observer,
        {"target": target},
        t_s=0.0,
        observer_belief=observer_belief,
    )["target"]
    relative_state = knowledge._tracks["target"].relative_belief
    assert relative_state is not None
    expected = ric_rect_state_to_eci(
        -relative_state.state,
        observer_belief.state[:3],
        observer_belief.state[3:6],
    )

    np.testing.assert_allclose(published.state, expected, atol=1.0e-12)
    assert np.trace(published.covariance) > 1.0


def test_relative_hcw_knowledge_estimator_updates_from_angles_range_rate_measurement() -> None:
    target = _truth(position=np.array([7000.0, 0.0, 0.0]), velocity=np.array([0.0, 7.546, 0.0]))
    chaser_rel_ric = np.array([0.1, -0.2, 0.05, 0.00001, -0.00002, 0.00003], dtype=float)
    prior_rel_ric = chaser_rel_ric + np.array([0.02, -0.01, 0.01, 0.00001, -0.00001, 0.0], dtype=float)
    chaser_state = ric_rect_state_to_eci(
        chaser_rel_ric,
        target.position_eci_km,
        target.velocity_eci_km_s,
    )
    observer = _truth(position=chaser_state[:3], velocity=chaser_state[3:])
    knowledge = ObjectKnowledgeBase(
        observer_id="chaser",
        tracked_objects=[
            TrackedObjectConfig(
                target_id="target",
                conditions=KnowledgeConditionConfig(refresh_rate_s=1.0),
                sensor_noise=KnowledgeNoiseConfig(
                    angle_sigma_rad=0.0,
                    range_sigma_km=0.0,
                    range_rate_sigma_km_s=0.0,
                ),
                estimator="relative_hcw_ekf",
                measurement_model="relative_angles_range_rate",
                ekf=KnowledgeEKFConfig(
                    process_noise_diag=np.ones(6) * 1e-12,
                    init_cov_diag=np.array([1e-2, 1e-2, 1e-2, 1e-6, 1e-6, 1e-6], dtype=float),
                    initial_state_ric=prior_rel_ric,
                ),
            )
        ],
        dt_s=1.0,
        rng=np.random.default_rng(5),
    )

    prior_belief = knowledge.update(observer, {"target": target}, t_s=0.0)["target"]
    updated = knowledge.update(observer, {"target": target}, t_s=1.0)["target"]

    assert knowledge.measurement_snapshot()["target"].shape == (4,)
    assert np.linalg.norm(updated.state[:3] - target.position_eci_km) < np.linalg.norm(
        prior_belief.state[:3] - target.position_eci_km
    )
    assert knowledge.consistency_summary()["target"]["update_count"] == 1


def test_relative_hcw_knowledge_range_rate_requires_relative_prior() -> None:
    observer = _truth(position=np.array([7000.1, 0.0, 0.0]), velocity=np.array([0.0, 7.5, 0.0]))
    target = _truth(position=np.array([7000.0, 0.0, 0.0]), velocity=np.array([0.0, 7.5001, 0.0]))
    knowledge = ObjectKnowledgeBase(
        observer_id="chaser",
        tracked_objects=[
            TrackedObjectConfig(
                target_id="target",
                conditions=KnowledgeConditionConfig(refresh_rate_s=1.0),
                sensor_noise=KnowledgeNoiseConfig(range_sigma_km=0.0, range_rate_sigma_km_s=0.0),
                estimator="relative_hcw_ekf",
                measurement_model="relative_range_rate",
                ekf=KnowledgeEKFConfig(init_cov_diag=np.ones(6)),
            )
        ],
        dt_s=1.0,
        rng=np.random.default_rng(5),
    )

    with pytest.raises(ValueError, match="initial_state_ric"):
        knowledge.update(observer, {"target": target}, t_s=0.0)


def test_relative_th_knowledge_estimator_updates_from_angles_range_rate_measurement() -> None:
    target = _truth(position=np.array([6650.0, 0.0, 0.0]), velocity=np.array([0.0, 8.1, 0.5]))
    chaser_rel_ric = np.array([0.04, -0.06, 0.025, 0.000005, -0.000003, 0.000002], dtype=float)
    prior_rel_ric = chaser_rel_ric + np.array([0.01, -0.008, 0.006, 0.000004, -0.000002, 0.000001], dtype=float)
    chaser_state = ric_rect_state_to_eci(
        chaser_rel_ric,
        target.position_eci_km,
        target.velocity_eci_km_s,
    )
    observer = _truth(position=chaser_state[:3], velocity=chaser_state[3:])
    knowledge = ObjectKnowledgeBase(
        observer_id="chaser",
        tracked_objects=[
            TrackedObjectConfig(
                target_id="target",
                conditions=KnowledgeConditionConfig(refresh_rate_s=1.0),
                sensor_noise=KnowledgeNoiseConfig(
                    angle_sigma_rad=0.0,
                    range_sigma_km=0.0,
                    range_rate_sigma_km_s=0.0,
                ),
                estimator="relative_th_ekf",
                measurement_model="relative_angles_range_rate",
                ekf=KnowledgeEKFConfig(
                    process_noise_diag=np.ones(6) * 1e-12,
                    init_cov_diag=np.array([1e-2, 1e-2, 1e-2, 1e-6, 1e-6, 1e-6], dtype=float),
                    initial_state_ric=prior_rel_ric,
                    integration_substep_s=1.0,
                ),
            )
        ],
        dt_s=1.0,
        rng=np.random.default_rng(5),
    )

    prior_belief = knowledge.update(observer, {"target": target}, t_s=0.0)["target"]
    track = knowledge._tracks["target"]
    assert track.estimator is not None
    np.testing.assert_allclose(track.estimator.chief_state_eci_km_s, prior_belief.state)
    assert np.linalg.norm(track.estimator.chief_state_eci_km_s[:3] - target.position_eci_km) > 1e-3
    updated = knowledge.update(observer, {"target": target}, t_s=1.0)["target"]

    assert knowledge.measurement_snapshot()["target"].shape == (4,)
    assert np.linalg.norm(updated.state[:3] - target.position_eci_km) < np.linalg.norm(
        prior_belief.state[:3] - target.position_eci_km
    )
    assert knowledge.consistency_summary()["target"]["update_count"] == 1


def test_relative_ya_knowledge_estimator_updates_from_angles_range_rate_measurement() -> None:
    target = _truth(position=np.array([6650.0, 0.0, 0.0]), velocity=np.array([0.0, 8.1, 0.5]))
    chaser_rel_ric = np.array([0.04, -0.06, 0.025, 0.000005, -0.000003, 0.000002], dtype=float)
    prior_rel_ric = chaser_rel_ric + np.array([0.01, -0.008, 0.006, 0.000004, -0.000002, 0.000001], dtype=float)
    chaser_state = ric_rect_state_to_eci(
        chaser_rel_ric,
        target.position_eci_km,
        target.velocity_eci_km_s,
    )
    observer = _truth(position=chaser_state[:3], velocity=chaser_state[3:])
    knowledge = ObjectKnowledgeBase(
        observer_id="chaser",
        tracked_objects=[
            TrackedObjectConfig(
                target_id="target",
                conditions=KnowledgeConditionConfig(refresh_rate_s=1.0),
                sensor_noise=KnowledgeNoiseConfig(
                    angle_sigma_rad=0.0,
                    range_sigma_km=0.0,
                    range_rate_sigma_km_s=0.0,
                ),
                estimator="relative_ya_ekf",
                measurement_model="relative_angles_range_rate",
                ekf=KnowledgeEKFConfig(
                    process_noise_diag=np.ones(6) * 1e-12,
                    init_cov_diag=np.array([1e-2, 1e-2, 1e-2, 1e-6, 1e-6, 1e-6], dtype=float),
                    initial_state_ric=prior_rel_ric,
                    integration_substep_s=1.0,
                ),
            )
        ],
        dt_s=1.0,
        rng=np.random.default_rng(5),
    )

    prior_belief = knowledge.update(observer, {"target": target}, t_s=0.0)["target"]
    updated = knowledge.update(observer, {"target": target}, t_s=1.0)["target"]

    assert knowledge.measurement_snapshot()["target"].shape == (4,)
    assert np.linalg.norm(updated.state[:3] - target.position_eci_km) < np.linalg.norm(
        prior_belief.state[:3] - target.position_eci_km
    )
    assert knowledge.consistency_summary()["target"]["update_count"] == 1


@pytest.mark.parametrize("sensor_kind", ["access", "joint", "own"])
@pytest.mark.parametrize("start_s", [0.0, 86400.0])
def test_fractional_sensor_cadence_does_not_drop_due_samples(sensor_kind, start_s) -> None:
    truth = _truth()
    if sensor_kind == "access":
        sensor = AccessModel(AccessConfig(update_cadence_s=0.3))
    elif sensor_kind == "joint":
        sensor = JointStateSensor(update_cadence_s=0.3)
    else:
        sensor = NoisyOwnStateSensor(0.0, 0.0, np.random.default_rng(0), update_cadence_s=0.3)
    def sample(t):
        if isinstance(sensor, AccessModel):
            return sensor.can_update(truth.position_eci_km, truth.position_eci_km, t)
        return sensor.measure(truth, {}, t) is not None

    for index in range(20):
        t_s = start_s + index * 0.3
        assert sample(t_s), f"due sample lost at {t_s}"
        assert not sample(t_s)
        assert not sample(t_s + 0.15)


def test_default_joint_sensor_rng_is_local_to_each_instance() -> None:
    first = JointStateSensor()
    second = JointStateSensor()
    assert first.rng is not second.rng
    np.testing.assert_array_equal(first.measure(_truth(), {}, 0.0).vector, second.measure(_truth(), {}, 0.0).vector)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_access_rejects_nonfinite_epoch_without_consuming_cadence(value) -> None:
    access = AccessModel(AccessConfig())
    position = np.array([7000.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="time_s must be finite"):
        access.can_update(position, position, value)
    assert access.can_update(position, position, 0.0)


@pytest.mark.parametrize("field", ["observer", "target", "boresight"])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_access_rejects_nonfinite_geometry_without_consuming_cadence(field, value) -> None:
    access = AccessModel(AccessConfig(max_range_km=10.0, fov_half_angle_rad=0.5))
    vectors = {
        "observer": np.array([7000.0, 0.0, 0.0]),
        "target": np.array([7001.0, 0.0, 0.0]),
        "boresight": np.array([1.0, 0.0, 0.0]),
    }
    vectors[field][1] = value
    with pytest.raises(ValueError, match="finite"):
        access.can_update(vectors["observer"], vectors["target"], 0.0, boresight_eci=vectors["boresight"])
    assert access.can_update(np.array([7000.0, 0.0, 0.0]), np.array([7001.0, 0.0, 0.0]), 0.0)
