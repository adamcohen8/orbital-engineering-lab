from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from sim.core.models import StateTruth
from sim.dynamics.orbit.environment import EARTH_MU_KM3_S2
from sim.dynamics.orbit.two_body import propagate_two_body_rk4
from sim.estimation.maneuver_detection import EKFManeuverDetectionConfig, EKFManeuverDetector, chi_square_threshold
from sim.knowledge.object_tracking import (
    KnowledgeConditionConfig,
    KnowledgeEKFConfig,
    KnowledgeNoiseConfig,
    ObjectKnowledgeBase,
    TrackedObjectConfig,
)


@dataclass(frozen=True)
class _Diagnostic:
    nis: float
    dim: int = 2
    update_applied: bool = True

    @property
    def innovation(self) -> np.ndarray:
        return np.ones(self.dim, dtype=float)

    @property
    def innovation_covariance(self) -> np.ndarray:
        return np.eye(self.dim)


def _truth(state: np.ndarray, *, t_s: float) -> StateTruth:
    x = np.asarray(state, dtype=float).reshape(6)
    return StateTruth(
        position_eci_km=x[:3],
        velocity_eci_km_s=x[3:],
        attitude_quat_bn=np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
        angular_rate_body_rad_s=np.zeros(3),
        mass_kg=100.0,
        t_s=float(t_s),
    )


def _knowledge(detector: EKFManeuverDetectionConfig) -> ObjectKnowledgeBase:
    return ObjectKnowledgeBase(
        observer_id="observer",
        tracked_objects=[
            TrackedObjectConfig(
                target_id="target",
                conditions=KnowledgeConditionConfig(refresh_rate_s=1.0),
                sensor_noise=KnowledgeNoiseConfig(pos_sigma_km=np.zeros(3), vel_sigma_km_s=np.zeros(3)),
                estimator="ekf",
                measurement_model="state",
                ekf=KnowledgeEKFConfig(
                    process_noise_diag=np.ones(6) * 1.0e-14,
                    meas_noise_diag=np.array([1e-10, 1e-10, 1e-10, 1e-14, 1e-14, 1e-14], dtype=float),
                    init_cov_diag=np.array([1e-8, 1e-8, 1e-8, 1e-12, 1e-12, 1e-12], dtype=float),
                ),
                maneuver_detection=detector,
            )
        ],
        dt_s=1.0,
        rng=np.random.default_rng(123),
    )


def _propagate_nominal_arc(initial_state: np.ndarray, samples: int) -> list[np.ndarray]:
    states = [np.asarray(initial_state, dtype=float).reshape(6)]
    for _idx in range(1, samples):
        states.append(
            propagate_two_body_rk4(
                x_eci=states[-1],
                dt_s=1.0,
                mu_km3_s2=EARTH_MU_KM3_S2,
                accel_cmd_eci_km_s2=np.zeros(3),
            )
        )
    return states


def _propagate_arc_with_impulse(initial_state: np.ndarray, samples: int, *, impulse_t_s: float) -> list[np.ndarray]:
    states = [np.asarray(initial_state, dtype=float).reshape(6)]
    for idx in range(1, samples):
        prev = states[-1].copy()
        if np.isclose(float(idx), float(impulse_t_s)):
            prev[4] += 0.003
        states.append(
            propagate_two_body_rk4(
                x_eci=prev,
                dt_s=1.0,
                mu_km3_s2=EARTH_MU_KM3_S2,
                accel_cmd_eci_km_s2=np.zeros(3),
            )
        )
    return states


@pytest.mark.parametrize("field", ["window_size", "warning_count", "detection_count", "min_updates", "cooldown_updates"])
def test_ekf_maneuver_config_rejects_noninteger_counts(field: str) -> None:
    with pytest.raises(ValueError, match="must be an integer"):
        EKFManeuverDetectionConfig(**{field: 3.5})


def test_chi_square_threshold_rejects_noninteger_degrees_of_freedom() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        chi_square_threshold(2.5, 0.99)


def test_ekf_maneuver_detector_confirms_persistent_high_nis() -> None:
    detector = EKFManeuverDetector(
        EKFManeuverDetectionConfig(
            enabled=True,
            window_size=3,
            warning_count=2,
            detection_count=2,
            min_updates=2,
            warning_probability=0.95,
            detection_probability=0.99,
        )
    )
    threshold = chi_square_threshold(2, 0.99)

    detector.update(_Diagnostic(nis=0.1), t_s=1.0)
    detector.update(_Diagnostic(nis=threshold * 2.0), t_s=2.0)
    update = detector.update(_Diagnostic(nis=threshold * 2.0), t_s=3.0)

    assert update.new_confirmed_event is True
    assert detector.summary()["status"] == "confirmed"
    assert detector.summary()["first_confirmed_t_s"] == 3.0


def test_ekf_maneuver_detector_rearms_after_confirmed_event_clears() -> None:
    detector = EKFManeuverDetector(
        EKFManeuverDetectionConfig(
            enabled=True,
            window_size=2,
            warning_count=2,
            detection_count=2,
            min_updates=2,
            warning_probability=0.95,
            detection_probability=0.99,
            cooldown_updates=1,
        )
    )
    high_nis = chi_square_threshold(2, 0.99) * 2.0

    detector.update(_Diagnostic(nis=high_nis), t_s=1.0)
    first = detector.update(_Diagnostic(nis=high_nis), t_s=2.0)
    detector.update(_Diagnostic(nis=0.1), t_s=3.0)
    detector.update(_Diagnostic(nis=0.1), t_s=4.0)
    detector.update(_Diagnostic(nis=high_nis), t_s=5.0)
    second = detector.update(_Diagnostic(nis=high_nis), t_s=6.0)

    assert first.new_confirmed_event is True
    assert second.new_confirmed_event is True
    assert detector.summary()["confirmed_event_count"] == 2
    assert detector.summary()["last_event_t_s"] == 6.0


def test_ekf_maneuver_detector_retains_only_its_rolling_window() -> None:
    detector = EKFManeuverDetector(
        EKFManeuverDetectionConfig(
            enabled=True,
            window_size=3,
            warning_count=2,
            detection_count=2,
            min_updates=2,
        )
    )

    for index in range(20):
        detector.update(_Diagnostic(nis=0.1), t_s=float(index))

    assert len(detector._history) == 3
    assert detector.sample_count == 20


def test_ekf_maneuver_detector_stays_quiet_on_model_consistent_arc() -> None:
    initial = np.array([7000.0, 0.0, 0.0, 0.0, 7.546, 0.0], dtype=float)
    observer = _truth(np.array([6800.0, 10.0, 0.0, 0.0, 7.6, 0.0], dtype=float), t_s=0.0)
    knowledge = _knowledge(
        EKFManeuverDetectionConfig(
            enabled=True,
            window_size=3,
            warning_count=2,
            detection_count=2,
            min_updates=2,
            warning_probability=0.95,
            detection_probability=0.99,
        )
    )

    for idx, state in enumerate(_propagate_nominal_arc(initial, 10)):
        t_s = float(idx)
        knowledge.update(observer, {"target": _truth(state, t_s=t_s)}, t_s=t_s)

    summary = knowledge.consistency_summary()["target"]
    assert summary["maneuver_detection_status"] == "nominal"
    assert summary["maneuver_confirmed_event_count"] == 0
    assert summary["maneuver_detection_sample_count"] == 9


def test_ekf_maneuver_detector_flags_synthetic_impulse() -> None:
    initial = np.array([7000.0, 0.0, 0.0, 0.0, 7.546, 0.0], dtype=float)
    observer = _truth(np.array([6800.0, 10.0, 0.0, 0.0, 7.6, 0.0], dtype=float), t_s=0.0)
    knowledge = _knowledge(
        EKFManeuverDetectionConfig(
            enabled=True,
            window_size=3,
            warning_count=2,
            detection_count=2,
            min_updates=2,
            warning_probability=0.95,
            detection_probability=0.99,
        )
    )

    for idx, state in enumerate(_propagate_arc_with_impulse(initial, 12, impulse_t_s=5.0)):
        t_s = float(idx)
        knowledge.update(observer, {"target": _truth(state, t_s=t_s)}, t_s=t_s)

    summary = knowledge.consistency_summary()["target"]
    assert summary["maneuver_detection_status"] == "confirmed"
    assert summary["maneuver_confirmed_event_count"] == 1
    assert summary["maneuver_first_confirmed_t_s"] is not None
    assert 5.0 <= float(summary["maneuver_first_confirmed_t_s"]) <= 8.0
    assert float(summary["maneuver_max_nis"]) > chi_square_threshold(6, 0.99)
