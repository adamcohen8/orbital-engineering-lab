from __future__ import annotations

from array import array
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from sim.core.models import Measurement, StateBelief, StateTruth
from sim.dynamics.orbit.environment import EARTH_MU_KM3_S2, EARTH_RADIUS_KM
from sim.estimation.innovation_recovery import InnovationRecoveryConfig, validate_recovery_config
from sim.estimation.maneuver_detection import EKFManeuverDetectionConfig, EKFManeuverDetector
from sim.estimation.orbit_ekf import OrbitEKFEstimator, OrbitEKFUpdateDiagnostics
from sim.estimation.relative_hcw_ekf import (
    HCWRelativeEKFEstimator,
    SSJ2RelativeEKFEstimator,
    hcw_measurement_dimension,
    hcw_measurement_vector,
    normalize_hcw_measurement_model,
)
from sim.estimation.relative_th_ekf import THRelativeEKFEstimator, YARelativeEKFEstimator
from sim.numeric_backend import normalize_numeric_backend
from sim.sensors.access import AccessConfig, AccessModel
from sim.utils.frames import eci_relative_to_ric_rect, ric_rect_state_to_eci
from sim.utils.quaternion import quaternion_to_dcm_bn

KnowledgeSummaryValue = Any


def _normalize_tracking_numeric_backend(value: str | None) -> str:
    backend = normalize_numeric_backend("python" if value is None else value, error_message="knowledge numeric_backend must be python or rust")
    return backend


def _float_history() -> array[float]:
    return array("d")


def _line_of_sight_clear(observer_eci_km: np.ndarray, target_eci_km: np.ndarray) -> bool:
    ro = np.array(observer_eci_km, dtype=float)
    rt = np.array(target_eci_km, dtype=float)
    d = rt - ro
    denom = float(np.dot(d, d))
    if denom <= 0.0:
        return True
    tau = float(np.clip(-np.dot(ro, d) / denom, 0.0, 1.0))
    closest = ro + tau * d
    return float(np.linalg.norm(closest)) > EARTH_RADIUS_KM


@dataclass(frozen=True)
class KnowledgeConditionConfig:
    refresh_rate_s: float = 10.0
    max_range_km: float | None = None
    fov_half_angle_rad: float | None = None
    solid_angle_sr: float | None = None
    require_line_of_sight: bool = False
    dropout_prob: float = 0.0
    sensor_position_body_m: np.ndarray = field(default_factory=lambda: np.zeros(3))
    sensor_boresight_body: np.ndarray | None = None

    def __post_init__(self) -> None:
        if not np.isfinite(float(self.refresh_rate_s)) or float(self.refresh_rate_s) <= 0.0:
            raise ValueError("refresh_rate_s must be positive and finite.")
        if not np.isfinite(float(self.dropout_prob)) or not 0.0 <= float(self.dropout_prob) <= 1.0:
            raise ValueError("dropout_prob must be finite and between 0 and 1.")
        if self.max_range_km is not None and (
            not np.isfinite(float(self.max_range_km)) or float(self.max_range_km) <= 0.0
        ):
            raise ValueError("max_range_km must be positive and finite when provided.")
        if self.fov_half_angle_rad is not None and (
            not np.isfinite(float(self.fov_half_angle_rad)) or not 0.0 <= float(self.fov_half_angle_rad) <= np.pi
        ):
            raise ValueError("fov_half_angle_rad must be finite and within [0, pi].")
        if self.solid_angle_sr is not None and (
            not np.isfinite(float(self.solid_angle_sr)) or not 0.0 <= float(self.solid_angle_sr) <= 4.0 * np.pi
        ):
            raise ValueError("solid_angle_sr must be finite and within [0, 4*pi].")
        position = np.asarray(self.sensor_position_body_m, dtype=float).reshape(-1)
        if position.size != 3 or not np.all(np.isfinite(position)):
            raise ValueError("sensor_position_body_m must contain three finite values.")
        if self.sensor_boresight_body is not None:
            boresight = np.asarray(self.sensor_boresight_body, dtype=float).reshape(-1)
            if boresight.size != 3 or not np.all(np.isfinite(boresight)):
                raise ValueError("sensor_boresight_body must contain three finite values.")
            if float(np.linalg.norm(boresight)) <= 0.0:
                raise ValueError("sensor_boresight_body must be nonzero when provided.")

    @classmethod
    def from_knowledge(cls, knowledge: dict, *, default_period_s: float) -> KnowledgeConditionConfig:
        """Resolve the same access settings for reporting and onboard sensors."""
        conditions = dict(knowledge.get("conditions", {}) or {})
        return cls(
            refresh_rate_s=float(knowledge.get("refresh_rate_s", default_period_s)),
            max_range_km=conditions.get("max_range_km"),
            fov_half_angle_rad=conditions.get("fov_half_angle_rad"),
            solid_angle_sr=conditions.get("solid_angle_sr"),
            require_line_of_sight=bool(conditions.get("require_line_of_sight", False)),
            dropout_prob=float(conditions.get("dropout_prob", 0.0)),
            sensor_position_body_m=np.asarray(conditions.get("sensor_position_body_m", [0.0, 0.0, 0.0]), dtype=float),
            sensor_boresight_body=(
                None
                if conditions.get("sensor_boresight_body") is None
                else np.asarray(conditions["sensor_boresight_body"], dtype=float)
            ),
        )


@dataclass(frozen=True)
class KnowledgeNoiseConfig:
    pos_sigma_km: np.ndarray = field(default_factory=lambda: np.array([1e-3, 1e-3, 1e-3]))
    vel_sigma_km_s: np.ndarray = field(default_factory=lambda: np.array([1e-5, 1e-5, 1e-5]))
    pos_bias_km: np.ndarray = field(default_factory=lambda: np.zeros(3))
    vel_bias_km_s: np.ndarray = field(default_factory=lambda: np.zeros(3))
    range_sigma_km: float = 1e-3
    range_rate_sigma_km_s: float = 1e-5
    angle_sigma_rad: float = 1e-4
    range_bias_km: float = 0.0
    range_rate_bias_km_s: float = 0.0
    az_bias_rad: float = 0.0
    el_bias_rad: float = 0.0

    def __post_init__(self) -> None:
        for name in ("pos_sigma_km", "vel_sigma_km_s"):
            values = np.asarray(getattr(self, name), dtype=float).reshape(-1)
            if values.size not in (1, 3):
                raise ValueError(f"{name} must be scalar or length-3.")
            if not np.all(np.isfinite(values)) or np.any(values < 0.0):
                raise ValueError(f"{name} must contain finite nonnegative values.")
        for name in ("pos_bias_km", "vel_bias_km_s"):
            values = np.asarray(getattr(self, name), dtype=float).reshape(-1)
            if values.size not in (1, 3):
                raise ValueError(f"{name} must be scalar or length-3.")
            if not np.all(np.isfinite(values)):
                raise ValueError(f"{name} must contain finite values.")
        for name in ("range_sigma_km", "range_rate_sigma_km_s", "angle_sigma_rad"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative.")
        for name in ("range_bias_km", "range_rate_bias_km_s", "az_bias_rad", "el_bias_rad"):
            if not np.isfinite(float(getattr(self, name))):
                raise ValueError(f"{name} must be finite.")


@dataclass(frozen=True)
class KnowledgeEKFConfig:
    process_noise_diag: np.ndarray = field(default_factory=lambda: np.array([1e-8, 1e-8, 1e-8, 1e-10, 1e-10, 1e-10]))
    meas_noise_diag: np.ndarray = field(default_factory=lambda: np.array([1e-6, 1e-6, 1e-6, 1e-10, 1e-10, 1e-10]))
    init_cov_diag: np.ndarray = field(default_factory=lambda: np.array([1.0, 1.0, 1.0, 1e-2, 1e-2, 1e-2]))
    initial_state_eci_km_s: np.ndarray | None = None
    initial_state_ric: np.ndarray | None = None
    mean_motion_rad_s: float | None = None
    measurement_origin: str = "deputy"
    integration_substep_s: float = 10.0
    innovation_recovery: InnovationRecoveryConfig | None = None

    def __post_init__(self) -> None:
        for name in ("process_noise_diag", "meas_noise_diag", "init_cov_diag"):
            values = np.array(getattr(self, name), dtype=float).reshape(-1)
            if values.size != 6 or not np.all(np.isfinite(values)) or np.any(values < 0.0):
                raise ValueError(f"{name} must contain six finite nonnegative values.")
        for name in ("initial_state_eci_km_s", "initial_state_ric"):
            raw = getattr(self, name)
            if raw is None:
                continue
            values = np.array(raw, dtype=float).reshape(-1)
            if values.size != 6 or not np.all(np.isfinite(values)):
                raise ValueError(f"{name} must contain six finite values.")
        if self.mean_motion_rad_s is not None and (
            not np.isfinite(float(self.mean_motion_rad_s)) or float(self.mean_motion_rad_s) <= 0.0
        ):
            raise ValueError("mean_motion_rad_s must be positive and finite when provided.")
        if not np.isfinite(float(self.integration_substep_s)) or float(self.integration_substep_s) <= 0.0:
            raise ValueError("integration_substep_s must be positive and finite.")


@dataclass(frozen=True)
class TrackedObjectConfig:
    target_id: str
    conditions: KnowledgeConditionConfig = KnowledgeConditionConfig()
    sensor_noise: KnowledgeNoiseConfig = KnowledgeNoiseConfig()
    estimator: str = "ekf"
    measurement_model: str = "state"
    ekf: KnowledgeEKFConfig = KnowledgeEKFConfig()
    maneuver_detection: EKFManeuverDetectionConfig = EKFManeuverDetectionConfig()
    numeric_backend: str = "rust"

    def __post_init__(self) -> None:
        object.__setattr__(self, "numeric_backend", _normalize_tracking_numeric_backend(self.numeric_backend))


class ObjectDetectionGate:
    """Shared observer access policy; measurement generation is a separate owner."""

    def __init__(self, conditions: KnowledgeConditionConfig, rng: np.random.Generator):
        self.conditions = conditions
        self.rng = rng
        self.access = AccessModel(
            AccessConfig(
                update_cadence_s=float(conditions.refresh_rate_s),
                max_range_km=conditions.max_range_km,
                fov_half_angle_rad=conditions.fov_half_angle_rad,
                solid_angle_sr=conditions.solid_angle_sr,
            )
        )
        self.last_detection_status: str | None = None

    def detect(self, observer_truth: StateTruth, target_truth: StateTruth, t_s: float) -> bool:
        sensor_position_eci_km, sensor_boresight_eci = self._sensor_pose_eci(observer_truth)
        access_ok, access_reason = self.access.evaluate(
            sensor_position_eci_km,
            target_truth.position_eci_km,
            t_s,
            boresight_eci=sensor_boresight_eci,
        )
        if not access_ok:
            self.last_detection_status = str(access_reason)
            return False
        if self.conditions.require_line_of_sight and not _line_of_sight_clear(
            sensor_position_eci_km, target_truth.position_eci_km
        ):
            self.last_detection_status = "line_of_sight"
            return False
        self.access._last_update_t_s = float(t_s)
        if self.rng.random() < float(self.conditions.dropout_prob):
            self.last_detection_status = "dropout"
            return False
        self.last_detection_status = "detected"
        return True

    def _sensor_pose_eci(self, observer_truth: StateTruth) -> tuple[np.ndarray, np.ndarray | None]:
        sensor_position_eci_km, _, sensor_boresight_eci = self._sensor_state_eci(observer_truth)
        return sensor_position_eci_km, sensor_boresight_eci

    def _sensor_state_eci(
        self,
        observer_truth: StateTruth,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        c_bn = quaternion_to_dcm_bn(observer_truth.attitude_quat_bn)
        pos_body_m = np.array(self.conditions.sensor_position_body_m, dtype=float).reshape(3)
        offset_eci_km = (c_bn.T @ pos_body_m) / 1e3
        sensor_position_eci_km = observer_truth.position_eci_km + offset_eci_km
        omega_eci_rad_s = c_bn.T @ np.asarray(observer_truth.angular_rate_body_rad_s, dtype=float).reshape(3)
        sensor_velocity_eci_km_s = observer_truth.velocity_eci_km_s + np.cross(
            omega_eci_rad_s,
            offset_eci_km,
        )
        boresight_body = self.conditions.sensor_boresight_body
        if boresight_body is None:
            if float(np.linalg.norm(pos_body_m)) > 1e-12:
                boresight_body = pos_body_m
            else:
                boresight_body = np.array([1.0, 0.0, 0.0], dtype=float)
        b = np.array(boresight_body, dtype=float).reshape(3)
        bn = float(np.linalg.norm(b))
        if not np.isfinite(bn) or bn <= 0.0:
            raise ValueError("sensor_boresight_body must be nonzero when provided.")
        sensor_boresight_eci = c_bn.T @ (b / bn)
        return sensor_position_eci_km, sensor_velocity_eci_km_s, sensor_boresight_eci


class _OtherObjectStateSensor(ObjectDetectionGate):
    def __init__(
        self,
        conditions: KnowledgeConditionConfig,
        noise: KnowledgeNoiseConfig,
        rng: np.random.Generator,
        *,
        numeric_backend: str = "rust",
    ):
        super().__init__(conditions, rng)
        self.noise = noise
        self.numeric_backend = _normalize_tracking_numeric_backend(numeric_backend)

    _detection_gate = ObjectDetectionGate.detect

    def measure(self, observer_truth: StateTruth, target_truth: StateTruth, t_s: float) -> Measurement | None:
        if not self._detection_gate(observer_truth, target_truth, t_s):
            return None

        pos_sigma = _expand3(self.noise.pos_sigma_km)
        vel_sigma = _expand3(self.noise.vel_sigma_km_s)
        pos_bias = _expand3(self.noise.pos_bias_km)
        vel_bias = _expand3(self.noise.vel_bias_km_s)
        z_pos = target_truth.position_eci_km + pos_bias + self.rng.normal(0.0, pos_sigma, size=3)
        z_vel = target_truth.velocity_eci_km_s + vel_bias + self.rng.normal(0.0, vel_sigma, size=3)
        return Measurement(vector=np.hstack((z_pos, z_vel)), t_s=t_s)

    def measure_relative(
        self,
        observer_truth: StateTruth,
        target_truth: StateTruth,
        t_s: float,
        measurement_model: str,
    ) -> Measurement | None:
        model = _normalize_measurement_model(measurement_model)
        if model == "state":
            return self.measure(observer_truth, target_truth, t_s)
        if not self._detection_gate(observer_truth, target_truth, t_s):
            return None
        sensor_position_eci_km, sensor_velocity_eci_km_s, _ = self._sensor_state_eci(observer_truth)
        observer_state = np.hstack((sensor_position_eci_km, sensor_velocity_eci_km_s))
        truth_state = np.hstack((target_truth.position_eci_km, target_truth.velocity_eci_km_s))
        ideal = _relative_measurement_vector(
            model,
            truth_state,
            observer_state,
            numeric_backend=self.numeric_backend,
        )
        sigma = _relative_measurement_sigma(model, self.noise)
        bias = _relative_measurement_bias(model, self.noise)
        return Measurement(vector=ideal + bias + self.rng.normal(0.0, sigma, size=ideal.size), t_s=t_s)


@dataclass
class _Track:
    target_id: str
    sensor: _OtherObjectStateSensor
    estimator: (
        OrbitEKFEstimator
        | HCWRelativeEKFEstimator
        | SSJ2RelativeEKFEstimator
        | THRelativeEKFEstimator
        | YARelativeEKFEstimator
        | None
    )
    estimator_type: str
    measurement_model: str
    init_cov_diag: np.ndarray
    process_noise_diag: np.ndarray
    estimator_dt_s: float
    initial_state_eci_km_s: np.ndarray | None = None
    initial_state_ric: np.ndarray | None = None
    hcw_mean_motion_rad_s: float | None = None
    hcw_measurement_origin: str = "deputy"
    th_integration_substep_s: float = 10.0
    innovation_recovery: InnovationRecoveryConfig | None = None
    maneuver_detector: EKFManeuverDetector = field(default_factory=EKFManeuverDetector)
    belief: StateBelief | None = None
    relative_belief: StateBelief | None = None
    step_count: int = 0
    initialization_count: int = 0
    measurement_count: int = 0
    update_count: int = 0
    last_measurement_t_s: float | None = None
    nis_values: array[float] = field(default_factory=_float_history)
    nees_values: array[float] = field(default_factory=_float_history)
    innovation_norm_values: array[float] = field(default_factory=_float_history)
    pos_error_norm_km_values: array[float] = field(default_factory=_float_history)
    vel_error_norm_km_s_values: array[float] = field(default_factory=_float_history)
    track_age_s_values: array[float] = field(default_factory=_float_history)
    detected_count: int = 0
    reacquisition_count: int = 0
    loss_of_detection_count: int = 0
    consecutive_missed_steps: int = 0
    max_consecutive_missed_steps: int = 0
    last_detected: bool = False
    time_since_last_detection_s_values: array[float] = field(default_factory=_float_history)
    detection_status_counts: dict[str, int] = field(default_factory=dict)
    last_measurement_vector: np.ndarray | None = None

    def step(
        self,
        observer_truth: StateTruth,
        target_truth: StateTruth,
        t_s: float,
        observer_belief: StateBelief | None = None,
    ) -> StateBelief | None:
        self.step_count += 1
        self.last_measurement_vector = None
        if self.estimator_type in {"relative_hcw_ekf", "relative_ss_j2_ekf", "relative_th_ekf", "relative_ya_ekf"}:
            return self._step_relative_hcw(observer_truth, target_truth, t_s, observer_belief)
        meas = self.sensor.measure_relative(observer_truth, target_truth, t_s, self.measurement_model)
        detect_status = str(self.sensor.last_detection_status or "unknown")
        self.detection_status_counts[detect_status] = int(self.detection_status_counts.get(detect_status, 0)) + 1
        if meas is not None:
            self.last_measurement_vector = np.array(meas.vector, dtype=float).reshape(-1)
            self.measurement_count += 1
            self.last_measurement_t_s = float(t_s)
            self.detected_count += 1
            if not self.last_detected and (self.detected_count > 1):
                self.reacquisition_count += 1
            self.last_detected = True
            self.consecutive_missed_steps = 0
            self.time_since_last_detection_s_values.append(0.0)
        else:
            if self.last_detected:
                self.loss_of_detection_count += 1
            self.last_detected = False
            self.consecutive_missed_steps += 1
            self.max_consecutive_missed_steps = max(self.max_consecutive_missed_steps, self.consecutive_missed_steps)
            if self.last_measurement_t_s is not None:
                self.time_since_last_detection_s_values.append(float(t_s - self.last_measurement_t_s))
        if self.estimator_type == "measured_state":
            if _normalize_measurement_model(self.measurement_model) != "state":
                raise ValueError("measured_state estimator requires measurement_model='state'.")
            if meas is None:
                return self.belief
            self.belief = StateBelief(
                state=np.array(meas.vector, dtype=float).reshape(6).copy(),
                covariance=_state_measurement_covariance(self.sensor.noise),
                last_update_t_s=float(t_s),
            )
            if self.initialization_count <= 0:
                self.initialization_count += 1
            self.update_count += 1
            self._record_consistency(target_truth, None, t_s)
            return self.belief
        if self.belief is None:
            if meas is None:
                return None
            if _normalize_measurement_model(self.measurement_model) == "state":
                init_state = meas.vector.copy()
            else:
                if self.initial_state_eci_km_s is None:
                    raise ValueError(
                        f"tracked target {self.target_id!r} uses measurement_model={self.measurement_model!r}; "
                        "relative-only tracking requires ekf.initial_state_eci_km_s or "
                        "estimation.initial_state_eci_km_s instead of truth-seeded initialization."
                    )
                init_state = np.array(self.initial_state_eci_km_s, dtype=float).reshape(6)
            self.belief = StateBelief(state=init_state, covariance=np.diag(self.init_cov_diag), last_update_t_s=t_s)
            self.initialization_count += 1
            self._record_consistency(target_truth, None, t_s)
            return self.belief
        assert self.estimator is not None
        if _normalize_measurement_model(self.measurement_model) == "state":
            self.belief = self.estimator.update(self.belief, meas, t_s)
        else:
            self.belief = self._relative_ekf_update(self.belief, meas, observer_truth, t_s)
        diag = self.estimator.last_update_diagnostics
        if diag is not None and diag.update_applied:
            self.update_count += 1
        self._record_consistency(target_truth, diag, t_s)
        return self.belief

    def _step_relative_hcw(
        self,
        observer_truth: StateTruth,
        target_truth: StateTruth,
        t_s: float,
        observer_belief: StateBelief | None,
    ) -> StateBelief | None:
        meas = self._measure_hcw(observer_truth, target_truth, t_s)
        if self.relative_belief is None:
            if meas is None:
                return self.belief
            init_state = self._initial_hcw_state_from_measurement_or_config(meas)
            self.relative_belief = StateBelief(
                state=init_state,
                covariance=np.diag(self.init_cov_diag),
                last_update_t_s=float(t_s),
            )
            self.initialization_count += 1
            self.belief = self._target_belief_from_relative(observer_truth, observer_belief)
            if self.belief is not None:
                self._ensure_relative_estimator(self.belief.state, t_s)
            self._record_consistency(target_truth, None, t_s)
            return self.belief

        reference_belief = (
            self.belief
            if self.belief is not None
            else self._target_belief_from_relative(observer_truth, observer_belief)
        )
        if reference_belief is None:
            return None
        reference_state = reference_belief.state
        self._ensure_relative_estimator(reference_state, t_s)
        assert isinstance(
            self.estimator,
            (HCWRelativeEKFEstimator, SSJ2RelativeEKFEstimator, THRelativeEKFEstimator, YARelativeEKFEstimator),
        )
        self.relative_belief = self.estimator.update(self.relative_belief, meas, t_s)
        diag = self.estimator.last_update_diagnostics
        if diag is not None and diag.update_applied:
            self.update_count += 1
        self.belief = self._target_belief_from_relative(observer_truth, observer_belief)
        if isinstance(self.estimator, THRelativeEKFEstimator) and self.belief is not None:
            self.estimator.set_reference_state(np.array(self.belief.state, dtype=float).reshape(6), float(t_s))
        self._record_consistency(target_truth, diag, t_s)
        return self.belief

    def _ensure_relative_estimator(self, reference_state_eci_km_s: np.ndarray, t_s: float) -> None:
        if self.estimator is not None:
            return
        reference_state = np.array(reference_state_eci_km_s, dtype=float).reshape(6)
        if self.estimator_type == "relative_ya_ekf":
            self.estimator = YARelativeEKFEstimator(
                chief_state_eci_km_s=reference_state,
                chief_epoch_t_s=float(t_s),
                dt_s=self.estimator_dt_s,
                process_noise_diag=np.array(self.process_noise_diag, dtype=float),
                meas_noise_diag=self._hcw_meas_noise_diag(),
                measurement_model=self.measurement_model,
                measurement_origin=self.hcw_measurement_origin,
                numeric_backend=self.sensor.numeric_backend,
                innovation_recovery=self.innovation_recovery,
                integration_substep_s=float(self.th_integration_substep_s),
            )
        elif self.estimator_type == "relative_th_ekf":
            self.estimator = THRelativeEKFEstimator(
                chief_state_eci_km_s=reference_state,
                chief_epoch_t_s=float(t_s),
                dt_s=self.estimator_dt_s,
                process_noise_diag=np.array(self.process_noise_diag, dtype=float),
                meas_noise_diag=self._hcw_meas_noise_diag(),
                measurement_model=self.measurement_model,
                measurement_origin=self.hcw_measurement_origin,
                numeric_backend=self.sensor.numeric_backend,
                innovation_recovery=self.innovation_recovery,
                integration_substep_s=float(self.th_integration_substep_s),
            )
        elif self.estimator_type == "relative_ss_j2_ekf":
            self.estimator = SSJ2RelativeEKFEstimator.from_chief_state(
                reference_state,
                dt_s=self.estimator_dt_s,
                process_noise_diag=np.array(self.process_noise_diag, dtype=float),
                meas_noise_diag=self._hcw_meas_noise_diag(),
                measurement_model=self.measurement_model,
                measurement_origin=self.hcw_measurement_origin,
                numeric_backend=self.sensor.numeric_backend,
                innovation_recovery=self.innovation_recovery,
            )
        else:
            self.estimator = HCWRelativeEKFEstimator(
                mean_motion_rad_s=self._hcw_mean_motion(reference_state),
                dt_s=self.estimator_dt_s,
                process_noise_diag=np.array(self.process_noise_diag, dtype=float),
                meas_noise_diag=self._hcw_meas_noise_diag(),
                measurement_model=self.measurement_model,
                measurement_origin=self.hcw_measurement_origin,
                numeric_backend=self.sensor.numeric_backend,
                innovation_recovery=self.innovation_recovery,
            )

    def _measure_hcw(
        self,
        observer_truth: StateTruth,
        target_truth: StateTruth,
        t_s: float,
    ) -> Measurement | None:
        detected = self.sensor._detection_gate(observer_truth, target_truth, t_s)
        detect_status = str(self.sensor.last_detection_status or "unknown")
        self.detection_status_counts[detect_status] = int(self.detection_status_counts.get(detect_status, 0)) + 1
        if not detected:
            if self.last_detected:
                self.loss_of_detection_count += 1
            self.last_detected = False
            self.consecutive_missed_steps += 1
            self.max_consecutive_missed_steps = max(self.max_consecutive_missed_steps, self.consecutive_missed_steps)
            if self.last_measurement_t_s is not None:
                self.time_since_last_detection_s_values.append(float(t_s - self.last_measurement_t_s))
            return None

        sensor_position_eci_km, sensor_velocity_eci_km_s, _ = self.sensor._sensor_state_eci(observer_truth)
        sensor_state = np.hstack((sensor_position_eci_km, sensor_velocity_eci_km_s))
        target_state = np.hstack((target_truth.position_eci_km, target_truth.velocity_eci_km_s))
        native_truth = eci_relative_to_ric_rect(sensor_state, target_state)
        ideal = hcw_measurement_vector(
            self.measurement_model,
            native_truth,
            measurement_origin=self.hcw_measurement_origin,
        )
        sigma = self._hcw_measurement_sigma()
        bias = self._hcw_measurement_bias()
        measurement = ideal + bias + self.sensor.rng.normal(0.0, sigma, size=ideal.size)
        self.last_measurement_vector = np.array(measurement, dtype=float).reshape(-1)
        self.measurement_count += 1
        self.last_measurement_t_s = float(t_s)
        self.detected_count += 1
        if not self.last_detected and (self.detected_count > 1):
            self.reacquisition_count += 1
        self.last_detected = True
        self.consecutive_missed_steps = 0
        self.time_since_last_detection_s_values.append(0.0)
        return Measurement(vector=measurement, t_s=t_s)

    def _initial_hcw_state_from_measurement_or_config(self, measurement: Measurement) -> np.ndarray:
        if self.initial_state_ric is not None:
            return np.array(self.initial_state_ric, dtype=float).reshape(6)
        model = normalize_hcw_measurement_model(self.measurement_model)
        if model == "relative_state":
            sign = -1.0 if self.hcw_measurement_origin == "deputy" else 1.0
            return sign * np.array(measurement.vector, dtype=float).reshape(6)
        raise ValueError(
            f"tracked target {self.target_id!r} uses estimator={self.estimator_type!r} with "
            f"measurement_model={self.measurement_model!r}; relative-only RPO tracking requires "
            "ekf.initial_state_ric or estimation.initial_state_ric."
        )

    def _target_belief_from_relative(
        self,
        observer_truth: StateTruth,
        observer_belief: StateBelief | None,
    ) -> StateBelief | None:
        if self.relative_belief is None:
            return None
        # Publish an approximate target ECI belief for existing consumers. The
        # relative RIC state remains the authoritative estimator state; this
        # conversion uses the observer local RIC frame and is valid for small RPO
        # separations.
        observer_state = np.hstack((observer_truth.position_eci_km, observer_truth.velocity_eci_km_s))
        observer_covariance = np.zeros((6, 6), dtype=float)
        if observer_belief is not None:
            candidate_state = np.asarray(observer_belief.state, dtype=float).reshape(-1)
            candidate_covariance = np.asarray(observer_belief.covariance, dtype=float)
            if candidate_state.size >= 6 and np.all(np.isfinite(candidate_state[:6])):
                observer_state = candidate_state[:6].copy()
            if candidate_covariance.shape[0] >= 6 and candidate_covariance.shape[1] >= 6:
                covariance_6x6 = candidate_covariance[:6, :6]
                if np.all(np.isfinite(covariance_6x6)):
                    observer_covariance = covariance_6x6.copy()
        relative_state = np.array(self.relative_belief.state, dtype=float).reshape(6)
        target_from_observer_ric = -relative_state
        target_state = ric_rect_state_to_eci(target_from_observer_ric, observer_state[:3], observer_state[3:])
        covariance = _relative_covariance_to_published_eci(
            relative_state,
            np.array(self.relative_belief.covariance, dtype=float),
            observer_state,
            observer_covariance,
        )
        return StateBelief(
            state=target_state,
            covariance=covariance,
            last_update_t_s=float(self.relative_belief.last_update_t_s),
        )

    def _hcw_mean_motion(self, reference_state_eci_km_s: np.ndarray) -> float:
        if self.hcw_mean_motion_rad_s is not None:
            return float(self.hcw_mean_motion_rad_s)
        r = float(np.linalg.norm(np.array(reference_state_eci_km_s, dtype=float).reshape(6)[:3]))
        if r <= 0.0:
            raise ValueError("Cannot derive HCW mean motion from zero reference radius.")
        return float(np.sqrt(EARTH_MU_KM3_S2 / (r * r * r)))

    def _hcw_measurement_sigma(self) -> np.ndarray:
        model = normalize_hcw_measurement_model(self.measurement_model)
        if model == "relative_state":
            return np.hstack((_expand3(self.sensor.noise.pos_sigma_km), _expand3(self.sensor.noise.vel_sigma_km_s)))
        return _relative_measurement_sigma(model, self.sensor.noise)

    def _hcw_measurement_bias(self) -> np.ndarray:
        model = normalize_hcw_measurement_model(self.measurement_model)
        if model == "relative_state":
            return np.hstack((_expand3(self.sensor.noise.pos_bias_km), _expand3(self.sensor.noise.vel_bias_km_s)))
        return _relative_measurement_bias(model, self.sensor.noise)

    def _hcw_meas_noise_diag(self) -> np.ndarray:
        sigma = self._hcw_measurement_sigma()
        expected = hcw_measurement_dimension(self.measurement_model)
        if sigma.size != expected:
            raise ValueError(f"HCW measurement noise shape mismatch: expected {expected}, got {sigma.size}.")
        return np.maximum(sigma**2, 1e-18)

    def _relative_ekf_update(
        self,
        belief: StateBelief,
        measurement: Measurement | None,
        observer_truth: StateTruth,
        t_s: float,
    ) -> StateBelief:
        assert self.estimator is not None
        predicted = self.estimator.update(belief, None, t_s)
        if measurement is None:
            return predicted
        model = _normalize_measurement_model(self.measurement_model)
        sensor_position_eci_km, sensor_velocity_eci_km_s, _ = self.sensor._sensor_state_eci(observer_truth)
        observer_state = np.hstack((sensor_position_eci_km, sensor_velocity_eci_km_s))
        z = np.asarray(measurement.vector, dtype=float).reshape(-1)
        h_pred, h_jac = _relative_measurement_and_jacobian(
            model,
            predicted.state,
            observer_state,
            numeric_backend=self.sensor.numeric_backend,
        )
        r = np.diag(_relative_measurement_sigma(model, self.sensor.noise) ** 2)
        innovation = _relative_innovation(model, z, h_pred)
        s = h_jac @ predicted.covariance @ h_jac.T + r
        native_result = None
        # Weakly observed and angular tracks amplify gain-solve roundoff over
        # repeated updates. Use the native solve only for noise-dominated
        # range models; retain the established solve elsewhere.
        if (self.sensor.numeric_backend == "rust" and model in {"relative_range", "relative_range_rate"}
                and np.all(np.diag(r) >= 0.9 * np.diag(s))):
            from sim.rust_estimation_backend import ekf_update_innovation

            try:
                native_result = ekf_update_innovation(
                    predicted.state, predicted.covariance, innovation, h_jac, r,
                )
            except ValueError as exc:
                # A singular innovation covariance retains the Python owner's
                # pseudoinverse policy, as in the ordinary orbit EKF path.
                if "matrix must be positive definite" not in str(exc):
                    raise
        if native_result is not None:
            x_upd, p_upd, _native_innovation, _native_s, nis = native_result
        else:
            hp_t = predicted.covariance @ h_jac.T
            try:
                k_gain = np.linalg.solve(s.T, hp_t.T).T
                s_y = np.linalg.solve(s, innovation)
            except np.linalg.LinAlgError:
                s_pinv = np.linalg.pinv(s)
                k_gain = hp_t @ s_pinv
                s_y = s_pinv @ innovation
            x_upd = predicted.state + k_gain @ innovation
            i_kh = np.eye(predicted.state.size) - k_gain @ h_jac
            p_upd = i_kh @ predicted.covariance @ i_kh.T + k_gain @ r @ k_gain.T
            p_upd = 0.5 * (p_upd + p_upd.T)
            nis = float(innovation.T @ s_y)
        self.estimator.last_update_diagnostics = OrbitEKFUpdateDiagnostics(
            measurement_available=True,
            update_applied=True,
            innovation=np.array(innovation, dtype=float),
            innovation_covariance=np.array(s, dtype=float),
            nis=float(nis),
            predicted_cov_trace=float(np.trace(predicted.covariance)),
            posterior_cov_trace=float(np.trace(p_upd)),
        )
        return StateBelief(state=x_upd, covariance=p_upd, last_update_t_s=t_s)

    def _record_consistency(self, target_truth: StateTruth, diag: object | None, t_s: float) -> None:
        if self.belief is None:
            return
        err = np.array(self.belief.state[:6], dtype=float) - np.hstack(
            (target_truth.position_eci_km, target_truth.velocity_eci_km_s)
        )
        pos_err = err[:3]
        vel_err = err[3:6]
        self.pos_error_norm_km_values.append(float(np.linalg.norm(pos_err)))
        self.vel_error_norm_km_s_values.append(float(np.linalg.norm(vel_err)))
        age_s = float(t_s - self.last_measurement_t_s) if self.last_measurement_t_s is not None else float("nan")
        self.track_age_s_values.append(age_s)
        cov = np.array(self.belief.covariance, dtype=float)
        if cov.shape == (6, 6):
            try:
                nees = float(err.T @ np.linalg.solve(cov, err))
                if np.isfinite(nees):
                    self.nees_values.append(nees)
            except np.linalg.LinAlgError:
                pass
        if diag is None:
            return
        nis = float(getattr(diag, "nis", float("nan")))
        if np.isfinite(nis):
            self.nis_values.append(nis)
        innovation = np.array(getattr(diag, "innovation", np.full(6, np.nan)), dtype=float).reshape(-1)
        if innovation.size:
            innovation_norm = float(np.linalg.norm(innovation))
            if np.isfinite(innovation_norm):
                self.innovation_norm_values.append(innovation_norm)
        self.maneuver_detector.update(diag, t_s=float(t_s))

    def consistency_summary(self) -> dict[str, KnowledgeSummaryValue]:
        update_rate = float(self.update_count / max(self.step_count, 1))
        measurement_rate = float(self.measurement_count / max(self.step_count, 1))
        summary: dict[str, KnowledgeSummaryValue] = {
            "step_count": int(self.step_count),
            "initialization_count": int(self.initialization_count),
            "measurement_count": int(self.measurement_count),
            "update_count": int(self.update_count),
            "measurement_rate": measurement_rate,
            "update_rate": update_rate,
            "nis_mean": _safe_stat_mean(self.nis_values),
            "nis_p95": _safe_stat_percentile(self.nis_values, 95.0),
            "nees_mean": _safe_stat_mean(self.nees_values),
            "nees_p95": _safe_stat_percentile(self.nees_values, 95.0),
            "innovation_norm_mean": _safe_stat_mean(self.innovation_norm_values),
            "innovation_norm_p95": _safe_stat_percentile(self.innovation_norm_values, 95.0),
            "pos_error_rms_km": _safe_stat_rms(self.pos_error_norm_km_values),
            "vel_error_rms_km_s": _safe_stat_rms(self.vel_error_norm_km_s_values),
            "track_age_s_mean": _safe_stat_mean(self.track_age_s_values),
            "track_age_s_p95": _safe_stat_percentile(self.track_age_s_values, 95.0),
        }
        if self.estimator is not None:
            recovery = getattr(self.estimator.last_update_diagnostics, "recovery_diagnostics", None)
            if recovery is not None:
                summary["innovation_recovery_classification"] = recovery.classification
                summary["innovation_recovery_action"] = recovery.action
        maneuver = self.maneuver_detector.summary()
        for key in (
            "recovery_hold_count", "recovery_reinitialization_count", "recovery_transient_count",
            "last_recovery_event_t_s", "last_recovery_decision_t_s",
        ):
            if key in maneuver:
                summary[key] = maneuver[key]
        summary.update(
            {
                "maneuver_detection_enabled": bool(maneuver.get("enabled", False)),
                "maneuver_detection_status": str(maneuver.get("status", "disabled")),
                "maneuver_detection_sample_count": int(maneuver.get("sample_count", 0) or 0),
                "maneuver_warning_sample_count": int(maneuver.get("warning_sample_count", 0) or 0),
                "maneuver_detection_sample_count_above_threshold": int(maneuver.get("detection_sample_count", 0) or 0),
                "maneuver_suspect_event_count": int(maneuver.get("suspect_event_count", 0) or 0),
                "maneuver_confirmed_event_count": int(maneuver.get("confirmed_event_count", 0) or 0),
                "maneuver_first_suspect_t_s": maneuver.get("first_suspect_t_s"),
                "maneuver_first_confirmed_t_s": maneuver.get("first_confirmed_t_s"),
                "maneuver_max_nis": maneuver.get("max_nis"),
            }
        )
        return summary

    def detection_summary(self) -> dict[str, float | int | dict[str, int] | None]:
        detection_rate = float(self.detected_count / max(self.step_count, 1))
        nondetection_rate = float(1.0 - detection_rate)
        return {
            "step_count": int(self.step_count),
            "detected_count": int(self.detected_count),
            "nondetected_count": int(max(self.step_count - self.detected_count, 0)),
            "detection_rate": detection_rate,
            "nondetection_rate": nondetection_rate,
            "reacquisition_count": int(self.reacquisition_count),
            "loss_of_detection_count": int(self.loss_of_detection_count),
            "max_consecutive_missed_steps": int(self.max_consecutive_missed_steps),
            "time_since_last_detection_s_mean": _safe_stat_mean(self.time_since_last_detection_s_values),
            "time_since_last_detection_s_p95": _safe_stat_percentile(self.time_since_last_detection_s_values, 95.0),
            "status_counts": {str(k): int(v) for k, v in sorted(self.detection_status_counts.items())},
        }


class ObjectKnowledgeBase:
    def __init__(
        self,
        observer_id: str,
        tracked_objects: list[TrackedObjectConfig],
        dt_s: float,
        rng: np.random.Generator | None = None,
        mu_km3_s2: float = EARTH_MU_KM3_S2,
    ):
        self.observer_id = observer_id
        self._rng = np.random.default_rng() if rng is None else rng
        self._tracks: dict[str, _Track] = {}

        target_ids = [str(cfg.target_id) for cfg in tracked_objects if str(cfg.target_id) != str(observer_id)]
        duplicates = sorted({target_id for target_id in target_ids if target_ids.count(target_id) > 1})
        if duplicates:
            raise ValueError(f"tracked object target_id values must be unique; duplicates: {', '.join(duplicates)}.")

        for i, cfg in enumerate(tracked_objects):
            if cfg.target_id == observer_id:
                continue
            estimator_type = _normalize_estimator_type(cfg.estimator)
            measurement_model = (
                normalize_hcw_measurement_model(cfg.measurement_model)
                if estimator_type in {"relative_hcw_ekf", "relative_ss_j2_ekf", "relative_th_ekf", "relative_ya_ekf"}
                else _normalize_measurement_model(cfg.measurement_model)
            )
            if estimator_type not in {
                "ekf",
                "measured_state",
                "relative_hcw_ekf",
                "relative_ss_j2_ekf",
                "relative_th_ekf",
                "relative_ya_ekf",
            }:
                raise ValueError(f"Unsupported estimator '{cfg.estimator}' for target '{cfg.target_id}'.")
            if estimator_type == "measured_state" and measurement_model != "state":
                raise ValueError(
                    f"tracked target {cfg.target_id!r} uses estimator='measured_state'; "
                    "measured_state requires measurement_model='state'."
                )
            validate_recovery_config(cfg.ekf.innovation_recovery, measurement_model)
            if cfg.ekf.innovation_recovery is not None and cfg.ekf.innovation_recovery.enabled and estimator_type == "measured_state":
                raise ValueError("innovation recovery requires an EKF estimator.")
            trng = np.random.default_rng(int(self._rng.integers(0, 2**31 - 1)) + i)
            sensor = _OtherObjectStateSensor(
                cfg.conditions,
                cfg.sensor_noise,
                trng,
                numeric_backend=cfg.numeric_backend,
            )
            ekf = (
                OrbitEKFEstimator(
                    mu_km3_s2=mu_km3_s2,
                    dt_s=dt_s,
                    process_noise_diag=np.array(cfg.ekf.process_noise_diag, dtype=float),
                    meas_noise_diag=np.array(cfg.ekf.meas_noise_diag, dtype=float),
                    numeric_backend=cfg.numeric_backend,
                    innovation_recovery=cfg.ekf.innovation_recovery,
                )
                if estimator_type == "ekf"
                else None
            )
            self._tracks[cfg.target_id] = _Track(
                target_id=cfg.target_id,
                sensor=sensor,
                estimator=ekf,
                estimator_type=estimator_type,
                measurement_model=measurement_model,
                init_cov_diag=np.array(cfg.ekf.init_cov_diag, dtype=float),
                process_noise_diag=np.array(cfg.ekf.process_noise_diag, dtype=float),
                estimator_dt_s=float(dt_s),
                initial_state_eci_km_s=(
                    None
                    if cfg.ekf.initial_state_eci_km_s is None
                    else np.array(cfg.ekf.initial_state_eci_km_s, dtype=float).reshape(6)
                ),
                initial_state_ric=(
                    None
                    if cfg.ekf.initial_state_ric is None
                    else np.array(cfg.ekf.initial_state_ric, dtype=float).reshape(6)
                ),
                hcw_mean_motion_rad_s=cfg.ekf.mean_motion_rad_s,
                hcw_measurement_origin=str(cfg.ekf.measurement_origin),
                th_integration_substep_s=float(cfg.ekf.integration_substep_s),
                innovation_recovery=cfg.ekf.innovation_recovery,
                maneuver_detector=EKFManeuverDetector(cfg.maneuver_detection),
            )

    def target_ids(self) -> list[str]:
        return sorted(self._tracks.keys())

    def update(
        self,
        observer_truth: StateTruth,
        world_truth: dict[str, StateTruth],
        t_s: float,
        observer_belief: StateBelief | None = None,
    ) -> dict[str, StateBelief]:
        out: dict[str, StateBelief] = {}
        for target_id, track in self._tracks.items():
            tgt = world_truth.get(target_id)
            if tgt is None:
                continue
            b = track.step(observer_truth, tgt, t_s, observer_belief)
            if b is not None:
                out[target_id] = b
        return out

    def snapshot(self) -> dict[str, StateBelief]:
        out: dict[str, StateBelief] = {}
        for target_id, track in self._tracks.items():
            if track.belief is not None:
                out[target_id] = track.belief
        return out

    def measurement_snapshot(self) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        for target_id, track in self._tracks.items():
            if track.last_measurement_vector is not None:
                out[target_id] = np.array(track.last_measurement_vector, dtype=float).reshape(-1)
        return out

    def consistency_summary(self) -> dict[str, dict[str, KnowledgeSummaryValue]]:
        return {str(target_id): track.consistency_summary() for target_id, track in sorted(self._tracks.items())}

    def detection_summary(self) -> dict[str, dict[str, float | int | dict[str, int] | None]]:
        return {str(target_id): track.detection_summary() for target_id, track in sorted(self._tracks.items())}


def _expand3(v: np.ndarray) -> np.ndarray:
    a = np.array(v, dtype=float).reshape(-1)
    if a.size == 1:
        return np.full(3, float(a[0]))
    if a.size == 3:
        return a
    raise ValueError("Expected scalar or length-3 array.")


def _observer_relative_to_target_ric(observer_truth: StateTruth, target_truth: StateTruth) -> np.ndarray:
    observer_state = np.hstack((observer_truth.position_eci_km, observer_truth.velocity_eci_km_s))
    target_state = np.hstack((target_truth.position_eci_km, target_truth.velocity_eci_km_s))
    return eci_relative_to_ric_rect(observer_state, target_state)


def _relative_covariance_to_published_eci(
    relative_state_ric: np.ndarray,
    covariance_ric: np.ndarray,
    observer_state_eci_km_s: np.ndarray,
    observer_covariance_eci: np.ndarray | None = None,
) -> np.ndarray:
    rel = np.asarray(relative_state_ric, dtype=float).reshape(6)
    cov = np.asarray(covariance_ric, dtype=float).reshape(6, 6)
    observer = np.asarray(observer_state_eci_km_s, dtype=float).reshape(6)
    eps = np.array([1.0e-5, 1.0e-5, 1.0e-5, 1.0e-8, 1.0e-8, 1.0e-8], dtype=float)

    def publish_state(state_ric: np.ndarray) -> np.ndarray:
        return ric_rect_state_to_eci(-state_ric, observer[:3], observer[3:])

    jac = np.zeros((6, 6), dtype=float)
    for idx, step in enumerate(eps):
        plus = rel.copy()
        minus = rel.copy()
        plus[idx] += step
        minus[idx] -= step
        jac[:, idx] = (publish_state(plus) - publish_state(minus)) / (2.0 * step)
    out = jac @ cov @ jac.T
    if observer_covariance_eci is not None:
        observer_covariance = np.asarray(observer_covariance_eci, dtype=float).reshape(6, 6)
        observer_jac = np.zeros((6, 6), dtype=float)
        for idx, step in enumerate(eps):
            plus = observer.copy()
            minus = observer.copy()
            plus[idx] += step
            minus[idx] -= step
            plus_state = ric_rect_state_to_eci(-rel, plus[:3], plus[3:])
            minus_state = ric_rect_state_to_eci(-rel, minus[:3], minus[3:])
            observer_jac[:, idx] = (plus_state - minus_state) / (2.0 * step)
        out += observer_jac @ observer_covariance @ observer_jac.T
    return 0.5 * (out + out.T)


def _normalize_estimator_type(value: str) -> str:
    raw = str(value or "ekf").strip().lower().replace("-", "_")
    aliases = {
        "measured": "measured_state",
        "sensor": "measured_state",
        "sensor_state": "measured_state",
        "trust_sensors": "measured_state",
        "hcw": "relative_hcw_ekf",
        "hcw_ekf": "relative_hcw_ekf",
        "relative_hcw": "relative_hcw_ekf",
        "relative_hcw_filter": "relative_hcw_ekf",
        "ss": "relative_ss_j2_ekf",
        "ss_j2": "relative_ss_j2_ekf",
        "ss_j2_ekf": "relative_ss_j2_ekf",
        "schweighart_sedwick": "relative_ss_j2_ekf",
        "relative_ss_j2": "relative_ss_j2_ekf",
        "relative_ss_j2_filter": "relative_ss_j2_ekf",
        "th": "relative_th_ekf",
        "relative_th": "relative_th_ekf",
        "relative_th_filter": "relative_th_ekf",
        "tschauner_hempel": "relative_th_ekf",
        "thya": "relative_ya_ekf",
        "thya_ekf": "relative_ya_ekf",
        "relative_thya": "relative_ya_ekf",
        "relative_thya_ekf": "relative_ya_ekf",
        "relative_thya_filter": "relative_ya_ekf",
        "ya": "relative_ya_ekf",
        "ya_ekf": "relative_ya_ekf",
        "relative_ya": "relative_ya_ekf",
        "relative_ya_filter": "relative_ya_ekf",
        "yamanaka_ankersen": "relative_ya_ekf",
    }
    return aliases.get(raw, raw)


def _state_measurement_covariance(noise: KnowledgeNoiseConfig) -> np.ndarray:
    sigmas = np.hstack((_expand3(noise.pos_sigma_km), _expand3(noise.vel_sigma_km_s)))
    variances = np.maximum(sigmas.astype(float) ** 2, 1.0e-18)
    return np.diag(variances)


def _normalize_measurement_model(model: str) -> str:
    raw = str(model or "state").strip().lower().replace("-", "_")
    aliases = {
        "full_state": "state",
        "eci_state": "state",
        "range": "relative_range",
        "range_rate": "relative_range_rate",
        "angles": "relative_angles",
        "angles_range": "relative_angles_range",
        "angles_range_rate": "relative_angles_range_rate",
    }
    normalized = aliases.get(raw, raw)
    valid = {
        "state",
        "relative_range",
        "relative_range_rate",
        "relative_angles",
        "relative_angles_range",
        "relative_angles_range_rate",
    }
    if normalized not in valid:
        valid_txt = ", ".join(sorted(valid))
        raise ValueError(f"Unsupported knowledge measurement_model '{model}'. Valid options: {valid_txt}")
    return normalized


def _relative_measurement_vector(
    model: str,
    target_state: np.ndarray,
    observer_state: np.ndarray,
    *,
    numeric_backend: str = "rust",
) -> np.ndarray:
    backend = _normalize_tracking_numeric_backend(numeric_backend)
    if backend == "rust" and model in {
        "relative_range",
        "relative_range_rate",
        "relative_angles",
        "relative_angles_range",
        "relative_angles_range_rate",
    }:
        from sim.rust_tracking_backend import relative_measurement

        return relative_measurement(target_state, observer_state, model)
    x = np.asarray(target_state, dtype=float).reshape(-1)
    obs = np.asarray(observer_state, dtype=float).reshape(-1)
    rel_r = x[:3] - obs[:3]
    rel_v = x[3:6] - obs[3:6]
    rng_km = float(np.linalg.norm(rel_r))
    if rng_km <= 0.0:
        los = np.zeros(3)
        range_rate = 0.0
    else:
        los = rel_r / rng_km
        range_rate = float(np.dot(rel_v, los))
    az = float(np.arctan2(los[1], los[0])) if rng_km > 0.0 else 0.0
    el = float(np.arcsin(np.clip(los[2], -1.0, 1.0))) if rng_km > 0.0 else 0.0
    if model == "relative_range":
        return np.array([rng_km], dtype=float)
    if model == "relative_range_rate":
        return np.array([rng_km, range_rate], dtype=float)
    if model == "relative_angles":
        return np.array([az, el], dtype=float)
    if model == "relative_angles_range":
        return np.array([az, el, rng_km], dtype=float)
    if model == "relative_angles_range_rate":
        return np.array([az, el, rng_km, range_rate], dtype=float)
    raise ValueError(f"Unsupported relative measurement model '{model}'.")


def _relative_measurement_sigma(model: str, noise: KnowledgeNoiseConfig) -> np.ndarray:
    if model == "relative_range":
        return np.array([float(noise.range_sigma_km)], dtype=float)
    if model == "relative_range_rate":
        return np.array([float(noise.range_sigma_km), float(noise.range_rate_sigma_km_s)], dtype=float)
    if model == "relative_angles":
        return np.array([float(noise.angle_sigma_rad), float(noise.angle_sigma_rad)], dtype=float)
    if model == "relative_angles_range":
        return np.array(
            [float(noise.angle_sigma_rad), float(noise.angle_sigma_rad), float(noise.range_sigma_km)], dtype=float
        )
    if model == "relative_angles_range_rate":
        return np.array(
            [
                float(noise.angle_sigma_rad),
                float(noise.angle_sigma_rad),
                float(noise.range_sigma_km),
                float(noise.range_rate_sigma_km_s),
            ],
            dtype=float,
        )
    return np.hstack((_expand3(noise.pos_sigma_km), _expand3(noise.vel_sigma_km_s)))


def _relative_measurement_bias(model: str, noise: KnowledgeNoiseConfig) -> np.ndarray:
    if model == "relative_range":
        return np.array([float(noise.range_bias_km)], dtype=float)
    if model == "relative_range_rate":
        return np.array([float(noise.range_bias_km), float(noise.range_rate_bias_km_s)], dtype=float)
    if model == "relative_angles":
        return np.array([float(noise.az_bias_rad), float(noise.el_bias_rad)], dtype=float)
    if model == "relative_angles_range":
        return np.array([float(noise.az_bias_rad), float(noise.el_bias_rad), float(noise.range_bias_km)], dtype=float)
    if model == "relative_angles_range_rate":
        return np.array(
            [
                float(noise.az_bias_rad),
                float(noise.el_bias_rad),
                float(noise.range_bias_km),
                float(noise.range_rate_bias_km_s),
            ],
            dtype=float,
        )
    return np.hstack((_expand3(noise.pos_bias_km), _expand3(noise.vel_bias_km_s)))


def _relative_measurement_jacobian(
    model: str,
    target_state: np.ndarray,
    observer_state: np.ndarray,
    *,
    numeric_backend: str = "rust",
) -> np.ndarray:
    backend = _normalize_tracking_numeric_backend(numeric_backend)
    if backend == "rust" and model in {
        "relative_range",
        "relative_range_rate",
        "relative_angles",
        "relative_angles_range",
        "relative_angles_range_rate",
    }:
        from sim.rust_tracking_backend import relative_measurement_and_jacobian

        _, jacobian = relative_measurement_and_jacobian(target_state, observer_state, model)
        return jacobian
    x = np.asarray(target_state, dtype=float).reshape(-1)
    h0 = _relative_measurement_vector(model, x, observer_state, numeric_backend=backend)
    jac = np.zeros((h0.size, x.size))
    eps = np.array([1e-3, 1e-3, 1e-3, 1e-6, 1e-6, 1e-6], dtype=float)
    for i in range(min(6, x.size)):
        xp = x.copy()
        xp[i] += eps[i]
        hp = _relative_measurement_vector(model, xp, observer_state, numeric_backend=backend)
        jac[:, i] = _relative_innovation(model, hp, h0) / eps[i]
    return jac


def _relative_measurement_and_jacobian(
    model: str,
    target_state: np.ndarray,
    observer_state: np.ndarray,
    *,
    numeric_backend: str = "rust",
) -> tuple[np.ndarray, np.ndarray]:
    backend = _normalize_tracking_numeric_backend(numeric_backend)
    if backend == "rust" and model in {
        "relative_range",
        "relative_range_rate",
        "relative_angles",
        "relative_angles_range",
        "relative_angles_range_rate",
    }:
        from sim.rust_tracking_backend import relative_measurement_and_jacobian

        return relative_measurement_and_jacobian(target_state, observer_state, model)
    return (
        _relative_measurement_vector(model, target_state, observer_state, numeric_backend=backend),
        _relative_measurement_jacobian(model, target_state, observer_state, numeric_backend=backend),
    )


def _relative_innovation(model: str, z: np.ndarray, h: np.ndarray) -> np.ndarray:
    innovation = np.asarray(z, dtype=float).reshape(-1) - np.asarray(h, dtype=float).reshape(-1)
    if model in {"relative_angles", "relative_angles_range", "relative_angles_range_rate"} and innovation.size >= 2:
        innovation[0] = _wrap_angle_rad(float(innovation[0]))
        innovation[1] = _wrap_angle_rad(float(innovation[1]))
    return innovation


def _wrap_angle_rad(value: float) -> float:
    return float((value + np.pi) % (2.0 * np.pi) - np.pi)


def _safe_stat_array(values: Sequence[float]) -> np.ndarray:
    arr = np.array(values, dtype=float)
    return arr[np.isfinite(arr)]


def _safe_stat_mean(values: Sequence[float]) -> float | None:
    arr = _safe_stat_array(values)
    return float(np.mean(arr)) if arr.size else None


def _safe_stat_percentile(values: Sequence[float], pct: float) -> float | None:
    arr = _safe_stat_array(values)
    return float(np.percentile(arr, pct)) if arr.size else None


def _safe_stat_rms(values: Sequence[float]) -> float | None:
    arr = _safe_stat_array(values)
    return float(np.sqrt(np.mean(arr**2))) if arr.size else None
