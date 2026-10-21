"""Tracked-object knowledge and EKF construction."""

from __future__ import annotations

from typing import Any

import numpy as np

from sim.dynamics.orbit.environment import EARTH_MU_KM3_S2
from sim.estimation.innovation_recovery import InnovationRecoveryConfig
from sim.estimation.maneuver_detection import EKFManeuverDetectionConfig
from sim.knowledge.object_tracking import (
    KnowledgeConditionConfig,
    KnowledgeEKFConfig,
    KnowledgeNoiseConfig,
    ObjectKnowledgeBase,
    TrackedObjectConfig,
)
from sim.numeric_backend import normalize_numeric_backend


def _knowledge_ekf_diag(value: Any, default: list[float]) -> np.ndarray:
    arr = np.array(value if value is not None else default, dtype=float).reshape(-1)
    if arr.size != 6:
        return np.array(default, dtype=float)
    return arr


def _knowledge_maneuver_detection_config(value: Any) -> EKFManeuverDetectionConfig:
    raw = dict(value or {})
    return EKFManeuverDetectionConfig(
        enabled=bool(raw.get("enabled", False)),
        warning_probability=float(raw.get("warning_probability", 0.99)),
        detection_probability=float(raw.get("detection_probability", 0.999)),
        window_size=int(raw.get("window_size", 5)),
        warning_count=int(raw.get("warning_count", 3)),
        detection_count=int(raw.get("detection_count", 3)),
        min_updates=int(raw.get("min_updates", 3)),
        cooldown_updates=int(raw.get("cooldown_updates", 0)),
    )


def _build_knowledge_base(
    observer_id: str, agent_cfg: Any, dt_s: float, rng: np.random.Generator
) -> ObjectKnowledgeBase | None:
    knowledge = dict(agent_cfg.knowledge or {})
    targets = list(knowledge.get("targets", []) or [])
    if not targets:
        return None
    noise = dict(knowledge.get("sensor_error", {}) or {})
    estimation = dict(knowledge.get("estimation", {}) or {})
    ekf_cfg = dict(estimation.get("ekf", knowledge.get("ekf", {})) or {})
    numeric_backend = normalize_numeric_backend(
        estimation.get(
            "numeric_backend",
            ekf_cfg.get("numeric_backend", knowledge.get("numeric_backend", "rust")),
        ),
        field_name=f"objects.{observer_id}.knowledge.numeric_backend",
    )
    maneuver_detection_cfg = dict(
        estimation.get("maneuver_detection", ekf_cfg.get("maneuver_detection", knowledge.get("maneuver_detection", {})))
        or {}
    )
    initial_track_state = ekf_cfg.get("initial_state_eci_km_s", estimation.get("initial_state_eci_km_s"))
    tracked: list[TrackedObjectConfig] = []
    for target_id in targets:
        tracked.append(
            TrackedObjectConfig(
                target_id=str(target_id),
                conditions=KnowledgeConditionConfig.from_knowledge(knowledge, default_period_s=dt_s),
                sensor_noise=KnowledgeNoiseConfig(
                    pos_sigma_km=np.array(noise.get("pos_sigma_km", [0.01, 0.01, 0.01]), dtype=float),
                    vel_sigma_km_s=np.array(noise.get("vel_sigma_km_s", [1e-4, 1e-4, 1e-4]), dtype=float),
                    pos_bias_km=np.array(noise.get("pos_bias_km", [0.0, 0.0, 0.0]), dtype=float),
                    vel_bias_km_s=np.array(noise.get("vel_bias_km_s", [0.0, 0.0, 0.0]), dtype=float),
                    range_sigma_km=float(noise.get("range_sigma_km", 0.01)),
                    range_rate_sigma_km_s=float(noise.get("range_rate_sigma_km_s", 1e-4)),
                    angle_sigma_rad=float(noise.get("angle_sigma_rad", 1e-4)),
                    range_bias_km=float(noise.get("range_bias_km", 0.0)),
                    range_rate_bias_km_s=float(noise.get("range_rate_bias_km_s", 0.0)),
                    az_bias_rad=float(noise.get("az_bias_rad", 0.0)),
                    el_bias_rad=float(noise.get("el_bias_rad", 0.0)),
                ),
                estimator=str(estimation.get("type", "ekf")),
                measurement_model=str(estimation.get("measurement_model", "state")),
                ekf=KnowledgeEKFConfig(
                    process_noise_diag=_knowledge_ekf_diag(
                        ekf_cfg.get("process_noise_diag"),
                        [1e-8, 1e-8, 1e-8, 1e-10, 1e-10, 1e-10],
                    ),
                    meas_noise_diag=_knowledge_ekf_diag(
                        ekf_cfg.get("meas_noise_diag"),
                        [1e-6, 1e-6, 1e-6, 1e-10, 1e-10, 1e-10],
                    ),
                    init_cov_diag=_knowledge_ekf_diag(
                        ekf_cfg.get("init_cov_diag"),
                        [1.0, 1.0, 1.0, 1e-2, 1e-2, 1e-2],
                    ),
                    initial_state_eci_km_s=(
                        None if initial_track_state is None else np.array(initial_track_state, dtype=float).reshape(6)
                    ),
                    initial_state_ric=(
                        None
                        if ekf_cfg.get("initial_state_ric", estimation.get("initial_state_ric")) is None
                        else np.array(
                            ekf_cfg.get("initial_state_ric", estimation.get("initial_state_ric")), dtype=float
                        ).reshape(6)
                    ),
                    mean_motion_rad_s=(
                        None
                        if ekf_cfg.get("mean_motion_rad_s", estimation.get("mean_motion_rad_s")) is None
                        else float(ekf_cfg.get("mean_motion_rad_s", estimation.get("mean_motion_rad_s")))
                    ),
                    measurement_origin=str(
                        ekf_cfg.get("measurement_origin", estimation.get("measurement_origin", "deputy"))
                    ),
                    integration_substep_s=float(ekf_cfg.get("integration_substep_s", 10.0)),
                    innovation_recovery=(
                        None if ekf_cfg.get("innovation_recovery") is None
                        else InnovationRecoveryConfig(**dict(ekf_cfg["innovation_recovery"]))
                    ),
                ),
                maneuver_detection=_knowledge_maneuver_detection_config(maneuver_detection_cfg),
                numeric_backend=numeric_backend,
            )
        )
    return ObjectKnowledgeBase(
        observer_id=observer_id, tracked_objects=tracked, dt_s=dt_s, rng=rng, mu_km3_s2=EARTH_MU_KM3_S2
    )
