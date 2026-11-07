# ruff: noqa: F401,I001
from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from sim.config.plugin_specs import instantiate_plugin_spec
from sim.control.attitude.pose_commands import PoseCommandGenerator
from sim.control.orbit.integrated import IntegratedManeuverCommand, ManeuverStrategy, OrbitalAttitudeManeuverCoordinator
from sim.core.models import Command, StateBelief, StateTruth
from sim.dynamics.orbit.elements import coes_target_state_at_current_true_anomaly, orbital_element_feedback_accel
from sim.dynamics.orbit.two_body import propagate_two_body_rk4
from sim.rocket.models import RocketState, RocketVehicleConfig
from sim.utils.frames import eci_relative_to_ric_rect, ric_dcm_ir_from_rv, ric_rect_state_to_eci, ric_rect_to_curv
from sim.utils.quaternion import dcm_to_quaternion_bn, normalize_quaternion, quaternion_to_dcm_bn

logger = logging.getLogger(__name__)
_ATTITUDE_MANEUVER_COORDINATOR = OrbitalAttitudeManeuverCoordinator()


@dataclass
class _MissionExecutiveMode:
    name: str
    strategy: Any | None
    execution: Any | None


def _unit(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.array(v, dtype=float).reshape(3)
    n = float(np.linalg.norm(x))
    if n <= eps:
        return np.zeros(3, dtype=float)
    return x / n


def _attitude_belief_from_truth(truth: StateTruth) -> StateBelief:
    """Build the canonical full-state attitude belief used by controllers.

    Attitude controllers consume the shared ``[r, v, q, w]`` state layout.
    Mission execution can still be called without an estimator belief, so the
    truth fallback must preserve that layout instead of passing a shortened
    ``[q, w]`` vector that causes the standard controllers to return zero.
    """
    state = np.hstack(
        (
            np.asarray(truth.position_eci_km, dtype=float).reshape(3),
            np.asarray(truth.velocity_eci_km_s, dtype=float).reshape(3),
            np.asarray(truth.attitude_quat_bn, dtype=float).reshape(4),
            np.asarray(truth.angular_rate_body_rad_s, dtype=float).reshape(3),
        )
    )
    return StateBelief(
        state=state,
        covariance=np.eye(13, dtype=float) * 1e-6,
        last_update_t_s=float(truth.t_s),
    )


def _estimate_stack_delta_v_m_s(rocket_state: RocketState, vehicle_cfg: RocketVehicleConfig) -> float:
    stages = vehicle_cfg.stack.stages
    if not stages:
        return 0.0
    i0 = int(max(rocket_state.active_stage_index, 0))
    if i0 >= len(stages):
        return 0.0
    prop_rem = np.array(rocket_state.stage_prop_remaining_kg, dtype=float).reshape(-1)
    dry = np.array([s.dry_mass_kg for s in stages], dtype=float)
    isp = np.array([s.isp_s for s in stages], dtype=float)
    g0 = 9.80665
    m_cur = float(rocket_state.mass_kg)
    dv = 0.0
    for i in range(i0, len(stages)):
        mp = float(prop_rem[i]) if i < prop_rem.size else 0.0
        if mp <= 0.0:
            m_cur -= float(dry[i])
            continue
        m0 = max(m_cur, 1e-6)
        mf = max(m_cur - mp, 1e-6)
        dv += float(isp[i] * g0 * np.log(max(m0 / mf, 1.0)))
        m_cur = mf - float(dry[i])
    return float(max(dv, 0.0))


def _estimate_needed_delta_v_m_s(current_truth: StateTruth, target_truth: StateTruth | None) -> float:
    if target_truth is None:
        return np.inf
    rel_v_km_s = np.array(target_truth.velocity_eci_km_s, dtype=float) - np.array(
        current_truth.velocity_eci_km_s, dtype=float
    )
    return float(np.linalg.norm(rel_v_km_s) * 1e3)


def _available_delta_v_from_truth_mass_km_s(
    *,
    truth: StateTruth,
    dry_mass_kg: float | None,
    orbital_isp_s: float | None,
    fallback_km_s: float | None = None,
) -> float:
    if (
        dry_mass_kg is None
        or orbital_isp_s is None
        or (not np.isfinite(float(dry_mass_kg)))
        or (not np.isfinite(float(orbital_isp_s)))
        or float(dry_mass_kg) <= 0.0
        or float(orbital_isp_s) <= 0.0
    ):
        if fallback_km_s is None or not np.isfinite(float(fallback_km_s)):
            return 0.0
        return float(max(float(fallback_km_s), 0.0))
    m_cur_kg = float(max(float(truth.mass_kg), 0.0))
    m_dry_kg = float(max(float(dry_mass_kg), 0.0))
    if m_cur_kg <= m_dry_kg:
        return 0.0
    return float((float(orbital_isp_s) * 9.80665 * np.log(m_cur_kg / m_dry_kg)) / 1e3)


def _resolve_angle_tolerance_rad(rad_value: float, deg_value: float | None) -> float:
    if deg_value is not None:
        return float(max(np.deg2rad(float(deg_value)), 0.0))
    return float(max(rad_value, 0.0))


def _resolve_target_state(
    *,
    target_id: str | None,
    use_knowledge_for_targeting: bool,
    own_knowledge: dict[str, StateBelief],
) -> tuple[np.ndarray, np.ndarray] | None:
    if target_id is None:
        return None
    if use_knowledge_for_targeting and target_id in own_knowledge:
        kb = own_knowledge[target_id]
        if kb.state.size >= 6:
            return np.array(kb.state[:3], dtype=float), np.array(kb.state[3:6], dtype=float)
    return None


def _axis_unit_ric(axis_mode: str) -> np.ndarray:
    token = str(axis_mode).strip().upper().replace(" ", "")
    m = {
        "+R": np.array([1.0, 0.0, 0.0], dtype=float),
        "-R": np.array([-1.0, 0.0, 0.0], dtype=float),
        "+I": np.array([0.0, 1.0, 0.0], dtype=float),
        "-I": np.array([0.0, -1.0, 0.0], dtype=float),
        "+C": np.array([0.0, 0.0, 1.0], dtype=float),
        "-C": np.array([0.0, 0.0, -1.0], dtype=float),
    }
    if token in m:
        return m[token]
    raise ValueError("axis_mode must be one of: +R, -R, +I, -I, +C, -C")


def _set_orbit_controller_target(controller: Any | None, desired_state_eci_6: np.ndarray) -> None:
    if controller is None:
        return
    x = np.array(desired_state_eci_6, dtype=float).reshape(-1)
    if x.size != 6:
        return
    if hasattr(controller, "set_target_state"):
        try:
            controller.set_target_state(x)
            return
        except (TypeError, ValueError, AttributeError) as exc:
            logger.warning("Failed to set orbit target state via set_target_state: %s", exc)
    if hasattr(controller, "target_state"):
        try:
            controller.target_state = x
            return
        except (TypeError, ValueError, AttributeError) as exc:
            logger.warning("Failed to set orbit target state via target_state assignment: %s", exc)


def _apply_orbit_controller_intent(controller: Any | None, intent: dict[str, Any]) -> None:
    if controller is None:
        return
    rel_rect = intent.get("desired_relative_ric_rect_6")
    if rel_rect is not None and hasattr(controller, "target_rel_ric_rect"):
        try:
            controller.target_rel_ric_rect = np.array(rel_rect, dtype=float).reshape(6)
        except (TypeError, ValueError, AttributeError) as exc:
            logger.warning("Failed to set orbit controller relative target: %s", exc)
    desired_eci = intent.get("desired_state_eci_6")
    if desired_eci is not None:
        _set_orbit_controller_target(controller, np.array(desired_eci, dtype=float).reshape(6))


def _pointer_dict_to_obj(pointer: dict[str, Any] | None) -> Any | None:
    if not isinstance(pointer, dict):
        return None
    return instantiate_plugin_spec(pointer, description="nested mission plugin")


def _call_plugin_method(obj: Any | None, method_names: tuple[str, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    if obj is None:
        return {}
    for method_name in method_names:
        if not hasattr(obj, method_name):
            continue
        method = getattr(obj, method_name)
        try:
            ret = method(**kwargs)
        except TypeError:
            ret = method(truth=kwargs.get("truth"), t_s=kwargs.get("t_s", 0.0))
        if isinstance(ret, dict):
            return ret
        return {}
    return {}


def _relative_pd_accel_eci(
    *,
    truth: StateTruth,
    target_state_eci: tuple[np.ndarray, np.ndarray] | None,
    desired_relative_ric_rect: np.ndarray,
    kp_pos: float,
    kd_vel: float,
    max_accel_km_s2: float,
) -> np.ndarray:
    if target_state_eci is None:
        return np.zeros(3, dtype=float)
    x_self = np.hstack((np.array(truth.position_eci_km, dtype=float), np.array(truth.velocity_eci_km_s, dtype=float)))
    x_tgt = np.hstack((target_state_eci[0], target_state_eci[1]))
    rel_err = eci_relative_to_ric_rect(x_dep_eci=x_self, x_chief_eci=x_tgt) - np.array(
        desired_relative_ric_rect, dtype=float
    ).reshape(6)
    a_cmd_ric = -(float(kp_pos) * rel_err[:3] + float(kd_vel) * rel_err[3:6])
    nrm = float(np.linalg.norm(a_cmd_ric))
    amax = float(max(max_accel_km_s2, 0.0))
    if nrm > amax > 0.0:
        a_cmd_ric *= amax / nrm
    c_ir = ric_dcm_ir_from_rv(target_state_eci[0], target_state_eci[1])
    return c_ir @ a_cmd_ric


def _absolute_pd_accel_eci(
    *,
    truth: StateTruth,
    desired_state_eci_6: np.ndarray,
    kp_pos: float,
    kd_vel: float,
    max_accel_km_s2: float,
) -> np.ndarray:
    x_self = np.hstack((np.array(truth.position_eci_km, dtype=float), np.array(truth.velocity_eci_km_s, dtype=float)))
    x_des = np.array(desired_state_eci_6, dtype=float).reshape(6)
    a_cmd = float(kp_pos) * (x_des[:3] - x_self[:3]) + float(kd_vel) * (x_des[3:6] - x_self[3:6])
    nrm = float(np.linalg.norm(a_cmd))
    amax = float(max(max_accel_km_s2, 0.0))
    if amax <= 0.0:
        return np.zeros(3, dtype=float)
    if nrm > amax:
        a_cmd *= amax / nrm
    return a_cmd


def _resolve_desired_state_from_inputs(
    *,
    target_id: str | None,
    desired_state_source: str,
    use_knowledge_for_targeting: bool,
    desired_position_eci_km: np.ndarray | None,
    desired_velocity_eci_km_s: np.ndarray | None,
    own_knowledge: dict[str, StateBelief],
) -> tuple[np.ndarray, np.ndarray] | None:
    src = str(desired_state_source).lower()
    if src == "explicit":
        if desired_position_eci_km is None or desired_velocity_eci_km_s is None:
            return None
        return (
            np.array(desired_position_eci_km, dtype=float).reshape(3),
            np.array(desired_velocity_eci_km_s, dtype=float).reshape(3),
        )
    return _resolve_target_state(
        target_id=target_id,
        use_knowledge_for_targeting=use_knowledge_for_targeting,
        own_knowledge=own_knowledge,
    )


def _desired_attitude_for_thrust(
    *,
    truth: StateTruth,
    thrust_eci_km_s2: np.ndarray,
    thruster_direction_body: np.ndarray,
) -> np.ndarray:
    q_req = _ATTITUDE_MANEUVER_COORDINATOR.maneuverer.required_attitude_for_delta_v(
        truth=truth,
        delta_v_eci_km_s=np.array(thrust_eci_km_s2, dtype=float),
        thruster_direction_body=np.array(thruster_direction_body, dtype=float),
    )
    if q_req is None:
        return np.array(truth.attitude_quat_bn, dtype=float)
    return np.array(q_req, dtype=float)

_ORIGINAL_ESTIMATE_STACK_DELTA_V = _estimate_stack_delta_v_m_s


def _compat_estimate_stack_delta_v_m_s(rocket_state, vehicle_cfg):
    facade = sys.modules.get("sim.mission.modules")
    current = getattr(facade, "_estimate_stack_delta_v_m_s", _ORIGINAL_ESTIMATE_STACK_DELTA_V)
    if current is not _estimate_stack_delta_v_m_s:
        return current(rocket_state, vehicle_cfg)
    return _ORIGINAL_ESTIMATE_STACK_DELTA_V(rocket_state, vehicle_cfg)


__all__ = [name for name in globals() if not name.startswith("__")]
