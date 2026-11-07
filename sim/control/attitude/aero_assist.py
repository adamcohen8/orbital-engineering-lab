from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from sim.control.attitude._validation import require_finite_scalar, require_finite_vector
from sim.control.attitude.baseline import ReactionWheelPDController
from sim.core.interfaces import Controller
from sim.core.models import Command, StateBelief
from sim.utils.frames import ric_dcm_ir_from_rv
from sim.utils.quaternion import dcm_to_quaternion_bn


def _unit3(value: Any, *, field_name: str) -> np.ndarray:
    arr = np.array(value, dtype=float).reshape(3)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{field_name} must contain finite values.")
    norm = float(np.linalg.norm(arr))
    if norm <= 0.0:
        raise ValueError(f"{field_name} must be non-zero.")
    return arr / norm


def _construct_pd(spec: Any) -> ReactionWheelPDController:
    if isinstance(spec, ReactionWheelPDController):
        return spec
    if spec is None:
        return ReactionWheelPDController()
    if isinstance(spec, dict):
        params = dict(spec.get("params", spec) or {})
        params.pop("module", None)
        params.pop("class_name", None)
        return ReactionWheelPDController(**params)
    raise TypeError("pd must be a ReactionWheelPDController, dict, or None.")


def _rotation_between_unit_vectors(source_unit: np.ndarray, target_unit: np.ndarray) -> np.ndarray:
    source = _unit3(source_unit, field_name="source_unit")
    target = _unit3(target_unit, field_name="target_unit")
    dot = float(np.clip(np.dot(source, target), -1.0, 1.0))
    if dot > 1.0 - 1e-12:
        return np.eye(3)
    if dot < -1.0 + 1e-12:
        trial = np.array([1.0, 0.0, 0.0], dtype=float)
        if abs(float(np.dot(trial, source))) > 0.9:
            trial = np.array([0.0, 1.0, 0.0], dtype=float)
        axis = _unit3(np.cross(source, trial), field_name="rotation_axis")
        return -np.eye(3) + 2.0 * np.outer(axis, axis)
    axis = np.cross(source, target)
    k = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ],
        dtype=float,
    )
    return np.eye(3) + k + (k @ k) * (1.0 / (1.0 + dot))


@dataclass
class AtmosphericLiftAxisController(Controller):
    """Point a body-frame lift axis along a requested RIC direction."""

    pd: ReactionWheelPDController | dict[str, Any] | None = None
    lift_axis_body: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))
    desired_lift_ric: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))
    desired_rate_body_rad_s: np.ndarray = field(default_factory=lambda: np.zeros(3))
    state_rv_slice: tuple[int, int] = (0, 6)
    flip_lift_after_s: float | None = None

    def __post_init__(self) -> None:
        self.pd = _construct_pd(self.pd)
        self.lift_axis_body = _unit3(self.lift_axis_body, field_name="lift_axis_body")
        self.desired_lift_ric = _unit3(self.desired_lift_ric, field_name="desired_lift_ric")
        self.desired_rate_body_rad_s = require_finite_vector(
            self.desired_rate_body_rad_s,
            "desired_rate_body_rad_s",
            size=3,
        )
        if self.flip_lift_after_s is not None:
            self.flip_lift_after_s = require_finite_scalar(self.flip_lift_after_s, "flip_lift_after_s")
        if self.state_rv_slice[1] - self.state_rv_slice[0] != 6:
            raise ValueError("state_rv_slice must select [r_eci(3), v_eci(3)].")

    def act(self, belief: StateBelief, t_s: float, budget_ms: float) -> Command:
        i0, i1 = self.state_rv_slice
        if belief.state.size < i1:
            return Command.zero()
        r_eci = np.array(belief.state[i0 : i0 + 3], dtype=float)
        v_eci = np.array(belief.state[i0 + 3 : i0 + 6], dtype=float)
        if float(np.linalg.norm(r_eci)) <= 0.0 or float(np.linalg.norm(v_eci)) <= 0.0:
            return Command.zero()

        desired_ric = np.array(self.desired_lift_ric, dtype=float)
        if self.flip_lift_after_s is not None and float(t_s) >= float(self.flip_lift_after_s):
            desired_ric = -desired_ric
        desired_lift_eci = ric_dcm_ir_from_rv(r_eci, v_eci) @ desired_ric
        desired_lift_eci = _unit3(desired_lift_eci, field_name="desired_lift_eci")

        c_nb_des = _rotation_between_unit_vectors(self.lift_axis_body, desired_lift_eci)
        q_des_bn = dcm_to_quaternion_bn(c_nb_des.T)
        assert isinstance(self.pd, ReactionWheelPDController)
        self.pd.set_target(q_des_bn, self.desired_rate_body_rad_s)
        cmd = self.pd.act(belief, t_s, budget_ms)
        mode_flags = dict(cmd.mode_flags or {})
        mode_flags.update(
            {
                "mode": "atmospheric_lift_axis",
                "desired_lift_ric": desired_ric.tolist(),
                "desired_lift_eci": desired_lift_eci.tolist(),
                "desired_attitude_quat_bn": q_des_bn.tolist(),
            }
        )
        return Command(thrust_eci_km_s2=cmd.thrust_eci_km_s2, torque_body_nm=cmd.torque_body_nm, mode_flags=mode_flags)
