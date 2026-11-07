"""Reference attitude-pointing controllers built on the quaternion PD law."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from sim.control.attitude._validation import require_finite_scalar
from sim.control.attitude.baseline import QuaternionPDController, _normalize_quaternion
from sim.control.attitude.pose_commands import _attitude_quat_align_primary
from sim.core.models import Command, StateBelief, StateTruth
from sim.utils.frames import ric_dcm_ir_from_rv


@dataclass
class ReferencePointingController(QuaternionPDController):
    """Point a body axis at a standard orbit-relative or supplied ECI reference."""

    pointing_mode: str = "nadir"
    primary_axis_body: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0]))
    secondary_axis_body: np.ndarray | None = None
    secondary_direction_eci: np.ndarray | None = None
    reference_direction_eci: np.ndarray | None = None
    target_position_eci_km: np.ndarray | None = None
    ric_direction: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0]))
    max_target_slew_rate_deg_s: float | None = None
    _last_target_t_s: float | None = field(default=None, init=False, repr=False)
    _commanded_quaternion: np.ndarray | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        aliases = {
            "nadir": "nadir",
            "velocity": "prograde",
            "prograde": "prograde",
            "retrograde": "retrograde",
            "sun": "sun",
            "thrust": "thrust",
            "target": "target",
            "ric": "ric",
            "fixed_eci": "fixed_eci",
            "quaternion": "quaternion",
        }
        token = str(self.pointing_mode).strip().lower()
        if token not in aliases:
            raise ValueError(f"Unknown pointing_mode {self.pointing_mode!r}; expected one of {sorted(aliases)}.")
        self.pointing_mode = aliases[token]
        self.primary_axis_body = _unit(self.primary_axis_body, "primary_axis_body")
        if self.secondary_axis_body is not None:
            self.secondary_axis_body = _unit(self.secondary_axis_body, "secondary_axis_body")
        if self.secondary_direction_eci is not None:
            self.secondary_direction_eci = _unit(self.secondary_direction_eci, "secondary_direction_eci")
        if self.reference_direction_eci is not None:
            self.reference_direction_eci = _unit(self.reference_direction_eci, "reference_direction_eci")
        if self.target_position_eci_km is not None:
            self.target_position_eci_km = _vector3(self.target_position_eci_km, "target_position_eci_km")
        self.ric_direction = _unit(self.ric_direction, "ric_direction")
        if self.max_target_slew_rate_deg_s is not None:
            self.max_target_slew_rate_deg_s = require_finite_scalar(
                self.max_target_slew_rate_deg_s,
                "max_target_slew_rate_deg_s",
                allow_positive_inf=True,
            )
            if self.max_target_slew_rate_deg_s <= 0.0:
                raise ValueError("max_target_slew_rate_deg_s must be positive when provided.")

    def set_target_direction(self, direction_eci: np.ndarray) -> None:
        self.reference_direction_eci = _unit(direction_eci, "direction_eci")

    def set_target_position(self, position_eci_km: np.ndarray) -> None:
        self.target_position_eci_km = _vector3(position_eci_km, "position_eci_km")

    def act(self, belief: StateBelief, t_s: float, budget_ms: float) -> Command:
        if belief.state.size < 13:
            return Command.zero()
        truth = StateTruth(
            position_eci_km=np.asarray(belief.state[0:3], dtype=float),
            velocity_eci_km_s=np.asarray(belief.state[3:6], dtype=float),
            attitude_quat_bn=np.asarray(belief.state[6:10], dtype=float),
            angular_rate_body_rad_s=np.asarray(belief.state[10:13], dtype=float),
            mass_kg=1.0,
            t_s=float(t_s),
        )
        direction = self._target_direction(truth)
        if direction is None:
            command = super().act(belief, t_s, budget_ms)
            command.mode_flags.update({"mode": "reference_quaternion_hold", "pointing_mode": self.pointing_mode})
            return command
        desired = _attitude_quat_align_primary(
            truth=truth,
            primary_axis_body=self.primary_axis_body,
            target_axis_eci=direction,
            secondary_axis_body=self.secondary_axis_body,
            secondary_axis_eci_hint=self.secondary_direction_eci,
        )
        desired = self._rate_limit_target(desired, t_s=float(t_s))
        self.desired_attitude_quat_bn = desired
        command = super().act(belief, t_s, budget_ms)
        command.mode_flags.update(
            {
                "mode": f"reference_{self.pointing_mode}_pointing",
                "pointing_mode": self.pointing_mode,
                "target_direction_eci": np.asarray(direction, dtype=float).tolist(),
                "desired_attitude_quat_bn": desired.tolist(),
                "target_slew_rate_limited": self.max_target_slew_rate_deg_s is not None,
            }
        )
        return command

    def _target_direction(self, truth: StateTruth) -> np.ndarray | None:
        if self.pointing_mode == "quaternion":
            return None
        if self.pointing_mode == "nadir":
            value = -np.asarray(truth.position_eci_km, dtype=float)
        elif self.pointing_mode == "prograde":
            value = np.asarray(truth.velocity_eci_km_s, dtype=float)
        elif self.pointing_mode == "retrograde":
            value = -np.asarray(truth.velocity_eci_km_s, dtype=float)
        elif self.pointing_mode == "ric":
            value = ric_dcm_ir_from_rv(truth.position_eci_km, truth.velocity_eci_km_s) @ self.ric_direction
        elif self.pointing_mode == "target":
            if self.target_position_eci_km is None:
                return None
            value = np.asarray(self.target_position_eci_km, dtype=float) - truth.position_eci_km
        else:
            if self.reference_direction_eci is None:
                return None
            value = np.asarray(self.reference_direction_eci, dtype=float)
        norm = float(np.linalg.norm(value))
        return None if norm <= 1.0e-12 else value / norm

    def _rate_limit_target(self, desired: np.ndarray, *, t_s: float) -> np.ndarray:
        desired = _normalize_quaternion(np.asarray(desired, dtype=float))
        previous = self._commanded_quaternion
        dt_s = 0.0 if self._last_target_t_s is None else max(t_s - self._last_target_t_s, 0.0)
        self._last_target_t_s = t_s
        if previous is None or self.max_target_slew_rate_deg_s is None or dt_s <= 0.0:
            self._commanded_quaternion = desired
            return desired
        if float(np.dot(previous, desired)) < 0.0:
            desired = -desired
        dot = float(np.clip(np.dot(previous, desired), -1.0, 1.0))
        angle = 2.0 * float(np.arccos(dot))
        max_angle = np.deg2rad(float(self.max_target_slew_rate_deg_s)) * dt_s
        if angle <= max_angle or angle <= 1.0e-12:
            result = desired
        else:
            fraction = float(np.clip(max_angle / angle, 0.0, 1.0))
            result = _normalize_quaternion((1.0 - fraction) * previous + fraction * desired)
        self._commanded_quaternion = result
        return result


@dataclass
class ThrustAlignController(ReferencePointingController):
    pointing_mode: str = "thrust"


@dataclass
class TargetTrackController(ReferencePointingController):
    pointing_mode: str = "target"


@dataclass
class NadirPointingController(ReferencePointingController):
    pointing_mode: str = "nadir"


@dataclass
class VelocityPointingController(ReferencePointingController):
    pointing_mode: str = "prograde"


@dataclass
class SunPointingController(ReferencePointingController):
    pointing_mode: str = "sun"


@dataclass
class RICAxisPointingController(ReferencePointingController):
    pointing_mode: str = "ric"


def _vector3(value: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float).reshape(-1)
    if result.size != 3 or not bool(np.all(np.isfinite(result))):
        raise ValueError(f"{name} must be a finite length-3 vector.")
    return result


def _unit(value: np.ndarray, name: str) -> np.ndarray:
    result = _vector3(value, name)
    norm = float(np.linalg.norm(result))
    if norm <= 0.0:
        raise ValueError(f"{name} must be nonzero.")
    return result / norm
