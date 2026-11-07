from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from sim.control.attitude._validation import require_finite_scalar, require_finite_vector
from sim.core.interfaces import Controller
from sim.core.models import Command, StateBelief
from sim.utils.frames import ric_dcm_ir_from_rv
from sim.utils.quaternion import dcm_to_quaternion_bn, normalize_quaternion, quaternion_to_dcm_bn


def _quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    a0, a1, a2, a3 = normalize_quaternion(np.array(q1, dtype=float).reshape(4))
    b0, b1, b2, b3 = normalize_quaternion(np.array(q2, dtype=float).reshape(4))
    return normalize_quaternion(
        np.array(
            [
                a0 * b0 - a1 * b1 - a2 * b2 - a3 * b3,
                a0 * b1 + a1 * b0 + a2 * b3 - a3 * b2,
                a0 * b2 - a1 * b3 + a2 * b0 + a3 * b1,
                a0 * b3 + a1 * b2 - a2 * b1 + a3 * b0,
            ],
            dtype=float,
        )
    )


def _quat_angle_rad(q_from: np.ndarray, q_to: np.ndarray) -> float:
    qa = normalize_quaternion(np.array(q_from, dtype=float).reshape(4))
    qb = normalize_quaternion(np.array(q_to, dtype=float).reshape(4))
    d = float(abs(np.dot(qa, qb)))
    d = float(np.clip(d, -1.0, 1.0))
    return float(2.0 * np.arccos(d))


def _quat_slerp(q_from: np.ndarray, q_to: np.ndarray, alpha: float) -> np.ndarray:
    qa = normalize_quaternion(np.array(q_from, dtype=float).reshape(4))
    qb = normalize_quaternion(np.array(q_to, dtype=float).reshape(4))
    a = float(np.clip(alpha, 0.0, 1.0))
    dot = float(np.dot(qa, qb))
    if dot < 0.0:
        qb = -qb
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        return normalize_quaternion((1.0 - a) * qa + a * qb)
    theta = float(np.arccos(dot))
    s = float(np.sin(theta))
    w1 = float(np.sin((1.0 - a) * theta) / s)
    w2 = float(np.sin(a * theta) / s)
    return normalize_quaternion(w1 * qa + w2 * qb)


def _small_angle_error_quat(sigma_rad: float, rng: np.random.Generator) -> np.ndarray:
    sig = max(float(sigma_rad), 0.0)
    if sig <= 0.0:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    dtheta = rng.normal(0.0, sig, size=3)
    ang = float(np.linalg.norm(dtheta))
    if ang <= 1e-15:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    axis = dtheta / ang
    h = 0.5 * ang
    return normalize_quaternion(
        np.array([np.cos(h), axis[0] * np.sin(h), axis[1] * np.sin(h), axis[2] * np.sin(h)], dtype=float)
    )


@dataclass
class _BaseSurrogateSnap(Controller):
    cancel_rate_mag_rad_s2: float = 1.0
    rate_tolerance_rad_s: float = 1e-3
    slew_time_180_s: float = 1.0
    pointing_sigma_deg: float = 0.0
    default_dt_s: float = 1.0
    rng_seed: int = 0
    _last_t_s: float | None = field(default=None, init=False, repr=False)
    _holding: bool = field(default=False, init=False, repr=False)
    _error_applied: bool = field(default=False, init=False, repr=False)
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        require_finite_scalar(self.cancel_rate_mag_rad_s2, "cancel_rate_mag_rad_s2")
        require_finite_scalar(self.rate_tolerance_rad_s, "rate_tolerance_rad_s")
        require_finite_scalar(self.slew_time_180_s, "slew_time_180_s")
        require_finite_scalar(self.pointing_sigma_deg, "pointing_sigma_deg")
        require_finite_scalar(self.default_dt_s, "default_dt_s")
        require_finite_scalar(self.rng_seed, "rng_seed")
        self._rng = np.random.default_rng(int(self.rng_seed))

    def _dt(self, t_s: float) -> float:
        if self._last_t_s is None:
            dt = float(self.default_dt_s)
        else:
            dt = max(float(t_s) - float(self._last_t_s), 1e-9)
        self._last_t_s = float(t_s)
        return dt

    def _cancel_rate_step(self, w_body: np.ndarray, dt_s: float) -> np.ndarray:
        w = np.array(w_body, dtype=float).reshape(3)
        wm = float(np.linalg.norm(w))
        tol = max(float(self.rate_tolerance_rad_s), 0.0)
        if wm <= tol:
            return np.zeros(3, dtype=float)
        max_drop = max(float(self.cancel_rate_mag_rad_s2), 0.0) * float(dt_s)
        new_mag = max(0.0, wm - max_drop)
        if new_mag <= tol or wm <= 1e-15:
            return np.zeros(3, dtype=float)
        return w * (new_mag / wm)

    def _slew_step(self, q_now: np.ndarray, q_target: np.ndarray, dt_s: float) -> tuple[np.ndarray, bool]:
        angle = _quat_angle_rad(q_now, q_target)
        if angle <= 1e-12:
            return normalize_quaternion(q_target), True
        rate_rad_s = np.pi / max(float(self.slew_time_180_s), 1e-9)
        max_step = max(rate_rad_s * float(dt_s), 0.0)
        if angle <= max_step:
            return normalize_quaternion(q_target), True
        return _quat_slerp(q_now, q_target, max_step / angle), False

    def _inject_error_if_needed(self, q_frame: np.ndarray) -> np.ndarray:
        if self._error_applied:
            return normalize_quaternion(q_frame)
        sigma_rad = np.deg2rad(max(float(self.pointing_sigma_deg), 0.0))
        if sigma_rad <= 0.0:
            self._error_applied = True
            return normalize_quaternion(q_frame)
        q_err = _small_angle_error_quat(sigma_rad=sigma_rad, rng=self._rng)
        self._error_applied = True
        return normalize_quaternion(_quat_multiply(q_err, q_frame))


@dataclass
class SurrogateSnapECIController(_BaseSurrogateSnap):
    desired_attitude_quat_bn: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0], dtype=float))
    _hold_q_bn: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0], dtype=float), init=False, repr=False
    )
    _target_q_bn: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0], dtype=float), init=False, repr=False
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        self.set_target(self.desired_attitude_quat_bn)

    def set_target(self, desired_attitude_quat_bn: np.ndarray) -> None:
        q = normalize_quaternion(require_finite_vector(desired_attitude_quat_bn, "desired_attitude_quat_bn", size=4))
        self.desired_attitude_quat_bn = q
        self._target_q_bn = q
        self._hold_q_bn = q.copy()
        self._holding = False
        self._error_applied = False

    def act(self, belief: StateBelief, t_s: float, budget_ms: float) -> Command:
        if belief.state.size < 13:
            return Command.zero()
        q_now = normalize_quaternion(require_finite_vector(belief.state[6:10], "belief attitude quaternion", size=4))
        w_now = require_finite_vector(belief.state[10:13], "belief angular rate", size=3)
        dt_s = self._dt(t_s)

        q_attr = normalize_quaternion(
            require_finite_vector(self.desired_attitude_quat_bn, "desired_attitude_quat_bn", size=4)
        )
        if float(np.linalg.norm(q_attr - self._target_q_bn)) > 1e-12:
            self.set_target(q_attr)

        if self._holding:
            q_next = self._hold_q_bn.copy()
            w_next = np.zeros(3, dtype=float)
            phase = "hold"
        else:
            w_next = self._cancel_rate_step(w_now, dt_s)
            if float(np.linalg.norm(w_next)) > max(float(self.rate_tolerance_rad_s), 0.0):
                q_next = q_now
                phase = "rate_cancel"
            else:
                q_step, reached = self._slew_step(q_now, self._target_q_bn, dt_s)
                if reached:
                    q_step = self._inject_error_if_needed(q_step)
                    self._hold_q_bn = q_step.copy()
                    self._holding = True
                    phase = "hold"
                else:
                    phase = "slew"
                q_next = q_step
                w_next = np.zeros(3, dtype=float)

        return Command(
            thrust_eci_km_s2=np.zeros(3),
            torque_body_nm=np.zeros(3),
            mode_flags={
                "mode": "surrogate_snap_eci",
                "attitude_state_override": {
                    "q_next_bn": normalize_quaternion(q_next).tolist(),
                    "w_next_body_rad_s": np.array(w_next, dtype=float).tolist(),
                    "phase": phase,
                },
            },
        )


@dataclass
class SurrogateSnapRICController(_BaseSurrogateSnap):
    desired_attitude_quat_br: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0], dtype=float))
    _hold_q_br: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0], dtype=float), init=False, repr=False
    )
    _target_q_br: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0], dtype=float), init=False, repr=False
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        self.set_target(self.desired_attitude_quat_br)

    def set_target(self, desired_attitude_quat_br: np.ndarray) -> None:
        q = normalize_quaternion(require_finite_vector(desired_attitude_quat_br, "desired_attitude_quat_br", size=4))
        self.desired_attitude_quat_br = q
        self._target_q_br = q
        self._hold_q_br = q.copy()
        self._holding = False
        self._error_applied = False

    def act(self, belief: StateBelief, t_s: float, budget_ms: float) -> Command:
        if belief.state.size < 13:
            return Command.zero()
        r = require_finite_vector(belief.state[0:3], "belief position", size=3)
        v = require_finite_vector(belief.state[3:6], "belief velocity", size=3)
        q_now_bn = normalize_quaternion(
            require_finite_vector(belief.state[6:10], "belief attitude quaternion", size=4)
        )
        w_now = require_finite_vector(belief.state[10:13], "belief angular rate", size=3)
        dt_s = self._dt(t_s)

        q_attr = normalize_quaternion(
            require_finite_vector(self.desired_attitude_quat_br, "desired_attitude_quat_br", size=4)
        )
        if float(np.linalg.norm(q_attr - self._target_q_br)) > 1e-12:
            self.set_target(q_attr)

        c_bn = quaternion_to_dcm_bn(q_now_bn)
        c_ir = ric_dcm_ir_from_rv(r, v)
        c_br_now = c_bn @ c_ir
        q_now_br = dcm_to_quaternion_bn(c_br_now)

        if self._holding:
            q_cmd_br = self._hold_q_br.copy()
            w_next = np.zeros(3, dtype=float)
            phase = "hold"
        else:
            w_next = self._cancel_rate_step(w_now, dt_s)
            if float(np.linalg.norm(w_next)) > max(float(self.rate_tolerance_rad_s), 0.0):
                q_cmd_br = q_now_br
                phase = "rate_cancel"
            else:
                q_step_br, reached = self._slew_step(q_now_br, self._target_q_br, dt_s)
                if reached:
                    q_step_br = self._inject_error_if_needed(q_step_br)
                    self._hold_q_br = q_step_br.copy()
                    self._holding = True
                    phase = "hold"
                else:
                    phase = "slew"
                q_cmd_br = q_step_br
                w_next = np.zeros(3, dtype=float)

        c_br_cmd = quaternion_to_dcm_bn(q_cmd_br)
        c_bn_cmd = c_br_cmd @ c_ir.T
        q_next_bn = dcm_to_quaternion_bn(c_bn_cmd)

        return Command(
            thrust_eci_km_s2=np.zeros(3),
            torque_body_nm=np.zeros(3),
            mode_flags={
                "mode": "surrogate_snap_ric",
                "attitude_state_override": {
                    "q_next_bn": normalize_quaternion(q_next_bn).tolist(),
                    "w_next_body_rad_s": np.array(w_next, dtype=float).tolist(),
                    "phase": phase,
                },
            },
        )
