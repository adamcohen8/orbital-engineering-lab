from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from sim.control.attitude._validation import require_finite_vector
from sim.control.attitude.baseline import SmallAngleLQRController
from sim.core.interfaces import Controller
from sim.core.models import Command, StateBelief
from sim.utils.frames import ric_dcm_ir_from_rv
from sim.utils.quaternion import dcm_to_quaternion_bn


def _rot_x(a: float) -> np.ndarray:
    ca = np.cos(a)
    sa = np.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, ca, sa], [0.0, -sa, ca]])


def _rot_y(a: float) -> np.ndarray:
    ca = np.cos(a)
    sa = np.sin(a)
    return np.array([[ca, 0.0, -sa], [0.0, 1.0, 0.0], [sa, 0.0, ca]])


def _rot_z(a: float) -> np.ndarray:
    ca = np.cos(a)
    sa = np.sin(a)
    return np.array([[ca, sa, 0.0], [-sa, ca, 0.0], [0.0, 0.0, 1.0]])


@dataclass
class RICFrameLQRController(Controller):
    lqr: SmallAngleLQRController
    desired_ric_euler_rad: np.ndarray = field(default_factory=lambda: np.zeros(3))
    desired_ric_rate_rad_s: np.ndarray = field(default_factory=lambda: np.zeros(3))
    state_rv_slice: tuple[int, int] = (0, 6)

    def __post_init__(self) -> None:
        e = require_finite_vector(self.desired_ric_euler_rad, "desired_ric_euler_rad", size=3)
        w = require_finite_vector(self.desired_ric_rate_rad_s, "desired_ric_rate_rad_s", size=3)
        if self.state_rv_slice[1] - self.state_rv_slice[0] != 6:
            raise ValueError("state_rv_slice must select [r_eci(3), v_eci(3)].")
        self.desired_ric_euler_rad = e
        self.desired_ric_rate_rad_s = w

    def set_desired_ric_state(
        self,
        yaw_r_rad: float,
        roll_i_rad: float,
        pitch_c_rad: float,
        w_ric_rad_s: np.ndarray | None = None,
    ) -> None:
        self.desired_ric_euler_rad = require_finite_vector(
            [yaw_r_rad, roll_i_rad, pitch_c_rad],
            "desired_ric_euler_rad",
            size=3,
        )
        if w_ric_rad_s is not None:
            self.desired_ric_rate_rad_s = require_finite_vector(w_ric_rad_s, "w_ric_rad_s", size=3)

    def act(self, belief: StateBelief, t_s: float, budget_ms: float) -> Command:
        i0, i1 = self.state_rv_slice
        if belief.state.size < i1:
            return Command.zero()

        r_eci = require_finite_vector(belief.state[i0 : i0 + 3], "belief position", size=3)
        v_eci = require_finite_vector(belief.state[i0 + 3 : i0 + 6], "belief velocity", size=3)
        r_norm = float(np.linalg.norm(r_eci))
        if r_norm <= 0.0:
            return Command.zero()

        c_ir = ric_dcm_ir_from_rv(r_eci, v_eci)
        c_ri = c_ir.T

        yaw_r, roll_i, pitch_c = self.desired_ric_euler_rad
        c_br = _rot_z(pitch_c) @ _rot_y(roll_i) @ _rot_x(yaw_r)
        c_bn_des = c_br @ c_ri
        q_des_bn = dcm_to_quaternion_bn(c_bn_des)

        h = np.cross(r_eci, v_eci)
        omega_ri_eci = h / max(r_norm * r_norm, 1e-12)
        omega_ri_ric = c_ri @ omega_ri_eci

        # Desired body rate wrt inertial = desired body wrt RIC + RIC frame rate.
        omega_bi_des_body = c_br @ (self.desired_ric_rate_rad_s + omega_ri_ric)
        self.lqr.set_target(q_des_bn, omega_bi_des_body)
        cmd = self.lqr.act(belief, t_s, budget_ms)
        mode_flags = dict(cmd.mode_flags)
        mode_flags["mode"] = "lqr_ric"
        mode_flags["desired_ric_euler_rad"] = self.desired_ric_euler_rad.tolist()
        return Command(thrust_eci_km_s2=cmd.thrust_eci_km_s2, torque_body_nm=cmd.torque_body_nm, mode_flags=mode_flags)
