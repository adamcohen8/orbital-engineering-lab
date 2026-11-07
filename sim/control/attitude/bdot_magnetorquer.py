from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from sim.control.attitude._validation import require_finite_scalar, require_finite_vector
from sim.core.interfaces import Controller
from sim.core.models import Command, StateBelief


def _unit(vec: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    arr = np.array(vec, dtype=float).reshape(3)
    if not np.all(np.isfinite(arr)):
        raise ValueError("vector must contain finite values.")
    mag = float(np.linalg.norm(arr))
    if mag <= eps:
        return np.zeros(3, dtype=float)
    return arr / mag


@dataclass
class MagnetorquerBdotController(Controller):
    magnetic_field_body_t: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 3.0e-5]))
    gain: float = 2.0e3
    max_torque_nm: float = 1.0e-4
    angular_rate_slice: tuple[int, int] = (10, 13)

    def __post_init__(self) -> None:
        self.magnetic_field_body_t = require_finite_vector(
            self.magnetic_field_body_t,
            "magnetic_field_body_t",
            size=3,
            allow_positive_inf=False,
        )
        self.gain = require_finite_scalar(self.gain, "gain")
        self.max_torque_nm = require_finite_scalar(self.max_torque_nm, "max_torque_nm", allow_positive_inf=True)

    def act(self, belief: StateBelief, t_s: float, budget_ms: float) -> Command:
        i0, i1 = self.angular_rate_slice
        if belief.state.size < i1:
            return Command.zero()
        omega = require_finite_vector(belief.state[i0:i1], "belief angular rate", size=3)
        b_body = require_finite_vector(self.magnetic_field_body_t, "magnetic_field_body_t", size=3)
        b_hat = _unit(b_body)
        if float(np.linalg.norm(b_hat)) <= 0.0:
            torque = np.zeros(3, dtype=float)
        else:
            omega_perp = omega - b_hat * float(np.dot(omega, b_hat))
            gain = require_finite_scalar(self.gain, "gain")
            torque = -gain * omega_perp * float(np.linalg.norm(b_body)) ** 2
            n = float(np.linalg.norm(torque))
            max_torque = require_finite_scalar(self.max_torque_nm, "max_torque_nm", allow_positive_inf=True)
            max_torque = float(max(max_torque, 0.0))
            if max_torque <= 0.0:
                torque = np.zeros(3, dtype=float)
            elif n > max_torque:
                torque *= max_torque / n
        return Command(
            thrust_eci_km_s2=np.zeros(3, dtype=float),
            torque_body_nm=torque,
            mode_flags={
                "mode": "magnetorquer_bdot",
                "magnetic_field_body_t": b_body.tolist(),
                "bdot_omega_perp_norm_rad_s": float(np.linalg.norm(omega - b_hat * float(np.dot(omega, b_hat)))),
            },
        )
