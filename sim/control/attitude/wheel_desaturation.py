from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from sim.control.attitude._validation import require_finite_scalar, require_finite_vector
from sim.core.interfaces import Controller
from sim.core.models import Command, StateBelief


@dataclass
class WheelDesaturationController(Controller):
    wheel_momentum_body_nms: np.ndarray = field(default_factory=lambda: np.zeros(3))
    momentum_state_slice: tuple[int, int] | None = None
    momentum_threshold_nms: float = 0.1
    unload_gain_s_inv: float = 0.02
    max_unload_torque_nm: float = 0.01

    def __post_init__(self) -> None:
        self.wheel_momentum_body_nms = require_finite_vector(
            self.wheel_momentum_body_nms,
            "wheel_momentum_body_nms",
            size=3,
        )
        require_finite_scalar(self.momentum_threshold_nms, "momentum_threshold_nms", allow_positive_inf=True)
        require_finite_scalar(self.unload_gain_s_inv, "unload_gain_s_inv")
        require_finite_scalar(self.max_unload_torque_nm, "max_unload_torque_nm", allow_positive_inf=True)

    def set_wheel_momentum(self, wheel_momentum_body_nms: np.ndarray) -> None:
        self.wheel_momentum_body_nms = require_finite_vector(
            wheel_momentum_body_nms,
            "wheel_momentum_body_nms",
            size=3,
        )

    def _momentum(self, belief: StateBelief) -> np.ndarray:
        if self.momentum_state_slice is not None:
            i0, i1 = self.momentum_state_slice
            if belief.state.size >= i1 and i1 - i0 == 3:
                return require_finite_vector(belief.state[i0:i1], "belief wheel momentum", size=3)
        return require_finite_vector(self.wheel_momentum_body_nms, "wheel_momentum_body_nms", size=3)

    def act(self, belief: StateBelief, t_s: float, budget_ms: float) -> Command:
        h_body = self._momentum(belief)
        h_norm = float(np.linalg.norm(h_body))
        threshold = require_finite_scalar(
            self.momentum_threshold_nms,
            "momentum_threshold_nms",
            allow_positive_inf=True,
        )
        threshold = float(max(threshold, 0.0))
        active = h_norm > threshold > 0.0
        torque = np.zeros(3, dtype=float)
        if active:
            gain = require_finite_scalar(self.unload_gain_s_inv, "unload_gain_s_inv")
            torque = -float(max(gain, 0.0)) * h_body
            n = float(np.linalg.norm(torque))
            max_torque = require_finite_scalar(
                self.max_unload_torque_nm,
                "max_unload_torque_nm",
                allow_positive_inf=True,
            )
            max_torque = float(max(max_torque, 0.0))
            if max_torque <= 0.0:
                torque = np.zeros(3, dtype=float)
            if n > max_torque > 0.0:
                torque *= max_torque / n
        return Command(
            thrust_eci_km_s2=np.zeros(3, dtype=float),
            torque_body_nm=torque,
            mode_flags={
                "mode": "wheel_desaturation",
                "wheel_desaturation_requested": bool(active),
                "wheel_momentum_body_nms": h_body.tolist(),
                "wheel_momentum_norm_nms": h_norm,
            },
        )
