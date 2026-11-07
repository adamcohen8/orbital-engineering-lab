from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from sim.config.plugin_specs import instantiate_plugin_spec
from sim.control.attitude._validation import require_finite_vector
from sim.core.interfaces import Controller
from sim.core.models import Command, StateBelief


def _construct_controller(spec: Any) -> Controller:
    if hasattr(spec, "act"):
        return spec
    if isinstance(spec, dict):
        return instantiate_plugin_spec(spec, description="controller")
    raise TypeError("base_controller must be a controller object or constructor dict.")


@dataclass
class CMGSteeringController(Controller):
    base_controller: Any
    max_torque_nm: np.ndarray | float = 0.2
    momentum_nms: np.ndarray | float = 1.0
    gimbal_rate_limit_rad_s: np.ndarray | float = 0.1

    def __post_init__(self) -> None:
        self.base_controller = _construct_controller(self.base_controller)
        require_finite_vector(self._vec(self.max_torque_nm), "max_torque_nm", size=3, allow_positive_inf=True)
        require_finite_vector(self._vec(self.momentum_nms), "momentum_nms", size=3, allow_positive_inf=True)
        # Positive infinity remains a valid unlimited gimbal-rate value.
        require_finite_vector(
            self._vec(self.gimbal_rate_limit_rad_s),
            "gimbal_rate_limit_rad_s",
            size=3,
            allow_positive_inf=True,
        )

    @staticmethod
    def _vec(value: np.ndarray | float, default: float = 0.0) -> np.ndarray:
        arr = np.array(value, dtype=float).reshape(-1)
        if arr.size == 1:
            return np.full(3, float(arr[0]), dtype=float)
        if arr.size != 3:
            return np.full(3, float(default), dtype=float)
        return arr

    def act(self, belief: StateBelief, t_s: float, budget_ms: float) -> Command:
        cmd = self.base_controller.act(belief, t_s, budget_ms)
        torque = require_finite_vector(cmd.torque_body_nm, "base controller torque", size=3)
        require_finite_vector(self._vec(self.max_torque_nm), "max_torque_nm", size=3, allow_positive_inf=True)
        require_finite_vector(self._vec(self.momentum_nms), "momentum_nms", size=3, allow_positive_inf=True)
        require_finite_vector(
            self._vec(self.gimbal_rate_limit_rad_s),
            "gimbal_rate_limit_rad_s",
            size=3,
            allow_positive_inf=True,
        )
        max_torque = np.abs(self._vec(self.max_torque_nm))
        momentum = np.abs(self._vec(self.momentum_nms))
        rate = np.abs(self._vec(self.gimbal_rate_limit_rad_s))
        gimbal_cap = np.zeros(3, dtype=float)
        finite_product = np.isfinite(momentum) & np.isfinite(rate)
        gimbal_cap[finite_product] = momentum[finite_product] * rate[finite_product]
        unlimited_product = (~finite_product) & (momentum > 0.0) & (rate > 0.0)
        gimbal_cap[unlimited_product] = np.inf
        cap = np.minimum(
            max_torque,
            gimbal_cap,
        )
        torque = np.clip(torque, -cap, cap)
        mode_flags = dict(cmd.mode_flags or {})
        mode_flags.update(
            {
                "mode": "cmg_steering",
                "cmg_base_mode": mode_flags.get("mode"),
                "cmg_torque_cap_nm": cap.tolist(),
            }
        )
        return Command(
            thrust_eci_km_s2=np.array(cmd.thrust_eci_km_s2, dtype=float),
            torque_body_nm=torque,
            mode_flags=mode_flags,
        )
