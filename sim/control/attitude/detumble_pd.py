from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from sim.config.plugin_specs import instantiate_plugin_spec
from sim.control.attitude._validation import require_finite_matrix, require_finite_vector
from sim.control.attitude.baseline import ReactionWheelPDController
from sim.core.interfaces import Controller
from sim.core.models import Command, StateBelief
from sim.utils.frames import ric_dcm_ir_from_rv
from sim.utils.quaternion import dcm_to_quaternion_bn, quaternion_to_dcm_bn


def _construct_pd(spec: Any) -> ReactionWheelPDController:
    if isinstance(spec, ReactionWheelPDController):
        return spec
    if isinstance(spec, dict):
        obj = instantiate_plugin_spec(spec, description="pd controller")
        if not isinstance(obj, ReactionWheelPDController):
            raise TypeError("pd spec must construct a ReactionWheelPDController.")
        return obj
    raise TypeError("pd must be a ReactionWheelPDController or a compatible constructor dict spec.")


@dataclass
class ECIDetumblePDController(Controller):
    pd: Any
    rate_only: bool = True
    lock_reference_on_first_call: bool = True
    _q_ref_bn: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.pd = _construct_pd(self.pd)

    def reset_reference(self) -> None:
        self._q_ref_bn = None

    def set_reference(self, q_ref_bn: np.ndarray) -> None:
        q = require_finite_vector(q_ref_bn, "q_ref_bn", size=4)
        n = float(np.linalg.norm(q))
        if n <= 0.0:
            raise ValueError("q_ref_bn must be nonzero.")
        self._q_ref_bn = q / n

    def act(self, belief: StateBelief, t_s: float, budget_ms: float) -> Command:
        if belief.state.size < 13:
            return Command.zero()
        q_cur = require_finite_vector(belief.state[6:10], "belief attitude quaternion", size=4)
        n = float(np.linalg.norm(q_cur))
        if n <= 0.0:
            return Command.zero()
        q_cur = q_cur / n

        if self.rate_only:
            # Pure detumble mode: damp body rates without trying to hold a fixed attitude.
            self.pd.set_target(q_cur, np.zeros(3))
        else:
            if self._q_ref_bn is None:
                if self.lock_reference_on_first_call:
                    self._q_ref_bn = q_cur.copy()
                else:
                    self._q_ref_bn = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
            self.pd.set_target(self._q_ref_bn, np.zeros(3))
        cmd = self.pd.act(belief, t_s, budget_ms)
        mode_flags = dict(cmd.mode_flags)
        mode_flags["mode"] = "pd_detumble_eci"
        return Command(thrust_eci_km_s2=cmd.thrust_eci_km_s2, torque_body_nm=cmd.torque_body_nm, mode_flags=mode_flags)


@dataclass
class RICDetumblePDController(Controller):
    pd: Any
    state_rv_slice: tuple[int, int] = (0, 6)
    rate_only: bool = True
    lock_reference_on_first_call: bool = True
    _c_br_ref: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.pd = _construct_pd(self.pd)

    def reset_reference(self) -> None:
        self._c_br_ref = None

    def set_reference_c_br(self, c_br: np.ndarray) -> None:
        c = require_finite_matrix(c_br, "c_br", shape=(3, 3))
        self._c_br_ref = c.copy()

    def act(self, belief: StateBelief, t_s: float, budget_ms: float) -> Command:
        i0, i1 = self.state_rv_slice
        if belief.state.size < max(13, i1):
            return Command.zero()

        r_eci = require_finite_vector(belief.state[i0 : i0 + 3], "belief position", size=3)
        v_eci = require_finite_vector(belief.state[i0 + 3 : i0 + 6], "belief velocity", size=3)
        r_norm = float(np.linalg.norm(r_eci))
        if r_norm <= 0.0:
            return Command.zero()

        q_cur = require_finite_vector(belief.state[6:10], "belief attitude quaternion", size=4)
        q_norm = float(np.linalg.norm(q_cur))
        if q_norm <= 0.0:
            return Command.zero()
        c_bn = quaternion_to_dcm_bn(q_cur / q_norm)
        c_ir = ric_dcm_ir_from_rv(r_eci, v_eci)
        c_ri = c_ir.T

        h = np.cross(r_eci, v_eci)
        omega_ri_eci = h / max(r_norm * r_norm, 1e-12)
        omega_ri_ric = c_ri @ omega_ri_eci
        if self.rate_only:
            c_br_cur = c_bn @ c_ir
            q_des_bn = q_cur / q_norm
            omega_bi_des_body = c_br_cur @ omega_ri_ric
        else:
            if self._c_br_ref is None:
                if self.lock_reference_on_first_call:
                    self._c_br_ref = c_bn @ c_ir
                else:
                    self._c_br_ref = np.eye(3)
            c_bn_des = self._c_br_ref @ c_ri
            q_des_bn = dcm_to_quaternion_bn(c_bn_des)
            omega_bi_des_body = self._c_br_ref @ omega_ri_ric

        self.pd.set_target(q_des_bn, omega_bi_des_body)
        cmd = self.pd.act(belief, t_s, budget_ms)
        mode_flags = dict(cmd.mode_flags)
        mode_flags["mode"] = "pd_detumble_ric"
        return Command(thrust_eci_km_s2=cmd.thrust_eci_km_s2, torque_body_nm=cmd.torque_body_nm, mode_flags=mode_flags)
