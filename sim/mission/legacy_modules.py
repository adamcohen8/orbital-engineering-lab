# ruff: noqa: F401,F403,F405,I001
from .strategies.base import *

@dataclass
class AttitudeDetumbleGateMissionModule:
    """
    Mission-side attitude mode gate:
    - Use detumble mode when |w_body| exceeds enter threshold.
    - Return to nominal mode when |w_body| falls below exit threshold.
    Requires attitude controller to implement `set_mode("detumble"|"nominal")`.
    """

    enter_rate_rad_s: float = 0.03
    exit_rate_rad_s: float = 0.015
    prefer_attitude_belief_rate: bool = True
    detumble_mode_name: str = "detumble"
    nominal_mode_name: str = "nominal"
    _detumble_latched: bool = False

    def __post_init__(self) -> None:
        self.enter_rate_rad_s = float(max(self.enter_rate_rad_s, 0.0))
        self.exit_rate_rad_s = float(max(self.exit_rate_rad_s, 0.0))
        if self.exit_rate_rad_s > self.enter_rate_rad_s:
            self.exit_rate_rad_s = self.enter_rate_rad_s

    def _rate_norm_rad_s(self, truth: StateTruth, att_belief: StateBelief | None) -> float:
        if self.prefer_attitude_belief_rate and att_belief is not None and att_belief.state.size >= 13:
            w = np.array(att_belief.state[10:13], dtype=float)
            if np.all(np.isfinite(w)):
                return float(np.linalg.norm(w))
        w_t = np.array(truth.angular_rate_body_rad_s, dtype=float)
        return float(np.linalg.norm(w_t))

    def update(
        self,
        *,
        truth: StateTruth,
        attitude_controller: Any | None = None,
        att_belief: StateBelief | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        w_norm = self._rate_norm_rad_s(truth=truth, att_belief=att_belief)
        if self._detumble_latched:
            if w_norm <= self.exit_rate_rad_s:
                self._detumble_latched = False
        else:
            if w_norm >= self.enter_rate_rad_s:
                self._detumble_latched = True

        desired_mode = self.detumble_mode_name if self._detumble_latched else self.nominal_mode_name
        controller_accepts_mode = bool(attitude_controller is not None and hasattr(attitude_controller, "set_mode"))
        if controller_accepts_mode:
            try:
                attitude_controller.set_mode(desired_mode)
            except (TypeError, ValueError, AttributeError) as exc:
                logger.warning("Unable to set attitude controller mode '%s': %s", desired_mode, exc)
                controller_accepts_mode = False

        return {
            "attitude_gate": {
                "mode": str(desired_mode),
                "rate_norm_rad_s": float(w_norm),
                "enter_rate_rad_s": float(self.enter_rate_rad_s),
                "exit_rate_rad_s": float(self.exit_rate_rad_s),
                "controller_accepts_mode": bool(controller_accepts_mode),
            }
        }


@dataclass
class SatelliteMissionModule:
    orbital_mode: str = "coast"  # coast|pursuit_knowledge|evade_knowledge|pursuit_blind|evade_blind
    attitude_mode: str = "hold_eci"  # hold_eci|hold_ric|spotlight|sun_track|pursuit|evade|sensing
    target_id: str | None = None
    max_accel_km_s2: float = 0.0
    blind_direction_eci: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0], dtype=float))
    hold_quat_bn: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0], dtype=float))
    hold_quat_br: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0], dtype=float))
    boresight_body: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0], dtype=float))
    spotlight_lat_deg: float = 0.0
    spotlight_lon_deg: float = 0.0
    spotlight_alt_km: float = 0.0
    spotlight_ric_direction: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0], dtype=float))
    use_knowledge_for_targeting: bool = True

    def _target_state(self, own_knowledge: dict[str, StateBelief]) -> tuple[np.ndarray, np.ndarray] | None:
        if self.target_id is None:
            return None
        if self.use_knowledge_for_targeting and self.target_id in own_knowledge:
            kb = own_knowledge[self.target_id]
            if kb.state.size >= 6:
                return np.array(kb.state[:3], dtype=float), np.array(kb.state[3:6], dtype=float)
        return None

    def _orbital_command(self, truth: StateTruth, own_knowledge: dict[str, StateBelief]) -> np.ndarray:
        amax = float(max(self.max_accel_km_s2, 0.0))
        if self.orbital_mode == "coast" or amax <= 0.0:
            return np.zeros(3, dtype=float)
        if self.orbital_mode in ("pursuit_knowledge", "evade_knowledge", "pursuit_blind", "evade_blind"):
            tgt = self._target_state(own_knowledge=own_knowledge)
            if tgt is None:
                d = _unit(np.array(self.blind_direction_eci, dtype=float))
            else:
                d = _unit(tgt[0] - np.array(truth.position_eci_km, dtype=float))
            if self.orbital_mode.startswith("evade"):
                d = -d
            return amax * d
        return np.zeros(3, dtype=float)

    def _attitude_command(
        self,
        truth: StateTruth,
        own_knowledge: dict[str, StateBelief],
        env: dict[str, Any] | None = None,
        orbital_accel_cmd: np.ndarray | None = None,
    ) -> np.ndarray:
        env = dict(env or {})
        orbital_accel_cmd = np.zeros(3, dtype=float) if orbital_accel_cmd is None else np.array(orbital_accel_cmd)
        mode = str(self.attitude_mode).lower()
        if mode == "hold_eci":
            return normalize_quaternion(np.array(self.hold_quat_bn, dtype=float))
        if mode == "hold_ric":
            c_ir = ric_dcm_ir_from_rv(truth.position_eci_km, truth.velocity_eci_km_s)
            c_br = quaternion_to_dcm_bn(np.array(self.hold_quat_br, dtype=float))
            c_bn = c_br @ c_ir.T
            return dcm_to_quaternion_bn(c_bn)
        if mode == "sun_track":
            sun_dir = np.array(env.get("sun_dir_eci", np.array([1.0, 0.0, 0.0])), dtype=float)
            return PoseCommandGenerator.sun_track(
                truth=truth,
                sun_dir_eci=sun_dir,
                panel_normal_body=np.array(self.boresight_body, dtype=float),
            )
        if mode == "spotlight":
            return PoseCommandGenerator.spotlight_latlon(
                truth=truth,
                latitude_deg=float(self.spotlight_lat_deg),
                longitude_deg=float(self.spotlight_lon_deg),
                altitude_km=float(self.spotlight_alt_km),
                boresight_body=np.array(self.boresight_body, dtype=float),
            )
        if mode == "sensing":
            return PoseCommandGenerator.spotlight_ric_direction(
                truth=truth,
                ric_direction=np.array(self.spotlight_ric_direction, dtype=float),
                boresight_body=np.array(self.boresight_body, dtype=float),
            )
        if mode in ("pursuit", "evade"):
            d = _unit(np.array(orbital_accel_cmd, dtype=float))
            if np.linalg.norm(d) <= 0.0:
                return normalize_quaternion(np.array(truth.attitude_quat_bn, dtype=float))
            if mode == "evade":
                d = -d
            return PoseCommandGenerator.sun_track(
                truth=truth,
                sun_dir_eci=d,
                panel_normal_body=np.array(self.boresight_body, dtype=float),
            )
        tgt = self._target_state(own_knowledge=own_knowledge)
        if tgt is not None:
            d = _unit(tgt[0] - np.array(truth.position_eci_km, dtype=float))
            return PoseCommandGenerator.sun_track(
                truth=truth,
                sun_dir_eci=d,
                panel_normal_body=np.array(self.boresight_body, dtype=float),
            )
        return normalize_quaternion(np.array(truth.attitude_quat_bn, dtype=float))

    def update(
        self,
        *,
        object_id: str,
        truth: StateTruth,
        own_knowledge: dict[str, StateBelief],
        env: dict[str, Any],
        t_s: float,
        dt_s: float,
        **kwargs: Any,
    ) -> dict[str, Any]:
        a_cmd = self._orbital_command(truth=truth, own_knowledge=own_knowledge)
        q_cmd = self._attitude_command(
            truth=truth,
            own_knowledge=own_knowledge,
            env=env,
            orbital_accel_cmd=a_cmd,
        )
        return {
            "thrust_eci_km_s2": np.array(a_cmd, dtype=float),
            "desired_attitude_quat_bn": np.array(q_cmd, dtype=float),
            "mission_mode": {"orbital": self.orbital_mode, "attitude": self.attitude_mode},
        }


@dataclass
class DefensiveRICAxisBurnMissionModule:
    """
    Basic defensive maneuver:
    - Select one fixed burn direction in the RIC frame: +R/-R/+I/-I/+C/-C.
    - Burn only when valid knowledge of the chaser is available.
    """

    chaser_id: str = "chaser"
    axis_mode: str = "+R"  # +R|-R|+I|-I|+C|-C
    burn_accel_km_s2: float = 2e-6
    require_finite_knowledge: bool = True
    thruster_direction_body: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0], dtype=float))
    alignment_tolerance_rad: float = np.deg2rad(5.0)
    alignment_tolerance_deg: float | None = None
    min_burn_accel_km_s2: float = 1e-12

    def __post_init__(self) -> None:
        self.alignment_tolerance_rad = _resolve_angle_tolerance_rad(
            self.alignment_tolerance_rad, self.alignment_tolerance_deg
        )

    def _has_chaser_knowledge(self, own_knowledge: dict[str, StateBelief]) -> bool:
        kb = own_knowledge.get(self.chaser_id)
        if kb is None or kb.state.size < 6:
            return False
        if not self.require_finite_knowledge:
            return True
        x = np.array(kb.state[:6], dtype=float)
        return bool(np.all(np.isfinite(x)))

    def _alignment(self, truth: StateTruth, accel_eci_km_s2: np.ndarray) -> tuple[bool, float]:
        a = np.array(accel_eci_km_s2, dtype=float).reshape(3)
        if float(np.linalg.norm(a)) <= 0.0:
            return True, 0.0
        c_bn = quaternion_to_dcm_bn(truth.attitude_quat_bn)
        t_body = _unit(np.array(self.thruster_direction_body, dtype=float))
        if float(np.linalg.norm(t_body)) <= 0.0:
            return False, float(np.pi)
        thrust_axis_eci = c_bn.T @ t_body
        target_axis_eci = -_unit(a)
        cosang = float(np.clip(np.dot(thrust_axis_eci, target_axis_eci), -1.0, 1.0))
        ang = float(np.arccos(cosang))
        return ang <= float(max(self.alignment_tolerance_rad, 0.0)), ang

    def update(
        self,
        *,
        truth: StateTruth,
        own_knowledge: dict[str, StateBelief],
        attitude_controller: Any | None = None,
        att_belief: StateBelief | None = None,
        t_s: float,
        dt_s: float,
        **kwargs: Any,
    ) -> dict[str, Any]:
        env = dict(kwargs.get("env", {}) or {})
        attitude_disabled = bool(env.get("attitude_disabled", False))
        out: dict[str, Any] = {}
        know = self._has_chaser_knowledge(own_knowledge)
        if not know:
            out["mission_use_integrated_command"] = True
            out["thrust_eci_km_s2"] = np.zeros(3, dtype=float)
            out["torque_body_nm"] = np.zeros(3, dtype=float)
            out["mission_mode"] = {
                "type": "defensive_ric_axis_burn",
                "axis_mode": str(self.axis_mode),
                "triggered": False,
                "has_chaser_knowledge": False,
                "alignment_ok": False,
            }
            return out

        a_mag = float(max(self.burn_accel_km_s2, 0.0))
        if a_mag <= 0.0:
            out["mission_use_integrated_command"] = True
            out["thrust_eci_km_s2"] = np.zeros(3, dtype=float)
            out["torque_body_nm"] = np.zeros(3, dtype=float)
            out["mission_mode"] = {
                "type": "defensive_ric_axis_burn",
                "axis_mode": str(self.axis_mode),
                "triggered": False,
                "has_chaser_knowledge": True,
                "alignment_ok": False,
                "reason": "zero_burn_accel",
            }
            return out

        dir_ric = _axis_unit_ric(self.axis_mode)
        c_ir = ric_dcm_ir_from_rv(
            np.array(truth.position_eci_km, dtype=float), np.array(truth.velocity_eci_km_s, dtype=float)
        )
        dir_eci = c_ir @ dir_ric
        thrust_cmd = a_mag * _unit(dir_eci)
        q_req = OrbitalAttitudeManeuverCoordinator().maneuverer.required_attitude_for_delta_v(
            truth=truth,
            delta_v_eci_km_s=np.array(thrust_cmd, dtype=float),
            thruster_direction_body=np.array(self.thruster_direction_body, dtype=float),
        )
        q_des = np.array(q_req if q_req is not None else truth.attitude_quat_bn, dtype=float)

        if (not attitude_disabled) and attitude_controller is not None and hasattr(attitude_controller, "set_target"):
            try:
                attitude_controller.set_target(q_des)
            except (TypeError, ValueError, AttributeError) as exc:
                logger.warning("Failed to set defensive burn attitude target: %s", exc)

        att_belief_eff = att_belief
        if (not attitude_disabled) and att_belief_eff is None and attitude_controller is not None:
            att_belief_eff = _attitude_belief_from_truth(truth)
        c_att = (
            attitude_controller.act(att_belief_eff, float(t_s), 2.0)
            if (not attitude_disabled) and attitude_controller is not None and att_belief_eff is not None
            else Command.zero()
        )

        align_ok, align_angle = self._alignment(truth=truth, accel_eci_km_s2=np.array(thrust_cmd, dtype=float))
        if attitude_disabled or attitude_controller is None:
            # Permit immediate burns when no attitude loop is present to execute slews.
            align_ok = True
        fire = bool(align_ok and float(np.linalg.norm(thrust_cmd)) > float(max(self.min_burn_accel_km_s2, 0.0)))

        out["mission_use_integrated_command"] = True
        out["torque_body_nm"] = np.array(c_att.torque_body_nm, dtype=float).reshape(3)
        out["command_mode_flags"] = dict(c_att.mode_flags or {})
        out["desired_attitude_quat_bn"] = q_des
        out["thrust_eci_km_s2"] = np.array(thrust_cmd, dtype=float) if fire else np.zeros(3, dtype=float)
        out["mission_mode"] = {
            "type": "defensive_ric_axis_burn",
            "axis_mode": str(self.axis_mode),
            "triggered": True,
            "has_chaser_knowledge": True,
            "alignment_ok": bool(align_ok),
            "alignment_angle_rad": float(align_angle),
            "fire": bool(fire),
        }
        return out


@dataclass
class SingleRICAxisBurnMissionModule:
    """
    One-shot burn in the target-centered RIC frame, then coast.

    `axis_mode` selects the requested RIC axis.
    `axis_kind="plume"` interprets that axis as the nozzle/plume direction,
    so the applied force is opposite that axis.
    """

    target_id: str = "target"
    use_knowledge_for_targeting: bool = True
    axis_mode: str = "+I"
    axis_kind: str = "plume"  # plume|force
    burn_accel_km_s2: float = 2e-6
    burn_start_s: float = 0.0
    burn_duration_s: float = 60.0
    slew_lead_time_s: float = 0.0
    thruster_direction_body: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0], dtype=float))
    require_finite_reference: bool = True

    def _reference_state(
        self,
        *,
        object_id: str | None,
        truth: StateTruth,
        own_knowledge: dict[str, StateBelief],
    ) -> tuple[np.ndarray, np.ndarray] | None:
        target_token = str(self.target_id or "").strip().lower()
        object_token = str(object_id or "").strip().lower()
        if target_token in {"", "self", "ownship"} or (object_token and target_token == object_token):
            return (
                np.array(truth.position_eci_km, dtype=float).reshape(3),
                np.array(truth.velocity_eci_km_s, dtype=float).reshape(3),
            )
        ref = _resolve_target_state(
            target_id=self.target_id,
            use_knowledge_for_targeting=bool(self.use_knowledge_for_targeting),
            own_knowledge=own_knowledge,
        )
        if ref is None:
            return None
        if not self.require_finite_reference:
            return ref
        r_ref = np.array(ref[0], dtype=float).reshape(3)
        v_ref = np.array(ref[1], dtype=float).reshape(3)
        if not (np.all(np.isfinite(r_ref)) and np.all(np.isfinite(v_ref))):
            return None
        return r_ref, v_ref

    def update(
        self,
        *,
        truth: StateTruth,
        own_knowledge: dict[str, StateBelief],
        t_s: float,
        dt_s: float,
        object_id: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        out: dict[str, Any] = {}
        ref = self._reference_state(object_id=object_id, truth=truth, own_knowledge=own_knowledge)
        burn_start_s = float(self.burn_start_s)
        burn_duration_s = float(max(self.burn_duration_s, 0.0))
        slew_lead_time_s = float(max(self.slew_lead_time_s, 0.0))
        burn_active = bool(float(t_s) >= burn_start_s and float(t_s) < (burn_start_s + burn_duration_s))
        slew_active = bool(float(t_s) >= (burn_start_s - slew_lead_time_s) and float(t_s) < burn_start_s)
        if ref is None:
            out["fallback_thrust_eci_km_s2"] = np.zeros(3, dtype=float)
            out["mission_mode"] = {
                "type": "single_ric_axis_burn",
                "phase": "hold_no_reference",
                "axis_mode": str(self.axis_mode),
                "axis_kind": str(self.axis_kind),
                "burn_active": False,
            }
            return out

        if (not burn_active) or float(max(self.burn_accel_km_s2, 0.0)) <= 0.0:
            if not slew_active or float(max(self.burn_accel_km_s2, 0.0)) <= 0.0:
                out["fallback_thrust_eci_km_s2"] = np.zeros(3, dtype=float)
                out["mission_mode"] = {
                    "type": "single_ric_axis_burn",
                    "phase": "coast",
                    "axis_mode": str(self.axis_mode),
                    "axis_kind": str(self.axis_kind),
                    "burn_active": bool(burn_active),
                }
                return out

        axis_ric = _axis_unit_ric(self.axis_mode)
        kind = str(self.axis_kind).strip().lower()
        if kind not in {"plume", "force"}:
            raise ValueError("axis_kind must be 'plume' or 'force'.")
        force_ric = -axis_ric if kind == "plume" else axis_ric
        r_ref, v_ref = ref
        c_ir = ric_dcm_ir_from_rv(r_ref, v_ref)
        force_eci = c_ir @ force_ric
        thrust_cmd = float(max(self.burn_accel_km_s2, 0.0)) * _unit(force_eci)
        required_q = _desired_attitude_for_thrust(
            truth=truth,
            thrust_eci_km_s2=thrust_cmd,
            thruster_direction_body=np.array(self.thruster_direction_body, dtype=float),
        )
        out["desired_attitude_quat_bn"] = np.array(required_q, dtype=float)
        if burn_active:
            out["fallback_thrust_eci_km_s2"] = thrust_cmd
            out["align_to_thrust"] = True
            phase = "burn"
        else:
            out["fallback_thrust_eci_km_s2"] = np.zeros(3, dtype=float)
            phase = "slew"
        out["mission_mode"] = {
            "type": "single_ric_axis_burn",
            "phase": phase,
            "axis_mode": str(self.axis_mode),
            "axis_kind": str(self.axis_kind),
            "burn_active": bool(burn_active),
            "slew_active": bool(slew_active),
            "burn_start_s": burn_start_s,
            "burn_duration_s": burn_duration_s,
            "slew_lead_time_s": slew_lead_time_s,
        }
        return out


@dataclass
class MultiRICAxisBurnMissionModule:
    """
    One finite burn with multiple simultaneous components in the object's
    instantaneous RIC frame.

    Each item in `burns` should provide:
    - `axis_mode`: +R, -R, +I, -I, +C, or -C
    - `target_delta_v_m_s`: non-negative component delta-v in m/s
    """

    burns: tuple[dict[str, Any], ...] | list[dict[str, Any]] = ()
    burn_start_s: float = 0.0
    burn_duration_s: float = 60.0

    def __post_init__(self) -> None:
        self.burn_start_s = float(self.burn_start_s)
        self.burn_duration_s = float(self.burn_duration_s)
        if self.burn_duration_s <= 0.0:
            raise ValueError("burn_duration_s must be positive.")

        parsed: list[dict[str, Any]] = []
        for burn in list(self.burns or []):
            item = dict(burn or {})
            axis_mode = str(item.get("axis_mode", "+R") or "+R").strip().upper()
            if len(axis_mode) == 1 and axis_mode in {"R", "I", "C"}:
                axis_mode = f"+{axis_mode}"
            _axis_unit_ric(axis_mode)
            target_delta_v_m_s = float(item.get("target_delta_v_m_s", 0.0))
            if target_delta_v_m_s < 0.0:
                raise ValueError("target_delta_v_m_s must be non-negative.")
            parsed.append({"axis_mode": axis_mode, "target_delta_v_m_s": target_delta_v_m_s})
        if not parsed:
            raise ValueError("burns must include at least one RIC component.")
        self.burns = tuple(parsed)

    def _delta_v_vector_m_s(self) -> np.ndarray:
        vec = np.zeros(3, dtype=float)
        for burn in self.burns:
            vec += _axis_unit_ric(str(burn["axis_mode"])) * float(burn["target_delta_v_m_s"])
        return vec

    def update(
        self,
        *,
        truth: StateTruth,
        t_s: float,
        dt_s: float | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        burn_active = bool(float(self.burn_start_s) <= float(t_s) < float(self.burn_start_s + self.burn_duration_s))
        accel_ric_km_s2 = np.zeros(3, dtype=float)
        if burn_active:
            accel_ric_km_s2 = (self._delta_v_vector_m_s() / 1000.0) / float(self.burn_duration_s)

        c_ir = ric_dcm_ir_from_rv(
            np.array(truth.position_eci_km, dtype=float), np.array(truth.velocity_eci_km_s, dtype=float)
        )
        accel_eci_km_s2 = c_ir @ accel_ric_km_s2
        total_delta_v_m_s = float(np.linalg.norm(self._delta_v_vector_m_s()))
        return {
            "mission_use_integrated_command": True,
            "thrust_eci_km_s2": accel_eci_km_s2,
            "torque_body_nm": np.zeros(3, dtype=float),
            "command_mode_flags": {
                "mission": "multi_ric_axis_burn",
                "axis_mode": "multi",
                "target_delta_v_m_s": total_delta_v_m_s,
                "burn_active": bool(burn_active),
            },
            "mission_mode": {
                "type": "multi_ric_axis_burn",
                "mode": "multi_ric_axis_burn",
                "burns": [dict(burn) for burn in self.burns],
                "target_delta_v_m_s": total_delta_v_m_s,
                "burn_active": bool(burn_active),
                "burn_start_s": float(self.burn_start_s),
                "burn_duration_s": float(self.burn_duration_s),
            },
        }


@dataclass
class ScheduledVectorBurnMissionModule:
    """Apply one scheduled vector burn through the normal mission fallback path."""

    target_id: str = "self"
    frame: str = "ric"  # ric|eci
    delta_v_m_s: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    burn_start_s: float = 0.0
    burn_duration_s: float = 1.0
    thruster_direction_body: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0], dtype=float))
    require_finite_reference: bool = True

    def _reference_state(
        self,
        *,
        object_id: str | None,
        truth: StateTruth,
        own_knowledge: dict[str, StateBelief],
    ) -> tuple[np.ndarray, np.ndarray] | None:
        target_token = str(self.target_id or "").strip().lower()
        object_token = str(object_id or "").strip().lower()
        if target_token in {"", "self", "ownship"} or (object_token and target_token == object_token):
            return (
                np.array(truth.position_eci_km, dtype=float).reshape(3),
                np.array(truth.velocity_eci_km_s, dtype=float).reshape(3),
            )
        ref = _resolve_target_state(
            target_id=self.target_id,
            use_knowledge_for_targeting=True,
            own_knowledge=own_knowledge,
        )
        if ref is None:
            return None
        r_ref = np.array(ref[0], dtype=float).reshape(3)
        v_ref = np.array(ref[1], dtype=float).reshape(3)
        if self.require_finite_reference and not (np.all(np.isfinite(r_ref)) and np.all(np.isfinite(v_ref))):
            return None
        return r_ref, v_ref

    def update(
        self,
        *,
        truth: StateTruth,
        own_knowledge: dict[str, StateBelief],
        t_s: float,
        dt_s: float,
        object_id: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        start = float(self.burn_start_s)
        duration = float(max(self.burn_duration_s, dt_s, 1.0e-12))
        active = bool(float(t_s) >= start and float(t_s) < start + duration)
        frame = str(self.frame or "ric").strip().lower()
        if frame not in {"ric", "eci"}:
            raise ValueError("ScheduledVectorBurnMissionModule frame must be 'ric' or 'eci'.")
        delta_v_m_s = np.array(self.delta_v_m_s, dtype=float).reshape(3)
        if not active or float(np.linalg.norm(delta_v_m_s)) <= 0.0:
            return {
                "fallback_thrust_eci_km_s2": np.zeros(3, dtype=float),
                "mission_mode": {
                    "type": "scheduled_vector_burn",
                    "phase": "coast",
                    "frame": frame,
                    "burn_active": False,
                    "burn_start_s": start,
                    "burn_duration_s": duration,
                },
            }
        if frame == "ric":
            ref = self._reference_state(object_id=object_id, truth=truth, own_knowledge=own_knowledge)
            if ref is None:
                return {
                    "fallback_thrust_eci_km_s2": np.zeros(3, dtype=float),
                    "mission_mode": {
                        "type": "scheduled_vector_burn",
                        "phase": "hold_no_reference",
                        "frame": frame,
                        "burn_active": False,
                        "burn_start_s": start,
                        "burn_duration_s": duration,
                    },
                }
            r_ref, v_ref = ref
            delta_v_eci_m_s = ric_dcm_ir_from_rv(r_ref, v_ref) @ delta_v_m_s
        else:
            delta_v_eci_m_s = delta_v_m_s
        thrust_cmd = (delta_v_eci_m_s / 1000.0) / duration
        required_q = _desired_attitude_for_thrust(
            truth=truth,
            thrust_eci_km_s2=thrust_cmd,
            thruster_direction_body=np.array(self.thruster_direction_body, dtype=float),
        )
        return {
            "desired_attitude_quat_bn": np.array(required_q, dtype=float),
            "fallback_thrust_eci_km_s2": np.array(thrust_cmd, dtype=float),
            "align_to_thrust": True,
            "mission_mode": {
                "type": "scheduled_vector_burn",
                "phase": "burn",
                "frame": frame,
                "burn_active": True,
                "burn_start_s": start,
                "burn_duration_s": duration,
                "delta_v_m_s": delta_v_m_s.tolist(),
                "delta_v_norm_m_s": float(np.linalg.norm(delta_v_m_s)),
            },
        }


@dataclass
class RocketMissionModule:
    launch_mode: str = "go_now"  # go_now|go_when_possible|wait_optimal_window
    orbital_goal: str = "pursuit"  # pursuit|predefined_orbit
    target_id: str | None = None
    go_when_possible_margin_m_s: float = 100.0
    window_period_s: float = 5400.0
    window_open_duration_s: float = 300.0
    predef_target_alt_km: float = 500.0
    predef_target_ecc: float = 0.0

    def _in_window(self, t_s: float) -> bool:
        p = max(float(self.window_period_s), 1.0)
        w = max(float(self.window_open_duration_s), 0.0)
        tau = float(t_s % p)
        return tau <= w

    def update(
        self,
        *,
        object_id: str,
        truth: StateTruth,
        t_s: float,
        rocket_state: RocketState | None = None,
        rocket_vehicle_cfg: RocketVehicleConfig | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        launch_authorized = True
        own_knowledge = dict(kwargs.get("own_knowledge", {}) or {})
        if self.launch_mode == "go_now":
            launch_authorized = True
        elif self.launch_mode == "wait_optimal_window":
            launch_authorized = self._in_window(float(t_s))
        elif self.launch_mode == "go_when_possible":
            if rocket_state is None or rocket_vehicle_cfg is None:
                launch_authorized = True
            else:
                target_state = _resolve_target_state(
                    target_id=(None if self.target_id is None else str(self.target_id)),
                    use_knowledge_for_targeting=True,
                    own_knowledge=own_knowledge,
                )
                dv_avail = _estimate_stack_delta_v_m_s(rocket_state=rocket_state, vehicle_cfg=rocket_vehicle_cfg)
                dv_need = (
                    np.inf
                    if target_state is None
                    else float(
                        np.linalg.norm(
                            np.array(target_state[1], dtype=float) - np.array(truth.velocity_eci_km_s, dtype=float)
                        )
                        * 1e3
                    )
                )
                launch_authorized = dv_need <= (dv_avail - float(self.go_when_possible_margin_m_s))
        out: dict[str, Any] = {"launch_authorized": bool(launch_authorized)}
        out["mission_mode"] = {"launch": self.launch_mode, "goal": self.orbital_goal}
        return out


@dataclass
class EndStateManeuverMissionModule:
    """
    Mission-level orbital/attitude coupling module.

    Flow:
    1) Build desired end state from explicit target or object knowledge.
    2) Compute required delta-v (current v -> desired v).
    3) Ask integrated maneuver coordinator for fire/slew/hold decision.
    4) Emit attitude target for alignment; emit thrust only when burn is allowed.
    """

    target_id: str | None = None
    use_knowledge_for_targeting: bool = True
    desired_position_eci_km: np.ndarray | None = None
    desired_velocity_eci_km_s: np.ndarray | None = None
    desired_state_source: str = "target"  # target|explicit
    strategy: ManeuverStrategy = "thrust_limited"
    max_thrust_n: float = 0.2
    min_thrust_n: float = 0.0
    burn_dt_s: float = 1.0
    available_delta_v_km_s: float = 0.5
    require_attitude_alignment: bool = True
    thruster_position_body_m: np.ndarray | None = None
    thruster_direction_body: np.ndarray | None = field(default_factory=lambda: np.array([1.0, 0.0, 0.0], dtype=float))
    alignment_tolerance_rad: float = np.deg2rad(5.0)
    alignment_tolerance_deg: float | None = None
    terminate_on_velocity_tolerance_km_s: float = 1e-5
    _coordinator: OrbitalAttitudeManeuverCoordinator = field(
        default_factory=OrbitalAttitudeManeuverCoordinator, init=False, repr=False
    )

    def __post_init__(self) -> None:
        self.alignment_tolerance_rad = _resolve_angle_tolerance_rad(
            self.alignment_tolerance_rad, self.alignment_tolerance_deg
        )

    def _resolve_desired_state(
        self,
        *,
        own_knowledge: dict[str, StateBelief],
    ) -> tuple[np.ndarray, np.ndarray] | None:
        src = str(self.desired_state_source).lower()
        if src == "explicit":
            if self.desired_position_eci_km is None or self.desired_velocity_eci_km_s is None:
                return None
            return (
                np.array(self.desired_position_eci_km, dtype=float).reshape(3),
                np.array(self.desired_velocity_eci_km_s, dtype=float).reshape(3),
            )
        if self.target_id is None:
            return None
        if self.use_knowledge_for_targeting:
            kb = own_knowledge.get(self.target_id)
            if kb is not None and kb.state.size >= 6:
                return np.array(kb.state[:3], dtype=float), np.array(kb.state[3:6], dtype=float)
        return None

    def update(
        self,
        *,
        object_id: str,
        truth: StateTruth,
        own_knowledge: dict[str, StateBelief],
        t_s: float,
        dt_s: float,
        dry_mass_kg: float | None = None,
        orbital_isp_s: float | None = None,
        orbit_command_period_s: float | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        env = dict(kwargs.get("env", {}) or {})
        attitude_disabled = bool(env.get("attitude_disabled", False))
        out: dict[str, Any] = {}
        desired = self._resolve_desired_state(own_knowledge=own_knowledge)
        if desired is None:
            out["mission_mode"] = {"type": "end_state", "phase": "hold_no_target"}
            return out

        _, v_des = desired
        dv_eci = np.array(v_des, dtype=float) - np.array(truth.velocity_eci_km_s, dtype=float)
        dv_mag = float(np.linalg.norm(dv_eci))
        if dv_mag <= max(float(self.terminate_on_velocity_tolerance_km_s), 0.0):
            out["mission_mode"] = {"type": "end_state", "phase": "on_target"}
            return out

        burn_window_s = float(max(orbit_command_period_s if orbit_command_period_s is not None else dt_s, 1e-6))
        available_delta_v_km_s = _available_delta_v_from_truth_mass_km_s(
            truth=truth,
            dry_mass_kg=dry_mass_kg,
            orbital_isp_s=orbital_isp_s,
            fallback_km_s=self.available_delta_v_km_s,
        )

        cmd = IntegratedManeuverCommand(
            delta_v_eci_km_s=dv_eci,
            available_delta_v_km_s=available_delta_v_km_s,
            strategy=str(self.strategy),  # type: ignore[arg-type]
            max_thrust_n=float(max(self.max_thrust_n, 0.0)),
            dt_s=burn_window_s,
            min_thrust_n=float(max(self.min_thrust_n, 0.0)),
            require_attitude_alignment=(bool(self.require_attitude_alignment) and (not attitude_disabled)),
            thruster_position_body_m=None
            if self.thruster_position_body_m is None
            else np.array(self.thruster_position_body_m, dtype=float),
            thruster_direction_body=None
            if self.thruster_direction_body is None
            else np.array(self.thruster_direction_body, dtype=float),
            alignment_tolerance_rad=float(max(self.alignment_tolerance_rad, 0.0)),
        )
        _, decision = self._coordinator.execute(truth=truth, command=cmd)
        if dry_mass_kg is None or orbital_isp_s is None:
            self.available_delta_v_km_s = float(max(decision.remaining_delta_v_km_s, 0.0))

        if decision.required_attitude_quat_bn is not None:
            out["desired_attitude_quat_bn"] = np.array(decision.required_attitude_quat_bn, dtype=float)

        if decision.executed and decision.applied_delta_v_km_s > 0.0:
            d = _unit(dv_eci)
            a_cmd = d * (float(decision.applied_delta_v_km_s) / burn_window_s)
            out["thrust_eci_km_s2"] = a_cmd

        out["mission_mode"] = {
            "type": "end_state",
            "phase": decision.action,
            "reason": decision.reason,
            "alignment_ok": bool(decision.alignment_ok),
            "remaining_delta_v_km_s": float(max(decision.remaining_delta_v_km_s, 0.0)),
            "applied_delta_v_km_s": float(decision.applied_delta_v_km_s),
        }
        return out


@dataclass
class IntegratedCommandMissionModule:
    """
    Base mission brain for integrated orbital+attitude command arbitration.

    Workflow each step:
    1) Determine desired end state from knowledge/world/explicit input.
    2) Update orbital controller target if supported.
    3) Ask orbital controller for burn command.
    4) Check alignment for that burn.
    5) If aligned -> burn (and optional attitude hold command).
       If not aligned -> zero burn and use attitude controller to slew.
    6) Output final actuator command for this timestep.
    """

    target_id: str | None = None
    desired_state_source: str = "target"  # target|explicit
    use_knowledge_for_targeting: bool = True
    desired_position_eci_km: np.ndarray | None = None
    desired_velocity_eci_km_s: np.ndarray | None = None
    require_attitude_alignment: bool = True
    thruster_direction_body: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0], dtype=float))
    alignment_tolerance_rad: float = np.deg2rad(5.0)
    alignment_tolerance_deg: float | None = None
    min_burn_accel_km_s2: float = 1e-12

    def __post_init__(self) -> None:
        self.alignment_tolerance_rad = _resolve_angle_tolerance_rad(
            self.alignment_tolerance_rad, self.alignment_tolerance_deg
        )

    def _resolve_desired_state(
        self,
        *,
        own_knowledge: dict[str, StateBelief],
    ) -> tuple[np.ndarray, np.ndarray] | None:
        src = str(self.desired_state_source).lower()
        if src == "explicit":
            if self.desired_position_eci_km is None or self.desired_velocity_eci_km_s is None:
                return None
            return (
                np.array(self.desired_position_eci_km, dtype=float).reshape(3),
                np.array(self.desired_velocity_eci_km_s, dtype=float).reshape(3),
            )
        if self.target_id is None:
            return None
        if self.use_knowledge_for_targeting:
            kb = own_knowledge.get(self.target_id)
            if kb is not None and kb.state.size >= 6:
                return np.array(kb.state[:3], dtype=float), np.array(kb.state[3:6], dtype=float)
        return None

    @staticmethod
    def _set_orbit_controller_target(controller: Any, desired_state_eci_6: np.ndarray) -> None:
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

    @staticmethod
    def _burn_alignment(
        *,
        truth: StateTruth,
        thrust_eci_km_s2: np.ndarray,
        thruster_direction_body: np.ndarray,
        alignment_tolerance_rad: float,
    ) -> tuple[bool, float, np.ndarray]:
        a = np.array(thrust_eci_km_s2, dtype=float).reshape(3)
        an = float(np.linalg.norm(a))
        if an <= 0.0:
            return True, 0.0, np.array(truth.attitude_quat_bn, dtype=float)
        t_body = _unit(np.array(thruster_direction_body, dtype=float))
        if np.linalg.norm(t_body) <= 0.0:
            return False, float(np.pi), np.array(truth.attitude_quat_bn, dtype=float)
        c_bn = quaternion_to_dcm_bn(truth.attitude_quat_bn)
        thrust_axis_eci = c_bn.T @ t_body
        target_axis_eci = -a / an
        cosang = float(np.clip(np.dot(thrust_axis_eci, target_axis_eci), -1.0, 1.0))
        angle = float(np.arccos(cosang))
        required_q = OrbitalAttitudeManeuverCoordinator().maneuverer.required_attitude_for_delta_v(
            truth=truth,
            delta_v_eci_km_s=a,  # direction-only usage
            thruster_direction_body=t_body,
        )
        return angle <= float(max(alignment_tolerance_rad, 0.0)), angle, required_q

    def update(
        self,
        *,
        object_id: str,
        truth: StateTruth,
        own_knowledge: dict[str, StateBelief],
        orbit_controller: Any | None = None,
        attitude_controller: Any | None = None,
        orb_belief: StateBelief | None = None,
        att_belief: StateBelief | None = None,
        t_s: float,
        dt_s: float,
        **kwargs: Any,
    ) -> dict[str, Any]:
        env = dict(kwargs.get("env", {}) or {})
        attitude_disabled = bool(env.get("attitude_disabled", False))
        out: dict[str, Any] = {}
        desired = self._resolve_desired_state(own_knowledge=own_knowledge)
        if desired is not None:
            x_des = np.hstack((desired[0], desired[1]))
            self._set_orbit_controller_target(orbit_controller, x_des)

        c_orb = Command.zero()
        if orbit_controller is not None and orb_belief is not None:
            c_orb = orbit_controller.act(orb_belief, float(t_s), 2.0)
        thrust_cmd = np.array(c_orb.thrust_eci_km_s2, dtype=float).reshape(3)
        burn_mag = float(np.linalg.norm(thrust_cmd))
        burn_requested = burn_mag > float(max(self.min_burn_accel_km_s2, 0.0))

        align_ok = True
        align_angle = 0.0
        required_q = np.array(truth.attitude_quat_bn, dtype=float)
        if burn_requested and self.require_attitude_alignment and (not attitude_disabled):
            align_ok, align_angle, required_q = self._burn_alignment(
                truth=truth,
                thrust_eci_km_s2=thrust_cmd,
                thruster_direction_body=np.array(self.thruster_direction_body, dtype=float),
                alignment_tolerance_rad=float(self.alignment_tolerance_rad),
            )

        if (not attitude_disabled) and attitude_controller is not None and hasattr(attitude_controller, "set_target"):
            try:
                attitude_controller.set_target(np.array(required_q, dtype=float))
            except (TypeError, ValueError, AttributeError) as exc:
                logger.warning("Failed to set attitude target in IntegratedCommandMissionModule: %s", exc)
        c_att = Command.zero()
        if (not attitude_disabled) and attitude_controller is not None and att_belief is not None:
            c_att = attitude_controller.act(att_belief, float(t_s), 2.0)

        if burn_requested and align_ok:
            out["thrust_eci_km_s2"] = thrust_cmd
            phase = "burn"
        else:
            out["thrust_eci_km_s2"] = np.zeros(3, dtype=float)
            phase = "slew" if burn_requested else "hold"
        out["torque_body_nm"] = np.array(c_att.torque_body_nm, dtype=float).reshape(3)
        out["desired_attitude_quat_bn"] = np.array(required_q, dtype=float)
        out["mission_use_integrated_command"] = True
        out["mission_mode"] = {
            "type": "integrated_brain",
            "phase": phase,
            "burn_requested": bool(burn_requested),
            "alignment_ok": bool(align_ok),
            "alignment_angle_rad": float(align_angle),
        }
        return out


@dataclass
class PredictiveIntegratedCommandMissionModule:
    """
    Predictive integrated mission brain.

    - Predicts forward by lead time.
    - Uses orbital controller at future state to determine thrust direction.
    - Commands attitude controller toward required burn attitude.
    - Fires exactly when burn time arrives if angular tolerance is met.
    """

    target_id: str = "target"
    use_knowledge_for_targeting: bool = True
    lead_time_s: float = 30.0
    predict_dt_s: float = 1.0
    alignment_tolerance_rad: float = np.deg2rad(5.0)
    alignment_tolerance_deg: float | None = None
    thruster_direction_body: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0], dtype=float))
    min_burn_accel_km_s2: float = 1e-12
    mu_km3_s2: float = 398600.4418
    orbit_controller_budget_ms: float = 2.0
    attitude_controller_budget_ms: float = 2.0
    planning_period_s: float | None = None
    skip_orbit_planning_in_detumble_mode: bool = True
    attitude_mode_attr: str = "mode"
    detumble_mode_tokens: tuple[str, ...] = ("detumble",)
    _countdown_s: float = field(default=-1.0, init=False, repr=False)
    _last_plan_t_s: float | None = field(default=None, init=False, repr=False)
    _planned_accel_eci_km_s2: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=float), init=False, repr=False
    )
    _planned_attitude_quat_bn: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0], dtype=float), init=False, repr=False
    )

    def __post_init__(self) -> None:
        self.alignment_tolerance_rad = _resolve_angle_tolerance_rad(
            self.alignment_tolerance_rad, self.alignment_tolerance_deg
        )

    def _target_state(self, own_knowledge: dict[str, StateBelief]) -> tuple[np.ndarray, np.ndarray] | None:
        if self.use_knowledge_for_targeting:
            kb = own_knowledge.get(self.target_id)
            if kb is not None and kb.state.size >= 6:
                r_k = np.array(kb.state[:3], dtype=float)
                v_k = np.array(kb.state[3:6], dtype=float)
                if np.all(np.isfinite(r_k)) and np.all(np.isfinite(v_k)):
                    return r_k, v_k
        return None

    def _predict_eci(self, x_eci: np.ndarray, horizon_s: float, dt_s: float) -> np.ndarray:
        x = np.array(x_eci, dtype=float).reshape(6)
        n_steps = int(max(np.floor(horizon_s / dt_s), 0))
        rem = float(max(horizon_s - n_steps * dt_s, 0.0))
        for _ in range(n_steps):
            x = propagate_two_body_rk4(x_eci=x, dt_s=dt_s, mu_km3_s2=self.mu_km3_s2, accel_cmd_eci_km_s2=np.zeros(3))
        if rem > 1e-9:
            x = propagate_two_body_rk4(x_eci=x, dt_s=rem, mu_km3_s2=self.mu_km3_s2, accel_cmd_eci_km_s2=np.zeros(3))
        return x

    def _predict_orb_belief_for_controller(
        self,
        orbit_controller: Any | None,
        self_truth: StateTruth,
        target_state_eci: tuple[np.ndarray, np.ndarray] | None,
    ) -> StateBelief:
        x_self = np.hstack(
            (np.array(self_truth.position_eci_km, dtype=float), np.array(self_truth.velocity_eci_km_s, dtype=float))
        )
        horizon = float(max(self.lead_time_s, 0.0))
        hdt = float(max(min(self.predict_dt_s, max(horizon, 1e-6)), 1e-6))
        x_self_p = self._predict_eci(x_self, horizon_s=horizon, dt_s=hdt)
        if target_state_eci is None:
            return StateBelief(state=x_self_p, covariance=np.eye(6) * 1e-4, last_update_t_s=float(self_truth.t_s))
        x_tgt = np.hstack((target_state_eci[0], target_state_eci[1]))
        x_tgt_p = self._predict_eci(x_tgt, horizon_s=horizon, dt_s=hdt)

        if orbit_controller is not None and hasattr(orbit_controller, "ric_curv_state_slice"):
            r_c = x_tgt_p[:3]
            v_c = x_tgt_p[3:6]
            r_s = x_self_p[:3]
            v_s = x_self_p[3:6]
            x_rect = eci_relative_to_ric_rect(
                x_dep_eci=np.hstack((r_s, v_s)),
                x_chief_eci=np.hstack((r_c, v_c)),
            )
            x_curv = ric_rect_to_curv(x_rect, r0_km=float(np.linalg.norm(r_c)))
            x = np.hstack((x_curv, np.hstack((r_c, v_c))))
            return StateBelief(state=x, covariance=np.eye(12) * 1e-4, last_update_t_s=float(self_truth.t_s))
        return StateBelief(state=x_self_p, covariance=np.eye(6) * 1e-4, last_update_t_s=float(self_truth.t_s))

    def _alignment(self, truth: StateTruth, accel_eci_km_s2: np.ndarray) -> tuple[bool, float]:
        a = np.array(accel_eci_km_s2, dtype=float).reshape(3)
        if float(np.linalg.norm(a)) <= 0.0:
            return True, 0.0
        c_bn = quaternion_to_dcm_bn(truth.attitude_quat_bn)
        t_body = _unit(np.array(self.thruster_direction_body, dtype=float))
        if float(np.linalg.norm(t_body)) <= 0.0:
            return False, float(np.pi)
        thrust_axis_eci = c_bn.T @ t_body
        target_axis_eci = -_unit(a)
        cosang = float(np.clip(np.dot(thrust_axis_eci, target_axis_eci), -1.0, 1.0))
        ang = float(np.arccos(cosang))
        return ang <= float(max(self.alignment_tolerance_rad, 0.0)), ang

    def _attitude_controller_in_detumble_mode(self, attitude_controller: Any | None) -> tuple[bool, str]:
        if attitude_controller is None:
            return False, ""
        attr = str(self.attitude_mode_attr).strip()
        if not attr:
            return False, ""
        try:
            mode_obj = getattr(attitude_controller, attr, "")
        except AttributeError:
            mode_obj = ""
        mode_str = str(mode_obj).strip().lower()
        if not mode_str:
            return False, ""
        tokens = [str(t).strip().lower() for t in tuple(self.detumble_mode_tokens or ()) if str(t).strip()]
        if any(tok in mode_str for tok in tokens):
            return True, mode_str
        return False, mode_str

    def _effective_planning_period_s(self, orbit_controller: Any | None, dt_s: float) -> float:
        if self.planning_period_s is not None:
            return float(max(self.planning_period_s, 1e-9))
        # Default to mission update cadence; explicit planning_period_s can be used
        # when slower planning is desired.
        return float(max(dt_s, 1e-9))

    def update(
        self,
        *,
        truth: StateTruth,
        own_knowledge: dict[str, StateBelief],
        orbit_controller: Any | None = None,
        attitude_controller: Any | None = None,
        orb_belief: StateBelief | None = None,
        att_belief: StateBelief | None = None,
        t_s: float,
        dt_s: float,
        **kwargs: Any,
    ) -> dict[str, Any]:
        env = dict(kwargs.get("env", {}) or {})
        attitude_disabled = bool(env.get("attitude_disabled", False))
        out: dict[str, Any] = {}
        in_detumble_mode, mode_str = self._attitude_controller_in_detumble_mode(attitude_controller)
        planning_blocked_by_detumble = bool(self.skip_orbit_planning_in_detumble_mode and in_detumble_mode)
        plan_period_s = self._effective_planning_period_s(orbit_controller=orbit_controller, dt_s=float(dt_s))
        target_state = self._target_state(own_knowledge=own_knowledge)
        plan_due = bool(
            self._last_plan_t_s is None or (float(t_s) - float(self._last_plan_t_s)) >= (plan_period_s - 1e-12)
        )
        planned_this_step = False

        if planning_blocked_by_detumble:
            # Cancel pending planned burn while detumbling.
            self._countdown_s = -1.0
            self._planned_accel_eci_km_s2 = np.zeros(3, dtype=float)
            self._planned_attitude_quat_bn = np.array(truth.attitude_quat_bn, dtype=float)
        elif self._countdown_s < 0.0 and plan_due:
            b_pred = self._predict_orb_belief_for_controller(
                orbit_controller=orbit_controller,
                self_truth=truth,
                target_state_eci=target_state,
            )
            c_orb_pred = (
                orbit_controller.act(b_pred, float(t_s), float(max(self.orbit_controller_budget_ms, 1e-9)))
                if orbit_controller is not None
                else Command.zero()
            )
            self._planned_accel_eci_km_s2 = np.array(c_orb_pred.thrust_eci_km_s2, dtype=float).reshape(3)
            if not np.all(np.isfinite(self._planned_accel_eci_km_s2)):
                self._planned_accel_eci_km_s2 = np.zeros(3, dtype=float)
            dv_pred = self._planned_accel_eci_km_s2 * float(max(self.predict_dt_s, 1e-6))
            q_req = OrbitalAttitudeManeuverCoordinator().maneuverer.required_attitude_for_delta_v(
                truth=truth,
                delta_v_eci_km_s=dv_pred,
                thruster_direction_body=np.array(self.thruster_direction_body, dtype=float),
            )
            self._planned_attitude_quat_bn = np.array(
                q_req if q_req is not None else truth.attitude_quat_bn, dtype=float
            )
            self._countdown_s = float(max(self.lead_time_s, 0.0))
            self._last_plan_t_s = float(t_s)
            planned_this_step = True

        # Slew/hold phase before gate
        if (not attitude_disabled) and attitude_controller is not None and hasattr(attitude_controller, "set_target"):
            try:
                attitude_controller.set_target(np.array(self._planned_attitude_quat_bn, dtype=float))
            except (TypeError, ValueError, AttributeError) as exc:
                logger.warning("Failed to set predictive attitude target: %s", exc)
        att_belief_eff = att_belief
        if (not attitude_disabled) and att_belief_eff is None and attitude_controller is not None:
            # Ensure integrated attitude logic can still run even if self-knowledge is not configured.
            att_belief_eff = _attitude_belief_from_truth(truth)
        c_att = (
            attitude_controller.act(att_belief_eff, float(t_s), float(max(self.attitude_controller_budget_ms, 1e-9)))
            if (not attitude_disabled) and attitude_controller is not None and att_belief_eff is not None
            else Command.zero()
        )

        fire = False
        align_ok, align_angle = self._alignment(truth=truth, accel_eci_km_s2=self._planned_accel_eci_km_s2)
        if attitude_disabled:
            align_ok = True
            align_angle = 0.0
        if planning_blocked_by_detumble:
            fire = False
        else:
            if float(max(self.lead_time_s, 0.0)) <= 0.0:
                # Zero-lead mode: continuous closed-loop burn eligibility each step.
                fire = bool(
                    align_ok
                    and float(np.linalg.norm(self._planned_accel_eci_km_s2))
                    > float(max(self.min_burn_accel_km_s2, 0.0))
                )
                self._countdown_s = 0.0
            elif self._countdown_s < 0.0:
                fire = False
            elif self._countdown_s <= float(max(dt_s, 1e-9)):
                if align_ok and float(np.linalg.norm(self._planned_accel_eci_km_s2)) > float(
                    max(self.min_burn_accel_km_s2, 0.0)
                ):
                    fire = True
                self._countdown_s = -1.0
            else:
                self._countdown_s -= float(max(dt_s, 1e-9))

        out["mission_use_integrated_command"] = True
        out["torque_body_nm"] = np.array(c_att.torque_body_nm, dtype=float).reshape(3)
        out["command_mode_flags"] = dict(c_att.mode_flags or {})
        out["desired_attitude_quat_bn"] = np.array(self._planned_attitude_quat_bn, dtype=float)
        out["thrust_eci_km_s2"] = self._planned_accel_eci_km_s2.copy() if fire else np.zeros(3, dtype=float)
        out["mission_mode"] = {
            "type": "predictive_integrated_brain",
            "countdown_s": float(self._countdown_s),
            "fire": bool(fire),
            "alignment_ok": bool(align_ok),
            "alignment_angle_rad": float(align_angle),
            "orbit_controller_budget_ms": float(self.orbit_controller_budget_ms),
            "attitude_controller_budget_ms": float(self.attitude_controller_budget_ms),
            "planning_blocked_by_detumble": bool(planning_blocked_by_detumble),
            "attitude_controller_mode": str(mode_str),
            "plan_due": bool(plan_due),
            "planned_this_step": bool(planned_this_step),
            "planning_period_s": float(plan_period_s),
        }
        return out
