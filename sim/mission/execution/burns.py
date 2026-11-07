# ruff: noqa: F401,F403,F405,I001
from ..strategies.base import *

MAX_PREDICTIVE_BURN_STEPS = 100_000

@dataclass
class PredictiveBurnExecution:
    target_id: str | None = None
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
    detumble_enter_rate_rad_s: float | None = None
    detumble_exit_rate_rad_s: float | None = None
    detumble_mode_name: str = "detumble"
    nominal_mode_name: str = "nominal"
    _detumble_latched: bool = field(default=False, init=False, repr=False)
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
        self._validate_prediction_horizon(self.lead_time_s, self.predict_dt_s)

    @staticmethod
    def _validate_prediction_horizon(horizon_s: float, dt_s: float) -> tuple[float, float]:
        horizon = float(horizon_s)
        step = float(dt_s)
        if not np.isfinite(horizon) or horizon < 0.0:
            raise ValueError("lead_time_s must be a nonnegative finite number.")
        if not np.isfinite(step) or step <= 0.0:
            raise ValueError("predict_dt_s must be a positive finite number.")
        if int(np.floor(horizon / step)) > MAX_PREDICTIVE_BURN_STEPS:
            raise ValueError(
                f"predictive burn horizon exceeds the {MAX_PREDICTIVE_BURN_STEPS}-step safety limit."
            )
        return horizon, step

    def _target_state(
        self,
        *,
        intent: dict[str, Any],
        own_knowledge: dict[str, StateBelief],
    ) -> tuple[np.ndarray, np.ndarray] | None:
        desired_state = intent.get("desired_state_eci_6")
        if desired_state is not None:
            x_des = np.array(desired_state, dtype=float).reshape(-1)
            if x_des.size >= 6 and np.all(np.isfinite(x_des[:6])):
                return np.array(x_des[:3], dtype=float), np.array(x_des[3:6], dtype=float)
        target_id = intent.get("target_id", self.target_id)
        use_knowledge = bool(intent.get("use_knowledge_for_targeting", self.use_knowledge_for_targeting))
        return _resolve_target_state(
            target_id=(None if target_id is None else str(target_id)),
            use_knowledge_for_targeting=use_knowledge,
            own_knowledge=own_knowledge,
        )

    def _predict_eci(self, x_eci: np.ndarray, horizon_s: float, dt_s: float) -> np.ndarray:
        x = np.array(x_eci, dtype=float).reshape(6)
        horizon_s, dt_s = self._validate_prediction_horizon(horizon_s, dt_s)
        n_steps = int(max(np.floor(horizon_s / dt_s), 0))
        rem = float(max(horizon_s - n_steps * dt_s, 0.0))
        for _ in range(n_steps):
            x = propagate_two_body_rk4(x_eci=x, dt_s=dt_s, mu_km3_s2=self.mu_km3_s2, accel_cmd_eci_km_s2=np.zeros(3))
        if rem > 1e-9:
            x = propagate_two_body_rk4(x_eci=x, dt_s=rem, mu_km3_s2=self.mu_km3_s2, accel_cmd_eci_km_s2=np.zeros(3))
        return x

    def _predict_orb_belief_for_controller(
        self,
        *,
        orbit_controller: Any | None,
        self_truth: StateTruth,
        target_state_eci: tuple[np.ndarray, np.ndarray] | None,
        lead_time_s: float,
    ) -> StateBelief:
        x_self = np.hstack(
            (np.array(self_truth.position_eci_km, dtype=float), np.array(self_truth.velocity_eci_km_s, dtype=float))
        )
        horizon = float(max(lead_time_s, 0.0))
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
            x_rect = eci_relative_to_ric_rect(x_dep_eci=np.hstack((r_s, v_s)), x_chief_eci=np.hstack((r_c, v_c)))
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
        return any(tok in mode_str for tok in tokens), mode_str

    def _effective_planning_period_s(self, dt_s: float) -> float:
        if self.planning_period_s is not None:
            return float(max(self.planning_period_s, 1e-9))
        return float(max(dt_s, 1e-9))

    def _maybe_update_mode(
        self, truth: StateTruth, att_belief: StateBelief | None, attitude_controller: Any | None
    ) -> None:
        if (
            self.detumble_enter_rate_rad_s is None
            or attitude_controller is None
            or not hasattr(attitude_controller, "set_mode")
        ):
            return
        if att_belief is not None and att_belief.state.size >= 13:
            w = np.array(att_belief.state[10:13], dtype=float)
        else:
            w = np.array(truth.angular_rate_body_rad_s, dtype=float)
        w_norm = float(np.linalg.norm(w))
        enter = float(max(self.detumble_enter_rate_rad_s, 0.0))
        exit_rate = float(
            max(self.detumble_exit_rate_rad_s if self.detumble_exit_rate_rad_s is not None else enter, 0.0)
        )
        if self._detumble_latched:
            if w_norm <= exit_rate:
                self._detumble_latched = False
        elif w_norm >= enter:
            self._detumble_latched = True
        mode = self.detumble_mode_name if self._detumble_latched else self.nominal_mode_name
        try:
            attitude_controller.set_mode(mode)
        except (TypeError, ValueError, AttributeError) as exc:
            logger.warning("Unable to set attitude controller mode '%s': %s", mode, exc)

    def update(
        self,
        *,
        intent: dict[str, Any],
        truth: StateTruth,
        own_knowledge: dict[str, StateBelief],
        orbit_controller: Any | None = None,
        attitude_controller: Any | None = None,
        att_belief: StateBelief | None = None,
        t_s: float,
        dt_s: float,
        **kwargs: Any,
    ) -> dict[str, Any]:
        env = dict(kwargs.get("env", {}) or {})
        attitude_disabled = bool(env.get("attitude_disabled", False))
        out: dict[str, Any] = {}

        self._maybe_update_mode(truth=truth, att_belief=att_belief, attitude_controller=attitude_controller)
        in_detumble_mode, mode_str = self._attitude_controller_in_detumble_mode(attitude_controller)
        planning_blocked_by_detumble = bool(self.skip_orbit_planning_in_detumble_mode and in_detumble_mode)
        plan_period_s = self._effective_planning_period_s(float(dt_s))
        target_state = self._target_state(intent=intent, own_knowledge=own_knowledge)
        _apply_orbit_controller_intent(orbit_controller, intent)
        plan_due = bool(
            self._last_plan_t_s is None or (float(t_s) - float(self._last_plan_t_s)) >= (plan_period_s - 1e-12)
        )
        planned_this_step = False
        lead_time_s = float(max(intent.get("lead_time_s", self.lead_time_s), 0.0))

        if planning_blocked_by_detumble:
            self._countdown_s = -1.0
            self._planned_accel_eci_km_s2 = np.zeros(3, dtype=float)
            self._planned_attitude_quat_bn = np.array(truth.attitude_quat_bn, dtype=float)
        elif self._countdown_s < 0.0 and plan_due:
            b_pred = self._predict_orb_belief_for_controller(
                orbit_controller=orbit_controller,
                self_truth=truth,
                target_state_eci=target_state,
                lead_time_s=lead_time_s,
            )
            c_orb_pred = (
                orbit_controller.act(b_pred, float(t_s), float(max(self.orbit_controller_budget_ms, 1e-9)))
                if orbit_controller is not None
                else Command.zero()
            )
            self._planned_accel_eci_km_s2 = np.array(c_orb_pred.thrust_eci_km_s2, dtype=float).reshape(3)
            if not np.all(np.isfinite(self._planned_accel_eci_km_s2)):
                self._planned_accel_eci_km_s2 = np.zeros(3, dtype=float)
            if float(np.linalg.norm(self._planned_accel_eci_km_s2)) <= 1e-15 and "fallback_thrust_eci_km_s2" in intent:
                self._planned_accel_eci_km_s2 = np.array(intent.get("fallback_thrust_eci_km_s2"), dtype=float).reshape(
                    3
                )
            dv_pred = self._planned_accel_eci_km_s2 * float(max(self.predict_dt_s, 1e-6))
            q_req = OrbitalAttitudeManeuverCoordinator().maneuverer.required_attitude_for_delta_v(
                truth=truth,
                delta_v_eci_km_s=dv_pred,
                thruster_direction_body=np.array(self.thruster_direction_body, dtype=float),
            )
            self._planned_attitude_quat_bn = np.array(
                q_req if q_req is not None else truth.attitude_quat_bn, dtype=float
            )
            self._countdown_s = lead_time_s
            self._last_plan_t_s = float(t_s)
            planned_this_step = True

        if (not attitude_disabled) and attitude_controller is not None and hasattr(attitude_controller, "set_target"):
            try:
                attitude_controller.set_target(np.array(self._planned_attitude_quat_bn, dtype=float))
            except (TypeError, ValueError, AttributeError) as exc:
                logger.warning("Failed to set predictive burn attitude target: %s", exc)
        att_belief_eff = att_belief
        if (not attitude_disabled) and att_belief_eff is None and attitude_controller is not None:
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
            if lead_time_s <= 0.0:
                fire = bool(
                    align_ok
                    and float(np.linalg.norm(self._planned_accel_eci_km_s2))
                    > float(max(self.min_burn_accel_km_s2, 0.0))
                )
                self._countdown_s = 0.0
            elif planned_this_step:
                # A command returned by this update applies to the interval
                # beginning at ``t_s``.  Keep a newly planned countdown intact
                # so a one-step lead time cannot fire in that same interval.
                fire = False
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
            **dict(intent.get("mission_mode", {}) or {}),
            "execution": "predictive_burn",
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

@dataclass
class ImpulsiveExecution:
    align_thruster_to_thrust: bool = True
    thruster_direction_body: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0], dtype=float))
    require_attitude_alignment: bool = True
    alignment_tolerance_rad: float = np.deg2rad(5.0)
    alignment_tolerance_deg: float | None = None
    use_strategy_fallback_thrust: bool = True
    pulse_period_s: float = 60.0
    pulse_width_s: float = 5.0
    pulse_phase_s: float = 0.0
    min_burn_accel_km_s2: float = 1e-12
    orbit_controller_budget_ms: float = 2.0
    attitude_controller_budget_ms: float = 2.0

    def __post_init__(self) -> None:
        self.alignment_tolerance_rad = _resolve_angle_tolerance_rad(
            self.alignment_tolerance_rad, self.alignment_tolerance_deg
        )

    def _pulse_active(self, t_s: float) -> bool:
        period = float(max(self.pulse_period_s, 1e-9))
        width = float(np.clip(self.pulse_width_s, 0.0, period))
        if width <= 0.0:
            return False
        if width >= period:
            return True
        phase = float(self.pulse_phase_s)
        tau = (float(t_s) - phase) % period
        return tau < width

    def _pulse_active_duration_s(self, t_s: float, dt_s: float) -> float:
        """Return pulse-on time in the half-open interval ``[t_s, t_s + dt_s)``."""

        interval_s = float(max(dt_s, 0.0))
        period = float(max(self.pulse_period_s, 1e-9))
        width = float(np.clip(self.pulse_width_s, 0.0, period))
        if interval_s <= 0.0 or width <= 0.0:
            return 0.0
        if width >= period:
            return interval_s

        def cumulative_active_time(relative_t_s: float) -> float:
            cycles = float(np.floor(relative_t_s / period))
            cycle_time_s = relative_t_s - cycles * period
            return cycles * width + min(cycle_time_s, width)

        start_s = float(t_s) - float(self.pulse_phase_s)
        active_s = cumulative_active_time(start_s + interval_s) - cumulative_active_time(start_s)
        return float(np.clip(active_s, 0.0, interval_s))

    def update(
        self,
        *,
        intent: dict[str, Any],
        truth: StateTruth,
        t_s: float,
        orbit_controller: Any | None = None,
        attitude_controller: Any | None = None,
        orb_belief: StateBelief | None = None,
        att_belief: StateBelief | None = None,
        dt_s: float | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        _apply_orbit_controller_intent(orbit_controller, intent)
        c_orb = (
            orbit_controller.act(orb_belief, float(t_s), float(max(self.orbit_controller_budget_ms, 1e-9)))
            if (orbit_controller is not None and orb_belief is not None)
            else Command.zero()
        )
        thrust_cmd = np.array(c_orb.thrust_eci_km_s2, dtype=float).reshape(3)
        if (
            float(np.linalg.norm(thrust_cmd)) <= 1e-15
            and self.use_strategy_fallback_thrust
            and "fallback_thrust_eci_km_s2" in intent
        ):
            thrust_cmd = np.array(intent.get("fallback_thrust_eci_km_s2"), dtype=float).reshape(3)

        q_des = intent.get("desired_attitude_quat_bn")
        if (
            q_des is None
            and bool(intent.get("align_to_thrust", self.align_thruster_to_thrust))
            and float(np.linalg.norm(thrust_cmd)) > 1e-15
        ):
            q_des = _desired_attitude_for_thrust(
                truth=truth,
                thrust_eci_km_s2=thrust_cmd,
                thruster_direction_body=np.array(self.thruster_direction_body, dtype=float),
            )
        q_des_arr = None if q_des is None else np.array(q_des, dtype=float).reshape(4)
        if q_des_arr is not None and attitude_controller is not None and hasattr(attitude_controller, "set_target"):
            try:
                attitude_controller.set_target(q_des_arr)
            except (TypeError, ValueError, AttributeError) as exc:
                logger.warning("Failed to set impulsive execution attitude target: %s", exc)
        c_att = (
            attitude_controller.act(att_belief, float(t_s), float(max(self.attitude_controller_budget_ms, 1e-9)))
            if (attitude_controller is not None and att_belief is not None)
            else Command.zero()
        )

        tol_rad = float(max(self.alignment_tolerance_rad, 0.0))
        alignment_ok = True
        alignment_angle_rad = 0.0
        if float(np.linalg.norm(thrust_cmd)) > float(max(self.min_burn_accel_km_s2, 0.0)):
            alignment_ok, alignment_angle_rad = PredictiveBurnExecution._alignment(
                self, truth=truth, accel_eci_km_s2=thrust_cmd
            )
        if dt_s is None:
            pulse_active_duration_s = 0.0
            pulse_duty_fraction = 1.0 if self._pulse_active(float(t_s)) else 0.0
        else:
            interval_s = float(max(dt_s, 0.0))
            pulse_active_duration_s = self._pulse_active_duration_s(float(t_s), interval_s)
            pulse_duty_fraction = pulse_active_duration_s / interval_s if interval_s > 0.0 else 0.0
        pulse_active = pulse_duty_fraction > 0.0
        fire = bool(
            pulse_active
            and float(np.linalg.norm(thrust_cmd)) > float(max(self.min_burn_accel_km_s2, 0.0))
            and ((not self.require_attitude_alignment) or alignment_ok)
        )
        if self.require_attitude_alignment and alignment_angle_rad > tol_rad:
            fire = False

        return {
            "mission_use_integrated_command": True,
            "mission_bypass_orbital_command_latch": True,
            "thrust_eci_km_s2": thrust_cmd * pulse_duty_fraction if fire else np.zeros(3, dtype=float),
            "torque_body_nm": np.array(c_att.torque_body_nm, dtype=float).reshape(3),
            "desired_attitude_quat_bn": q_des_arr,
            "command_mode_flags": {
                **dict(c_att.mode_flags or {}),
                "execution": "impulsive",
                "pulse_active": bool(pulse_active),
                "pulse_active_duration_s": float(pulse_active_duration_s),
                "pulse_duty_fraction": float(pulse_duty_fraction),
                "alignment_ok": bool(alignment_ok),
            },
            "mission_mode": {
                **dict(intent.get("mission_mode", {}) or {}),
                "execution": "impulsive",
                "pulse_active": bool(pulse_active),
                "pulse_active_duration_s": float(pulse_active_duration_s),
                "pulse_duty_fraction": float(pulse_duty_fraction),
                "fire": bool(fire),
                "alignment_ok": bool(alignment_ok),
                "alignment_angle_rad": float(alignment_angle_rad),
            },
        }


@dataclass
class BudgetedEndStateExecution:
    strategy: ManeuverStrategy = "thrust_limited"
    max_thrust_n: float = 0.2
    min_thrust_n: float = 0.0
    burn_dt_s: float = 1.0
    available_delta_v_km_s: float = 0.5
    require_attitude_alignment: bool = True
    thruster_position_body_m: np.ndarray | None = field(default_factory=lambda: np.zeros(3, dtype=float))
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

    def update(
        self,
        *,
        intent: dict[str, Any],
        truth: StateTruth,
        dt_s: float = 1.0,
        dry_mass_kg: float | None = None,
        orbital_isp_s: float | None = None,
        orbit_command_period_s: float | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        env = dict(kwargs.get("env", {}) or {})
        attitude_disabled = bool(env.get("attitude_disabled", False))
        out: dict[str, Any] = {}
        x_des = intent.get("desired_state_eci_6")
        if x_des is None:
            out["mission_mode"] = {
                **dict(intent.get("mission_mode", {}) or {}),
                "execution": "budgeted_end_state",
                "phase": "hold_no_target",
            }
            return out
        x_des_arr = np.array(x_des, dtype=float).reshape(-1)
        if x_des_arr.size != 6:
            out["mission_mode"] = {
                **dict(intent.get("mission_mode", {}) or {}),
                "execution": "budgeted_end_state",
                "phase": "hold_no_target",
            }
            return out
        dv_eci = x_des_arr[3:6] - np.array(truth.velocity_eci_km_s, dtype=float)
        if float(np.linalg.norm(dv_eci)) <= max(float(self.terminate_on_velocity_tolerance_km_s), 0.0):
            out["mission_mode"] = {
                **dict(intent.get("mission_mode", {}) or {}),
                "execution": "budgeted_end_state",
                "phase": "on_target",
            }
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
            out["thrust_eci_km_s2"] = _unit(dv_eci) * (float(decision.applied_delta_v_km_s) / burn_window_s)

        out["mission_mode"] = {
            **dict(intent.get("mission_mode", {}) or {}),
            "execution": "budgeted_end_state",
            "phase": decision.action,
            "reason": decision.reason,
            "alignment_ok": bool(decision.alignment_ok),
            "remaining_delta_v_km_s": float(max(decision.remaining_delta_v_km_s, 0.0)),
            "applied_delta_v_km_s": float(decision.applied_delta_v_km_s),
        }
        return out
