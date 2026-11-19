from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from time import perf_counter
from typing import Any

import numpy as np

from sim.core.models import Command, StateBelief, StateTruth
from sim.dynamics.model import OrbitalAttitudeDynamics
from sim.dynamics.orbit.cr3bp import cr3bp_system
from sim.dynamics.orbit.environment import EARTH_RADIUS_KM
from sim.dynamics.orbit.epoch import TIME_DEPENDENT_ENV_CACHE_KEY
from sim.dynamics.reentry import evaluate_reentry_termination, locate_reentry_termination_crossing
from sim.execution.passive_history import PassiveHistoryCache
from sim.rocket.navigation import build_rocket_nav_state
from sim.rocket.touchdown import RocketTouchdown
from sim.runtime.satellites.flight_software_runtime import SatellitePhysicalCommand
from sim.runtime_support import (
    AgentRuntime,
    _earth_impact_policy_for_object,
    _orbital_elements_basic,
    _rocket_altitude_km,
    _rocket_state_to_truth,
    _truth_state6,
)
from sim.utils.quaternion import quaternion_to_dcm_bn

logger = logging.getLogger(__name__)


class _BudgetedControllerProxy:
    """Apply the configured runtime policy to controller calls made by mission code."""

    def __init__(self, base: Any, *, budget_ms: float, deadline_policy: str) -> None:
        self.base = base
        self.budget_ms = float(budget_ms)
        self.deadline_policy = str(deadline_policy)
        self.runtime_ms = 0.0
        self.deadline_missed = False
        self.call_count = 0

    def act(self, belief: StateBelief, t_s: float, budget_ms: float | None = None) -> Command:
        started = perf_counter()
        command = self.base.act(belief, t_s, self.budget_ms)
        elapsed_ms = (perf_counter() - started) * 1000.0
        self.runtime_ms += float(elapsed_ms)
        self.call_count += 1
        missed = elapsed_ms > self.budget_ms
        self.deadline_missed = bool(self.deadline_missed or missed)
        if missed and self.deadline_policy == "error":
            raise TimeoutError(
                f"Controller deadline missed: runtime={elapsed_ms:.6g} ms, budget={self.budget_ms:.6g} ms."
            )
        if missed and self.deadline_policy == "zero_command":
            return Command.zero()
        return command

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base, name)


@dataclass
class _DecisionContext:
    agent: AgentRuntime
    internal_world_truth: dict[str, StateTruth]
    t_s: float
    dt_s: float
    env: dict[str, Any]
    orbit_controller: Any | None = None
    attitude_controller: Any | None = None
    orb_belief: StateBelief | None = None
    att_belief: StateBelief | None = None


class _DecisionContextBuilder:
    def __init__(
        self,
        *,
        base_environment: dict[str, Any],
        attitude_enabled: bool,
        orbit_command_period_s: float,
    ) -> None:
        self.base_environment = base_environment
        self.attitude_enabled = bool(attitude_enabled)
        self.orbit_command_period_s = float(orbit_command_period_s)

    def outer_context(
        self,
        *,
        agent: AgentRuntime,
        internal_world_truth: dict[str, StateTruth],
        t_s: float,
        dt_s: float,
    ) -> _DecisionContext:
        return _DecisionContext(
            agent=agent,
            internal_world_truth=internal_world_truth,
            t_s=float(t_s),
            dt_s=float(dt_s),
            env={**self.base_environment, "attitude_disabled": (not self.attitude_enabled)},
        )

    def satellite_context(
        self,
        *,
        agent: AgentRuntime,
        internal_world_truth: dict[str, StateTruth],
        t_s: float,
        dt_s: float,
        orb_belief: StateBelief | None,
        att_belief: StateBelief | None,
    ) -> _DecisionContext:
        return _DecisionContext(
            agent=agent,
            internal_world_truth=internal_world_truth,
            t_s=float(t_s),
            dt_s=float(dt_s),
            env={**self.base_environment, "orbit_command_period_s": self.orbit_command_period_s},
            orbit_controller=agent.orbit_controller,
            attitude_controller=(agent.attitude_controller if self.attitude_enabled else None),
            orb_belief=orb_belief,
            att_belief=(att_belief if self.attitude_enabled else None),
        )


@dataclass
class _SatelliteStepResult:
    truth: StateTruth
    average_thrust_eci_km_s2: np.ndarray
    average_torque_body_nm: np.ndarray
    delta_v_m_s: float
    max_accel_km_s2: float
    burned: bool


@dataclass
class _RocketStepResult:
    truth: StateTruth
    throttle: float
    thrust_eci_km_s2: np.ndarray
    torque_body_nm: np.ndarray
    delta_v_m_s: float
    max_accel_km_s2: float
    burned: bool
    stage_index: float | None = None
    q_dyn_pa: float | None = None
    mach: float | None = None
    metrics: dict[str, float] | None = None
    touchdown: RocketTouchdown | None = None


@dataclass
class _RocketStepper:
    def __init__(self, engine: Any) -> None:
        self.engine = engine

    def _guidance_phase_code(self, guidance: Any) -> float:
        phase_map = {
            "ascent": 1.0,
            "coast_to_apogee": 2.0,
            "circularize": 3.0,
            "complete": 4.0,
        }
        current = guidance
        for _ in range(8):
            phase = getattr(current, "_phase", None)
            if phase is not None:
                return float(phase_map.get(str(phase), np.nan))
            current = getattr(current, "base_guidance", None)
            if current is None:
                break
        return float("nan")

    def step(
        self,
        *,
        agent: AgentRuntime,
        world_truth_decision: dict[str, StateTruth],
        t_s: float,
        t_next: float,
    ) -> _RocketStepResult:
        e = self.engine
        step_dt_s = float(t_next) - float(t_s)
        if step_dt_s <= 0.0:
            raise ValueError("Rocket step interval must be positive.")
        mission_out = e._run_agent_decision(
            e.decision_contexts.outer_context(
                agent=agent,
                internal_world_truth=world_truth_decision,
                t_s=t_s,
                dt_s=step_dt_s,
            ),
        )
        launch_auth = bool(mission_out.get("launch_authorized", True))
        agent.waiting_for_launch = not launch_auth
        if not launch_auth:
            agent.rocket_sim.hold_on_launch_pad(agent.rocket_state, t_s=float(t_next))
            agent.truth = _rocket_state_to_truth(agent.rocket_state)
            if agent.belief is not None:
                agent.belief.state[:6] = _truth_state6(agent.truth, agent.belief.state[:6])
                agent.belief.last_update_t_s = t_next
            return _RocketStepResult(
                truth=agent.truth,
                throttle=0.0,
                thrust_eci_km_s2=e.zero3.copy(),
                torque_body_nm=e.zero3.copy(),
                delta_v_m_s=0.0,
                max_accel_km_s2=0.0,
                burned=False,
                stage_index=float(agent.rocket_state.active_stage_index),
                q_dyn_pa=0.0,
                mach=0.0,
                metrics={},
            )

        cmd = agent.rocket_guidance.command(agent.rocket_state, agent.rocket_sim.sim_cfg, agent.rocket_sim.vehicle_cfg)
        if "landing_gear_commands" in mission_out:
            cmd = replace(cmd, landing_gear_commands=tuple(mission_out["landing_gear_commands"]))
        if "guidance_throttle" in mission_out:
            cmd = replace(cmd, throttle=float(mission_out.get("guidance_throttle", cmd.throttle)))
        throttle = float(np.clip(cmd.throttle, 0.0, 1.0))
        start = agent.rocket_state
        agent.rocket_state = agent.rocket_sim.step(start, cmd, dt_s=step_dt_s)
        touchdown = agent.rocket_sim.resolve_touchdown(start, agent.rocket_state, cmd)
        agent.truth = _rocket_state_to_truth(agent.rocket_state)
        if agent.belief is not None:
            agent.belief.state[:6] = _truth_state6(agent.truth, agent.belief.state[:6])
            agent.belief.last_update_t_s = agent.rocket_state.t_s
        thrust_n = float(getattr(agent.rocket_state, "_last_step_thrust_n", 0.0))
        missing_acceleration = object()
        step_acceleration = getattr(agent.rocket_state, "_last_step_thrust_accel_eci_km_s2", missing_acceleration)
        if step_acceleration is missing_acceleration:
            fallback_axis_eci = quaternion_to_dcm_bn(agent.rocket_state.attitude_quat_bn).T @ np.array(
                getattr(agent.rocket_state, "thrust_vector_body", agent.rocket_sim.vehicle_cfg.thrust_axis_body),
                dtype=float,
            )
            step_acceleration = (thrust_n / max(agent.rocket_state.mass_kg, 1e-9)) * fallback_axis_eci / 1e3
        accel = np.array(step_acceleration, dtype=float)
        accel_mag = float(np.linalg.norm(accel))
        nav = build_rocket_nav_state(
            agent.rocket_state,
            agent.rocket_sim.sim_cfg,
            agent.rocket_sim.vehicle_cfg,
            throttle_cmd=throttle,
            thrust_n=thrust_n,
        )
        metrics = {
            "altitude_km": float(nav.altitude_km),
            "speed_km_s": float(nav.speed_km_s),
            "vertical_speed_km_s": float(nav.vertical_speed_km_s),
            "horizontal_speed_km_s": float(nav.horizontal_speed_km_s),
            "flight_path_angle_deg": float(nav.flight_path_angle_deg),
            "apoapsis_alt_km": float(nav.apoapsis_alt_km),
            "periapsis_alt_km": float(nav.periapsis_alt_km),
            "eccentricity": float(nav.eccentricity),
            "alpha_deg": float(getattr(agent.rocket_state, "_last_step_alpha_deg", nav.alpha_deg)),
            "beta_deg": float(getattr(agent.rocket_state, "_last_step_beta_deg", nav.beta_deg)),
            "tvc_gimbal_deg": float(getattr(agent.rocket_state, "_last_step_tvc_gimbal_deg", 0.0)),
            "aero_force_n": float(getattr(agent.rocket_state, "_last_step_aero_force_n", 0.0)),
            "aero_moment_nm": float(getattr(agent.rocket_state, "_last_step_aero_moment_nm", 0.0)),
            "thrust_to_weight": float(nav.thrust_to_weight),
            "propellant_remaining_kg": float(nav.propellant_remaining_kg),
            "propellant_remaining_fraction": float(nav.propellant_remaining_fraction),
            "guidance_phase_code": self._guidance_phase_code(agent.rocket_guidance),
        }
        return _RocketStepResult(
            truth=agent.truth,
            throttle=throttle,
            thrust_eci_km_s2=accel,
            torque_body_nm=np.array(getattr(agent.rocket_state, "_last_step_torque_body_nm", e.zero3), dtype=float).reshape(3),
            delta_v_m_s=accel_mag * (agent.rocket_state.t_s - start.t_s) * 1e3,
            max_accel_km_s2=accel_mag,
            burned=bool(accel_mag > 1e-15),
            stage_index=float(agent.rocket_state.active_stage_index),
            q_dyn_pa=float(getattr(agent.rocket_state, "_last_step_q_dyn_pa", 0.0)),
            mach=float(getattr(agent.rocket_state, "_last_step_mach", 0.0)),
            metrics=metrics,
            touchdown=touchdown,
        )


class _SatelliteStepper:
    def __init__(self, engine: Any) -> None:
        self.engine = engine
        self.passive_history = PassiveHistoryCache(engine)
        self._realization_cursor_by_object: dict[str, int] = {}

    def step(
        self,
        *,
        aid: str,
        agent: AgentRuntime,
        initial_truth: StateTruth,
        world_truth_decision: dict[str, StateTruth],
        t_s: float,
        t_next: float,
        sample_index: int,
    ) -> _SatelliteStepResult:
        if agent.flight_software_runtime is None:
            if str(getattr(agent, "runtime_profile", "flight_software")) == "trajectory_only":
                return self._step_trajectory_only(
                    aid=aid,
                    agent=agent,
                    initial_truth=initial_truth,
                    world_truth_decision=world_truth_decision,
                    t_s=t_s,
                    t_next=t_next,
                )
            raise RuntimeError(f"satellite {aid!r} has no complete v2 flight-software runtime")
        return self._step_v2(
            aid=aid,
            agent=agent,
            initial_truth=initial_truth,
            world_truth_decision=world_truth_decision,
            t_s=t_s,
            t_next=t_next,
            sample_index=sample_index,
        )

    def _step_v2(
        self,
        *,
        aid: str,
        agent: AgentRuntime,
        initial_truth: StateTruth,
        world_truth_decision: dict[str, StateTruth],
        t_s: float,
        t_next: float,
        sample_index: int,
    ) -> _SatelliteStepResult:
        """Advance a satellite whose sole decision owner is its v2 stack."""

        e = self.engine
        runtime = agent.flight_software_runtime
        if runtime is None:
            raise RuntimeError("v2 satellite step requires a flight-software runtime")
        start_interval_ns = int(round(float(t_s) * 1.0e9))
        final_interval_ns = int(round(float(t_next) * 1.0e9))
        current_ns = start_interval_ns
        interval_s = max(float(t_next) - float(t_s), 1.0e-12)
        tr_inner = initial_truth
        accel_time_integral = e.zero3.copy()
        torque_time_integral = e.zero3.copy()
        step_delta_v_m_s = 0.0
        step_max_accel_km_s2 = 0.0
        burned_this_step = False
        env_inner = {
            **e.base_environment,
            "object_id": aid,
            "world_truth": dict(world_truth_decision),
            "attitude_disabled": (not e.attitude_enabled),
            TIME_DEPENDENT_ENV_CACHE_KEY: getattr(e, "_time_dependent_env_cache", {}),
        }
        missing_forecast_dynamics = object()
        substep_ns = max(1, int(round(float(e.sim_substep_s) * 1.0e9)))
        while current_ns < final_interval_ns:
            start_ns = current_ns
            maximum_end_ns = min(start_ns + substep_ns, final_interval_ns)
            control_available_ns = (
                None
                if agent.control_available_time_s is None
                else int(round(float(agent.control_available_time_s) * 1.0e9))
            )
            if control_available_ns is not None and start_ns < control_available_ns <= maximum_end_ns:
                maximum_end_ns = control_available_ns
            forecast_dynamics_by_object = getattr(
                e,
                "_forecast_dynamics_by_object",
                missing_forecast_dynamics,
            )
            if forecast_dynamics_by_object is missing_forecast_dynamics:
                forecast_dynamics_by_object = {
                    object_id: runtime.dynamics for object_id, runtime in e.agents.items()
                }
            world_truth_inner = _retime_decision_truth(
                world_truth_decision,
                source_time_s=t_s,
                target_time_s=start_ns / 1.0e9,
                own_id=aid,
                own_truth=tr_inner,
                dynamics_by_object=forecast_dynamics_by_object,
                environment=env_inner,
                forecast_cache=getattr(e, "_forecast_truth_cache", None),
            )
            runtime.prepare_interval(
                tr_inner,
                start_time_ns=start_ns,
                world_truth=world_truth_inner,
            )
            end_ns = runtime.next_hard_boundary_ns(
                after_time_ns=start_ns,
                before_time_ns=maximum_end_ns,
            )
            h = (end_ns - start_ns) / 1.0e9
            if h <= 0.0:
                raise RuntimeError("satellite event scheduler produced a nonpositive interval")
            if control_available_ns is not None and start_ns < control_available_ns:
                # Initialization delay is a physical command inhibit.  The
                # onboard clock and tasks may run, but no queued command can
                # reach hardware until the declared availability boundary.
                physical = SatellitePhysicalCommand(
                    (0.0, 0.0, 0.0),
                    (0.0, 0.0, 0.0),
                    (0.0, 0.0, 0.0),
                    0.0,
                    (),
                )
            else:
                physical = runtime.command_interval(
                    tr_inner,
                    start_time_ns=start_ns,
                    end_time_ns=end_ns,
                    world_truth=world_truth_inner,
                )
            c_bn_start = quaternion_to_dcm_bn(np.asarray(tr_inner.attitude_quat_bn, dtype=float))
            total_force_start_eci = np.asarray(physical.force_eci_n, dtype=float) + c_bn_start.T @ np.asarray(
                physical.force_body_n,
                dtype=float,
            )
            acceleration = total_force_start_eci / max(float(tr_inner.mass_kg), 1.0e-12)
            acceleration_km_s2 = acceleration / 1.0e3
            torque_body = np.asarray(physical.torque_body_n_m, dtype=float)
            command = Command(
                acceleration_km_s2,
                torque_body,
                {
                    "physical_force_eci_n": tuple(float(value) for value in physical.force_eci_n),
                    "physical_force_body_n": tuple(float(value) for value in physical.force_body_n),
                    "mass_flow_kg_s": float(physical.mass_flow_kg_s),
                    "delta_mass_kg": float(physical.mass_flow_kg_s) * h,
                    "min_mass_kg": float(runtime.dry_mass_kg),
                },
            )
            physical_env = {**env_inner, **runtime.physics_environment(tr_inner, physical)}
            tr_inner = agent.dynamics.step(state=tr_inner, command=command, env=physical_env, dt_s=h)
            applied_acceleration = np.asarray(command.thrust_eci_km_s2, dtype=float)
            accel_time_integral += applied_acceleration * h
            torque_time_integral += torque_body * h
            accel_mag = float(np.linalg.norm(applied_acceleration))
            step_delta_v_m_s += accel_mag * h * 1.0e3
            step_max_accel_km_s2 = max(step_max_accel_km_s2, accel_mag)
            burned_this_step = burned_this_step or accel_mag > 1.0e-15
            current_ns = end_ns
        realizations, cursor = runtime.realizations_since(self._realization_cursor_by_object.get(aid, 0))
        self._realization_cursor_by_object[aid] = cursor
        e.command_decision_hist[aid].extend(
            {
                "sample_index": sample_index,
                "time_s": float(item.interval_start_ns) / 1.0e9,
                "interval_end_time_s": float(item.interval_end_ns) / 1.0e9,
                "object_id": aid,
                "boundary": "SatelliteFlightSoftware",
                "actuator_id": item.actuator_id,
                "source_command_id": (
                    None
                    if item.source_command_id is None
                    else {
                        "source_id": item.source_command_id.source_id,
                        "boot_id": item.source_command_id.boot_id,
                        "sequence": item.source_command_id.sequence,
                    }
                ),
                "requested_force_n": list(item.requested_force_n),
                "requested_torque_n_m": list(item.requested_torque_n_m),
                "realized_force_n": list(item.realized_force_n),
                "realized_torque_n_m": list(item.realized_torque_n_m),
                "demand_mode": item.demand_mode.value,
                "saturated": item.saturated,
            }
            for item in realizations
        )
        return _SatelliteStepResult(
            truth=tr_inner,
            average_thrust_eci_km_s2=accel_time_integral / interval_s,
            average_torque_body_nm=e.zero3 if not e.attitude_enabled else torque_time_integral / interval_s,
            delta_v_m_s=step_delta_v_m_s,
            max_accel_km_s2=step_max_accel_km_s2,
            burned=burned_this_step,
        )

    def _step_trajectory_only(
        self,
        *,
        aid: str,
        agent: AgentRuntime,
        initial_truth: StateTruth,
        world_truth_decision: dict[str, StateTruth],
        t_s: float,
        t_next: float,
    ) -> _SatelliteStepResult:
        """Advance deterministic dynamics without constructing an onboard runtime."""

        e = self.engine
        native_truth = self.passive_history.step(
            aid=aid, agent=agent, initial=initial_truth, t_s=t_s, t_next=t_next,
        )
        if native_truth is not None:
            return _SatelliteStepResult(
                truth=native_truth, average_thrust_eci_km_s2=e.zero3.copy(),
                average_torque_body_nm=e.zero3.copy(), delta_v_m_s=0.0,
                max_accel_km_s2=0.0, burned=False,
            )
        tr_inner = initial_truth
        current_s = float(t_s)
        final_s = float(t_next)
        substep_s = max(float(e.sim_substep_s), 1.0e-12)
        environment = {
            **e.base_environment,
            "object_id": aid,
            "world_truth": dict(world_truth_decision),
            "attitude_disabled": (not e.attitude_enabled),
            TIME_DEPENDENT_ENV_CACHE_KEY: getattr(e, "_time_dependent_env_cache", {}),
        }
        # Checked-in dynamics treat commands as immutable inputs. Reuse one
        # zero command across internal substeps while retaining the historical
        # fresh-command behavior for third-party dynamics implementations.
        zero_command = Command.zero() if isinstance(agent.dynamics, OrbitalAttitudeDynamics) else None
        while current_s < final_s:
            h = min(substep_s, final_s - current_s)
            if h <= 0.0:
                raise RuntimeError("trajectory-only dynamics produced a nonpositive interval")
            environment["world_truth"] = {**world_truth_decision, aid: tr_inner}
            tr_inner = agent.dynamics.step(
                state=tr_inner,
                command=zero_command if zero_command is not None else Command.zero(),
                env=environment,
                dt_s=h,
            )
            current_s += h
        return _SatelliteStepResult(
            truth=tr_inner,
            average_thrust_eci_km_s2=e.zero3.copy(),
            average_torque_body_nm=e.zero3.copy(),
            delta_v_m_s=0.0,
            max_accel_km_s2=0.0,
            burned=False,
        )


def _retime_decision_truth(
    world_truth: dict[str, StateTruth],
    *,
    source_time_s: float,
    target_time_s: float,
    own_id: str,
    own_truth: StateTruth,
    dynamics_by_object: dict[str, Any] | None = None,
    environment: dict[str, Any] | None = None,
    forecast_cache: dict[tuple[float, float, str, int, int], StateTruth] | None = None,
) -> dict[str, StateTruth]:
    """Put decision-snapshot objects at the exact onboard sample time.

    Other satellites are coast-propagated by their configured deterministic
    dynamics model.  This keeps gravity, perturbations, and frame rate
    consistent at faster onboard releases without exposing another object's
    future commanded trajectory.  Non-dynamic objects retain a documented
    constant-velocity fallback.
    """

    from sim.dynamics.orbit.rust_forecast import eligible, forecast_batch

    elapsed = float(target_time_s) - float(source_time_s)
    resolved: dict[str, StateTruth] = {own_id: own_truth}
    prediction_environment = {
        **dict(environment or {}), "world_truth": dict(world_truth), "attitude_disabled": True,
    }
    pending = []

    def store(object_id, cache_key, predicted):
        if forecast_cache is not None:
            forecast_cache[cache_key] = predicted
        resolved[object_id] = predicted.copy()

    def flush():
        for entry, predicted in forecast_batch(pending, elapsed, prediction_environment):
            store(entry[0], entry[1], predicted)
        pending.clear()

    for object_id, truth in world_truth.items():
        if object_id == own_id:
            continue
        dynamics = None if dynamics_by_object is None else dynamics_by_object.get(object_id)
        cache_key = (float(source_time_s), float(target_time_s), str(object_id), id(truth), id(dynamics))
        predicted = None if forecast_cache is None else forecast_cache.get(cache_key)
        if predicted is None and eligible(dynamics, truth, elapsed, prediction_environment):
            pending.append((object_id, cache_key, dynamics, truth))
            continue
        # Flush before a callback, cached row or fallback to retain serial order.
        if pending:
            flush()
        if predicted is None:
            if elapsed > 0.0 and dynamics is not None:
                predicted = dynamics.step(state=truth.copy(), command=Command.zero(),
                                          env=dict(prediction_environment), dt_s=elapsed)
            else:
                predicted = truth.copy()
                predicted.position_eci_km = np.asarray(truth.position_eci_km, dtype=float) + elapsed * np.asarray(
                    truth.velocity_eci_km_s, dtype=float)
                predicted.t_s = float(target_time_s)
            store(object_id, cache_key, predicted)
        else:
            resolved[object_id] = predicted.copy()
    if pending:
        flush()

    return resolved


class _KnowledgeSynchronizer:
    def __init__(self, engine: Any) -> None:
        self.engine = engine

    def initialize(self) -> None:
        e = self.engine
        initial_world_truth = {
            aid: (agent.truth if agent.kind == "satellite" else _rocket_state_to_truth(agent.rocket_state))
            for aid, agent in e.agents.items()
            if agent.active
        }
        if getattr(e, "target_reference_truth", None) is not None:
            initial_world_truth["target_reference"] = e.target_reference_truth.copy()
        for aid, agent in e.agents.items():
            if not agent.active or agent.knowledge_base is None:
                continue
            observer_truth = initial_world_truth.get(aid)
            if observer_truth is None:
                continue
            agent.knowledge_base.update(
                observer_truth=observer_truth,
                world_truth=initial_world_truth,
                t_s=0.0,
                observer_belief=agent.belief,
            )
            self._record_snapshot(aid=aid, sample_index=0)

    def update_after_step(
        self,
        *,
        world_truth: dict[str, StateTruth],
        sample_index: int,
        t_s: float,
    ) -> None:
        e = self.engine
        for aid, agent in e.agents.items():
            if not agent.active or agent.knowledge_base is None:
                continue
            observer_truth = world_truth.get(aid)
            if observer_truth is None:
                continue
            agent.knowledge_base.update(
                observer_truth=observer_truth,
                world_truth=world_truth,
                t_s=t_s,
                observer_belief=agent.belief,
            )
            self._record_snapshot(aid=aid, sample_index=sample_index)

    def _record_snapshot(self, *, aid: str, sample_index: int) -> None:
        e = self.engine
        agent = e.agents[aid]
        if agent.knowledge_base is None:
            return
        snap = agent.knowledge_base.snapshot()
        measurements = agent.knowledge_base.measurement_snapshot()
        for tid, hist in e.knowledge_hist.get(aid, {}).items():
            belief = snap.get(tid)
            if belief is not None:
                hist[sample_index, :] = belief.state[:6]
            elif sample_index > 0:
                hist[sample_index, :] = hist[sample_index - 1, :]
            meas = measurements.get(tid)
            measurement_hist = getattr(e, "knowledge_measurement_hist", {}).get(aid, {}).get(tid)
            if meas is not None and measurement_hist is not None:
                n = min(int(meas.size), int(measurement_hist.shape[1]))
                measurement_hist[sample_index, :n] = meas[:n]


class _TerminationMonitor:
    def __init__(self, engine: Any) -> None:
        self.engine = engine

    def check_rocket_touchdown(
        self, *, t_s: float, object_id: str, contact: RocketTouchdown,
    ) -> None:
        e = self.engine
        # Other termination monitors operate on the trial step. Retain the
        # earliest event they or the rocket model localized on that interval.
        candidates = [(contact.state.t_s, contact.termination_reason, object_id)]
        if e.termination_time_s is not None:
            candidates.append((e.termination_time_s, e.termination_reason, e.termination_object_id))
        for monitor in (self.check_reentry, self.check_earth_impact):
            if monitor(t_s=t_s):
                candidates.append((e.termination_time_s, e.termination_reason, e.termination_object_id))
        e.termination_time_s, e.termination_reason, e.termination_object_id = min(
            candidates, key=lambda event: event[0],
        )
        e.terminated_early = True
        if (e.termination_time_s, e.termination_reason, e.termination_object_id) == candidates[0]:
            e.rocket_terminal_state = {"object_id": object_id, **contact.evidence}
            event = contact.state.gear_touchdown if contact.state.gear_terminal is not None else contact.evidence
            e.rocket_touchdown = {"object_id": object_id, **event}

    def check_earth_impact(self, *, t_s: float, include_rockets: bool = True) -> bool:
        e = self.engine
        orbit_config = dict(getattr(e.cfg.simulator.dynamics, "orbit", {}) or {})
        orbit_model = str(orbit_config.get("model", "two_body") or "two_body").strip().lower()
        earth_center = np.zeros(3, dtype=float)
        if orbit_model == "cr3bp":
            system = cr3bp_system(str(orbit_config.get("cr3bp_system", "earth_moon")))
            earth_center[0] = -float(system.mu) * float(system.distance_km)
        for aid, agent in e.agents.items():
            if agent.kind == "rocket" and not include_rockets:
                continue
            policy = _earth_impact_policy_for_object(e.cfg.simulator.termination, aid)
            if not bool(policy.get("earth_impact_enabled", True)):
                continue
            if agent.kind == "rocket" and "touchdown" in policy:
                continue
            re = float(policy.get("earth_radius_km", EARTH_RADIUS_KM))
            if not agent.active:
                continue
            if agent.kind == "rocket" and agent.waiting_for_launch:
                continue
            truth = agent.truth if agent.kind == "satellite" else _rocket_state_to_truth(agent.rocket_state)
            relative_position = np.asarray(truth.position_eci_km, dtype=float) - earth_center
            impact = float(np.linalg.norm(relative_position)) <= re
            crossing_fraction: float | None = None
            if agent.kind == "satellite" and int(getattr(e, "current_index", 0)) > 0:
                hist = getattr(e, "truth_hist", {}).get(aid)
                idx = int(e.current_index)
                if hist is not None and idx < int(hist.shape[0]):
                    r0 = np.asarray(hist[idx - 1, :3], dtype=float) - earth_center
                    r1 = np.asarray(hist[idx, :3], dtype=float) - earth_center
                    # A straight chord between coarse output samples is not a
                    # valid continuous orbit approximation.  Localize only a
                    # bracketed outside-to-inside endpoint crossing.
                    if float(np.linalg.norm(r1)) <= re:
                        crossing_fraction = _segment_sphere_entry_fraction(r0, r1, re)
                    impact = bool(impact or crossing_fraction is not None)
            if agent.kind == "rocket" and agent.rocket_sim is not None:
                impact = bool(_rocket_altitude_km(truth.position_eci_km, truth.t_s, agent.rocket_sim.sim_cfg) <= 0.0)
            if impact:
                e.terminated_early = True
                e.termination_reason = "earth_impact"
                if crossing_fraction is not None and int(e.current_index) > 0:
                    t0 = float(e.t_s[int(e.current_index) - 1])
                    e.termination_time_s = t0 + crossing_fraction * (float(t_s) - t0)
                else:
                    e.termination_time_s = float(t_s)
                e.termination_object_id = aid
                return True
        return False
    def check_reentry(self, *, t_s: float) -> bool:
        e = self.engine
        reentry_cfg = getattr(e, "reentry_cfg", None)
        if reentry_cfg is None or not bool(getattr(reentry_cfg, "enabled", False)):
            return False
        metrics_by_object = getattr(e, "reentry_metric_hists", {}) or {}
        sample_index = int(getattr(e, "current_index", 0))
        for aid, metrics_by_key in dict(metrics_by_object).items():
            agent = e.agents.get(aid)
            if agent is None or not agent.active:
                continue
            sample_metrics: dict[str, float] = {}
            for key, series in dict(metrics_by_key).items():
                arr = np.array(series, dtype=float).reshape(-1)
                if sample_index < arr.size:
                    sample_metrics[str(key)] = float(arr[sample_index])
            reason = evaluate_reentry_termination(sample_metrics, reentry_cfg, object_id=aid)
            if reason is None:
                continue
            previous_metrics: dict[str, float] = {}
            if sample_index > 0:
                for key, series in dict(metrics_by_key).items():
                    arr = np.array(series, dtype=float).reshape(-1)
                    if sample_index - 1 < arr.size:
                        previous_metrics[str(key)] = float(arr[sample_index - 1])
            crossing = locate_reentry_termination_crossing(
                previous_metrics,
                sample_metrics,
                reentry_cfg,
                object_id=aid,
            )
            e.terminated_early = True
            e.termination_reason = reason if crossing is None else crossing[0]
            if crossing is not None and sample_index > 0:
                t0 = float(e.t_s[sample_index - 1])
                e.termination_time_s = t0 + float(crossing[1]) * (float(t_s) - t0)
            else:
                e.termination_time_s = float(t_s)
            e.termination_object_id = aid
            return True
        return False

    def update_rocket_insertion(self, *, t_s: float, dt_s: float | None = None) -> None:
        e = self.engine
        rocket = e.rocket
        if (
            rocket is None
            or not rocket.active
            or rocket.waiting_for_launch
            or rocket.rocket_state is None
            or rocket.rocket_sim is None
        ):
            return
        rs = rocket.rocket_state
        sim_cfg = rocket.rocket_sim.sim_cfg
        alt_km = _rocket_altitude_km(rs.position_eci_km, rs.t_s, sim_cfg)
        near_alt = abs(float(alt_km) - float(sim_cfg.target_altitude_km)) <= float(sim_cfg.target_altitude_tolerance_km)
        _, ecc_now = _orbital_elements_basic(
            np.array(rs.position_eci_km, dtype=float), np.array(rs.velocity_eci_km_s, dtype=float)
        )
        low_e = float(ecc_now) <= float(sim_cfg.target_eccentricity_max)
        stages_done = int(rs.active_stage_index) >= len(rocket.rocket_sim.vehicle_cfg.stack.stages)
        if near_alt and low_e and stages_done:
            applied_dt_s = float(e.dt if dt_s is None else dt_s)
            if not np.isfinite(applied_dt_s) or applied_dt_s <= 0.0:
                raise ValueError("rocket insertion update dt_s must be positive and finite.")
            e.rocket_insertion_hold_s += applied_dt_s
            if (not e.rocket_inserted) and e.rocket_insertion_hold_s >= float(sim_cfg.insertion_hold_time_s):
                e.rocket_inserted = True
                e.rocket_insertion_time_s = float(t_s)
        else:
            e.rocket_insertion_hold_s = 0.0
        has_rocket_inserted_satellite = any(
            bool(getattr(agent_cfg, "enabled", False))
            and str(dict(getattr(agent_cfg, "initial_state", {}) or {}).get("source", "") or "").strip().lower()
            == "rocket_insertion"
            for agent_cfg in getattr(e.cfg, "objects", {}).values()
        )
        if e.rocket_inserted and not has_rocket_inserted_satellite:
            e.terminated_early = True
            e.termination_reason = "rocket_orbit_insertion"
            e.termination_time_s = float(e.rocket_insertion_time_s if e.rocket_insertion_time_s is not None else t_s)
            e.termination_object_id = "rocket"


def _segment_sphere_entry_fraction(r0: np.ndarray, r1: np.ndarray, radius_km: float) -> float | None:
    """Return the first line-segment crossing of a spherical impact surface."""

    start = np.asarray(r0, dtype=float).reshape(3)
    end = np.asarray(r1, dtype=float).reshape(3)
    if not (np.all(np.isfinite(start)) and np.all(np.isfinite(end))):
        return None
    radius = float(radius_km)
    if float(np.linalg.norm(start)) <= radius:
        return 0.0
    delta = end - start
    a = float(np.dot(delta, delta))
    if a <= 0.0:
        return None
    b = 2.0 * float(np.dot(start, delta))
    c = float(np.dot(start, start) - radius * radius)
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return None
    root = float(np.sqrt(max(disc, 0.0)))
    candidates = [
        value
        for value in ((-b - root) / (2.0 * a), (-b + root) / (2.0 * a))
        if 0.0 <= value <= 1.0
    ]
    return None if not candidates else float(min(candidates))
