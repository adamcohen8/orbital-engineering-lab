from __future__ import annotations

import logging
import os
from contextlib import nullcontext
from copy import copy
from dataclasses import replace
from pathlib import Path
from threading import RLock
from time import perf_counter
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from sim.acceleration.settings import acceleration_context_from_config, acceleration_settings_from_config
from sim.aero import resolve_vehicle_aero_properties
from sim.config import (
    SimulationScenarioConfig,
    configured_objects,
    default_reference_object_id,
    relative_reference_for_object,
    scenario_config_from_dict,
)
from sim.core.models import Command, StateBelief, StateTruth
from sim.dynamics.attitude.rigid_body import (
    activate_attitude_guardrail_stats,
    new_attitude_guardrail_stats,
)
from sim.dynamics.orbit.atmosphere import altitude_km_from_eci
from sim.dynamics.orbit.epoch import TIME_DEPENDENT_ENV_CACHE_KEY
from sim.dynamics.orbit.frames import FrameContext, frame_context_from_mapping
from sim.dynamics.orbit.spherical_harmonics import configure_spherical_harmonics_env
from sim.dynamics.orbit.tle import tle_block_initialization_metadata
from sim.dynamics.reentry import (
    REENTRY_METRIC_KEYS,
    ReentryObjectProperties,
    reentry_config_from_dynamics,
    reentry_metrics_for_state,
)
from sim.execution.collisions import SphericalCollisionStepper
from sim.execution.object_step_coordinator import ObjectStepCoordinator
from sim.execution.object_workers import (
    ObjectKnowledgeSyncResult,
    ObjectStepBackendUnavailable,
    ObjectStepExecutor,
    ObjectStepInput,
    ObjectStepResult,
    SerialObjectStepExecutor,
)
from sim.execution.runtime_profile import _RuntimeProfiler
from sim.execution.single_run_history import SingleRunHistoryStore
from sim.execution.system_forces import SystemForceStepper
from sim.numeric_backend import normalize_numeric_backend
from sim.reporting.run_payload_assembly import SingleRunPayloadAssembler, _SingleRunPayloadParts
from sim.resource_limits import (
    HistoryMemoryEstimate,
    enforce_history_memory_budget,
    estimate_history_memory_from_config,
)
from sim.rocket.navigation import build_rocket_nav_state
from sim.runtime_support import (
    AgentRuntime,
    _apply_relative_cislunar_init_from_reference,
    _apply_relative_init_from_reference,
    _build_knowledge_base,
    _create_rocket_runtime,
    _create_satellite_runtime,
    _deploy_from_rocket,
    _rocket_state_to_truth,
    _run_mission_execution,
    _run_mission_modules,
    _run_mission_strategy,
    _scenario_uses_aerodynamic_lift,
)
from sim.single_run_support import (
    _BudgetedControllerProxy,
    _DecisionContext,
    _DecisionContextBuilder,
    _KnowledgeSynchronizer,
    _RocketStepper,
    _SatelliteStepper,
    _TerminationMonitor,
)

if TYPE_CHECKING:
    from sim.dynamics.orbit.sgp4 import SGP4EphemerisProvider

_AUTO_OBJECT_PARALLEL_RUN_WORK_THRESHOLD = 250.0

logger = logging.getLogger(__name__)

OBJECT_WORKER_BUDGET_ENV = "OEL_OBJECT_WORKER_BUDGET"
CAMPAIGN_WORKER_COUNT_ENV = "OEL_CAMPAIGN_WORKER_COUNT"
TOTAL_PROCESS_BUDGET_ENV = "OEL_TOTAL_PROCESS_BUDGET"


def _effective_propagation_method(cfg: SimulationScenarioConfig, agent_cfg: Any) -> str:
    orbit = dict((cfg.simulator.dynamics or {}).get("orbit", {}) or {})
    default_method = str(orbit.get("propagation_method", "special") or "special").strip().lower()
    return str(getattr(agent_cfg, "propagation_method", "") or default_method or "special").strip().lower()


def _object_initialization_metadata(
    cfg: SimulationScenarioConfig, object_configs: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    metadata: dict[str, dict[str, Any]] = {}
    for object_id, agent_cfg in sorted(dict(object_configs or {}).items()):
        if _effective_propagation_method(cfg, agent_cfg) == "general":
            continue
        initial_state = dict(getattr(agent_cfg, "initial_state", {}) or {})
        tle = initial_state.get("tle")
        if not isinstance(tle, dict):
            continue
        metadata[str(object_id)] = tle_block_initialization_metadata(
            tle,
            target_jd_utc=getattr(cfg.simulator, "initial_jd_utc", None),
            duration_s=float(getattr(cfg.simulator, "duration_s", 0.0) or 0.0),
        )
    return metadata


def _apply_frame_context_to_environment(
    env: dict[str, Any],
    *,
    frame_context: FrameContext,
    orbit_cfg: dict[str, Any],
) -> dict[str, Any]:
    out = dict(env or {})
    out["frame_context"] = frame_context
    out["frame_model"] = frame_context.legacy_frame_model
    out["frame_model_canonical"] = frame_context.model
    out["frame_provenance"] = frame_context.metadata()
    out["time_scale_model"] = frame_context.time_scale_model
    out["eop_extrapolation"] = frame_context.eop_extrapolation
    out["tt_minus_utc_s"] = float(frame_context.tt_minus_utc_s)
    if frame_context.eop_path is not None:
        out["eop_path"] = frame_context.eop_path
    if frame_context.dut1_s is not None:
        out["dut1_s"] = float(frame_context.dut1_s)
    if frame_context.xp_arcsec is not None:
        out["xp_arcsec"] = float(frame_context.xp_arcsec)
    if frame_context.yp_arcsec is not None:
        out["yp_arcsec"] = float(frame_context.yp_arcsec)
    if frame_context.dat_s is not None:
        out["dat_s"] = float(frame_context.dat_s)
    out["ddpsi_rad"] = float(frame_context.ddpsi_rad)
    out["ddeps_rad"] = float(frame_context.ddeps_rad)

    legacy_model = frame_context.legacy_frame_model
    if "drag_frame_model" not in out and orbit_cfg.get("drag_frame_model") is None:
        out["drag_frame_model"] = legacy_model
    if "density_frame_model" not in out:
        out["density_frame_model"] = str(out.get("drag_frame_model", legacy_model))
    if frame_context.eop_path is not None:
        if "drag_eop_path" not in out and orbit_cfg.get("drag_eop_path") is None:
            out["drag_eop_path"] = frame_context.eop_path
        if "density_eop_path" not in out:
            out["density_eop_path"] = str(out.get("drag_eop_path", frame_context.eop_path))
        if "spherical_harmonics_eop_path" not in out:
            out["spherical_harmonics_eop_path"] = frame_context.eop_path
    if "spherical_harmonics_frame_model" not in out:
        out["spherical_harmonics_frame_model"] = legacy_model
    return out


def _write_state_truth(row: np.ndarray, truth: StateTruth) -> None:
    """Write one canonical truth row without allocating a temporary array."""
    row[0:3] = truth.position_eci_km
    row[3:6] = truth.velocity_eci_km_s
    row[6:10] = truth.attitude_quat_bn
    row[10:13] = truth.angular_rate_body_rad_s
    row[13] = truth.mass_kg


def _onboard_belief_vector(agent: AgentRuntime) -> np.ndarray | None:
    runtime = agent.flight_software_runtime
    return None if runtime is None else runtime.onboard_state_vector()


class _SingleRunEngine:
    def __init__(
        self,
        cfg: SimulationScenarioConfig,
        *,
        step_callback: Callable[[int, int], None] | None = None,
        history_mode: str = "full",
        initial_history_capacity: int = 4096,
        max_history_samples: int = 4096,
    ) -> None:
        # The engine is the owner of live simulation state.  A threaded
        # object-step backend may be running while a session-facing caller
        # publishes an input or requests a snapshot, so keep those operations
        # on one re-entrant owner boundary.  Worker snapshots clear this
        # transient lock before transport/pickle; they never own live state.
        self._owner_state_lock = RLock()
        self.cfg = cfg
        self.active_step_callback = step_callback
        self.history_mode = str(history_mode or "full").strip().lower()
        if self.history_mode not in {"full", "dynamic"}:
            raise ValueError("history_mode must be 'full' or 'dynamic'.")
        self._object_worker_compact_buffers = False
        self.dt = float(cfg.simulator.dt_s)
        controller_execution = dict(dict(getattr(cfg.simulator, "execution", {}) or {}).get("controller", {}) or {})
        self.orbit_controller_budget_ms = float(controller_execution.get("orbit_budget_ms", 2.0) or 2.0)
        self.attitude_controller_budget_ms = float(controller_execution.get("attitude_budget_ms", 2.0) or 2.0)
        self.controller_deadline_policy = str(
            controller_execution.get("deadline_policy", "record") or "record"
        ).strip().lower()
        self.planned_samples = int(np.floor(float(cfg.simulator.duration_s) / self.dt)) + 1
        self.sample_offset = 0
        self.max_history_samples = int(max(2, max_history_samples))
        if self.history_mode == "dynamic":
            self.n = int(max(2, min(self.planned_samples, self.max_history_samples, int(initial_history_capacity))))
        else:
            self.n = self.planned_samples
        self.outdir = Path(cfg.outputs.output_dir)

        seed = int(cfg.metadata.get("seed", 123))
        rng = np.random.default_rng(seed)
        dynamics_cfg = dict(cfg.simulator.dynamics or {})
        orbit_cfg = dict(dynamics_cfg.get("orbit", {}) or {})
        att_cfg = dict(dynamics_cfg.get("attitude", {}) or {})
        self.attitude_guardrail_stats = new_attitude_guardrail_stats(
            policy=str(att_cfg.get("guardrail_policy", "error") or "error")
        )
        activate_attitude_guardrail_stats(self.attitude_guardrail_stats)
        self.planetary_environment = None
        if cfg.simulator.central_body != "earth":
            from sim.dynamics.planetary.preparation import prepare_environment
            self.planetary_environment = prepare_environment(cfg)
            self.planetary_environment.verify_resources()
            self.frame_context = None
            self.base_environment = {}
        else:
            self.frame_context = frame_context_from_mapping(
                dict(getattr(cfg.simulator, "frames", {}) or {}),
                jd_utc_start=cfg.simulator.initial_jd_utc,
            )
            self.base_environment = configure_spherical_harmonics_env(dict(cfg.simulator.environment or {}), orbit_cfg)
            atmosphere_env = self.base_environment.pop("atmosphere_env", None)
            if isinstance(atmosphere_env, dict):
                self.base_environment = {**dict(atmosphere_env), **self.base_environment}
            if orbit_cfg.get("atmosphere_model") not in (None, "") and "atmosphere_model" not in self.base_environment:
                self.base_environment["atmosphere_model"] = str(orbit_cfg.get("atmosphere_model")).strip().lower()
            if cfg.simulator.initial_jd_utc is not None and "jd_utc_start" not in self.base_environment:
                self.base_environment["jd_utc_start"] = float(cfg.simulator.initial_jd_utc)
            self.base_environment = _apply_frame_context_to_environment(
                self.base_environment,
                frame_context=self.frame_context,
                orbit_cfg=orbit_cfg,
            )
        self._time_dependent_env_cache: dict[tuple, dict[str, np.ndarray]] = {}
        self.reentry_cfg = reentry_config_from_dynamics(dynamics_cfg)
        self.attitude_enabled = bool(att_cfg.get("enabled", True))
        orbit_substep_s = float(max(float(orbit_cfg.get("orbit_substep_s", self.dt) or self.dt), 1e-9))
        attitude_substep_s = float(max(float(att_cfg.get("attitude_substep_s", self.dt) or self.dt), 1e-9))
        self.orbit_command_period_s = orbit_substep_s
        self.sim_substep_s = (
            float(min(orbit_substep_s, attitude_substep_s)) if self.attitude_enabled else orbit_substep_s
        )
        self.eye6 = np.eye(6) * 1e-4
        self.eye12 = np.eye(12) * 1e-4
        self.zero3 = np.zeros(3, dtype=float)
        self.decision_contexts = _DecisionContextBuilder(
            base_environment=self.base_environment,
            attitude_enabled=self.attitude_enabled,
            orbit_command_period_s=self.orbit_command_period_s,
        )
        self.rocket_stepper = _RocketStepper(self)
        self.satellite_stepper = _SatelliteStepper(self)
        if self.planetary_environment is None:
            self.termination_monitor = _TerminationMonitor(self)
        else:
            from sim.dynamics.planetary.runtime import PlanetaryTerminationMonitor
            self.termination_monitor = PlanetaryTerminationMonitor(self)

        self.agents: dict[str, AgentRuntime] = {}
        self.object_configs = configured_objects(cfg)
        scenario_uses_aerodynamic_lift = _scenario_uses_aerodynamic_lift(cfg)
        for aid, agent_cfg in self.object_configs.items():
            if not bool(agent_cfg.enabled):
                continue
            if str(agent_cfg.kind).strip().lower() == "rocket":
                self.agents[aid] = _create_rocket_runtime(cfg, object_id=aid, agent_cfg=agent_cfg)
            else:
                self.agents[aid] = _create_satellite_runtime(
                    aid,
                    agent_cfg,
                    cfg,
                    np.random.default_rng(int(rng.integers(0, 2**31 - 1))),
                    scenario_uses_aerodynamic_lift=scenario_uses_aerodynamic_lift,
                    planetary_environment=self.planetary_environment,
                )
        for agent in self.agents.values():
            if agent.kind == "satellite" and agent.deploy_source in {"rocket_deployment", "rocket_insertion"}:
                agent.active = False
        self._forecast_dynamics_by_object = {
            object_id: runtime.dynamics
            for object_id, runtime in self.agents.items()
            if runtime.kind == "satellite" and runtime.dynamics is not None
        }
        self._forecast_truth_cache: dict[tuple[float, float, str, int, int], StateTruth] = {}

        self.general_propagation: dict[str, SGP4EphemerisProvider] = {}
        for aid, agent in self.agents.items():
            agent_cfg = self.object_configs.get(aid)
            if agent.kind != "satellite" or agent_cfg is None:
                continue
            if _effective_propagation_method(cfg, agent_cfg) != "general":
                continue
            initial_state = dict(getattr(agent_cfg, "initial_state", {}) or {})
            general = dict(getattr(agent_cfg, "general", {}) or {})
            if str(general.get("model", "") or "").strip().lower() != "sgp4":
                continue
            if agent.truth is None:
                continue
            from sim.dynamics.orbit.sgp4 import SGP4EphemerisProvider

            provider_kwargs = dict(
                mass_kg=float(agent.truth.mass_kg),
                start_jd_utc=cfg.simulator.initial_jd_utc,
                duration_s=float(cfg.simulator.duration_s),
                output_frame=str(general.get("output_frame", "teme") or "teme"),
                frame_transform=general.get("frame_transform"),
                attitude_quat_bn=agent.truth.attitude_quat_bn,
                angular_rate_body_rad_s=agent.truth.angular_rate_body_rad_s,
                max_tle_age_days_warning=(
                    None
                    if general.get("max_tle_age_days_warning") is None
                    else float(general.get("max_tle_age_days_warning"))
                ),
                acceleration_mode=acceleration_settings_from_config(cfg).requested_mode,
                numeric_backend=normalize_numeric_backend(
                    general.get("numeric_backend", "rust"),
                    field_name=f"objects.{aid}.general.numeric_backend",
                ),
            )
            if isinstance(initial_state.get("ogp_mean_elements"), dict):
                provider = SGP4EphemerisProvider.from_mean_elements(
                    dict(initial_state["ogp_mean_elements"]), **provider_kwargs
                )
            else:
                provider = SGP4EphemerisProvider.from_tle_block(
                    dict(initial_state.get("tle", {}) or {}), **provider_kwargs
                )
            self.general_propagation[aid] = provider
            agent.truth = provider.canonical_state_at(0.0)
            if agent.belief is not None and agent.belief.state.size >= 6:
                agent.belief.state[:6] = np.hstack((agent.truth.position_eci_km, agent.truth.velocity_eci_km_s))
                agent.belief.last_update_t_s = 0.0

        self.rocket = self.agents.get("rocket") or next((a for a in self.agents.values() if a.kind == "rocket"), None)
        self.chaser = self.agents.get("chaser")
        self.target = self.agents.get("target")
        execution_cfg = dict(getattr(cfg.simulator, "execution", {}) or {})
        runtime_profiler_cfg = dict(execution_cfg.get("runtime_profiler", {}) or {})
        self.runtime_profiler = _RuntimeProfiler(
            object_ids=list(self.agents.keys()),
            enabled=bool(runtime_profiler_cfg.get("enabled", True)),
        )
        self.controller_debug_enabled = bool(getattr(cfg.outputs.stats, "controller_debug", True))
        self.object_step_executor: ObjectStepExecutor = SerialObjectStepExecutor(self)
        self.reentry_object_ids = self._resolve_reentry_object_ids()
        self.reentry_active_by_object = {aid: False for aid in self.reentry_object_ids}

        relative_forms = {
            "relative_to_target_ric", "relative_ric_rect", "relative_ric_curv",
            "relative_to_target_cislunar", "relative_cislunar",
        }
        pending_relative: dict[str, tuple[Any, dict[str, Any], str]] = {}
        for aid, agent in self.agents.items():
            if agent.kind != "satellite" or agent.deploy_source in {"rocket_deployment", "rocket_insertion"}:
                continue
            if aid in self.general_propagation:
                continue
            agent_cfg = self.object_configs.get(aid)
            initial_state = dict(getattr(agent_cfg, "initial_state", {}) or {})
            if not relative_forms.intersection(initial_state):
                continue
            reference_id = str(relative_reference_for_object(cfg, aid) or "").strip()
            pending_relative[aid] = (agent, initial_state, reference_id)
        while pending_relative:
            ready = sorted(
                aid for aid, (_agent, _state, reference_id) in pending_relative.items()
                if reference_id not in pending_relative
            )
            if not ready:
                raise ValueError("Relative initial-state references contain an unresolved cycle.")
            for aid in ready:
                agent, initial_state, reference_id = pending_relative.pop(aid)
                reference = self.agents.get(reference_id)
                if reference is None:
                    raise ValueError(f"Relative initial-state reference {reference_id!r} is unavailable.")
                _apply_relative_init_from_reference(agent=agent, reference=reference, initial_state=initial_state)
                _apply_relative_cislunar_init_from_reference(
                    agent=agent,
                    reference=reference,
                    initial_state=initial_state,
                )

        target_reference_id = default_reference_object_id(cfg, available_ids=self.agents.keys()) or "target"
        target_reference_section = self.object_configs.get(target_reference_id)
        target_reference_cfg = dict(
            (target_reference_section.reference_orbit if target_reference_section is not None else {}) or {}
        )
        target_reference_agent = self.agents.get(target_reference_id)
        self.target_reference_truth = None
        self.target_reference_dynamics = None
        self.target_reference_orbit_hist = None
        if (
            bool(target_reference_cfg.get("enabled", False))
            and target_reference_agent is not None
            and target_reference_agent.truth is not None
        ):
            self.target_reference_truth = target_reference_agent.truth.copy()
            self.target_reference_dynamics = replace(
                target_reference_agent.dynamics,
                disturbance_model=None,
                propagate_attitude=False,
                use_rectangular_prism_for_aero_srp=False,
                rectangular_prism_dims_m=None,
            )

        for aid, agent in self.agents.items():
            cfg_src = self.object_configs[aid]
            if agent.runtime_profile == "trajectory_only":
                agent.knowledge_base = None
            else:
                agent.knowledge_base = _build_knowledge_base(
                    observer_id=aid,
                    agent_cfg=cfg_src,
                    dt_s=self.dt,
                    rng=np.random.default_rng(int(rng.integers(0, 2**31 - 1))),
                )
        initial_world_truth = {
            aid: (
                agent.truth.copy()
                if agent.kind == "satellite"
                else _rocket_state_to_truth(agent.rocket_state).copy()
            )
            for aid, agent in self.agents.items()
            if agent.active
        }
        for _aid, agent in self.agents.items():
            if agent.kind == "satellite" and agent.active and agent.flight_software_runtime is not None:
                agent.flight_software_runtime.prepare_interval(
                    agent.truth,
                    start_time_ns=0,
                    world_truth=initial_world_truth,
                )

        self.history_memory_estimate = self._estimate_history_memory()
        enforce_history_memory_budget(self.history_memory_estimate)

        self.t_s = np.arange(self.n, dtype=float)
        self.t_s *= self.dt
        self.outdir.mkdir(parents=True, exist_ok=True)

        if self.target_reference_truth is not None and self.target_reference_orbit_hist is None:
            self.target_reference_orbit_hist = np.full((self.n, 6), np.nan)
            self.target_reference_orbit_hist[0, 0:3] = self.target_reference_truth.position_eci_km
            self.target_reference_orbit_hist[0, 3:6] = self.target_reference_truth.velocity_eci_km_s

        self.truth_hist = {aid: np.full((self.n, 14), np.nan) for aid in self.agents.keys()}
        self.belief_hist = {
            aid: np.full(
                (
                    self.n,
                    int(agent.belief.state.size)
                    if agent.belief is not None
                    else int(_onboard_belief_vector(agent).size)
                    if _onboard_belief_vector(agent) is not None
                    else 0,
                ),
                np.nan,
            )
            for aid, agent in self.agents.items()
        }
        self.thrust_hist = {aid: np.full((self.n, 3), np.nan) for aid in self.agents.keys()}
        self.torque_hist = {aid: np.full((self.n, 3), np.nan) for aid in self.agents.keys()}
        self.desired_attitude_hist = {aid: np.full((self.n, 4), np.nan) for aid in self.agents.keys()}
        self.controller_debug_hist: dict[str, list[dict[str, Any]]] = {aid: [] for aid in self.agents.keys()}
        self.command_decision_hist: dict[str, list[dict[str, Any]]] = {aid: [] for aid in self.agents.keys()}
        self._last_orbital_command_eval_t_s: dict[str, float | None] = {aid: None for aid in self.agents.keys()}
        self._latched_orbital_thrust_cmd_by_object: dict[str, np.ndarray] = {
            aid: self.zero3.copy() for aid in self.agents.keys()
        }
        self.throttle_hist = {
            aid: np.full(self.n, np.nan) for aid, agent in self.agents.items() if agent.kind == "rocket"
        }
        self.rocket_stage_hist = np.full(self.n, np.nan) if self.rocket is not None else None
        self.rocket_q_dyn_hist = np.full(self.n, np.nan) if self.rocket is not None else None
        self.rocket_mach_hist = np.full(self.n, np.nan) if self.rocket is not None else None
        self.rocket_metric_hist_keys = [
            "altitude_km",
            "speed_km_s",
            "vertical_speed_km_s",
            "horizontal_speed_km_s",
            "flight_path_angle_deg",
            "apoapsis_alt_km",
            "periapsis_alt_km",
            "eccentricity",
            "alpha_deg",
            "beta_deg",
            "tvc_gimbal_deg",
            "aero_force_n",
            "aero_moment_nm",
            "thrust_to_weight",
            "propellant_remaining_kg",
            "propellant_remaining_fraction",
            "guidance_phase_code",
        ]
        self.rocket_metric_hists = (
            {key: np.full(self.n, np.nan) for key in self.rocket_metric_hist_keys}
            if self.rocket is not None
            else {}
        )
        self.reentry_metric_hists = {
            aid: {key: np.full(self.n, np.nan) for key in REENTRY_METRIC_KEYS}
            for aid in self.reentry_object_ids
        }
        self.knowledge_hist: dict[str, dict[str, np.ndarray]] = {}
        self.knowledge_measurement_hist: dict[str, dict[str, np.ndarray]] = {}
        self.bridge_hist: dict[str, list[dict[str, Any]]] = {aid: [] for aid in self.agents.keys()}
        for aid, agent in self.agents.items():
            if agent.knowledge_base is not None:
                self.knowledge_hist[aid] = {}
                self.knowledge_measurement_hist[aid] = {}
                for tid in agent.knowledge_base.target_ids():
                    self.knowledge_hist[aid][tid] = np.full((self.n, 6), np.nan)
                    self.knowledge_measurement_hist[aid][tid] = np.full((self.n, 6), np.nan)
        self.knowledge_sync = _KnowledgeSynchronizer(self)
        self.history_store = SingleRunHistoryStore(self)
        self.knowledge_sync.initialize()
        self.worker_knowledge_detection_by_observer: dict[str, Any] | None = None
        self.worker_knowledge_consistency_by_observer: dict[str, Any] | None = None

        self.terminated_early = False
        self.termination_reason: str | None = None
        self.termination_time_s: float | None = None
        self.termination_object_id: str | None = None
        self.rocket_inserted = False
        self.rocket_insertion_time_s: float | None = None
        self.rocket_insertion_hold_s = 0.0
        self.total_dv_m_s_by_object = {aid: 0.0 for aid in self.agents.keys()}
        self.impulsive_maneuvers: list[dict[str, Any]] = []
        self.collision_events: list[dict[str, Any]] = []
        self.burn_samples_by_object = {aid: 0 for aid in self.agents.keys()}
        self.max_accel_km_s2_by_object = {aid: 0.0 for aid in self.agents.keys()}
        self.current_index = 0
        self.resource_hist = {}

        for aid, agent in self.agents.items():
            if not agent.active:
                continue
            truth = agent.truth if agent.kind == "satellite" else _rocket_state_to_truth(agent.rocket_state)
            _write_state_truth(self.truth_hist[aid][0, :], truth)
            if truth.resource_state is not None:
                self.resource_hist[aid] = [{**truth.resource_state, "time_s": float(truth.t_s)}]
            if agent.belief is not None:
                self._ensure_belief_hist_width(aid, agent.belief.state.size)
                self.belief_hist[aid][0, : agent.belief.state.size] = agent.belief.state
            else:
                onboard = _onboard_belief_vector(agent)
                if onboard is not None:
                    self._ensure_belief_hist_width(aid, onboard.size)
                    self.belief_hist[aid][0, : onboard.size] = onboard
            if agent.kind == "rocket" and agent.rocket_state is not None and self.rocket_stage_hist is not None:
                self.rocket_stage_hist[0] = float(agent.rocket_state.active_stage_index)
                if self.rocket_q_dyn_hist is not None:
                    self.rocket_q_dyn_hist[0] = float(getattr(agent.rocket_state, "_last_step_q_dyn_pa", 0.0))
                if self.rocket_mach_hist is not None:
                    self.rocket_mach_hist[0] = float(getattr(agent.rocket_state, "_last_step_mach", 0.0))
                if agent.rocket_sim is not None:
                    nav = build_rocket_nav_state(
                        agent.rocket_state,
                        agent.rocket_sim.sim_cfg,
                        agent.rocket_sim.vehicle_cfg,
                    )
                    initial_metrics = {
                        "altitude_km": nav.altitude_km,
                        "speed_km_s": nav.speed_km_s,
                        "vertical_speed_km_s": nav.vertical_speed_km_s,
                        "horizontal_speed_km_s": nav.horizontal_speed_km_s,
                        "flight_path_angle_deg": nav.flight_path_angle_deg,
                        "apoapsis_alt_km": nav.apoapsis_alt_km,
                        "periapsis_alt_km": nav.periapsis_alt_km,
                        "eccentricity": nav.eccentricity,
                        "alpha_deg": nav.alpha_deg,
                        "beta_deg": nav.beta_deg,
                        "tvc_gimbal_deg": 0.0,
                        "aero_force_n": 0.0,
                        "aero_moment_nm": 0.0,
                        "thrust_to_weight": nav.thrust_to_weight,
                        "propellant_remaining_kg": nav.propellant_remaining_kg,
                        "propellant_remaining_fraction": nav.propellant_remaining_fraction,
                    }
                    for metric_key, metric_value in initial_metrics.items():
                        if metric_key in self.rocket_metric_hists:
                            self.rocket_metric_hists[metric_key][0] = float(metric_value)
            self._record_reentry_metrics(aid=aid, truth=truth, sample_index=0, dt_s=0.0)

        # Satellites may arrive invalidly inside Earth and must fail at the
        # initial sample. Launch vehicles legitimately begin on the pad at the
        # surface, so their impact policy starts after the first propagation.
        if not self.termination_monitor.check_earth_impact(
            t_s=float(self.t_s[0]), include_rockets=False
        ):
            self.termination_monitor.check_reentry(t_s=float(self.t_s[0]))
        self.ground_segment = None
        if cfg.ground_segment.get("enabled", False):
            from sim.ground_segment import SimulatedGroundSegment

            self.ground_segment = SimulatedGroundSegment(
                cfg.ground_segment,
                stations=cfg.ground_stations,
                object_ids=list(self.agents),
                jd_utc_start=cfg.simulator.initial_jd_utc,
                frame_context=self.frame_context,
            )
            self._advance_ground_segment(0)
        self._emit_step_callback(0)
        self.object_step_executor = self._build_object_step_executor()
        self.system_force_stepper = (
            SystemForceStepper(
                pointers=list(cfg.simulator.system_force_models),
                initial_jd_utc=float(cfg.simulator.initial_jd_utc),
                substep_s=self.sim_substep_s,
                numeric_backend=normalize_numeric_backend(
                    dict(cfg.simulator.dynamics.get("orbit", {}) or {}).get("numeric_backend", "rust"),
                    field_name="simulator.dynamics.orbit.numeric_backend",
                ),
            )
            if cfg.simulator.system_force_models else None
        )
        self.collision_stepper = (
            SphericalCollisionStepper(
                self,
                dict(cfg.simulator.collisions["radii_m"]),
                numeric_backend=normalize_numeric_backend(
                    cfg.simulator.collisions.get("numeric_backend", "rust"),
                    field_name="simulator.collisions.numeric_backend",
                ),
            )
            if cfg.simulator.collisions.get("enabled", False) else None
        )

    def _build_object_step_executor(self) -> ObjectStepExecutor:
        return ObjectStepCoordinator(self, cpu_count=os.cpu_count).build_executor()

    def _estimated_object_parallel_work_score(self, active_ids: list[str]) -> float:
        return ObjectStepCoordinator(self, cpu_count=os.cpu_count).estimated_work_score(active_ids)

    @staticmethod
    def _object_worker_allocation(object_ids: list[str], workers: int) -> dict[str, list[str]]:
        return ObjectStepCoordinator.object_worker_allocation(object_ids, workers)

    def _resolve_reentry_object_ids(self) -> list[str]:
        if not bool(self.reentry_cfg.enabled):
            return []
        configured = [str(item).strip() for item in self.reentry_cfg.object_ids if str(item).strip()]
        if configured and not any(item in {"*", "all"} for item in configured):
            return [aid for aid in configured if aid in self.agents]
        return [aid for aid, agent in self.agents.items() if agent.kind == "satellite"]

    def _reentry_properties_for_agent(self, aid: str, agent: AgentRuntime, truth: StateTruth) -> ReentryObjectProperties:
        agent_cfg = self.object_configs.get(aid)
        specs = dict(getattr(agent_cfg, "specs", {}) or {}) if agent_cfg is not None else {}
        dynamics = getattr(agent, "dynamics", None)
        aero = resolve_vehicle_aero_properties(
            specs,
            default_reference_area_m2=1.0,
            default_cd=2.2,
            default_cl=0.0,
            default_nose_radius_m=self.reentry_cfg.default_nose_radius_m,
        )
        area_m2 = float(getattr(dynamics, "area_m2", aero.reference_area_m2) or aero.reference_area_m2)
        drag_area_m2 = getattr(dynamics, "drag_area_m2", None)
        if drag_area_m2 is None:
            drag_area_m2 = aero.drag_area_m2 if aero.drag_area_m2 is not None else area_m2
        cd = float(getattr(dynamics, "cd", aero.cd) or aero.cd)
        lift_area_m2 = getattr(dynamics, "lift_area_m2", None)
        if lift_area_m2 is None:
            lift_area_m2 = aero.lift_area_m2 if aero.lift_area_m2 is not None else drag_area_m2
        cl = float(getattr(dynamics, "lift_coefficient", aero.cl) or 0.0)
        return ReentryObjectProperties(
            mass_kg=float(max(float(truth.mass_kg), 1e-12)),
            drag_area_m2=float(max(float(drag_area_m2), 0.0)),
            cd=cd,
            nose_radius_m=float(aero.nose_radius_m),
            lift_area_m2=None if lift_area_m2 is None else float(max(float(lift_area_m2), 0.0)),
            cl=cl,
        )

    def _record_reentry_metrics(self, *, aid: str, truth: StateTruth, sample_index: int, dt_s: float) -> None:
        if aid not in self.reentry_metric_hists:
            return
        altitude_km = altitude_km_from_eci(truth.position_eci_km, truth.t_s, env=self.base_environment)
        active = bool(altitude_km <= self.reentry_cfg.begin_altitude_km)
        self.reentry_active_by_object[aid] = bool(self.reentry_active_by_object.get(aid, False) or active)
        prev_heat = 0.0
        prev_heat_rate: float | None = None
        if sample_index > 0:
            prev_heat = float(self.reentry_metric_hists[aid]["heat_load_j_m2"][sample_index - 1])
            prev_heat_rate = float(self.reentry_metric_hists[aid]["heat_rate_w_m2"][sample_index - 1])
        metrics = reentry_metrics_for_state(
            r_eci_km=truth.position_eci_km,
            v_eci_km_s=truth.velocity_eci_km_s,
            t_s=truth.t_s,
            dt_s=dt_s,
            cfg=self.reentry_cfg,
            props=self._reentry_properties_for_agent(aid, self.agents[aid], truth),
            env=self.base_environment,
            active=active,
            previous_heat_load_j_m2=prev_heat,
            previous_heat_rate_w_m2=prev_heat_rate,
        )
        for key, value in metrics.items():
            if key in self.reentry_metric_hists[aid]:
                self.reentry_metric_hists[aid][key][sample_index] = float(value)

    def _estimate_history_memory(self) -> HistoryMemoryEstimate:
        return estimate_history_memory_from_config(self.cfg, samples=int(max(self.n, 0)))

    @property
    def total_steps(self) -> int:
        return max(self.planned_samples - 1, 0)

    @property
    def retained_start_step(self) -> int:
        return int(self.sample_offset)

    @property
    def retained_end_step(self) -> int:
        return int(self.sample_offset + int(self.current_index))

    @property
    def retained_sample_count(self) -> int:
        return int(self.current_index + 1)

    @property
    def allocated_history_samples(self) -> int:
        return int(self.n)

    @property
    def done(self) -> bool:
        current_time_s = 0.0
        if hasattr(self, "t_s") and self.t_s.size:
            current_time_s = float(self.t_s[min(int(self.current_index), self.t_s.size - 1)])
        return bool(
            self.terminated_early
            or current_time_s >= float(self.cfg.simulator.duration_s) - max(1.0e-9, 1.0e-9 * abs(float(self.cfg.simulator.duration_s)))
        )

    def _emit_step_callback(self, step: int) -> None:
        if self.active_step_callback is None:
            return
        try:
            self.active_step_callback(int(self.sample_offset + int(step)), self.total_steps)
        except (TypeError, ValueError) as exc:
            logger.warning("Disabling step callback after runtime error: %s", exc)
            self.active_step_callback = None

    def _owner_state_context(self):
        """Return the live-owner lock, or a no-op context for worker copies."""

        lock = getattr(self, "_owner_state_lock", None)
        return nullcontext(None) if lock is None else lock

    def _ensure_belief_hist_width(self, aid: str, width: int) -> None:
        self.history_store.ensure_belief_width(aid, width)

    def _grow_axis0(self, arr: np.ndarray | None, rows: int, *, fill: float = np.nan) -> np.ndarray | None:
        return self.history_store.grow_axis0(arr, rows, fill=fill)

    def _compact_axis0_latest(
        self,
        arr: np.ndarray | None,
        *,
        start: int,
        count: int,
        fill: float = np.nan,
    ) -> np.ndarray | None:
        return self.history_store.compact_axis0_latest(arr, start=start, count=count, fill=fill)

    def _compact_event_history_latest(self, rows: list[dict[str, Any]], *, retained_start_time_s: float) -> list[dict[str, Any]]:
        return self.history_store.compact_event_history_latest(
            rows,
            retained_start_time_s=retained_start_time_s,
        )

    def _compact_dynamic_history_if_needed(self, *, keep_latest: int | None = None) -> None:
        self.history_store.compact_if_needed(keep_latest=keep_latest)

    def _ensure_sample_capacity(self, sample_index: int) -> None:
        self.history_store.ensure_sample_capacity(sample_index)

    def _advance_ground_segment(self, index: int) -> None:
        if self.ground_segment is not None:
            from sim.spacecraft_resources.artifacts import resource_snapshot

            t = float(self.t_s[index])
            self.ground_segment.step(
                t,
                truth={oid: hist[index] for oid, hist in self.truth_hist.items()},
                resources=resource_snapshot(self.resource_hist, t),
            )

    def snapshot(
        self,
        step_index: int | None = None,
        *,
        include_flight_software: bool = True,
    ) -> dict[str, Any]:
        # Snapshot reads are part of the same owner boundary as live writes;
        # callbacks may call back into this method because the owner lock is
        # re-entrant.
        with self._owner_state_context():
            return self._snapshot_owned(
                step_index,
                include_flight_software=include_flight_software,
            )

    def _snapshot_owned(
        self,
        step_index: int | None = None,
        *,
        include_flight_software: bool = True,
    ) -> dict[str, Any]:
        if step_index is None:
            idx = self.current_index
        elif self.history_mode == "dynamic":
            idx = int(step_index) - int(self.sample_offset)
        else:
            idx = int(step_index)
        if idx < 0 or idx > int(self.current_index) or idx >= self.n:
            raise IndexError(
                f"step_index {step_index} is outside retained steps "
                f"{self.retained_start_step}..{self.retained_end_step}."
            )
        truth = {oid: np.array(hist[idx], dtype=float) for oid, hist in self.truth_hist.items()}
        if self.target_reference_orbit_hist is not None:
            ref_state = np.array(self.target_reference_orbit_hist[idx], dtype=float).reshape(-1)
            if ref_state.size >= 6 and np.all(np.isfinite(ref_state[:6])):
                target_mass_kg = 0.0
                target_truth = truth.get("target")
                if target_truth is not None and np.array(target_truth).reshape(-1).size >= 14:
                    target_mass_kg = float(np.array(target_truth, dtype=float).reshape(-1)[13])
                truth["target_reference"] = np.hstack(
                    (
                        ref_state[:6],
                        np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, target_mass_kg]),
                    )
                )
        from sim.spacecraft_resources.artifacts import resource_snapshot

        flight_software = {}
        if include_flight_software:
            for object_id, agent in self.agents.items():
                runtime = agent.flight_software_runtime
                if runtime is not None and (checkpoint := runtime.checkpoint()) is not None:
                    flight_software[object_id] = checkpoint
        return {
            "step_index": int(self.sample_offset + idx),
            "time_s": float(self.t_s[idx]),
            "truth": truth,
            "belief": {oid: np.array(hist[idx], dtype=float) for oid, hist in self.belief_hist.items()},
            "applied_thrust": {oid: np.array(hist[idx], dtype=float) for oid, hist in self.thrust_hist.items()},
            "applied_torque": {oid: np.array(hist[idx], dtype=float) for oid, hist in self.torque_hist.items()},
            "flight_software": flight_software,
            **({"state_metadata": {"central_body": self.planetary_environment.body.body_id,
               "origin_center_id": self.planetary_environment.body.center_id, "inertial_axes": "J2000",
               "environment_sha256": self.planetary_environment.digest}} if self.planetary_environment is not None else {}),
            "spacecraft_resources": resource_snapshot(self.resource_hist, float(self.t_s[idx])),
            "ground_segment": self.ground_segment.ground.snapshot(float(self.t_s[idx])) if self.ground_segment else {},
        }

    def publish_fsw_input(self, object_id: str, event: object) -> None:
        with self._owner_state_context():
            self._publish_fsw_input_owned(object_id, event)

    def _publish_fsw_input_owned(self, object_id: str, event: object) -> None:
        agent = self.agents.get(str(object_id))
        runtime = None if agent is None else agent.flight_software_runtime
        if runtime is None:
            raise ValueError(f"Object {object_id!r} does not use a v2 SatelliteFlightSoftware runtime.")
        runtime.enqueue(event)

    def add_fsw_input_publisher(self, object_id: str, publisher: Callable[..., object]) -> None:
        with self._owner_state_context():
            self._add_fsw_input_publisher_owned(object_id, publisher)

    def _add_fsw_input_publisher_owned(self, object_id: str, publisher: Callable[..., object]) -> None:
        agent = self.agents.get(str(object_id))
        runtime = None if agent is None else agent.flight_software_runtime
        if runtime is None:
            raise ValueError(f"Object {object_id!r} does not use a v2 SatelliteFlightSoftware runtime.")
        runtime.add_input_publisher(publisher)

    def request_fsw_input_publisher_poll(self, object_id: str) -> None:
        with self._owner_state_context():
            self._request_fsw_input_publisher_poll_owned(object_id)

    def _request_fsw_input_publisher_poll_owned(self, object_id: str) -> None:
        agent = self.agents.get(str(object_id))
        runtime = None if agent is None else agent.flight_software_runtime
        if runtime is None:
            raise ValueError(f"Object {object_id!r} does not use a v2 SatelliteFlightSoftware runtime.")
        time_ns = int(round(float(self.t_s[self.current_index]) * 1.0e9))
        runtime.request_input_publisher_poll(time_ns=time_ns)

    def _run_agent_decision(self, ctx: _DecisionContext) -> dict[str, Any]:
        from sim.gnc.contracts import merge_mission_intent_layers

        agent = ctx.agent
        orbit_proxy = (
            None
            if ctx.orbit_controller is None
            else _BudgetedControllerProxy(
                ctx.orbit_controller,
                budget_ms=self.orbit_controller_budget_ms,
                deadline_policy=self.controller_deadline_policy,
            )
        )
        attitude_proxy = (
            None
            if ctx.attitude_controller is None
            else _BudgetedControllerProxy(
                ctx.attitude_controller,
                budget_ms=self.attitude_controller_budget_ms,
                deadline_policy=self.controller_deadline_policy,
            )
        )
        module_out = _run_mission_modules(
            agent=agent,
            t_s=ctx.t_s,
            dt_s=ctx.dt_s,
            env=ctx.env,
            orbit_controller=orbit_proxy,
            attitude_controller=attitude_proxy,
            orb_belief=ctx.orb_belief,
            att_belief=ctx.att_belief,
        )
        strategy_out = _run_mission_strategy(
            agent=agent,
            t_s=ctx.t_s,
            dt_s=ctx.dt_s,
            env=ctx.env,
            orbit_controller=orbit_proxy,
            attitude_controller=attitude_proxy,
            orb_belief=ctx.orb_belief,
            att_belief=ctx.att_belief,
        )
        pre_execution = merge_mission_intent_layers(
            (
                ("mission_modules", module_out),
                ("mission_strategy", strategy_out),
            )
        )
        execution_out = _run_mission_execution(
            agent=agent,
            intent=dict(pre_execution.values),
            t_s=ctx.t_s,
            dt_s=ctx.dt_s,
            env=ctx.env,
            orbit_controller=orbit_proxy,
            attitude_controller=attitude_proxy,
            orb_belief=ctx.orb_belief,
            att_belief=ctx.att_belief,
        )
        mission_out = merge_mission_intent_layers(
            (
                ("mission_modules", module_out),
                ("mission_strategy", strategy_out),
                ("mission_execution", execution_out),
            )
        ).to_runtime_dict()
        if orbit_proxy is not None and orbit_proxy.call_count:
            mission_out["_integrated_orbit_runtime_ms"] = float(orbit_proxy.runtime_ms)
            mission_out["_integrated_orbit_deadline_missed"] = bool(orbit_proxy.deadline_missed)
        if attitude_proxy is not None and attitude_proxy.call_count:
            mission_out["_integrated_attitude_runtime_ms"] = float(attitude_proxy.runtime_ms)
            mission_out["_integrated_attitude_deadline_missed"] = bool(attitude_proxy.deadline_missed)
        return mission_out

    def _build_object_step_inputs(
        self,
        *,
        world_truth_start: dict[str, StateTruth],
        t_s: float,
        t_next: float,
        sample_index: int,
    ) -> list[ObjectStepInput]:
        inputs: list[ObjectStepInput] = []
        for aid, agent in self.agents.items():
            if not agent.active:
                continue
            inputs.append(
                ObjectStepInput(
                    object_id=aid,
                    agent=agent,
                    initial_truth=world_truth_start[aid],
                    # The step uses Jacobi-style immutable start-of-interval
                    # truth. Share one mapping across inputs so process chunks
                    # serialize it once rather than once per object.
                    world_truth_decision=world_truth_start,
                    t_s=float(t_s),
                    t_next=float(t_next),
                    sample_index=int(sample_index),
                )
            )
        return inputs

    def _step_object_serial(self, item: ObjectStepInput) -> ObjectStepResult:
        aid = item.object_id
        agent = item.agent
        object_t0 = perf_counter()

        if agent.kind == "rocket":
            rocket_result = self.rocket_stepper.step(
                agent=agent,
                world_truth_decision=item.world_truth_decision,
                t_s=item.t_s,
                t_next=item.t_next,
            )
            agent.truth = rocket_result.truth
            result = ObjectStepResult(
                object_id=aid,
                stage="rocket_step",
                elapsed_s=perf_counter() - object_t0,
                truth=rocket_result.truth,
                thrust_eci_km_s2=np.array(rocket_result.thrust_eci_km_s2, dtype=float),
                torque_body_nm=np.array(rocket_result.torque_body_nm, dtype=float),
                delta_v_m_s=float(rocket_result.delta_v_m_s),
                max_accel_km_s2=float(rocket_result.max_accel_km_s2),
                burned=bool(rocket_result.burned),
                throttle=rocket_result.throttle,
                stage_index=rocket_result.stage_index,
                q_dyn_pa=rocket_result.q_dyn_pa,
                mach=rocket_result.mach,
                metrics=dict(rocket_result.metrics or {}),
            )
        else:
            provider = self.general_propagation.get(aid)
            if provider is not None:
                native_truth = self.satellite_stepper.passive_history.step(
                    aid=aid, agent=agent, initial=item.initial_truth, t_s=item.t_s, t_next=item.t_next,
                )
                agent.truth = native_truth if native_truth is not None else provider.canonical_state_at(item.t_next)
                if agent.belief is not None and agent.belief.state.size >= 6:
                    agent.belief.state[:6] = np.hstack((agent.truth.position_eci_km, agent.truth.velocity_eci_km_s))
                    agent.belief.last_update_t_s = item.t_next
                result = ObjectStepResult(
                    object_id=aid,
                    stage="general_propagation_step",
                    elapsed_s=perf_counter() - object_t0,
                    truth=agent.truth,
                    thrust_eci_km_s2=self.zero3.copy(),
                    torque_body_nm=self.zero3.copy(),
                )
            else:
                sat_result = self.satellite_stepper.step(
                    aid=aid,
                    agent=agent,
                    initial_truth=item.initial_truth,
                    world_truth_decision=item.world_truth_decision,
                    t_s=item.t_s,
                    t_next=item.t_next,
                    sample_index=item.sample_index,
                )
                agent.truth = sat_result.truth
                result = ObjectStepResult(
                    object_id=aid,
                    stage="satellite_step",
                    elapsed_s=perf_counter() - object_t0,
                    truth=sat_result.truth,
                    thrust_eci_km_s2=np.array(sat_result.average_thrust_eci_km_s2, dtype=float),
                    torque_body_nm=np.array(sat_result.average_torque_body_nm, dtype=float),
                    delta_v_m_s=float(sat_result.delta_v_m_s),
                    max_accel_km_s2=float(sat_result.max_accel_km_s2),
                    burned=bool(sat_result.burned),
                )

        if agent.bridge is None:
            return result

        bridge_t0 = perf_counter()
        evt = {"t_s": float(item.t_next), "object_id": aid}
        if hasattr(agent.bridge, "step"):
            try:
                ret = agent.bridge.step(evt)
                if ret is not None:
                    evt["bridge"] = ret
            except Exception as ex:
                if bool(getattr(self.cfg.simulator.plugin_validation, "strict_runtime", False)):
                    raise RuntimeError(f"{aid} bridge.step failed at t={float(item.t_next):.6g} s") from ex
                evt["bridge_error"] = str(ex)
        return replace(
            result,
            bridge_events=[evt],
            bridge_elapsed_s=perf_counter() - bridge_t0,
        )

    def _apply_object_step_result(self, result: ObjectStepResult, *, sample_index: int) -> None:
        aid = result.object_id
        if result.updated_agent is not None:
            result.updated_agent.knowledge_base = self.agents[aid].knowledge_base
            self.agents[aid] = result.updated_agent
            self._refresh_agent_role_pointers(aid, result.updated_agent)
        agent = self.agents[aid]
        agent.truth = result.truth
        for name, delta in result.attitude_guardrail_count_deltas.items():
            if hasattr(self.attitude_guardrail_stats, name):
                setattr(
                    self.attitude_guardrail_stats,
                    name,
                    int(getattr(self.attitude_guardrail_stats, name)) + int(delta),
                )
        if result.belief_state is not None:
            agent.belief = StateBelief(
                state=np.array(result.belief_state, dtype=float),
                covariance=(
                    self.eye6.copy()
                    if result.belief_covariance is None
                    else np.array(result.belief_covariance, dtype=float)
                ),
                last_update_t_s=(
                    float(result.belief_last_update_t_s)
                    if result.belief_last_update_t_s is not None
                    else float(result.truth.t_s)
                ),
            )
        k = int(sample_index)

        if result.stage == "rocket_step":
            if aid in self.throttle_hist and result.throttle is not None:
                self.throttle_hist[aid][k] = float(result.throttle)
            if self.rocket_stage_hist is not None and result.stage_index is not None:
                self.rocket_stage_hist[k + 1] = float(result.stage_index)
            if self.rocket_q_dyn_hist is not None and result.q_dyn_pa is not None:
                self.rocket_q_dyn_hist[k + 1] = float(result.q_dyn_pa)
            if self.rocket_mach_hist is not None and result.mach is not None:
                self.rocket_mach_hist[k + 1] = float(result.mach)
            for metric_key, metric_value in dict(result.metrics or {}).items():
                if metric_key in self.rocket_metric_hists:
                    self.rocket_metric_hists[metric_key][k + 1] = float(metric_value)

        self.thrust_hist[aid][k + 1, :] = result.thrust_eci_km_s2
        self.torque_hist[aid][k + 1, :] = result.torque_body_nm
        if result.desired_attitude_quat_bn is not None and aid in self.desired_attitude_hist:
            self.desired_attitude_hist[aid][k + 1, :] = np.array(result.desired_attitude_quat_bn, dtype=float)
        self.total_dv_m_s_by_object[aid] += float(result.delta_v_m_s)
        self.max_accel_km_s2_by_object[aid] = max(
            self.max_accel_km_s2_by_object[aid],
            float(result.max_accel_km_s2),
        )
        if result.burned:
            self.burn_samples_by_object[aid] += 1

        self.runtime_profiler.record_object(aid, result.stage, float(result.elapsed_s))
        self.runtime_profiler.record_stage("object_step", float(result.elapsed_s))
        if self.runtime_profiler.enabled:
            for stage, elapsed_s in dict(result.profile_stage_seconds).items():
                count = int(dict(result.profile_stage_counts).get(stage, 0))
                if count <= 0:
                    continue
                self.runtime_profiler.object_stage_seconds.setdefault(aid, {})[str(stage)] = float(
                    self.runtime_profiler.object_stage_seconds.setdefault(aid, {}).get(str(stage), 0.0)
                    + float(elapsed_s)
                )
                self.runtime_profiler.object_stage_counts.setdefault(aid, {})[str(stage)] = int(
                    self.runtime_profiler.object_stage_counts.setdefault(aid, {}).get(str(stage), 0) + count
                )
        if result.controller_debug_events:
            self.controller_debug_hist[aid].extend(result.controller_debug_events)
        if result.command_decision_events:
            self.command_decision_hist[aid].extend(result.command_decision_events)
        if result.updated_agent is not None or result.last_orbital_command_eval_t_s is not None:
            self._last_orbital_command_eval_t_s[aid] = result.last_orbital_command_eval_t_s
        if result.latched_orbital_thrust_cmd is not None:
            self._latched_orbital_thrust_cmd_by_object[aid] = np.array(
                result.latched_orbital_thrust_cmd,
                dtype=float,
            )
        if result.bridge_events:
            self.bridge_hist[aid].extend(result.bridge_events)
            self.runtime_profiler.record_object(aid, "bridge_step", float(result.bridge_elapsed_s))
            self.runtime_profiler.record_stage("bridge_step", float(result.bridge_elapsed_s))

    def _refresh_agent_role_pointers(self, aid: str, agent: AgentRuntime) -> None:
        if aid == "rocket" or agent.kind == "rocket":
            self.rocket = agent
        if aid == "chaser":
            self.chaser = agent
        if aid == "target":
            self.target = agent

    def _apply_worker_knowledge_sync_results(
        self,
        results: list[ObjectKnowledgeSyncResult],
        *,
        sample_index: int,
    ) -> None:
        detection: dict[str, Any] = {}
        consistency: dict[str, Any] = {}
        for result in results:
            aid = str(result.object_id)
            if result.detection_summary:
                detection[aid] = dict(result.detection_summary)
            if result.consistency_summary:
                consistency[aid] = dict(result.consistency_summary)
            snapshot = dict(result.knowledge_snapshot)
            for tid, hist in self.knowledge_hist.get(aid, {}).items():
                if tid not in snapshot and sample_index > 0:
                    hist[sample_index, :] = hist[sample_index - 1, :]
            for tid, belief in dict(result.knowledge_snapshot).items():
                hist = self.knowledge_hist.get(aid, {}).get(tid)
                if hist is None:
                    continue
                hist[sample_index, :] = np.array(belief.state, dtype=float).reshape(-1)[:6]
            for tid, meas in dict(result.measurement_snapshot).items():
                measurement_hist = self.knowledge_measurement_hist.get(aid, {}).get(tid)
                if measurement_hist is None:
                    continue
                arr = np.array(meas, dtype=float).reshape(-1)
                n = min(int(arr.size), int(measurement_hist.shape[1]))
                measurement_hist[sample_index, :n] = arr[:n]
        if detection:
            self.worker_knowledge_detection_by_observer = detection
        if consistency:
            self.worker_knowledge_consistency_by_observer = consistency

    def _process_worker_engine_snapshot(self, item: ObjectStepInput) -> _SingleRunEngine:
        aid = item.object_id
        worker = copy(self)
        # The owner lock protects live state but is process-local and must not
        # cross the worker transport boundary.  Worker stepping is isolated to
        # its copied object state and uses the no-op context above.
        worker._owner_state_lock = None
        worker.active_step_callback = None
        worker._forecast_dynamics_by_object = {
            object_id: runtime.dynamics
            for object_id, runtime in self.agents.items()
            if runtime.active and runtime.kind == "satellite" and runtime.dynamics is not None
        }
        worker.agents = {aid: item.agent}
        worker.rocket = worker.agents.get(aid) if item.agent.kind == "rocket" else None
        worker.chaser = worker.agents.get(aid) if aid == "chaser" else None
        worker.target = worker.agents.get(aid) if aid == "target" else None
        worker.general_propagation = {
            oid: provider for oid, provider in self.general_propagation.items() if oid == aid
        }
        worker.controller_debug_hist = {aid: []}
        worker.command_decision_hist = {aid: []}
        worker.bridge_hist = {aid: []}
        worker.runtime_profiler = _RuntimeProfiler(
            object_ids=[aid],
            enabled=bool(getattr(self.runtime_profiler, "enabled", True)),
        )
        worker.object_step_executor = SerialObjectStepExecutor(worker)
        worker.rocket_stepper = _RocketStepper(worker)
        worker.satellite_stepper = _SatelliteStepper(worker)
        worker.termination_monitor = _TerminationMonitor(worker)
        worker.history_store = SingleRunHistoryStore(worker)
        worker.knowledge_sync = None
        worker._object_worker_compact_buffers = True
        worker.truth_hist = {}
        worker.belief_hist = {}
        worker.thrust_hist = {}
        worker.torque_hist = {}
        worker.desired_attitude_hist = {aid: np.full((1, 4), np.nan)}
        worker.knowledge_hist = {}
        worker.knowledge_measurement_hist = {}
        worker._last_orbital_command_eval_t_s = {
            aid: self._last_orbital_command_eval_t_s.get(aid)
        }
        worker._latched_orbital_thrust_cmd_by_object = {
            aid: np.array(self._latched_orbital_thrust_cmd_by_object.get(aid, self.zero3), dtype=float)
        }
        return worker

    @staticmethod
    def _validate_object_step_results(
        inputs: list[ObjectStepInput],
        results: list[ObjectStepResult],
    ) -> None:
        input_ids = [item.object_id for item in inputs]
        result_ids = [item.object_id for item in results]
        if result_ids != input_ids:
            raise RuntimeError(
                "Object step executor must return exactly one result per input in input order. "
                f"expected={input_ids!r}, received={result_ids!r}"
            )

    def apply_impulse(
        self,
        object_id: str,
        delta_v_eci_m_s: tuple[float, float, float] | np.ndarray,
        *,
        maneuver_id: str,
    ) -> dict[str, Any]:
        """Apply an exact ECI velocity jump at the current sample boundary.

        This explicit run operation is limited to passive ONP satellites. The
        pre/post pair is retained separately because object_state has one row
        per object and sample, whose event-time row is the post-impulse state.
        """
        with self._owner_state_context():
            return self._apply_impulse_owned(
                object_id,
                delta_v_eci_m_s,
                maneuver_id=maneuver_id,
            )

    def _apply_impulse_owned(
        self,
        object_id: str,
        delta_v_eci_m_s: tuple[float, float, float] | np.ndarray,
        *,
        maneuver_id: str,
    ) -> dict[str, Any]:
        if self.terminated_early or self.done:
            raise ValueError("Cannot apply an impulse after the run has ended")
        agent = self.agents.get(object_id)
        if agent is None or not agent.active or agent.kind != "satellite" or agent.truth is None:
            raise ValueError("Impulse object must be an active satellite")
        if object_id in self.general_propagation or agent.dynamics is None:
            raise ValueError("Impulse requires an ONP numerical satellite")
        if agent.flight_software_runtime is not None and str(agent.runtime_profile) != "trajectory_only":
            raise ValueError("Impulse execution currently requires a passive trajectory_only object")
        if not maneuver_id or any(row["maneuver_id"] == maneuver_id for row in self.impulsive_maneuvers):
            raise ValueError("maneuver_id must be nonempty and unique in this run")
        dv = np.asarray(delta_v_eci_m_s, dtype=float)
        if dv.shape != (3,) or not np.isfinite(dv).all() or not 0.0 < float(np.linalg.norm(dv)) <= 100_000.0:
            raise ValueError("Impulse delta_v_eci_m_s must be a finite nonzero three-vector within 100 km/s")
        index = int(self.current_index)
        t_s = float(self.t_s[index])
        pre = np.concatenate((agent.truth.position_eci_km, agent.truth.velocity_eci_km_s))
        agent.truth.velocity_eci_km_s = agent.truth.velocity_eci_km_s + dv / 1000.0
        post = np.concatenate((agent.truth.position_eci_km, agent.truth.velocity_eci_km_s))
        _write_state_truth(self.truth_hist[object_id][index, :], agent.truth)
        self._forecast_truth_cache = {}
        row = {
            "maneuver_id": maneuver_id,
            "object_id": object_id,
            "sample_index": index,
            "time_s": t_s,
            "frame": "eci",
            "delta_v_eci_m_s": dv.tolist(),
            "pre_state_eci": pre.tolist(),
            "post_state_eci": post.tolist(),
        }
        self.impulsive_maneuvers.append(row)
        return dict(row)

    def step(self, dt_s: float | None = None) -> dict[str, Any]:
        return self._advance_step(dt_s=dt_s, return_snapshot=True)

    def _advance_step(
        self, *, dt_s: float | None = None, return_snapshot: bool = True,
    ) -> dict[str, Any]:
        # Keep each owner step atomic with respect to session-facing live
        # input/maneuver/publisher operations.  RLock is intentional: the
        # owner callback and compatibility seams may call back into the engine.
        with self._owner_state_context():
            return self._advance_step_owned(dt_s=dt_s, return_snapshot=return_snapshot)

    def _advance_step_owned(
        self, *, dt_s: float | None = None, return_snapshot: bool = True,
    ) -> dict[str, Any]:
        activate_attitude_guardrail_stats(self.attitude_guardrail_stats)
        if not bool(getattr(self, "_acceleration_context_active", False)):
            with acceleration_context_from_config(self.cfg):
                self._acceleration_context_active = True
                try:
                    return self._advance_step_owned(dt_s=dt_s, return_snapshot=return_snapshot)
                finally:
                    self._acceleration_context_active = False
        if self.done:
            return self.snapshot(include_flight_software=False) if return_snapshot else {}

        step_wall_t0 = perf_counter()
        compact_t0 = perf_counter()
        self._compact_dynamic_history_if_needed()
        self.runtime_profiler.record_stage("dynamic_history_compaction", perf_counter() - compact_t0)
        k = int(self.current_index)
        t = float(self.t_s[k])
        if self.ground_segment is not None and dt_s is not None and float(dt_s) != self.dt:
            raise ValueError("ground_segment requires the configured fixed simulator.dt_s")
        step_dt = self.dt if dt_s is None else float(dt_s)
        if not np.isfinite(step_dt) or step_dt <= 0.0:
            raise ValueError("step dt_s must be positive.")
        remaining_s = max(float(self.cfg.simulator.duration_s) - t, 0.0)
        step_dt = min(step_dt, remaining_s)
        if step_dt <= 0.0:
            return self.snapshot(include_flight_software=False)
        t_next = float(t + step_dt)
        boundary_eps = max(1.0e-9, 1.0e-9 * max(abs(t), abs(t_next), 1.0))
        if self.rocket is not None:
            deployment_times = [
                float(agent.deploy_time_s)
                for agent in self.agents.values()
                if (
                    agent.kind == "satellite"
                    and not agent.active
                    and agent.deploy_source == "rocket_deployment"
                    and agent.deploy_time_s is not None
                    and t + boundary_eps < float(agent.deploy_time_s) < t_next - boundary_eps
                )
            ]
            if deployment_times:
                # Make a time-gated deployment an actual sample boundary. The
                # payload is copied from the rocket at deployment and is only
                # propagated during the interval that follows it.
                return self.step(dt_s=min(deployment_times) - t)
        capacity_t0 = perf_counter()
        self._ensure_sample_capacity(k + 1)
        self.runtime_profiler.record_stage("history_capacity", perf_counter() - capacity_t0)
        self.t_s[k + 1] = t_next
        self._time_dependent_env_cache = {}
        self._forecast_truth_cache = {}

        if self.rocket is not None:
            for agent in self.agents.values():
                if agent.kind == "satellite" and not agent.active and agent.deploy_source == "rocket_deployment":
                    if agent.deploy_time_s is not None and float(agent.deploy_time_s) <= t + boundary_eps:
                        _deploy_from_rocket(agent, self.rocket, t)
                        if agent.active and agent.truth is not None:
                            _write_state_truth(self.truth_hist[agent.object_id][k, :], agent.truth)
                            if agent.belief is not None:
                                self._ensure_belief_hist_width(agent.object_id, agent.belief.state.size)
                                self.belief_hist[agent.object_id][k, : agent.belief.state.size] = agent.belief.state

        snapshot_t0 = perf_counter()
        world_truth_start = {
            aid: (
                agent.truth.copy()
                if agent.kind == "satellite"
                else _rocket_state_to_truth(agent.rocket_state).copy()
            )
            for aid, agent in self.agents.items()
            if agent.active
        }
        if self.target_reference_truth is not None:
            world_truth_start["target_reference"] = self.target_reference_truth.copy()
        world_truth_live = dict(world_truth_start)
        self.runtime_profiler.record_stage("world_snapshot", perf_counter() - snapshot_t0)

        if self.target_reference_truth is not None and self.target_reference_dynamics is not None:
            target_ref_t0 = perf_counter()
            env_ref = {
                **self.base_environment,
                "world_truth": world_truth_live,
                "attitude_disabled": True,
                TIME_DEPENDENT_ENV_CACHE_KEY: self._time_dependent_env_cache,
            }
            self.target_reference_truth = self.target_reference_dynamics.step(
                state=self.target_reference_truth,
                command=Command.zero(),
                env=env_ref,
                dt_s=step_dt,
            )
            assert self.target_reference_orbit_hist is not None
            self.target_reference_orbit_hist[k + 1, 0:3] = self.target_reference_truth.position_eci_km
            self.target_reference_orbit_hist[k + 1, 3:6] = self.target_reference_truth.velocity_eci_km_s
            world_truth_live["target_reference"] = self.target_reference_truth.copy()
            self.runtime_profiler.record_stage("target_reference_step", perf_counter() - target_ref_t0)

        object_inputs = self._build_object_step_inputs(
            world_truth_start=world_truth_start,
            t_s=t,
            t_next=t_next,
            sample_index=k,
        )
        try:
            object_results = (
                self.collision_stepper.step_objects(object_inputs)
                if self.collision_stepper is not None
                else self.system_force_stepper.step_objects(object_inputs)
                if self.system_force_stepper is not None
                else self.object_step_executor.step_objects(object_inputs)
            )
        except ObjectStepBackendUnavailable as exc:
            policy = str(dict(self.object_execution_plan or {}).get("policy", "configured"))
            if policy != "auto":
                raise
            self.object_step_executor.shutdown()
            self.object_step_executor = SerialObjectStepExecutor(self)
            self.object_execution_plan["initial_selected_backend"] = self.object_execution_plan.get(
                "selected_backend",
                "process_pool",
            )
            self.object_execution_plan["selected_backend"] = "serial"
            self.object_execution_plan["selected_workers"] = 1
            self.object_execution_plan["runtime_fallback_reason"] = str(exc)
            self.object_execution_plan["reason"] = (
                "auto planner fell back to serial after object-worker transport became unavailable"
            )
            object_results = self.object_step_executor.step_objects(object_inputs)
        except ValueError as exc:
            if self.planetary_environment is None or not self.termination_monitor.handle_gravity_domain_exit(exc, object_inputs):
                raise
            self.runtime_profiler.record_stage("step_wall", perf_counter() - step_wall_t0)
            return self.snapshot(include_flight_software=False) if return_snapshot else {}
        self._validate_object_step_results(object_inputs, object_results)
        for result in object_results:
            self._apply_object_step_result(result, sample_index=k)
            agent = self.agents[result.object_id]
            world_truth_live[result.object_id] = (
                agent.truth if agent.kind == "satellite" else _rocket_state_to_truth(agent.rocket_state)
            )
        if self.collision_stepper is not None:
            self.collision_events.extend(self.collision_stepper.last_events)

        if self.rocket is not None:
            for aid, agent in self.agents.items():
                if (
                    agent.kind == "satellite"
                    and not agent.active
                    and agent.deploy_source == "rocket_deployment"
                    and agent.deploy_time_s is not None
                    and float(agent.deploy_time_s) <= t_next + boundary_eps
                ):
                    _deploy_from_rocket(agent, self.rocket, t_next)
                    if agent.active and agent.truth is not None:
                        world_truth_live[aid] = agent.truth

        if self.rocket is not None:
            for aid, agent in self.agents.items():
                if (
                    agent.kind == "satellite"
                    and not agent.active
                    and agent.deploy_source == "rocket_deployment"
                    and agent.deploy_time_s is not None
                    and float(agent.deploy_time_s) <= t_next + boundary_eps
                ):
                    _deploy_from_rocket(agent, self.rocket, t_next)
                    if agent.active and agent.truth is not None:
                        world_truth_live[aid] = agent.truth

        termination_update_t0 = perf_counter()
        self.termination_monitor.update_rocket_insertion(t_s=t_next, dt_s=step_dt)
        if self.rocket is not None and self.rocket_inserted:
            for aid, agent in self.agents.items():
                if agent.kind == "satellite" and not agent.active and agent.deploy_source == "rocket_insertion":
                    _deploy_from_rocket(agent, self.rocket, t_next)
                    if agent.active and agent.truth is not None:
                        world_truth_live[aid] = agent.truth
        self.runtime_profiler.record_stage("termination_update", perf_counter() - termination_update_t0)

        knowledge_t0 = perf_counter()
        worker_knowledge_results = self.object_step_executor.sync_after_step(
            world_truth=world_truth_live,
            sample_index=k + 1,
            t_s=t_next,
        )
        if worker_knowledge_results is None:
            self.knowledge_sync.update_after_step(
                world_truth=world_truth_live,
                sample_index=k + 1,
                t_s=t_next,
            )
        else:
            self._apply_worker_knowledge_sync_results(
                worker_knowledge_results,
                sample_index=k + 1,
            )
        self.runtime_profiler.record_stage("knowledge_sync", perf_counter() - knowledge_t0)

        history_t0 = perf_counter()
        for aid, agent in self.agents.items():
            if not agent.active:
                continue
            truth = agent.truth if agent.kind == "satellite" else _rocket_state_to_truth(agent.rocket_state)
            _write_state_truth(self.truth_hist[aid][k + 1, :], truth)
            if truth.resource_state is not None:
                self.resource_hist.setdefault(aid, []).append(dict(truth.resource_state))
            if agent.belief is not None:
                self._ensure_belief_hist_width(aid, agent.belief.state.size)
                self.belief_hist[aid][k + 1, : agent.belief.state.size] = agent.belief.state
            else:
                runtime = agent.flight_software_runtime
                if runtime is not None:
                    runtime.prepare_interval(
                        truth,
                        start_time_ns=int(round(t_next * 1.0e9)),
                        world_truth=world_truth_live,
                    )
                onboard = _onboard_belief_vector(agent)
                if onboard is not None:
                    self._ensure_belief_hist_width(aid, onboard.size)
                    self.belief_hist[aid][k + 1, : onboard.size] = onboard
            self._record_reentry_metrics(aid=aid, truth=truth, sample_index=k + 1, dt_s=step_dt)
        self.runtime_profiler.record_stage("history_write", perf_counter() - history_t0)

        self.current_index = k + 1
        self._advance_ground_segment(self.current_index)
        self._emit_step_callback(self.current_index)

        termination_check_t0 = perf_counter()
        if self.termination_monitor.check_reentry(t_s=t_next):
            self.runtime_profiler.record_stage("termination_check", perf_counter() - termination_check_t0)
            self.runtime_profiler.record_stage("step_wall", perf_counter() - step_wall_t0)
            return self.snapshot(include_flight_software=False) if return_snapshot else {}
        if self.termination_monitor.check_earth_impact(t_s=t_next):
            self.runtime_profiler.record_stage("termination_check", perf_counter() - termination_check_t0)
            self.runtime_profiler.record_stage("step_wall", perf_counter() - step_wall_t0)
            return self.snapshot(include_flight_software=False) if return_snapshot else {}
        self.runtime_profiler.record_stage("termination_check", perf_counter() - termination_check_t0)
        self.runtime_profiler.record_stage("step_wall", perf_counter() - step_wall_t0)
        return self.snapshot(include_flight_software=False) if return_snapshot else {}

    def run(self) -> dict[str, Any]:
        self._ensure_full_history_payload_allowed()
        if not bool(getattr(self, "_acceleration_context_active", False)):
            with acceleration_context_from_config(self.cfg):
                self._acceleration_context_active = True
                try:
                    return self.run()
                finally:
                    self._acceleration_context_active = False
        try:
            while not self.done:
                if getattr(self.step, "__func__", None) is _SingleRunEngine.step:
                    # Full runs retain the same sample/lifecycle boundaries,
                    # but no caller consumes their per-step copied snapshot.
                    self._advance_step(return_snapshot=False)
                else:
                    # Preserve downstream step overrides and patch seams.
                    self.step()
            self._shutdown_flight_software()
            return self.build_payload()
        finally:
            try:
                self._shutdown_flight_software()
            finally:
                self.object_step_executor.shutdown()

    def _shutdown_flight_software(self) -> None:
        shutdown_time_ns = int(round(float(self.t_s[self.current_index]) * 1.0e9))
        for agent in self.agents.values():
            runtime = agent.flight_software_runtime
            if runtime is None:
                continue
            aligned_time_ns = shutdown_time_ns - shutdown_time_ns % runtime.tick_period_ns
            runtime.shutdown(time_ns=aligned_time_ns)

    def _build_payload_parts(self) -> _SingleRunPayloadParts:
        return SingleRunPayloadAssembler(
            self,
            initialization_metadata_builder=_object_initialization_metadata,
        ).build_parts()

    def _payload_from_parts(self, parts: _SingleRunPayloadParts) -> dict[str, Any]:
        return SingleRunPayloadAssembler(
            self,
            initialization_metadata_builder=_object_initialization_metadata,
        ).payload_from_parts(parts)

    def _primary_rocket_throttle_history(self) -> np.ndarray:
        return SingleRunPayloadAssembler(
            self,
            initialization_metadata_builder=_object_initialization_metadata,
        ).primary_rocket_throttle_history()

    def _ensure_full_history_payload_allowed(self) -> None:
        SingleRunPayloadAssembler(
            self,
            initialization_metadata_builder=_object_initialization_metadata,
        ).ensure_full_history_allowed()

    def build_run_payload(self) -> dict[str, Any]:
        """Build the in-memory run payload without rendering or writing artifacts."""

        return SingleRunPayloadAssembler(
            self,
            initialization_metadata_builder=_object_initialization_metadata,
        ).build_run_payload()

    def _write_artifacts(self, payload: dict[str, Any], parts: _SingleRunPayloadParts) -> dict[str, Any]:
        return SingleRunPayloadAssembler(
            self,
            initialization_metadata_builder=_object_initialization_metadata,
        ).write_artifacts(payload, parts)

    def build_payload(self) -> dict[str, Any]:
        return SingleRunPayloadAssembler(
            self,
            initialization_metadata_builder=_object_initialization_metadata,
        ).build_payload()


def _run_single_config(
    cfg: SimulationScenarioConfig,
    step_callback: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    # The public-safe budget helper owns the execution-local marker.  Import
    # the private contract module only for an active thread campaign so public
    # single-run imports do not cross the private campaign boundary.
    from sim.execution.worker_budget import current_thread_campaign_runtime_scope

    with acceleration_context_from_config(cfg):
        engine = _SingleRunEngine(cfg, step_callback=step_callback)
        try:
            runtime_scope = current_thread_campaign_runtime_scope()
            if runtime_scope is not None:
                from sim.execution.thread_campaign import preflight_thread_campaign_engine_if_active

                preflight_thread_campaign_engine_if_active(engine)
        except BaseException:
            # Engine construction creates the selected object executor.  If a
            # runtime contract catches an unsafe materialized hook, close that
            # executor before the worker future reports the failure.
            try:
                engine.object_step_executor.shutdown()
            except Exception:
                pass
            raise
        return engine.run()


def _is_truthy_env(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _coerce_noninteractive_for_automation(cfg: SimulationScenarioConfig) -> SimulationScenarioConfig:
    if not (_is_truthy_env("SIM_AUTOMATION") or _is_truthy_env("CI")):
        return cfg
    root = cfg.to_dict()
    outputs = root.setdefault("outputs", {})
    mode = str(outputs.get("mode", "interactive")).strip().lower()
    if mode == "interactive":
        outputs["mode"] = "save"
    return scenario_config_from_dict(root, source_path=getattr(cfg, "source_path", None))
