from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from sim.config import (
    SimulationScenarioConfig,
    validate_scenario_plugins,
)
from sim.config.object_refs import configured_objects
from sim.core.models import Command
from sim.execution import create_single_run_engine, run_simulation_scenario
from sim.public_api.config import (
    ControllerFactory,
    SimulationConfig,
    _api_sealed_policy,
)
from sim.public_api.controller_adapters import (
    _controller_object,
    _mission_object,
)
from sim.public_api.results import (
    SimulationResult,
)
from sim.public_api.snapshots import SimulationSnapshot
from sim.scenarios import ScenarioArtifact
from sim.scenarios import ScenarioBuilder as ScenarioBuilder
from sim.scenarios import ValidationIssue as ValidationIssue
from sim.security import ConfigPathPolicy
from sim.security.sealed_mode import SealedModePolicy, validate_sealed_mode


class SimulationSession:
    def __init__(
        self,
        config: ScenarioArtifact | SimulationConfig | SimulationScenarioConfig | dict[str, Any],
        *,
        sealed_mode: bool = False,
        sealed_policy: SealedModePolicy | None = None,
        history_mode: str = "full",
        initial_history_capacity: int = 4096,
        max_history_samples: int = 4096,
    ):
        self._sealed_policy = _api_sealed_policy(sealed_mode=sealed_mode, sealed_policy=sealed_policy)
        self._base_config = self._coerce_config(config)
        self._enforce_sealed_mode(self._base_config)
        self._active_config = self._base_config
        self._history_mode = str(history_mode or "full").strip().lower()
        self._initial_history_capacity = int(max(2, initial_history_capacity))
        self._max_history_samples = int(max(2, max_history_samples))
        self._result: SimulationResult | None = None
        self._step_index = 0
        self._done = False
        self._engine: Any | None = None
        self._controller_overrides: dict[tuple[str, str], ControllerFactory] = {}
        self._mission_overrides: dict[tuple[str, str], ControllerFactory] = {}
        self._controller_originals: dict[tuple[str, str], Any] = {}
        self._mission_originals: dict[tuple[str, str], Any] = {}

    @classmethod
    def from_config(
        cls,
        config: ScenarioArtifact | SimulationConfig | SimulationScenarioConfig | dict[str, Any],
        *,
        sealed_mode: bool = False,
        sealed_policy: SealedModePolicy | None = None,
        history_mode: str = "full",
        initial_history_capacity: int = 4096,
        max_history_samples: int = 4096,
    ) -> SimulationSession:
        return cls(
            config,
            sealed_mode=sealed_mode,
            sealed_policy=sealed_policy,
            history_mode=history_mode,
            initial_history_capacity=initial_history_capacity,
            max_history_samples=max_history_samples,
        )

    @classmethod
    def from_yaml(
        cls,
        path: str | Path,
        *,
        sealed_mode: bool = False,
        sealed_policy: SealedModePolicy | None = None,
        history_mode: str = "full",
        initial_history_capacity: int = 4096,
        max_history_samples: int = 4096,
    ) -> SimulationSession:
        return cls(
            SimulationConfig.from_yaml(path),
            sealed_mode=sealed_mode,
            sealed_policy=sealed_policy,
            history_mode=history_mode,
            initial_history_capacity=initial_history_capacity,
            max_history_samples=max_history_samples,
        )

    @staticmethod
    def _coerce_config(
        config: ScenarioArtifact | SimulationConfig | SimulationScenarioConfig | dict[str, Any],
    ) -> SimulationConfig:
        if isinstance(config, ScenarioArtifact):
            return config.to_config()
        if isinstance(config, SimulationConfig):
            return config
        if isinstance(config, SimulationScenarioConfig):
            return SimulationConfig(config)
        if isinstance(config, dict):
            return SimulationConfig.from_dict(config)
        raise TypeError(f"Unsupported config type: {type(config)!r}")

    @property
    def config(self) -> SimulationConfig:
        return self._active_config

    @property
    def result(self) -> SimulationResult | None:
        return self._result

    @property
    def done(self) -> bool:
        if self._engine is not None:
            return bool(self._engine.done)
        return bool(self._done)

    def reset(self, seed: int | None = None) -> SimulationSnapshot | None:
        self._shutdown_engine_workers()
        self._active_config = self._base_config.with_seed(seed) if seed is not None else self._base_config
        self._enforce_sealed_mode(self._active_config)
        self._result = None
        self._step_index = 0
        self._done = False
        self._engine = None
        if self._is_batch_analysis(self._active_config.scenario):
            return None
        self._ensure_engine()
        assert self._engine is not None
        snap = self._engine.snapshot(0)
        return SimulationSnapshot(
            step_index=int(snap["step_index"]),
            time_s=float(snap["time_s"]),
            truth=dict(snap["truth"]),
            belief=dict(snap["belief"]),
            applied_thrust=dict(snap["applied_thrust"]),
            applied_torque=dict(snap["applied_torque"]),
            spacecraft_resources=dict(snap.get("spacecraft_resources", {})),
            ground_segment=snap.get("ground_segment", {}),
            state_metadata=dict(snap.get("state_metadata", {})),
        )

    def run(self, *, step_callback: Any | None = None) -> SimulationResult:
        if self._is_batch_analysis(self._active_config.scenario):
            if self._controller_overrides or self._mission_overrides:
                raise RuntimeError("Runtime API controller/mission overrides are only supported for single-run sessions.")
            payload = self._run_batch_analysis(self._active_config)
            self._result = SimulationResult(config=self._active_config, payload=payload)
            self._done = True
            return self._result

        self._ensure_engine(step_callback=step_callback)
        assert self._engine is not None
        payload = self._engine.run()
        self._result = SimulationResult(config=self._active_config, payload=payload)
        self._step_index = max(self._result.num_steps - 1, 0)
        self._done = True
        return self._result

    def step(self, dt_s: float | None = None) -> SimulationSnapshot:
        if self._is_batch_analysis(self._active_config.scenario):
            raise RuntimeError("SimulationSession.step() is only available for single-run scenarios.")
        self._ensure_engine()
        assert self._engine is not None
        try:
            if dt_s is None:
                snap = self._engine.step()
            else:
                snap = self._engine.step(dt_s=float(dt_s))
        except BaseException:
            self._shutdown_engine_workers(suppress_errors=True)
            raise
        self._step_index = int(snap["step_index"])
        self._done = bool(self._engine.done)
        if self._done:
            self._shutdown_engine_workers()
        return SimulationSnapshot(
            step_index=int(snap["step_index"]),
            time_s=float(snap["time_s"]),
            truth=dict(snap["truth"]),
            belief=dict(snap["belief"]),
            applied_thrust=dict(snap["applied_thrust"]),
            applied_torque=dict(snap["applied_torque"]),
            spacecraft_resources=dict(snap.get("spacecraft_resources", {})),
            ground_segment=snap.get("ground_segment", {}),
            state_metadata=dict(snap.get("state_metadata", {})),
        )

    def publish_fsw_input(self, object_id: str, event: object) -> None:
        """Publish one typed input event to a satellite's flight-software bus."""

        self._ensure_engine()
        assert self._engine is not None
        self._engine.publish_fsw_input(str(object_id), event)

    def flight_software_observations(self, object_id: str) -> dict:
        """Return copies of published onboard telemetry and actuator packets."""
        from sim.public_api.fsw_observations import flight_software_observations

        self._ensure_engine()
        return flight_software_observations(self._engine, str(object_id))

    def add_fsw_input_publisher(self, object_id: str, publisher: Callable[..., object]) -> None:
        """Attach a truth-free input source sampled at flight-software releases."""

        self._ensure_engine()
        assert self._engine is not None
        self._engine.add_fsw_input_publisher(str(object_id), publisher)

    def request_fsw_input_publisher_poll(self, object_id: str) -> None:
        """Request one attached input-publisher poll at the current simulation time."""

        self._ensure_engine()
        assert self._engine is not None
        self._engine.request_fsw_input_publisher_poll(str(object_id))

    def set_orbit_controller(self, object_id: str, controller: Any | None) -> None:
        """Attach a trusted Python orbit controller object or callable to a single-run session."""

        self._set_controller_override("orbit", object_id, None if controller is None else lambda: controller)

    def set_orbit_controller_factory(self, object_id: str, factory: ControllerFactory | None) -> None:
        """Attach a factory that creates a fresh orbit controller for each engine reset."""

        self._set_controller_override("orbit", object_id, factory)

    def set_attitude_controller(self, object_id: str, controller: Any | None) -> None:
        """Attach a trusted Python attitude controller object or callable to a single-run session."""

        self._set_controller_override("attitude", object_id, None if controller is None else lambda: controller)

    def set_attitude_controller_factory(self, object_id: str, factory: ControllerFactory | None) -> None:
        """Attach a factory that creates a fresh attitude controller for each engine reset."""

        self._set_controller_override("attitude", object_id, factory)

    def set_mission_strategy(self, object_id: str, strategy: Any | None) -> None:
        self._set_mission_override("strategy", object_id, None if strategy is None else lambda: strategy)

    def set_mission_strategy_factory(self, object_id: str, factory: ControllerFactory | None) -> None:
        self._set_mission_override("strategy", object_id, factory)

    def set_mission_execution(self, object_id: str, execution: Any | None) -> None:
        self._set_mission_override("execution", object_id, None if execution is None else lambda: execution)

    def set_mission_execution_factory(self, object_id: str, factory: ControllerFactory | None) -> None:
        self._set_mission_override("execution", object_id, factory)

    def _set_controller_override(
        self,
        controller_kind: str,
        object_id: str,
        factory: ControllerFactory | None,
    ) -> None:
        kind = str(controller_kind)
        oid = str(object_id)
        key = (kind, oid)
        if factory is None:
            self._controller_overrides.pop(key, None)
        elif not callable(factory):
            raise TypeError("Controller factory must be callable.")
        else:
            self._reject_satellite_v1_override(oid, f"{kind} controller")
            self._ensure_runtime_override_allowed("controller overrides")
            self._controller_overrides[key] = factory
        if self._engine is not None:
            self._apply_single_controller_override(kind, oid, factory)

    def _set_mission_override(
        self,
        mission_kind: str,
        object_id: str,
        factory: ControllerFactory | None,
    ) -> None:
        kind = str(mission_kind)
        oid = str(object_id)
        key = (kind, oid)
        if factory is None:
            self._mission_overrides.pop(key, None)
        elif not callable(factory):
            raise TypeError("Mission factory must be callable.")
        else:
            self._reject_satellite_v1_override(oid, f"mission {kind}")
            self._ensure_runtime_override_allowed("mission overrides")
            self._mission_overrides[key] = factory
        if self._engine is not None:
            self._apply_single_mission_override(kind, oid, factory)

    def _ensure_engine(self, *, step_callback: Any | None = None) -> None:
        if self._engine is not None:
            if step_callback is not None:
                self._engine.active_step_callback = step_callback
                emit = getattr(self._engine, "_emit_step_callback", None)
                if callable(emit):
                    emit(getattr(self._engine, "current_index", 0))
            return
        scenario = self._active_config.to_scenario_config()
        self._enforce_sealed_mode(self._active_config)
        self._validate_plugins_if_strict(scenario)
        self._engine = create_single_run_engine(
            scenario,
            step_callback=step_callback,
            history_mode=self._history_mode,
            initial_history_capacity=self._initial_history_capacity,
            max_history_samples=self._max_history_samples,
        )
        self._controller_originals.clear()
        self._mission_originals.clear()
        self._apply_runtime_overrides()

    def _shutdown_engine_workers(self, *, suppress_errors: bool = False) -> None:
        engine = self._engine
        executor = None if engine is None else getattr(engine, "object_step_executor", None)
        shutdown = None if executor is None else getattr(executor, "shutdown", None)
        if not callable(shutdown):
            return
        try:
            shutdown()
        except Exception:
            if not suppress_errors:
                raise

    def _apply_runtime_overrides(self) -> None:
        for (kind, object_id), factory in self._controller_overrides.items():
            self._apply_single_controller_override(kind, object_id, factory)
        for (kind, object_id), factory in self._mission_overrides.items():
            self._apply_single_mission_override(kind, object_id, factory)

    def _agent_for_override(self, object_id: str) -> Any:
        assert self._engine is not None
        agents = getattr(self._engine, "agents", {})
        oid = str(object_id)
        if oid not in agents:
            raise KeyError(f"No active object with id '{oid}' in this session.")
        return agents[oid]

    def _apply_single_controller_override(
        self,
        kind: str,
        object_id: str,
        factory: ControllerFactory | None,
    ) -> None:
        key = (str(kind), str(object_id))
        attr = "attitude_controller" if kind == "attitude" else "orbit_controller"
        if factory is None:
            if self._engine is not None and key in self._controller_originals:
                agent = self._agent_for_override(object_id)
                current = getattr(agent, attr, None)
                original = self._controller_originals.pop(key)
                if current is not None and hasattr(current, "base"):
                    current.base = original
                    if hasattr(current, "_last_eval_t_s"):
                        current._last_eval_t_s = None
                    if hasattr(current, "_last_cmd"):
                        current._last_cmd = Command.zero()
                else:
                    setattr(agent, attr, original)
            return
        agent = self._agent_for_override(object_id)
        if getattr(agent, "kind", None) == "satellite":
            raise RuntimeError(
                "GNC v1 runtime overrides are unavailable for satellites; select or provide a complete "
                "SatelliteFlightSoftware stack in objects.<id>.flight_software."
            )
        controller = _controller_object(factory(), command_kind=kind)
        current = getattr(agent, attr, None)
        if current is not None and hasattr(current, "base"):
            self._controller_originals.setdefault(key, current.base)
            current.base = controller
            if hasattr(current, "_last_eval_t_s"):
                current._last_eval_t_s = None
            if hasattr(current, "_last_cmd"):
                current._last_cmd = Command.zero()
        else:
            self._controller_originals.setdefault(key, current)
            setattr(agent, attr, controller)

    def _apply_single_mission_override(
        self,
        kind: str,
        object_id: str,
        factory: ControllerFactory | None,
    ) -> None:
        key = (str(kind), str(object_id))
        attr = "mission_execution" if kind == "execution" else "mission_strategy"
        if factory is None:
            if self._engine is not None and key in self._mission_originals:
                agent = self._agent_for_override(object_id)
                setattr(agent, attr, self._mission_originals.pop(key))
            return
        agent = self._agent_for_override(object_id)
        if getattr(agent, "kind", None) == "satellite":
            raise RuntimeError(
                "GNC v1 runtime overrides are unavailable for satellites; select or provide a complete "
                "SatelliteFlightSoftware stack in objects.<id>.flight_software."
            )
        self._mission_originals.setdefault(key, getattr(agent, attr, None))
        setattr(agent, attr, _mission_object(factory()))

    @staticmethod
    def _validate_plugins_if_strict(config: SimulationScenarioConfig) -> None:
        if not bool(config.simulator.plugin_validation.get("strict", True)):
            return
        errors = validate_scenario_plugins(config)
        if errors:
            msg = "Plugin validation failed:\n- " + "\n- ".join(errors)
            raise ValueError(msg)

    @staticmethod
    def _is_batch_analysis(config: SimulationScenarioConfig) -> bool:
        return bool(config.monte_carlo.enabled or config.analysis.enabled)

    @staticmethod
    def _run_batch_analysis(config: SimulationConfig) -> dict[str, Any]:
        return run_simulation_scenario(config.to_scenario_config(), source_path=config.source_path)

    def _enforce_sealed_mode(self, config: SimulationConfig) -> None:
        if self._sealed_policy is None:
            return
        errors = validate_sealed_mode(config.to_scenario_config(), self._sealed_policy)
        if errors:
            raise ValueError("Sealed mode validation failed:\n- " + "\n- ".join(errors))

    def _ensure_runtime_override_allowed(self, surface: str) -> None:
        if self._sealed_policy is not None:
            raise PermissionError(f"Sealed mode blocks Python API {surface}; express trusted behavior in scenario YAML.")

    def _reject_satellite_v1_override(self, object_id: str, surface: str) -> None:
        scenario = self._active_config.to_scenario_config()
        object_cfg = configured_objects(scenario).get(str(object_id))
        if object_cfg is not None and object_cfg.kind == "satellite":
            raise RuntimeError(
                f"Cannot set {surface} on satellite {object_id!r}: GNC v2 accepts only complete "
                "SatelliteFlightSoftware stacks through objects.<id>.flight_software."
            )


TrustedSimulationSession = SimulationSession


class HostedSimulationSession(SimulationSession):
    """Sealed-by-construction session for hosted or untrusted callers."""

    def __init__(
        self,
        config: ScenarioArtifact | SimulationConfig | SimulationScenarioConfig | dict[str, Any],
        *,
        sealed_policy: SealedModePolicy | None = None,
        history_mode: str = "full",
        initial_history_capacity: int = 4096,
        max_history_samples: int = 4096,
    ) -> None:
        super().__init__(
            config,
            sealed_mode=True,
            sealed_policy=sealed_policy,
            history_mode=history_mode,
            initial_history_capacity=initial_history_capacity,
            max_history_samples=max_history_samples,
        )

    @classmethod
    def from_config(cls, config, **kwargs) -> HostedSimulationSession:
        return cls(config, **kwargs)

    @classmethod
    def from_yaml(cls, path: str | Path, **kwargs) -> HostedSimulationSession:
        policy = ConfigPathPolicy.default(config_path=path, allow_config_dir_writes=False)
        return cls(SimulationConfig.from_yaml(path, path_policy=policy), **kwargs)
