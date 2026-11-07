from __future__ import annotations

import importlib
import multiprocessing as mp
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

_GYMNASIUM_IMPORT_ERROR: Exception | None = None

try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception as exc:  # pragma: no cover
    _GYMNASIUM_IMPORT_ERROR = exc

    class _FallbackEnv:
        metadata: dict[str, Any] = {}

        def reset(self, *, seed: int | None = None) -> None:
            _ = seed

    class _FallbackBox:
        def __init__(self, low: np.ndarray, high: np.ndarray, shape: tuple[int, ...], dtype: Any):
            self.low = np.array(low, dtype=dtype)
            self.high = np.array(high, dtype=dtype)
            self.shape = shape
            self.dtype = dtype

    class _FallbackSpaces:
        Box = _FallbackBox

    class _FallbackGym:
        Env = _FallbackEnv

    gym = _FallbackGym()
    spaces = _FallbackSpaces()

from sim.config import AlgorithmPointer, MonteCarloVariation, SimulationScenarioConfig, scenario_config_from_dict
from sim.dynamics.orbit.environment import EARTH_RADIUS_KM
from sim.flight_software import (
    ControlAxisSample,
    InputEvent,
    InputKind,
    PacketId,
    PilotInputPayload,
    Quality,
)
from sim.runtime_support import (
    _apply_chaser_relative_init_from_target,
    _build_knowledge_base,
    _create_satellite_runtime,
    _deep_set,
    _sample_variation,
)
from sim.single_run_support import _SatelliteStepper

_LEGACY_SATELLITE_GNC_FIELDS = (
    "orbit_control",
    "attitude_control",
    "mission_strategy",
    "mission_execution",
    "mission_objectives",
    "guidance_modifiers",
    "bridge",
)


def _prepare_rl_v2_scenario(
    scenario: dict[str, Any],
    *,
    controlled_ids: tuple[str, ...],
    action_fields_by_agent: dict[str, tuple[ActionField, ...]],
) -> None:
    dt_s = float(dict(scenario.get("simulator", {}) or {}).get("dt_s", 1.0) or 1.0)
    objects = dict(scenario.get("objects", {}) or {})
    for object_id, raw in objects.items():
        obj = raw
        kind = str(obj.get("kind", "satellite") or "satellite").strip().lower()
        if kind != "satellite":
            continue
        for field_name in _LEGACY_SATELLITE_GNC_FIELDS:
            obj.pop(field_name, None)
        if object_id not in controlled_ids:
            obj["flight_software"] = {
                "stack": "fsw.passive",
                "hardware_profile": "hardware.passive.v1",
                "task_period_s": dt_s,
            }
            continue
        fields = action_fields_by_agent.get(object_id, ())
        acceleration_bounds = [
            max(abs(float(field.low)), abs(float(field.high))) * 1.0e3
            for field in fields
            if "thrust_eci_km_s2" in field.key
        ]
        max_acceleration = max(acceleration_bounds, default=0.02)
        reference_id = "target" if object_id != "target" and "target" in objects else next(
            (candidate for candidate in objects if candidate != object_id),
            object_id,
        )
        obj["flight_software"] = {
            "stack": "fsw.game_pilot_reference",
            "hardware_profile": "game.ideal_wrench.v1",
            "task_period_s": dt_s,
            "params": {
                "control_mode": "direct_eci",
                "input_profile": f"rl.direct_eci.{object_id}",
                "reference_object_id": reference_id,
                "max_acceleration_m_s2": max_acceleration,
            },
        }


def _enqueue_rl_action(owner: Any, agent_id: str, action_values: dict[str, float]) -> None:
    agent = owner.agents[agent_id]
    runtime = agent.flight_software_runtime
    if runtime is None or not action_values:
        return
    adapter = owner.action_adapter if hasattr(owner, "action_adapter") else owner.action_adapters_by_agent.get(agent_id)
    if adapter is None:
        adapter = DirectActionAdapter()
    intent = adapter.adapt(action_values=action_values, env=owner) if hasattr(adapter, "adapt") else adapter(
        action_values=action_values, env=owner
    )
    acceleration = intent.get("thrust_eci_km_s2", intent.get("fallback_thrust_eci_km_s2", (0.0, 0.0, 0.0)))
    values = [0.0 if value is None else float(value) for value in list(acceleration)]
    if len(values) != 3:
        values = [0.0, 0.0, 0.0]
    max_acceleration = max(float(runtime.stack.config.max_acceleration_m_s2), 1.0e-15)
    axes = tuple(
        ControlAxisSample(name, float(np.clip(value * 1.0e3 / max_acceleration, -1.0, 1.0)))
        for name, value in zip(("translate_r", "translate_i", "translate_c"), values, strict=True)
    ) + (ControlAxisSample("throttle", 1.0),)
    at = runtime.clock_tag(int(round(owner.current_time_s * 1.0e9)))
    sequence = int(getattr(owner, "_rl_input_sequence", 0))
    owner._rl_input_sequence = sequence + 1
    runtime.enqueue(
        InputEvent(
            PacketId(f"rl/{agent_id}", runtime.boot_id, sequence),
            InputKind.PILOT_INPUT,
            at,
            at,
            Quality(),
            PilotInputPayload(runtime.stack.config.profile.profile_id, axes),
        )
    )


def _step_agents_with_production_runtime(
    owner: Any,
    action_values_by_agent: dict[str, dict[str, float]],
) -> None:
    assert owner.scenario_cfg is not None
    simulator = owner.scenario_cfg.simulator
    dynamics = dict(simulator.dynamics or {})
    orbit = dict(dynamics.get("orbit", {}) or {})
    attitude = dict(dynamics.get("attitude", {}) or {})
    owner.base_environment = dict(simulator.environment or {})
    owner.attitude_enabled = bool(attitude.get("enabled", True))
    orbit_substep = max(float(orbit.get("orbit_substep_s", simulator.dt_s) or simulator.dt_s), 1.0e-9)
    attitude_substep = max(float(attitude.get("attitude_substep_s", simulator.dt_s) or simulator.dt_s), 1.0e-9)
    owner.sim_substep_s = min(orbit_substep, attitude_substep) if owner.attitude_enabled else orbit_substep
    owner.zero3 = np.zeros(3, dtype=float)
    if not hasattr(owner, "command_decision_hist"):
        owner.command_decision_hist = defaultdict(list)
    world_truth = {agent_id: agent.truth for agent_id, agent in owner.agents.items() if agent.active}
    stepper = _SatelliteStepper(owner)
    t_next = owner.current_time_s + float(simulator.dt_s)
    for agent_id, agent in owner.agents.items():
        if not agent.active:
            continue
        _enqueue_rl_action(owner, agent_id, action_values_by_agent.get(agent_id, {}))
        result = stepper.step(
            aid=agent_id,
            agent=agent,
            initial_truth=agent.truth,
            world_truth_decision=world_truth,
            t_s=owner.current_time_s,
            t_next=t_next,
            sample_index=owner.step_count + 1,
        )
        agent.truth = result.truth


class SupportsActionAdapter(Protocol):
    def adapt(self, *, action_values: dict[str, float], env: GymSimulationEnv) -> dict[str, Any]: ...


class SupportsReward(Protocol):
    def compute_reward(
        self,
        *,
        env: GymSimulationEnv,
        previous_snapshot: dict[str, Any] | None,
        snapshot: dict[str, Any],
        action_values: dict[str, float],
        terminated: bool,
        truncated: bool,
    ) -> float: ...


class SupportsTermination(Protocol):
    def check_termination(
        self,
        *,
        env: GymSimulationEnv,
        previous_snapshot: dict[str, Any] | None,
        snapshot: dict[str, Any],
        action_values: dict[str, float],
    ) -> tuple[bool, bool, dict[str, Any]]: ...


@dataclass(frozen=True)
class ObservationField:
    path: str
    scale: float = 1.0


@dataclass(frozen=True)
class ActionField:
    key: str
    low: float
    high: float


@dataclass(frozen=True)
class GymEnvConfig:
    scenario: SimulationScenarioConfig | dict[str, Any]
    controlled_agent_id: str = "chaser"
    observation_fields: tuple[ObservationField, ...] = ()
    action_fields: tuple[ActionField, ...] = ()
    episode_variations: tuple[MonteCarloVariation, ...] = ()
    action_adapter: Any | None = None
    reward_fn: Any | None = None
    termination_fn: Any | None = None
    max_steps: int | None = None


@dataclass(frozen=True)
class VectorEnvConfig:
    env_cfg: GymEnvConfig
    num_envs: int
    parallel: bool = False
    auto_reset: bool = False


@dataclass(frozen=True)
class MultiAgentEnvConfig:
    scenario: SimulationScenarioConfig | dict[str, Any]
    controlled_agent_ids: tuple[str, ...] = ("chaser", "target")
    observation_fields_by_agent: dict[str, tuple[ObservationField, ...]] = field(default_factory=dict)
    action_fields_by_agent: dict[str, tuple[ActionField, ...]] = field(default_factory=dict)
    episode_variations: tuple[MonteCarloVariation, ...] = ()
    action_adapters_by_agent: dict[str, Any] = field(default_factory=dict)
    reward_fns_by_agent: dict[str, Any] = field(default_factory=dict)
    termination_fn: Any | None = None
    max_steps: int | None = None


def default_observation_fields_for_agent(agent_id: str) -> tuple[ObservationField, ...]:
    """Belief-owned default observation vector for autonomy policies.

    Simulator truth remains available through explicit `truth.*` paths, but those
    paths are treated as oracle-baseline observations by the policy-card helpers.
    """

    return (
        ObservationField(path=f"belief.{agent_id}.state[0]"),
        ObservationField(path=f"belief.{agent_id}.state[1]"),
        ObservationField(path=f"belief.{agent_id}.state[2]"),
        ObservationField(path=f"belief.{agent_id}.state[3]"),
        ObservationField(path=f"belief.{agent_id}.state[4]"),
        ObservationField(path=f"belief.{agent_id}.state[5]"),
    )


@dataclass(frozen=True)
class EnvFactory:
    env_cfg: GymEnvConfig

    def __call__(self) -> GymSimulationEnv:
        return GymSimulationEnv(self.env_cfg)


def _pointer_from_any(spec: Any) -> AlgorithmPointer | None:
    if spec is None or callable(spec):
        return None
    if isinstance(spec, AlgorithmPointer):
        return spec
    if isinstance(spec, str):
        module_name, _, attr = spec.rpartition(":")
        if module_name and attr:
            return AlgorithmPointer(module=module_name, function=attr)
        return AlgorithmPointer(module=spec)
    if isinstance(spec, dict):
        return AlgorithmPointer(
            module=spec.get("module"),
            class_name=spec.get("class_name"),
            function=spec.get("function"),
            params=dict(spec.get("params", {}) or {}),
        )
    return None


def _load_callable(spec: Any) -> Any:
    if spec is None or callable(spec):
        return spec
    pointer = _pointer_from_any(spec)
    if pointer is None or pointer.module is None:
        return None
    mod = importlib.import_module(pointer.module)
    if pointer.class_name:
        cls = getattr(mod, pointer.class_name)
        return cls(**dict(pointer.params or {}))
    if pointer.function:
        fn = getattr(mod, pointer.function)
        params = dict(pointer.params or {})
        if params:
            return lambda **kwargs: fn(**params, **kwargs)
        return fn
    return mod


def _path_tokens(path: str) -> list[str | int]:
    tokens: list[str | int] = []
    for part in str(path).split("."):
        cur = part
        while "[" in cur and cur.endswith("]"):
            base, idx_txt = cur[:-1].split("[", 1)
            if base:
                tokens.append(base)
            tokens.append(int(idx_txt))
            cur = ""
        if cur:
            tokens.append(cur)
    return tokens


def _lookup_path(root: Any, path: str) -> Any:
    cur = root
    for token in _path_tokens(path):
        if isinstance(token, int):
            cur = cur[token]
        elif isinstance(cur, dict):
            cur = cur[token]
        else:
            cur = getattr(cur, token)
    return cur


def _assign_path(root: dict[str, Any], path: str, value: Any) -> None:
    tokens = _path_tokens(path)
    cur: Any = root
    for i, token in enumerate(tokens):
        last = i == len(tokens) - 1
        nxt = None if last else tokens[i + 1]
        if isinstance(token, str):
            if last:
                cur[token] = value
                return
            if token not in cur or cur[token] is None:
                cur[token] = [] if isinstance(nxt, int) else {}
            cur = cur[token]
            continue
        while len(cur) <= token:
            cur.append(None)
        if last:
            cur[token] = value
            return
        if cur[token] is None:
            cur[token] = [] if isinstance(nxt, int) else {}
        cur = cur[token]


def _enabled_satellite_ids_from_scenario_dict(scenario_dict: dict[str, Any]) -> tuple[str, ...]:
    enabled: list[str] = []
    objects = dict(scenario_dict.get("objects", {}) or {})
    candidate_ids = list(objects) if objects else ["chaser", "target"]
    for agent_id in candidate_ids:
        agent_cfg = dict((objects.get(agent_id) if objects else scenario_dict.get(agent_id, {})) or {})
        # Scenario object configs default ``enabled`` to true in the canonical
        # loader.  Keep the pre-parse observation probe aligned with that
        # contract so raw dict scenarios do not lose their default-enabled
        # satellites before validation/building.
        if bool(agent_cfg.get("enabled", True)):
            kind = str(agent_cfg.get("kind", "satellite") or "satellite").strip().lower()
            if kind != "rocket":
                enabled.append(str(agent_id))
    return tuple(enabled)


def _observation_probe_from_scenario_dict(scenario_dict: dict[str, Any]) -> dict[str, Any]:
    enabled_ids = _enabled_satellite_ids_from_scenario_dict(scenario_dict)
    truth = {
        agent_id: {
            "position_eci_km": np.zeros(3, dtype=float),
            "velocity_eci_km_s": np.zeros(3, dtype=float),
            "attitude_quat_bn": np.zeros(4, dtype=float),
            "angular_rate_body_rad_s": np.zeros(3, dtype=float),
            "mass_kg": 0.0,
            "t_s": 0.0,
        }
        for agent_id in enabled_ids
    }
    belief = {
        agent_id: {
            "state": np.zeros(13, dtype=float),
            "last_update_t_s": 0.0,
        }
        for agent_id in enabled_ids
    }
    knowledge: dict[str, dict[str, dict[str, Any]]] = {}
    objects = dict(scenario_dict.get("objects", {}) or {})
    for observer_id in enabled_ids:
        agent_cfg = dict((objects.get(observer_id) if objects else scenario_dict.get(observer_id, {})) or {})
        knowledge_cfg = dict(agent_cfg.get("knowledge", {}) or {})
        targets = []
        for target_id in list(knowledge_cfg.get("targets", []) or []):
            target = str(target_id)
            if target != observer_id and target in enabled_ids:
                targets.append(target)
        if targets:
            knowledge[observer_id] = {
                target_id: {
                    "state": np.zeros(6, dtype=float),
                    "last_update_t_s": 0.0,
                }
                for target_id in sorted(set(targets))
            }
    return {
        "truth": truth,
        "belief": belief,
        "knowledge": knowledge,
        "metrics": {"range_km": 0.0, "closest_range_km": 0.0, "step": 0.0, "time_s": 0.0},
        "sampled_parameters": {},
    }


def _lookup_observation_value(snapshot: dict[str, Any], path: str, probe: dict[str, Any]) -> np.ndarray:
    try:
        value = _lookup_path(snapshot, path)
    except (AttributeError, IndexError, KeyError, TypeError):
        value = _lookup_path(probe, path)
    return np.array(value, dtype=float).reshape(-1)


def _observation_dim_from_fields(fields: tuple[ObservationField, ...], probe: dict[str, Any]) -> int:
    dim = 0
    for obs_field in fields:
        try:
            value = np.array(_lookup_path(probe, obs_field.path), dtype=float).reshape(-1)
        except (AttributeError, IndexError, KeyError, TypeError) as exc:
            raise ValueError(
                f"Observation field '{obs_field.path}' is unavailable for the enabled agents or configured knowledge targets."
            ) from exc
        dim += int(value.size)
    return dim


def _snapshot_truth(truth: Any) -> dict[str, Any]:
    return {
        "position_eci_km": np.array(truth.position_eci_km, dtype=float),
        "velocity_eci_km_s": np.array(truth.velocity_eci_km_s, dtype=float),
        "attitude_quat_bn": np.array(truth.attitude_quat_bn, dtype=float),
        "angular_rate_body_rad_s": np.array(truth.angular_rate_body_rad_s, dtype=float),
        "mass_kg": float(truth.mass_kg),
        "t_s": float(truth.t_s),
    }


def _sample_episode_variations(
    scenario_dict: dict[str, Any],
    variations: tuple[MonteCarloVariation, ...],
    rng: np.random.Generator,
) -> dict[str, Any]:
    sampled: dict[str, Any] = {}
    for variation in variations:
        value = _sample_variation(variation, rng)
        _deep_set(scenario_dict, variation.parameter_path, value)
        sampled[variation.parameter_path] = value
    return sampled


@dataclass
class DirectActionAdapter:
    def adapt(self, *, action_values: dict[str, float], env: GymSimulationEnv) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in action_values.items():
            _assign_path(out, key, float(value))
        return out


@dataclass
class ThrustVectorToPointingAdapter:
    direction_key_prefix: str = "thrust_direction_eci"
    throttle_key: str = "throttle"
    align_to_thrust: bool = True

    def adapt(self, *, action_values: dict[str, float], env: GymSimulationEnv) -> dict[str, Any]:
        direction = np.array(
            [
                float(action_values.get(f"{self.direction_key_prefix}[0]", 0.0)),
                float(action_values.get(f"{self.direction_key_prefix}[1]", 0.0)),
                float(action_values.get(f"{self.direction_key_prefix}[2]", 0.0)),
            ],
            dtype=float,
        )
        throttle = float(np.clip(action_values.get(self.throttle_key, 0.0), 0.0, 1.0))
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-12 or throttle <= 0.0:
            thrust = np.zeros(3, dtype=float)
        else:
            direction /= norm
            controlled = env.agents[env.controlled_agent_id]
            max_accel = float(controlled.limits["orbital"].max_accel_km_s2)
            thrust = direction * throttle * max_accel
        return {
            "fallback_thrust_eci_km_s2": thrust.tolist(),
            "align_to_thrust": bool(self.align_to_thrust),
            "mission_mode": {"policy_adapter": "thrust_vector_to_pointing"},
        }


@dataclass
class RelativeDistanceReward:
    controlled_agent_id: str = "chaser"
    target_id: str = "target"
    scale: float = 1000.0
    terminal_bonus: float = 0.0
    sign: float = 1.0

    def compute_reward(
        self,
        *,
        env: GymSimulationEnv,
        previous_snapshot: dict[str, Any] | None,
        snapshot: dict[str, Any],
        action_values: dict[str, float],
        terminated: bool,
        truncated: bool,
    ) -> float:
        _ = (env, action_values, truncated)
        current = float(snapshot["metrics"].get("range_km", np.nan))
        if previous_snapshot is None:
            prev = current
        else:
            prev = float(previous_snapshot["metrics"].get("range_km", current))
        reward = float((prev - current) * self.scale * self.sign)
        if terminated:
            reward += float(self.terminal_bonus * self.sign)
        return reward


@dataclass
class RangeTermination:
    controlled_agent_id: str = "chaser"
    target_id: str = "target"
    capture_radius_km: float = 0.01

    def check_termination(
        self,
        *,
        env: GymSimulationEnv,
        previous_snapshot: dict[str, Any] | None,
        snapshot: dict[str, Any],
        action_values: dict[str, float],
    ) -> tuple[bool, bool, dict[str, Any]]:
        _ = (env, previous_snapshot, action_values)
        range_km = float(snapshot["metrics"].get("range_km", np.inf))
        capture = bool(range_km <= self.capture_radius_km)
        return capture, False, {"capture": capture}


class GymSimulationEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, cfg: GymEnvConfig):
        super().__init__()
        self.cfg = cfg
        self.controlled_agent_id = str(cfg.controlled_agent_id)
        self.base_scenario_dict = (
            cfg.scenario.to_dict()
            if isinstance(cfg.scenario, SimulationScenarioConfig)
            else deepcopy(dict(cfg.scenario))
        )
        self.rng = np.random.default_rng(int(self.base_scenario_dict.get("metadata", {}).get("seed", 0)))
        self.action_adapter = _load_callable(cfg.action_adapter) or DirectActionAdapter()
        self.reward_fn = _load_callable(cfg.reward_fn) or RelativeDistanceReward(
            controlled_agent_id=self.controlled_agent_id
        )
        self.termination_fn = _load_callable(cfg.termination_fn)
        self.agents: dict[str, Any] = {}
        self.scenario_cfg: SimulationScenarioConfig | None = None
        self.sampled_parameters: dict[str, Any] = {}
        self.step_count = 0
        self.max_steps = 1
        self.current_time_s = 0.0
        self.closest_range_km = np.inf
        self.last_snapshot: dict[str, Any] | None = None
        self._observation_probe = _observation_probe_from_scenario_dict(self.base_scenario_dict)

        obs_fields = tuple(cfg.observation_fields) or default_observation_fields_for_agent(self.controlled_agent_id)
        self.observation_fields = obs_fields
        low = np.array([float(field.low) for field in cfg.action_fields], dtype=np.float32)
        high = np.array([float(field.high) for field in cfg.action_fields], dtype=np.float32)
        if low.size == 0:
            low = np.zeros(0, dtype=np.float32)
            high = np.zeros(0, dtype=np.float32)
        self.action_fields = tuple(cfg.action_fields)
        self.action_space = spaces.Box(low=low, high=high, shape=(low.size,), dtype=np.float32)
        obs_dim = _observation_dim_from_fields(self.observation_fields, self._observation_probe)
        self.observation_space = spaces.Box(
            low=np.full(obs_dim, -np.inf, dtype=np.float32),
            high=np.full(obs_dim, np.inf, dtype=np.float32),
            shape=(obs_dim,),
            dtype=np.float32,
        )

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(int(seed))
        scenario_dict = deepcopy(self.base_scenario_dict)
        if seed is not None:
            scenario_dict.setdefault("metadata", {})["seed"] = int(seed)
        options = dict(options or {})
        self.sampled_parameters = _sample_episode_variations(
            scenario_dict, tuple(self.cfg.episode_variations), self.rng
        )
        for path, value in dict(options.get("override_parameters", {}) or {}).items():
            _deep_set(scenario_dict, path, value)
            self.sampled_parameters[path] = value
        _prepare_rl_v2_scenario(
            scenario_dict,
            controlled_ids=(self.controlled_agent_id,),
            action_fields_by_agent={self.controlled_agent_id: self.action_fields},
        )
        self.scenario_cfg = scenario_config_from_dict(scenario_dict)
        runtime_probe = _observation_probe_from_scenario_dict(scenario_dict)
        obs_dim = _observation_dim_from_fields(self.observation_fields, runtime_probe)
        if obs_dim != int(self.observation_space.shape[0]):
            raise ValueError(
                f"Observation fields resolved to dimension {obs_dim}, expected {self.observation_space.shape[0]}."
            )
        self._observation_probe = runtime_probe
        self._build_agents()
        dt = float(self.scenario_cfg.simulator.dt_s)
        duration_s = float(self.scenario_cfg.simulator.duration_s)
        configured_steps = max(1, int(np.floor(duration_s / dt)))
        self.max_steps = int(self.cfg.max_steps) if self.cfg.max_steps is not None else configured_steps
        self.step_count = 0
        self.current_time_s = 0.0
        self.closest_range_km = self._compute_range_km()
        self.last_snapshot = self._snapshot()
        return self._observation_from_snapshot(self.last_snapshot), self._info_dict(self.last_snapshot)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self.scenario_cfg is None:
            raise RuntimeError("Environment must be reset before step().")
        action_arr = np.array(action, dtype=np.float32).reshape(-1)
        if action_arr.size != len(self.action_fields):
            raise ValueError(f"Expected action of size {len(self.action_fields)}, got {action_arr.size}.")
        action_values = {
            field.key: float(np.clip(action_arr[i], field.low, field.high))
            for i, field in enumerate(self.action_fields)
        }
        previous_snapshot = self.last_snapshot
        self._step_all_agents(action_values)
        self.step_count += 1
        self.current_time_s = float(self.scenario_cfg.simulator.dt_s) * self.step_count
        snapshot = self._snapshot()
        terminated, truncated, term_info = self._check_done(previous_snapshot, snapshot, action_values)
        reward = self._compute_reward(previous_snapshot, snapshot, action_values, terminated, truncated)
        self.last_snapshot = snapshot
        info = self._info_dict(snapshot)
        info.update(term_info)
        return self._observation_from_snapshot(snapshot), float(reward), bool(terminated), bool(truncated), info

    def _build_agents(self) -> None:
        if self.scenario_cfg is None:
            raise RuntimeError("Scenario config not initialized.")
        if self.scenario_cfg.rocket.enabled:
            raise NotImplementedError("GymSimulationEnv currently supports satellite scenarios only.")
        root_seed = int(self.scenario_cfg.metadata.get("seed", 0))
        rng = np.random.default_rng(root_seed)
        agents: dict[str, Any] = {}
        if self.scenario_cfg.target.enabled:
            agents["target"] = _create_satellite_runtime(
                "target",
                self.scenario_cfg.target,
                self.scenario_cfg,
                np.random.default_rng(int(rng.integers(0, 2**31 - 1))),
            )
        if self.scenario_cfg.chaser.enabled:
            agents["chaser"] = _create_satellite_runtime(
                "chaser",
                self.scenario_cfg.chaser,
                self.scenario_cfg,
                np.random.default_rng(int(rng.integers(0, 2**31 - 1))),
            )
        if "chaser" in agents and "target" in agents:
            _apply_chaser_relative_init_from_target(
                chaser=agents["chaser"],
                target=agents["target"],
                initial_state=dict(self.scenario_cfg.chaser.initial_state or {}),
            )
        for aid, agent in agents.items():
            cfg_src = self.scenario_cfg.chaser if aid == "chaser" else self.scenario_cfg.target
            agent.knowledge_base = _build_knowledge_base(
                observer_id=aid,
                agent_cfg=cfg_src,
                dt_s=float(self.scenario_cfg.simulator.dt_s),
                rng=np.random.default_rng(int(rng.integers(0, 2**31 - 1))),
            )
        if self.controlled_agent_id not in agents:
            raise ValueError(f"Controlled agent '{self.controlled_agent_id}' is not enabled in the scenario.")
        self.agents = agents

    def _step_all_agents(self, action_values: dict[str, float]) -> None:
        _step_agents_with_production_runtime(self, {self.controlled_agent_id: action_values})
        return
    def _adapt_action(self, action_values: dict[str, float]) -> dict[str, Any]:
        adapter = self.action_adapter
        if hasattr(adapter, "adapt"):
            return adapter.adapt(action_values=action_values, env=self)
        return adapter(action_values=action_values, env=self)

    def _compute_reward(
        self,
        previous_snapshot: dict[str, Any] | None,
        snapshot: dict[str, Any],
        action_values: dict[str, float],
        terminated: bool,
        truncated: bool,
    ) -> float:
        fn = self.reward_fn
        if hasattr(fn, "compute_reward"):
            return float(
                fn.compute_reward(
                    env=self,
                    previous_snapshot=previous_snapshot,
                    snapshot=snapshot,
                    action_values=action_values,
                    terminated=terminated,
                    truncated=truncated,
                )
            )
        return float(
            fn(
                env=self,
                previous_snapshot=previous_snapshot,
                snapshot=snapshot,
                action_values=action_values,
                terminated=terminated,
                truncated=truncated,
            )
        )

    def _check_done(
        self,
        previous_snapshot: dict[str, Any] | None,
        snapshot: dict[str, Any],
        action_values: dict[str, float],
    ) -> tuple[bool, bool, dict[str, Any]]:
        terminated = False
        truncated = bool(self.step_count >= self.max_steps)
        info: dict[str, Any] = {}
        for aid, agent in self.agents.items():
            radius = float(np.linalg.norm(agent.truth.position_eci_km))
            if radius <= EARTH_RADIUS_KM:
                terminated = True
                info["termination_reason"] = "earth_impact"
                info["termination_object_id"] = aid
                break
        if self.termination_fn is not None:
            checker = self.termination_fn
            if hasattr(checker, "check_termination"):
                extra_terminated, extra_truncated, extra_info = checker.check_termination(
                    env=self,
                    previous_snapshot=previous_snapshot,
                    snapshot=snapshot,
                    action_values=action_values,
                )
            else:
                extra_terminated, extra_truncated, extra_info = checker(
                    env=self,
                    previous_snapshot=previous_snapshot,
                    snapshot=snapshot,
                    action_values=action_values,
                )
            terminated = bool(terminated or extra_terminated)
            truncated = bool(truncated or extra_truncated)
            info.update(dict(extra_info or {}))
        return terminated, truncated, info

    def _compute_range_km(self) -> float:
        controlled = self.agents.get(self.controlled_agent_id)
        target = self.agents.get("target")
        if controlled is None or target is None:
            return float("nan")
        return float(np.linalg.norm(controlled.truth.position_eci_km - target.truth.position_eci_km))

    def _snapshot(self) -> dict[str, Any]:
        truth = {aid: _snapshot_truth(agent.truth) for aid, agent in self.agents.items() if agent.active}
        belief = {
            aid: {
                "state": np.array(agent.belief.state, dtype=float),
                "last_update_t_s": float(agent.belief.last_update_t_s),
            }
            for aid, agent in self.agents.items()
            if agent.active and agent.belief is not None
        }
        knowledge: dict[str, Any] = {}
        for aid, agent in self.agents.items():
            if not agent.active or agent.knowledge_base is None:
                continue
            knowledge[aid] = {}
            for target_id, kb in agent.knowledge_base.snapshot().items():
                knowledge[aid][target_id] = {
                    "state": np.array(kb.state, dtype=float),
                    "last_update_t_s": float(kb.last_update_t_s),
                }
        range_km = self._compute_range_km()
        if np.isfinite(range_km):
            self.closest_range_km = min(float(self.closest_range_km), float(range_km))
        return {
            "truth": truth,
            "belief": belief,
            "knowledge": knowledge,
            "metrics": {
                "range_km": float(range_km),
                "closest_range_km": float(self.closest_range_km),
                "step": int(self.step_count),
                "time_s": float(self.current_time_s),
            },
            "sampled_parameters": dict(self.sampled_parameters),
        }

    def _observation_from_snapshot(self, snapshot: dict[str, Any]) -> np.ndarray:
        parts: list[np.ndarray] = []
        for obs_field in self.observation_fields:
            value = _lookup_observation_value(snapshot, obs_field.path, self._observation_probe)
            parts.append(value * float(obs_field.scale))
        if not parts:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(parts).astype(np.float32)

    def _info_dict(self, snapshot: dict[str, Any]) -> dict[str, Any]:
        return {
            "controlled_agent_id": self.controlled_agent_id,
            "sampled_parameters": dict(self.sampled_parameters),
            "metrics": dict(snapshot["metrics"]),
        }


class MultiAgentSimulationEnv:
    metadata = {"render_modes": []}

    def __init__(self, cfg: MultiAgentEnvConfig):
        self.cfg = cfg
        self.controlled_agent_ids = tuple(str(agent_id) for agent_id in cfg.controlled_agent_ids)
        if not self.controlled_agent_ids:
            raise ValueError("controlled_agent_ids must not be empty.")
        self.base_scenario_dict = (
            cfg.scenario.to_dict()
            if isinstance(cfg.scenario, SimulationScenarioConfig)
            else deepcopy(dict(cfg.scenario))
        )
        self.rng = np.random.default_rng(int(self.base_scenario_dict.get("metadata", {}).get("seed", 0)))
        self.observation_fields_by_agent = {
            str(agent_id): tuple(fields) for agent_id, fields in dict(cfg.observation_fields_by_agent or {}).items()
        }
        self.action_fields_by_agent = {
            str(agent_id): tuple(fields) for agent_id, fields in dict(cfg.action_fields_by_agent or {}).items()
        }
        self.action_adapters_by_agent = {
            str(agent_id): (_load_callable(adapter) or DirectActionAdapter())
            for agent_id, adapter in dict(cfg.action_adapters_by_agent or {}).items()
        }
        self.reward_fns_by_agent = {
            str(agent_id): _load_callable(fn) for agent_id, fn in dict(cfg.reward_fns_by_agent or {}).items()
        }
        self.termination_fn = _load_callable(cfg.termination_fn)
        self.scenario_cfg: SimulationScenarioConfig | None = None
        self.agents: dict[str, Any] = {}
        self.sampled_parameters: dict[str, Any] = {}
        self.step_count = 0
        self.max_steps = 1
        self.current_time_s = 0.0
        self.closest_range_km = np.inf
        self.last_snapshot: dict[str, Any] | None = None
        self._observation_probe = _observation_probe_from_scenario_dict(self.base_scenario_dict)
        self.action_spaces = {}
        self.observation_spaces = {}
        for agent_id in self.controlled_agent_ids:
            action_fields = self.action_fields_by_agent.get(agent_id, ())
            low = np.array([float(field.low) for field in action_fields], dtype=np.float32)
            high = np.array([float(field.high) for field in action_fields], dtype=np.float32)
            if low.size == 0:
                low = np.zeros(0, dtype=np.float32)
                high = np.zeros(0, dtype=np.float32)
            self.action_spaces[agent_id] = spaces.Box(low=low, high=high, shape=(low.size,), dtype=np.float32)
            obs_dim = _observation_dim_from_fields(self._fields_for_agent(agent_id), self._observation_probe)
            self.observation_spaces[agent_id] = spaces.Box(
                low=np.full(obs_dim, -np.inf, dtype=np.float32),
                high=np.full(obs_dim, np.inf, dtype=np.float32),
                shape=(obs_dim,),
                dtype=np.float32,
            )

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
        if seed is not None:
            self.rng = np.random.default_rng(int(seed))
        scenario_dict = deepcopy(self.base_scenario_dict)
        if seed is not None:
            scenario_dict.setdefault("metadata", {})["seed"] = int(seed)
        options = dict(options or {})
        self.sampled_parameters = _sample_episode_variations(
            scenario_dict, tuple(self.cfg.episode_variations), self.rng
        )
        for path, value in dict(options.get("override_parameters", {}) or {}).items():
            _deep_set(scenario_dict, path, value)
            self.sampled_parameters[path] = value
        _prepare_rl_v2_scenario(
            scenario_dict,
            controlled_ids=self.controlled_agent_ids,
            action_fields_by_agent=self.action_fields_by_agent,
        )
        self.scenario_cfg = scenario_config_from_dict(scenario_dict)
        runtime_probe = _observation_probe_from_scenario_dict(scenario_dict)
        for agent_id in self.controlled_agent_ids:
            obs_dim = _observation_dim_from_fields(self._fields_for_agent(agent_id), runtime_probe)
            expected = int(self.observation_spaces[agent_id].shape[0])
            if obs_dim != expected:
                raise ValueError(
                    f"Observation fields for '{agent_id}' resolved to dimension {obs_dim}, expected {expected}."
                )
        self._observation_probe = runtime_probe
        self._build_agents()
        dt = float(self.scenario_cfg.simulator.dt_s)
        duration_s = float(self.scenario_cfg.simulator.duration_s)
        configured_steps = max(1, int(np.floor(duration_s / dt)))
        self.max_steps = int(self.cfg.max_steps) if self.cfg.max_steps is not None else configured_steps
        self.step_count = 0
        self.current_time_s = 0.0
        self.closest_range_km = self._compute_range_km()
        self.last_snapshot = self._snapshot()
        observations = {
            agent_id: self._observation_from_snapshot(self.last_snapshot, self._fields_for_agent(agent_id))
            for agent_id in self.controlled_agent_ids
        }
        infos = {agent_id: self._info_dict(self.last_snapshot, agent_id) for agent_id in self.controlled_agent_ids}
        return observations, infos

    def step(
        self,
        actions: dict[str, np.ndarray | list[float]],
    ) -> tuple[dict[str, np.ndarray], dict[str, float], dict[str, bool], dict[str, bool], dict[str, dict[str, Any]]]:
        if self.scenario_cfg is None:
            raise RuntimeError("Environment must be reset before step().")
        action_values_by_agent = {
            agent_id: self._parse_agent_action(agent_id, dict(actions or {}).get(agent_id))
            for agent_id in self.controlled_agent_ids
        }
        previous_snapshot = self.last_snapshot
        self._step_all_agents_multi(action_values_by_agent)
        self.step_count += 1
        self.current_time_s = float(self.scenario_cfg.simulator.dt_s) * self.step_count
        snapshot = self._snapshot()
        terminated, truncated, term_info = self._check_done_multi(previous_snapshot, snapshot, action_values_by_agent)
        rewards = {
            agent_id: self._compute_reward_multi(
                agent_id,
                previous_snapshot,
                snapshot,
                action_values_by_agent[agent_id],
                terminated[agent_id],
                truncated[agent_id],
            )
            for agent_id in self.controlled_agent_ids
        }
        self.last_snapshot = snapshot
        observations = {
            agent_id: self._observation_from_snapshot(snapshot, self._fields_for_agent(agent_id))
            for agent_id in self.controlled_agent_ids
        }
        infos = {agent_id: self._info_dict(snapshot, agent_id) for agent_id in self.controlled_agent_ids}
        for agent_id in self.controlled_agent_ids:
            infos[agent_id].update(dict(term_info.get(agent_id, {})))
        return observations, rewards, terminated, truncated, infos

    def _fields_for_agent(self, agent_id: str) -> tuple[ObservationField, ...]:
        fields = self.observation_fields_by_agent.get(agent_id)
        if fields:
            return fields
        return default_observation_fields_for_agent(agent_id)

    def _parse_agent_action(self, agent_id: str, action: np.ndarray | list[float] | None) -> dict[str, float]:
        action_fields = self.action_fields_by_agent.get(agent_id, ())
        if not action_fields:
            return {}
        if action is None:
            action_arr = np.zeros(len(action_fields), dtype=np.float32)
        else:
            action_arr = np.array(action, dtype=np.float32).reshape(-1)
        if action_arr.size != len(action_fields):
            raise ValueError(f"Expected action of size {len(action_fields)} for '{agent_id}', got {action_arr.size}.")
        return {
            field.key: float(np.clip(action_arr[i], field.low, field.high)) for i, field in enumerate(action_fields)
        }

    def _build_agents(self) -> None:
        if self.scenario_cfg is None:
            raise RuntimeError("Scenario config not initialized.")
        if self.scenario_cfg.rocket.enabled:
            raise NotImplementedError("MultiAgentSimulationEnv currently supports satellite scenarios only.")
        root_seed = int(self.scenario_cfg.metadata.get("seed", 0))
        rng = np.random.default_rng(root_seed)
        agents: dict[str, Any] = {}
        if self.scenario_cfg.target.enabled:
            agents["target"] = _create_satellite_runtime(
                "target",
                self.scenario_cfg.target,
                self.scenario_cfg,
                np.random.default_rng(int(rng.integers(0, 2**31 - 1))),
            )
        if self.scenario_cfg.chaser.enabled:
            agents["chaser"] = _create_satellite_runtime(
                "chaser",
                self.scenario_cfg.chaser,
                self.scenario_cfg,
                np.random.default_rng(int(rng.integers(0, 2**31 - 1))),
            )
        if "chaser" in agents and "target" in agents:
            _apply_chaser_relative_init_from_target(
                chaser=agents["chaser"],
                target=agents["target"],
                initial_state=dict(self.scenario_cfg.chaser.initial_state or {}),
            )
        for aid, agent in agents.items():
            cfg_src = self.scenario_cfg.chaser if aid == "chaser" else self.scenario_cfg.target
            agent.knowledge_base = _build_knowledge_base(
                observer_id=aid,
                agent_cfg=cfg_src,
                dt_s=float(self.scenario_cfg.simulator.dt_s),
                rng=np.random.default_rng(int(rng.integers(0, 2**31 - 1))),
            )
        missing = [agent_id for agent_id in self.controlled_agent_ids if agent_id not in agents]
        if missing:
            raise ValueError(f"Controlled agent(s) not enabled in scenario: {missing}.")
        self.agents = agents

    def _step_all_agents_multi(self, action_values_by_agent: dict[str, dict[str, float]]) -> None:
        _step_agents_with_production_runtime(self, action_values_by_agent)
        return
    def _check_done_multi(
        self,
        previous_snapshot: dict[str, Any] | None,
        snapshot: dict[str, Any],
        action_values_by_agent: dict[str, dict[str, float]],
    ) -> tuple[dict[str, bool], dict[str, bool], dict[str, dict[str, Any]]]:
        terminations = {agent_id: False for agent_id in self.controlled_agent_ids}
        truncations = {agent_id: bool(self.step_count >= self.max_steps) for agent_id in self.controlled_agent_ids}
        infos = {agent_id: {} for agent_id in self.controlled_agent_ids}
        impacted_agents = []
        for aid, agent in self.agents.items():
            radius = float(np.linalg.norm(agent.truth.position_eci_km))
            if radius <= EARTH_RADIUS_KM:
                impacted_agents.append(aid)
        if impacted_agents:
            for agent_id in self.controlled_agent_ids:
                terminations[agent_id] = True
                infos[agent_id]["termination_reason"] = "earth_impact"
                infos[agent_id]["termination_object_ids"] = list(impacted_agents)
        if self.termination_fn is not None:
            checker = self.termination_fn
            if hasattr(checker, "check_termination"):
                extra_terminated, extra_truncated, extra_info = checker.check_termination(
                    env=self,
                    previous_snapshot=previous_snapshot,
                    snapshot=snapshot,
                    action_values=action_values_by_agent,
                )
            else:
                extra_terminated, extra_truncated, extra_info = checker(
                    env=self,
                    previous_snapshot=previous_snapshot,
                    snapshot=snapshot,
                    action_values=action_values_by_agent,
                )
            if isinstance(extra_terminated, dict):
                for agent_id, value in extra_terminated.items():
                    if agent_id in terminations:
                        terminations[agent_id] = bool(terminations[agent_id] or value)
            else:
                for agent_id in terminations:
                    terminations[agent_id] = bool(terminations[agent_id] or extra_terminated)
            if isinstance(extra_truncated, dict):
                for agent_id, value in extra_truncated.items():
                    if agent_id in truncations:
                        truncations[agent_id] = bool(truncations[agent_id] or value)
            else:
                for agent_id in truncations:
                    truncations[agent_id] = bool(truncations[agent_id] or extra_truncated)
            if isinstance(extra_info, dict):
                for agent_id, value in extra_info.items():
                    if agent_id in infos and isinstance(value, dict):
                        infos[agent_id].update(value)
        return terminations, truncations, infos

    def _compute_reward_multi(
        self,
        agent_id: str,
        previous_snapshot: dict[str, Any] | None,
        snapshot: dict[str, Any],
        action_values: dict[str, float],
        terminated: bool,
        truncated: bool,
    ) -> float:
        reward_fn = self.reward_fns_by_agent.get(agent_id)
        if reward_fn is None:
            counterpart = "target" if agent_id == "chaser" else "chaser"
            sign = 1.0 if agent_id == "chaser" else -1.0
            reward_fn = RelativeDistanceReward(
                controlled_agent_id=agent_id,
                target_id=counterpart,
                sign=sign,
            )
        if hasattr(reward_fn, "compute_reward"):
            return float(
                reward_fn.compute_reward(
                    env=self,
                    previous_snapshot=previous_snapshot,
                    snapshot=snapshot,
                    action_values=action_values,
                    terminated=terminated,
                    truncated=truncated,
                )
            )
        return float(
            reward_fn(
                env=self,
                previous_snapshot=previous_snapshot,
                snapshot=snapshot,
                action_values=action_values,
                terminated=terminated,
                truncated=truncated,
            )
        )

    def _compute_range_km(self) -> float:
        chaser = self.agents.get("chaser")
        target = self.agents.get("target")
        if chaser is None or target is None:
            return float("nan")
        return float(np.linalg.norm(chaser.truth.position_eci_km - target.truth.position_eci_km))

    def _snapshot(self) -> dict[str, Any]:
        truth = {aid: _snapshot_truth(agent.truth) for aid, agent in self.agents.items() if agent.active}
        belief = {
            aid: {
                "state": np.array(agent.belief.state, dtype=float),
                "last_update_t_s": float(agent.belief.last_update_t_s),
            }
            for aid, agent in self.agents.items()
            if agent.active and agent.belief is not None
        }
        knowledge: dict[str, Any] = {}
        for aid, agent in self.agents.items():
            if not agent.active or agent.knowledge_base is None:
                continue
            knowledge[aid] = {}
            for target_id, kb in agent.knowledge_base.snapshot().items():
                knowledge[aid][target_id] = {
                    "state": np.array(kb.state, dtype=float),
                    "last_update_t_s": float(kb.last_update_t_s),
                }
        range_km = self._compute_range_km()
        if np.isfinite(range_km):
            self.closest_range_km = min(float(self.closest_range_km), float(range_km))
        return {
            "truth": truth,
            "belief": belief,
            "knowledge": knowledge,
            "metrics": {
                "range_km": float(range_km),
                "closest_range_km": float(self.closest_range_km),
                "step": int(self.step_count),
                "time_s": float(self.current_time_s),
            },
            "sampled_parameters": dict(self.sampled_parameters),
        }

    def _observation_from_snapshot(self, snapshot: dict[str, Any], fields: tuple[ObservationField, ...]) -> np.ndarray:
        parts: list[np.ndarray] = []
        for obs_field in fields:
            value = _lookup_observation_value(snapshot, obs_field.path, self._observation_probe)
            parts.append(value * float(obs_field.scale))
        if not parts:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(parts).astype(np.float32)

    def _info_dict(self, snapshot: dict[str, Any], agent_id: str) -> dict[str, Any]:
        return {
            "controlled_agent_id": agent_id,
            "sampled_parameters": dict(self.sampled_parameters),
            "metrics": dict(snapshot["metrics"]),
        }


def _stack_info_dicts(infos: list[dict[str, Any]]) -> dict[str, list[Any]]:
    keys: set[str] = set()
    for info in infos:
        keys.update(info.keys())
    return {key: [info.get(key) for info in infos] for key in sorted(keys)}


class SyncVectorSimulationEnv:
    def __init__(self, env_fns: list[Any], *, auto_reset: bool = False):
        if not env_fns:
            raise ValueError("env_fns must not be empty.")
        self.envs = [fn() for fn in env_fns]
        self.num_envs = len(self.envs)
        self.auto_reset = bool(auto_reset)
        self.single_observation_space = self.envs[0].observation_space
        self.single_action_space = self.envs[0].action_space

    def reset(
        self,
        *,
        seed: int | list[int] | None = None,
        options: dict[str, Any] | list[dict[str, Any]] | None = None,
    ) -> tuple[np.ndarray, dict[str, list[Any]]]:
        seeds = self._broadcast_seed(seed)
        options_list = self._broadcast_options(options)
        obs_list = []
        info_list = []
        for env, env_seed, env_options in zip(self.envs, seeds, options_list):
            obs, info = env.reset(seed=env_seed, options=env_options)
            obs_list.append(np.array(obs, dtype=np.float32))
            info_list.append(dict(info))
        return np.stack(obs_list, axis=0), _stack_info_dicts(info_list)

    def step(
        self,
        actions: np.ndarray | list[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, list[Any]]]:
        action_arr = np.array(actions, dtype=np.float32)
        if action_arr.ndim == 1 and self.num_envs == 1:
            action_arr = action_arr.reshape(1, -1)
        if action_arr.shape[0] != self.num_envs:
            raise ValueError(f"Expected actions for {self.num_envs} envs, got shape {action_arr.shape}.")
        obs_list = []
        reward_list = []
        terminated_list = []
        truncated_list = []
        info_list = []
        for env, action in zip(self.envs, action_arr):
            obs, reward, terminated, truncated, info = env.step(action)
            final_observation = None
            final_info = None
            if self.auto_reset and (terminated or truncated):
                final_observation = np.array(obs, dtype=np.float32)
                final_info = dict(info)
                obs, info = env.reset()
                info = dict(info)
                info["final_observation"] = final_observation
                info["final_info"] = final_info
            obs_list.append(np.array(obs, dtype=np.float32))
            reward_list.append(float(reward))
            terminated_list.append(bool(terminated))
            truncated_list.append(bool(truncated))
            info_list.append(dict(info))
        return (
            np.stack(obs_list, axis=0),
            np.array(reward_list, dtype=np.float32),
            np.array(terminated_list, dtype=bool),
            np.array(truncated_list, dtype=bool),
            _stack_info_dicts(info_list),
        )

    def close(self) -> None:
        for env in self.envs:
            close_fn = getattr(env, "close", None)
            if callable(close_fn):
                close_fn()

    def _broadcast_seed(self, seed: int | list[int] | None) -> list[int | None]:
        if seed is None:
            return [None] * self.num_envs
        if isinstance(seed, list):
            if len(seed) != self.num_envs:
                raise ValueError(f"Expected {self.num_envs} seeds, got {len(seed)}.")
            return [None if s is None else int(s) for s in seed]
        base = int(seed)
        return [base + i for i in range(self.num_envs)]

    def _broadcast_options(self, options: dict[str, Any] | list[dict[str, Any]] | None) -> list[dict[str, Any] | None]:
        if options is None:
            return [None] * self.num_envs
        if isinstance(options, list):
            if len(options) != self.num_envs:
                raise ValueError(f"Expected {self.num_envs} options entries, got {len(options)}.")
            return [dict(opt) if opt is not None else None for opt in options]
        return [dict(options) for _ in range(self.num_envs)]


def _vector_worker(conn: Any, env_cfg: GymEnvConfig, auto_reset: bool) -> None:
    env = GymSimulationEnv(env_cfg)
    try:
        while True:
            cmd, payload = conn.recv()
            if cmd == "reset":
                seed = payload.get("seed")
                options = payload.get("options")
                conn.send(env.reset(seed=seed, options=options))
                continue
            if cmd == "step":
                obs, reward, terminated, truncated, info = env.step(payload["action"])
                if auto_reset and (terminated or truncated):
                    final_observation = np.array(obs, dtype=np.float32)
                    final_info = dict(info)
                    obs, info = env.reset()
                    info = dict(info)
                    info["final_observation"] = final_observation
                    info["final_info"] = final_info
                conn.send((obs, reward, terminated, truncated, info))
                continue
            if cmd == "close":
                conn.close()
                break
            raise ValueError(f"Unknown command: {cmd}")
    finally:
        close_fn = getattr(env, "close", None)
        if callable(close_fn):
            close_fn()


class AsyncVectorSimulationEnv:
    def __init__(self, env_cfgs: list[GymEnvConfig], *, auto_reset: bool = False):
        if not env_cfgs:
            raise ValueError("env_cfgs must not be empty.")
        ctx = mp.get_context("spawn")
        self.parents = []
        self.processes = []
        self.num_envs = len(env_cfgs)
        self.auto_reset = bool(auto_reset)
        probe_env = GymSimulationEnv(env_cfgs[0])
        self.single_observation_space = probe_env.observation_space
        self.single_action_space = probe_env.action_space
        close_fn = getattr(probe_env, "close", None)
        if callable(close_fn):
            close_fn()
        for env_cfg in env_cfgs:
            parent_conn, child_conn = ctx.Pipe()
            proc = ctx.Process(target=_vector_worker, args=(child_conn, env_cfg, self.auto_reset))
            proc.daemon = True
            proc.start()
            child_conn.close()
            self.parents.append(parent_conn)
            self.processes.append(proc)

    def reset(
        self,
        *,
        seed: int | list[int] | None = None,
        options: dict[str, Any] | list[dict[str, Any]] | None = None,
    ) -> tuple[np.ndarray, dict[str, list[Any]]]:
        seeds = self._broadcast_seed(seed)
        options_list = self._broadcast_options(options)
        for conn, env_seed, env_options in zip(self.parents, seeds, options_list):
            conn.send(("reset", {"seed": env_seed, "options": env_options}))
        results = [conn.recv() for conn in self.parents]
        obs_list = [np.array(obs, dtype=np.float32) for obs, _ in results]
        info_list = [dict(info) for _, info in results]
        return np.stack(obs_list, axis=0), _stack_info_dicts(info_list)

    def step(
        self,
        actions: np.ndarray | list[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, list[Any]]]:
        action_arr = np.array(actions, dtype=np.float32)
        if action_arr.ndim == 1 and self.num_envs == 1:
            action_arr = action_arr.reshape(1, -1)
        if action_arr.shape[0] != self.num_envs:
            raise ValueError(f"Expected actions for {self.num_envs} envs, got shape {action_arr.shape}.")
        for conn, action in zip(self.parents, action_arr):
            conn.send(("step", {"action": action}))
        results = [conn.recv() for conn in self.parents]
        obs_list = [np.array(result[0], dtype=np.float32) for result in results]
        reward_list = [float(result[1]) for result in results]
        terminated_list = [bool(result[2]) for result in results]
        truncated_list = [bool(result[3]) for result in results]
        info_list = [dict(result[4]) for result in results]
        return (
            np.stack(obs_list, axis=0),
            np.array(reward_list, dtype=np.float32),
            np.array(terminated_list, dtype=bool),
            np.array(truncated_list, dtype=bool),
            _stack_info_dicts(info_list),
        )

    def close(self) -> None:
        for conn in self.parents:
            try:
                conn.send(("close", {}))
            except (BrokenPipeError, EOFError, OSError):
                pass
        for conn in self.parents:
            try:
                conn.close()
            except OSError:
                pass
        for proc in self.processes:
            proc.join(timeout=1.0)
            if proc.is_alive():
                proc.terminate()

    def _broadcast_seed(self, seed: int | list[int] | None) -> list[int | None]:
        if seed is None:
            return [None] * self.num_envs
        if isinstance(seed, list):
            if len(seed) != self.num_envs:
                raise ValueError(f"Expected {self.num_envs} seeds, got {len(seed)}.")
            return [None if s is None else int(s) for s in seed]
        base = int(seed)
        return [base + i for i in range(self.num_envs)]

    def _broadcast_options(self, options: dict[str, Any] | list[dict[str, Any]] | None) -> list[dict[str, Any] | None]:
        if options is None:
            return [None] * self.num_envs
        if isinstance(options, list):
            if len(options) != self.num_envs:
                raise ValueError(f"Expected {self.num_envs} options entries, got {len(options)}.")
            return [dict(opt) if opt is not None else None for opt in options]
        return [dict(options) for _ in range(self.num_envs)]


def make_env_fn(env_cfg: GymEnvConfig) -> EnvFactory:
    return EnvFactory(env_cfg)


def make_vector_env(vector_cfg: VectorEnvConfig) -> SyncVectorSimulationEnv | AsyncVectorSimulationEnv:
    if int(vector_cfg.num_envs) <= 0:
        raise ValueError("num_envs must be positive.")
    env_cfgs = [deepcopy(vector_cfg.env_cfg) for _ in range(int(vector_cfg.num_envs))]
    if vector_cfg.parallel:
        return AsyncVectorSimulationEnv(env_cfgs, auto_reset=bool(vector_cfg.auto_reset))
    return SyncVectorSimulationEnv([make_env_fn(cfg) for cfg in env_cfgs], auto_reset=bool(vector_cfg.auto_reset))
