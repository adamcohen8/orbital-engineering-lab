"""Object-step records, process transport, and native preparation threads."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import copy
from dataclasses import dataclass, field, fields, replace
from multiprocessing.context import BaseContext
from time import monotonic
from traceback import format_exc
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from sim.acceleration.settings import acceleration_context_from_config
from sim.core.models import StateBelief, StateTruth
from sim.dynamics.attitude.rigid_body import (
    activate_attitude_guardrail_stats,
    get_attitude_guardrail_stats,
)
from sim.execution.runtime_profile import _RuntimeProfiler
from sim.platform_compat import process_context
from sim.runtime_support import AgentRuntime
from sim.utils.parallel import restore_env_vars, set_parallel_worker_thread_limits

if TYPE_CHECKING:
    from sim.single_run import _SingleRunEngine


_DEFAULT_OBJECT_STEP_RESPONSE_TIMEOUT_S = 300.0
_DEFAULT_OBJECT_WORKER_SHUTDOWN_TIMEOUT_S = 5.0


@dataclass(frozen=True)
class ObjectStepInput:
    object_id: str
    agent: AgentRuntime
    initial_truth: StateTruth
    world_truth_decision: dict[str, StateTruth]
    t_s: float
    t_next: float
    sample_index: int


@dataclass(frozen=True)
class ObjectStepMessage:
    object_id: str
    knowledge_base: Any | None
    initial_truth: StateTruth
    world_truth_decision: dict[str, StateTruth]
    t_s: float
    t_next: float
    sample_index: int
    orbit_controller: Any | None = None
    attitude_controller: Any | None = None
    mission_strategy: Any | None = None
    mission_execution: Any | None = None


@dataclass(frozen=True)
class ObjectKnowledgeSyncResult:
    object_id: str
    knowledge_snapshot: dict[str, StateBelief] = field(default_factory=dict)
    measurement_snapshot: dict[str, np.ndarray] = field(default_factory=dict)
    detection_summary: dict[str, Any] = field(default_factory=dict)
    consistency_summary: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ObjectStepResult:
    object_id: str
    stage: str
    elapsed_s: float
    truth: StateTruth
    thrust_eci_km_s2: np.ndarray
    torque_body_nm: np.ndarray
    delta_v_m_s: float = 0.0
    max_accel_km_s2: float = 0.0
    burned: bool = False
    throttle: float | None = None
    stage_index: float | None = None
    q_dyn_pa: float | None = None
    mach: float | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    bridge_events: list[dict[str, Any]] = field(default_factory=list)
    bridge_elapsed_s: float = 0.0
    updated_agent: AgentRuntime | None = None
    controller_debug_events: list[dict[str, Any]] = field(default_factory=list)
    command_decision_events: list[dict[str, Any]] = field(default_factory=list)
    profile_stage_seconds: dict[str, float] = field(default_factory=dict)
    profile_stage_counts: dict[str, int] = field(default_factory=dict)
    last_orbital_command_eval_t_s: float | None = None
    latched_orbital_thrust_cmd: np.ndarray | None = None
    desired_attitude_quat_bn: np.ndarray | None = None
    belief_state: np.ndarray | None = None
    belief_covariance: np.ndarray | None = None
    belief_last_update_t_s: float | None = None
    attitude_guardrail_count_deltas: dict[str, int] = field(default_factory=dict)
    flight_software_evidence_offsets: dict[str, int] | None = None


class ObjectStepExecutor(Protocol):
    backend_name: str

    def step_objects(self, inputs: list[ObjectStepInput]) -> list[ObjectStepResult]:
        """Advance one timestep for each object input and return ordered results."""

    def shutdown(self) -> None:
        """Release executor resources."""

    def sync_after_step(
        self,
        *,
        world_truth: dict[str, StateTruth],
        sample_index: int,
        t_s: float,
    ) -> list[ObjectKnowledgeSyncResult] | None:
        """Synchronize worker knowledge after the orchestrator has the step truth."""


class ObjectStepBackendUnavailable(RuntimeError):
    """Raised when process transport fails independently of object physics."""


class SerialObjectStepExecutor:
    backend_name = "serial"
    max_workers = 1

    def __init__(self, engine: _SingleRunEngine) -> None:
        self.engine = engine

    def step_objects(self, inputs: list[ObjectStepInput]) -> list[ObjectStepResult]:
        return [self.engine._step_object_serial(item) for item in inputs]

    def shutdown(self) -> None:
        return None

    def sync_after_step(
        self,
        *,
        world_truth: dict[str, StateTruth],
        sample_index: int,
        t_s: float,
    ) -> list[ObjectKnowledgeSyncResult] | None:
        return None


class OGPBatchStepExecutor(SerialObjectStepExecutor):
    """Public bounded native OGP preparation, with ordinary ordered stepping."""

    def __init__(self, engine: _SingleRunEngine, *, max_workers: int) -> None:
        super().__init__(engine)
        from sim.execution.ogp_history import OGPHistoryPreparation

        self.preparation = OGPHistoryPreparation(engine, workers=max(1, int(max_workers)))

    def step_objects(self, inputs: list[ObjectStepInput]) -> list[ObjectStepResult]:
        self.preparation.prepare(inputs)
        return super().step_objects(inputs)


class ThreadPoolHistoryStepExecutor(SerialObjectStepExecutor):
    """Parallel native history preparation followed by ordinary serial steps."""

    backend_name = "thread_pool"

    def __init__(self, engine: _SingleRunEngine, *, max_workers: int) -> None:
        super().__init__(engine)
        self.max_workers = max(1, int(max_workers))
        self._pool: ThreadPoolExecutor | None = None

    def step_objects(self, inputs: list[ObjectStepInput]) -> list[ObjectStepResult]:
        if len(inputs) > 1:
            if self._pool is None:
                self._pool = ThreadPoolExecutor(
                    max_workers=self.max_workers, thread_name_prefix="oel-native-history",
                )
            self.engine.satellite_stepper.passive_history.prepare_objects(
                inputs, self._pool, max_workers=self.max_workers,
            )
        return super().step_objects(inputs)

    def shutdown(self) -> None:
        pool, self._pool = self._pool, None
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)


class ProcessPoolObjectStepExecutor:
    backend_name = "process_pool"

    def __init__(
        self,
        engine: _SingleRunEngine,
        *,
        max_workers: int,
        context: BaseContext | None = None,
        response_timeout_s: float = _DEFAULT_OBJECT_STEP_RESPONSE_TIMEOUT_S,
        shutdown_timeout_s: float = _DEFAULT_OBJECT_WORKER_SHUTDOWN_TIMEOUT_S,
    ) -> None:
        self.engine = engine
        self.max_workers = int(max(1, max_workers))
        self._context = context if context is not None else process_context()
        self.response_timeout_s = max(0.0, float(response_timeout_s))
        self.shutdown_timeout_s = max(0.0, float(shutdown_timeout_s))
        self._worker_processes: list[Any] = []
        self._worker_connections: list[Any] = []
        self._executor_index_by_object: dict[str, int] = {}

    def _initialize_workers(self, inputs: list[ObjectStepInput]) -> None:
        worker_count = min(self.max_workers, len(inputs))
        groups: list[list[ObjectStepInput]] = [[] for _ in range(worker_count)]
        for index, item in enumerate(inputs):
            groups[index % worker_count].append(item)
        thread_env_previous = set_parallel_worker_thread_limits(default_threads="1")
        try:
            for worker_index, group in enumerate(groups):
                snapshots = {
                    item.object_id: self.engine._process_worker_engine_snapshot(item)
                    for item in group
                }
                parent_connection, worker_connection = self._context.Pipe(duplex=True)
                process = self._context.Process(
                    target=_persistent_object_worker_loop,
                    args=(worker_connection, snapshots),
                    name=f"oel-object-worker-{worker_index}",
                )
                process.start()
                worker_connection.close()
                self._worker_processes.append(process)
                self._worker_connections.append(parent_connection)
                for item in group:
                    self._executor_index_by_object[item.object_id] = worker_index
        except Exception as exc:
            self.shutdown()
            raise ObjectStepBackendUnavailable(
                "ProcessPoolObjectStepExecutor is unavailable in this environment. "
                "Use simulator.execution.object_parallelism.backend=serial or disable object_parallelism."
            ) from exc
        finally:
            restore_env_vars(thread_env_previous)

    def step_objects(self, inputs: list[ObjectStepInput]) -> list[ObjectStepResult]:
        if len(inputs) <= 1:
            return [self.engine._step_object_serial(item) for item in inputs]
        if not self._worker_processes:
            self._initialize_workers(inputs)
        input_ids = {item.object_id for item in inputs}
        if input_ids != set(self._executor_index_by_object):
            raise RuntimeError("Persistent object worker membership changed after initialization.")
        chunks: list[list[tuple[int, ObjectStepMessage]]] = [[] for _ in self._worker_processes]
        for index, item in enumerate(inputs):
            chunks[self._executor_index_by_object[item.object_id]].append(
                (
                    index,
                    ObjectStepMessage(
                        object_id=item.object_id,
                        knowledge_base=item.agent.knowledge_base,
                        initial_truth=item.initial_truth,
                        world_truth_decision=item.world_truth_decision,
                        t_s=item.t_s,
                        t_next=item.t_next,
                        sample_index=item.sample_index,
                        orbit_controller=item.agent.orbit_controller,
                        attitude_controller=item.agent.attitude_controller,
                        mission_strategy=item.agent.mission_strategy,
                        mission_execution=item.agent.mission_execution,
                    ),
                )
            )
        active_indices = [index for index, chunk in enumerate(chunks) if chunk]
        try:
            for index in active_indices:
                self._worker_connections[index].send(chunks[index])
            indexed: list[tuple[int, ObjectStepResult]] = []
            response_deadline = monotonic() + self.response_timeout_s
            for index in active_indices:
                remaining_s = max(0.0, response_deadline - monotonic())
                if not self._worker_connections[index].poll(remaining_s):
                    raise ObjectStepBackendUnavailable(
                        "Persistent object worker response timed out after "
                        f"{self.response_timeout_s:.3f} s."
                    )
                status, payload = self._worker_connections[index].recv()
                if status != "ok":
                    message, worker_traceback = payload
                    raise RuntimeError(
                        f"Persistent object worker {index} failed: {message}\n{worker_traceback}"
                    )
                indexed.extend(payload)
        except (BrokenPipeError, EOFError, OSError) as exc:
            raise ObjectStepBackendUnavailable(
                f"Persistent object-worker transport failed: {type(exc).__name__}: {exc}"
            ) from exc
        indexed.sort(key=lambda row: row[0])
        for _index, result in indexed:
            _restore_flight_software_evidence(self.engine, result)
        return [result for _index, result in indexed]

    def sync_after_step(
        self,
        *,
        world_truth: dict[str, StateTruth],
        sample_index: int,
        t_s: float,
    ) -> list[ObjectKnowledgeSyncResult] | None:
        return None

    def shutdown(self) -> None:
        for connection in self._worker_connections:
            try:
                connection.send(None)
            except (BrokenPipeError, EOFError, OSError):
                pass
        for connection in self._worker_connections:
            try:
                connection.close()
            except OSError:
                pass
        shutdown_deadline = monotonic() + self.shutdown_timeout_s
        for process in self._worker_processes:
            process.join(timeout=max(0.0, shutdown_deadline - monotonic()))
        for process in self._worker_processes:
            if process.is_alive():
                process.terminate()
        for process in self._worker_processes:
            if process.is_alive():
                process.join(timeout=max(0.0, shutdown_deadline - monotonic()))
        for process in self._worker_processes:
            if process.is_alive() and hasattr(process, "kill"):
                process.kill()
        for process in self._worker_processes:
            if process.is_alive():
                process.join(timeout=0.0)
        self._worker_connections.clear()
        self._worker_processes.clear()
        self._executor_index_by_object.clear()


_PERSISTENT_OBJECT_WORKER_ENGINES: dict[str, _SingleRunEngine] = {}


def _persistent_object_workers_init(engines: dict[str, _SingleRunEngine]) -> None:
    global _PERSISTENT_OBJECT_WORKER_ENGINES
    _PERSISTENT_OBJECT_WORKER_ENGINES = dict(engines)


def _persistent_object_worker_loop(connection: Any, engines: dict[str, _SingleRunEngine]) -> None:
    _persistent_object_workers_init(engines)
    try:
        while True:
            try:
                chunk = connection.recv()
            except (EOFError, OSError):
                return
            if chunk is None:
                return
            try:
                connection.send(("ok", _persistent_object_step_batch_worker(chunk)))
            except Exception as exc:
                connection.send(("error", (str(exc), format_exc())))
                return
    finally:
        connection.close()


def _persistent_object_step_worker(message: ObjectStepMessage) -> ObjectStepResult:
    aid = str(message.object_id)
    engine = _PERSISTENT_OBJECT_WORKER_ENGINES.get(aid)
    if engine is None:
        raise RuntimeError(f"Persistent object worker for {aid!r} has not been initialized.")
    return _run_object_step_worker(engine, message)


def _persistent_object_step_batch_worker(
    chunk: list[tuple[int, ObjectStepMessage]],
) -> list[tuple[int, ObjectStepResult]]:
    return [(index, _persistent_object_step_worker(message)) for index, message in chunk]


def _sync_runtime_components(agent: AgentRuntime, message: ObjectStepMessage) -> None:
    agent.knowledge_base = message.knowledge_base
    agent.orbit_controller = message.orbit_controller
    agent.attitude_controller = message.attitude_controller
    agent.mission_strategy = message.mission_strategy
    agent.mission_execution = message.mission_execution


def _run_object_step_worker(engine: _SingleRunEngine, message: ObjectStepMessage) -> ObjectStepResult:
    aid = str(message.object_id)
    agent = engine.agents[aid]
    _sync_runtime_components(agent, message)
    activate_attitude_guardrail_stats(engine.attitude_guardrail_stats)
    guardrail_counts_before = get_attitude_guardrail_stats(engine.attitude_guardrail_stats)
    profiler_enabled = bool(getattr(engine.runtime_profiler, "enabled", True))
    engine.runtime_profiler = _RuntimeProfiler(object_ids=[aid], enabled=profiler_enabled)
    engine.controller_debug_hist = {aid: []}
    engine.command_decision_hist = {aid: []}
    desired_attitude_hist = engine.desired_attitude_hist.get(aid)
    if desired_attitude_hist is None or desired_attitude_hist.shape != (1, 4):
        desired_attitude_hist = np.full((1, 4), np.nan)
    else:
        desired_attitude_hist[0, :] = np.nan
    engine.desired_attitude_hist = {aid: desired_attitude_hist}
    item = ObjectStepInput(
        object_id=aid,
        agent=agent,
        initial_truth=message.initial_truth,
        world_truth_decision=message.world_truth_decision,
        t_s=message.t_s,
        t_next=message.t_next,
        sample_index=message.sample_index,
    )
    # These caches belong to one owner-side object-step interval.  Persistent
    # process and thread workers retain their engine between messages, so a
    # prior interval's environment/forecast entries must never influence the
    # next message's start-of-step inputs.
    engine._time_dependent_env_cache = {}
    engine._forecast_truth_cache = {}
    with acceleration_context_from_config(engine.cfg):
        result = engine._step_object_serial(item)
    guardrail_counts_after = get_attitude_guardrail_stats(engine.attitude_guardrail_stats)
    guardrail_count_deltas = {
        name: int(value) - int(guardrail_counts_before.get(name, 0))
        for name, value in guardrail_counts_after.items()
    }
    belief = agent.belief
    updated_agent = replace(agent, knowledge_base=None)
    evidence_offsets = _compact_flight_software_evidence(engine, updated_agent)
    return replace(
        result,
        updated_agent=updated_agent,
        controller_debug_events=list(engine.controller_debug_hist.get(aid, [])),
        command_decision_events=list(engine.command_decision_hist.get(aid, [])),
        profile_stage_seconds=dict(engine.runtime_profiler.object_stage_seconds.get(aid, {})),
        profile_stage_counts=dict(engine.runtime_profiler.object_stage_counts.get(aid, {})),
        last_orbital_command_eval_t_s=engine._last_orbital_command_eval_t_s.get(aid),
        latched_orbital_thrust_cmd=np.array(
            engine._latched_orbital_thrust_cmd_by_object.get(aid, engine.zero3),
            dtype=float,
        ),
        desired_attitude_quat_bn=np.array(
            engine.desired_attitude_hist[aid][0, :],
            dtype=float,
        )
        if aid in engine.desired_attitude_hist
        else None,
        belief_state=(None if belief is None else np.array(belief.state, dtype=float)),
        belief_covariance=(None if belief is None else np.array(belief.covariance, dtype=float)),
        belief_last_update_t_s=(None if belief is None else float(belief.last_update_t_s)),
        attitude_guardrail_count_deltas=guardrail_count_deltas,
        flight_software_evidence_offsets=evidence_offsets,
    )


def _compact_flight_software_evidence(
    engine: _SingleRunEngine, updated_agent: AgentRuntime,
) -> dict[str, int] | None:
    """Send append-only boundary evidence once while retaining full worker state.

    Returning the complete runtime after every sample used to serialize all
    past typed packets repeatedly. Only the reporting evidence lists are
    compacted; stack state, inputs, hardware and scheduler remain ordinary
    per-step runtime snapshots.
    """
    from sim.runtime.satellites.flight_software_runtime import SatelliteFlightSoftwareRuntime

    runtime = updated_agent.flight_software_runtime
    if type(runtime) is not SatelliteFlightSoftwareRuntime:
        # Custom adapters may change historical evidence rather than append it.
        return None
    offsets = getattr(engine, "_object_worker_evidence_offsets", {})
    evidence = runtime.evidence
    names = tuple(item.name for item in fields(evidence))
    compact = replace(evidence, **{
        name: getattr(evidence, name)[int(offsets.get(name, 0)):]
        for name in names
    })
    previous = {name: int(offsets.get(name, 0)) for name in names}
    engine._object_worker_evidence_offsets = {
        name: len(getattr(evidence, name)) for name in names
    }
    updated_agent.flight_software_runtime = copy(runtime)
    updated_agent.flight_software_runtime.evidence = compact
    return previous


def _restore_flight_software_evidence(engine: _SingleRunEngine, result: ObjectStepResult) -> None:
    offsets = getattr(result, "flight_software_evidence_offsets", None)
    if offsets is None or result.updated_agent is None:
        return
    runtime = result.updated_agent.flight_software_runtime
    previous_runtime = engine.agents[result.object_id].flight_software_runtime
    for name, offset in offsets.items():
        previous = getattr(previous_runtime.evidence, name)
        if len(previous) < offset:
            raise RuntimeError(f"Object worker evidence prefix is missing for {result.object_id!r}/{name}.")
        # Parent-side boundary observation can append the next invocation.
        # The next worker result is authoritative for that same boundary,
        # matching the previous complete-runtime replacement semantics.
        del previous[offset:]
        previous.extend(getattr(runtime.evidence, name))
        setattr(runtime.evidence, name, previous)
