"""Public object-step records and deterministic serial execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from multiprocessing.context import BaseContext
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from sim.core.models import StateBelief, StateTruth
from sim.pro_features import FEATURE_OBJECT_PARALLELISM, require_pro_feature
from sim.runtime_support import AgentRuntime

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

class ThreadPoolHistoryStepExecutor(SerialObjectStepExecutor):
    """Parallel native history preparation followed by ordinary serial steps."""
    backend_name = 'thread_pool'

    def __init__(self, engine: _SingleRunEngine, *, max_workers: int) -> None:
        require_pro_feature(FEATURE_OBJECT_PARALLELISM)

    def step_objects(self, inputs: list[ObjectStepInput]) -> list[ObjectStepResult]:
        require_pro_feature(FEATURE_OBJECT_PARALLELISM)

    def shutdown(self) -> None:
        require_pro_feature(FEATURE_OBJECT_PARALLELISM)

class ProcessPoolObjectStepExecutor:
    backend_name = 'process_pool'

    def __init__(self, engine: _SingleRunEngine, *, max_workers: int, context: BaseContext | None=None, response_timeout_s: float=_DEFAULT_OBJECT_STEP_RESPONSE_TIMEOUT_S, shutdown_timeout_s: float=_DEFAULT_OBJECT_WORKER_SHUTDOWN_TIMEOUT_S) -> None:
        require_pro_feature(FEATURE_OBJECT_PARALLELISM)

    def _initialize_workers(self, inputs: list[ObjectStepInput]) -> None:
        require_pro_feature(FEATURE_OBJECT_PARALLELISM)

    def step_objects(self, inputs: list[ObjectStepInput]) -> list[ObjectStepResult]:
        require_pro_feature(FEATURE_OBJECT_PARALLELISM)

    def sync_after_step(self, *, world_truth: dict[str, StateTruth], sample_index: int, t_s: float) -> list[ObjectKnowledgeSyncResult] | None:
        require_pro_feature(FEATURE_OBJECT_PARALLELISM)

    def shutdown(self) -> None:
        require_pro_feature(FEATURE_OBJECT_PARALLELISM)

_PERSISTENT_OBJECT_WORKER_ENGINES: dict[str, _SingleRunEngine] = {}

def _persistent_object_workers_init(engines: dict[str, _SingleRunEngine]) -> None:
    require_pro_feature(FEATURE_OBJECT_PARALLELISM)

def _persistent_object_worker_loop(connection: Any, engines: dict[str, _SingleRunEngine]) -> None:
    require_pro_feature(FEATURE_OBJECT_PARALLELISM)

def _persistent_object_step_worker(message: ObjectStepMessage) -> ObjectStepResult:
    require_pro_feature(FEATURE_OBJECT_PARALLELISM)

def _persistent_object_step_batch_worker(chunk: list[tuple[int, ObjectStepMessage]]) -> list[tuple[int, ObjectStepResult]]:
    require_pro_feature(FEATURE_OBJECT_PARALLELISM)

def _sync_runtime_components(agent: AgentRuntime, message: ObjectStepMessage) -> None:
    require_pro_feature(FEATURE_OBJECT_PARALLELISM)

def _run_object_step_worker(engine: _SingleRunEngine, message: ObjectStepMessage) -> ObjectStepResult:
    require_pro_feature(FEATURE_OBJECT_PARALLELISM)

def _compact_flight_software_evidence(engine: _SingleRunEngine, updated_agent: AgentRuntime) -> dict[str, int] | None:
    require_pro_feature(FEATURE_OBJECT_PARALLELISM)

def _restore_flight_software_evidence(engine: _SingleRunEngine, result: ObjectStepResult) -> None:
    require_pro_feature(FEATURE_OBJECT_PARALLELISM)
