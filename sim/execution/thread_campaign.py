"""Bounded, owner-coordinated thread execution for independent campaigns.

The process campaign paths deliberately keep their process transport and
initializer globals.  This module is the separate opt-in thread transport.
Tasks are serialized before they enter the pool and results are serialized
before they cross back to the owner.  That keeps a worker's configuration and
result graph from aliasing the campaign owner while still allowing one process
to host the run threads.

The owner remains responsible for progress callbacks, checkpoints, retention,
and final report writing.  A thread cannot be killed safely by Python, so
``cancel`` prevents queued work from starting and asks running workers to stop
through the per-execution cancellation event.  ``shutdown`` waits for those
workers before releasing the pool.
"""

from __future__ import annotations

import pickle
from collections.abc import Callable, Iterable, Iterator, Mapping
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from functools import lru_cache
from importlib import import_module
from queue import Empty, Queue
from threading import Event, Lock
from time import monotonic
from typing import Any

from sim.execution.thread_math import thread_math_limits
from sim.execution.worker_budget import (
    ThreadCampaignRuntimeScope,
    current_thread_campaign_runtime_scope,
    thread_campaign_runtime_scope,
)


class ThreadCampaignCancelled(RuntimeError):
    """Raised when a threaded campaign notices owner cancellation."""


class ThreadCampaignBackendUnavailable(RuntimeError):
    """Raised when task or result transport cannot cross the thread boundary."""


@dataclass(frozen=True)
class ThreadCampaignContext:
    """Execution-local cancellation and progress transport."""

    cancel_event: Event
    progress_queue: Queue[dict[str, Any]]

    def cancelled(self) -> bool:
        return bool(self.cancel_event.is_set())


_THREAD_UNSAFE_ENVIRONMENT_KEYS = frozenset(
    {
        "ephemeris_body_callable",
        "ephemeris_callable",
        "jacchia70_density_callable",
        "jb2006_density_callable",
        "jb2008_density_callable",
        "msis86_density_callable",
        "nrlmsise00_density_callable",
        "spice_body_ephemeris_callable",
        "spice_ephemeris_callable",
    }
)
_DEFAULT_THREAD_CAMPAIGN_TIMEOUT_S = 24.0 * 60.0 * 60.0
_UNSET_TIMEOUT = object()

# ``thread_worker_incompatibilities`` serves the narrower object-step
# executor as well as complete scenario workers.  These four reasons describe
# cross-object lifecycle/coupling that is safe when the outer campaign worker
# runs its own scenario serially.  Keep this allowlist exact: callback,
# provider, bridge, and custom component reasons must still fail closed.
_OBJECT_STEP_ONLY_REASON_PREFIXES = (
    "general propagation providers require the process/general-propagation path",
    "system force stepping couples objects and requires its coordinated executor",
    "collision stepping couples objects and requires its coordinated executor",
    "dynamic rocket deployment changes object-worker membership",
)


@lru_cache(maxsize=1)
def _registered_builtin_pointer_names() -> frozenset[tuple[str, str]]:
    """Return exact OEL class pointers that the full thread contract knows.

    Scenario pointers are data and must not be trusted by module prefix alone.
    The runtime thread contract already maintains the authoritative exact-type
    registry, so static campaign preflight may use its class identities without
    importing an arbitrary user module or constructing an extension.
    """

    try:
        from sim.execution.thread_contract import _registered_builtin_types

        return frozenset(
            (str(cls.__module__), str(cls.__name__))
            for cls in _registered_builtin_types()
            if isinstance(cls, type)
        )
    except (ImportError, AttributeError, TypeError):
        return frozenset()


def _is_registered_builtin_pointer(value: Mapping[str, Any]) -> bool:
    module = value.get("module")
    class_name = value.get("class_name")
    if not isinstance(module, str) or not isinstance(class_name, str):
        return False
    # Resolve the class identity from the registry rather than trusting an
    # arbitrary module path.  Importing the exact registered module is only a
    # secondary guard against stale or malformed serialized names.
    if (module, class_name) not in _registered_builtin_pointer_names():
        return False
    try:
        return getattr(import_module(module), class_name, None) is not None
    except (ImportError, AttributeError, TypeError):
        return False


def thread_campaign_incompatibilities(
    config: Mapping[str, Any],
    *,
    label: str = "scenario",
    allow_outer_analysis: bool = False,
) -> list[str]:
    """Return static reasons a complete scenario is unsafe in campaign threads.

    The complete scenario engine is still constructed inside each worker.  This
    read-only check catches the extension surfaces that cannot inherit the
    process boundary: SPICE's mutable kernel context, callable environment
    hooks, custom plugin pointers, and nested process object pools.  Exact
    registered OEL module/class pointers remain eligible; unregistered
    module/class/function/file pointers are deliberately fail-closed before a
    worker can create output.
    """

    root = dict(config or {})
    reasons: list[str] = []

    def enabled(value: Any) -> bool:
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)

    def add(reason: str) -> None:
        if reason not in reasons:
            reasons.append(reason)

    def walk_pointers(value: Any, path: str) -> None:
        if isinstance(value, Mapping):
            if value.get("builtin") not in (None, ""):
                # Built-in pointer params are data, not executable extension
                # hooks.  Do not interpret a parameter called ``module`` as a
                # custom pointer.
                return
            custom_fields = tuple(
                key
                for key in ("module", "class_name", "function", "file")
                if value.get(key) not in (None, "")
            )
            if custom_fields:
                if (
                    set(custom_fields).issubset({"module", "class_name"})
                    and _is_registered_builtin_pointer(value)
                ):
                    # OEL's exact built-in guidance/controller/runtime class
                    # is covered by the full object thread contract.  Its
                    # parameter map remains data; custom function/file
                    # pointers continue through the reject path below.
                    return
                add(
                    f"{label}{path} uses custom pointer field(s) "
                    + ", ".join(custom_fields)
                    + "; campaign threads require a built-in or explicitly isolated worker extension"
                )
                return
            for key, child in value.items():
                key_text = str(key)
                if key_text in {"metadata", "params", "values", "description"}:
                    continue
                walk_pointers(child, f"{path}.{key_text}")
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                walk_pointers(child, f"{path}[{index}]")

    simulator = root.get("simulator")
    simulator = simulator if isinstance(simulator, Mapping) else {}
    environment = simulator.get("environment")
    environment = environment if isinstance(environment, Mapping) else {}
    atmosphere_env = environment.get("atmosphere_env")
    atmosphere_env = atmosphere_env if isinstance(atmosphere_env, Mapping) else {}
    for env_path, env in ((".simulator.environment", environment), (".simulator.environment.atmosphere_env", atmosphere_env)):
        mode = str(env.get("ephemeris_mode", "") or "").strip().lower()
        if mode in {"spice", "spiceypy"}:
            add(f"{label}{env_path}.ephemeris_mode={mode!r} uses mutable SPICE kernel state")
        if env.get("spice_kernels"):
            add(f"{label}{env_path}.spice_kernels requires a process-isolated SPICE context")
        for key in _THREAD_UNSAFE_ENVIRONMENT_KEYS:
            if env.get(key) not in (None, ""):
                add(f"{label}{env_path}.{key} is a native/custom environment callback")
        if env.get("extensions"):
            add(f"{label}{env_path}.extensions contains custom environment state")

    outputs = root.get("outputs")
    outputs = outputs if isinstance(outputs, Mapping) else {}
    plots = outputs.get("plots")
    plots = plots if isinstance(plots, Mapping) else {}
    if enabled(plots.get("enabled", False)):
        add(f"{label}.outputs.plots.enabled is not supported inside campaign threads; write plots owner-side")
    animations = outputs.get("animations")
    animations = animations if isinstance(animations, Mapping) else {}
    if enabled(animations.get("enabled", False)):
        add(f"{label}.outputs.animations.enabled is not supported inside campaign threads")

    # A config-queue worker can be handed a scenario that starts its own
    # analysis workflow.  That nested workflow must not launch the default
    # process pool after the outer campaign has crossed into a thread.  The
    # Monte Carlo and sensitivity campaign owners pass ``allow_outer_analysis``
    # because their ``analysis`` section describes the owner-managed campaign,
    # while each prepared child executes one serial scenario.
    analysis = root.get("analysis")
    analysis = analysis if isinstance(analysis, Mapping) else {}
    analysis_execution = analysis.get("execution")
    analysis_execution = analysis_execution if isinstance(analysis_execution, Mapping) else {}
    analysis_backend = str(
        analysis_execution.get("parallel_backend", "process_pool") or "process_pool"
    ).strip().lower()
    if (
        not allow_outer_analysis
        and enabled(analysis.get("enabled", False))
        and enabled(analysis_execution.get("parallel_enabled", False))
        and analysis_backend == "process_pool"
    ):
        add(
            f"{label}.analysis.execution requests a nested process pool while analysis is enabled; "
            "thread campaigns require owner-managed analysis or an explicit thread backend"
        )

    execution = simulator.get("execution")
    execution = execution if isinstance(execution, Mapping) else {}
    object_parallelism = execution.get("object_parallelism")
    object_parallelism = object_parallelism if isinstance(object_parallelism, Mapping) else {}
    # ``policy`` is normalized on the simulator execution mapping, while the
    # backend and enabled bit live under ``object_parallelism``.
    object_policy = str(execution.get("policy", "configured") or "configured").strip().lower()
    object_backend = str(object_parallelism.get("backend", "serial") or "serial").strip().lower()
    object_enabled = enabled(object_parallelism.get("enabled", False))
    nested_process_requested = object_backend == "process_pool" or (
        object_policy in {"parallel", "auto"} and object_backend != "thread_pool"
    )
    if nested_process_requested and (object_enabled or object_policy in {"parallel", "auto"}):
        add(
            f"{label}.simulator.execution.object_parallelism requests a nested process pool; "
            "thread campaigns require the thread-scoped object executor (including policy-normalized defaults)"
        )

    # Recurse through the normalized scenario while leaving arbitrary metadata
    # and plugin params opaque.  The explicit pointer fields above are enough
    # to reject custom GNC/force/bridge implementations without importing them.
    walk_pointers(root, "")
    objects = root.get("objects")
    if isinstance(objects, Mapping):
        for object_id, obj in objects.items():
            if isinstance(obj, Mapping) and obj.get("bridge") not in (None, {}, ""):
                bridge = obj.get("bridge")
                if isinstance(bridge, Mapping) and bridge.get("builtin") in (None, ""):
                    add(f"{label}.objects.{object_id}.bridge is an external/custom object hook")
                elif not isinstance(bridge, Mapping):
                    add(f"{label}.objects.{object_id}.bridge is an external/custom object hook")
    return reasons


def ensure_thread_campaign_compatible(
    config: Mapping[str, Any],
    *,
    label: str = "scenario",
    allow_outer_analysis: bool = False,
) -> None:
    """Reject a statically unsafe scenario before a threaded worker starts."""

    reasons = thread_campaign_incompatibilities(
        config,
        label=label,
        allow_outer_analysis=allow_outer_analysis,
    )
    if reasons:
        raise ThreadCampaignBackendUnavailable(
            "Thread campaign backend is unavailable for this scenario:\n- " + "\n- ".join(reasons)
        )


def _iter_callable_values(value: Any, path: str) -> Iterator[tuple[str, Any]]:
    """Yield callable values from a bounded runtime parameter container."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            yield from _iter_callable_values(item, f"{path}[{key!r}]")
    elif isinstance(value, (tuple, list, set, frozenset)):
        for index, item in enumerate(value):
            yield from _iter_callable_values(item, f"{path}[{index}]")
    elif callable(value):
        yield path, value


def thread_campaign_engine_incompatibilities(
    engine: Any,
    *,
    active_ids: Iterable[str] | None = None,
) -> list[str]:
    """Return hook-safety reasons for a constructed campaign scenario engine.

    The full engine contract is authoritative for object-owned components.  A
    complete campaign task may still use serial object stepping, so only the
    object-step coordination reasons are filtered here.  Runtime parameter
    maps are checked after factories have materialized the engine; this closes
    the static preflight escape hatch where a callable hidden below a built-in
    pointer's ``params`` map would otherwise remain opaque.
    """

    from sim.execution.thread_contract import (
        _component_incompatibility,
        thread_worker_incompatibilities,
    )

    ids = list(active_ids) if active_ids is not None else list((getattr(engine, "agents", {}) or {}).keys())
    reasons = [
        reason
        for reason in thread_worker_incompatibilities(engine, ids)
        if not reason.startswith(_OBJECT_STEP_ONLY_REASON_PREFIXES)
    ]
    agents = getattr(engine, "agents", {}) or {}
    for object_id in ids:
        agent = agents.get(object_id)
        if agent is None:
            continue
        runtime = getattr(agent, "flight_software_runtime", None)
        profile_params = getattr(runtime, "profile_params", None)
        for path, callback in _iter_callable_values(
            profile_params,
            f"object {object_id!r}.flight_software_runtime.profile_params",
        ):
            reason = _component_incompatibility(callback, path)
            if reason is not None:
                reasons.append(reason)
    return list(dict.fromkeys(reasons))


def ensure_thread_campaign_engine_compatible(
    engine: Any,
    *,
    label: str = "thread campaign scenario",
    active_ids: Iterable[str] | None = None,
) -> None:
    """Reject an unsafe constructed engine before its run emits artifacts."""

    reasons = thread_campaign_engine_incompatibilities(engine, active_ids=active_ids)
    if reasons:
        raise ThreadCampaignBackendUnavailable(
            "Thread campaign backend is unavailable for this constructed scenario "
            f"({label}):\n- " + "\n- ".join(reasons)
        )


def preflight_thread_campaign_engine_if_active(engine: Any) -> None:
    """Validate *engine* only when called inside a thread campaign worker."""

    scope = current_thread_campaign_runtime_scope()
    if scope is None:
        return
    ensure_thread_campaign_engine_compatible(engine, label=scope.label)


class ThreadCampaignExecutor:
    """Run independent campaign tasks in a bounded, private thread pool.

    ``worker`` functions retain their process-worker signature and receive a
    task dictionary.  The executor adds ``cancel_event`` and
    ``progress_queue`` only after the task has been serialized, so those
    execution-local objects are never part of the user task's transport or
    shared global state.
    """

    def __init__(
        self,
        *,
        max_workers: int,
        max_in_flight: int | None = None,
        overall_timeout_s: float | None = _DEFAULT_THREAD_CAMPAIGN_TIMEOUT_S,
    ) -> None:
        workers = int(max_workers)
        if workers <= 0:
            raise ValueError("ThreadCampaignExecutor max_workers must be positive.")
        self.max_workers = workers
        self.max_in_flight = max(workers, int(max_in_flight or 2 * workers))
        self.overall_timeout_s = (
            None if overall_timeout_s is None else max(0.0, float(overall_timeout_s))
        )
        self.context = ThreadCampaignContext(cancel_event=Event(), progress_queue=Queue())
        self._pool: ThreadPoolExecutor | None = ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix="oel-campaign",
        )
        self._lock = Lock()
        self._futures: set[Future[Any]] = set()
        self._closed = False

    @property
    def progress_queue(self) -> Queue[dict[str, Any]]:
        return self.context.progress_queue

    @property
    def cancel_event(self) -> Event:
        return self.context.cancel_event

    def iter_completed(
        self,
        worker: Callable[[dict[str, Any]], Any],
        tasks: Iterable[dict[str, Any]],
        *,
        poll_interval_s: float = 0.1,
        overall_timeout_s: Any = _UNSET_TIMEOUT,
    ) -> Iterator[tuple[Future[Any] | None, dict[str, Any] | None]]:
        """Yield completed futures with bounded submission.

        ``(None, None)`` is yielded while no future has completed so the owner
        can drain progress and run cancellation-aware callbacks.  Completed
        results are yielded in completion order, while task IDs in the result
        preserve campaign ordering for checkpoint and aggregation ownership.
        """

        task_iter = iter(tasks)
        pending: dict[Future[Any], dict[str, Any]] = {}
        if overall_timeout_s is _UNSET_TIMEOUT:
            timeout = self.overall_timeout_s
        elif overall_timeout_s is None:
            timeout = None
        else:
            timeout = max(0.0, float(overall_timeout_s))
        deadline = None if timeout is None else monotonic() + timeout

        def deadline_error() -> TimeoutError:
            return TimeoutError(
                "Thread campaign execution exceeded the overall timeout of "
                f"{float(timeout):.3f} s with {len(pending)} task(s) pending."
            )

        def enforce_deadline() -> None:
            if deadline is not None and monotonic() >= deadline:
                self.cancel()
                raise deadline_error()

        def fill() -> None:
            while len(pending) < self.max_in_flight:
                enforce_deadline()
                if self.cancelled:
                    return
                try:
                    task = next(task_iter)
                except StopIteration:
                    return
                future = self.submit(worker, task)
                pending[future] = task

        fill()
        while pending:
            enforce_deadline()
            done, _ = wait(
                set(pending),
                timeout=(
                    max(0.0, float(poll_interval_s))
                    if deadline is None
                    else min(max(0.0, float(poll_interval_s)), max(0.0, deadline - monotonic()))
                ),
                return_when=FIRST_COMPLETED,
            )
            if not done:
                enforce_deadline()
                yield None, None
                continue
            for future in done:
                task = pending.pop(future)
                yield future, task
            fill()

    def submit(self, worker: Callable[[dict[str, Any]], Any], task: dict[str, Any]) -> Future[Any]:
        """Submit one isolated task to the pool."""

        with self._lock:
            if self._closed:
                raise ThreadCampaignBackendUnavailable("Thread campaign executor is closed.")
            if self.cancelled:
                raise ThreadCampaignCancelled("Thread campaign execution was cancelled.")
            pool = self._pool
            if pool is None:
                raise ThreadCampaignBackendUnavailable("Thread campaign pool is unavailable.")
            try:
                isolated_task = _pickle_roundtrip(task, context="thread campaign task")
            except ThreadCampaignBackendUnavailable:
                raise
            future = pool.submit(_run_thread_task, worker, isolated_task, self.context)
            self._futures.add(future)
        # ``Future.add_done_callback`` invokes the callback synchronously when
        # the future already completed.  Register after releasing ``_lock`` so
        # a fast worker cannot re-enter ``_forget`` while submit still owns it.
        future.add_done_callback(self._forget)
        return future

    @property
    def cancelled(self) -> bool:
        return bool(self.context.cancel_event.is_set())

    def cancel(self) -> None:
        """Request cancellation and cancel work that has not started."""

        self.context.cancel_event.set()
        with self._lock:
            futures = tuple(self._futures)
        for future in futures:
            future.cancel()

    def drain_progress(self) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        while True:
            try:
                event = self.progress_queue.get_nowait()
            except Empty:
                return events
            except Exception:
                return events
            if isinstance(event, dict):
                events.append(dict(event))

    def shutdown(self, *, wait_for_running: bool = True) -> None:
        """Close the pool; running tasks cannot be forcibly terminated."""

        with self._lock:
            if self._closed:
                return
            self._closed = True
            pool, self._pool = self._pool, None
        self.cancel()
        if pool is not None:
            pool.shutdown(wait=bool(wait_for_running), cancel_futures=True)
        with self._lock:
            self._futures.clear()

    def __enter__(self) -> ThreadCampaignExecutor:
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        if exc_type is not None:
            self.cancel()
        self.shutdown(wait_for_running=True)

    def _forget(self, future: Future[Any]) -> None:
        with self._lock:
            self._futures.discard(future)


def _run_thread_task(
    worker: Callable[[dict[str, Any]], Any],
    task: dict[str, Any],
    context: ThreadCampaignContext,
) -> Any:
    if context.cancelled():
        raise ThreadCampaignCancelled("Thread campaign execution was cancelled.")
    task = dict(task)
    task["cancel_event"] = context.cancel_event
    task["progress_queue"] = context.progress_queue
    task["_thread_campaign"] = True
    runtime_label = task.get("_thread_campaign_label") or task.get("label") or "thread campaign scenario"
    budget = task.get("worker_budget")
    if isinstance(budget, Mapping):
        try:
            object_workers = int(budget.get("object_workers_per_run", 0))
            campaign_workers = int(budget.get("campaign_workers", 1))
            total_process_budget = int(budget.get("total_process_budget", 1))
        except Exception as exc:
            raise ThreadCampaignBackendUnavailable(
                "Thread campaign worker budget is invalid; expected a hierarchical "
                "worker budget with plain numeric fields."
            ) from exc
        from sim.execution.worker_budget import worker_budget_context

        with thread_math_limits():
            with thread_campaign_runtime_scope(str(runtime_label)):
                with worker_budget_context(
                    object_workers_per_run=object_workers,
                    campaign_workers=campaign_workers,
                    total_process_budget=total_process_budget,
                ):
                    result = worker(task)
    else:
        with thread_math_limits():
            with thread_campaign_runtime_scope(str(runtime_label)):
                result = worker(task)
    return _pickle_roundtrip(result, context="thread campaign result")


def _pickle_roundtrip(value: Any, *, context: str) -> Any:
    try:
        return pickle.loads(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))
    except Exception as exc:
        raise ThreadCampaignBackendUnavailable(
            f"Cannot serialize {context}; threaded campaign execution requires "
            "process-serializable task and result data."
        ) from exc


__all__ = [
    "ThreadCampaignBackendUnavailable",
    "ThreadCampaignCancelled",
    "ThreadCampaignContext",
    "ThreadCampaignExecutor",
    "ThreadCampaignRuntimeScope",
    "current_thread_campaign_runtime_scope",
    "ensure_thread_campaign_compatible",
    "ensure_thread_campaign_engine_compatible",
    "preflight_thread_campaign_engine_if_active",
    "thread_campaign_engine_incompatibilities",
    "thread_campaign_incompatibilities",
    "thread_campaign_runtime_scope",
]
