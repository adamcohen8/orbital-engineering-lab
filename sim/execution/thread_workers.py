"""Process-equivalent, bounded threaded execution for complete object steps.

The native-history thread executor lives beside this module, but it only
prepares a narrow passive trajectory.  This executor handles the full object
step contract used by the persistent process workers.  Each object receives a
private engine snapshot and every message and result crosses a pickle
round-trip before it crosses the thread boundary.  That deliberately keeps
the ownership and aliasing rules close to the process implementation while
retaining one process for the whole simulation.

Threads do not have process failure isolation.  A running Python thread
cannot be forcibly stopped, so a failed step cancels queued work, marks the
executor unusable, and lets already-running calls finish during ``shutdown``.
The caller must not retry an executor after an exception unless it has first
shut it down and constructed a new one.
"""

from __future__ import annotations

import pickle
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from contextlib import ExitStack
from threading import Lock
from time import monotonic
from typing import TYPE_CHECKING, Any

from sim.execution.object_workers import (
    _DEFAULT_OBJECT_STEP_RESPONSE_TIMEOUT_S,
    ObjectKnowledgeSyncResult,
    ObjectStepBackendUnavailable,
    ObjectStepInput,
    ObjectStepMessage,
    ObjectStepResult,
    _restore_flight_software_evidence,
    _run_object_step_worker,
)
from sim.execution.thread_math import thread_math_limits
from sim.execution.worker_budget import current_worker_budget, worker_budget_context

if TYPE_CHECKING:
    from sim.single_run import _SingleRunEngine


class ThreadPoolObjectStepExecutor:
    """Run complete object steps on persistent, privately-owned engines.

    ``max_workers`` bounds both the native thread pool and the number of
    futures submitted ahead of the ordered result cursor.  Object engines are
    initialized lazily on the first multi-object step and then retained for
    the lifetime of this executor, just as a process worker retains its
    engine.  A single-object call follows the existing process executor's
    serial path and never submits work to the pool.
    """

    backend_name = "thread_pool"

    def __init__(
        self,
        engine: _SingleRunEngine,
        *,
        max_workers: int,
        response_timeout_s: float = _DEFAULT_OBJECT_STEP_RESPONSE_TIMEOUT_S,
    ) -> None:
        self.engine = engine
        self.max_workers = max(1, int(max_workers))
        self.response_timeout_s = max(0.0, float(response_timeout_s))
        self._pool: ThreadPoolExecutor | None = None
        self._worker_engines: dict[str, _SingleRunEngine] = {}
        self._object_ids: tuple[str, ...] = ()
        self._state_lock = Lock()
        self._step_lock = Lock()
        self._inflight: set[Future[Any]] = set()
        self._broken_error: BaseException | None = None
        self._shutdown = False

    def step_objects(self, inputs: list[ObjectStepInput]) -> list[ObjectStepResult]:
        """Advance objects in input order, preserving process-style failures."""

        with self._step_lock:
            return self._step_objects_locked(inputs)

    def _step_objects_locked(self, inputs: list[ObjectStepInput]) -> list[ObjectStepResult]:
        self._ensure_available()
        if not inputs:
            return []
        # Match ProcessPoolObjectStepExecutor: a one-object timestep remains
        # on the owner even after the pool has been initialized.
        if len(inputs) <= 1:
            return [self.engine._step_object_serial(item) for item in inputs]

        self._validate_thread_contract(inputs)
        self._ensure_workers(inputs)
        if tuple(item.object_id for item in inputs) != self._object_ids:
            raise RuntimeError(
                "Persistent threaded object-worker membership changed after initialization."
            )

        try:
            return self._step_bounded(inputs)
        except BaseException as exc:
            self._mark_broken(exc)
            raise

    def sync_after_step(
        self,
        *,
        world_truth: dict[str, Any],
        sample_index: int,
        t_s: float,
    ) -> list[ObjectKnowledgeSyncResult] | None:
        del world_truth, sample_index, t_s
        return None

    def shutdown(self) -> None:
        """Cancel queued work and wait for running calls before releasing state.

        Python supplies no safe way to kill a running thread.  Consequently
        this method may wait for a worker that is inside user code or a native
        extension; it never pretends that an unfinished object step was
        cancelled.  The executor is permanently closed after this method.
        """

        # Serialize teardown with the submission loop.  Without this gate a
        # concurrent shutdown could close the pool between two bounded
        # submissions, turning a clean caller-side shutdown into an ambiguous
        # ``cannot schedule new futures`` failure.
        with self._step_lock:
            with self._state_lock:
                pool, self._pool = self._pool, None
                self._shutdown = True
                inflight = tuple(self._inflight)
            for future in inflight:
                future.cancel()
            if pool is not None:
                pool.shutdown(wait=True, cancel_futures=True)
            with self._state_lock:
                self._worker_engines.clear()
                self._object_ids = ()
                self._inflight.clear()

    def _ensure_available(self) -> None:
        with self._state_lock:
            if self._shutdown:
                raise ObjectStepBackendUnavailable(
                    "ThreadPoolObjectStepExecutor has been shut down."
                )
            if self._broken_error is not None:
                raise ObjectStepBackendUnavailable(
                    "ThreadPoolObjectStepExecutor cannot be reused after a worker failure; "
                    "shutdown it and construct a new executor."
                ) from self._broken_error

    def _validate_thread_contract(self, inputs: list[ObjectStepInput]) -> None:
        """Recheck object-owned callbacks before every threaded submission.

        Runtime configuration can replace a controller, force model, sensor,
        or flight-software component after the initial planner decision.  The
        automatic policy can then use the normal owner-side fallback, while a
        configured/forced policy receives an explicit backend-unavailable
        error before this timestep has been submitted to a worker.
        """

        from sim.execution.thread_contract import thread_worker_incompatibilities

        reasons = thread_worker_incompatibilities(
            self.engine,
            (item.object_id for item in inputs),
        )
        if reasons:
            raise ObjectStepBackendUnavailable(
                "ThreadPoolObjectStepExecutor contract became incompatible: "
                + "; ".join(reasons)
            )

    def _ensure_workers(self, inputs: list[ObjectStepInput]) -> None:
        if self._worker_engines:
            return
        object_ids = tuple(item.object_id for item in inputs)
        if len(set(object_ids)) != len(object_ids):
            raise RuntimeError("Threaded object-step inputs must contain unique object IDs.")

        engines: dict[str, _SingleRunEngine] = {}
        try:
            for item in inputs:
                # _process_worker_engine_snapshot intentionally remains the
                # single source of worker construction.  The pickle roundtrip
                # is what converts its shallow parent-side copy into a truly
                # private object graph, matching process transport semantics.
                snapshot = self.engine._process_worker_engine_snapshot(item)
                engines[item.object_id] = _pickle_roundtrip(
                    snapshot,
                    context=f"threaded engine snapshot for {item.object_id!r}",
                )
            pool = ThreadPoolExecutor(
                max_workers=min(self.max_workers, len(inputs)),
                thread_name_prefix="oel-object-step",
            )
        except Exception as exc:
            raise ObjectStepBackendUnavailable(
                "ThreadPoolObjectStepExecutor could not initialize private object engines."
            ) from exc
        with self._state_lock:
            if self._shutdown:
                pool.shutdown(wait=True, cancel_futures=True)
                raise ObjectStepBackendUnavailable(
                    "ThreadPoolObjectStepExecutor was shut down during initialization."
                )
            self._worker_engines = engines
            self._object_ids = object_ids
            self._pool = pool

    def _step_bounded(self, inputs: list[ObjectStepInput]) -> list[ObjectStepResult]:
        pool = self._pool
        if pool is None:
            raise ObjectStepBackendUnavailable("Threaded object-step pool is not initialized.")

        pending: deque[tuple[int, Future[ObjectStepResult]]] = deque()
        results: list[ObjectStepResult | None] = [None] * len(inputs)
        next_index = 0
        deadline = monotonic() + self.response_timeout_s

        def submit_one(index: int) -> None:
            item = inputs[index]
            message = _pickle_roundtrip(
                _object_step_message(item),
                context=f"threaded step message for {item.object_id!r}",
            )
            future = pool.submit(
                self._run_one,
                item.object_id,
                message,
                current_worker_budget(),
            )
            with self._state_lock:
                self._inflight.add(future)
            future.add_done_callback(self._forget_future)
            pending.append((index, future))

        try:
            while next_index < len(inputs) and len(pending) < self.max_workers:
                submit_one(next_index)
                next_index += 1

            while pending:
                index, future = pending.popleft()
                # Waiting in submission order makes both successful results
                # and the first reported error deterministic.
                remaining_s = max(0.0, deadline - monotonic())
                try:
                    results[index] = future.result(timeout=remaining_s)
                except FutureTimeoutError as exc:
                    raise ThreadPoolObjectStepTimeout(
                        "ThreadPoolObjectStepExecutor response timed out after "
                        f"{self.response_timeout_s:.3f} s; running threads cannot be killed, "
                        "so the object outcome is unknown and this executor is unusable."
                    ) from exc
                if next_index < len(inputs):
                    submit_one(next_index)
                    next_index += 1
        except BaseException:
            for _index, future in pending:
                future.cancel()
            raise

        ordered = [result for result in results if result is not None]
        for result in ordered:
            _restore_flight_software_evidence(self.engine, result)
        return ordered

    def _run_one(
        self,
        object_id: str,
        message: ObjectStepMessage,
        budget: Any | None,
    ) -> ObjectStepResult:
        worker = self._worker_engines.get(object_id)
        if worker is None:
            raise ObjectStepBackendUnavailable(
                f"Persistent threaded object worker for {object_id!r} is unavailable."
            )
        with ExitStack() as stack:
            if budget is not None:
                stack.enter_context(
                    worker_budget_context(
                        object_workers_per_run=budget.object_workers_per_run,
                        campaign_workers=budget.campaign_workers,
                        total_process_budget=budget.total_process_budget,
                    )
                )
            stack.enter_context(thread_math_limits())
            result = _run_object_step_worker(worker, message)
        # A thread shares Python objects with its owner.  Round-tripping the
        # result before returning is the boundary that prevents a mutable
        # worker result from aliasing the owner's state.
        return _pickle_roundtrip(result, context=f"threaded step result for {object_id!r}")

    def _forget_future(self, future: Future[Any]) -> None:
        with self._state_lock:
            self._inflight.discard(future)

    def _mark_broken(self, exc: BaseException) -> None:
        with self._state_lock:
            if self._broken_error is None:
                self._broken_error = exc
            inflight = tuple(self._inflight)
        for future in inflight:
            future.cancel()


def _object_step_message(item: ObjectStepInput) -> ObjectStepMessage:
    agent = item.agent
    return ObjectStepMessage(
        object_id=item.object_id,
        knowledge_base=agent.knowledge_base,
        initial_truth=item.initial_truth,
        world_truth_decision=item.world_truth_decision,
        t_s=item.t_s,
        t_next=item.t_next,
        sample_index=item.sample_index,
        orbit_controller=agent.orbit_controller,
        attitude_controller=agent.attitude_controller,
        mission_strategy=agent.mission_strategy,
        mission_execution=agent.mission_execution,
    )


def _pickle_roundtrip(value: Any, *, context: str) -> Any:
    try:
        return pickle.loads(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))
    except Exception as exc:
        raise ObjectStepBackendUnavailable(
            f"ThreadPoolObjectStepExecutor could not serialize {context}; "
            "thread execution requires process-serializable object state."
        ) from exc


class ThreadPoolObjectStepTimeout(RuntimeError):
    """Fatal deadline expiry; callers must not replay the unfinished step."""


__all__ = ["ThreadPoolObjectStepExecutor", "ThreadPoolObjectStepTimeout"]
