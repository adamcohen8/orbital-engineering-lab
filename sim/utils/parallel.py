from __future__ import annotations

import logging
import os
import queue as queue_mod
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import FIRST_COMPLETED, CancelledError, wait
from time import monotonic
from typing import Any

PARALLEL_WORKER_THREAD_ENV_VARS = (
    "VECLIB_MAXIMUM_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
)

_WORKER_PROGRESS_QUEUE: Any | None = None
_DEFAULT_BOUNDED_FUTURES_TIMEOUT_S = 24.0 * 60.0 * 60.0


def set_parallel_worker_thread_limits(default_threads: str = "1") -> dict[str, str | None]:
    """Limit native math threads inherited by newly spawned worker processes."""
    previous: dict[str, str | None] = {}
    for name in PARALLEL_WORKER_THREAD_ENV_VARS:
        previous[name] = os.environ.get(name)
        if previous[name] is None:
            os.environ[name] = str(default_threads)
    return previous


def restore_env_vars(previous: dict[str, str | None]) -> None:
    """Restore environment variables captured by a worker-launch policy."""
    for name, value in previous.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


def initialize_worker_progress_queue(progress_queue: Any) -> None:
    """Install a launch-time queue in a process-pool worker."""
    global _WORKER_PROGRESS_QUEUE
    _WORKER_PROGRESS_QUEUE = progress_queue


def worker_progress_queue() -> Any | None:
    """Return the process-local progress queue installed by the pool initializer."""
    return _WORKER_PROGRESS_QUEUE


def abort_process_pool(executor: Any) -> None:
    """Cancel queued work and terminate only the supplied executor's children."""

    for process in list(dict(getattr(executor, "_processes", {}) or {}).values()):
        try:
            process.terminate()
        except (AttributeError, OSError):
            pass
    shutdown = getattr(executor, "shutdown", None)
    if not callable(shutdown):
        return
    try:
        shutdown(wait=False, cancel_futures=True)
    except TypeError:  # pragma: no cover - compatibility with older Python
        shutdown(wait=False)


def iter_bounded_futures(
    executor: Any,
    worker: Callable[[Any], Any],
    tasks: Iterable[Any],
    *,
    max_in_flight: int,
    poll_interval_s: float = 0.1,
    overall_timeout_s: float | None = _DEFAULT_BOUNDED_FUTURES_TIMEOUT_S,
) -> Iterator[tuple[Any | None, Any | None]]:
    """Submit bounded tasks and stop waiting after the optional overall deadline."""
    task_iter = iter(tasks)
    pending: dict[Any, Any] = {}
    limit = max(1, int(max_in_flight))
    deadline = None if overall_timeout_s is None else monotonic() + max(0.0, float(overall_timeout_s))

    def _fill() -> None:
        while len(pending) < limit:
            try:
                task = next(task_iter)
            except StopIteration:
                return
            pending[executor.submit(worker, task)] = task

    _fill()
    while pending:
        remaining_s = None if deadline is None else deadline - monotonic()
        if remaining_s is not None and remaining_s <= 0.0:
            for future in pending:
                future.cancel()
            # ProcessPoolExecutor cancellation cannot stop work that has
            # already started. Terminate only this executor's children so a
            # surrounding context manager cannot block indefinitely on exit.
            abort_process_pool(executor)
            raise TimeoutError(
                "Parallel execution exceeded the overall timeout of "
                f"{float(overall_timeout_s):.3f} s with {len(pending)} task(s) pending."
            )
        wait_timeout_s = max(float(poll_interval_s), 0.0)
        if remaining_s is not None:
            wait_timeout_s = min(wait_timeout_s, max(remaining_s, 0.0))
        done_now, _ = wait(
            set(pending),
            timeout=wait_timeout_s,
            return_when=FIRST_COMPLETED,
        )
        if not done_now:
            yield None, None
            continue
        for future in done_now:
            yield future, pending.pop(future)
        _fill()


def format_parallel_fallback_reason(exc: BaseException) -> str:
    detail = str(exc).strip()
    if detail:
        return f"{type(exc).__name__}: {detail}"
    return f"{type(exc).__name__}: worker process exited without an error message"


_LOGGER = logging.getLogger(__name__)


def _raise_if_cancellation(exc: BaseException) -> None:
    if isinstance(exc, (CancelledError, KeyboardInterrupt, SystemExit)) or type(exc).__name__ in {
        "MCPExecutionCancelled",
        "ThreadCampaignCancelled",
    }:
        raise exc


def _drain_parallel_progress_events(
    progress_queue: Any,
    *,
    step_callback: Callable[[int, int], None] | None,
    batch_progress_callback: Callable[[dict[str, Any]], None] | None,
    callback_label: str,
) -> tuple[Callable[[int, int], None] | None, Callable[[dict[str, Any]], None] | None]:
    """Relay worker progress events to callbacks owned by the parent process."""
    if progress_queue is None:
        return step_callback, batch_progress_callback
    while True:
        try:
            event = dict(progress_queue.get_nowait() or {})
        except queue_mod.Empty:
            break
        except Exception:
            break
        if step_callback is not None and str(event.get("event", "")) == "step":
            try:
                step_callback(int(event.get("step", 0)), int(event.get("total", 0)))
            except Exception as exc:
                _raise_if_cancellation(exc)
                _LOGGER.warning("Disabling %s step callback after runtime error: %s", callback_label, exc)
                step_callback = None
        if batch_progress_callback is not None:
            try:
                batch_progress_callback(event)
            except Exception as exc:
                _raise_if_cancellation(exc)
                _LOGGER.warning("Disabling %s progress callback after runtime error: %s", callback_label, exc)
                batch_progress_callback = None
    return step_callback, batch_progress_callback
