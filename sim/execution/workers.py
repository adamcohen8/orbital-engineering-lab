from __future__ import annotations

import os
from typing import Any

from sim.config import scenario_config_from_dict, validate_scenario_plugins
from sim.execution.campaign_retention import compact_iteration_result, policy_from_config
from sim.execution.metrics import (
    closest_approach_from_run_payload,
    relative_motion_summary_from_run_payload,
    relative_range_series_from_run_payload,
)
from sim.single_run import _run_single_config
from sim.utils.parallel import (
    initialize_worker_progress_queue,
    restore_env_vars,
    set_parallel_worker_thread_limits,
    worker_progress_queue,
)

_WORKER_CANCEL_EVENT: Any | None = None


def initialize_campaign_worker_transport(progress_queue: Any, cancel_event: Any) -> None:
    """Install cancellation at launch so ordinary steps use shared memory.

    A multiprocessing Event must be passed during process creation, rather
    than as a subsequently pickled task argument. Its local check avoids a
    Manager proxy request/response for every simulation sample.
    """
    global _WORKER_CANCEL_EVENT
    initialize_worker_progress_queue(progress_queue)
    _WORKER_CANCEL_EVENT = cancel_event


def run_mc_iteration_from_dict(task: dict[str, Any]) -> dict[str, Any]:
    iteration = int(task.get("iteration", 0))
    cdict = dict(task.get("config_dict", {}) or {})
    strict_plugins = bool(task.get("strict_plugins", True))
    progress_queue = task.get("progress_queue")
    cancel_event = task.get("cancel_event", _WORKER_CANCEL_EVENT)
    if progress_queue is None:
        progress_queue = worker_progress_queue()
    emit_every = int(task.get("progress_emit_every", 20) or 20)
    emit_every = max(1, emit_every)
    collect_relative_range_series = bool(task.get("collect_relative_range_series", False))
    collect_payload = bool(task.get("collect_payload", False))
    ci = scenario_config_from_dict(cdict)
    if strict_plugins:
        errs = validate_scenario_plugins(ci)
        if errs:
            msg = f"Plugin validation failed in Monte Carlo iteration {iteration}:\n- " + "\n- ".join(errs)
            raise ValueError(msg)

    last_emit = -(10**9)

    def _on_step(step: int, total: int) -> None:
        nonlocal last_emit
        if cancel_event is not None and cancel_event.is_set():
            if bool(task.get("_thread_campaign", False)):
                from sim.execution.thread_campaign import ThreadCampaignCancelled

                raise ThreadCampaignCancelled("Thread campaign worker cancelled.")
            raise RuntimeError("Parallel campaign worker cancelled.")
        if progress_queue is None:
            return
        s = max(int(step), 0)
        t = max(int(total), 0)
        should_emit = (s == 0) or (t > 0 and s >= t) or (s - last_emit >= emit_every)
        if not should_emit:
            return
        last_emit = s
        try:
            progress_queue.put(
                {
                    "event": "step",
                    "pid": int(os.getpid()),
                    "iteration": int(iteration),
                    "step": int(s),
                    "total": int(t),
                }
            )
        except Exception:
            pass

    ro = _run_single_config(
        ci,
        step_callback=_on_step if progress_queue is not None or cancel_event is not None else None,
    )
    if progress_queue is not None:
        try:
            progress_queue.put(
                {
                    "event": "done",
                    "pid": int(os.getpid()),
                    "iteration": int(iteration),
                }
            )
        except Exception:
            pass
    result = {
        "iteration": iteration,
        "summary": ro["summary"],
        "closest_approach_km": closest_approach_from_run_payload(ro),
        "derived": {
            "relative_motion": relative_motion_summary_from_run_payload(ro),
        },
    }
    rel = dict(result["derived"]["relative_motion"])
    for key in ("initial_range_km", "final_range_km", "final_relative_speed_m_s", "max_relative_speed_m_s"):
        if key in rel:
            result["derived"][key] = rel.get(key)
    if collect_payload:
        result["payload"] = ro
    if collect_relative_range_series:
        result["relative_range_series"] = relative_range_series_from_run_payload(ro)
    policy = policy_from_config(cdict)
    if policy:
        result = compact_iteration_result(
            ro, result, policy=policy,
            metric_paths=list(task.get("metric_paths", []) or []),
            output_dir=str(dict(cdict.get("outputs", {}) or {}).get("output_dir", "")),
        )
    return result


_set_parallel_worker_thread_limits = set_parallel_worker_thread_limits
_restore_env_vars = restore_env_vars
_run_mc_iteration_from_dict = run_mc_iteration_from_dict
