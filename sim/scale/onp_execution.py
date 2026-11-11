from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import time
from concurrent.futures import FIRST_COMPLETED, CancelledError, Future, ProcessPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import yaml

from sim.api import SimulationWorkspace
from sim.config import load_simulation_yaml
from sim.execution.service import run_simulation_config_file
from sim.execution.thread_campaign import (
    ThreadCampaignCancelled,
    ThreadCampaignExecutor,
    ensure_thread_campaign_compatible,
)
from sim.execution.worker_budget import (
    CAMPAIGN_WORKER_COUNT_ENV,
    TOTAL_PROCESS_BUDGET_ENV,
    current_worker_budget,
    worker_budget_value,
)
from sim.review.workspace import ReviewWorkspace
from sim.scale._output import bind_output_writer, logicalize
from sim.scale.planner import (
    ScalePlannerLimitError,
    check_active_scale_planner_limits,
    enforce_scale_planner_limits,
    load_scale_planner_config,
    resolve_selected_object_ids,
)
from sim.scale.store import (
    HpopHandoffPacketRecord,
    OnpRefinementCaseRecord,
    OnpRefinementRunRecord,
    claim_onp_refinement_runs,
    finish_onp_refinement_run,
    load_hpop_handoff_packets,
    load_onp_refinement_cases,
    load_onp_refinement_runs,
    mark_onp_refinement_run_running,
    recover_incomplete_onp_refinement_runs,
)
from sim.security.output_writes import OutputWriter
from sim.utils.parallel import (
    abort_process_pool,
    iter_bounded_futures,
    restore_env_vars,
    set_parallel_worker_thread_limits,
)

ONP_EXECUTION_SCHEMA_VERSION = 1
ONP_CLOSEST_APPROACH_QUERY = """
SELECT sample_index, time_s, deputy_id, chief_id, range_km,
       range_rate_km_s, v_radial_km_s, v_intrack_km_s, v_crosstrack_km_s,
       (
           SELECT COUNT(*)
           FROM relative_state AS tied
           WHERE ABS(tied.range_km - relative_state.range_km) <= 1.0e-12
       ) AS minimum_range_tie_count
FROM relative_state
ORDER BY range_km ASC, time_s ASC
LIMIT 1
""".strip()


@enforce_scale_planner_limits("ONP refinement queue")
def run_onp_refinement_queue(
    config_path: str | Path,
    *,
    workers: int = 0,
    backend: str | None = None,
    retry_failed: bool = False,
    recover_incomplete: bool = False,
    limit: int | None = None,
    writer: OutputWriter | None = None,
) -> dict[str, Any]:
    """Claim, execute, and compare validated Scale ONP refinement cases.

    The default keeps the historical process-pool transport; ``thread_pool``
    is an explicit opt-in for independent generated ONP cases.
    """

    cfg = load_scale_planner_config(config_path)
    if cfg.mode != "run":
        raise ValueError("ONP refinement execution requires scale_catalog.mode: run.")
    selected_ids = resolve_selected_object_ids(cfg)
    cases = load_onp_refinement_cases(cfg.store_path, object_ids=selected_ids)
    validated_cases = [case for case in cases if case.status == "validated" and case.validation_status == "ok"]
    recovered = (
        recover_incomplete_onp_refinement_runs(
            cfg.store_path,
            case_ids=tuple(case.case_id for case in validated_cases),
        )
        if recover_incomplete
        else 0
    )
    packet_by_id = {
        packet.packet_id: packet for packet in load_hpop_handoff_packets(cfg.store_path, object_ids=selected_ids)
    }
    missing_packets = sorted(case.packet_id for case in validated_cases if case.packet_id not in packet_by_id)
    if missing_packets:
        raise ValueError(
            "Validated ONP refinement cases reference non-current handoff packets: " + ", ".join(missing_packets)
        )
    worker_count = _resolve_worker_count(workers, len(validated_cases))
    backend = _resolve_execution_backend(backend, worker_count=worker_count)
    if backend == "serial":
        worker_count = 1
    batch_run_id = _batch_run_id()
    claims = claim_onp_refinement_runs(
        cfg.store_path,
        cases=validated_cases,
        batch_run_id=batch_run_id,
        backend=backend,
        retry_failed=retry_failed,
        retry_interrupted=recover_incomplete,
        limit=limit,
    )
    worker_count = min(worker_count, max(len(claims), 1))
    case_by_id = {case.case_id: case for case in validated_cases}
    tasks = [
        _worker_task(
            run=claim,
            case=case_by_id[claim.case_id],
            packet=packet_by_id.get(case_by_id[claim.case_id].packet_id),
            store_path=cfg.store_path,
            allow_preparation_error=True,
        )
        for claim in claims
    ]

    results: list[dict[str, Any]] = []
    if tasks and worker_count == 1:
        for task in tasks:
            check_active_scale_planner_limits()
            results.append(_run_and_persist(cfg.store_path, task))
            check_active_scale_planner_limits()
    elif tasks:
        for task in tasks:
            mark_onp_refinement_run_running(cfg.store_path, execution_id=str(task["execution_id"]))
        if backend == "thread_pool":
            thread_budget = _thread_worker_budget(worker_count)
            thread_tasks = [{**task, "worker_budget": thread_budget} for task in tasks]
            thread_results, thread_error = _run_thread_pool_tasks(
                cfg.store_path,
                thread_tasks,
                worker_count=worker_count,
            )
            results.extend(thread_results)
            if thread_error is not None:
                # The owner has reconciled every submitted future before this
                # point.  Any remaining orchestration error, including an
                # owner-store persistence fault, must remain terminal instead
                # of being reported as a completed queue.
                raise thread_error
        else:
            previous_env = set_parallel_worker_thread_limits(default_threads="1")
            try:
                try:
                    executor = ProcessPoolExecutor(max_workers=worker_count)
                except (NotImplementedError, OSError, PermissionError) as exc:
                    fallback_warning = f"Parallel CPU backend unavailable; used serial fallback: {type(exc).__name__}: {exc}"
                    for task in tasks:
                        check_active_scale_planner_limits()
                        result = _execute_onp_case_worker(task)
                        result["backend"] = "serial_fallback"
                        result["warnings"] = [*result.get("warnings", []), fallback_warning]
                        _persist_result(cfg.store_path, result)
                        results.append(result)
                        check_active_scale_planner_limits()
                else:
                    try:
                        with executor:
                            try:
                                check_active_scale_planner_limits()
                                future_iter = iter_bounded_futures(
                                    executor,
                                    _execute_onp_case_worker,
                                    tasks,
                                    max_in_flight=max(1, worker_count * 2),
                                )
                                for future, task in future_iter:
                                    if future is None or task is None:
                                        check_active_scale_planner_limits()
                                        continue
                                    check_active_scale_planner_limits()
                                    try:
                                        result = dict(future.result())
                                    except ScalePlannerLimitError:
                                        raise
                                    except Exception as exc:
                                        result = _error_result(task, exc)
                                    _persist_result(cfg.store_path, result)
                                    results.append(result)
                                    check_active_scale_planner_limits()
                            except ScalePlannerLimitError:
                                # Process workers can be terminated safely at
                                # this boundary.  The owner then persists
                                # interrupted claims below; no serial fallback
                                # is allowed after a planner breach.
                                abort_process_pool(executor)
                                raise
                    except ScalePlannerLimitError as exc:
                        completed_ids = {str(result["execution_id"]) for result in results}
                        for task in tasks:
                            if str(task["execution_id"]) in completed_ids:
                                continue
                            result = _interrupted_result(task, exc)
                            _persist_result(cfg.store_path, result)
                            results.append(result)
                        raise
                    except Exception as exc:
                        completed_ids = {str(result["execution_id"]) for result in results}
                        for task in tasks:
                            if str(task["execution_id"]) in completed_ids:
                                continue
                            result = _interrupted_result(task, exc)
                            _persist_result(cfg.store_path, result)
                            results.append(result)
            except ScalePlannerLimitError as exc:
                # This also covers a planner breach during the serial fallback
                # used only when pool construction is unavailable.  Preserve
                # claim evidence and stop before starting another case.
                if executor is not None:
                    abort_process_pool(executor)
                completed_ids = {str(result["execution_id"]) for result in results}
                for task in tasks:
                    if str(task["execution_id"]) in completed_ids:
                        continue
                    result = _interrupted_result(task, exc)
                    _persist_result(cfg.store_path, result)
                    results.append(result)
                raise
            finally:
                restore_env_vars(previous_env)

    run_records = load_onp_refinement_runs(cfg.store_path, batch_run_id=batch_run_id)
    artifact_root = cfg.hpop_handoff_output_dir / "onp_refinement" / "executions" / _safe_batch_stem(batch_run_id)
    artifacts = _write_execution_artifacts(
        artifact_root,
        batch_run_id=batch_run_id,
        config_path=cfg.config_path,
        store_path=cfg.store_path,
        backend=backend,
        worker_count=worker_count,
        recovered=recovered,
        runs=run_records,
        writer=bind_output_writer(writer, cfg.hpop_handoff_output_dir),
    )
    completed = sum(run.status == "completed" for run in run_records)
    failed = sum(run.status in {"error", "blocked", "interrupted"} for run in run_records)
    return {
        "schema_version": ONP_EXECUTION_SCHEMA_VERSION,
        "status": "completed" if failed == 0 else "completed_with_errors",
        "workflow": {"name": cfg.name, "config_path": str(cfg.config_path), "mode": cfg.mode},
        "store_path": str(cfg.store_path),
        "batch_run_id": batch_run_id,
        "backend": backend,
        "actual_backends": sorted({run.backend for run in run_records}),
        "workers": worker_count,
        "validated_cases_seen": len(validated_cases),
        "cases_claimed": len(claims),
        "cases_completed": completed,
        "cases_failed": failed,
        "incomplete_claims_recovered": recovered,
        "retry_failed": bool(retry_failed),
        "output_dir": str(artifact_root),
        "artifacts": artifacts,
        "executions": [_run_manifest_entry(run) for run in run_records],
        "non_claims": [
            "OGP-vs-ONP deltas compare the handoff product to sampled ONP review-store evidence.",
            "The queue does not add interpolation, covariance, probability-of-collision, or external HPOP validation.",
        ],
    }


def _run_and_persist(store_path: Path, task: dict[str, Any]) -> dict[str, Any]:
    mark_onp_refinement_run_running(store_path, execution_id=str(task["execution_id"]))
    result = _execute_onp_case_worker(task)
    _persist_result(store_path, result)
    return result


def _run_thread_pool_tasks(
    store_path: Path,
    tasks: list[dict[str, Any]],
    *,
    worker_count: int,
) -> tuple[list[dict[str, Any]], BaseException | None]:
    """Run bounded ONP thread tasks and reconcile every submitted future.

    The generic campaign iterator intentionally yields task completions to its
    owner.  The ONP queue also needs a durable answer for futures that are
    still running when cancellation or a deadline arrives, so this narrow
    owner loop retains each future until shutdown and reconciles late results
    before any claim is marked interrupted.
    """

    if not tasks:
        return [], None
    try:
        executor = ThreadCampaignExecutor(
            max_workers=max(1, int(worker_count)),
            max_in_flight=max(1, int(worker_count)),
        )
    except (NotImplementedError, OSError, PermissionError, RuntimeError) as exc:
        fallback_warning = f"Thread backend unavailable; used serial fallback: {type(exc).__name__}: {exc}"
        fallback_results: list[dict[str, Any]] = []
        for task in tasks:
            check_active_scale_planner_limits()
            result = _execute_onp_case_worker(task)
            result["backend"] = "serial_fallback"
            result["warnings"] = [*result.get("warnings", []), fallback_warning]
            _persist_result(store_path, result)
            fallback_results.append(result)
            check_active_scale_planner_limits()
        return fallback_results, None

    pending: dict[Future[Any], dict[str, Any]] = {}
    unsubmitted: dict[str, dict[str, Any]] = {}
    results: list[dict[str, Any]] = []
    persisted_ids: set[str] = set()
    task_iter = iter(tasks)
    orchestration_error: BaseException | None = None
    persistence_failure_triggered = False
    persistence_failures: dict[str, BaseException] = {}
    deadline = (
        None
        if executor.overall_timeout_s is None
        else time.monotonic() + float(executor.overall_timeout_s)
    )

    def _persist_once(result: dict[str, Any]) -> None:
        execution_id = str(result["execution_id"])
        if execution_id in persisted_ids:
            return
        try:
            _persist_result(store_path, result)
        except BaseException as exc:
            persistence_failures[execution_id] = exc
            raise
        persistence_failures.pop(execution_id, None)
        persisted_ids.add(execution_id)
        results.append(result)

    def _fill() -> None:
        while len(pending) < max(1, int(worker_count)):
            try:
                task = next(task_iter)
            except StopIteration:
                return
            try:
                future = executor.submit(_execute_onp_case_worker, task)
            except BaseException:
                # Retain the consumed claim until the owner can persist its
                # interrupted terminal result; a submit/persistence fault
                # must not make the claim disappear from reconciliation.
                unsubmitted[str(task["execution_id"])] = task
                raise
            pending[future] = task

    def _deadline_error() -> TimeoutError:
        timeout = float(executor.overall_timeout_s or 0.0)
        return TimeoutError(
            "Thread campaign execution exceeded the overall timeout of "
            f"{timeout:.3f} s with {len(pending)} task(s) pending."
        )

    def _reconcile_future(future: Future[Any], task: dict[str, Any]) -> None:
        execution_id = str(task["execution_id"])
        if execution_id in persisted_ids:
            return
        if future.cancelled():
            _persist_once(
                _interrupted_result(
                    task,
                    orchestration_error
                    or ThreadCampaignCancelled("ONP thread task was cancelled before it started."),
                )
            )
            return
        try:
            result = dict(future.result())
        except ThreadCampaignCancelled as exc:
            _persist_once(_interrupted_result(task, exc))
        except CancelledError as exc:
            _persist_once(_interrupted_result(task, exc))
        except BaseException as exc:
            _persist_once(_error_result(task, exc))
        else:
            _persist_once(result)

    try:
        check_active_scale_planner_limits()
        _fill()
        while pending:
            check_active_scale_planner_limits()
            if deadline is not None and time.monotonic() >= deadline:
                orchestration_error = _deadline_error()
                executor.cancel()
                break
            timeout = 0.1
            if deadline is not None:
                timeout = min(timeout, max(0.0, deadline - time.monotonic()))
            done, _ = wait(tuple(pending), timeout=timeout, return_when=FIRST_COMPLETED)
            if not done:
                check_active_scale_planner_limits()
                continue
            for future in done:
                task = pending[future]
                check_active_scale_planner_limits()
                try:
                    result = dict(future.result())
                except ThreadCampaignCancelled as exc:
                    orchestration_error = exc
                    executor.cancel()
                    _persist_once(_interrupted_result(task, exc))
                    pending.pop(future, None)
                    break
                except CancelledError as exc:
                    orchestration_error = exc
                    executor.cancel()
                    _persist_once(_interrupted_result(task, exc))
                    pending.pop(future, None)
                    break
                except BaseException as exc:
                    _persist_once(_error_result(task, exc))
                    pending.pop(future, None)
                else:
                    _persist_once(result)
                    pending.pop(future, None)
                    check_active_scale_planner_limits()
            if orchestration_error is not None:
                break
            _fill()
    except ThreadCampaignCancelled as exc:
        if orchestration_error is None:
            orchestration_error = exc
        executor.cancel()
    except BaseException as exc:
        persistence_failure_triggered = bool(persistence_failures) and orchestration_error is None
        if orchestration_error is None:
            orchestration_error = exc
        executor.cancel()
    finally:
        if orchestration_error is not None:
            executor.cancel()
            # Claims that were never submitted cannot produce a late result.
            # Resolve them before waiting for running workers, while retaining
            # pending futures for exact completion/error reconciliation below.
            for task in task_iter:
                unsubmitted[str(task["execution_id"])] = task
            for execution_id, task in tuple(unsubmitted.items()):
                try:
                    _persist_once(_interrupted_result(task, orchestration_error))
                except BaseException as exc:
                    persistence_failure_triggered = True
                    orchestration_error = exc
                else:
                    unsubmitted.pop(execution_id, None)
        executor.shutdown(wait_for_running=True)
        # Futures that were running at cancellation can finish successfully
        # while shutdown waits.  Persist those exact results before recording
        # an interrupted/unknown outcome for anything that never produced one.
        for future, task in tuple(pending.items()):
            try:
                _reconcile_future(future, task)
            except BaseException as exc:
                # Keep the future in ``pending`` until its durable terminal
                # result is accepted.  A persistence fault must remain an
                # orchestration error rather than being reported complete.
                orchestration_error = exc
            else:
                pending.pop(future, None)
        if not pending and not unsubmitted and not persistence_failures and persistence_failure_triggered:
            # A transient owner-store fault was retried successfully during
            # reconciliation; no unresolved orchestration error remains.
            orchestration_error = None
        elif persistence_failures:
            orchestration_error = list(persistence_failures.values())[-1]

    return results, orchestration_error


def _raise_if_thread_task_cancelled(task: dict[str, Any]) -> None:
    event = task.get("cancel_event")
    if event is not None and bool(event.is_set()):
        raise ThreadCampaignCancelled("ONP thread worker cancelled cooperatively.")


def _thread_step_callback(task: dict[str, Any]):
    if task.get("cancel_event") is None:
        return None

    def _check_cancelled(_step: int, _total: int) -> None:
        _raise_if_thread_task_cancelled(task)

    return _check_cancelled


def _execute_onp_case_worker(task: dict[str, Any]) -> dict[str, Any]:
    config_path = Path(str(task["config_path"])).expanduser().resolve()
    output_dir = Path(str(task["output_dir"])).expanduser().resolve()
    _raise_if_thread_task_cancelled(task)
    base = {
        "execution_id": str(task["execution_id"]),
        "backend": str(task["backend"]),
        "output_dir": str(output_dir),
        "attempt": int(task.get("attempt", 1) or 1),
        "source_config_path": str(task.get("source_config_path", config_path)),
        "source_config_digest": str(task.get("source_config_digest", task.get("config_digest", ""))),
        "review_store_path": "",
        "run_summary": {},
        "comparison": {},
        "warnings": [],
    }
    try:
        preparation_error = str(task.get("_preparation_error", "") or "")
        if preparation_error:
            return {
                **base,
                "status": "blocked",
                "error": f"ONP execution preparation failed before worker submission: {preparation_error}",
            }
        actual_digest = hashlib.sha256(config_path.read_bytes()).hexdigest()
        if actual_digest != str(task["config_digest"]):
            return {
                **base,
                "status": "blocked",
                "error": "Generated ONP config digest changed after materialization; rematerialize before execution.",
            }
        validation = SimulationWorkspace().validate_candidate_config(config_path, trust_plugins=True)
        if not bool(validation.get("ok", False)):
            return {
                **base,
                "status": "blocked",
                "error": "Canonical safe and trusted validation failed before ONP execution.",
                "run_summary": {"validation": _json_safe(validation)},
            }
        if bool(task.get("_thread_campaign", False)):
            ensure_thread_campaign_compatible(
                load_simulation_yaml(config_path).to_dict(),
                label=f"Scale ONP refinement {task['execution_id']}",
            )
        _raise_if_thread_task_cancelled(task)
        step_callback = _thread_step_callback(task)
        payload = run_simulation_config_file(config_path, step_callback=step_callback)
        # The engine has returned a quiescent payload.  A late owner cancel
        # after the final step callback must not relabel a physically complete
        # run as interrupted; the saved review store remains the authority.
        workspace = ReviewWorkspace.open(output_dir)
        query_result = workspace.query(ONP_CLOSEST_APPROACH_QUERY, max_rows=1)
        if not query_result.rows:
            raise RuntimeError("ONP review store contains no relative_state samples.")
        closest = query_result.rows[0]
        relative_speed = math.sqrt(
            float(closest["v_radial_km_s"]) ** 2
            + float(closest["v_intrack_km_s"]) ** 2
            + float(closest["v_crosstrack_km_s"]) ** 2
        )
        onp_tca_jd_utc = float(task["analysis_start_jd_utc"]) + float(closest["time_s"]) / 86400.0
        comparison = {
            "evidence_query": ONP_CLOSEST_APPROACH_QUERY,
            "packet_id": str(task["packet_id"]),
            "candidate_id": str(task["candidate_id"]),
            "sample_index": int(closest["sample_index"]),
            "minimum_range_tie_count": int(closest["minimum_range_tie_count"]),
            "onp_closest_time_s": float(closest["time_s"]),
            "deputy_id": str(task["deputy_id"]),
            "chief_id": str(task["chief_id"]),
            "ogp_tca_jd_utc": float(task["ogp_tca_jd_utc"]),
            "onp_tca_jd_utc": onp_tca_jd_utc,
            "tca_delta_s": (onp_tca_jd_utc - float(task["ogp_tca_jd_utc"])) * 86400.0,
            "ogp_miss_distance_km": float(task["ogp_miss_distance_km"]),
            "onp_miss_distance_km": float(closest["range_km"]),
            "miss_distance_delta_km": float(closest["range_km"]) - float(task["ogp_miss_distance_km"]),
            "ogp_relative_speed_km_s": float(task["ogp_relative_speed_km_s"]),
            "onp_relative_speed_km_s": relative_speed,
            "relative_speed_delta_km_s": relative_speed - float(task["ogp_relative_speed_km_s"]),
            "onp_range_rate_km_s": _optional_float(closest["range_rate_km_s"]),
        }
        summary = {
            "scenario_name": str(payload.get("scenario_name", "")),
            "config_path": str(config_path),
            "validation": _json_safe(validation),
            "run": _json_safe(dict(payload.get("run", {}) or {})),
        }
        return {
            **base,
            "status": "completed",
            "review_store_path": str(workspace.db_path),
            "run_summary": summary,
            "comparison": comparison,
            "error": "",
        }
    except ThreadCampaignCancelled:
        raise
    except Exception as exc:
        return _error_result(task, exc, base=base)


def _worker_task(
    *,
    run: OnpRefinementRunRecord,
    case: OnpRefinementCaseRecord,
    packet: HpopHandoffPacketRecord | None,
    store_path: str | Path | None = None,
    allow_preparation_error: bool = False,
) -> dict[str, Any]:
    if packet is None:
        raise ValueError(f"ONP refinement case {case.case_id!r} has no current source handoff packet.")
    prior_thread_attempt = False
    preparation_error = ""
    try:
        if store_path is not None and int(run.attempt) > 1:
            prior_thread_attempt = any(
                prior.backend == "thread_pool" and int(prior.attempt) < int(run.attempt)
                for prior in load_onp_refinement_runs(store_path, case_id=case.case_id)
            )
        execution_config_path, execution_config_digest, execution_output_dir = _execution_provenance_paths(
            run=run,
            case=case,
            fresh_provenance=(str(run.backend) == "thread_pool" or prior_thread_attempt),
        )
    except Exception as exc:
        if not allow_preparation_error:
            raise
        preparation_error = f"{type(exc).__name__}: {exc}"
        execution_config_path = str(Path(case.generated_config_path).expanduser().resolve())
        execution_config_digest = str(case.config_digest)
        execution_output_dir = str(Path(case.scenario_output_dir).expanduser().resolve())
    task = {
        "execution_id": run.execution_id,
        "backend": run.backend,
        "case_id": case.case_id,
        "config_path": execution_config_path,
        "config_digest": execution_config_digest,
        "source_config_path": case.generated_config_path,
        "source_config_digest": case.config_digest,
        "output_dir": execution_output_dir,
        "attempt": int(run.attempt),
        "analysis_start_jd_utc": case.analysis_start_jd_utc,
        "packet_id": packet.packet_id,
        "candidate_id": packet.candidate_id,
        "deputy_id": packet.deputy_id,
        "chief_id": packet.chief_id,
        "ogp_tca_jd_utc": packet.tca_jd_utc,
        "ogp_miss_distance_km": packet.miss_distance_km,
        "ogp_relative_speed_km_s": packet.relative_speed_km_s,
    }
    if preparation_error:
        task["_preparation_error"] = preparation_error
    return task


def _execution_provenance_paths(
    *,
    run: OnpRefinementRunRecord,
    case: OnpRefinementCaseRecord,
    fresh_provenance: bool | None = None,
) -> tuple[str, str, str]:
    """Return a fresh config/output identity for explicit recovery attempts."""

    source_path = Path(case.generated_config_path).expanduser().resolve()
    source_output_dir = Path(case.scenario_output_dir).expanduser().resolve()
    if fresh_provenance is None:
        fresh_provenance = str(run.backend) == "thread_pool"
    # Keep the established process-pool retry identity unless this case has
    # crossed a prior thread attempt.  Thread retries, and mixed transport
    # retries after them, need a fresh identity so a late completion cannot
    # look like a replay of the same artifact.
    if int(run.attempt) <= 1 or not fresh_provenance:
        return str(source_path), str(case.config_digest), str(source_output_dir)

    attempt_tag = f"attempt-{int(run.attempt)}-{str(run.execution_id)[:8]}"
    execution_output_dir = source_output_dir.with_name(f"{source_output_dir.name}__{attempt_tag}")
    execution_config_path = source_path.with_name(f"{source_path.stem}__{attempt_tag}{source_path.suffix}")
    source_bytes = source_path.read_bytes()
    source_digest = hashlib.sha256(source_bytes).hexdigest()
    if source_digest != str(case.config_digest):
        raise ValueError(
            "Generated ONP source config changed after materialization; "
            f"expected digest {case.config_digest}, found {source_digest}. Rematerialize before recovery: {source_path}"
        )
    raw = yaml.safe_load(source_bytes)
    if not isinstance(raw, dict):
        raise ValueError(f"Generated ONP config is not a mapping: {source_path}")
    outputs = raw.setdefault("outputs", {})
    if not isinstance(outputs, dict):
        raise ValueError(f"Generated ONP config outputs must be a mapping: {source_path}")
    outputs["output_dir"] = str(execution_output_dir)
    metadata = raw.setdefault("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
        raw["metadata"] = metadata
    provenance = metadata.setdefault("scale_provenance", {})
    if not isinstance(provenance, dict):
        provenance = {}
        metadata["scale_provenance"] = provenance
    provenance.update(
        {
            "source_config_path": str(source_path),
            "source_config_digest": str(case.config_digest),
            "execution_attempt": int(run.attempt),
            "execution_id": str(run.execution_id),
            "execution_output_dir": str(execution_output_dir),
        }
    )
    execution_config_path.parent.mkdir(parents=True, exist_ok=True)
    execution_config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    execution_config_digest = hashlib.sha256(execution_config_path.read_bytes()).hexdigest()
    return str(execution_config_path), execution_config_digest, str(execution_output_dir)


def _persist_result(store_path: Path, result: dict[str, Any]) -> None:
    finish_onp_refinement_run(
        store_path,
        execution_id=str(result["execution_id"]),
        status=str(result["status"]),
        backend=str(result.get("backend", "")),
        output_dir=str(result.get("output_dir", "")),
        review_store_path=str(result.get("review_store_path", "")),
        run_summary=dict(result.get("run_summary", {}) or {}),
        comparison=dict(result.get("comparison", {}) or {}),
        error=str(result.get("error", "")),
        warnings=tuple(str(item) for item in result.get("warnings", [])),
    )


def _error_result(
    task: dict[str, Any],
    exc: BaseException,
    *,
    base: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result = dict(
        base
        or {
            "execution_id": str(task["execution_id"]),
            "backend": str(task["backend"]),
            "output_dir": str(task.get("output_dir", "")),
            "review_store_path": "",
            "run_summary": {},
            "comparison": {},
            "warnings": [],
        }
    )
    result.update({"status": "error", "error": f"{type(exc).__name__}: {exc}"})
    return result


def _interrupted_result(task: dict[str, Any], exc: BaseException) -> dict[str, Any]:
    result = _error_result(task, exc)
    result["status"] = "interrupted"
    if str(task.get("backend", "")) == "thread_pool":
        warning = "Thread-pool orchestration stopped before this case produced a terminal result."
    else:
        warning = "Process-pool orchestration stopped before this case produced a terminal result."
    result["warnings"] = [warning]
    return result


def _resolve_worker_count(requested: int, case_count: int) -> int:
    if int(requested) < 0:
        raise ValueError("workers must be zero (auto) or a positive integer.")
    if int(requested) == 0:
        requested = max((os.cpu_count() or 1) - 1, 1)
    worker_count = max(1, min(int(requested), max(int(case_count), 1)))
    for name in (CAMPAIGN_WORKER_COUNT_ENV, TOTAL_PROCESS_BUDGET_ENV):
        raw_budget = worker_budget_value(name)
        if raw_budget in (None, ""):
            continue
        try:
            budget = int(raw_budget)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be an integer when set.") from exc
        if budget > 0:
            worker_count = min(worker_count, budget)
    return max(1, worker_count)


def _resolve_execution_backend(requested: str | None, *, worker_count: int) -> str:
    """Normalize the explicit ONP queue transport without changing old calls."""

    value = str(requested or "").strip().lower()
    if not value:
        return "serial" if int(worker_count) == 1 else "parallel_cpu"
    if value == "serial":
        return "serial"
    if value in {"parallel_cpu", "process_pool"}:
        return "serial" if int(worker_count) == 1 else "parallel_cpu"
    if value == "thread_pool":
        return "serial" if int(worker_count) == 1 else "thread_pool"
    raise ValueError(
        "ONP refinement execution backend must be serial, process_pool, parallel_cpu, or thread_pool."
    )


def _thread_worker_budget(worker_count: int) -> dict[str, int]:
    """Return plain budget data for one isolated ONP thread task."""

    inherited = current_worker_budget()
    if inherited is None:
        return {
            "object_workers_per_run": 0,
            "campaign_workers": max(1, int(worker_count)),
            "total_process_budget": max(1, int(worker_count)),
        }
    return {
        "object_workers_per_run": max(0, int(inherited.object_workers_per_run)),
        "campaign_workers": max(1, min(int(inherited.campaign_workers), int(worker_count))),
        "total_process_budget": max(1, min(int(inherited.total_process_budget), int(worker_count))),
    }


def _batch_run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"onp_queue:{stamp}:{uuid4().hex[:8]}"


def _safe_batch_stem(batch_run_id: str) -> str:
    return batch_run_id.replace(":", "_")


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _write_execution_artifacts(
    root: Path,
    *,
    batch_run_id: str,
    config_path: Path,
    store_path: Path,
    backend: str,
    worker_count: int,
    recovered: int,
    runs: list[OnpRefinementRunRecord],
    writer: Any | None = None,
) -> dict[str, str]:
    if writer is None:
        root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    comparison_path = root / "ogp_vs_onp_comparison.csv"
    summary_json_path = root / "summary.json"
    summary_md_path = root / "summary.md"
    entries = [_run_manifest_entry(run) for run in runs]
    manifest = {
        "schema_version": ONP_EXECUTION_SCHEMA_VERSION,
        "batch_run_id": batch_run_id,
        "config_path": str(config_path),
        "store_path": str(store_path),
        "backend": backend,
        "workers": worker_count,
        "incomplete_claims_recovered": recovered,
        "closest_approach_query": ONP_CLOSEST_APPROACH_QUERY,
        "executions": entries,
    }
    if writer is not None:
        manifest = logicalize(manifest, writer=writer)
    serialized_manifest = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    if writer is None:
        manifest_path.write_text(serialized_manifest, encoding="utf-8")
    else:
        writer.write_text(manifest_path, serialized_manifest)
    comparison_fields = [
        "execution_id",
        "case_id",
        "packet_id",
        "candidate_id",
        "deputy_id",
        "chief_id",
        "sample_index",
        "minimum_range_tie_count",
        "onp_closest_time_s",
        "ogp_tca_jd_utc",
        "onp_tca_jd_utc",
        "tca_delta_s",
        "ogp_miss_distance_km",
        "onp_miss_distance_km",
        "miss_distance_delta_km",
        "ogp_relative_speed_km_s",
        "onp_relative_speed_km_s",
        "relative_speed_delta_km_s",
        "onp_range_rate_km_s",
        "review_store_path",
    ]
    comparison_stream = io.StringIO(newline="")
    csv_writer = csv.DictWriter(comparison_stream, fieldnames=comparison_fields)
    csv_writer.writeheader()
    for run in runs:
        if not run.comparison:
            continue
        row = {field: run.comparison.get(field, "") for field in comparison_fields}
        row.update({"execution_id": run.execution_id, "case_id": run.case_id})
        row["review_store_path"] = run.review_store_path
        if writer is not None:
            row = logicalize(row, writer=writer)
        csv_writer.writerow(row)
    comparison_payload = comparison_stream.getvalue().encode("utf-8")
    if writer is None:
        comparison_path.write_bytes(comparison_payload)
    else:
        writer.write_bytes(comparison_path, comparison_payload)
    completed = sum(run.status == "completed" for run in runs)
    failed = sum(run.status in {"error", "blocked", "interrupted"} for run in runs)
    comparisons = []
    for run in runs:
        if not run.comparison:
            continue
        comparison = dict(run.comparison)
        comparison.update(
            {
                "execution_id": run.execution_id,
                "case_id": run.case_id,
                "review_store_path": run.review_store_path,
            }
        )
        comparisons.append(comparison)
    interpretation = _comparison_interpretation(comparisons)
    non_claims = [
        "Closest approach is the minimum sampled range returned by the recorded review-store query.",
        "The configured ONP case is a deterministic numerical refinement; this packet does not establish broad higher-fidelity superiority.",
        "No interpolation, covariance propagation, collision probability, calibrated risk class, or external validation is included.",
    ]
    summary = {
        "schema_version": ONP_EXECUTION_SCHEMA_VERSION,
        "batch_run_id": batch_run_id,
        "backend": backend,
        "cases_claimed": len(runs),
        "cases_completed": completed,
        "cases_failed": failed,
        "comparison_rows": len(comparisons),
        "closest_approach_query": ONP_CLOSEST_APPROACH_QUERY,
        "comparisons": comparisons,
        "interpretation": interpretation,
        "non_claims": non_claims,
    }
    if writer is not None:
        summary = logicalize(summary, writer=writer)
    serialized_summary = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    serialized_markdown = _comparison_markdown(
        batch_run_id=batch_run_id,
        runs=runs,
        completed=completed,
        failed=failed,
        comparisons=comparisons,
        interpretation=interpretation,
        non_claims=non_claims,
    )
    if writer is None:
        summary_json_path.write_text(serialized_summary, encoding="utf-8")
        summary_md_path.write_text(serialized_markdown, encoding="utf-8")
    else:
        writer.write_text(summary_json_path, serialized_summary)
        writer.write_text(summary_md_path, serialized_markdown)
    return {
        "manifest_json": str(manifest_path),
        "comparison_csv": str(comparison_path),
        "summary_json": str(summary_json_path),
        "summary_markdown": str(summary_md_path),
    }


def _comparison_interpretation(comparisons: list[dict[str, Any]]) -> list[str]:
    if not comparisons:
        return ["No completed OGP-vs-ONP comparison row was produced."]
    messages = []
    for comparison in comparisons:
        miss_delta = float(comparison["miss_distance_delta_km"])
        tca_delta = float(comparison["tca_delta_s"])
        speed_delta = float(comparison["relative_speed_delta_km_s"])
        tie_count = int(comparison.get("minimum_range_tie_count", 1))
        if tie_count > 1:
            messages.append(
                f"Case {comparison['case_id']} has {tie_count} ONP samples tied at the minimum range; "
                f"the ordered query reports the earliest sample, so its {tca_delta:.12g} s TCA delta is not unique. "
                f"Sampled miss-distance delta is {miss_delta:.12g} km and relative-speed delta is {speed_delta:.12g} km/s."
            )
        elif all(abs(value) <= 1.0e-12 for value in (miss_delta, tca_delta, speed_delta)):
            messages.append(
                f"Case {comparison['case_id']} produced no sampled OGP-vs-ONP change in TCA, miss distance, or relative speed."
            )
        else:
            messages.append(
                f"Case {comparison['case_id']} changed sampled miss distance by {miss_delta:.12g} km, "
                f"TCA by {tca_delta:.12g} s, and relative speed by {speed_delta:.12g} km/s."
            )
    return messages


def _comparison_markdown(
    *,
    batch_run_id: str,
    runs: list[OnpRefinementRunRecord],
    completed: int,
    failed: int,
    comparisons: list[dict[str, Any]],
    interpretation: list[str],
    non_claims: list[str],
) -> str:
    lines = [
        "# OGP vs ONP sampled closest-approach comparison",
        "",
        f"- Batch run: `{batch_run_id}`",
        f"- Claimed cases: {len(runs)}",
        f"- Completed cases: {completed}",
        f"- Failed or blocked cases: {failed}",
        f"- Comparison rows: {len(comparisons)}",
        "",
    ]
    if comparisons:
        lines.extend(
            [
                "| Case | OGP miss (km) | ONP sampled miss (km) | Delta (km) | TCA delta (s) | Min-range ties | Relative-speed delta (km/s) |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for comparison in comparisons:
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(comparison["case_id"]),
                        _format_metric(comparison["ogp_miss_distance_km"]),
                        _format_metric(comparison["onp_miss_distance_km"]),
                        _format_metric(comparison["miss_distance_delta_km"]),
                        _format_metric(comparison["tca_delta_s"]),
                        str(comparison.get("minimum_range_tie_count", 1)),
                        _format_metric(comparison["relative_speed_delta_km_s"]),
                    ]
                )
                + " |"
            )
        lines.append("")
    lines.extend(["## Interpretation", ""])
    lines.extend(f"- {item}" for item in interpretation)
    lines.extend(["", "## Evidence query", "", "```sql", ONP_CLOSEST_APPROACH_QUERY, "```", ""])
    lines.extend(["## Non-claims", ""])
    lines.extend(f"- {item}" for item in non_claims)
    lines.append("")
    return "\n".join(lines)


def _format_metric(value: Any) -> str:
    if value is None:
        return "undefined"
    return f"{float(value):.12g}"


def _run_manifest_entry(run: OnpRefinementRunRecord) -> dict[str, Any]:
    return {
        "execution_id": run.execution_id,
        "batch_run_id": run.batch_run_id,
        "case_id": run.case_id,
        "config_digest": run.config_digest,
        "backend": run.backend,
        "attempt": run.attempt,
        "status": run.status,
        "output_dir": run.output_dir,
        "review_store_path": run.review_store_path,
        "input_digest": run.input_digest,
        "comparison": run.comparison,
        "error": run.error,
        "warnings": list(run.warnings),
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return value
