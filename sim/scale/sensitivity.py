from __future__ import annotations

import hashlib
import json
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from dataclasses import replace
from itertools import islice, product
from pathlib import Path
from typing import Any

import numpy as np

from sim.execution.thread_campaign import (
    ThreadCampaignBackendUnavailable,
    ThreadCampaignCancelled,
    ThreadCampaignExecutor,
)
from sim.numeric_backend import normalize_numeric_backend
from sim.scale.campaign_execution import (
    build_campaign_trial_result,
    ensure_scale_thread_builder_compatible,
    load_worker_scale_planner_config,
    raise_if_thread_task_cancelled,
    resolve_scale_campaign_worker_count,
    scale_thread_worker_budget,
)
from sim.scale.monte_carlo import (
    attach_numba_chunked_propagation_summaries,
    campaign_run_telemetry,
    evaluate_quality_gates,
    latest_results_by_trial,
    normalize_propagation_summary_backend,
    serial_metric_stats,
    summary_metric_names,
)
from sim.scale.monte_carlo import (
    pending_trials as select_pending_trials,
)
from sim.scale.monte_carlo import (
    resolved_chunk_duration_s as resolve_chunk_duration_s,
)
from sim.scale.planner import (
    ScalePlannerConfig,
    ScalePlannerLimitError,
    check_active_scale_planner_limits,
    enforce_scale_planner_limits,
    load_scale_planner_config,
    resolve_selected_object_ids,
)
from sim.scale.store import (
    CampaignTrialResultRecord,
    SensitivityCampaignRecord,
    SensitivityTrialRecord,
    load_campaign_trial_results,
    load_current_tle_states,
    load_sensitivity_trials,
    write_campaign_trial_results,
    write_sensitivity_campaign,
)
from sim.utils.parallel import abort_process_pool, restore_env_vars, set_parallel_worker_thread_limits


@enforce_scale_planner_limits("sensitivity materialization")
def run_sensitivity_campaign(config_path: str | Path) -> dict[str, Any]:
    cfg = load_scale_planner_config(config_path)
    if cfg.mode != "run":
        raise ValueError("Sensitivity campaign materialization requires scale_catalog.mode: run.")
    if not cfg.sensitivity_enabled:
        raise ValueError("Sensitivity campaign materialization requires scale_catalog.sensitivity.enabled: true.")
    campaign, trials = build_sensitivity_campaign(cfg)
    cfg.sensitivity_output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = _write_campaign_manifest(cfg, campaign=campaign, trials=trials)
    write_summary = write_sensitivity_campaign(cfg.store_path, campaign=campaign, trials=trials)
    return {
        "schema_version": 1,
        "status": "completed",
        "workflow": {
            "name": cfg.name,
            "config_path": str(cfg.config_path),
            "mode": cfg.mode,
        },
        "store_path": str(cfg.store_path),
        "output_dir": str(cfg.sensitivity_output_dir),
        "manifest_path": str(manifest_path),
        "campaign_id": campaign.campaign_id,
        "campaign_type": campaign.campaign_type,
        "uncertainty_interpretation": campaign.uncertainty_interpretation,
        "target": campaign.target,
        "target_object_count": len(trials[0].target_object_ids) if trials else 0,
        "trials_seen": write_summary["trials_seen"],
        "trials_recorded": write_summary["trials_inserted"],
        "campaigns_recorded": write_summary["campaigns_inserted"],
        "non_claims": [
            "Sensitivity v1 materializes analyst-defined campaign trials only.",
            "No user trial builder was imported or executed.",
            "No perturbed SGP4 products or covariance-derived probability were generated.",
        ],
    }


@enforce_scale_planner_limits("serial sensitivity execution")
def run_sensitivity_serial(
    config_path: str | Path,
    *,
    limit: int = 0,
    propagate_sgp4: bool = False,
    duration_s: float | None = None,
    step_s: float | None = None,
    retry_failed: bool = False,
    propagation_summary_backend: str = "scalar_trial",
    chunk_duration_s: float | None = None,
    numeric_backend: str = "rust",
) -> dict[str, Any]:
    return _run_sensitivity_execution(
        config_path,
        backend="serial",
        max_workers=1,
        limit=limit,
        propagate_sgp4=propagate_sgp4,
        duration_s=duration_s,
        step_s=step_s,
        retry_failed=retry_failed,
        propagation_summary_backend=propagation_summary_backend,
        chunk_duration_s=chunk_duration_s,
        numeric_backend=numeric_backend,
    )


@enforce_scale_planner_limits("parallel sensitivity execution")
def run_sensitivity_parallel_cpu(
    config_path: str | Path,
    *,
    limit: int = 0,
    workers: int = 0,
    propagate_sgp4: bool = False,
    duration_s: float | None = None,
    step_s: float | None = None,
    retry_failed: bool = False,
    propagation_summary_backend: str = "scalar_trial",
    chunk_duration_s: float | None = None,
    numeric_backend: str = "rust",
    backend: str = "parallel_cpu",
) -> dict[str, Any]:
    execution_backend = _normalize_parallel_backend(backend)
    worker_count = resolve_scale_campaign_worker_count(workers)
    return _run_sensitivity_execution(
        config_path,
        backend=execution_backend,
        max_workers=worker_count,
        limit=limit,
        propagate_sgp4=propagate_sgp4,
        duration_s=duration_s,
        step_s=step_s,
        retry_failed=retry_failed,
        propagation_summary_backend=propagation_summary_backend,
        chunk_duration_s=chunk_duration_s,
        numeric_backend=numeric_backend,
    )


@enforce_scale_planner_limits("threaded sensitivity execution")
def run_sensitivity_threaded(
    config_path: str | Path,
    *,
    limit: int = 0,
    workers: int = 0,
    propagate_sgp4: bool = False,
    duration_s: float | None = None,
    step_s: float | None = None,
    retry_failed: bool = False,
    propagation_summary_backend: str = "scalar_trial",
    chunk_duration_s: float | None = None,
    numeric_backend: str = "python",
) -> dict[str, Any]:
    """Run independent Scale sensitivity trials on an explicit thread pool."""

    worker_count = resolve_scale_campaign_worker_count(workers)
    return _run_sensitivity_execution(
        config_path,
        backend="thread_pool",
        max_workers=worker_count,
        limit=limit,
        propagate_sgp4=propagate_sgp4,
        duration_s=duration_s,
        step_s=step_s,
        retry_failed=retry_failed,
        propagation_summary_backend=propagation_summary_backend,
        chunk_duration_s=chunk_duration_s,
        numeric_backend=numeric_backend,
    )


def _run_sensitivity_execution(
    config_path: str | Path,
    *,
    backend: str,
    max_workers: int,
    limit: int = 0,
    propagate_sgp4: bool = False,
    duration_s: float | None = None,
    step_s: float | None = None,
    retry_failed: bool = False,
    propagation_summary_backend: str = "scalar_trial",
    chunk_duration_s: float | None = None,
    numeric_backend: str = "rust",
) -> dict[str, Any]:
    wall_start = time.perf_counter()
    phase_durations: dict[str, float] = {}
    phase_start = time.perf_counter()
    cfg = load_scale_planner_config(config_path)
    if cfg.mode != "run":
        raise ValueError("Sensitivity execution requires scale_catalog.mode: run.")
    if not cfg.sensitivity_enabled:
        raise ValueError("Sensitivity execution requires scale_catalog.sensitivity.enabled: true.")
    numeric_backend = normalize_numeric_backend(numeric_backend)
    if numeric_backend not in {"python", "rust"}:
        raise ValueError("numeric_backend must be python or rust.")
    summary_backend = normalize_propagation_summary_backend(
        propagate_sgp4=propagate_sgp4,
        propagation_summary_backend=propagation_summary_backend,
    )
    resolved_chunk_duration_s = (
        None if summary_backend != "numba_chunked" else resolve_chunk_duration_s(chunk_duration_s)
    )

    campaign, planned_trials = build_sensitivity_campaign(cfg)
    write_sensitivity_campaign(cfg.store_path, campaign=campaign, trials=planned_trials)
    trials = load_sensitivity_trials(cfg.store_path, campaign_id=campaign.campaign_id, limit=max(int(limit), 0))
    if not trials:
        raise ValueError(f"No materialized sensitivity trials found for campaign {campaign.campaign_id}.")
    run_id = _sensitivity_run_id(
        cfg,
        campaign=campaign,
        trials=trials,
        backend=backend,
        limit=limit,
        propagate_sgp4=propagate_sgp4,
        duration_s=duration_s,
        step_s=step_s,
        propagation_summary_backend=summary_backend,
        chunk_duration_s=resolved_chunk_duration_s,
        numeric_backend=numeric_backend,
    )
    existing_results = load_campaign_trial_results(cfg.store_path, campaign_id=campaign.campaign_id, run_id=run_id)
    existing_by_trial = latest_results_by_trial(existing_results, trials=trials)
    pending_trials = select_pending_trials(trials, existing_by_trial=existing_by_trial, retry_failed=retry_failed)
    execution_cfg = _sensitivity_execution_cfg(cfg)
    trial_builder_propagate_sgp4 = bool(propagate_sgp4 and summary_backend == "scalar_trial")
    phase_durations["setup_s"] = time.perf_counter() - phase_start
    phase_start = time.perf_counter()
    if backend == "serial":
        new_results = []
        parallel_fallback_reason = ""
        for trial in pending_trials:
            check_active_scale_planner_limits()
            result = _sensitivity_trial_result(
                cfg,
                campaign=campaign,
                trial=trial,
                run_id=run_id,
                backend=backend,
                propagate_sgp4=trial_builder_propagate_sgp4,
                duration_s=duration_s,
                step_s=step_s,
            )
            check_active_scale_planner_limits()
            write_campaign_trial_results(cfg.store_path, results=[result])
            new_results.append(result)
    elif backend == "parallel_cpu":
        new_results, parallel_fallback_reason = _parallel_sensitivity_trial_results(
            cfg,
            campaign=campaign,
            trials=pending_trials,
            run_id=run_id,
            max_workers=max_workers,
            propagate_sgp4=trial_builder_propagate_sgp4,
            duration_s=duration_s,
            step_s=step_s,
        )
    elif backend == "thread_pool":
        new_results, parallel_fallback_reason = _thread_pool_sensitivity_trial_results(
            cfg,
            campaign=campaign,
            trials=pending_trials,
            run_id=run_id,
            max_workers=max_workers,
            propagate_sgp4=trial_builder_propagate_sgp4,
            duration_s=duration_s,
            step_s=step_s,
        )
    else:
        raise ValueError(f"Unsupported sensitivity execution backend: {backend}")
    phase_durations["trial_execution_s"] = time.perf_counter() - phase_start
    propagation_summary_telemetry: dict[str, Any] = {}
    results_to_write = list(new_results)
    if propagate_sgp4 and summary_backend == "numba_chunked":
        fresh_trial_ids = {result.trial_id for result in new_results}
        resumable_results = [
            result
            for trial_id, result in existing_by_trial.items()
            if trial_id not in fresh_trial_ids and needs_numba_chunked_summary(result)
        ]
        summary_candidates = [*new_results, *resumable_results]
        if summary_candidates:
            results_to_write, propagation_summary_telemetry = attach_numba_chunked_propagation_summaries(
                execution_cfg,
                results=summary_candidates,
                duration_s=duration_s,
                step_s=step_s,
                chunk_duration_s=float(resolved_chunk_duration_s),
            )
            phase_durations["chunked_propagation_summary_s"] = float(
                propagation_summary_telemetry.get("wall_time_s", 0.0)
            )
    phase_start = time.perf_counter()
    write_summary = (
        write_campaign_trial_results(cfg.store_path, results=results_to_write, replace_existing=True)
        if results_to_write
        else {"results_seen": 0, "results_inserted": 0}
    )
    phase_durations["result_store_write_s"] = time.perf_counter() - phase_start

    if results_to_write:
        latest_results = load_campaign_trial_results(cfg.store_path, campaign_id=campaign.campaign_id, run_id=run_id)
        results_by_trial = latest_results_by_trial(latest_results, trials=trials)
    else:
        results_by_trial = existing_by_trial
    results = [results_by_trial[trial.trial_id] for trial in trials if trial.trial_id in results_by_trial]
    if not results:
        raise ValueError(f"No sensitivity trial results were available for run {run_id}.")
    status_counts: dict[str, int] = {}
    for result in results:
        status_counts[result.status] = status_counts.get(result.status, 0) + 1
    run_status = "error" if status_counts.get("error") else "blocked" if status_counts.get("blocked") else "completed"
    telemetry = campaign_run_telemetry(
        results=results,
        fresh_results=new_results,
        propagate_sgp4=propagate_sgp4,
        propagation_summary_backend=summary_backend,
        phase_durations=phase_durations,
        propagation_summary_telemetry=propagation_summary_telemetry,
        wall_time_s=time.perf_counter() - wall_start,
    )
    cfg.sensitivity_output_dir.mkdir(parents=True, exist_ok=True)
    fresh_backends = {result.backend for result in new_results}
    reported_backend = backend
    if backend in {"parallel_cpu", "thread_pool"} and parallel_fallback_reason:
        reported_backend = (
            f"{backend}_with_serial_fallback"
            if backend in fresh_backends
            else "serial_fallback"
        )
    if parallel_fallback_reason:
        telemetry["parallel_fallback_reason"] = parallel_fallback_reason
        telemetry["requested_backend"] = backend
        telemetry["actual_backend"] = reported_backend
    manifest_path = _write_run_manifest(
        cfg,
        campaign=campaign,
        run_id=run_id,
        results=results,
        limit=limit,
        backend=reported_backend,
        max_workers=max_workers,
        retry_failed=retry_failed,
        propagate_sgp4=propagate_sgp4,
        duration_s=duration_s,
        step_s=step_s,
        propagation_summary_backend=summary_backend,
        chunk_duration_s=resolved_chunk_duration_s,
        telemetry=telemetry,
        numeric_backend=numeric_backend,
    )
    summary_path, summary_markdown_path, summary = _write_run_summary(
        cfg,
        campaign=campaign,
        run_id=run_id,
        results=results,
        limit=limit,
        backend=reported_backend,
        max_workers=max_workers,
        retry_failed=retry_failed,
        propagate_sgp4=propagate_sgp4,
        duration_s=duration_s,
        step_s=step_s,
        run_status=run_status,
        status_counts=status_counts,
        manifest_path=manifest_path,
        propagation_summary_backend=summary_backend,
        chunk_duration_s=resolved_chunk_duration_s,
        telemetry=telemetry,
        execution_cfg=execution_cfg,
        numeric_backend=numeric_backend,
    )
    payload = {
        "schema_version": 1,
        "status": run_status,
        "quality_status": summary["quality"]["status"],
        "workflow": {
            "name": cfg.name,
            "config_path": str(cfg.config_path),
            "mode": cfg.mode,
        },
        "store_path": str(cfg.store_path),
        "output_dir": str(cfg.sensitivity_output_dir),
        "manifest_path": str(manifest_path),
        "summary_path": str(summary_path),
        "summary_markdown_path": str(summary_markdown_path),
        "campaign_id": campaign.campaign_id,
        "run_id": run_id,
        "backend": backend,
        "max_workers": max_workers,
        "propagate_sgp4": bool(propagate_sgp4),
        "propagation_summary_backend": summary_backend,
        "chunk_duration_s": resolved_chunk_duration_s,
        "trials_seen": len(trials),
        "trials_pending": len(pending_trials),
        "trials_reused": len(trials) - len(pending_trials),
        "retry_failed": bool(retry_failed),
        "results_attempted": write_summary["results_seen"],
        "results_recorded": write_summary["results_inserted"],
        "status_counts": status_counts,
        "telemetry": summary["telemetry"],
        "non_claims": _execution_non_claims(propagate_sgp4=propagate_sgp4),
    }
    if numeric_backend == "rust":
        payload["numeric_backend"] = "rust"
    return payload


def build_sensitivity_campaign(
    cfg: ScalePlannerConfig,
) -> tuple[SensitivityCampaignRecord, list[SensitivityTrialRecord]]:
    if cfg.sensitivity_campaign_type != "parameter_sweep":
        raise ValueError("Phase 1F supports only sensitivity.campaign_type: parameter_sweep.")
    if cfg.sensitivity_target != "selected_objects":
        raise ValueError("Phase 1F supports only sensitivity.target: selected_objects.")
    if cfg.sensitivity_sweep_mode not in {"zip", "grid"}:
        raise ValueError("scale_catalog.sensitivity.sweep_mode must be zip or grid.")
    target_states = load_current_tle_states(
        cfg.store_path, object_ids=resolve_selected_object_ids(cfg)
    )
    target_object_ids = tuple(state.object_id for state in target_states)
    target_state_inputs = [
        {
            "object_id": state.object_id,
            "od_solution_id": state.od_solution_id,
            "input_digest": state.input_digest,
        }
        for state in target_states
    ]
    trial_parameters = _trial_parameter_sets(
        parameters=list(cfg.sensitivity_parameters),
        sweep_mode=cfg.sensitivity_sweep_mode,
        requested_trial_count=cfg.sensitivity_trial_count,
    )
    for row in trial_parameters:
        _apply_derived_parameters(row, cfg.sensitivity_derived_parameters)
    campaign_digest = _digest(
        {
            "workflow_name": cfg.name,
            "campaign_type": cfg.sensitivity_campaign_type,
            "target": cfg.sensitivity_target,
            "uncertainty_interpretation": cfg.sensitivity_uncertainty_interpretation,
            "seed": cfg.sensitivity_seed,
            "sweep_mode": cfg.sensitivity_sweep_mode,
            "parameters": list(cfg.sensitivity_parameters),
            "derived_parameters": list(cfg.sensitivity_derived_parameters),
            "frame": cfg.sensitivity_frame,
            "trial_builder": cfg.sensitivity_trial_builder,
            "outputs": cfg.sensitivity_outputs,
            "target_object_ids": list(target_object_ids),
            "target_state_inputs": target_state_inputs,
        }
    )
    campaign_id = f"sensitivity:{campaign_digest}"
    campaign = SensitivityCampaignRecord(
        campaign_id=campaign_id,
        workflow_name=cfg.name,
        campaign_type=cfg.sensitivity_campaign_type,
        target=cfg.sensitivity_target,
        uncertainty_interpretation=cfg.sensitivity_uncertainty_interpretation,
        seed=cfg.sensitivity_seed,
        requested_trial_count=cfg.sensitivity_trial_count,
        realized_trial_count=len(trial_parameters),
        sweep_mode=cfg.sensitivity_sweep_mode,
        parameters=cfg.sensitivity_parameters,
        frame=cfg.sensitivity_frame,
        trial_builder=cfg.sensitivity_trial_builder,
        outputs=cfg.sensitivity_outputs,
        output_dir=str(cfg.sensitivity_output_dir),
        input_digest=campaign_digest,
        status="planned",
        warnings=_campaign_warnings(cfg, target_object_ids=target_object_ids),
    )
    trials = [
        _trial_record(
            campaign_id=campaign_id,
            trial_index=index,
            parameters=params,
            target_object_ids=target_object_ids,
            frame=cfg.sensitivity_frame,
            trial_builder=cfg.sensitivity_trial_builder,
            campaign_digest=campaign_digest,
        )
        for index, params in enumerate(trial_parameters)
    ]
    return campaign, trials


def _trial_parameter_sets(
    *,
    parameters: list[dict[str, Any]],
    sweep_mode: str,
    requested_trial_count: int,
) -> list[dict[str, Any]]:
    if not parameters:
        return [{} for _ in range(max(requested_trial_count, 0))]
    values_by_name = [(str(parameter.get("name", "")).strip(), _parameter_values(parameter, requested_trial_count)) for parameter in parameters]
    if any(not name for name, _ in values_by_name):
        raise ValueError("Every sensitivity parameter requires a non-empty name.")
    names = [name for name, _ in values_by_name]
    if len(set(names)) != len(names):
        raise ValueError("Sensitivity parameter names must be unique.")
    if sweep_mode == "grid":
        value_grid = product(*(values for _, values in values_by_name))
        selected = islice(value_grid, requested_trial_count) if requested_trial_count > 0 else value_grid
        return [dict(zip(names, values, strict=True)) for values in selected]
    count = requested_trial_count or max(len(values) for _, values in values_by_name)
    out: list[dict[str, Any]] = []
    for trial_index in range(count):
        row: dict[str, Any] = {}
        for name, values in values_by_name:
            if len(values) == 1:
                row[name] = values[0]
            elif len(values) == count:
                row[name] = values[trial_index]
            else:
                raise ValueError(
                    f"Sensitivity parameter {name!r} has {len(values)} values but zip sweep requires 1 or {count}."
                )
        out.append(row)
    return out


def _parameter_values(parameter: dict[str, Any], requested_trial_count: int) -> list[Any]:
    mode = str(parameter.get("mode", "fixed") or "fixed").strip().lower()
    name = str(parameter.get("name", "")).strip() or "<unnamed>"
    if mode == "fixed":
        values = [parameter.get("value")]
        _validate_sensitivity_values(values, name=name)
        return values
    if mode == "values":
        values = list(parameter.get("values", []) or [])
        if not values:
            raise ValueError(f"Sensitivity parameter {parameter.get('name')!r} values mode requires values.")
        _validate_sensitivity_values(values, name=name)
        return values
    if mode == "linspace":
        raw_count = parameter.get("count", 0) or requested_trial_count or 0
        numeric_count = float(raw_count)
        if not np.isfinite(numeric_count) or not numeric_count.is_integer():
            raise ValueError(f"Sensitivity parameter {parameter.get('name')!r} linspace count must be an integer.")
        count = int(numeric_count)
        if count <= 0:
            raise ValueError(f"Sensitivity parameter {parameter.get('name')!r} linspace mode requires count or trial_count.")
        try:
            start = float(parameter.get("start", 0.0))
            stop = float(parameter.get("stop", 0.0))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Sensitivity parameter {name!r} linspace start/stop must be finite numbers.") from exc
        if not np.isfinite(start) or not np.isfinite(stop):
            raise ValueError(f"Sensitivity parameter {name!r} linspace start/stop must be finite numbers.")
        if count == 1:
            return [start]
        step = (stop - start) / float(count - 1)
        return [start + index * step for index in range(count)]
    raise ValueError(f"Unsupported sensitivity parameter mode: {mode}")


def _validate_sensitivity_values(values: list[Any], *, name: str) -> None:
    for value in values:
        if isinstance(value, (int, float, np.integer, np.floating)) and not np.isfinite(value):
            raise ValueError(f"Sensitivity parameter {name!r} values must be finite.")


def _apply_derived_parameters(row: dict[str, Any], derived_parameters: tuple[dict[str, Any], ...]) -> None:
    for derived in derived_parameters:
        name = str(derived.get("name", "")).strip()
        op = str(derived.get("op", "add") or "add").strip().lower()
        terms = [str(term).strip() for term in list(derived.get("terms", []) or []) if str(term).strip()]
        if not name:
            raise ValueError("Every sensitivity derived parameter requires a non-empty name.")
        if op != "add":
            raise ValueError("Sensitivity derived parameters v1 supports only op: add.")
        if not terms:
            raise ValueError(f"Sensitivity derived parameter {name!r} requires terms.")
        if name in row:
            raise ValueError("Sensitivity parameter and derived parameter names must be unique.")
        missing_terms = [term for term in terms if term not in row]
        if missing_terms:
            raise ValueError(
                f"Sensitivity derived parameter {name!r} references unknown term(s): "
                f"{', '.join(missing_terms)}."
            )
        try:
            derived_value = float(sum(float(row[term]) for term in terms))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Sensitivity derived parameter {name!r} terms must be numeric.") from exc
        if not np.isfinite(derived_value):
            raise ValueError(f"Sensitivity derived parameter {name!r} must be finite.")
        row[name] = derived_value


def needs_numba_chunked_summary(result: CampaignTrialResultRecord) -> bool:
    if result.status != "completed":
        return False
    product = dict(result.metrics.get("builder_product", {}) or {})
    fit = dict(product.get("sgp4_fit", {}) or {})
    return (
        fit.get("status") == "completed"
        and isinstance(fit.get("estimated_elements"), dict)
        and not isinstance(product.get("sgp4_trial_propagation"), dict)
    )


def _parallel_sensitivity_trial_results(
    cfg: ScalePlannerConfig,
    *,
    campaign: SensitivityCampaignRecord,
    trials: list[SensitivityTrialRecord],
    run_id: str,
    max_workers: int,
    propagate_sgp4: bool,
    duration_s: float | None,
    step_s: float | None,
) -> tuple[list[CampaignTrialResultRecord], str]:
    if not trials:
        return [], ""
    worker_count = max(1, min(int(max_workers), len(trials)))
    def payload_for(trial: SensitivityTrialRecord) -> dict[str, Any]:
        return {
            "config_path": str(cfg.config_path),
            "campaign": campaign,
            "trial": trial,
            "run_id": run_id,
            "backend": "parallel_cpu",
            "propagate_sgp4": bool(propagate_sgp4),
            "duration_s": duration_s,
            "step_s": step_s,
        }
    thread_env_previous = set_parallel_worker_thread_limits(default_threads="1")
    results: list[CampaignTrialResultRecord] = []
    completed_trial_ids: set[str] = set()
    parallel_error: BaseException | None = None
    executor: ProcessPoolExecutor | None = None
    planner_limit_aborted = False
    try:
        try:
            executor = ProcessPoolExecutor(max_workers=worker_count)
            trial_iterator = iter(trials)
            future_to_trial: dict[Any, SensitivityTrialRecord] = {}

            def fill_queue() -> None:
                while len(future_to_trial) < 2 * worker_count:
                    check_active_scale_planner_limits()
                    try:
                        trial = next(trial_iterator)
                    except StopIteration:
                        return
                    future_to_trial[executor.submit(_parallel_sensitivity_trial_worker, payload_for(trial))] = trial

            fill_queue()
            while future_to_trial:
                completed, _pending = wait(
                    tuple(future_to_trial),
                    timeout=0.1,
                    return_when=FIRST_COMPLETED,
                )
                if not completed:
                    # Planner limits are owner-side ceilings.  Poll while
                    # process workers are pending and abort this owned pool on
                    # a breach instead of waiting for a trial to finish.
                    check_active_scale_planner_limits()
                    continue
                for future in completed:
                    trial = future_to_trial.pop(future)
                    try:
                        result = future.result()
                    except (BrokenProcessPool, EOFError, OSError) as exc:
                        parallel_error = exc
                        abort_process_pool(executor)
                        break
                    except BaseException:
                        abort_process_pool(executor)
                        raise
                    results.append(result)
                    completed_trial_ids.add(trial.trial_id)
                    write_campaign_trial_results(cfg.store_path, results=[result])
                    check_active_scale_planner_limits()
                if parallel_error is not None:
                    break
                fill_queue()
        except ScalePlannerLimitError:
            planner_limit_aborted = True
            if executor is not None:
                abort_process_pool(executor)
            raise
        except (BrokenProcessPool, EOFError, OSError, RuntimeError) as exc:
            parallel_error = exc
            if executor is not None:
                abort_process_pool(executor)
        finally:
            if executor is not None and parallel_error is None and not planner_limit_aborted:
                executor.shutdown(wait=True, cancel_futures=False)
    finally:
        restore_env_vars(thread_env_previous)
    if parallel_error is not None:
        for trial in trials:
            if trial.trial_id in completed_trial_ids:
                continue
            check_active_scale_planner_limits()
            result = _sensitivity_trial_result(
                cfg,
                campaign=campaign,
                trial=trial,
                run_id=run_id,
                backend="serial_fallback",
                propagate_sgp4=propagate_sgp4,
                duration_s=duration_s,
                step_s=step_s,
            )
            results.append(result)
            write_campaign_trial_results(cfg.store_path, results=[result])
            check_active_scale_planner_limits()
    reason = "" if parallel_error is None else f"{type(parallel_error).__name__}: {parallel_error}"
    return results, reason


def _thread_pool_sensitivity_trial_results(
    cfg: ScalePlannerConfig,
    *,
    campaign: SensitivityCampaignRecord,
    trials: list[SensitivityTrialRecord],
    run_id: str,
    max_workers: int,
    propagate_sgp4: bool,
    duration_s: float | None,
    step_s: float | None,
) -> tuple[list[CampaignTrialResultRecord], str]:
    if not trials:
        return [], ""
    ensure_scale_thread_builder_compatible(campaign.trial_builder, workflow_label="sensitivity")
    worker_count = max(1, min(int(max_workers), len(trials)))
    worker_budget = scale_thread_worker_budget(worker_count)

    def payload_for(trial: SensitivityTrialRecord) -> dict[str, Any]:
        return {
            "config_path": str(cfg.config_path),
            "campaign": campaign,
            "trial": trial,
            "run_id": run_id,
            "backend": "thread_pool",
            "propagate_sgp4": bool(propagate_sgp4),
            "duration_s": duration_s,
            "step_s": step_s,
            "worker_budget": worker_budget,
        }

    results: list[CampaignTrialResultRecord] = []
    completed_trial_ids: set[str] = set()
    thread_error: BaseException | None = None
    executor: ThreadCampaignExecutor | None = None
    try:
        try:
            # Keep only one batch resident so owner cancellation can prevent
            # the next batch from entering the pool's internal queue.
            executor = ThreadCampaignExecutor(
                max_workers=worker_count,
                max_in_flight=worker_count,
            )
            for future, task in executor.iter_completed(
                _thread_pool_sensitivity_trial_worker,
                (payload_for(trial) for trial in trials),
            ):
                if future is None or task is None:
                    check_active_scale_planner_limits()
                    continue
                result = future.result()
                results.append(result)
                completed_trial_ids.add(str(task["trial"].trial_id))
                write_campaign_trial_results(cfg.store_path, results=[result])
                check_active_scale_planner_limits()
        except ThreadCampaignCancelled:
            if executor is not None:
                executor.cancel()
            raise
        except (ThreadCampaignBackendUnavailable, EOFError, OSError) as exc:
            thread_error = exc
            if executor is not None:
                executor.cancel()
        except BaseException:
            if executor is not None:
                executor.cancel()
            raise
    finally:
        if executor is not None:
            # Python threads and native kernels cannot be forcibly stopped.
            # Owner polling requests cooperative cancellation, then shutdown
            # waits for active workers to reach their cancellation checks.
            executor.shutdown(wait_for_running=True)
    if thread_error is not None:
        for trial in trials:
            if trial.trial_id in completed_trial_ids:
                continue
            check_active_scale_planner_limits()
            result = _sensitivity_trial_result(
                cfg,
                campaign=campaign,
                trial=trial,
                run_id=run_id,
                backend="serial_fallback",
                propagate_sgp4=propagate_sgp4,
                duration_s=duration_s,
                step_s=step_s,
            )
            results.append(result)
            write_campaign_trial_results(cfg.store_path, results=[result])
            check_active_scale_planner_limits()
    reason = "" if thread_error is None else f"{type(thread_error).__name__}: {thread_error}"
    return results, reason


def _parallel_sensitivity_trial_worker(payload: dict[str, Any]) -> CampaignTrialResultRecord:
    cfg = load_worker_scale_planner_config(str(payload["config_path"]))
    return _sensitivity_trial_result(
        cfg,
        campaign=payload["campaign"],
        trial=payload["trial"],
        run_id=str(payload["run_id"]),
        backend=str(payload["backend"]),
        propagate_sgp4=bool(payload["propagate_sgp4"]),
        duration_s=payload["duration_s"],
        step_s=payload["step_s"],
    )


def _thread_pool_sensitivity_trial_worker(payload: dict[str, Any]) -> CampaignTrialResultRecord:
    raise_if_thread_task_cancelled(payload)
    # The immutable planner config is shared through the existing bounded
    # worker cache; only the trial/result records cross the task boundary.
    cfg = load_worker_scale_planner_config(str(payload["config_path"]))
    result = _sensitivity_trial_result(
        cfg,
        campaign=payload["campaign"],
        trial=payload["trial"],
        run_id=str(payload["run_id"]),
        backend=str(payload["backend"]),
        propagate_sgp4=bool(payload["propagate_sgp4"]),
        duration_s=payload["duration_s"],
        step_s=payload["step_s"],
    )
    raise_if_thread_task_cancelled(payload)
    return result


def _sensitivity_trial_result(
    cfg: ScalePlannerConfig,
    *,
    campaign: SensitivityCampaignRecord,
    trial: SensitivityTrialRecord,
    run_id: str,
    backend: str,
    propagate_sgp4: bool = False,
    duration_s: float | None = None,
    step_s: float | None = None,
) -> CampaignTrialResultRecord:
    return build_campaign_trial_result(
        _sensitivity_execution_cfg(cfg),
        campaign=campaign,
        trial=trial,
        run_id=run_id,
        backend=backend,
        trial_builder=cfg.sensitivity_trial_builder,
        result_prefix="sensitivity_result",
        workflow_label="sensitivity",
        propagate_sgp4=propagate_sgp4,
        duration_s=duration_s,
        step_s=step_s,
    )


def _sensitivity_execution_cfg(cfg: ScalePlannerConfig) -> ScalePlannerConfig:
    return replace(
        cfg,
        monte_carlo_trial_builder=dict(cfg.sensitivity_trial_builder),
        monte_carlo_outputs=dict(cfg.sensitivity_outputs),
        monte_carlo_output_dir=cfg.sensitivity_output_dir,
    )


def _sensitivity_run_id(
    cfg: ScalePlannerConfig,
    *,
    campaign: SensitivityCampaignRecord,
    trials: list[SensitivityTrialRecord],
    backend: str,
    limit: int,
    propagate_sgp4: bool,
    duration_s: float | None,
    step_s: float | None,
    propagation_summary_backend: str,
    chunk_duration_s: float | None,
    numeric_backend: str = "rust",
) -> str:
    identity = {
        "workflow_name": cfg.name,
        "campaign_id": campaign.campaign_id,
        "backend": backend,
        "execution_contract": _execution_contract(backend),
        "limit": int(limit or 0),
        "propagate_sgp4": bool(propagate_sgp4),
        "duration_s": duration_s,
        "step_s": step_s,
        "propagation_summary_backend": propagation_summary_backend,
        "chunk_duration_s": chunk_duration_s,
        "trial_ids": [trial.trial_id for trial in trials],
    }
    normalized_numeric_backend = normalize_numeric_backend(numeric_backend)
    if normalized_numeric_backend != "python":
        identity["numeric_backend"] = normalized_numeric_backend
    digest = _digest(identity)
    return f"sensitivity_{backend}_run:{digest}"


def _execution_contract(backend: str) -> str:
    if backend == "serial":
        return "sgp4_sensitivity_serial_v1"
    if backend == "parallel_cpu":
        return "sgp4_sensitivity_parallel_cpu_v1"
    return f"sgp4_sensitivity_{backend}_v1"


def _normalize_parallel_backend(backend: str | None) -> str:
    value = str(backend or "parallel_cpu").strip().lower()
    if value == "process_pool":
        value = "parallel_cpu"
    if value not in {"parallel_cpu", "thread_pool"}:
        raise ValueError("Sensitivity parallel backend must be parallel_cpu/process_pool or thread_pool.")
    return value


def _run_artifact_stem(*, backend: str, run_id: str) -> str:
    return f"sensitivity_{backend}_{_run_id_suffix(run_id)}"


def _run_id_suffix(run_id: str) -> str:
    suffix = str(run_id).rsplit(":", 1)[-1]
    safe = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in suffix)
    return safe[:16] if safe else _digest({"run_id": run_id})[:16]


def _write_run_manifest(
    cfg: ScalePlannerConfig,
    *,
    campaign: SensitivityCampaignRecord,
    run_id: str,
    results: list[CampaignTrialResultRecord],
    limit: int,
    backend: str,
    max_workers: int,
    retry_failed: bool,
    propagate_sgp4: bool,
    duration_s: float | None,
    step_s: float | None,
    propagation_summary_backend: str,
    chunk_duration_s: float | None,
    telemetry: dict[str, Any],
    numeric_backend: str = "rust",
) -> Path:
    manifest_path = cfg.sensitivity_output_dir / f"{_run_artifact_stem(backend=backend, run_id=run_id)}_run_manifest.json"
    status_counts: dict[str, int] = {}
    for result in results:
        status_counts[result.status] = status_counts.get(result.status, 0) + 1
    manifest = {
        "schema_version": 1,
        "workflow_name": cfg.name,
        "config_path": str(cfg.config_path),
        "store_path": str(cfg.store_path),
        "run": {
            "run_id": run_id,
            "campaign_id": campaign.campaign_id,
            "backend": backend,
            "execution_contract": _execution_contract(backend),
            "max_workers": int(max_workers),
            "requested_limit": int(limit or 0),
            "retry_failed": bool(retry_failed),
            "propagate_sgp4": bool(propagate_sgp4),
            "duration_s": duration_s,
            "step_s": step_s,
            "propagation_summary_backend": propagation_summary_backend,
            "chunk_duration_s": chunk_duration_s,
            "trial_count": len(results),
            "status_counts": status_counts,
        },
        "results": [_result_payload(result) for result in results],
        "telemetry": telemetry,
        "non_claims": _execution_non_claims(propagate_sgp4=propagate_sgp4),
    }
    if normalize_numeric_backend(numeric_backend) == "rust":
        manifest["run"]["numeric_backend"] = "rust"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest_path


def _write_run_summary(
    cfg: ScalePlannerConfig,
    *,
    campaign: SensitivityCampaignRecord,
    run_id: str,
    results: list[CampaignTrialResultRecord],
    limit: int,
    backend: str,
    max_workers: int,
    retry_failed: bool,
    propagate_sgp4: bool,
    duration_s: float | None,
    step_s: float | None,
    run_status: str,
    status_counts: dict[str, int],
    manifest_path: Path,
    propagation_summary_backend: str,
    chunk_duration_s: float | None,
    telemetry: dict[str, Any],
    execution_cfg: ScalePlannerConfig,
    numeric_backend: str = "rust",
) -> tuple[Path, Path, dict[str, Any]]:
    artifact_stem = _run_artifact_stem(backend=backend, run_id=run_id)
    summary_path = cfg.sensitivity_output_dir / f"{artifact_stem}_summary.json"
    markdown_path = cfg.sensitivity_output_dir / f"{artifact_stem}_summary.md"
    metric_names = summary_metric_names(execution_cfg, propagate_sgp4=propagate_sgp4)
    metric_stats = serial_metric_stats(
        results,
        metric_names=metric_names,
        numeric_backend=numeric_backend,
    )
    quality = evaluate_quality_gates(
        execution_cfg,
        results=results,
        status_counts=status_counts,
        metric_stats=metric_stats,
    )
    summary = {
        "schema_version": 1,
        "workflow_name": cfg.name,
        "config_path": str(cfg.config_path),
        "store_path": str(cfg.store_path),
        "campaign": {
            "campaign_id": campaign.campaign_id,
            "campaign_type": campaign.campaign_type,
            "target": campaign.target,
            "uncertainty_interpretation": campaign.uncertainty_interpretation,
            "seed": campaign.seed,
            "requested_trial_count": campaign.requested_trial_count,
            "realized_trial_count": campaign.realized_trial_count,
            "sweep_mode": campaign.sweep_mode,
        },
        "run": {
            "run_id": run_id,
            "backend": backend,
            "execution_contract": _execution_contract(backend),
            "status": run_status,
            "max_workers": int(max_workers),
            "requested_limit": int(limit or 0),
            "retry_failed": bool(retry_failed),
            "trial_count": len(results),
            "status_counts": dict(sorted(status_counts.items())),
            "propagate_sgp4": bool(propagate_sgp4),
            "duration_s": duration_s,
            "step_s": step_s,
            "propagation_summary_backend": propagation_summary_backend,
            "chunk_duration_s": chunk_duration_s,
        },
        "metrics": metric_stats,
        "sensitivity_deltas": _sensitivity_delta_table(results),
        "quality": quality,
        "telemetry": telemetry,
        "artifacts": {
            "manifest_json": str(manifest_path),
            "summary_json": str(summary_path),
            "summary_markdown": str(markdown_path),
        },
        "non_claims": _execution_non_claims(propagate_sgp4=propagate_sgp4),
    }
    if normalize_numeric_backend(numeric_backend) == "rust":
        summary["run"]["numeric_backend"] = "rust"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path.write_text(_render_run_summary_markdown(summary), encoding="utf-8")
    return summary_path, markdown_path, summary


def _result_payload(result: CampaignTrialResultRecord) -> dict[str, Any]:
    return {
        "result_id": result.result_id,
        "trial_id": result.trial_id,
        "trial_index": result.trial_index,
        "backend": result.backend,
        "status": result.status,
        "builder_status": result.builder_status,
        "metrics": result.metrics,
        "product_refs": list(result.product_refs),
        "error": result.error,
        "warnings": list(result.warnings),
    }


def _sensitivity_delta_table(results: list[CampaignTrialResultRecord]) -> list[dict[str, Any]]:
    completed = [result for result in results if result.status == "completed"]
    if not completed:
        return []
    baseline = min(completed, key=lambda item: item.trial_index)
    metric_names = (
        "final_separation_km",
        "max_separation_km",
        "min_separation_km",
        "time_of_min_separation_s",
        "time_of_max_separation_s",
        "sgp4_fit_position_rms_m",
        "sgp4_fit_velocity_rms_mm_s",
    )
    baseline_values = {name: _coerce_float(baseline.metrics.get(name)) for name in metric_names}
    rows = []
    for result in sorted(results, key=lambda item: item.trial_index):
        parameters = result.metrics.get("parameters")
        if not isinstance(parameters, dict):
            product = result.metrics.get("builder_product")
            parameters = dict(product.get("parameters", {})) if isinstance(product, dict) else {}
        row: dict[str, Any] = {
            "trial_index": int(result.trial_index),
            "trial_id": result.trial_id,
            "status": result.status,
            "parameters": parameters,
            "baseline_trial_index": int(baseline.trial_index),
        }
        for name in metric_names:
            value = _coerce_float(result.metrics.get(name))
            row[name] = value
            baseline_value = baseline_values.get(name)
            row[f"delta_{name}"] = (
                float(value - baseline_value)
                if value is not None and baseline_value is not None
                else None
            )
        rows.append(row)
    return rows


def _render_run_summary_markdown(summary: dict[str, Any]) -> str:
    run = summary["run"]
    quality = summary["quality"]
    lines = [
        f"# {summary['workflow_name']} Sensitivity Summary",
        "",
        f"- Run status: `{run['status']}`",
        f"- Quality status: `{quality['status']}`",
        f"- Backend: `{run['backend']}`",
        f"- Trials: `{run['trial_count']}`",
        f"- Status counts: `{json.dumps(run['status_counts'], sort_keys=True)}`",
        f"- SGP4 propagation: `{run['propagate_sgp4']}`",
    ]
    if run.get("propagate_sgp4"):
        lines.extend(
            [
                f"- Propagation summary backend: `{run.get('propagation_summary_backend')}`",
                f"- Propagation duration: `{run['duration_s']}` s",
                f"- Propagation step: `{run['step_s']}` s",
            ]
        )
    lines.extend(["", "## Metric Summary", "", "| Metric | Count | Min | Mean | Median | P95 | Max | Missing |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for name, stats in summary["metrics"].items():
        lines.append(
            "| "
            + " | ".join(
                [
                    name,
                    str(stats["count"]),
                    _format_summary_value(stats["min"]),
                    _format_summary_value(stats["mean"]),
                    _format_summary_value(stats["median"]),
                    _format_summary_value(stats["p95"]),
                    _format_summary_value(stats["max"]),
                    str(stats["missing_count"]),
                ]
            )
            + " |"
        )
    deltas = list(summary.get("sensitivity_deltas", []) or [])
    if deltas:
        lines.extend(["", "## Sensitivity Deltas", "", "| Trial | Parameters | Delta Final Sep (km) | Delta Max Sep (km) |", "| ---: | --- | ---: | ---: |"])
        for row in deltas:
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row.get("trial_index")),
                        json.dumps(row.get("parameters", {}), sort_keys=True),
                        _format_summary_value(row.get("delta_final_separation_km")),
                        _format_summary_value(row.get("delta_max_separation_km")),
                    ]
                )
                + " |"
            )
    telemetry = summary.get("telemetry")
    if isinstance(telemetry, dict) and telemetry:
        propagation = telemetry.get("propagation")
        if not isinstance(propagation, dict):
            propagation = {}
        lines.extend(
            [
                "",
                "## Run Telemetry",
                "",
                f"- Wall time: `{_format_summary_value(telemetry.get('wall_time_s'))}` s",
                f"- Propagation state evaluations: `{int(propagation.get('state_evaluations', 0) or 0)}`",
                f"- Propagation evals/sec: `{_format_summary_value(propagation.get('state_evaluations_per_propagation_s'))}`",
                f"- Wall evals/sec: `{_format_summary_value(propagation.get('state_evaluations_per_wall_s'))}`",
            ]
        )
    lines.extend(["", "## Quality Gates", "", "| Gate | Status | Observed | Comparator | Threshold |", "| --- | --- | ---: | --- | ---: |"])
    for check in quality["checks"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    check["name"],
                    check["status"],
                    _format_summary_value(check["observed"]),
                    check["comparator"],
                    _format_summary_value(check["threshold"]),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Non-Claims", ""])
    lines.extend(f"- {item}" for item in summary["non_claims"])
    lines.append("")
    return "\n".join(lines)


def _execution_non_claims(*, propagate_sgp4: bool) -> list[str]:
    claims = [
        "Sensitivity execution v1 executes deterministic analyst-defined parameter sweeps.",
        "No external trial builder was imported or executed.",
        "Sensitivity deltas compare against the first completed trial as a baseline; they are not covariance probabilities.",
    ]
    if propagate_sgp4:
        claims.append("SGP4 propagation uses synthetic fitted mean elements, not observed catalog TLEs.")
    else:
        claims.append("No perturbed SGP4 trajectory propagation products were generated.")
    return claims


def _coerce_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric):
        return None
    return numeric


def _format_summary_value(value: Any) -> str:
    numeric = _coerce_float(value)
    if numeric is None:
        return ""
    return f"{numeric:.6g}"


def _trial_record(
    *,
    campaign_id: str,
    trial_index: int,
    parameters: dict[str, Any],
    target_object_ids: tuple[str, ...],
    frame: dict[str, Any],
    trial_builder: dict[str, Any],
    campaign_digest: str,
) -> SensitivityTrialRecord:
    digest = _digest(
        {
            "campaign_digest": campaign_digest,
            "trial_index": trial_index,
            "parameters": parameters,
            "target_object_ids": list(target_object_ids),
        }
    )
    builder_kind = str(trial_builder.get("kind", "external") or "external")
    return SensitivityTrialRecord(
        trial_id=f"sensitivity_trial:{digest}",
        campaign_id=campaign_id,
        trial_index=trial_index,
        parameters=parameters,
        target_object_ids=target_object_ids,
        frame=frame,
        builder_status=f"pending_{builder_kind}_builder",
        trial_product_refs=(),
        input_digest=digest,
        status="planned",
    )


def _campaign_warnings(cfg: ScalePlannerConfig, *, target_object_ids: tuple[str, ...]) -> tuple[str, ...]:
    warnings: list[str] = []
    if cfg.sensitivity_uncertainty_interpretation != "analyst_defined_sensitivity":
        warnings.append("Only analyst-defined sensitivity is supported in Phase 1F.")
    if not target_object_ids:
        warnings.append("No current target objects were available for this sensitivity campaign.")
    if not cfg.sensitivity_trial_builder:
        warnings.append("No trial_builder was provided; trials are parameter rows only.")
    return tuple(warnings)


def _write_campaign_manifest(
    cfg: ScalePlannerConfig,
    *,
    campaign: SensitivityCampaignRecord,
    trials: list[SensitivityTrialRecord],
) -> Path:
    manifest_path = cfg.sensitivity_output_dir / "sensitivity_campaign_manifest.json"
    manifest = {
        "schema_version": 1,
        "workflow_name": cfg.name,
        "config_path": str(cfg.config_path),
        "store_path": str(cfg.store_path),
        "campaign": {
            "campaign_id": campaign.campaign_id,
            "campaign_type": campaign.campaign_type,
            "target": campaign.target,
            "uncertainty_interpretation": campaign.uncertainty_interpretation,
            "seed": campaign.seed,
            "requested_trial_count": campaign.requested_trial_count,
            "realized_trial_count": campaign.realized_trial_count,
            "sweep_mode": campaign.sweep_mode,
            "parameters": list(campaign.parameters),
            "derived_parameters": list(cfg.sensitivity_derived_parameters),
            "frame": campaign.frame,
            "trial_builder": campaign.trial_builder,
            "outputs": campaign.outputs,
            "warnings": list(campaign.warnings),
        },
        "trials": [
            {
                "trial_id": trial.trial_id,
                "trial_index": trial.trial_index,
                "parameters": trial.parameters,
                "target_object_ids": list(trial.target_object_ids),
                "builder_status": trial.builder_status,
                "status": trial.status,
            }
            for trial in trials
        ],
        "non_claims": [
            "Manifest records analyst-defined sensitivity trials only.",
            "No external trial builder was imported or executed.",
            "No perturbed SGP4 propagation products were generated.",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest_path


def _digest(payload: dict[str, Any]) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
