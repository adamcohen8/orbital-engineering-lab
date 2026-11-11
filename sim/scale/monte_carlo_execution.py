from __future__ import annotations

import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
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
    ensure_scale_thread_builder_compatible,
    load_worker_scale_planner_config,
    raise_if_thread_task_cancelled,
    resolve_scale_campaign_worker_count,
    scale_thread_worker_budget,
)
from sim.scale.monte_carlo_batch_parity import (
    _attach_numba_chunked_propagation_summaries,
    _write_batch_propagation_parity_artifact,
)
from sim.scale.monte_carlo_planning import (
    _digest,
    _execution_contract,
    _trial_result,
    build_monte_carlo_campaign,
)
from sim.scale.monte_carlo_quality import _coerce_float
from sim.scale.monte_carlo_reporting import (
    _serial_non_claims,
    _write_campaign_manifest,
    _write_serial_run_manifest,
    _write_serial_run_summary,
)
from sim.scale.planner import (
    ScalePlannerConfig,
    ScalePlannerLimitError,
    check_active_scale_planner_limits,
    enforce_scale_planner_limits,
    load_scale_planner_config,
)
from sim.scale.store import (
    CampaignTrialResultRecord,
    SensitivityCampaignRecord,
    SensitivityTrialRecord,
    load_campaign_trial_results,
    load_sensitivity_trials,
    write_campaign_trial_results,
    write_sensitivity_campaign,
)
from sim.utils.parallel import abort_process_pool, restore_env_vars, set_parallel_worker_thread_limits

PROPAGATION_SUMMARY_BACKENDS = {"scalar_trial", "numba_chunked"}
DEFAULT_NUMBA_CHUNK_DURATION_S = 21600.0


@enforce_scale_planner_limits("Monte Carlo materialization")
def run_monte_carlo_campaign(config_path: str | Path) -> dict[str, Any]:
    cfg = load_scale_planner_config(config_path)
    if cfg.mode != "run":
        raise ValueError("Monte Carlo campaign materialization requires scale_catalog.mode: run.")
    if not cfg.monte_carlo_enabled:
        raise ValueError("Monte Carlo campaign materialization requires scale_catalog.monte_carlo.enabled: true.")
    campaign, trials = build_monte_carlo_campaign(cfg)
    cfg.monte_carlo_output_dir.mkdir(parents=True, exist_ok=True)
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
        "output_dir": str(cfg.monte_carlo_output_dir),
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
            "Monte Carlo v1 materializes sampled trial inputs only.",
            "No user trial builder was imported or executed.",
            "No perturbed SGP4 products or covariance-derived probability were generated.",
        ],
    }


@enforce_scale_planner_limits("serial Monte Carlo execution")
def run_monte_carlo_serial(
    config_path: str | Path,
    *,
    limit: int = 0,
    propagate_sgp4: bool = False,
    duration_s: float | None = None,
    step_s: float | None = None,
    retry_failed: bool = False,
    sgp4_batch_backend: str = "",
    propagation_summary_backend: str = "scalar_trial",
    chunk_duration_s: float | None = None,
    numeric_backend: str = "rust",
) -> dict[str, Any]:
    return _run_monte_carlo_execution(
        config_path,
        backend="serial",
        max_workers=1,
        limit=limit,
        propagate_sgp4=propagate_sgp4,
        duration_s=duration_s,
        step_s=step_s,
        retry_failed=retry_failed,
        sgp4_batch_backend=sgp4_batch_backend,
        propagation_summary_backend=propagation_summary_backend,
        chunk_duration_s=chunk_duration_s,
        numeric_backend=numeric_backend,
    )


@enforce_scale_planner_limits("parallel Monte Carlo execution")
def run_monte_carlo_parallel_cpu(
    config_path: str | Path,
    *,
    limit: int = 0,
    workers: int = 0,
    propagate_sgp4: bool = False,
    duration_s: float | None = None,
    step_s: float | None = None,
    retry_failed: bool = False,
    sgp4_batch_backend: str = "",
    propagation_summary_backend: str = "scalar_trial",
    chunk_duration_s: float | None = None,
    numeric_backend: str = "rust",
    backend: str = "parallel_cpu",
) -> dict[str, Any]:
    execution_backend = _normalize_parallel_backend(backend)
    worker_count = resolve_scale_campaign_worker_count(workers)
    return _run_monte_carlo_execution(
        config_path,
        backend=execution_backend,
        max_workers=worker_count,
        limit=limit,
        propagate_sgp4=propagate_sgp4,
        duration_s=duration_s,
        step_s=step_s,
        retry_failed=retry_failed,
        sgp4_batch_backend=sgp4_batch_backend,
        propagation_summary_backend=propagation_summary_backend,
        chunk_duration_s=chunk_duration_s,
        numeric_backend=numeric_backend,
    )


@enforce_scale_planner_limits("threaded Monte Carlo execution")
def run_monte_carlo_threaded(
    config_path: str | Path,
    *,
    limit: int = 0,
    workers: int = 0,
    propagate_sgp4: bool = False,
    duration_s: float | None = None,
    step_s: float | None = None,
    retry_failed: bool = False,
    sgp4_batch_backend: str = "",
    propagation_summary_backend: str = "scalar_trial",
    chunk_duration_s: float | None = None,
    numeric_backend: str = "rust",
) -> dict[str, Any]:
    """Run independent Scale Monte Carlo trials on an explicit thread pool."""

    worker_count = resolve_scale_campaign_worker_count(workers)
    return _run_monte_carlo_execution(
        config_path,
        backend="thread_pool",
        max_workers=worker_count,
        limit=limit,
        propagate_sgp4=propagate_sgp4,
        duration_s=duration_s,
        step_s=step_s,
        retry_failed=retry_failed,
        sgp4_batch_backend=sgp4_batch_backend,
        propagation_summary_backend=propagation_summary_backend,
        chunk_duration_s=chunk_duration_s,
        numeric_backend=numeric_backend,
    )


def _run_monte_carlo_execution(
    config_path: str | Path,
    *,
    backend: str,
    max_workers: int,
    limit: int = 0,
    propagate_sgp4: bool = False,
    duration_s: float | None = None,
    step_s: float | None = None,
    retry_failed: bool = False,
    sgp4_batch_backend: str = "",
    propagation_summary_backend: str = "scalar_trial",
    chunk_duration_s: float | None = None,
    numeric_backend: str = "rust",
) -> dict[str, Any]:
    wall_start = time.perf_counter()
    phase_durations: dict[str, float] = {}
    phase_start = time.perf_counter()
    cfg = load_scale_planner_config(config_path)
    if cfg.mode != "run":
        raise ValueError("Monte Carlo execution requires scale_catalog.mode: run.")
    if not cfg.monte_carlo_enabled:
        raise ValueError("Monte Carlo execution requires scale_catalog.monte_carlo.enabled: true.")
    numeric_backend = normalize_numeric_backend(numeric_backend)
    if numeric_backend not in {"python", "rust"}:
        raise ValueError("numeric_backend must be python or rust.")
    batch_backend = str(sgp4_batch_backend or "").strip().lower()
    if batch_backend and not propagate_sgp4:
        raise ValueError("SGP4 batch propagation parity requires propagate_sgp4=True.")
    if batch_backend and batch_backend not in {"scalar_reference", "numba"}:
        raise ValueError("SGP4 batch propagation parity currently supports backend: scalar_reference or numba.")
    summary_backend = _normalize_propagation_summary_backend(
        propagate_sgp4=propagate_sgp4,
        propagation_summary_backend=propagation_summary_backend,
    )
    resolved_chunk_duration_s = (
        None if summary_backend != "numba_chunked" else _resolved_chunk_duration_s(chunk_duration_s)
    )

    campaign, planned_trials = build_monte_carlo_campaign(cfg)
    write_sensitivity_campaign(cfg.store_path, campaign=campaign, trials=planned_trials)
    trials = load_sensitivity_trials(cfg.store_path, campaign_id=campaign.campaign_id, limit=max(int(limit), 0))
    if not trials:
        raise ValueError(f"No materialized Monte Carlo trials found for campaign {campaign.campaign_id}.")

    run_id = _monte_carlo_run_id(
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
        sgp4_batch_backend=batch_backend,
        numeric_backend=numeric_backend,
    )
    existing_results = load_campaign_trial_results(cfg.store_path, campaign_id=campaign.campaign_id, run_id=run_id)
    existing_by_trial = _latest_results_by_trial(existing_results, trials=trials)
    pending_trials = _pending_trials(trials, existing_by_trial=existing_by_trial, retry_failed=retry_failed)
    trial_builder_propagate_sgp4 = bool(propagate_sgp4 and summary_backend == "scalar_trial")
    phase_durations["setup_s"] = time.perf_counter() - phase_start
    phase_start = time.perf_counter()
    if backend == "serial":
        new_results = []
        parallel_fallback_reason = ""
        for trial in pending_trials:
            check_active_scale_planner_limits()
            result = _trial_result(
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
        new_results, parallel_fallback_reason = _parallel_cpu_trial_results(
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
        new_results, parallel_fallback_reason = _thread_pool_trial_results(
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
        raise ValueError(f"Unsupported Monte Carlo execution backend: {backend}")
    phase_durations["trial_execution_s"] = time.perf_counter() - phase_start
    propagation_summary_telemetry: dict[str, Any] = {}
    results_to_write = list(new_results)
    if propagate_sgp4 and summary_backend == "numba_chunked":
        fresh_trial_ids = {result.trial_id for result in new_results}
        resumable_results = [
            result
            for trial_id, result in existing_by_trial.items()
            if trial_id not in fresh_trial_ids and _needs_numba_chunked_summary(result)
        ]
        summary_candidates = [*new_results, *resumable_results]
        if summary_candidates:
            results_to_write, propagation_summary_telemetry = _attach_numba_chunked_propagation_summaries(
                cfg,
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
        results_by_trial = _latest_results_by_trial(latest_results, trials=trials)
    else:
        results_by_trial = existing_by_trial
    results = [results_by_trial[trial.trial_id] for trial in trials if trial.trial_id in results_by_trial]
    if not results:
        raise ValueError(f"No Monte Carlo trial results were available for run {run_id}.")

    cfg.monte_carlo_output_dir.mkdir(parents=True, exist_ok=True)
    status_counts: dict[str, int] = {}
    for result in results:
        status_counts[result.status] = status_counts.get(result.status, 0) + 1
    run_status = "error" if status_counts.get("error") else "blocked" if status_counts.get("blocked") else "completed"
    phase_start = time.perf_counter()
    batch_parity_path, batch_parity = _write_batch_propagation_parity_artifact(
        cfg,
        run_id=run_id,
        results=results,
        batch_backend=batch_backend,
        execution_backend=backend,
    )
    phase_durations["batch_parity_write_s"] = time.perf_counter() - phase_start
    run_telemetry = _monte_carlo_run_telemetry(
        results=results,
        fresh_results=new_results,
        propagate_sgp4=propagate_sgp4,
        propagation_summary_backend=summary_backend,
        phase_durations=phase_durations,
        propagation_summary_telemetry=propagation_summary_telemetry,
        wall_time_s=time.perf_counter() - wall_start,
    )
    fresh_backends = {result.backend for result in new_results}
    reported_backend = backend
    if backend in {"parallel_cpu", "thread_pool"} and parallel_fallback_reason:
        reported_backend = (
            f"{backend}_with_serial_fallback"
            if backend in fresh_backends
            else "serial_fallback"
        )
    if parallel_fallback_reason:
        run_telemetry["parallel_fallback_reason"] = parallel_fallback_reason
        run_telemetry["requested_backend"] = backend
        run_telemetry["actual_backend"] = reported_backend
    manifest_path = _write_serial_run_manifest(
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
        sgp4_batch_backend=batch_backend,
        telemetry=run_telemetry,
        numeric_backend=numeric_backend,
    )
    summary_path, summary_markdown_path, serial_summary = _write_serial_run_summary(
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
        batch_parity_path=batch_parity_path,
        batch_parity=batch_parity,
        propagation_summary_backend=summary_backend,
        chunk_duration_s=resolved_chunk_duration_s,
        telemetry=run_telemetry,
        numeric_backend=numeric_backend,
    )
    payload = {
        "schema_version": 1,
        "status": run_status,
        "quality_status": serial_summary["quality"]["status"],
        "workflow": {
            "name": cfg.name,
            "config_path": str(cfg.config_path),
            "mode": cfg.mode,
        },
        "store_path": str(cfg.store_path),
        "output_dir": str(cfg.monte_carlo_output_dir),
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
        "sgp4_batch_backend": batch_backend,
        "results_attempted": write_summary["results_seen"],
        "results_recorded": write_summary["results_inserted"],
        "batch_parity_path": str(batch_parity_path) if batch_parity_path is not None else "",
        "batch_parity_status": str((batch_parity or {}).get("status", "")),
        "status_counts": status_counts,
        "telemetry": serial_summary["telemetry"],
        "non_claims": _serial_non_claims(propagate_sgp4=propagate_sgp4),
    }
    payload["numeric_backend"] = numeric_backend
    return payload


def _monte_carlo_run_id(
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
    sgp4_batch_backend: str,
    numeric_backend: str = "rust",
) -> str:
    effective_duration_s = cfg.duration_s if duration_s is None else float(duration_s)
    effective_step_s = cfg.step_s if step_s is None else float(step_s)
    run_digest = _digest(
        {
            "workflow_name": cfg.name,
            "campaign_id": campaign.campaign_id,
            "start_jd_utc": cfg.start_jd_utc,
            "backend": backend,
            "execution_contract": _execution_contract(backend),
            "limit": int(limit or 0),
            "propagate_sgp4": bool(propagate_sgp4),
            "duration_s": effective_duration_s,
            "step_s": effective_step_s,
            "propagation_summary_backend": propagation_summary_backend,
            "chunk_duration_s": chunk_duration_s,
            "sgp4_batch_backend": sgp4_batch_backend,
            "numeric_backend": str(numeric_backend).strip().lower(),
            "trial_ids": [trial.trial_id for trial in trials],
        }
    )
    return f"monte_carlo_{backend}_run:{run_digest}"


def _normalize_propagation_summary_backend(
    *,
    propagate_sgp4: bool,
    propagation_summary_backend: str,
) -> str:
    backend = str(propagation_summary_backend or "scalar_trial").strip().lower()
    if not propagate_sgp4:
        if backend not in {"", "scalar_trial"}:
            raise ValueError("Propagation summary backends require propagate_sgp4=True.")
        return "scalar_trial"
    if backend not in PROPAGATION_SUMMARY_BACKENDS:
        supported = ", ".join(sorted(PROPAGATION_SUMMARY_BACKENDS))
        raise ValueError(f"Unsupported propagation summary backend {backend!r}; supported backends: {supported}.")
    return backend


def _normalize_parallel_backend(backend: str | None) -> str:
    value = str(backend or "parallel_cpu").strip().lower()
    if value == "process_pool":
        value = "parallel_cpu"
    if value not in {"parallel_cpu", "thread_pool"}:
        raise ValueError("Monte Carlo parallel backend must be parallel_cpu/process_pool or thread_pool.")
    return value


def _resolved_chunk_duration_s(chunk_duration_s: float | None) -> float:
    value = DEFAULT_NUMBA_CHUNK_DURATION_S if chunk_duration_s is None else float(chunk_duration_s)
    if value <= 0.0 or not np.isfinite(value):
        raise ValueError("Numba chunked propagation requires a positive finite chunk_duration_s.")
    return float(value)


def _monte_carlo_run_telemetry(
    *,
    results: list[CampaignTrialResultRecord],
    fresh_results: list[CampaignTrialResultRecord],
    propagate_sgp4: bool,
    propagation_summary_backend: str,
    phase_durations: dict[str, float],
    propagation_summary_telemetry: dict[str, Any],
    wall_time_s: float,
) -> dict[str, Any]:
    propagation = _propagation_summary_telemetry_from_results(results)
    fresh_propagation = _propagation_summary_telemetry_from_results(fresh_results)
    if propagation_summary_telemetry:
        for key, value in propagation_summary_telemetry.items():
            if key not in {"state_evaluations", "trial_count", "sample_count_min", "sample_count_max", "status"}:
                propagation[key] = value
        fresh_propagation.update(propagation_summary_telemetry)
    fresh_state_evaluations = int(fresh_propagation.get("state_evaluations", 0) or 0)
    reused_trial_count = max(len(results) - len(fresh_results), 0)
    propagation["fresh_trial_count"] = int(len(fresh_results))
    propagation["reused_trial_count"] = int(reused_trial_count)
    propagation["fresh_state_evaluations"] = fresh_state_evaluations
    propagation_seconds = _coerce_float(propagation_summary_telemetry.get("wall_time_s"))
    if propagation_seconds is None and propagation_summary_backend == "scalar_trial":
        propagation_seconds = _coerce_float(phase_durations.get("trial_execution_s"))
    if fresh_state_evaluations and propagation_seconds and propagation_seconds > 0.0:
        propagation["state_evaluations_per_propagation_s"] = float(fresh_state_evaluations / propagation_seconds)
        propagation["throughput_scope"] = "fresh_results"
    if fresh_state_evaluations and wall_time_s > 0.0:
        propagation["state_evaluations_per_wall_s"] = float(fresh_state_evaluations / wall_time_s)
        propagation["throughput_scope"] = "fresh_results"
    execution_mode = "fresh"
    if reused_trial_count and fresh_results:
        execution_mode = "partial_resume"
    elif reused_trial_count:
        execution_mode = "reused"
    return {
        "schema_version": 1,
        "wall_time_s": float(wall_time_s),
        "phase_durations_s": {name: float(value) for name, value in sorted(phase_durations.items())},
        "trial_count": len(results),
        "fresh_trial_count": int(len(fresh_results)),
        "reused_trial_count": int(reused_trial_count),
        "execution_mode": execution_mode,
        "propagate_sgp4": bool(propagate_sgp4),
        "propagation_summary_backend": propagation_summary_backend,
        "propagation": propagation,
    }


def _propagation_summary_telemetry_from_results(results: list[CampaignTrialResultRecord]) -> dict[str, Any]:
    summaries = []
    for result in results:
        product = result.metrics.get("builder_product")
        if not isinstance(product, dict):
            continue
        summary = product.get("sgp4_trial_propagation")
        if isinstance(summary, dict):
            summaries.append(summary)
    if not summaries:
        return {
            "status": "not_run",
            "trial_count": 0,
            "state_evaluations": 0,
        }
    sample_counts = [int(summary.get("sample_count", 0) or 0) for summary in summaries]
    chunk_counts = [int(summary.get("chunk_count", 0) or 0) for summary in summaries if summary.get("chunk_count")]
    state_evaluations = int(sum(sample_count * 2 for sample_count in sample_counts))
    telemetry: dict[str, Any] = {
        "status": "completed",
        "trial_count": int(len(summaries)),
        "sample_count_min": int(min(sample_counts)),
        "sample_count_max": int(max(sample_counts)),
        "state_evaluations": state_evaluations,
        "model": "sgp4",
    }
    if chunk_counts:
        telemetry["chunk_count_min"] = int(min(chunk_counts))
        telemetry["chunk_count_max"] = int(max(chunk_counts))
    return telemetry


def _latest_results_by_trial(
    results: list[CampaignTrialResultRecord],
    *,
    trials: list[SensitivityTrialRecord],
) -> dict[str, CampaignTrialResultRecord]:
    selected_trial_ids = {trial.trial_id for trial in trials}
    out: dict[str, CampaignTrialResultRecord] = {}
    for result in results:
        if result.trial_id in selected_trial_ids:
            out[result.trial_id] = result
    return out


def _pending_trials(
    trials: list[SensitivityTrialRecord],
    *,
    existing_by_trial: dict[str, CampaignTrialResultRecord],
    retry_failed: bool,
) -> list[SensitivityTrialRecord]:
    retryable = {"blocked", "error"} if retry_failed else set()
    pending: list[SensitivityTrialRecord] = []
    for trial in trials:
        existing = existing_by_trial.get(trial.trial_id)
        if existing is None or existing.status in retryable:
            pending.append(trial)
    return pending


def _needs_numba_chunked_summary(result: CampaignTrialResultRecord) -> bool:
    if result.status != "completed":
        return False
    product = dict(result.metrics.get("builder_product", {}) or {})
    fit = dict(product.get("sgp4_fit", {}) or {})
    return (
        fit.get("status") == "completed"
        and isinstance(fit.get("estimated_elements"), dict)
        and not isinstance(product.get("sgp4_trial_propagation"), dict)
    )


def _parallel_cpu_trial_results(
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
                    future_to_trial[executor.submit(_parallel_cpu_trial_worker, payload_for(trial))] = trial

            fill_queue()
            while future_to_trial:
                completed, _pending = wait(
                    tuple(future_to_trial),
                    timeout=0.1,
                    return_when=FIRST_COMPLETED,
                )
                if not completed:
                    # Planner limits are owner-side ceilings.  A process worker
                    # cannot safely share the owner's ContextVar, so poll while
                    # it is pending and terminate this owned pool on breach.
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
            result = _trial_result(
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


def _thread_pool_trial_results(
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
    ensure_scale_thread_builder_compatible(campaign.trial_builder, workflow_label="Monte Carlo")
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
            for future, task in executor.iter_completed(_thread_pool_trial_worker, (payload_for(trial) for trial in trials)):
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
            result = _trial_result(
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


def _parallel_cpu_trial_worker(payload: dict[str, Any]) -> CampaignTrialResultRecord:
    cfg = load_worker_scale_planner_config(str(payload["config_path"]))
    return _trial_result(
        cfg,
        campaign=payload["campaign"],
        trial=payload["trial"],
        run_id=str(payload["run_id"]),
        backend=str(payload["backend"]),
        propagate_sgp4=bool(payload["propagate_sgp4"]),
        duration_s=payload["duration_s"],
        step_s=payload["step_s"],
    )


def _thread_pool_trial_worker(payload: dict[str, Any]) -> CampaignTrialResultRecord:
    raise_if_thread_task_cancelled(payload)
    # The immutable planner config is shared through the existing bounded
    # worker cache; only the trial/result records cross the task boundary.
    cfg = load_worker_scale_planner_config(str(payload["config_path"]))
    result = _trial_result(
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
