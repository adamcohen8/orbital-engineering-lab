from __future__ import annotations

import hashlib
import json
import os
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any

from sim.execution.thread_campaign import ThreadCampaignBackendUnavailable, ThreadCampaignCancelled
from sim.execution.worker_budget import (
    CAMPAIGN_WORKER_COUNT_ENV,
    TOTAL_PROCESS_BUDGET_ENV,
    current_worker_budget,
    worker_budget_value,
)
from sim.scale.planner import ScalePlannerConfig, load_scale_planner_config
from sim.scale.store import CampaignTrialResultRecord, SensitivityCampaignRecord, SensitivityTrialRecord
from sim.scale.trial_builders import build_deployment_ric_state_product


def digest_json(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _scale_config_digest(config_path: str) -> str:
    digest = hashlib.sha256()
    with Path(config_path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=8)
def _load_worker_scale_planner_config_cached(
    config_path: str,
    config_digest: str,
) -> ScalePlannerConfig:
    """Load a content-identified config while guarding the validated read."""

    before_digest = _scale_config_digest(config_path)
    if before_digest != config_digest:
        raise RuntimeError(f"Scale planner config changed before validation: {config_path}")
    config = load_scale_planner_config(config_path)
    after_digest = _scale_config_digest(config_path)
    if after_digest != config_digest:
        raise RuntimeError(f"Scale planner config changed during validation: {config_path}")
    return config


def load_worker_scale_planner_config(config_path: str) -> ScalePlannerConfig:
    """Load a content-identified campaign config and return a task-local copy."""

    source = str(Path(config_path).expanduser().resolve())
    expected_digest = _scale_config_digest(source)
    config = _load_worker_scale_planner_config_cached(source, expected_digest)
    if _scale_config_digest(source) != expected_digest:
        raise RuntimeError(f"Scale planner config changed during worker setup: {source}")
    return deepcopy(config)


def resolve_scale_campaign_worker_count(requested: int, trial_count: int | None = None) -> int:
    """Resolve bounded worker slots without changing the process default."""

    requested_count = int(requested or 0)
    if requested_count < 0:
        raise ValueError("workers must be zero (auto) or a positive integer.")
    if requested_count == 0:
        requested_count = max(1, int(os.cpu_count() or 1) - 1)
    worker_count = max(1, requested_count)
    if trial_count is not None:
        worker_count = min(worker_count, max(int(trial_count), 1))
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


def scale_thread_worker_budget(worker_count: int) -> dict[str, int]:
    """Return plain hierarchical budget data for a Scale trial thread."""

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


def ensure_scale_thread_builder_compatible(
    trial_builder: dict[str, Any],
    *,
    workflow_label: str,
) -> None:
    """Reject unregistered Scale trial builders before a thread worker starts."""

    builder_kind = str((trial_builder or {}).get("kind", "external") or "external").strip().lower()
    if builder_kind not in {"deployment_ric_state", "external"}:
        raise ThreadCampaignBackendUnavailable(
            f"Thread {workflow_label} requires a built-in trial builder; unsupported kind: {builder_kind!r}."
        )


def raise_if_thread_task_cancelled(task: dict[str, Any]) -> None:
    """Honor owner cancellation before a Scale thread starts physics work."""

    event = task.get("cancel_event")
    if event is not None and bool(event.is_set()):
        raise ThreadCampaignCancelled("Scale campaign thread execution was cancelled.")


def build_campaign_trial_result(
    cfg: ScalePlannerConfig,
    *,
    campaign: SensitivityCampaignRecord,
    trial: SensitivityTrialRecord,
    run_id: str,
    backend: str,
    trial_builder: dict[str, Any],
    result_prefix: str,
    workflow_label: str,
    propagate_sgp4: bool = False,
    duration_s: float | None = None,
    step_s: float | None = None,
) -> CampaignTrialResultRecord:
    builder_kind = str(trial_builder.get("kind", "external") or "external").strip().lower()
    if builder_kind == "external":
        status = "blocked"
        builder_status = "external_builder_required"
        metrics: dict[str, Any] = {}
        product_refs: tuple[str, ...] = ()
        warnings = (
            "Runner stopped before physics execution because no trusted trial builder is available.",
        )
        error = (
            f"External {workflow_label} trial builders are recorded but not executed by the runner yet. "
            "Provide a trusted in-process builder contract before generating perturbed SGP4 products."
        )
    elif builder_kind == "deployment_ric_state":
        try:
            product = build_deployment_ric_state_product(
                cfg,
                trial=trial,
                propagate_sgp4=propagate_sgp4,
                propagation_duration_s=duration_s,
                propagation_step_s=step_s,
            )
            status = product.status
            builder_status = "completed" if product.status == "completed" else product.error
            metrics = product.metrics
            product_refs = product.product_refs
            warnings = product.warnings
            error = product.error
        except Exception as exc:
            status = "error"
            builder_status = "builder_error"
            metrics = {}
            product_refs = ()
            warnings = ("deployment_ric_state builder failed before producing a trial product.",)
            error = str(exc)
    else:
        status = "blocked"
        builder_status = f"unsupported_{builder_kind}_builder"
        metrics = {}
        product_refs = ()
        warnings = (
            "Runner stopped before physics execution because the configured builder kind is unsupported.",
        )
        error = f"Unsupported {workflow_label} trial_builder.kind for execution: {builder_kind}"
    digest = digest_json(
        {
            "run_id": run_id,
            "campaign_id": campaign.campaign_id,
            "trial_id": trial.trial_id,
            "trial_index": trial.trial_index,
            "backend": backend,
            "status": status,
            "builder_status": builder_status,
            "parameters": trial.parameters,
            "target_object_ids": list(trial.target_object_ids),
            "product_refs": list(product_refs),
            "propagate_sgp4": bool(propagate_sgp4),
            "duration_s": duration_s,
            "step_s": step_s,
            "result_prefix": result_prefix,
        }
    )
    return CampaignTrialResultRecord(
        result_id=f"{result_prefix}:{digest}",
        run_id=run_id,
        campaign_id=campaign.campaign_id,
        trial_id=trial.trial_id,
        trial_index=trial.trial_index,
        backend=backend,
        status=status,
        builder_status=builder_status,
        metrics=metrics,
        product_refs=product_refs,
        input_digest=digest,
        error=error,
        warnings=warnings,
    )
