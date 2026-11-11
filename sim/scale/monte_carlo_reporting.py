from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from sim.numeric_backend import normalize_numeric_backend
from sim.scale.monte_carlo_planning import _execution_contract, _run_artifact_stem
from sim.scale.monte_carlo_quality import _coerce_float, _evaluate_quality_gates
from sim.scale.planner import ScalePlannerConfig
from sim.scale.store import (
    CampaignTrialResultRecord,
    SensitivityCampaignRecord,
    SensitivityTrialRecord,
)


def _write_campaign_manifest(
    cfg: ScalePlannerConfig,
    *,
    campaign: SensitivityCampaignRecord,
    trials: list[SensitivityTrialRecord],
) -> Path:
    manifest_path = cfg.monte_carlo_output_dir / "monte_carlo_campaign_manifest.json"
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
            "sampling_mode": campaign.sweep_mode,
            "parameters": list(campaign.parameters),
            "derived_parameters": list(cfg.monte_carlo_derived_parameters),
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
            "Manifest records sampled Monte Carlo trial inputs only.",
            "No external trial builder was imported or executed.",
            "No perturbed SGP4 propagation products were generated.",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest_path


def _write_serial_run_manifest(
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
    sgp4_batch_backend: str,
    telemetry: dict[str, Any],
    numeric_backend: str = "rust",
) -> Path:
    manifest_path = cfg.monte_carlo_output_dir / f"{_run_artifact_stem(backend=backend, run_id=run_id)}_run_manifest.json"
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
            "sgp4_batch_backend": sgp4_batch_backend,
            "trial_count": len(results),
            "status_counts": status_counts,
        },
        "results": [
            {
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
            for result in results
        ],
        "telemetry": telemetry,
        "non_claims": _serial_non_claims(propagate_sgp4=propagate_sgp4),
    }
    manifest["run"]["numeric_backend"] = str(numeric_backend).strip().lower()
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest_path
def _write_serial_run_summary(
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
    batch_parity_path: Path | None = None,
    batch_parity: dict[str, Any] | None = None,
    propagation_summary_backend: str = "scalar_trial",
    chunk_duration_s: float | None = None,
    telemetry: dict[str, Any] | None = None,
    numeric_backend: str = "rust",
) -> tuple[Path, Path, dict[str, Any]]:
    artifact_stem = _run_artifact_stem(backend=backend, run_id=run_id)
    summary_path = cfg.monte_carlo_output_dir / f"{artifact_stem}_summary.json"
    markdown_path = cfg.monte_carlo_output_dir / f"{artifact_stem}_summary.md"
    metric_names = _summary_metric_names(cfg, propagate_sgp4=propagate_sgp4)
    metric_stats = _serial_metric_stats(
        results,
        metric_names=metric_names,
        numeric_backend=numeric_backend,
    )
    quality = _evaluate_quality_gates(
        cfg,
        results=results,
        status_counts=status_counts,
        metric_stats=metric_stats,
        batch_parity=batch_parity,
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
        "telemetry": telemetry or {},
        "quality": quality,
        "artifacts": {
            "manifest_json": str(manifest_path),
            "summary_json": str(summary_path),
            "summary_markdown": str(markdown_path),
        },
        "non_claims": _serial_non_claims(propagate_sgp4=propagate_sgp4),
    }
    summary["run"]["numeric_backend"] = str(numeric_backend).strip().lower()
    if batch_parity_path is not None and batch_parity is not None:
        summary["artifacts"]["batch_propagation_parity_json"] = str(batch_parity_path)
        summary["batch_propagation_parity"] = {
            "status": batch_parity.get("status"),
            "backend": batch_parity.get("backend"),
            "trial_count": batch_parity.get("trial_count"),
            "sample_count": batch_parity.get("sample_count"),
            "max_absolute_differences": batch_parity.get("max_absolute_differences", {}),
        }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path.write_text(_render_serial_summary_markdown(summary), encoding="utf-8")
    return summary_path, markdown_path, summary


def _summary_metric_names(cfg: ScalePlannerConfig, *, propagate_sgp4: bool) -> tuple[str, ...]:
    configured = [
        str(item).strip()
        for item in list(cfg.monte_carlo_outputs.get("summary_metrics", []) or [])
        if str(item).strip()
    ]
    defaults = [
        "deployment_speed_m_s",
        "effective_az_deg",
        "effective_el_deg",
        "sgp4_fit_position_rms_m",
        "sgp4_fit_velocity_rms_mm_s",
    ]
    if propagate_sgp4:
        defaults.extend(
            [
                "final_separation_km",
                "max_separation_km",
                "min_separation_km",
                "time_of_min_separation_s",
                "time_of_max_separation_s",
                "propagation_sample_count",
                "propagation_chunk_count",
            ]
        )
    return tuple(dict.fromkeys([*configured, *defaults]))


def _serial_metric_stats(
    results: list[CampaignTrialResultRecord],
    *,
    metric_names: tuple[str, ...],
    numeric_backend: str = "rust",
) -> dict[str, dict[str, Any]]:
    backend = normalize_numeric_backend(numeric_backend)
    if backend not in {"python", "rust"}:
        raise ValueError("numeric_backend must be python or rust.")
    out: dict[str, dict[str, Any]] = {}
    for name in metric_names:
        values = [_extract_numeric_metric(result.metrics, name) for result in results]
        numeric_values = [value for value in values if value is not None and np.isfinite(value)]
        if backend == "rust":
            from sim.rust_scale_backend import aggregate_metric

            out[name] = aggregate_metric(numeric_values, attempted_count=len(results))
        else:
            out[name] = _numeric_stats(numeric_values, attempted_count=len(results))
    return out


def _extract_numeric_metric(metrics: dict[str, Any], name: str) -> float | None:
    if name in metrics:
        return _coerce_float(metrics.get(name))
    parameters = metrics.get("parameters")
    if isinstance(parameters, dict) and name in parameters:
        return _coerce_float(parameters.get(name))
    builder_product = metrics.get("builder_product")
    if isinstance(builder_product, dict):
        sampled_parameters = builder_product.get("sampled_parameters")
        if isinstance(sampled_parameters, dict) and name in sampled_parameters:
            return _coerce_float(sampled_parameters.get(name))
    return None


def _numeric_stats(values: list[float], *, attempted_count: int) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "missing_count": attempted_count,
            "min": None,
            "p05": None,
            "mean": None,
            "median": None,
            "p95": None,
            "max": None,
            "std": None,
        }
    array = np.asarray(values, dtype=float)
    return {
        "count": int(array.size),
        "missing_count": int(max(attempted_count - array.size, 0)),
        "min": float(np.min(array)),
        "p05": float(np.percentile(array, 5)),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
        "std": float(np.std(array, ddof=1)) if array.size > 1 else 0.0,
    }
def _render_serial_summary_markdown(summary: dict[str, Any]) -> str:
    run = summary["run"]
    quality = summary["quality"]
    backend_labels = {"serial": "Serial", "parallel_cpu": "Parallel CPU"}
    backend_label = backend_labels.get(str(run["backend"]), str(run["backend"]).replace("_", " ").title())
    lines = [
        f"# {summary['workflow_name']} Monte Carlo {backend_label} Summary",
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
        if run.get("chunk_duration_s") is not None:
            lines.append(f"- Propagation chunk duration: `{run['chunk_duration_s']}` s")
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
    parity = summary.get("batch_propagation_parity")
    if isinstance(parity, dict):
        lines.extend(
            [
                "",
                "## Batch Propagation Parity",
                "",
                f"- Status: `{parity.get('status')}`",
                f"- Backend: `{parity.get('backend')}`",
                f"- Trials: `{parity.get('trial_count')}`",
                f"- Samples: `{parity.get('sample_count')}`",
                "",
                "| Difference | Max Abs |",
                "| --- | ---: |",
            ]
        )
        for name, value in dict(parity.get("max_absolute_differences", {}) or {}).items():
            lines.append(f"| {name} | {_format_summary_value(value)} |")
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
        if propagation.get("estimated_peak_position_velocity_mib") is not None:
            lines.append(
                f"- Estimated peak position/velocity arrays: "
                f"`{_format_summary_value(propagation.get('estimated_peak_position_velocity_mib'))}` MiB"
            )
        phase_durations = telemetry.get("phase_durations_s")
        if isinstance(phase_durations, dict) and phase_durations:
            lines.extend(["", "| Phase | Seconds |", "| --- | ---: |"])
            for name, value in phase_durations.items():
                lines.append(f"| {name} | {_format_summary_value(value)} |")
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


def _format_summary_value(value: Any) -> str:
    numeric = _coerce_float(value)
    if numeric is None:
        return ""
    return f"{numeric:.6g}"


def _serial_non_claims(*, propagate_sgp4: bool) -> list[str]:
    claims = [
        "Monte Carlo execution v1 executes trial builders and records per-trial products/results.",
        "No external trial builder was imported or executed.",
    ]
    if propagate_sgp4:
        claims.append("SGP4 propagation uses synthetic fitted mean elements, not observed catalog TLEs.")
    else:
        claims.append("No perturbed SGP4 trajectory propagation products were generated.")
    return claims
