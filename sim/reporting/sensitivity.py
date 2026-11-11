from __future__ import annotations

import csv
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from sim.config import SimulationScenarioConfig
from sim.execution.metrics import (
    extract_study_metric,
    normalize_study_metric_specs,
)
from sim.plotting.style import artifact_metadata, save_oel_figure
from sim.reporting.output_index import write_output_index
from sim.review import write_workflow_review
from sim.utils.io import write_json


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _deep_get(root: dict[str, Any], path: str, default: Any = None) -> Any:
    parts = path.split(".")
    cur: Any = root
    for tok in parts:
        if "[" in tok and tok.endswith("]"):
            key, idx_txt = tok[:-1].split("[", 1)
            idx = int(idx_txt)
            if key:
                if not isinstance(cur, dict) or key not in cur:
                    return default
                cur = cur[key]
            if not isinstance(cur, list) or idx < 0 or idx >= len(cur):
                return default
            cur = cur[idx]
            continue
        if not isinstance(cur, dict) or tok not in cur:
            return default
        cur = cur[tok]
    return cur


def analysis_metric_specs(cfg: SimulationScenarioConfig) -> list[dict[str, str]]:
    raw_metrics = list(cfg.analysis.metrics or [])
    if raw_metrics:
        return normalize_study_metric_specs(raw_metrics)
    return normalize_study_metric_specs([
        "summary.duration_s",
        "summary.terminated_early",
        "summary.termination_reason",
        "derived.closest_approach_km",
    ])


def analysis_metrics(cfg: SimulationScenarioConfig) -> list[str]:
    return [str(spec["name"]) for spec in analysis_metric_specs(cfg)]


def extract_analysis_metric(run_payload: dict[str, Any], metric_path: str) -> Any:
    return extract_study_metric(run_payload, metric_path)


def extract_analysis_metrics(run_payload: dict[str, Any], metric_specs: list[Any]) -> dict[str, Any]:
    specs = normalize_study_metric_specs(metric_specs)
    return {str(spec["name"]): extract_analysis_metric(run_payload, str(spec["path"])) for spec in specs}


def _serialize_analysis_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.12g}"
    if isinstance(value, (int, bool)):
        return str(value)
    if value is None:
        return "null"
    return json.dumps(value, sort_keys=True)


def _artifact_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if np.isnan(value):
            return "nan"
        if np.isposinf(value):
            return "inf"
        if np.isneginf(value):
            return "-inf"
        return f"{value:.12g}"
    if isinstance(value, (int, bool)):
        return str(value)
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True)
    return str(value)


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: _artifact_cell(row.get(name)) for name in fieldnames})


def _artifact_slug(value: Any) -> str:
    text = str(value or "value").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text[:80] or "value"


def _finite_xy(x_values: list[Any], y_values: list[Any]) -> tuple[np.ndarray, np.ndarray]:
    x: list[float] = []
    y: list[float] = []
    for xv, yv in zip(x_values, y_values):
        if isinstance(xv, bool) or isinstance(yv, bool):
            continue
        if not isinstance(xv, (int, float, np.integer, np.floating)):
            continue
        if not isinstance(yv, (int, float, np.integer, np.floating)):
            continue
        xf = float(xv)
        yf = float(yv)
        if np.isfinite(xf) and np.isfinite(yf):
            x.append(xf)
            y.append(yf)
    return np.array(x, dtype=float), np.array(y, dtype=float)


def build_lhs_parameter_summaries(
    *,
    cfg: SimulationScenarioConfig,
    metric_paths: list[str],
    run_details: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not run_details:
        return [], []
    summaries: list[dict[str, Any]] = []
    rankings: list[dict[str, Any]] = []
    for param in cfg.analysis.sensitivity.parameters:
        parameter_path = str(param.parameter_path)
        x_vals: list[float] = []
        for detail in run_details:
            value = dict(detail.get("sampled_parameters", {}) or {}).get(parameter_path)
            if isinstance(value, bool):
                x_vals.append(1.0 if value else 0.0)
            elif isinstance(value, (int, float, np.integer, np.floating)):
                x_vals.append(float(value))
            else:
                x_vals.append(float("nan"))
        x = np.array(x_vals, dtype=float)
        finite_x = np.isfinite(x)
        metric_correlations: dict[str, dict[str, float]] = {}
        max_abs_corr = 0.0
        for metric_path in metric_paths:
            y_vals: list[float] = []
            for detail in run_details:
                value = dict(detail.get("metrics", {}) or {}).get(metric_path)
                if isinstance(value, bool):
                    y_vals.append(1.0 if value else 0.0)
                elif isinstance(value, (int, float, np.integer, np.floating)):
                    y_vals.append(float(value))
                else:
                    y_vals.append(float("nan"))
            y = np.array(y_vals, dtype=float)
            finite = finite_x & np.isfinite(y)
            corr = float("nan")
            if int(np.sum(finite)) >= 3:
                x_ok = x[finite]
                y_ok = y[finite]
                if not np.allclose(np.std(x_ok), 0.0) and not np.allclose(np.std(y_ok), 0.0):
                    corr = float(np.corrcoef(x_ok, y_ok)[0, 1])
            abs_corr = abs(corr) if np.isfinite(corr) else float("nan")
            if np.isfinite(abs_corr):
                max_abs_corr = max(max_abs_corr, abs_corr)
            metric_correlations[metric_path] = {
                "correlation": corr,
                "abs_correlation": abs_corr,
            }
        finite_param = x[finite_x]
        summaries.append(
            {
                "parameter_path": parameter_path,
                "distribution": str(param.distribution),
                "sample_count": int(finite_param.size),
                "value_stats": {
                    "mean": float(np.mean(finite_param)) if finite_param.size else float("nan"),
                    "min": float(np.min(finite_param)) if finite_param.size else float("nan"),
                    "max": float(np.max(finite_param)) if finite_param.size else float("nan"),
                    "std": float(np.std(finite_param)) if finite_param.size else float("nan"),
                },
                "metric_correlations": metric_correlations,
            }
        )
        rankings.append(
            {
                "parameter_path": parameter_path,
                "distribution": str(param.distribution),
                "sample_count": int(finite_param.size),
                "max_abs_correlation": float(max_abs_corr),
            }
        )
    rankings.sort(key=lambda row: float(row.get("max_abs_correlation", 0.0)), reverse=True)
    return summaries, rankings


def aggregate_sensitivity_parameter_runs(
    *,
    parameter_path: str,
    metric_paths: list[str],
    run_details: list[dict[str, Any]],
    baseline_metrics: dict[str, Any],
) -> dict[str, Any]:
    grouped: dict[str, dict[str, Any]] = {}
    for detail in run_details:
        if str(detail.get("status", "completed")) != "completed":
            continue
        if str(detail.get("parameter_path", "")) != parameter_path:
            continue
        value = detail.get("parameter_value")
        value_key = _serialize_analysis_value(value)
        bucket = grouped.setdefault(
            value_key,
            {
                "parameter_value": value,
                "runs": [],
                "metrics": {},
            },
        )
        bucket["runs"].append(detail)

    for bucket in grouped.values():
        parameter_value = bucket.get("parameter_value")
        metric_summary: dict[str, Any] = {}
        for metric_path in metric_paths:
            values = [run["metrics"].get(metric_path) for run in bucket["runs"]]
            numeric = np.array(
                [
                    float(v)
                    for v in values
                    if isinstance(v, (int, float)) and not isinstance(v, bool) and np.isfinite(float(v))
                ],
                dtype=float,
            )
            entry: dict[str, Any] = {"sample_count": int(len(values))}
            if numeric.size:
                mean_value = float(np.mean(numeric))
                entry["mean"] = mean_value
                entry["min"] = float(np.min(numeric))
                entry["max"] = float(np.max(numeric))
                baseline_value = baseline_metrics.get(metric_path)
                if (
                    isinstance(baseline_value, (int, float))
                    and not isinstance(baseline_value, bool)
                    and np.isfinite(float(baseline_value))
                ):
                    baseline_float = float(baseline_value)
                    delta = float(mean_value - baseline_float)
                    entry["baseline"] = baseline_float
                    entry["delta_from_baseline"] = delta
                    if not np.isclose(baseline_float, 0.0):
                        entry["percent_change_from_baseline"] = float(100.0 * delta / abs(baseline_float))
                    if isinstance(parameter_value, (int, float)) and not isinstance(parameter_value, bool):
                        param_float = float(parameter_value)
                        if not np.isclose(param_float, 0.0):
                            entry["delta_per_parameter_unit"] = float(delta / param_float)
                            if not np.isclose(baseline_float, 0.0):
                                entry["elasticity_vs_zero"] = float((delta / baseline_float) / param_float)
            else:
                non_null = [v for v in values if v is not None]
                if non_null:
                    entry["values"] = non_null
                    baseline_value = baseline_metrics.get(metric_path)
                    if baseline_value is not None:
                        entry["matches_baseline"] = bool(all(v == baseline_value for v in non_null))
            metric_summary[metric_path] = entry
        bucket["metrics"] = metric_summary

    response_curve = [grouped[key] for key in sorted(grouped.keys())]
    max_abs_delta = 0.0
    max_abs_percent_change = 0.0
    max_abs_delta_per_unit = 0.0
    for bucket in response_curve:
        for metric_entry in dict(bucket.get("metrics", {}) or {}).values():
            delta = metric_entry.get("delta_from_baseline")
            if isinstance(delta, (int, float)) and np.isfinite(float(delta)):
                max_abs_delta = max(max_abs_delta, abs(float(delta)))
            percent_change = metric_entry.get("percent_change_from_baseline")
            if isinstance(percent_change, (int, float)) and np.isfinite(float(percent_change)):
                max_abs_percent_change = max(max_abs_percent_change, abs(float(percent_change)))
            delta_per_unit = metric_entry.get("delta_per_parameter_unit")
            if isinstance(delta_per_unit, (int, float)) and np.isfinite(float(delta_per_unit)):
                max_abs_delta_per_unit = max(max_abs_delta_per_unit, abs(float(delta_per_unit)))
    return {
        "parameter_path": parameter_path,
        "value_count": int(len(response_curve)),
        "max_abs_delta_from_baseline": float(max_abs_delta),
        "max_abs_percent_change_from_baseline": float(max_abs_percent_change),
        "max_abs_delta_per_parameter_unit": float(max_abs_delta_per_unit),
        "response_curve": response_curve,
    }


def build_two_parameter_grid_summaries(
    *,
    cfg: SimulationScenarioConfig,
    metric_paths: list[str],
    run_details: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    params = list(cfg.analysis.sensitivity.parameters or [])
    if len(params) != 2:
        return [], []
    x_param, y_param = params
    x_path = str(x_param.parameter_path)
    y_path = str(y_param.parameter_path)
    x_values = list(x_param.values or [])
    y_values = list(y_param.values or [])
    cells: list[dict[str, Any]] = []
    max_abs_span = 0.0
    metric_spans: dict[str, float] = {}

    for metric_path in metric_paths:
        z_grid = np.full((len(x_values), len(y_values)), np.nan, dtype=float)
        for run in run_details:
            if str(run.get("status", "completed")) != "completed":
                continue
            sampled = dict(run.get("sampled_parameters", {}) or {})
            try:
                i = x_values.index(sampled.get(x_path))
                j = y_values.index(sampled.get(y_path))
            except ValueError:
                continue
            value = dict(run.get("metrics", {}) or {}).get(metric_path)
            if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
                continue
            z_grid[i, j] = float(value)
        finite = z_grid[np.isfinite(z_grid)]
        span = float(np.max(finite) - np.min(finite)) if finite.size else 0.0
        metric_spans[metric_path] = span
        max_abs_span = max(max_abs_span, abs(span))
        cells.append(
            {
                "metric_path": metric_path,
                "x_parameter_path": x_path,
                "y_parameter_path": y_path,
                "x_values": x_values,
                "y_values": y_values,
                "z_grid": z_grid.tolist(),
                "finite_count": int(finite.size),
                "min": float(np.min(finite)) if finite.size else float("nan"),
                "max": float(np.max(finite)) if finite.size else float("nan"),
                "span": span,
            }
        )

    summary = {
        "interaction": f"{x_path}|{y_path}",
        "x_parameter_path": x_path,
        "y_parameter_path": y_path,
        "x_value_count": int(len(x_values)),
        "y_value_count": int(len(y_values)),
        "run_count": int(len(x_values) * len(y_values)),
        "metric_surfaces": cells,
        "metric_spans": metric_spans,
        "max_abs_metric_span": float(max_abs_span),
    }
    ranking = {
        "parameter_path": f"{x_path}|{y_path}",
        "x_parameter_path": x_path,
        "y_parameter_path": y_path,
        "value_count": int(len(x_values) * len(y_values)),
        "max_abs_metric_span": float(max_abs_span),
    }
    return [summary], [ranking]


def build_sensitivity_report_payload(
    *,
    cfg: SimulationScenarioConfig,
    config_path: str | Path,
    prepared: list[dict[str, Any]],
    completed: dict[int, dict[str, Any]],
    baseline: dict[str, Any] | None,
    metric_paths: list[str],
    metric_specs: list[Any] | None = None,
    sensitivity_method: str,
    parallel_enabled: bool,
    parallel_active: bool,
    parallel_workers: int,
    parallel_fallback_reason: str | None,
    preflight: dict[str, Any] | None = None,
    failure_policy: str = "fail_fast",
) -> dict[str, Any]:
    extraction_specs = list(metric_specs or metric_paths)
    runs: list[dict[str, Any]] = []
    for prepared_run in sorted(prepared, key=lambda entry: int(entry["iteration"])):
        idx = int(prepared_run["iteration"])
        completed_run = dict(completed.get(idx, {}) or {})
        status = str(completed_run.get("status", "completed" if completed_run else "missing"))
        payload = dict(completed_run.get("payload", {}) or {})
        if not payload:
            payload = {
                "summary": dict(completed_run.get("summary", {}) or {}),
                "metric_cache": dict(completed_run.get("metric_cache", {}) or {}),
            }
        payload.setdefault("closest_approach_km", _safe_float(completed_run.get("closest_approach_km")))
        metrics = (
            extract_analysis_metrics(payload, extraction_specs)
            if status == "completed"
            else {path: None for path in metric_paths}
        )
        runs.append(
            {
                "iteration": idx,
                "status": status,
                "error": str(completed_run.get("error", "") or ""),
                "parameter_path": prepared_run.get("parameter_path"),
                "parameter_value": prepared_run.get("parameter_value"),
                "value_index": int(prepared_run["value_index"]),
                "sampled_parameters": dict(prepared_run.get("sampled_parameters", {}) or {}),
                "summary": dict(completed_run.get("summary", payload.get("summary", {})) or {}),
                "closest_approach_km": _safe_float(completed_run.get("closest_approach_km")),
                "metrics": metrics,
            }
        )

    successful_run_count = sum(1 for run in runs if str(run.get("status")) == "completed")
    failed_run_count = sum(1 for run in runs if str(run.get("status")) == "failed")
    baseline_metrics = dict(baseline.get("metrics", {}) or {}) if baseline is not None else {}
    interaction_summaries: list[dict[str, Any]] = []
    if sensitivity_method == "lhs":
        parameter_summaries, parameter_rankings = build_lhs_parameter_summaries(
            cfg=cfg,
            metric_paths=metric_paths,
            run_details=runs,
        )
    elif sensitivity_method == "two_parameter_grid":
        interaction_summaries, parameter_rankings = build_two_parameter_grid_summaries(
            cfg=cfg,
            metric_paths=metric_paths,
            run_details=runs,
        )
        parameter_summaries = []
    else:
        parameter_summaries = [
            aggregate_sensitivity_parameter_runs(
                parameter_path=str(param.parameter_path),
                metric_paths=metric_paths,
                run_details=runs,
                baseline_metrics=baseline_metrics,
            )
            for param in cfg.analysis.sensitivity.parameters
        ]
        parameter_rankings = sorted(
            (
                {
                    "parameter_path": str(summary.get("parameter_path", "")),
                    "max_abs_delta_from_baseline": float(summary.get("max_abs_delta_from_baseline", 0.0)),
                    "max_abs_percent_change_from_baseline": float(
                        summary.get("max_abs_percent_change_from_baseline", 0.0)
                    ),
                    "max_abs_delta_per_parameter_unit": float(summary.get("max_abs_delta_per_parameter_unit", 0.0)),
                    "value_count": int(summary.get("value_count", 0)),
                }
                for summary in parameter_summaries
            ),
            key=lambda entry: (
                float(entry["max_abs_percent_change_from_baseline"]),
                float(entry["max_abs_delta_from_baseline"]),
            ),
            reverse=True,
        )

    cfg_json = json.dumps(cfg.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)
    source_path = Path(config_path).resolve()
    reproducibility = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": hashlib.sha256(cfg_json.encode("utf-8")).hexdigest(),
        "source_config_path": str(source_path),
        "source_config_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest()
        if source_path.is_file()
        else None,
        "sensitivity_method": sensitivity_method,
        "seed": int(cfg.analysis.sensitivity.seed),
    }
    agg = {
        "generated_utc": reproducibility["generated_utc"],
        "config_path": str(Path(config_path).resolve()),
        "scenario_name": cfg.scenario_name,
        "scenario_description": cfg.scenario_description,
        "analysis": {
            "enabled": True,
            "study_type": "sensitivity",
            "method": sensitivity_method,
            "parallel_enabled": bool(parallel_active),
            "parallel_requested": bool(parallel_enabled and len(prepared) > 1),
            "parallel_workers": int(parallel_workers if parallel_active else 1),
            "metrics": metric_paths,
            "parameter_count": int(len(cfg.analysis.sensitivity.parameters)),
            "run_count": int(len(runs)),
            "successful_run_count": int(successful_run_count),
            "failed_run_count": int(failed_run_count),
            "failure_policy": str(failure_policy or "fail_fast"),
            "samples": int(cfg.analysis.sensitivity.samples if sensitivity_method == "lhs" else len(runs)),
            "seed": int(cfg.analysis.sensitivity.seed),
            "preflight": dict(preflight or {"run_count": int(len(prepared)), "error_count": 0, "errors": []}),
        },
        "monte_carlo": {"enabled": False},
        "baseline": baseline,
        "parameter_summaries": parameter_summaries,
        "interaction_summaries": interaction_summaries,
        "parameter_rankings": parameter_rankings,
        "reproducibility": reproducibility,
        "runs": runs,
        "artifacts": {},
    }
    if parallel_fallback_reason is not None:
        agg["analysis"]["parallel_fallback_reason"] = str(parallel_fallback_reason)
    return agg


def _build_sensitivity_runs_csv_rows(payload: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    metric_paths = list(dict(payload.get("analysis", {}) or {}).get("metrics", []) or [])
    base_fields = [
        "iteration",
        "status",
        "error",
        "parameter_path",
        "parameter_value",
        "value_index",
        "sampled_parameters",
        "terminated_early",
        "termination_reason",
        "duration_s",
        "closest_approach_km",
    ]
    fieldnames = list(base_fields) + [f"metric:{path}" for path in metric_paths]
    rows: list[dict[str, Any]] = []
    for run in list(payload.get("runs", []) or []):
        summary = dict(run.get("summary", {}) or {})
        metrics = dict(run.get("metrics", {}) or {})
        row: dict[str, Any] = {
            "iteration": run.get("iteration"),
            "status": run.get("status"),
            "error": run.get("error"),
            "parameter_path": run.get("parameter_path"),
            "parameter_value": run.get("parameter_value"),
            "value_index": run.get("value_index"),
            "sampled_parameters": dict(run.get("sampled_parameters", {}) or {}),
            "terminated_early": summary.get("terminated_early"),
            "termination_reason": summary.get("termination_reason"),
            "duration_s": summary.get("duration_s"),
            "closest_approach_km": run.get("closest_approach_km"),
        }
        for path in metric_paths:
            row[f"metric:{path}"] = metrics.get(path)
        rows.append(row)
    return rows, fieldnames


def _build_sensitivity_rankings_csv_rows(payload: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    rankings = [dict(row or {}) for row in list(payload.get("parameter_rankings", []) or [])]
    preferred = [
        "rank",
        "parameter_path",
        "value_count",
        "sample_count",
        "distribution",
        "max_abs_delta_from_baseline",
        "max_abs_percent_change_from_baseline",
        "max_abs_delta_per_parameter_unit",
        "max_abs_correlation",
        "x_parameter_path",
        "y_parameter_path",
        "max_abs_metric_span",
    ]
    extra = sorted({key for row in rankings for key in row.keys()} - set(preferred))
    fieldnames = preferred + extra
    rows: list[dict[str, Any]] = []
    for idx, row in enumerate(rankings, start=1):
        out = dict(row)
        out["rank"] = idx
        rows.append(out)
    return rows, fieldnames


def _format_metric_lines(metrics: dict[str, Any], *, limit: int = 6) -> list[str]:
    lines: list[str] = []
    for idx, (key, value) in enumerate(metrics.items()):
        if idx >= limit:
            lines.append(f"- ... {len(metrics) - limit} more")
            break
        lines.append(f"- `{key}`: `{_artifact_cell(value)}`")
    return lines


def _build_sensitivity_markdown_report(payload: dict[str, Any]) -> str:
    analysis = dict(payload.get("analysis", {}) or {})
    baseline = dict(payload.get("baseline", {}) or {})
    preflight = dict(analysis.get("preflight", {}) or {})
    rankings = [dict(row or {}) for row in list(payload.get("parameter_rankings", []) or [])]
    artifacts = dict(payload.get("artifacts", {}) or {})

    lines = [
        f"# Sensitivity Analysis Report: {payload.get('scenario_name', 'unknown')}",
        "",
    ]
    description = str(payload.get("scenario_description", "") or "").strip()
    if description:
        lines.extend([description, ""])

    lines.extend(
        [
            "## Study",
            "",
            f"- Method: `{analysis.get('method', 'unknown')}`",
            f"- Runs: `{analysis.get('run_count', 0)}`",
            f"- Successful runs: `{analysis.get('successful_run_count', analysis.get('run_count', 0))}`",
            f"- Failed runs: `{analysis.get('failed_run_count', 0)}`",
            f"- Failure policy: `{analysis.get('failure_policy', 'fail_fast')}`",
            f"- Parameters: `{analysis.get('parameter_count', 0)}`",
            f"- Metrics: `{', '.join(str(x) for x in list(analysis.get('metrics', []) or []))}`",
            f"- Preflight: `{preflight.get('error_count', 0)}` error(s) across `{preflight.get('run_count', 0)}` generated run(s)",
            "",
            "## Baseline",
            "",
        ]
    )
    if baseline:
        lines.append(f"- Source: `{baseline.get('source', 'unknown')}`")
        if baseline.get("path"):
            lines.append(f"- Path: `{baseline.get('path')}`")
        if baseline.get("output_dir"):
            lines.append(f"- Output directory: `{baseline.get('output_dir')}`")
        metric_lines = _format_metric_lines(dict(baseline.get("metrics", {}) or {}))
        if metric_lines:
            lines.extend(metric_lines)
    else:
        lines.append("- Source: `none`")

    failed_runs = [
        dict(run or {}) for run in list(payload.get("runs", []) or []) if str(dict(run or {}).get("status")) == "failed"
    ]
    if failed_runs:
        lines.extend(["", "## Failed Runs", ""])
        lines.append("| Iteration | Parameter | Value | Error |")
        lines.append("|---:|---|---:|---|")
        for run in failed_runs[:10]:
            error = str(run.get("error", "") or "").replace("|", "\\|")
            lines.append(
                f"| {run.get('iteration')} | `{run.get('parameter_path', '')}` | {_artifact_cell(run.get('parameter_value'))} | {error} |"
            )
        if len(failed_runs) > 10:
            lines.append(f"\nAdditional failed runs: `{len(failed_runs) - 10}`")

    lines.extend(["", "## Top Parameter Drivers", ""])
    if rankings:
        lines.append("| Rank | Parameter | Score | Secondary |")
        lines.append("|---:|---|---:|---:|")
        for idx, row in enumerate(rankings[:10], start=1):
            if "max_abs_percent_change_from_baseline" in row:
                score = _artifact_cell(row.get("max_abs_percent_change_from_baseline"))
                secondary = _artifact_cell(row.get("max_abs_delta_from_baseline"))
            elif "max_abs_metric_span" in row:
                score = _artifact_cell(row.get("max_abs_metric_span"))
                secondary = _artifact_cell(row.get("value_count"))
            else:
                score = _artifact_cell(row.get("max_abs_correlation"))
                secondary = _artifact_cell(row.get("sample_count"))
            lines.append(f"| {idx} | `{row.get('parameter_path', '')}` | {score} | {secondary} |")
    else:
        lines.append("No parameter rankings were produced.")

    lines.extend(["", "## Artifacts", ""])
    for key in sorted(artifacts):
        lines.append(f"- `{key}`: `{artifacts[key]}`")
    lines.append("")
    return "\n".join(lines)


def write_sensitivity_plot_artifacts(
    *,
    outdir: Path,
    payload: dict[str, Any],
) -> dict[str, Any]:
    analysis = dict(payload.get("analysis", {}) or {})
    method = str(analysis.get("method", "") or "").strip().lower()
    metric_paths = [str(x) for x in list(analysis.get("metrics", []) or []) if str(x)]
    artifacts = dict(payload.get("artifacts", {}) or {})
    dpi = 150
    metadata = artifact_metadata(scenario_name=str(payload.get("scenario_name", "sensitivity") or "sensitivity"))
    style_name = "oel_dark"

    try:
        import matplotlib

        # Sensitivity plots are always saved to disk, so force a non-interactive
        # backend instead of inheriting an interactive backend that can abort headless runs.
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        from sim.utils.figure_size import cap_figsize
    except Exception as exc:
        artifacts["sensitivity_plot_error"] = f"{type(exc).__name__}: {exc}"
        payload["artifacts"] = artifacts
        return payload

    def _save(fig: Any, path: Path) -> str:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        save_oel_figure(fig, path, dpi=dpi, metadata=metadata, artifact_id=path.stem, style_name=style_name)
        plt.close(fig)
        return str(path)

    if method == "one_at_a_time":
        response_paths: dict[str, str] = {}
        for summary in list(payload.get("parameter_summaries", []) or []):
            parameter_path = str(dict(summary or {}).get("parameter_path", "") or "")
            curve = list(dict(summary or {}).get("response_curve", []) or [])
            if not parameter_path or not curve:
                continue
            for metric_path in metric_paths:
                x_values = [dict(point or {}).get("parameter_value") for point in curve]
                y_values: list[Any] = []
                for point in curve:
                    metric_entry = dict(dict(point or {}).get("metrics", {}) or {}).get(metric_path)
                    y_values.append(dict(metric_entry or {}).get("mean") if isinstance(metric_entry, dict) else None)
                x, y = _finite_xy(x_values, y_values)
                if x.size == 0:
                    continue
                order = np.argsort(x)
                x = x[order]
                y = y[order]
                fig, ax = plt.subplots(figsize=cap_figsize(7.5, 4.8))
                ax.plot(x, y, marker="o", linewidth=1.6)
                ax.set_title(f"{parameter_path} vs {metric_path}")
                ax.set_xlabel(parameter_path)
                ax.set_ylabel(metric_path)
                ax.grid(True, alpha=0.3)
                key = f"{parameter_path}|{metric_path}"
                path = (
                    outdir
                    / f"master_analysis_sensitivity_response_{_artifact_slug(parameter_path)}__{_artifact_slug(metric_path)}.png"
                )
                response_paths[key] = _save(fig, path)
        if response_paths:
            artifacts["response_curve_pngs"] = response_paths

    if method == "lhs":
        scatter_paths: dict[str, str] = {}
        runs = [dict(row or {}) for row in list(payload.get("runs", []) or [])]
        for ranking in list(payload.get("parameter_rankings", []) or []):
            parameter_path = str(dict(ranking or {}).get("parameter_path", "") or "")
            if not parameter_path:
                continue
            for metric_path in metric_paths:
                x_values = [dict(run.get("sampled_parameters", {}) or {}).get(parameter_path) for run in runs]
                y_values = [dict(run.get("metrics", {}) or {}).get(metric_path) for run in runs]
                x, y = _finite_xy(x_values, y_values)
                if x.size == 0:
                    continue
                fig, ax = plt.subplots(figsize=cap_figsize(7.5, 4.8))
                ax.scatter(x, y, s=28, alpha=0.85)
                ax.set_title(f"{parameter_path} vs {metric_path}")
                ax.set_xlabel(parameter_path)
                ax.set_ylabel(metric_path)
                ax.grid(True, alpha=0.3)
                key = f"{parameter_path}|{metric_path}"
                path = (
                    outdir
                    / f"master_analysis_sensitivity_lhs_scatter_{_artifact_slug(parameter_path)}__{_artifact_slug(metric_path)}.png"
                )
                scatter_paths[key] = _save(fig, path)
        if scatter_paths:
            artifacts["lhs_scatter_pngs"] = scatter_paths

    if method == "two_parameter_grid":
        heatmap_paths: dict[str, str] = {}
        for summary in list(payload.get("interaction_summaries", []) or []):
            summary_dict = dict(summary or {})
            x_path = str(summary_dict.get("x_parameter_path", "") or "")
            y_path = str(summary_dict.get("y_parameter_path", "") or "")
            x_values = list(summary_dict.get("x_values", []) or [])
            y_values = list(summary_dict.get("y_values", []) or [])
            for surface in list(summary_dict.get("metric_surfaces", []) or []):
                surface_dict = dict(surface or {})
                metric_path = str(surface_dict.get("metric_path", "") or "")
                z = np.array(surface_dict.get("z_grid", []), dtype=float)
                if not x_path or not y_path or not metric_path or z.size == 0:
                    continue
                fig, ax = plt.subplots(
                    figsize=cap_figsize(max(6.5, 0.7 * len(y_values) + 4.0), max(4.8, 0.55 * len(x_values) + 3.0))
                )
                im = ax.imshow(z, aspect="auto", origin="lower")
                ax.set_xticks(np.arange(len(y_values), dtype=float))
                ax.set_yticks(np.arange(len(x_values), dtype=float))
                ax.set_xticklabels([_artifact_cell(v) for v in y_values], rotation=30, ha="right")
                ax.set_yticklabels([_artifact_cell(v) for v in x_values])
                ax.set_xlabel(y_path)
                ax.set_ylabel(x_path)
                ax.set_title(f"{metric_path} interaction surface")
                fig.colorbar(im, ax=ax, label=metric_path)
                key = f"{x_path}|{y_path}|{metric_path}"
                path = outdir / (
                    "master_analysis_sensitivity_grid_"
                    f"{_artifact_slug(x_path)}__{_artifact_slug(y_path)}__{_artifact_slug(metric_path)}.png"
                )
                heatmap_paths[key] = _save(fig, path)
        if heatmap_paths:
            artifacts["grid_heatmap_pngs"] = heatmap_paths

    rankings = [dict(row or {}) for row in list(payload.get("parameter_rankings", []) or [])]
    if rankings:
        if method == "lhs":
            score_key = "max_abs_correlation"
            ylabel = "max |correlation|"
        elif method == "two_parameter_grid":
            score_key = "max_abs_metric_span"
            ylabel = "max metric span"
        else:
            score_key = "max_abs_percent_change_from_baseline"
            ylabel = "max |% change from baseline|"
        labels: list[str] = []
        scores: list[float] = []
        for row in rankings[:12]:
            score = row.get(score_key)
            if not isinstance(score, (int, float, np.integer, np.floating)) or not np.isfinite(float(score)):
                continue
            labels.append(str(row.get("parameter_path", "")))
            scores.append(float(score))
        if scores:
            fig, ax = plt.subplots(figsize=cap_figsize(max(7.0, 0.8 * len(scores) + 4.0), 4.8))
            idx = np.arange(len(scores), dtype=float)
            ax.bar(idx, scores, alpha=0.85)
            ax.set_xticks(idx)
            ax.set_xticklabels(labels, rotation=30, ha="right")
            ax.set_ylabel(ylabel)
            ax.set_title("Sensitivity Parameter Rankings")
            ax.grid(True, axis="y", alpha=0.3)
            artifacts["ranking_bar_png"] = _save(fig, outdir / "master_analysis_sensitivity_rankings.png")

    baseline_metrics = dict(dict(payload.get("baseline", {}) or {}).get("metrics", {}) or {})
    if baseline_metrics and metric_paths:
        labels: list[str] = []
        deltas: list[float] = []
        for metric_path in metric_paths:
            baseline_value = baseline_metrics.get(metric_path)
            if not isinstance(baseline_value, (int, float, np.integer, np.floating)) or isinstance(
                baseline_value, bool
            ):
                continue
            baseline_float = float(baseline_value)
            if not np.isfinite(baseline_float):
                continue
            values: list[float] = []
            for run in list(payload.get("runs", []) or []):
                value = dict(dict(run or {}).get("metrics", {}) or {}).get(metric_path)
                if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
                    value_float = float(value)
                    if np.isfinite(value_float):
                        values.append(value_float)
            if not values:
                continue
            delta = float(max(values, key=lambda item: abs(float(item) - baseline_float)) - baseline_float)
            labels.append(metric_path)
            deltas.append(delta)
        if deltas:
            order = np.argsort(np.abs(np.array(deltas, dtype=float)))[::-1][:12]
            plot_labels = [labels[int(i)] for i in order]
            plot_deltas = [deltas[int(i)] for i in order]
            fig, ax = plt.subplots(figsize=cap_figsize(max(7.5, 0.75 * len(plot_deltas) + 4.0), 4.8))
            idx = np.arange(len(plot_deltas), dtype=float)
            colors = ["tab:green" if value >= 0.0 else "tab:red" for value in plot_deltas]
            ax.bar(idx, plot_deltas, color=colors, alpha=0.85)
            ax.axhline(0.0, color="k", linewidth=1.0)
            ax.set_xticks(idx)
            ax.set_xticklabels(plot_labels, rotation=30, ha="right")
            ax.set_ylabel("max delta from baseline")
            ax.set_title("Sensitivity Metric Delta From Baseline")
            ax.grid(True, axis="y", alpha=0.3)
            artifacts["baseline_delta_bar_png"] = _save(
                fig, outdir / "master_analysis_sensitivity_baseline_delta.png"
            )

    payload["artifacts"] = artifacts
    return payload


def write_sensitivity_summary_artifact(
    *,
    outdir: Path,
    payload: dict[str, Any],
) -> dict[str, Any]:
    artifacts = dict(payload.get("artifacts", {}) or {})
    summary_path = outdir / "master_analysis_sensitivity_summary.json"
    artifacts["summary_json"] = str(summary_path)
    payload["artifacts"] = artifacts
    payload = write_sensitivity_plot_artifacts(outdir=outdir, payload=payload)
    artifacts = dict(payload.get("artifacts", {}) or {})

    runs_csv_path = outdir / "master_analysis_sensitivity_runs.csv"
    runs_rows, runs_fields = _build_sensitivity_runs_csv_rows(payload)
    _write_csv(runs_csv_path, runs_rows, runs_fields)
    artifacts["runs_csv"] = str(runs_csv_path)

    rankings_csv_path = outdir / "master_analysis_sensitivity_rankings.csv"
    ranking_rows, ranking_fields = _build_sensitivity_rankings_csv_rows(payload)
    _write_csv(rankings_csv_path, ranking_rows, ranking_fields)
    artifacts["rankings_csv"] = str(rankings_csv_path)

    report_md_path = outdir / "master_analysis_sensitivity_report.md"
    payload["artifacts"] = artifacts
    report_md_path.write_text(_build_sensitivity_markdown_report(payload), encoding="utf-8")
    artifacts["report_md"] = str(report_md_path)

    payload["artifacts"] = artifacts
    index_path = write_output_index(
        outdir=outdir,
        workflow="sensitivity",
        title=str(payload.get("scenario_name", "sensitivity")),
        payload=payload,
        artifacts=artifacts,
    )
    artifacts["output_index_md"] = str(index_path)
    payload["artifacts"] = artifacts
    write_json(str(summary_path), payload)
    review_outputs = write_workflow_review(
        output_dir=outdir,
        workflow_type="sensitivity",
        title=str(payload.get("scenario_name", "sensitivity")),
        scenario_name=str(payload.get("scenario_name", "sensitivity")),
        status="complete",
        summary={
            "method": dict(payload.get("analysis", {}) or {}).get("method"),
            "run_count": dict(payload.get("analysis", {}) or {}).get("run_count"),
            "successful_run_count": dict(payload.get("analysis", {}) or {}).get("successful_run_count"),
            "failed_run_count": dict(payload.get("analysis", {}) or {}).get("failed_run_count"),
            "parameter_count": dict(payload.get("analysis", {}) or {}).get("parameter_count"),
        },
        artifacts=artifacts,
        recommended_queries=[
            {
                "name": "workflow_metadata",
                "description": "Sensitivity workflow metadata.",
                "sql": "SELECT * FROM workflow_metadata",
            },
            {
                "name": "sensitivity_rankings",
                "description": "Ranked sensitivity drivers.",
                "sql": "SELECT rank, parameter_path, metric_path, effect_size FROM sensitivity_rankings ORDER BY rank",
            },
            {
                "name": "sensitivity_runs",
                "description": "Generated sensitivity runs and changed values.",
                "sql": "SELECT run_id, status, parameter_path, parameter_value, output_dir FROM sensitivity_runs",
            },
        ],
        recommended_review_order=[
            "Open master_analysis_sensitivity_report.md for the human-readable study summary.",
            "Query sensitivity_rankings for ranked driver evidence.",
            "Query sensitivity_runs when a ranking needs run-level inspection.",
        ],
        source_config=str(payload.get("config_path", "") or ""),
        provenance={"writer": "sim.reporting.sensitivity.write_sensitivity_summary_artifact"},
        tables={
            "sensitivity_runs": _sensitivity_review_run_rows(payload),
            "sensitivity_rankings": _sensitivity_review_ranking_rows(payload),
        },
    )
    artifacts.update(review_outputs)
    payload["artifacts"] = artifacts
    index_path = write_output_index(
        outdir=outdir,
        workflow="sensitivity",
        title=str(payload.get("scenario_name", "sensitivity")),
        payload=payload,
        artifacts=artifacts,
    )
    artifacts["output_index_md"] = str(index_path)
    payload["artifacts"] = artifacts
    write_json(str(summary_path), payload)
    return payload


def _sensitivity_review_run_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in list(payload.get("runs", []) or []):
        summary = dict(run.get("summary", {}) or {})
        rows.append(
            {
                "run_id": run.get("iteration"),
                "status": run.get("status"),
                "error": run.get("error"),
                "parameter_path": run.get("parameter_path"),
                "parameter_value": run.get("parameter_value"),
                "value_index": run.get("value_index"),
                "closest_approach_km": run.get("closest_approach_km"),
                "duration_s": summary.get("duration_s"),
                "terminated_early": summary.get("terminated_early"),
                "termination_reason": summary.get("termination_reason"),
                "output_dir": summary.get("output_dir"),
                "sampled_parameters_json": json.dumps(dict(run.get("sampled_parameters", {}) or {}), sort_keys=True),
                "metrics_json": json.dumps(dict(run.get("metrics", {}) or {}), sort_keys=True),
            }
        )
    return rows


def _sensitivity_review_ranking_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    analysis = dict(payload.get("analysis", {}) or {})
    method = str(analysis.get("method", "") or "")
    rows: list[dict[str, Any]] = []
    for idx, ranking in enumerate(list(payload.get("parameter_rankings", []) or []), start=1):
        row = dict(ranking or {})
        effect_size = (
            row.get("max_abs_percent_change_from_baseline")
            or row.get("max_abs_delta_from_baseline")
            or row.get("max_abs_correlation")
            or row.get("max_abs_metric_span")
        )
        row.setdefault("rank", idx)
        row.setdefault("metric_path", "")
        row.setdefault("method", method)
        row.setdefault("effect_size", effect_size)
        rows.append(row)
    return rows
