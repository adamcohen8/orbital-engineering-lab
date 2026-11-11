from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from sim.analysis.orbital_delivery import aggregate_orbital_delivery_runs
from sim.config import SimulationScenarioConfig
from sim.execution.metrics import normalize_study_metric_specs
from sim.utils.io import write_json


def _study_metric_summary(run_details: list[dict[str, Any]], specs: list[dict[str, str]]) -> dict[str, dict[str, Any]]:
    from sim.execution.monte_carlo_support import quantile_stats

    summary: dict[str, dict[str, Any]] = {}
    for spec in specs:
        name = str(spec["name"])
        values = [dict(detail.get("metrics", {}) or {}).get(name) for detail in run_details]
        numeric: list[float] = []
        category_counts: dict[str, int] = {}
        available = 0
        for value in values:
            if isinstance(value, bool):
                category = str(value).lower()
            elif isinstance(value, (int, float, np.integer, np.floating)):
                number = float(value)
                if not np.isfinite(number):
                    continue
                numeric.append(number)
                available += 1
                continue
            elif isinstance(value, str):
                category = value
            else:
                continue
            category_counts[category] = category_counts.get(category, 0) + 1
            available += 1
        entry: dict[str, Any] = {
            "path": str(spec["path"]),
            "run_count": len(values),
            "available_count": available,
            "missing_count": len(values) - available,
        }
        if numeric:
            entry["numeric_stats"] = quantile_stats(numeric, (5.0, 50.0, 95.0))
        if category_counts:
            entry["value_counts"] = dict(sorted(category_counts.items()))
        summary[name] = entry
    return summary


def _comparison_summary(
    cfg: SimulationScenarioConfig, run_details: list[dict[str, Any]], metric_specs: list[dict[str, str]],
) -> dict[str, Any] | None:
    from sim.execution.monte_carlo_support import quantile_stats, safe_float

    comparison = cfg.monte_carlo.comparison
    if comparison is None:
        return None
    bundle = next(bundle for bundle in cfg.monte_carlo.bundles if bundle.name == comparison.bundle)
    labels = [option.label for option in bundle.options]
    grouped = {
        label: [detail for detail in run_details if dict(detail.get("comparison", {}) or {}).get("option") == label]
        for label in labels
    }
    options = {
        label: {
            "run_count": len(group),
            "pass_rate": float(np.mean([bool(detail.get("pass")) for detail in group])) if group else None,
            "study_metric_summary": _study_metric_summary(group, metric_specs),
        }
        for label, group in grouped.items()
    }
    summary: dict[str, Any] = {
        "bundle": bundle.name,
        "paired": bool(comparison.paired),
        "runs_per_option": int(comparison.runs_per_option),
        "reference_option": labels[0],
        "options": options,
    }
    if comparison.paired:
        by_replicate = {
            label: {int(detail["comparison"]["replicate"]): detail for detail in group}
            for label, group in grouped.items()
        }
        deltas: dict[str, Any] = {}
        reference = by_replicate[labels[0]]
        for label in labels[1:]:
            paired_ids = sorted(set(reference) & set(by_replicate[label]))
            metric_deltas: dict[str, Any] = {}
            for spec in metric_specs:
                name = str(spec["name"])
                values = []
                for replicate in paired_ids:
                    left = safe_float(dict(by_replicate[label][replicate].get("metrics", {}) or {}).get(name))
                    right = safe_float(dict(reference[replicate].get("metrics", {}) or {}).get(name))
                    if np.isfinite(left) and np.isfinite(right):
                        values.append(left - right)
                metric_deltas[name] = {"paired_count": len(values), "delta_stats": quantile_stats(values, (5.0, 50.0, 95.0))}
            pass_deltas = [
                int(bool(by_replicate[label][replicate].get("pass")))
                - int(bool(reference[replicate].get("pass")))
                for replicate in paired_ids
            ]
            deltas[label] = {
                "paired_count": len(paired_ids),
                "pass_delta_stats": quantile_stats(pass_deltas, (5.0, 50.0, 95.0)),
                "study_metric_deltas": metric_deltas,
            }
        summary["paired_differences_vs_reference"] = deltas
    return summary


def build_monte_carlo_report_payload(
    *,
    cfg: SimulationScenarioConfig,
    config_path: str | Path,
    root: dict[str, Any],
    repo_root: Path,
    runs: list[dict[str, Any]],
    run_details: list[dict[str, Any]],
    closest_approach_km_runs: list[float],
    duration_runs_s: list[float],
    total_dv_runs_m_s: list[float],
    guardrail_event_runs: list[int],
    failure_mode_counts: dict[str, int],
    dv_budget_m_s_by_object: dict[str, float],
    gates: Any,
    mc_out_cfg: dict[str, Any],
    varies_metadata_seed: bool,
    parallel_active: bool,
    parallel_enabled: bool,
    total_iters: int,
    parallel_workers: int,
    parallel_fallback_reason: str | None = None,
) -> dict[str, Any]:
    from sim.execution.monte_carlo_support import (
        aggregate_knowledge_consistency_from_runs,
        aggregate_knowledge_detection_from_runs,
        build_parameter_sensitivity_rankings,
        get_git_commit_sha,
        infer_model_profile,
        quantile_stats,
        safe_float,
    )

    legacy_gates = dict(gates if isinstance(gates, dict) else {})
    durations_s = np.array([float(dict(r.get("summary", {}) or {}).get("duration_s", 0.0)) for r in runs], dtype=float)
    terminated_early_flags = np.array(
        [1.0 if bool(dict(r.get("summary", {}) or {}).get("terminated_early", False)) else 0.0 for r in runs],
        dtype=float,
    )
    termination_reason_counts: dict[str, int] = {}
    dv_by_object: dict[str, list[float]] = {}
    burn_samples_by_object: dict[str, list[float]] = {}
    actuator_samples_by_object: dict[str, list[float]] = {}
    rcs_samples_by_object: dict[str, list[float]] = {}
    magnetorquer_samples_by_object: dict[str, list[float]] = {}
    cmg_samples_by_object: dict[str, list[float]] = {}
    wheel_desat_samples_by_object: dict[str, list[float]] = {}
    max_gimbal_rad_by_object: dict[str, list[float]] = {}
    max_rcs_force_n_by_object: dict[str, list[float]] = {}
    max_ep_thrust_n_by_object: dict[str, list[float]] = {}
    for entry in runs:
        summary = dict(entry.get("summary", {}) or {})
        term_reason = summary.get("termination_reason")
        if term_reason is not None:
            key = str(term_reason)
            termination_reason_counts[key] = int(termination_reason_counts.get(key, 0) + 1)
        thrust_stats = dict(summary.get("thrust_stats", {}) or {})
        for oid, thrust in thrust_stats.items():
            thrust_dict = dict(thrust or {})
            dv_by_object.setdefault(str(oid), []).append(float(thrust_dict.get("total_dv_m_s", 0.0)))
            burn_samples_by_object.setdefault(str(oid), []).append(float(thrust_dict.get("burn_samples", 0.0)))
        actuator_summary = dict(summary.get("actuator_diagnostics_summary", {}) or {})
        for oid, raw_diag in actuator_summary.items():
            diag = dict(raw_diag or {})
            key = str(oid)
            actuator_samples_by_object.setdefault(key, []).append(float(diag.get("actuator_stack_samples", 0.0)))
            rcs_samples_by_object.setdefault(key, []).append(float(diag.get("rcs_active_samples", 0.0)))
            magnetorquer_samples_by_object.setdefault(key, []).append(float(diag.get("magnetorquer_active_samples", 0.0)))
            cmg_samples_by_object.setdefault(key, []).append(float(diag.get("cmg_active_samples", 0.0)))
            wheel_desat_samples_by_object.setdefault(key, []).append(float(diag.get("wheel_desaturation_samples", 0.0)))
            max_gimbal_rad_by_object.setdefault(key, []).append(float(diag.get("max_gimbal_angle_rad", 0.0)))
            max_rcs_force_n_by_object.setdefault(key, []).append(
                float(diag.get("max_commanded_rcs_thruster_force_n", 0.0))
            )
            max_ep_thrust_n_by_object.setdefault(key, []).append(
                float(diag.get("max_electric_propulsion_thrust_n", 0.0))
            )

    dv_remaining_m_s_by_object: dict[str, list[float]] = {}
    for oid in sorted(dv_budget_m_s_by_object.keys()):
        vals: list[float] = []
        for detail in run_details:
            remaining = safe_float(dict(detail.get("delta_v_remaining_m_s_by_object", {}) or {}).get(oid))
            if np.isfinite(remaining):
                vals.append(float(remaining))
        if vals:
            dv_remaining_m_s_by_object[oid] = vals

    by_object_stats: dict[str, dict[str, float]] = {}
    all_obj_ids = sorted(set(dv_by_object.keys()) | set(burn_samples_by_object.keys()))
    for oid in all_obj_ids:
        dv_arr = np.array(dv_by_object.get(oid, []), dtype=float)
        burn_arr = np.array(burn_samples_by_object.get(oid, []), dtype=float)
        dv_finite = dv_arr[np.isfinite(dv_arr)]
        burn_finite = burn_arr[np.isfinite(burn_arr)]
        by_object_stats[oid] = {
            "total_dv_m_s_mean": float(np.mean(dv_finite)) if dv_finite.size else float("nan"),
            "total_dv_m_s_min": float(np.min(dv_finite)) if dv_finite.size else float("nan"),
            "total_dv_m_s_max": float(np.max(dv_finite)) if dv_finite.size else float("nan"),
            "total_dv_m_s_p95": float(np.percentile(dv_finite, 95)) if dv_finite.size else float("nan"),
            "burn_samples_mean": float(np.mean(burn_finite)) if burn_finite.size else float("nan"),
            "burn_samples_p95": float(np.percentile(burn_finite, 95)) if burn_finite.size else float("nan"),
        }

    ca_arr_full = np.array(closest_approach_km_runs, dtype=float)
    ca_finite = ca_arr_full[np.isfinite(ca_arr_full)]
    duration_arr = np.array(duration_runs_s, dtype=float)
    duration_finite = duration_arr[np.isfinite(duration_arr)]
    total_dv_arr = np.array(total_dv_runs_m_s, dtype=float)
    total_dv_finite = total_dv_arr[np.isfinite(total_dv_arr)]
    guardrail_arr = np.array(guardrail_event_runs, dtype=float)
    guardrail_finite = guardrail_arr[np.isfinite(guardrail_arr)]
    pass_flags = np.array([1.0 if bool(d.get("pass", False)) else 0.0 for d in run_details], dtype=float)
    pass_rate = float(np.mean(pass_flags)) if pass_flags.size else 0.0
    guardrail_violation_flags = np.array(
        [1.0 if int(d.get("guardrail_events", 0)) > 0 else 0.0 for d in run_details], dtype=float
    )

    aggregate_stats = {
        "runs": int(len(run_details)),
        "runs_requested": int(total_iters),
        "runs_executed": int(len(run_details)),
        "duration_s_mean": float(np.mean(duration_finite)) if duration_finite.size else float("nan"),
        "duration_s_min": float(np.min(duration_finite)) if duration_finite.size else float("nan"),
        "duration_s_max": float(np.max(duration_finite)) if duration_finite.size else float("nan"),
        "duration_s_p50": float(np.percentile(duration_finite, 50)) if duration_finite.size else float("nan"),
        "duration_s_p90": float(np.percentile(duration_finite, 90)) if duration_finite.size else float("nan"),
        "duration_s_p95": float(np.percentile(duration_finite, 95)) if duration_finite.size else float("nan"),
        "duration_s_p99": float(np.percentile(duration_finite, 99)) if duration_finite.size else float("nan"),
        "terminated_early_rate": float(np.mean(terminated_early_flags)) if terminated_early_flags.size else 0.0,
        "closest_approach_km_mean": float(np.mean(ca_finite)) if ca_finite.size else float("nan"),
        "closest_approach_km_min": float(np.min(ca_finite)) if ca_finite.size else float("nan"),
        "closest_approach_km_max": float(np.max(ca_finite)) if ca_finite.size else float("nan"),
        "closest_approach_km_p05": float(np.percentile(ca_finite, 5)) if ca_finite.size else float("nan"),
        "closest_approach_km_p50": float(np.percentile(ca_finite, 50)) if ca_finite.size else float("nan"),
        "closest_approach_km_p95": float(np.percentile(ca_finite, 95)) if ca_finite.size else float("nan"),
        "total_dv_m_s_mean": float(np.mean(total_dv_finite)) if total_dv_finite.size else float("nan"),
        "total_dv_m_s_p50": float(np.percentile(total_dv_finite, 50)) if total_dv_finite.size else float("nan"),
        "total_dv_m_s_p90": float(np.percentile(total_dv_finite, 90)) if total_dv_finite.size else float("nan"),
        "total_dv_m_s_p95": float(np.percentile(total_dv_finite, 95)) if total_dv_finite.size else float("nan"),
        "total_dv_m_s_p99": float(np.percentile(total_dv_finite, 99)) if total_dv_finite.size else float("nan"),
        "guardrail_events_mean": float(np.mean(guardrail_finite)) if guardrail_finite.size else float("nan"),
        "guardrail_events_p95": float(np.percentile(guardrail_finite, 95)) if guardrail_finite.size else float("nan"),
        "pass_rate": pass_rate,
        "fail_rate": 1.0 - pass_rate,
        "guardrail_violation_rate": float(np.mean(guardrail_violation_flags))
        if guardrail_violation_flags.size
        else float("nan"),
        "failure_mode_counts": failure_mode_counts,
        "termination_reason_counts": termination_reason_counts,
        "by_object": by_object_stats,
        "delta_v_budget_m_s_by_object": dict(dv_budget_m_s_by_object),
        "delta_v_remaining_m_s_by_object": {
            oid: quantile_stats(vals, (50.0, 90.0, 99.0)) for oid, vals in sorted(dv_remaining_m_s_by_object.items())
        },
        "knowledge_detection_by_observer": aggregate_knowledge_detection_from_runs(run_details),
        "knowledge_consistency_by_observer": aggregate_knowledge_consistency_from_runs(run_details),
        "actuator_diagnostics_by_object": {
            oid: {
                "actuator_stack_samples_mean": float(np.mean(actuator_samples_by_object.get(oid, [0.0]))),
                "rcs_active_samples_mean": float(np.mean(rcs_samples_by_object.get(oid, [0.0]))),
                "magnetorquer_active_samples_mean": float(np.mean(magnetorquer_samples_by_object.get(oid, [0.0]))),
                "cmg_active_samples_mean": float(np.mean(cmg_samples_by_object.get(oid, [0.0]))),
                "wheel_desaturation_samples_mean": float(np.mean(wheel_desat_samples_by_object.get(oid, [0.0]))),
                "max_gimbal_angle_rad_max": float(np.max(max_gimbal_rad_by_object.get(oid, [0.0]))),
                "max_commanded_rcs_thruster_force_n_max": float(np.max(max_rcs_force_n_by_object.get(oid, [0.0]))),
                "max_electric_propulsion_thrust_n_max": float(np.max(max_ep_thrust_n_by_object.get(oid, [0.0]))),
            }
            for oid in sorted(
                set(actuator_samples_by_object)
                | set(rcs_samples_by_object)
                | set(magnetorquer_samples_by_object)
                | set(cmg_samples_by_object)
                | set(wheel_desat_samples_by_object)
                | set(max_gimbal_rad_by_object)
                | set(max_rcs_force_n_by_object)
                | set(max_ep_thrust_n_by_object)
            )
        },
    }
    orbital_delivery = aggregate_orbital_delivery_runs(runs)
    if orbital_delivery:
        aggregate_stats["orbital_delivery_rpo_feasibility_rate"] = float(
            orbital_delivery["rpo_feasibility_rate"]
        )

    keepout_threshold = safe_float(legacy_gates.get("min_closest_approach_km"))
    if not np.isfinite(keepout_threshold):
        keepout_threshold = safe_float(mc_out_cfg.get("keepout_radius_km"))
    p_keepout_violation = float("nan")
    if np.isfinite(keepout_threshold) and ca_finite.size:
        p_keepout_violation = float(np.mean(ca_finite < keepout_threshold))

    catastrophic_failure_reasons = [
        str(x) for x in (mc_out_cfg.get("catastrophic_failure_reasons", ["terminated_early:earth_impact"]) or [])
    ]
    catastrophic_count = 0
    for detail in run_details:
        reasons = set(str(x) for x in list(detail.get("fail_reasons", []) or []))
        if any(reason in reasons for reason in catastrophic_failure_reasons):
            catastrophic_count += 1
    p_catastrophic_outcome = float(catastrophic_count / max(len(run_details), 1))

    max_duration_gate = safe_float(legacy_gates.get("max_duration_s"))
    p_exceed_time_budget = float("nan")
    if np.isfinite(max_duration_gate) and duration_finite.size:
        p_exceed_time_budget = float(np.mean(duration_finite > max_duration_gate))

    max_total_dv_gate = safe_float(legacy_gates.get("max_total_dv_m_s"))
    p_exceed_dv_budget = float("nan")
    if np.isfinite(max_total_dv_gate) and total_dv_finite.size:
        p_exceed_dv_budget = float(np.mean(total_dv_finite > max_total_dv_gate))

    if np.isfinite(max_total_dv_gate) and total_dv_finite.size:
        fuel_margin_stats = quantile_stats(max_total_dv_gate - total_dv_finite, (5.0, 50.0, 95.0))
    else:
        fuel_margin_stats = quantile_stats([], (5.0, 50.0, 95.0))
    if np.isfinite(max_duration_gate) and duration_finite.size:
        time_margin_stats = quantile_stats(max_duration_gate - duration_finite, (5.0, 50.0, 95.0))
    else:
        time_margin_stats = quantile_stats([], (5.0, 50.0, 95.0))
    aggregate_stats["p_keepout_violation"] = p_keepout_violation
    aggregate_stats["p_catastrophic_outcome"] = p_catastrophic_outcome
    aggregate_stats["p_exceed_dv_budget"] = p_exceed_dv_budget
    aggregate_stats["p_exceed_time_budget"] = p_exceed_time_budget
    metric_gate_summary: dict[str, dict[str, Any]] = {}
    metric_gate_names = sorted(
        {
            str(result.get("name"))
            for detail in run_details
            for result in list(detail.get("metric_gate_results", []) or [])
            if str(result.get("name", "") or "")
        }
    )
    for name in metric_gate_names:
        results = [
            dict(result or {})
            for detail in run_details
            for result in list(detail.get("metric_gate_results", []) or [])
            if str(dict(result or {}).get("name")) == name
        ]
        actual_values = [safe_float(result.get("actual")) for result in results]
        metric_gate_summary[name] = {
            "metric": str(results[0].get("metric", "")) if results else "",
            "op": str(results[0].get("op", "")) if results else "",
            "value": results[0].get("value") if results else None,
            "pass_rate": float(np.mean([1.0 if bool(result.get("pass", False)) else 0.0 for result in results]))
            if results
            else float("nan"),
            "actual_stats": quantile_stats(actual_values, (50.0, 90.0, 95.0, 99.0)),
        }
    if metric_gate_summary:
        aggregate_stats["metric_gate_summary"] = metric_gate_summary
    metric_specs = normalize_study_metric_specs(cfg.analysis.metrics)
    study_metric_summary = _study_metric_summary(run_details, metric_specs)
    comparison_summary = _comparison_summary(cfg, run_details, metric_specs)
    if study_metric_summary:
        aggregate_stats["study_metric_summary"] = study_metric_summary

    top_failure_modes: list[dict[str, Any]] = []
    if failure_mode_counts:
        for reason, count in sorted(failure_mode_counts.items(), key=lambda kv: int(kv[1]), reverse=True)[:3]:
            top_failure_modes.append(
                {
                    "reason": str(reason),
                    "count": int(count),
                    "rate": float(int(count) / max(len(run_details), 1)),
                }
            )

    cfg_json = json.dumps(root, sort_keys=True, separators=(",", ":"))
    seed_semantics = {
        "base_seed": int(cfg.monte_carlo.base_seed),
        "metadata_seed_varied": bool(varies_metadata_seed),
        "per_run_seed_rule": (
            "metadata.seed is sampled by the configured monte_carlo variation."
            if varies_metadata_seed
            else (
                "metadata.seed = monte_carlo.base_seed + paired replicate; model options share the seed."
                if cfg.monte_carlo.comparison is not None and cfg.monte_carlo.comparison.paired
                else "metadata.seed = monte_carlo.base_seed + iteration; root metadata.seed is overwritten for each run."
            )
        ),
    }
    reproducibility = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit_sha": get_git_commit_sha(repo_root),
        "config_sha256": hashlib.sha256(cfg_json.encode("utf-8")).hexdigest(),
        "model_profile": infer_model_profile(root),
        "random_seed_policy": seed_semantics["per_run_seed_rule"],
        "seed_semantics": seed_semantics,
    }
    sensitivity_rankings = build_parameter_sensitivity_rankings(run_details)
    top_parameter_drivers = [
        {
            "parameter_path": str(row.get("parameter_path")),
            "importance_score": safe_float(row.get("importance_score"), default=0.0),
            "abs_corr_pass": safe_float(row.get("abs_corr_pass")),
            "abs_corr_closest_approach_km": safe_float(row.get("abs_corr_closest_approach_km")),
            "abs_corr_total_dv_m_s": safe_float(row.get("abs_corr_total_dv_m_s")),
            "abs_corr_study_metrics": dict(row.get("abs_corr_study_metrics", {}) or {}),
        }
        for row in sensitivity_rankings[:5]
    ]
    unique_seeds = len(set(int(safe_float(d.get("seed"), default=-1)) for d in run_details))
    finite_ca_rate = float(np.mean(np.isfinite(ca_arr_full))) if ca_arr_full.size else float("nan")
    analysis_confidence = {
        "runs_executed": int(len(run_details)),
        "run_count_sufficient_for_tail_estimates": bool(len(run_details) >= 100),
        "unique_seed_count": int(unique_seeds),
        "finite_closest_approach_rate": finite_ca_rate,
        "finite_duration_rate": float(np.mean(np.isfinite(duration_arr))) if duration_arr.size else float("nan"),
        "finite_total_dv_rate": float(np.mean(np.isfinite(total_dv_arr))) if total_dv_arr.size else float("nan"),
        "varied_parameter_count": int(len(
            {str(v.parameter_path) for v in cfg.monte_carlo.variations}
            | {path for bundle in cfg.monte_carlo.bundles for path in bundle.options[0].values}
            | {path for group in cfg.monte_carlo.correlated_normals for path in group.parameter_paths}
        )),
        "model_profile": reproducibility.get("model_profile"),
        "git_commit_sha": reproducibility.get("git_commit_sha"),
        "config_sha256": reproducibility.get("config_sha256"),
    }
    if comparison_summary is not None:
        option_counts = [
            int(dict(option or {}).get("run_count", 0) or 0)
            for option in dict(comparison_summary.get("options", {}) or {}).values()
        ]
        analysis_confidence["run_count_sufficient_for_tail_estimates"] = bool(
            option_counts and min(option_counts) >= 100
        )
        analysis_confidence["tail_estimate_scope"] = "minimum_runs_per_comparison_option"
    commander_brief = {
        "scenario_name": cfg.scenario_name,
        "runs": int(cfg.monte_carlo.iterations),
        "p_success": pass_rate,
        "p_fail": 1.0 - pass_rate,
        "p_keepout_violation": p_keepout_violation,
        "p_catastrophic_outcome": p_catastrophic_outcome,
        "p_exceed_dv_budget": p_exceed_dv_budget,
        "p_exceed_time_budget": p_exceed_time_budget,
        "keepout_threshold_km": keepout_threshold if np.isfinite(keepout_threshold) else None,
        "worst_case_closest_approach_km": float(np.min(ca_finite)) if ca_finite.size else float("nan"),
        "timeline_confidence_bands_s": quantile_stats(duration_runs_s, (50.0, 90.0, 99.0)),
        "fuel_confidence_bands_total_dv_m_s": quantile_stats(total_dv_runs_m_s, (50.0, 90.0, 99.0)),
        "fuel_confidence_bands_dv_m_s_by_object": {
            oid: quantile_stats(dv_by_object.get(oid, []), (50.0, 90.0, 99.0)) for oid in sorted(dv_by_object.keys())
        },
        "delta_v_remaining_confidence_bands_m_s_by_object": {
            oid: quantile_stats(vals, (50.0, 90.0, 99.0)) for oid, vals in sorted(dv_remaining_m_s_by_object.items())
        },
        "resource_margin": {
            "fuel_margin_m_s_vs_budget": fuel_margin_stats,
            "time_margin_s_vs_budget": time_margin_stats,
        },
        "constraint_violation_summary": {
            "p_guardrail_violation": float(np.mean(guardrail_violation_flags))
            if guardrail_violation_flags.size
            else float("nan"),
            "guardrail_events_per_run": quantile_stats(guardrail_event_runs, (50.0, 90.0, 99.0)),
            "compute_deadline_overrun_available": False,
            "control_saturation_available": False,
        },
        "top_failure_modes": top_failure_modes,
        "top_parameter_drivers": top_parameter_drivers,
        "analysis_confidence": analysis_confidence,
    }
    if comparison_summary is not None:
        # The root campaign statistics remain useful as a pooled view, but
        # make that scope explicit and carry the per-option rates into the
        # human-facing brief so they cannot be mistaken for a model-specific
        # probability.
        commander_brief["comparison_summary"] = comparison_summary
        commander_brief["comparison_aggregation"] = "pooled_across_options"
    study_brief = {
        **commander_brief,
        "brief_type": "engineering_study",
        "metric_gate_summary": metric_gate_summary,
        "study_metric_summary": study_metric_summary,
        "comparison_summary": comparison_summary,
    }

    analyst_pack = {
        "scenario_name": cfg.scenario_name,
        "reproducibility": reproducibility,
        "gates": gates,
        "study_metrics": {"specs": metric_specs, "summary": study_metric_summary},
        "comparison_summary": comparison_summary,
        "run_details": run_details,
        "failure_mode_counts": failure_mode_counts,
        "sensitivity_rankings": sensitivity_rankings,
        "catastrophic_failure_reasons": catastrophic_failure_reasons,
    }
    if orbital_delivery:
        commander_brief["orbital_delivery"] = orbital_delivery
        study_brief["orbital_delivery"] = orbital_delivery
        analyst_pack["orbital_delivery"] = orbital_delivery

    agg = {
        "config_path": str(Path(config_path).resolve()),
        "scenario_name": cfg.scenario_name,
        "scenario_description": cfg.scenario_description,
        "monte_carlo": {
            "enabled": True,
            "iterations": int(cfg.monte_carlo.iterations),
            "base_seed": int(cfg.monte_carlo.base_seed),
            "seed_semantics": seed_semantics,
            "parallel_enabled": bool(parallel_active),
            "parallel_requested": bool(parallel_enabled and total_iters > 1),
            "parallel_workers": int(parallel_workers if parallel_active else 1),
        },
        "aggregate_stats": aggregate_stats,
        "study_metrics": {"specs": metric_specs, "summary": study_metric_summary},
        "comparison_summary": comparison_summary,
        "commander_brief": commander_brief,
        "study_brief": study_brief,
        "reproducibility": reproducibility,
        "analyst_pack": analyst_pack,
        "artifacts": {},
        "runs": runs,
    }
    if orbital_delivery:
        agg["orbital_delivery"] = orbital_delivery
    if parallel_fallback_reason is not None:
        agg["monte_carlo"]["parallel_fallback_reason"] = str(parallel_fallback_reason)

    return {
        "agg": agg,
        "commander_brief": commander_brief,
        "study_brief": study_brief,
        "analyst_pack": analyst_pack,
        "durations_s": durations_s,
        "ca_finite": ca_finite,
        "all_obj_ids": all_obj_ids,
        "dv_by_object": dv_by_object,
        "dv_remaining_m_s_by_object": dv_remaining_m_s_by_object,
        "run_details": run_details,
        "keepout_threshold": keepout_threshold,
        "failure_mode_counts": failure_mode_counts,
    }


def write_monte_carlo_report_artifacts(
    *,
    cfg: SimulationScenarioConfig,
    outdir: Path,
    agg: dict[str, Any],
    commander_brief: dict[str, Any],
    analyst_pack: dict[str, Any],
    run_details: list[dict[str, Any]],
    study_brief: dict[str, Any] | None = None,
    mc_out_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from sim.execution.monte_carlo_support import write_commander_brief_markdown

    mc_outputs = dict(mc_out_cfg or cfg.outputs.monte_carlo or {})
    if not bool(mc_outputs.get("save_aggregate_summary", True)):
        return agg

    summary_path = outdir / "master_monte_carlo_summary.json"
    commander_json_path = outdir / "master_monte_carlo_commander_brief.json"
    commander_md_path = outdir / "master_monte_carlo_commander_brief.md"
    study_brief_json_path = outdir / "master_monte_carlo_study_brief.json"
    study_brief_md_path = outdir / "master_monte_carlo_study_brief.md"
    analyst_path = outdir / "master_monte_carlo_analyst_pack.json"
    study_payload = dict(study_brief or agg.get("study_brief", {}) or commander_brief)

    write_json(str(commander_json_path), commander_brief)
    write_commander_brief_markdown(commander_md_path, commander_brief)
    write_json(str(study_brief_json_path), study_payload)
    write_commander_brief_markdown(study_brief_md_path, study_payload, title="Monte Carlo Study Brief")
    write_json(str(analyst_path), analyst_pack)

    artifacts = dict(agg.get("artifacts", {}) or {})
    artifacts["summary_json"] = str(summary_path)
    artifacts["commander_brief_json"] = str(commander_json_path)
    artifacts["commander_brief_md"] = str(commander_md_path)
    artifacts["study_brief_json"] = str(study_brief_json_path)
    artifacts["study_brief_md"] = str(study_brief_md_path)
    artifacts["analyst_pack_json"] = str(analyst_path)

    metric_specs = normalize_study_metric_specs(cfg.analysis.metrics)
    if metric_specs:
        metrics_path = outdir / "master_monte_carlo_study_metrics.csv"
        names = list(dict.fromkeys(str(spec["name"]) for spec in metric_specs))
        comparison_columns = ["comparison_bundle", "comparison_option", "replicate", "paired"] if cfg.monte_carlo.comparison else []
        with metrics_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["iteration", "pass", "sampled_parameters", *comparison_columns, *names])
            writer.writeheader()
            for detail in run_details:
                row: dict[str, Any] = {
                    "iteration": detail.get("iteration"),
                    "pass": detail.get("pass"),
                    "sampled_parameters": json.dumps(dict(detail.get("sampled_parameters", {}) or {}), sort_keys=True),
                }
                if comparison_columns:
                    comparison = dict(detail.get("comparison", {}) or {})
                    row.update({
                        "comparison_bundle": comparison.get("bundle"),
                        "comparison_option": comparison.get("option"),
                        "replicate": comparison.get("replicate"),
                        "paired": comparison.get("paired"),
                    })
                for name in names:
                    value = dict(detail.get("metrics", {}) or {}).get(name)
                    if isinstance(value, (float, np.floating)) and not np.isfinite(float(value)):
                        value = None
                    row[name] = (
                        json.dumps(value, sort_keys=True) if isinstance(value, (dict, list, tuple)) else value
                    )
                writer.writerow(row)
        artifacts["study_metrics_csv"] = str(metrics_path)

    if bool(mc_outputs.get("save_raw_runs", False)):
        runs_path = outdir / "master_monte_carlo_run_details.json"
        write_json(str(runs_path), {"scenario_name": cfg.scenario_name, "run_details": run_details})
        artifacts["run_details_json"] = str(runs_path)

    agg["artifacts"] = artifacts
    write_json(str(summary_path), agg)
    return agg


def apply_monte_carlo_baseline_comparison(
    *,
    agg: dict[str, Any],
    commander_brief: dict[str, Any],
    study_brief: dict[str, Any] | None = None,
    config_path: str | Path,
    baseline_summary_json: str,
) -> dict[str, Any]:
    from sim.execution.monte_carlo_support import build_baseline_comparison, load_json_file

    baseline_path_text = str(baseline_summary_json or "").strip()
    if not baseline_path_text:
        return agg
    baseline_path = Path(baseline_path_text)
    if not baseline_path.is_absolute():
        baseline_path = Path(config_path).resolve().parent / baseline_path
    baseline_payload = load_json_file(baseline_path)
    if baseline_payload is not None:
        comparison = build_baseline_comparison(agg, baseline_payload)
        agg["baseline_comparison"] = comparison
        commander_brief["baseline_comparison"] = comparison
        if study_brief is not None:
            study_brief["baseline_comparison"] = comparison
    else:
        agg["baseline_comparison_error"] = f"Unable to load baseline summary: {str(baseline_path)}"
    return agg
