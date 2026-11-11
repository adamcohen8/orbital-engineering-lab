from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from sim.config import SimulationScenarioConfig, iter_object_sections, object_parameter_prefix
from sim.execution.metrics import evaluate_study_metric_gates
from sim.presets.thrusters import BASIC_CHEMICAL_BOTTOM_Z


def safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(out):
        return default
    return out


def fmt_float(x: float, digits: int = 3) -> str:
    return f"{float(x):.{digits}f}"


def quantile_stats(
    values: list[float] | np.ndarray, quantiles: tuple[float, ...] = (50.0, 90.0, 95.0, 99.0)
) -> dict[str, float]:
    arr = np.array(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        out = {
            "mean": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }
        for q in quantiles:
            out[f"p{int(q)}"] = float("nan")
        return out
    out = {
        "mean": float(np.mean(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }
    for q in quantiles:
        out[f"p{int(q)}"] = float(np.percentile(arr, q))
    return out


def coerce_numeric_map(value: Any) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}
    out: dict[str, float] = {}
    for k, v in value.items():
        fv = safe_float(v)
        if np.isfinite(fv):
            out[str(k)] = fv
    return out


def get_git_commit_sha(repo_root: Path) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return None
    return out or None


def infer_model_profile(root_cfg: dict[str, Any]) -> str:
    metadata = dict(root_cfg.get("metadata", {}) or {})
    simulator = dict(root_cfg.get("simulator", {}) or {})
    dynamics = dict(simulator.get("dynamics", {}) or {})
    environment = dict(simulator.get("environment", {}) or {})
    for src in (metadata, simulator, dynamics, environment):
        for key in ("profile", "profile_name", "fidelity_profile"):
            val = src.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
    return "custom"


def resolve_satellite_isp_s(specs: dict[str, Any]) -> float:
    if "isp_s" in specs:
        return float(specs.get("isp_s", 0.0))
    if "thruster_isp_s" in specs:
        return float(specs.get("thruster_isp_s", 0.0))
    thr = str(specs.get("thruster", "")).strip().upper()
    if thr in ("BASIC_CHEMICAL_BOTTOM_Z", "BASIC_CHEMICAL_Z_BOTTOM"):
        return float(BASIC_CHEMICAL_BOTTOM_Z.isp_s)
    return 0.0


def satellite_initial_delta_v_budget_m_s(agent_cfg: Any) -> float:
    specs = dict(getattr(agent_cfg, "specs", {}) or {})
    dry_mass_kg = safe_float(specs.get("dry_mass_kg"))
    fuel_mass_kg = safe_float(specs.get("fuel_mass_kg"))
    if not (np.isfinite(dry_mass_kg) and np.isfinite(fuel_mass_kg)):
        return float("nan")
    if dry_mass_kg <= 0.0 or fuel_mass_kg < 0.0:
        return float("nan")
    m0_kg = dry_mass_kg + fuel_mass_kg
    if m0_kg <= dry_mass_kg:
        return 0.0
    isp_s = resolve_satellite_isp_s(specs)
    if isp_s <= 0.0:
        return float("nan")
    return float(isp_s * 9.80665 * np.log(m0_kg / dry_mass_kg))


def assess_mc_run(
    *,
    run_entry: dict[str, Any],
    gates: Any,
    success_termination_reasons: set[str],
    require_rocket_insertion: bool,
) -> dict[str, Any]:
    legacy_gates = dict(gates if isinstance(gates, dict) else {})
    if isinstance(gates, list):
        metric_gates = list(gates)
    else:
        metric_gates = []
        for key in ("metric_gates", "metrics", "generic"):
            raw = legacy_gates.get(key)
            if isinstance(raw, list):
                metric_gates.extend(raw)
    summary = dict(run_entry.get("summary", {}) or {})
    term_reason = summary.get("termination_reason")
    term_reason_txt = str(term_reason) if term_reason is not None else "none"
    terminated_early = bool(summary.get("terminated_early", False))
    closest_approach_km = safe_float(run_entry.get("closest_approach_km"))
    duration_s = safe_float(summary.get("duration_s"))
    guardrail_raw = summary.get("attitude_guardrail_stats")
    guardrail_map = dict(guardrail_raw or {}) if isinstance(guardrail_raw, dict) else {}
    guardrail_events = int(sum(int(v) for v in guardrail_map.values())) if guardrail_map else 0
    thrust_stats_raw = summary.get("thrust_stats")
    thrust_stats = dict(thrust_stats_raw or {}) if isinstance(thrust_stats_raw, dict) else {}
    total_dv_m_s_by_object = {
        str(oid): safe_float(dict(ts or {}).get("total_dv_m_s")) for oid, ts in thrust_stats.items()
    }
    total_dv_m_s_total = (
        float(np.sum(np.array(list(total_dv_m_s_by_object.values()), dtype=float)))
        if total_dv_m_s_by_object and all(np.isfinite(value) for value in total_dv_m_s_by_object.values())
        else float("nan")
    )

    fail_reasons: list[str] = []
    if not np.isfinite(duration_s):
        fail_reasons.append("missing:duration_s")
    if not np.isfinite(total_dv_m_s_total):
        fail_reasons.append("missing:total_dv_m_s")
    if terminated_early and term_reason_txt not in success_termination_reasons:
        fail_reasons.append(f"terminated_early:{term_reason_txt}")
    if require_rocket_insertion and (not bool(summary.get("rocket_insertion_achieved", False))):
        fail_reasons.append("rocket_insertion_not_achieved")

    min_closest_approach_km = safe_float(legacy_gates.get("min_closest_approach_km"))
    if np.isfinite(min_closest_approach_km):
        if not np.isfinite(closest_approach_km) or closest_approach_km < min_closest_approach_km:
            fail_reasons.append("gate:min_closest_approach_km")

    max_duration_s = safe_float(legacy_gates.get("max_duration_s"))
    if np.isfinite(max_duration_s):
        if not np.isfinite(duration_s) or duration_s > max_duration_s:
            fail_reasons.append("gate:max_duration_s")

    max_guardrail_events = safe_float(legacy_gates.get("max_guardrail_events"))
    if np.isfinite(max_guardrail_events):
        if not isinstance(guardrail_raw, dict) or float(guardrail_events) > max_guardrail_events:
            fail_reasons.append("gate:max_guardrail_events")

    max_total_dv_m_s = safe_float(legacy_gates.get("max_total_dv_m_s"))
    if np.isfinite(max_total_dv_m_s):
        if not np.isfinite(total_dv_m_s_total) or total_dv_m_s_total > max_total_dv_m_s:
            fail_reasons.append("gate:max_total_dv_m_s")

    max_dv_by_object = coerce_numeric_map(legacy_gates.get("max_total_dv_m_s_by_object"))
    for oid, dv_limit in max_dv_by_object.items():
        dv = safe_float(total_dv_m_s_by_object.get(oid))
        if not np.isfinite(dv) or dv > dv_limit:
            fail_reasons.append(f"gate:max_total_dv_m_s_by_object:{oid}")

    metric_gate_results = evaluate_study_metric_gates(run_entry, metric_gates)
    for result in metric_gate_results:
        if not bool(result.get("pass", False)):
            fail_reasons.append(f"gate:{result.get('name') or result.get('metric')}")

    return {
        "pass": len(fail_reasons) == 0,
        "fail_reasons": sorted(set(fail_reasons)),
        "duration_s": duration_s,
        "closest_approach_km": closest_approach_km,
        "guardrail_events": guardrail_events,
        "termination_reason": term_reason_txt,
        "terminated_early": terminated_early,
        "rocket_insertion_achieved": bool(summary.get("rocket_insertion_achieved", False)),
        "total_dv_m_s_total": total_dv_m_s_total,
        "total_dv_m_s_by_object": total_dv_m_s_by_object,
        "metric_gate_results": metric_gate_results,
        "gate_metrics": {str(item.get("name")): item.get("actual") for item in metric_gate_results},
    }


def build_parameter_sensitivity_rankings(run_details: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not run_details:
        return []
    all_paths: set[str] = set()
    for d in run_details:
        for path in dict(d.get("sampled_parameters", {}) or {}).keys():
            all_paths.add(str(path))
    rankings: list[dict[str, Any]] = []
    pass_arr = np.array([1.0 if bool(d.get("pass", False)) else 0.0 for d in run_details], dtype=float)
    ca_arr = np.array([safe_float(d.get("closest_approach_km")) for d in run_details], dtype=float)
    dv_arr = np.array([safe_float(d.get("total_dv_m_s_total"), default=0.0) for d in run_details], dtype=float)
    gate_metric_names = sorted(
        {
            str(name)
            for detail in run_details
            for name in dict(detail.get("gate_metrics", {}) or {}).keys()
            if str(name)
        }
    )
    gate_metric_arrays = {
        name: np.array([safe_float(dict(d.get("gate_metrics", {}) or {}).get(name)) for d in run_details], dtype=float)
        for name in gate_metric_names
    }
    study_metric_names = sorted(
        {
            str(name)
            for detail in run_details
            for name in dict(detail.get("metrics", {}) or {}).keys()
            if str(name)
        }
    )
    study_metric_arrays = {
        name: np.array([safe_float(dict(d.get("metrics", {}) or {}).get(name)) for d in run_details], dtype=float)
        for name in study_metric_names
    }

    for path in sorted(all_paths):
        vals: list[float] = []
        ok: list[bool] = []
        for d in run_details:
            sv = dict(d.get("sampled_parameters", {}) or {}).get(path)
            if isinstance(sv, bool):
                vals.append(1.0 if sv else 0.0)
                ok.append(True)
            elif isinstance(sv, (int, float, np.integer, np.floating)):
                vals.append(float(sv))
                ok.append(np.isfinite(float(sv)))
            else:
                vals.append(float("nan"))
                ok.append(False)
        x = np.array(vals, dtype=float)
        finite_x = np.isfinite(x)
        if int(np.sum(finite_x)) < 3:
            continue

        def _abs_corr(y: np.ndarray, *, x_values: np.ndarray = x, finite_x_mask: np.ndarray = finite_x) -> float:
            finite = finite_x_mask & np.isfinite(y)
            if int(np.sum(finite)) < 3:
                return float("nan")
            x_ok = x_values[finite]
            y_ok = y[finite]
            if np.allclose(np.std(x_ok), 0.0) or np.allclose(np.std(y_ok), 0.0):
                return float("nan")
            return float(abs(np.corrcoef(x_ok, y_ok)[0, 1]))

        corr_pass = _abs_corr(pass_arr)
        corr_ca = _abs_corr(ca_arr)
        corr_dv = _abs_corr(dv_arr)
        metric_corrs = {name: _abs_corr(arr) for name, arr in gate_metric_arrays.items()}
        study_metric_corrs = {name: _abs_corr(arr) for name, arr in study_metric_arrays.items()}
        finite_corrs = np.array(
            [corr_pass, corr_ca, corr_dv, *metric_corrs.values(), *study_metric_corrs.values()], dtype=float
        )
        finite_corrs = finite_corrs[np.isfinite(finite_corrs)]
        importance = float(np.max(finite_corrs)) if finite_corrs.size else float("nan")
        if not np.isfinite(importance):
            continue
        rankings.append(
            {
                "parameter_path": path,
                "samples": int(np.sum(finite_x)),
                "abs_corr_pass": corr_pass,
                "abs_corr_closest_approach_km": corr_ca,
                "abs_corr_total_dv_m_s": corr_dv,
                "abs_corr_gate_metrics": metric_corrs,
                "abs_corr_study_metrics": study_metric_corrs,
                "importance_score": importance,
            }
        )
    rankings.sort(key=lambda x: float(x.get("importance_score", 0.0)), reverse=True)
    return rankings


def load_json_file(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    return raw if isinstance(raw, dict) else None


def extract_baseline_metrics(payload: dict[str, Any]) -> dict[str, float]:
    commander = dict(payload.get("commander_brief", {}) or {})
    aggregate = dict(payload.get("aggregate_stats", {}) or {})
    p_success = safe_float(commander.get("p_success"))
    p_fail = safe_float(commander.get("p_fail"))
    duration_p95 = safe_float(dict(commander.get("timeline_confidence_bands_s", {}) or {}).get("p95"))
    dv_total_p95 = safe_float(dict(commander.get("fuel_confidence_bands_total_dv_m_s", {}) or {}).get("p95"))
    min_closest = safe_float(aggregate.get("closest_approach_km_min"))
    return {
        "p_success": p_success,
        "p_fail": p_fail,
        "duration_s_p95": duration_p95,
        "total_dv_m_s_p95": dv_total_p95,
        "closest_approach_km_min": min_closest,
    }


def aggregate_knowledge_consistency_from_runs(run_details: list[dict[str, Any]]) -> dict[str, Any]:
    buckets: dict[tuple[str, str, str], list[float]] = {}
    for detail in run_details:
        summary = dict(detail.get("summary", {}) or {})
        by_observer = dict(summary.get("knowledge_consistency_by_observer", {}) or {})
        for observer_id, by_target in by_observer.items():
            for target_id, metrics in dict(by_target or {}).items():
                for metric_name, value in dict(metrics or {}).items():
                    try:
                        v = float(value)
                    except (TypeError, ValueError):
                        continue
                    if np.isfinite(v):
                        buckets.setdefault((str(observer_id), str(target_id), str(metric_name)), []).append(v)
    out: dict[str, dict[str, dict[str, float]]] = {}
    for (observer_id, target_id, metric_name), values in sorted(buckets.items()):
        obs_map = out.setdefault(observer_id, {})
        tgt_map = obs_map.setdefault(target_id, {})
        arr = np.array(values, dtype=float)
        tgt_map[metric_name] = float(np.mean(arr)) if arr.size else float("nan")
    return out


def aggregate_knowledge_detection_from_runs(run_details: list[dict[str, Any]]) -> dict[str, Any]:
    buckets: dict[tuple[str, str, str], list[float]] = {}
    status_counts: dict[tuple[str, str, str], int] = {}
    for detail in run_details:
        summary = dict(detail.get("summary", {}) or {})
        by_observer = dict(summary.get("knowledge_detection_by_observer", {}) or {})
        for observer_id, by_target in by_observer.items():
            for target_id, metrics in dict(by_target or {}).items():
                for metric_name, value in dict(metrics or {}).items():
                    if metric_name == "status_counts" and isinstance(value, dict):
                        for status, count in value.items():
                            key = (str(observer_id), str(target_id), str(status))
                            status_counts[key] = int(status_counts.get(key, 0)) + int(count)
                        continue
                    try:
                        v = float(value)
                    except (TypeError, ValueError):
                        continue
                    if np.isfinite(v):
                        buckets.setdefault((str(observer_id), str(target_id), str(metric_name)), []).append(v)
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for (observer_id, target_id, metric_name), values in sorted(buckets.items()):
        obs_map = out.setdefault(observer_id, {})
        tgt_map = obs_map.setdefault(target_id, {})
        arr = np.array(values, dtype=float)
        tgt_map[metric_name] = float(np.mean(arr)) if arr.size else float("nan")
    for (observer_id, target_id, status), count in sorted(status_counts.items()):
        obs_map = out.setdefault(observer_id, {})
        tgt_map = obs_map.setdefault(target_id, {})
        tgt_map.setdefault("status_counts", {})[status] = int(count)
    return out


def build_baseline_comparison(current_payload: dict[str, Any], baseline_payload: dict[str, Any]) -> dict[str, Any]:
    cur = extract_baseline_metrics(current_payload)
    base = extract_baseline_metrics(baseline_payload)
    deltas: dict[str, float] = {}
    for k in sorted(set(cur.keys()) | set(base.keys())):
        cv = safe_float(cur.get(k))
        bv = safe_float(base.get(k))
        if np.isfinite(cv) and np.isfinite(bv):
            deltas[k] = float(cv - bv)
    return {
        "baseline_metrics": base,
        "current_metrics": cur,
        "delta_current_minus_baseline": deltas,
    }


def write_commander_brief_markdown(
    path: Path,
    brief: dict[str, Any],
    *,
    title: str = "Monte Carlo Commander Brief",
) -> None:
    top_fail = list(brief.get("top_failure_modes", []) or [])
    neutral_study = "study" in str(title).lower()
    comparison = dict(brief.get("comparison_summary", {}) or {})
    success_label = "P(success, pooled across comparison options)" if comparison else "P(success)"
    lines = [
        f"# {title}",
        "",
        f"- Scenario: {brief.get('scenario_name', 'unknown')}",
        f"- Runs: {int(brief.get('runs', 0))}",
        f"- {success_label}: {100.0 * safe_float(brief.get('p_success'), default=0.0):.1f}%",
        f"- P(fail): {100.0 * safe_float(brief.get('p_fail'), default=0.0):.1f}%",
    ]
    comparison = dict(brief.get("comparison_summary", {}) or {})
    if comparison:
        lines.append("- Probability scope: pooled across comparison options")
        options = dict(comparison.get("options", {}) or {})
        if options:
            lines.append(
                "- P(success) by option: "
                + ", ".join(
                    f"{label}={100.0 * safe_float(dict(option or {}).get('pass_rate'), default=0.0):.1f}%"
                    for label, option in options.items()
                )
            )
    if not neutral_study:
        lines.extend(
            [
                f"- P(keepout violation): {100.0 * safe_float(brief.get('p_keepout_violation'), default=0.0):.1f}%",
                f"- Worst-case closest approach (km): "
                f"{fmt_float(safe_float(brief.get('worst_case_closest_approach_km'), default=0.0), 3)}",
            ]
        )
    lines.extend(["", "## Confidence Bands"])
    timeline = dict(brief.get("timeline_confidence_bands_s", {}) or {})
    fuel = dict(brief.get("fuel_confidence_bands_total_dv_m_s", {}) or {})
    lines.extend(
        [
            f"- Timeline (s): P50={fmt_float(safe_float(timeline.get('p50'), default=0.0), 1)}, "
            f"P90={fmt_float(safe_float(timeline.get('p90'), default=0.0), 1)}, "
            f"P99={fmt_float(safe_float(timeline.get('p99'), default=0.0), 1)}",
            f"- Total dV (m/s): P50={fmt_float(safe_float(fuel.get('p50'), default=0.0), 2)}, "
            f"P90={fmt_float(safe_float(fuel.get('p90'), default=0.0), 2)}, "
            f"P99={fmt_float(safe_float(fuel.get('p99'), default=0.0), 2)}",
            "",
            "## Risk Metrics",
        ]
    )
    lines.append(f"- P(catastrophic outcome): {100.0 * safe_float(brief.get('p_catastrophic_outcome'), default=0.0):.1f}%")
    lines.append(f"- P(exceed dV budget): {100.0 * safe_float(brief.get('p_exceed_dv_budget'), default=0.0):.1f}%")
    lines.append(f"- P(exceed time budget): {100.0 * safe_float(brief.get('p_exceed_time_budget'), default=0.0):.1f}%")
    lines.extend(["", "## Top Failure Modes"])
    if top_fail:
        for row in top_fail:
            reason = str(row.get("reason", "unknown"))
            count = int(row.get("count", 0))
            frac = 100.0 * safe_float(row.get("rate"), default=0.0)
            lines.append(f"- {reason}: {count} runs ({frac:.1f}%)")
    else:
        lines.append("- none")
    metric_gates = dict(brief.get("metric_gate_summary", {}) or {})
    if metric_gates:
        lines.extend(["", "## Metric Gates"])
        for name, raw in sorted(metric_gates.items()):
            row = dict(raw or {})
            pass_rate = 100.0 * safe_float(row.get("pass_rate"), default=0.0)
            stats = dict(row.get("actual_stats", {}) or {})
            lines.append(
                f"- {name}: pass={pass_rate:.1f}%, "
                f"P95={fmt_float(safe_float(stats.get('p95'), default=0.0), 3)}"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def mc_initial_relative_ric_curv_samples(
    cfg: SimulationScenarioConfig,
    run_details: list[dict[str, Any]],
) -> dict[str, np.ndarray]:
    object_id = ""
    rel_block: dict[str, Any] = {}
    for candidate_id, agent_cfg in iter_object_sections(cfg):
        candidate_block = dict((getattr(agent_cfg, "initial_state", {}) or {}).get("relative_to_target_ric", {}) or {})
        if candidate_block:
            object_id = str(candidate_id)
            rel_block = candidate_block
            break
    frame = str(rel_block.get("frame", "rect")).strip().lower()
    base_state = np.array(rel_block.get("state", []), dtype=float).reshape(-1)
    if frame != "curv" or base_state.size != 6 or not run_details:
        return {}
    path_prefixes = [object_parameter_prefix(object_id)]
    canonical_prefix = f"objects.{object_id}"
    if canonical_prefix not in path_prefixes:
        path_prefixes.append(canonical_prefix)
    legacy_prefix = str(object_id)
    if legacy_prefix not in path_prefixes:
        path_prefixes.append(legacy_prefix)

    paths = {
        "radial_sep_km": [f"{prefix}.initial_state.relative_to_target_ric.state[0]" for prefix in path_prefixes],
        "in_track_sep_km": [f"{prefix}.initial_state.relative_to_target_ric.state[1]" for prefix in path_prefixes],
        "cross_track_sep_km": [f"{prefix}.initial_state.relative_to_target_ric.state[2]" for prefix in path_prefixes],
        "radial_vel_km_s": [f"{prefix}.initial_state.relative_to_target_ric.state[3]" for prefix in path_prefixes],
        "in_track_vel_km_s": [f"{prefix}.initial_state.relative_to_target_ric.state[4]" for prefix in path_prefixes],
        "cross_track_vel_km_s": [f"{prefix}.initial_state.relative_to_target_ric.state[5]" for prefix in path_prefixes],
    }
    index_by_name = {
        "radial_sep_km": 0,
        "in_track_sep_km": 1,
        "cross_track_sep_km": 2,
        "radial_vel_km_s": 3,
        "in_track_vel_km_s": 4,
        "cross_track_vel_km_s": 5,
    }
    out: dict[str, np.ndarray] = {}
    for name, path_candidates in paths.items():
        idx = index_by_name[name]
        vals: list[float] = []
        for rd in run_details:
            sampled = dict(rd.get("sampled_parameters", {}) or {})
            value = None
            for path in path_candidates:
                if path in sampled:
                    value = sampled.get(path)
                    break
            vals.append(float(safe_float(value, default=float(base_state[idx]))))
        out[name] = np.array(vals, dtype=float)
    return out


_safe_float = safe_float
_quantile_stats = quantile_stats
_coerce_numeric_map = coerce_numeric_map
_get_git_commit_sha = get_git_commit_sha
_infer_model_profile = infer_model_profile
_assess_mc_run = assess_mc_run
_build_parameter_sensitivity_rankings = build_parameter_sensitivity_rankings
_load_json_file = load_json_file
_aggregate_knowledge_consistency_from_runs = aggregate_knowledge_consistency_from_runs
_aggregate_knowledge_detection_from_runs = aggregate_knowledge_detection_from_runs
_build_baseline_comparison = build_baseline_comparison
_write_commander_brief_markdown = write_commander_brief_markdown
_mc_initial_relative_ric_curv_samples = mc_initial_relative_ric_curv_samples
_satellite_initial_delta_v_budget_m_s = satellite_initial_delta_v_budget_m_s
