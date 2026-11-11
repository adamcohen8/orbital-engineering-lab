# ruff: noqa: F401,F403,F405,I001
from .bench_common import *
from .config import *
from .criteria import *

def _failure_mode_for_metric(metric_name: str, reason: str, objective_kind: str = "") -> tuple[str, str, str]:
    metric = str(metric_name or "").strip()
    metric_l = metric.lower()
    reason_l = str(reason or "").strip().lower()
    objective_kind_l = str(objective_kind or "").strip().lower()

    if reason_l in {"missing_metric", "invalid_metric", "non_numeric_metric"}:
        if metric_l.endswith("settle_time_s"):
            return (
                "attitude_not_settled",
                "The attitude objective never reached the requested error/rate settle band for the required hold time.",
                "Relax the settle band only if it reflects the mission need; otherwise increase control authority, retune gains, or lengthen the maneuver window.",
            )
        if metric_l.endswith("insertion_time_s"):
            return (
                "rocket_insertion_missed",
                "The rocket did not produce an insertion timestamp, usually because insertion was not achieved.",
                "Inspect final apogee, perigee, eccentricity, throttle limiting, and pitch-program timing before widening the insertion tolerance.",
            )
        if objective_kind_l == "rocket_orbit_insertion" or metric_l.startswith("rocket_"):
            return (
                "rocket_missing_gnc_evidence",
                "The rocket GNC objective could not find the expected insertion, trajectory, or control-limit evidence.",
                "Run a rocket_ascent case with rocket metrics enabled and confirm the rocket object id and objective fields match the scenario.",
            )
        return (
            "missing_or_invalid_evidence",
            "The benchmark could not produce a valid numeric value for this criterion.",
            "Check that the metric kind, object id, reference object id, and saved payload histories match the suite definition.",
        )
    if metric_l.endswith("terminated_early"):
        return (
            "early_termination",
            "The scenario ended before the benchmark completed normally.",
            "Inspect the termination reason and case geometry before changing controller gains.",
        )
    if metric_l.endswith("insertion_achieved"):
        return (
            "rocket_insertion_missed",
            "The rocket did not satisfy the configured orbit-insertion condition.",
            "Inspect final apogee, perigee, eccentricity, throttle limiting, and pitch-program timing before widening the insertion tolerance.",
        )
    if metric_l.endswith("insertion_time_s"):
        return (
            "rocket_slow_insertion",
            "The rocket reached insertion later than the objective allows.",
            "Tune pitch-over timing and throttle limits, then check whether staging or payload mass is driving the delay.",
        )
    if metric_l.endswith("final_apoapsis_alt_km") or metric_l.endswith("final_periapsis_alt_km"):
        return (
            "rocket_orbit_altitude_error",
            "The final orbit altitude is outside the allowed insertion band.",
            "Adjust pitch-program duration and terminal pitch target, then compare vertical and horizontal speed histories near cutoff.",
        )
    if metric_l.endswith("final_eccentricity"):
        return (
            "rocket_orbit_shape_error",
            "The insertion orbit is more eccentric than the objective allows.",
            "Bias guidance toward circularization near cutoff and inspect whether the burn ends before horizontal speed converges.",
        )
    if metric_l.endswith("final_propellant_remaining_fraction"):
        return (
            "rocket_propellant_margin_low",
            "The rocket meets too little propellant margin at the end of the case.",
            "Trade payload mass, throttle limiting, pitch schedule, or target orbit before tightening terminal orbit tolerances.",
        )
    if metric_l.endswith("max_dynamic_pressure_pa"):
        return (
            "rocket_max_q_high",
            "The ascent exceeds the configured dynamic-pressure limit.",
            "Retune the max-Q throttle limiter or pitch profile so aero loads stay inside the design envelope.",
        )
    if metric_l.endswith("max_abs_alpha_deg"):
        return (
            "rocket_alpha_high",
            "The ascent exceeds the configured angle-of-attack limit.",
            "Reduce pitch-program aggressiveness or improve TVC/attitude tracking before accepting the trajectory.",
        )
    if metric_l.endswith("max_tvc_gimbal_deg"):
        return (
            "rocket_tvc_authority_limited",
            "The commanded TVC gimbal approaches or exceeds the configured authority limit.",
            "Reduce guidance demand, retune TVC dynamics, or revisit the actuator authority assumption.",
        )
    if metric_l.endswith("max_aero_force_n"):
        return (
            "rocket_aero_load_high",
            "The ascent produces higher aerodynamic force than the objective allows.",
            "Review max-Q, angle-of-attack, and reference geometry before increasing the load limit.",
        )
    if metric_l.endswith("settled_hold_duration_s"):
        return (
            "attitude_not_settled",
            "The controller entered the settle band for too little continuous time.",
            "Increase damping or authority, reduce residual body rate, or make the settle requirement match the available maneuver duration.",
        )
    if metric_l.endswith("settle_time_s"):
        return (
            "attitude_slow_settle",
            "The controller settled later than the objective allows.",
            "Tune proportional/damping gains together and check torque saturation before simply raising gains.",
        )
    if metric_l.endswith("max_attitude_error_deg"):
        return (
            "attitude_peak_error",
            "The attitude transient exceeded the allowed peak pointing error.",
            "Reduce overshoot with more damping, lower aggressive proportional gains, or add command shaping for large slews.",
        )
    if metric_l.endswith("rms_attitude_error_deg") or metric_l.endswith("final_attitude_error_deg"):
        return (
            "attitude_pointing_error",
            "The final or aggregate pointing error is outside the objective limit.",
            "Check desired quaternion convention, then retune gains or increase maneuver time/control authority.",
        )
    if metric_l.endswith("body_rate_norm_rad_s"):
        return (
            "attitude_rate_residual",
            "Residual body rate is above the allowed limit.",
            "Increase damping, verify desired rate is zero, and check whether torque limits are preventing rate bleed-off.",
        )
    if metric_l.endswith("total_torque_norm_n_m_s"):
        return (
            "attitude_control_effort_high",
            "The attitude controller used more integrated torque effort than allowed.",
            "Trade settle speed against effort, reduce gain aggressiveness, or revisit actuator sizing assumptions.",
        )
    if metric_l.endswith("time_inside_keepout_s"):
        return (
            "keepout_violation",
            "The trajectory spent too much time inside the keepout radius.",
            "Adjust approach corridor, braking timing, or keepout logic before optimizing terminal accuracy.",
        )
    if metric_l.endswith("final_relative_distance_km"):
        return (
            "rendezvous_terminal_range_error",
            "The terminal relative range is outside the allowed rendezvous box.",
            "Tune closing authority or extend the maneuver duration; inspect RIC axis errors to see which direction dominates.",
        )
    if metric_l.endswith("final_relative_speed_km_s"):
        return (
            "rendezvous_terminal_speed_error",
            "The terminal relative speed is too high.",
            "Bias tuning toward braking and damping rather than only reducing final position error.",
        )
    if metric_l.endswith("_radial_error_km"):
        axis = "radial"
    elif metric_l.endswith("_intrack_error_km"):
        axis = "in-track"
    elif metric_l.endswith("_crosstrack_error_km"):
        axis = "cross-track"
    else:
        axis = ""
    if axis:
        return (
            f"rendezvous_{axis.replace('-', '')}_bias",
            f"The terminal {axis} RIC component is outside the allowed box.",
            "Retune the axis weighting/gain for that channel and check whether the initial-condition envelope is exposing a persistent bias.",
        )
    if metric_l.endswith("_radial_rate_km_s"):
        axis = "radial"
    elif metric_l.endswith("_intrack_rate_km_s"):
        axis = "in-track"
    elif metric_l.endswith("_crosstrack_rate_km_s"):
        axis = "cross-track"
    else:
        axis = ""
    if axis:
        return (
            f"rendezvous_{axis.replace('-', '')}_rate_error",
            f"The terminal {axis} relative rate is outside the allowed box.",
            "Increase damping or terminal velocity weighting for that axis before tightening position tolerances.",
        )
    if metric_l.endswith("total_dv_m_s") or metric_l.endswith("fuel_used_kg"):
        return (
            "orbit_control_effort_high",
            "The orbit controller used more control effort or propellant than allowed.",
            "Compare terminal miss distance against fuel use and tune the trade between authority, duration, and terminal accuracy.",
        )
    if "runtime" in metric_l:
        return (
            "controller_runtime_slow",
            "The controller runtime exceeded the configured performance limit.",
            "Profile the controller path and cache or simplify repeated computations before increasing the runtime budget.",
        )
    if objective_kind_l == "terminal_rendezvous_box":
        return (
            "terminal_box_failure",
            "The terminal docking-box objective failed one or more constraints.",
            "Use the component metrics to identify whether position, rate, or effort is driving the miss.",
        )
    return (
        "threshold_violation",
        "A benchmark criterion failed its configured threshold.",
        "Inspect the metric value, threshold, and case geometry to decide whether to tune the controller or adjust the requirement.",
    )


def _annotate_failure_diagnoses(failures: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for failure_raw in failures:
        failure = dict(failure_raw or {})
        mode, diagnosis, suggestion = _failure_mode_for_metric(
            str(failure.get("metric", "") or ""),
            str(failure.get("reason", "") or ""),
            str(failure.get("objective_kind", "") or ""),
        )
        failure["failure_mode"] = mode
        failure["diagnosis"] = diagnosis
        failure["suggestion"] = suggestion
        out.append(failure)
    return out


def _controller_runtime_ledger(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    debug_map = dict(payload.get("controller_debug_by_object", {}) or {})
    out: dict[str, dict[str, Any]] = {}
    for object_id, rows_raw in debug_map.items():
        rows = [dict(row or {}) for row in list(rows_raw or [])]
        runtimes: list[float] = []
        for row in rows:
            value_f = _safe_metric_float(row.get("controller_runtime_ms"))
            if value_f is not None and math.isfinite(value_f):
                runtimes.append(value_f)
        out[str(object_id)] = {
            "samples": int(len(rows)),
            "mean_runtime_ms": float(sum(runtimes) / len(runtimes)) if runtimes else None,
            "max_runtime_ms": float(max(runtimes)) if runtimes else None,
        }
    return out


def _constraint_ledger_from_payload(
    payload: dict[str, Any],
    metrics: dict[str, Any],
    objectives: tuple[ControllerBenchObjective, ...],
) -> dict[str, Any]:
    summary = dict(payload.get("summary", {}) or {})
    thrust_stats = {str(k): dict(v or {}) for k, v in dict(summary.get("thrust_stats", {}) or {}).items()}
    objective_constraints: list[dict[str, Any]] = []
    for objective in objectives:
        prefix = _slug(objective.name or objective.kind)
        item: dict[str, Any] = {
            "name": objective.name or objective.kind,
            "kind": objective.kind,
            "object_id": objective.object_id,
            "reference_object_id": objective.reference_object_id,
        }
        for suffix in (
            "final_relative_distance_km",
            "final_relative_speed_km_s",
            "final_relative_radial_error_km",
            "final_relative_intrack_error_km",
            "final_relative_crosstrack_error_km",
            "final_relative_radial_rate_km_s",
            "final_relative_intrack_rate_km_s",
            "final_relative_crosstrack_rate_km_s",
            "closest_approach_km",
            "time_inside_keepout_s",
            "total_dv_m_s",
            "fuel_used_kg",
            "final_attitude_error_deg",
            "rms_attitude_error_deg",
            "max_attitude_error_deg",
            "final_body_rate_norm_rad_s",
            "max_body_rate_norm_rad_s",
            "settle_time_s",
            "settled_hold_duration_s",
            "total_torque_norm_n_m_s",
            "rocket_insertion_achieved",
            "rocket_insertion_time_s",
            "rocket_final_altitude_km",
            "rocket_final_speed_km_s",
            "rocket_final_apoapsis_alt_km",
            "rocket_final_periapsis_alt_km",
            "rocket_final_eccentricity",
            "rocket_final_propellant_remaining_fraction",
            "rocket_max_dynamic_pressure_pa",
            "rocket_max_mach",
            "rocket_max_abs_alpha_deg",
            "rocket_max_tvc_gimbal_deg",
            "rocket_max_aero_force_n",
            "rocket_total_dv_m_s",
            "rocket_burn_samples",
            "terminated_early",
        ):
            metric_name = _metric_name(prefix, suffix)
            if metric_name in metrics:
                item[suffix] = metrics.get(metric_name)
        objective_constraints.append(item)
    return {
        "terminated_early": bool(summary.get("terminated_early", False)),
        "termination_reason": summary.get("termination_reason"),
        "termination_time_s": summary.get("termination_time_s"),
        "termination_object_id": summary.get("termination_object_id"),
        "thrust_stats": thrust_stats,
        "controller_runtime_by_object": _controller_runtime_ledger(payload),
        "objective_constraints": objective_constraints,
    }


def _failure_analysis_for_run(
    run: dict[str, Any],
    metrics: dict[str, Any],
    criteria: tuple[ControllerBenchPassCriterion, ...],
    objectives: tuple[ControllerBenchObjective, ...],
    payload: dict[str, Any],
) -> dict[str, Any]:
    # objectives imports this module for shared diagnostics, so resolve the
    # objective expander lazily after both owner modules have initialized.
    from .objectives import _objective_specifications

    summary = dict(payload.get("summary", {}) or {})
    failure_records = _criterion_failure_records(metrics, criteria)
    objective_failures: list[dict[str, Any]] = []
    for objective in objectives:
        objective_metrics, objective_criteria = _objective_specifications(objective)
        scoped_metrics = {metric.name: metrics.get(metric.name) for metric in objective_metrics}
        objective_failures.extend(
            _criterion_failure_records(
                scoped_metrics,
                objective_criteria,
                objective_name=objective.name or objective.kind,
                objective_kind=objective.kind,
            )
        )
    seen_keys: set[tuple[str, str, str, str]] = set()
    all_failures: list[dict[str, Any]] = []
    for item in [*objective_failures, *failure_records]:
        key = (
            str(item.get("metric", "")),
            str(item.get("reason", "")),
            str(item.get("op", "")),
            str(item.get("threshold", "")),
        )
        if key in seen_keys:
            continue
        seen_keys.add(key)
        all_failures.append(item)
    all_failures = _annotate_failure_diagnoses(all_failures)
    missing_or_invalid = [
        row
        for row in all_failures
        if str(row.get("reason", "")) in {"missing_metric", "invalid_metric", "non_numeric_metric"}
    ]
    failure_mode_counts: dict[str, int] = {}
    suggestions: list[dict[str, Any]] = []
    seen_suggestions: set[tuple[str, str]] = set()
    for row in all_failures:
        mode = str(row.get("failure_mode", "unknown") or "unknown")
        failure_mode_counts[mode] = int(failure_mode_counts.get(mode, 0) + 1)
        suggestion = str(row.get("suggestion", "") or "").strip()
        if not suggestion:
            continue
        key = (mode, suggestion)
        if key in seen_suggestions:
            continue
        seen_suggestions.add(key)
        suggestions.append(
            {
                "failure_mode": mode,
                "suggestion": suggestion,
            }
        )
    return {
        "passed": bool(run.get("passed", False)),
        "failed_criteria": list(run.get("failed_criteria", []) or []),
        "failure_count": int(len(all_failures)),
        "failures": all_failures,
        "failure_mode_counts": failure_mode_counts,
        "suggestions": suggestions,
        "missing_or_invalid_evidence": missing_or_invalid,
        "terminated_early": bool(summary.get("terminated_early", False)),
        "termination_reason": summary.get("termination_reason"),
        "termination_time_s": summary.get("termination_time_s"),
        "termination_object_id": summary.get("termination_object_id"),
        "artifacts": dict(run.get("artifacts", {}) or {}),
    }


def _slug(text: str) -> str:
    out = re.sub(r"[^A-Za-z0-9]+", "_", str(text or "").strip().lower()).strip("_")
    return out or "objective"


def _controller_linear_system_summary(pointer: dict[str, Any]) -> dict[str, Any] | None:
    module_name = str(pointer.get("module", "") or "").strip()
    class_name = str(pointer.get("class_name", "") or "").strip()
    params = dict(pointer.get("params", {}) or {})
    if not module_name or not class_name:
        return None
    try:
        mod = importlib.import_module(module_name)
        cls = getattr(mod, class_name)
        ctrl = cls(**params)
    except Exception:
        return None
    fn = getattr(ctrl, "linear_system_summary", None)
    if not callable(fn):
        return None
    try:
        summary = fn()
    except Exception:
        return None
    return dict(summary or {}) if isinstance(summary, dict) else None


def _build_relative_rendezvous_leaderboards(
    runs: list[dict[str, Any]],
    variants: list[ControllerVariant],
) -> list[dict[str, Any]]:
    objective_runs: dict[str, dict[str, list[dict[str, Any]]]] = {}
    objective_kinds: dict[str, str] = {}
    for run in runs:
        variant_name = str(run.get("variant_name", "") or "").strip()
        for objective_result in list(run.get("objective_results", []) or []):
            kind = str(objective_result.get("kind", "") or "").strip()
            if kind not in {
                "relative_rendezvous",
                "terminal_rendezvous_box",
            }:
                continue
            objective_name = str(objective_result.get("name", "") or "").strip()
            if not objective_name or not variant_name:
                continue
            objective_kinds[objective_name] = kind
            objective_runs.setdefault(objective_name, {}).setdefault(variant_name, []).append(dict(objective_result))

    metric_specs = [
        ("final_relative_distance_km", "Mean Final Range (km)", "asc"),
        ("rms_relative_distance_km", "Mean RMS Range (km)", "asc"),
        ("final_relative_speed_km_s", "Mean Final Speed (km/s)", "asc"),
        ("final_relative_radial_error_km", "Mean Final Radial Error (km)", "asc"),
        ("final_relative_intrack_error_km", "Mean Final In-Track Error (km)", "asc"),
        ("final_relative_crosstrack_error_km", "Mean Final Cross-Track Error (km)", "asc"),
        ("total_dv_m_s", "Mean Total dV (m/s)", "asc"),
        ("fuel_used_kg", "Mean Fuel Used (kg)", "asc"),
        ("time_inside_keepout_s", "Mean Keepout Time (s)", "asc"),
    ]

    leaderboards: list[dict[str, Any]] = []
    for objective_name, by_variant in objective_runs.items():
        rankings: list[dict[str, Any]] = []

        pass_entries: list[dict[str, Any]] = []
        for variant in variants:
            rows = by_variant.get(variant.name, [])
            if not rows:
                continue
            run_count = len(rows)
            pass_rate = float(sum(1 for row in rows if bool(row.get("passed", False))) / run_count)
            pass_entries.append(
                {
                    "variant_name": variant.name,
                    "value": pass_rate,
                    "run_count": run_count,
                }
            )
        pass_entries.sort(key=lambda row: (-float(row["value"]), str(row["variant_name"])))
        if pass_entries:
            rankings.append(
                {
                    "metric": "objective_pass_rate",
                    "label": "Objective Pass Rate",
                    "direction": "desc",
                    "entries": [
                        {
                            "rank": idx + 1,
                            "variant_name": str(row["variant_name"]),
                            "value": float(row["value"]),
                            "run_count": int(row["run_count"]),
                        }
                        for idx, row in enumerate(pass_entries)
                    ],
                }
            )

        for metric_suffix, label, direction in metric_specs:
            entries: list[dict[str, Any]] = []
            for variant in variants:
                rows = by_variant.get(variant.name, [])
                if not rows:
                    continue
                values: list[float] = []
                for row in rows:
                    prefix = str(row.get("metric_prefix", "") or "").strip()
                    metric_value = dict(row.get("metrics", {}) or {}).get(f"{prefix}_{metric_suffix}")
                    try:
                        value_f = float(metric_value)
                    except (TypeError, ValueError):
                        continue
                    if not math.isfinite(value_f):
                        continue
                    values.append(value_f)
                if values:
                    entries.append(
                        {
                            "variant_name": variant.name,
                            "value": float(sum(values) / len(values)),
                            "sample_count": len(values),
                        }
                    )
            if not entries:
                continue
            reverse = direction == "desc"
            entries.sort(key=lambda row: float(row["value"]), reverse=reverse)
            if not reverse:
                entries.sort(key=lambda row: (float(row["value"]), str(row["variant_name"])))
            else:
                entries.sort(key=lambda row: (-float(row["value"]), str(row["variant_name"])))
            rankings.append(
                {
                    "metric": metric_suffix,
                    "label": label,
                    "direction": direction,
                    "entries": [
                        {
                            "rank": idx + 1,
                            "variant_name": str(row["variant_name"]),
                            "value": float(row["value"]),
                            "sample_count": int(row["sample_count"]),
                        }
                        for idx, row in enumerate(entries)
                    ],
                }
            )

        leaderboards.append(
            {
                "objective_name": objective_name,
                "kind": objective_kinds.get(objective_name, "relative_rendezvous"),
                "rankings": rankings,
            }
        )
    return leaderboards


def _build_rocket_insertion_leaderboards(
    runs: list[dict[str, Any]],
    variants: list[ControllerVariant],
) -> list[dict[str, Any]]:
    objective_runs: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for run in runs:
        variant_name = str(run.get("variant_name", "") or "").strip()
        if not variant_name:
            continue
        for objective_result in list(run.get("objective_results", []) or []):
            if str(objective_result.get("kind", "") or "").strip() != "rocket_orbit_insertion":
                continue
            objective_name = str(objective_result.get("name", "") or "").strip()
            if objective_name:
                objective_runs.setdefault(objective_name, {}).setdefault(variant_name, []).append(dict(objective_result))

    metric_specs = [
        ("rocket_insertion_time_s", "Mean Insertion Time (s)", "asc"),
        ("rocket_final_apoapsis_alt_km", "Mean Final Apogee Altitude (km)", "asc"),
        ("rocket_final_periapsis_alt_km", "Mean Final Perigee Altitude (km)", "asc"),
        ("rocket_final_eccentricity", "Mean Final Eccentricity", "asc"),
        ("rocket_final_propellant_remaining_fraction", "Mean Propellant Remaining Fraction", "desc"),
        ("rocket_max_dynamic_pressure_pa", "Mean Max Dynamic Pressure (Pa)", "asc"),
        ("rocket_max_abs_alpha_deg", "Mean Max |Alpha| (deg)", "asc"),
        ("rocket_max_tvc_gimbal_deg", "Mean Max TVC Gimbal (deg)", "asc"),
        ("rocket_total_dv_m_s", "Mean Total dV (m/s)", "asc"),
    ]

    leaderboards: list[dict[str, Any]] = []
    for objective_name, by_variant in objective_runs.items():
        rankings: list[dict[str, Any]] = []
        pass_entries: list[dict[str, Any]] = []
        for variant in variants:
            rows = by_variant.get(variant.name, [])
            if not rows:
                continue
            run_count = len(rows)
            pass_rate = float(sum(1 for row in rows if bool(row.get("passed", False))) / run_count)
            pass_entries.append(
                {
                    "variant_name": variant.name,
                    "value": pass_rate,
                    "run_count": run_count,
                }
            )
        pass_entries.sort(key=lambda row: (-float(row["value"]), str(row["variant_name"])))
        if pass_entries:
            rankings.append(
                {
                    "metric": "objective_pass_rate",
                    "label": "Objective Pass Rate",
                    "direction": "desc",
                    "entries": [
                        {
                            "rank": idx + 1,
                            "variant_name": str(row["variant_name"]),
                            "value": float(row["value"]),
                            "run_count": int(row["run_count"]),
                        }
                        for idx, row in enumerate(pass_entries)
                    ],
                }
            )

        for metric_suffix, label, direction in metric_specs:
            entries: list[dict[str, Any]] = []
            for variant in variants:
                rows = by_variant.get(variant.name, [])
                values: list[float] = []
                for row in rows:
                    prefix = str(row.get("metric_prefix", "") or "").strip()
                    metric_value = dict(row.get("metrics", {}) or {}).get(f"{prefix}_{metric_suffix}")
                    try:
                        value_f = float(metric_value)
                    except (TypeError, ValueError):
                        continue
                    if math.isfinite(value_f):
                        values.append(value_f)
                if values:
                    entries.append(
                        {
                            "variant_name": variant.name,
                            "value": float(sum(values) / len(values)),
                            "sample_count": len(values),
                        }
                    )
            if not entries:
                continue
            entries.sort(
                key=lambda row: (
                    -float(row["value"]) if direction == "desc" else float(row["value"]),
                    str(row["variant_name"]),
                )
            )
            rankings.append(
                {
                    "metric": metric_suffix,
                    "label": label,
                    "direction": direction,
                    "entries": [
                        {
                            "rank": idx + 1,
                            "variant_name": str(row["variant_name"]),
                            "value": float(row["value"]),
                            "sample_count": int(row["sample_count"]),
                        }
                        for idx, row in enumerate(entries)
                    ],
                }
            )
        leaderboards.append(
            {
                "objective_name": objective_name,
                "kind": "rocket_orbit_insertion",
                "rankings": rankings,
            }
        )
    return leaderboards


def _max_record(current: dict[str, Any] | None, candidate: dict[str, Any]) -> dict[str, Any]:
    if current is None:
        return candidate
    try:
        if float(candidate.get("value", float("-inf"))) > float(current.get("value", float("-inf"))):
            return candidate
    except (TypeError, ValueError):
        return current
    return current


def _aggregate_constraint_summary(runs: list[dict[str, Any]]) -> dict[str, Any]:
    failed_runs: list[dict[str, Any]] = []
    failure_reason_counts: dict[str, int] = {}
    failed_metric_counts: dict[str, int] = {}
    failure_mode_counts: dict[str, int] = {}
    suggestions_by_mode: dict[str, str] = {}
    early_termination_count = 0
    termination_reason_counts: dict[str, int] = {}
    worst: dict[str, dict[str, Any] | None] = {
        "fuel_used_kg": None,
        "total_dv_m_s": None,
        "total_torque_norm_n_m_s": None,
        "controller_runtime_ms": None,
        "time_inside_keepout_s": None,
        "final_relative_distance_km": None,
        "final_relative_speed_km_s": None,
        "final_relative_radial_error_km": None,
        "final_relative_intrack_error_km": None,
        "final_relative_crosstrack_error_km": None,
        "final_relative_radial_rate_km_s": None,
        "final_relative_intrack_rate_km_s": None,
        "final_relative_crosstrack_rate_km_s": None,
        "final_attitude_error_deg": None,
        "rms_attitude_error_deg": None,
        "max_attitude_error_deg": None,
        "final_body_rate_norm_rad_s": None,
        "max_body_rate_norm_rad_s": None,
        "settle_time_s": None,
        "rocket_insertion_time_s": None,
        "rocket_final_apoapsis_alt_km": None,
        "rocket_final_periapsis_alt_km": None,
        "rocket_final_eccentricity": None,
        "rocket_max_dynamic_pressure_pa": None,
        "rocket_max_mach": None,
        "rocket_max_abs_alpha_deg": None,
        "rocket_max_tvc_gimbal_deg": None,
        "rocket_max_aero_force_n": None,
        "rocket_total_dv_m_s": None,
    }
    for run in runs:
        variant_name = str(run.get("variant_name", ""))
        case_name = str(run.get("case_name", ""))
        failure_analysis = dict(run.get("failure_analysis", {}) or {})
        if not bool(run.get("passed", False)):
            run_failure_modes = dict(failure_analysis.get("failure_mode_counts", {}) or {})
            failed_runs.append(
                {
                    "variant_name": variant_name,
                    "case_name": case_name,
                    "failed_criteria": list(run.get("failed_criteria", []) or []),
                    "failure_count": int(failure_analysis.get("failure_count", 0) or 0),
                    "failure_modes": sorted(run_failure_modes.keys()),
                    "termination_reason": failure_analysis.get("termination_reason"),
                }
            )
        for failure in list(failure_analysis.get("failures", []) or []):
            failure_row = dict(failure or {})
            reason = str(failure_row.get("reason", "unknown") or "unknown")
            metric = str(failure_row.get("metric", "unknown") or "unknown")
            mode = str(failure_row.get("failure_mode", "unknown") or "unknown")
            failure_reason_counts[reason] = int(failure_reason_counts.get(reason, 0) + 1)
            failed_metric_counts[metric] = int(failed_metric_counts.get(metric, 0) + 1)
            failure_mode_counts[mode] = int(failure_mode_counts.get(mode, 0) + 1)
            suggestion = str(failure_row.get("suggestion", "") or "").strip()
            if suggestion and mode not in suggestions_by_mode:
                suggestions_by_mode[mode] = suggestion
        ledger = dict(run.get("constraint_ledger", {}) or {})
        if bool(ledger.get("terminated_early", False)):
            early_termination_count += 1
            reason = str(ledger.get("termination_reason", "unknown") or "unknown")
            termination_reason_counts[reason] = int(termination_reason_counts.get(reason, 0) + 1)
        for object_id, stats_raw in dict(ledger.get("thrust_stats", {}) or {}).items():
            stats = dict(stats_raw or {})
            for metric_name in ("fuel_used_kg", "total_dv_m_s"):
                value = _safe_metric_float(stats.get(metric_name))
                if value is None or not math.isfinite(value):
                    continue
                worst[metric_name] = _max_record(
                    worst[metric_name],
                    {
                        "value": float(value),
                        "variant_name": variant_name,
                        "case_name": case_name,
                        "object_id": str(object_id),
                    },
                )
        for object_id, stats_raw in dict(ledger.get("controller_runtime_by_object", {}) or {}).items():
            stats = dict(stats_raw or {})
            value = _safe_metric_float(stats.get("max_runtime_ms"))
            if value is None or not math.isfinite(value):
                continue
            worst["controller_runtime_ms"] = _max_record(
                worst["controller_runtime_ms"],
                {
                    "value": float(value),
                    "variant_name": variant_name,
                    "case_name": case_name,
                    "object_id": str(object_id),
                },
            )
        for metric_name, metric_value in dict(run.get("metrics", {}) or {}).items():
            metric_key = str(metric_name)
            if metric_key not in worst:
                continue
            value = _safe_metric_float(metric_value)
            if value is None or not math.isfinite(value):
                continue
            worst[metric_key] = _max_record(
                worst[metric_key],
                {
                    "value": float(value),
                    "variant_name": variant_name,
                    "case_name": case_name,
                    "objective": "",
                },
            )
        for objective in list(ledger.get("objective_constraints", []) or []):
            objective_row = dict(objective or {})
            objective_name = str(objective_row.get("name", ""))
            for metric_name in (
                "time_inside_keepout_s",
                "final_relative_distance_km",
                "final_relative_speed_km_s",
                "final_relative_radial_error_km",
                "final_relative_intrack_error_km",
                "final_relative_crosstrack_error_km",
                "final_relative_radial_rate_km_s",
                "final_relative_intrack_rate_km_s",
                "final_relative_crosstrack_rate_km_s",
                "final_attitude_error_deg",
                "rms_attitude_error_deg",
                "max_attitude_error_deg",
                "final_body_rate_norm_rad_s",
                "max_body_rate_norm_rad_s",
                "settle_time_s",
                "total_torque_norm_n_m_s",
                "rocket_insertion_time_s",
                "rocket_final_apoapsis_alt_km",
                "rocket_final_periapsis_alt_km",
                "rocket_final_eccentricity",
                "rocket_max_dynamic_pressure_pa",
                "rocket_max_mach",
                "rocket_max_abs_alpha_deg",
                "rocket_max_tvc_gimbal_deg",
                "rocket_max_aero_force_n",
                "rocket_total_dv_m_s",
            ):
                value = _safe_metric_float(objective_row.get(metric_name))
                if value is None or not math.isfinite(value):
                    continue
                worst[metric_name] = _max_record(
                    worst[metric_name],
                    {
                        "value": float(value),
                        "variant_name": variant_name,
                        "case_name": case_name,
                        "objective": objective_name,
                    },
                )
    return {
        "run_count": int(len(runs)),
        "failed_run_count": int(len(failed_runs)),
        "early_termination_count": int(early_termination_count),
        "failure_reason_counts": failure_reason_counts,
        "failed_metric_counts": failed_metric_counts,
        "failure_mode_counts": failure_mode_counts,
        "suggestions_by_mode": suggestions_by_mode,
        "termination_reason_counts": termination_reason_counts,
        "failed_runs": failed_runs,
        "worst": {key: value for key, value in worst.items() if value is not None},
    }

__all__ = [name for name in globals() if not name.startswith("__")]
