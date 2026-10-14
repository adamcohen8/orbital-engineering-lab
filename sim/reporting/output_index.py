from __future__ import annotations

import os
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _scalar(value: Any) -> str:
    if value is None:
        return "not available"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.4g}"
    text = str(value).strip()
    return text if text else "not available"


def _path_link(path_text: Any, *, base_dir: Path) -> str:
    text = str(path_text or "").strip()
    if not text:
        return "not available"
    try:
        path = Path(text)
        if path.is_absolute():
            resolved = path
        else:
            cwd_candidate = Path.cwd() / path
            base_candidate = base_dir / path
            try:
                cwd_candidate.resolve().relative_to(base_dir.resolve())
                is_already_output_relative = True
            except ValueError:
                is_already_output_relative = False
            resolved = cwd_candidate if cwd_candidate.exists() or is_already_output_relative else base_candidate
        display = str(resolved.resolve().relative_to(base_dir.resolve()))
    except ValueError:
        try:
            path = Path(text)
            resolved = path if path.is_absolute() else base_dir / path
            display = str(Path(os.path.relpath(str(resolved.resolve()), str(base_dir.resolve()))))
        except Exception:
            display = text
    except Exception:
        display = text
    href = f"<{display}>" if any(ch.isspace() for ch in display) else display
    return f"[`{display}`]({href})"


def _literal(value: Any) -> str:
    text = _scalar(value).replace("`", "\\`")
    return f"`{text}`"


def _artifact_value(key: str, value: Any, *, base_dir: Path) -> str:
    key_lower = key.lower()
    if "error" in key_lower or "warning" in key_lower or key_lower.endswith("status"):
        return _literal(value)
    if isinstance(value, (str, os.PathLike)):
        return _path_link(value, base_dir=base_dir)
    return _literal(value)


def _flatten_artifacts(value: Any, *, prefix: str = "") -> list[tuple[str, Any]]:
    if isinstance(value, dict):
        rows: list[tuple[str, Any]] = []
        for key, child in sorted(value.items()):
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            rows.extend(_flatten_artifacts(child, prefix=child_prefix))
        return rows
    if isinstance(value, list):
        rows = []
        for idx, child in enumerate(value):
            child_prefix = f"{prefix}[{idx}]" if prefix else f"[{idx}]"
            rows.extend(_flatten_artifacts(child, prefix=child_prefix))
        return rows
    return [(prefix or "artifact", value)]


def _artifact_lines(artifacts: dict[str, Any], *, base_dir: Path, limit: int = 60) -> list[str]:
    rows = [(key, value) for key, value in _flatten_artifacts(artifacts) if str(value or "").strip()]
    if not rows:
        return ["- No saved artifacts were listed for this run."]
    lines = [f"- `{key}`: {_artifact_value(key, value, base_dir=base_dir)}" for key, value in rows[:limit]]
    if len(rows) > limit:
        lines.append(f"- ... {len(rows) - limit} additional artifacts omitted from this index")
    return lines


def _artifact_path(artifacts: dict[str, Any], key: str) -> str:
    value = artifacts.get(key)
    if isinstance(value, (str, os.PathLike)) and str(value).strip():
        return str(value)
    return ""


def _nested_artifact_path(artifacts: dict[str, Any], group: str, key: str) -> str:
    value = dict(artifacts.get(group, {}) or {}).get(key)
    if isinstance(value, (str, os.PathLike)) and str(value).strip():
        return str(value)
    return ""


def _has_artifact_group(artifacts: dict[str, Any], *keys: str) -> bool:
    for key in keys:
        value = artifacts.get(key)
        if isinstance(value, dict) and any(str(child or "").strip() for child in value.values()):
            return True
        if isinstance(value, list) and any(str(child or "").strip() for child in value):
            return True
        if isinstance(value, (str, os.PathLike)) and str(value).strip():
            return True
    return False


def _linked_artifact_name(artifacts: dict[str, Any], key: str, fallback: str, *, base_dir: Path) -> str:
    path_text = _artifact_path(artifacts, key)
    if path_text:
        return _path_link(path_text, base_dir=base_dir)
    return f"`{fallback}`"


def _linked_nested_artifact_name(
    artifacts: dict[str, Any], group: str, key: str, fallback: str, *, base_dir: Path
) -> str:
    path_text = _nested_artifact_path(artifacts, group, key)
    if path_text:
        return _path_link(path_text, base_dir=base_dir)
    return f"`{fallback}`"


def _default_next_steps(*, workflow: str, artifacts: dict[str, Any], base_dir: Path) -> list[str]:
    steps: list[str] = []
    if "workflow_manifest_json" in artifacts:
        name = _linked_artifact_name(
            artifacts, "workflow_manifest_json", "review/workflow_manifest.json", base_dir=base_dir
        )
        steps.append(f"Open {name} for the common workflow review manifest.")
    if workflow == "single_run":
        if _nested_artifact_path(artifacts, "plots", "run_dashboard"):
            name = _linked_nested_artifact_name(
                artifacts, "plots", "run_dashboard", "run_dashboard.png", base_dir=base_dir
            )
            steps.append(f"Open {name} for the fastest visual overview.")
        if "summary_json" in artifacts:
            name = _linked_artifact_name(artifacts, "summary_json", "master_run_summary.json", base_dir=base_dir)
            steps.append(f"Open {name} for stable run metadata and metrics.")
        if "run_log_json" in artifacts:
            name = _linked_artifact_name(artifacts, "run_log_json", "master_run_log.json", base_dir=base_dir)
            steps.append(f"Open {name} for saved time histories and custom plotting data.")
        if _has_artifact_group(artifacts, "plots", "animations"):
            steps.append("Inspect generated plot or animation artifacts listed below.")
    elif workflow == "monte_carlo":
        if "summary_json" in artifacts:
            name = _linked_artifact_name(
                artifacts, "summary_json", "master_monte_carlo_summary.json", base_dir=base_dir
            )
            steps.append(f"Open {name} for aggregate campaign results.")
        if "commander_brief_md" in artifacts:
            name = _linked_artifact_name(
                artifacts, "commander_brief_md", "master_monte_carlo_commander_brief.md", base_dir=base_dir
            )
            steps.append(f"Open {name} for the human-readable campaign brief.")
        steps.append("Inspect campaign plots and AI report artifacts when present.")
    elif workflow == "sensitivity":
        if "report_md" in artifacts:
            name = _linked_artifact_name(
                artifacts, "report_md", "master_analysis_sensitivity_report.md", base_dir=base_dir
            )
            steps.append(f"Open {name} for the human-readable study report.")
        if "rankings_csv" in artifacts:
            name = _linked_artifact_name(
                artifacts, "rankings_csv", "master_analysis_sensitivity_rankings.csv", base_dir=base_dir
            )
            steps.append(f"Open {name} to inspect ranked parameter effects.")
        steps.append("Inspect generated response, scatter, grid, or ranking figures listed below.")
    else:
        steps.append("Inspect the artifact inventory below.")
    return steps or ["Inspect the artifact inventory below."]


def _single_run_status(summary: dict[str, Any]) -> str:
    review_status = str(summary.get("review_store_status", "") or "").strip()
    if review_status.startswith("failed:"):
        return f"partial - review store {review_status}"
    if bool(summary.get("terminated_early", False)):
        reason = str(summary.get("termination_reason") or "unknown").strip()
        if reason == "rocket_orbit_insertion":
            return "success - rocket orbit insertion achieved"
        return f"stopped early - {reason or 'unknown reason'}"
    return "nominal - full duration reached"


def _single_run_provenance(summary: dict[str, Any]) -> list[str]:
    propagation = dict(summary.get("object_propagation", {}) or {})
    frame = dict(summary.get("frame_provenance", {}) or {})
    if summary.get("central_body", "earth") != "earth":
        frame["model"] = f"{summary['central_body']}-centered {summary.get('inertial_axes', 'J2000')}"
    review_status = str(summary.get("review_store_status", "not requested") or "not requested")
    installation = dict(summary.get("installation_provenance", {}) or {})
    families = sorted(
        {
            str(dict(item or {}).get("propagator_name") or dict(item or {}).get("propagator_family") or "").strip()
            for item in propagation.values()
            if str(dict(item or {}).get("propagator_name") or dict(item or {}).get("propagator_family") or "").strip()
        }
    )
    return [
        f"- Config source: `{_scalar(summary.get('config_source_path'))}`",
        f"- Config SHA-256: `{_scalar(summary.get('config_sha256'))}`",
        f"- Run generated UTC: `{_scalar(summary.get('generated_utc'))}`",
        f"- OEL version: `{_scalar(summary.get('oel_version', installation.get('engine_version')))}`",
        f"- Installation disposition: `{_scalar(installation.get('installation_disposition'))}`",
        f"- Propagator(s): `{', '.join(families) if families else 'see saved configuration'}`",
        f"- Frame model: `{_scalar(frame.get('model'))}`",
        f"- Review evidence status: `{review_status}`",
        f"- Review detail: `{_scalar(summary.get('review_detail'))}`",
    ]


def _single_run_claim_limits(summary: dict[str, Any]) -> list[str]:
    lines = [
        "- Engine completion means the configured duration or termination condition was reached; it is not, by itself, a mission-success verdict.",
        "- Results apply only to the configured deterministic models, inputs, cadence, and retained samples.",
    ]
    if dict(summary.get("ground_station_access", {}) or {}):
        lines.append("- Access windows use retained-sample bounds unless a refinement disposition says otherwise; run-boundary windows may be censored.")
    if dict(summary.get("orbital_analysis", {}) or {}).get("directed_links"):
        lines.append("- Link availability is conditional on the configured terminals, thresholds, geometry, and link-budget assumptions.")
    if str(summary.get("review_detail", "standard") or "standard") == "compact":
        lines.append(
            "- Compact review detail omits diagnostic evidence; rerun with `outputs.review.detail: standard` "
            "or `full` when controller or FSW diagnostics are required."
        )
    return lines


def _guardrail_event_count(summary: dict[str, Any]) -> int:
    total = 0
    for value in dict(summary.get("attitude_guardrail_stats", {}) or {}).values():
        try:
            total += int(value)
        except (TypeError, ValueError):
            pass
    return total


def _relative_range_summary(summary: dict[str, Any]) -> dict[str, Any]:
    return dict(summary.get("relative_range_summary", {}) or {})


def _format_pair(pair: Any) -> str:
    values = [str(item) for item in list(pair or []) if str(item).strip()]
    return " to ".join(values) if len(values) == 2 else "primary pair"


def _single_run_narrative(summary: dict[str, Any], artifacts: dict[str, Any]) -> list[str]:
    objects = [str(item) for item in list(summary.get("objects", []) or [])]
    object_text = ", ".join(objects) if objects else "the configured objects"
    duration = _scalar(summary.get("duration_s"))
    samples = _scalar(summary.get("samples"))
    status = _single_run_status(summary)
    total_dv = _single_run_total_dv(summary)
    range_summary = _relative_range_summary(summary)
    plot_count = len(dict(summary.get("plot_outputs", {}) or {}))
    animation_count = len(dict(summary.get("animation_outputs", {}) or {}))

    lines = [
        (
            f"This single-run simulation completed with status **{status}** for objects: {object_text}. "
            f"It covered `{duration} s` across `{samples}` saved samples."
        )
    ]
    if range_summary:
        pair = _format_pair(range_summary.get("object_pair"))
        lines.append(
            f"The closest {pair} range was `{_scalar(range_summary.get('closest_approach_km'))} km` "
            f"at `t={_scalar(range_summary.get('closest_approach_time_s'))} s`; "
            f"the final range was `{_scalar(range_summary.get('final_range_km'))} km`."
        )
    orbital = dict(summary.get("orbital_analysis", {}) or {})
    coverage_rows = [dict(row or {}) for row in list(orbital.get("coverage", []) or [])]
    link_rows = [dict(row or {}) for row in list(orbital.get("directed_links", []) or [])]
    if coverage_rows:
        coverage_text = "; ".join(
            f"{row.get('analysis_id')}: mean `{100.0 * float(row.get('time_weighted_mean_covered_fraction') or 0.0):.3f}%`, "
            f"ever `{100.0 * float(row.get('ever_covered_fraction') or 0.0):.3f}%`"
            for row in coverage_rows
        )
        lines.append(f"Whole-Earth coverage results: {coverage_text}.")
    if link_rows:
        link_text = "; ".join(
            f"{row.get('analysis_id')}: available `{100.0 * float(row.get('sampled_available_fraction') or 0.0):.3f}%` "
            f"for `{float(row.get('available_duration_s') or 0.0):.3f} s`"
            for row in link_rows
        )
        lines.append(f"Directed-link results: {link_text}.")
    lines.append(
        f"The run used `{total_dv:.4g} m/s` total delta-v and recorded `{_guardrail_event_count(summary)}` "
        "attitude guardrail events."
    )
    if plot_count or animation_count:
        lines.append(
            f"It generated `{plot_count}` plot artifact(s) and `{animation_count}` animation artifact(s); "
            "open the listed visual artifacts before diving into raw JSON."
        )
    elif "summary_json" in artifacts:
        lines.append("No plot or animation artifacts were generated for this run; start with the summary JSON.")
    return lines


def _single_run_total_dv(summary: dict[str, Any]) -> float:
    thrust_stats = dict(summary.get("thrust_stats", {}) or {})
    total_dv = 0.0
    for row in thrust_stats.values():
        try:
            total_dv += float(dict(row or {}).get("total_dv_m_s", 0.0))
        except (TypeError, ValueError):
            pass
    return total_dv


def _single_run_next_command(summary: dict[str, Any]) -> str:
    def command_arg(value: str) -> str:
        if os.name == "nt":
            return subprocess.list2cmdline([value])
        return shlex.quote(value)

    review_db = str(summary.get("review_sqlite_path", "") or "").strip()
    if review_db:
        review_root = Path(review_db).parent.parent
        if dict(summary.get("orbital_analysis", {}) or {}).get("directed_links"):
            query_name = "directed_link_summary"
        elif dict(summary.get("ground_station_access_summary", {}) or {}):
            query_name = "ground_access_summary"
        elif _relative_range_summary(summary):
            query_name = "rendezvous_metrics"
        else:
            query_name = "object_final_state"
        return f"python -m sim.review {command_arg(str(review_root))} --saved-query {query_name}"
    config_source_path = str(summary.get("config_source_path", "") or "").strip()
    if config_source_path:
        return f"python run_simulation.py --config {command_arg(config_source_path)} --validate-only"
    return ""


def _single_run_metrics(summary: dict[str, Any]) -> list[str]:
    total_dv = _single_run_total_dv(summary)
    range_summary = _relative_range_summary(summary)
    lines = [
        f"- Samples: `{_scalar(summary.get('samples'))}`",
        f"- Duration: `{_scalar(summary.get('duration_s'))} s`",
        f"- Objects: `{_scalar(', '.join(list(summary.get('objects', []) or [])))}`",
        f"- Status: `{_single_run_status(summary)}`",
        f"- Total delta-v: `{total_dv:.4g} m/s`",
        f"- Attitude guardrail events: `{_guardrail_event_count(summary)}`",
        f"- Plots: `{len(dict(summary.get('plot_outputs', {}) or {}))}`",
        f"- Animations: `{len(dict(summary.get('animation_outputs', {}) or {}))}`",
    ]
    if range_summary:
        lines.extend(
            [
                f"- Primary range pair: `{_format_pair(range_summary.get('object_pair'))}`",
                f"- Initial range: `{_scalar(range_summary.get('initial_range_km'))} km`",
                f"- Closest approach: `{_scalar(range_summary.get('closest_approach_km'))} km`",
                f"- Closest approach time: `{_scalar(range_summary.get('closest_approach_time_s'))} s`",
                f"- Final range: `{_scalar(range_summary.get('final_range_km'))} km`",
            ]
        )
    ground_access = dict(summary.get("ground_station_access_summary", {}) or {})
    if ground_access:
        access_pairs = 0
        access_duration = 0.0
        access_samples = 0
        total_samples = 0
        reason_counts: dict[str, int] = {}
        thresholds: set[str] = set()
        for by_target in ground_access.values():
            for row in dict(by_target or {}).values():
                item = dict(row or {})
                access_pairs += 1
                try:
                    access_duration += float(item.get("access_duration_s", 0.0))
                    access_samples += int(item.get("access_samples", 0))
                    total_samples += int(item.get("samples", 0))
                except (TypeError, ValueError):
                    pass
                for reason, count in dict(item.get("reason_sample_count", {}) or {}).items():
                    reason_counts[str(reason)] = reason_counts.get(str(reason), 0) + int(count)
                thresholds.add(
                    f"elevation >= {_scalar(item.get('minimum_elevation_deg'))} deg; "
                    f"range <= {_scalar(item.get('maximum_range_km'))} km"
                )
        lines.append(f"- Ground stations: `{len(ground_access)}`")
        lines.append(f"- Ground-station access pairs: `{access_pairs}`")
        lines.append(f"- Ground-station accessible samples: `{access_samples}/{total_samples}`")
        lines.append(f"- Ground-station access duration sum: `{access_duration:.4g} s`")
        lines.append(f"- Ground-station thresholds: `{'; '.join(sorted(thresholds))}`")
        lines.append(
            "- Ground-station reason samples: `"
            + ", ".join(f"{name}={count}" for name, count in sorted(reason_counts.items()))
            + "`"
        )
    orbital = dict(summary.get("orbital_analysis", {}) or {})
    for row in list(orbital.get("coverage", []) or []):
        item = dict(row or {})
        lines.append(
            f"- Coverage `{item.get('analysis_id')}` time-weighted mean: "
            f"`{100.0 * float(item.get('time_weighted_mean_covered_fraction') or 0.0):.4g}%`"
        )
        lines.append(
            f"- Coverage `{item.get('analysis_id')}` ever covered: "
            f"`{100.0 * float(item.get('ever_covered_fraction') or 0.0):.4g}%`"
        )
    for row in list(orbital.get("directed_links", []) or []):
        item = dict(row or {})
        lines.append(
            f"- Link `{item.get('analysis_id')}` sampled availability: "
            f"`{100.0 * float(item.get('sampled_available_fraction') or 0.0):.4g}%`"
        )
        lines.append(
            f"- Link `{item.get('analysis_id')}` available duration: "
            f"`{float(item.get('available_duration_s') or 0.0):.4g} s`"
        )
        lines.append(
            f"- Link `{item.get('analysis_id')}` available samples/windows: "
            f"`{int(item.get('available_sample_count') or 0)}/{int(item.get('sample_count') or 0)}; "
            f"windows={int(item.get('interval_count') or 0)}`"
        )
        lines.append(
            f"- Link `{item.get('analysis_id')}` thresholds: "
            f"`required Eb/N0={_scalar(item.get('required_eb_n0_db'))} dB; "
            f"elevation >= {_scalar(item.get('minimum_fixed_site_elevation_deg'))} deg; "
            f"range <= {_scalar(item.get('maximum_range_km_threshold'))} km`"
        )
        reasons = dict(item.get("primary_reason_sample_count", {}) or {})
        lines.append(
            f"- Link `{item.get('analysis_id')}` reason samples: `"
            + ", ".join(f"{name}={count}" for name, count in sorted(reasons.items()) if int(count) > 0)
            + "`"
        )
    return lines


def _monte_carlo_metrics(payload: dict[str, Any]) -> list[str]:
    aggregate = dict(payload.get("aggregate_stats", {}) or {})
    commander = dict(payload.get("commander_brief", {}) or {})
    comparison = dict(payload.get("comparison_summary", {}) or commander.get("comparison_summary", {}) or {})
    runs = list(payload.get("runs", []) or [])
    pass_label = "Pass rate (pooled across comparison options)" if comparison else "Pass rate"
    lines = [
        f"- Iterations: `{len(runs) if runs else _scalar(dict(payload.get('monte_carlo', {}) or {}).get('iterations'))}`",
        f"- {pass_label}: `{_scalar(aggregate.get('pass_rate', commander.get('p_success')))}`",
        f"- Closest approach mean: `{_scalar(aggregate.get('closest_approach_km_mean'))} km`",
        f"- Keepout violation probability: `{_scalar(aggregate.get('p_keepout_violation', commander.get('p_keepout_violation')))}`",
        f"- Total delta-v mean: `{_scalar(aggregate.get('total_dv_m_s_mean'))} m/s`",
    ]
    delivery = dict(payload.get("orbital_delivery", {}) or {})
    if delivery:
        lines.extend(
            [
                f"- Payload delivery success rate: `{_scalar(delivery.get('delivery_success_rate'))}`",
                f"- RPO insertion feasibility rate: `{_scalar(delivery.get('rpo_feasibility_rate'))}`",
                f"- Delivery position error P95: `{_scalar(dict(delivery.get('position_error_km', {}) or {}).get('p95'))} km`",
                f"- Delivery velocity error P95: `{_scalar(dict(delivery.get('velocity_error_m_s', {}) or {}).get('p95'))} m/s`",
            ]
        )
    return lines


def _sensitivity_metrics(payload: dict[str, Any]) -> list[str]:
    analysis = dict(payload.get("analysis", {}) or {})
    rankings = list(payload.get("parameter_rankings", []) or [])
    top_driver = "not available"
    if rankings:
        top_driver = str(dict(rankings[0] or {}).get("parameter_path", "not available"))
    return [
        f"- Method: `{_scalar(analysis.get('method'))}`",
        f"- Total runs: `{_scalar(analysis.get('run_count'))}`",
        f"- Successful runs: `{_scalar(analysis.get('successful_run_count'))}`",
        f"- Failed runs: `{_scalar(analysis.get('failed_run_count'))}`",
        f"- Top ranked parameter: `{top_driver}`",
    ]


def write_output_index(
    *,
    outdir: Path,
    workflow: str,
    title: str,
    summary: dict[str, Any] | None = None,
    payload: dict[str, Any] | None = None,
    artifacts: dict[str, Any] | None = None,
    next_steps: list[str] | None = None,
) -> Path:
    """Write a human-readable index for an output directory."""
    outdir.mkdir(parents=True, exist_ok=True)
    index_path = outdir / "index.md"
    summary = dict(summary or {})
    payload = dict(payload or {})
    artifacts = dict(artifacts or {})
    scenario_name = summary.get("scenario_name", payload.get("scenario_name", title))
    scenario_description = str(
        summary.get("scenario_description", payload.get("scenario_description", "")) or ""
    ).strip()

    if workflow == "single_run":
        key_metrics = _single_run_metrics(summary)
        status = _single_run_status(summary)
        narrative = _single_run_narrative(summary, artifacts)
        next_command = _single_run_next_command(summary)
    elif workflow == "monte_carlo":
        key_metrics = _monte_carlo_metrics(payload)
        status = "campaign complete"
        narrative = []
        next_command = ""
    elif workflow == "sensitivity":
        key_metrics = _sensitivity_metrics(payload)
        status = "analysis complete"
        narrative = []
        next_command = ""
    else:
        key_metrics = ["- No workflow-specific metrics are available yet."]
        status = "complete"
        narrative = []
        next_command = ""

    steps = list(next_steps or _default_next_steps(workflow=workflow, artifacts=artifacts, base_dir=outdir))
    lines = [
        "# Start Here",
        "",
        "## Run Status",
        f"- Status: `{status}`",
        f"- Workflow: `{workflow}`",
        f"- Scenario: `{_scalar(scenario_name)}`",
    ]
    if scenario_description:
        lines.append(f"- Description: {scenario_description}")
    lines.extend(
        [
            f"- Output directory: `{outdir}`",
            f"- Generated UTC: `{datetime.now(timezone.utc).isoformat()}`",
            "",
            "## What Happened",
            *(narrative or ["Review the key results and artifact inventory below."]),
            "",
            "## Key Results",
            *key_metrics,
            "",
            "## Evidence Provenance",
            *(_single_run_provenance(summary) if workflow == "single_run" else ["- See the workflow artifacts below."]),
            "",
            "## Claim Limits And Success Criteria",
            *(_single_run_claim_limits(summary) if workflow == "single_run" else ["- Interpret results within the configured study scope."]),
            "",
            "## Open First",
            *[f"{idx}. {step}" for idx, step in enumerate(steps, start=1)],
            "",
            "## Next Command",
            (
                "Activate the OEL virtual environment first; see `docs/installation.md` "
                "for Windows and POSIX commands.\n\n"
                f"```text\n{next_command}\n```"
                if next_command
                else "No default next command is defined for this workflow."
            ),
            "",
            "## Artifact Inventory",
            *_artifact_lines(artifacts, base_dir=outdir),
            "",
        ]
    )
    index_path.write_text("\n".join(lines), encoding="utf-8")
    return index_path
