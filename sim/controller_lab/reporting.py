from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from sim.reporting.output_index import write_output_index
from sim.review import write_workflow_review


def write_controller_bench_reports(result: dict[str, Any], outdir: Path) -> dict[str, str]:
    outdir.mkdir(parents=True, exist_ok=True)

    summary_json = outdir / "controller_bench_summary.json"
    summary_md = outdir / "controller_bench_summary.md"
    comparison_csv = outdir / "controller_bench_comparison.csv"
    leaderboard_csv = outdir / "controller_bench_leaderboard.csv"
    failures_csv = outdir / "controller_bench_failures.csv"
    evidence_manifest_json = outdir / "controller_bench_evidence_manifest.json"
    artifact_paths = {
        "summary_json": str(summary_json),
        "summary_md": str(summary_md),
        "comparison_csv": str(comparison_csv),
        "leaderboard_csv": str(leaderboard_csv),
        "failures_csv": str(failures_csv),
        "evidence_manifest_json": str(evidence_manifest_json),
    }
    current_artifacts = dict(result.get("artifacts", {}) or {})
    result["artifacts"] = {**current_artifacts, **artifact_paths}

    summary_json.write_text(json.dumps(result, indent=2), encoding="utf-8")

    runs = list(result.get("runs", []) or [])
    metric_names: list[str] = []
    for run in runs:
        for name in dict(run.get("metrics", {}) or {}).keys():
            if name not in metric_names:
                metric_names.append(str(name))

    with comparison_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["variant", "case", "passed", "output_dir", *metric_names],
        )
        writer.writeheader()
        for run in runs:
            row = {
                "variant": run.get("variant_name"),
                "case": run.get("case_name"),
                "passed": run.get("passed"),
                "output_dir": run.get("artifacts", {}).get("output_dir"),
            }
            for name in metric_names:
                row[name] = dict(run.get("metrics", {}) or {}).get(name)
            writer.writerow(row)

    leaderboard_rows: list[dict[str, Any]] = []
    leaderboards_by_kind = {
        str(kind): list(rows or [])
        for kind, rows in dict(result.get("leaderboards", {}) or {}).items()
    }
    relative_rendezvous = list(leaderboards_by_kind.get("relative_rendezvous", []) or [])
    rocket_orbit_insertion = list(leaderboards_by_kind.get("rocket_orbit_insertion", []) or [])
    with leaderboard_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["kind", "objective", "metric", "label", "direction", "rank", "variant", "value", "samples"],
        )
        writer.writeheader()
        for kind, objectives in leaderboards_by_kind.items():
            for objective in objectives:
                objective_name = str(objective.get("objective_name", "") or "")
                for ranking in list(objective.get("rankings", []) or []):
                    for entry in list(ranking.get("entries", []) or []):
                        row = {
                            "kind": kind,
                            "objective": objective_name,
                            "metric": ranking.get("metric"),
                            "label": ranking.get("label"),
                            "direction": ranking.get("direction"),
                            "rank": entry.get("rank"),
                            "variant": entry.get("variant_name"),
                            "value": entry.get("value"),
                            "samples": entry.get("sample_count", entry.get("run_count")),
                        }
                        writer.writerow(row)
                        leaderboard_rows.append(row)

    with failures_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "variant",
                "case",
                "objective",
                "objective_kind",
                "metric",
                "op",
                "threshold",
                "value",
                "reason",
                "failure_mode",
                "diagnosis",
                "suggestion",
                "margin",
                "termination_reason",
                "output_dir",
            ],
        )
        writer.writeheader()
        for run in runs:
            failure_analysis = dict(run.get("failure_analysis", {}) or {})
            artifacts = dict(run.get("artifacts", {}) or {})
            failures = list(failure_analysis.get("failures", []) or [])
            if not failures and not bool(run.get("passed", False)):
                failures = [{"reason": "unknown"}]
            for failure_raw in failures:
                failure = dict(failure_raw or {})
                writer.writerow(
                    {
                        "variant": run.get("variant_name"),
                        "case": run.get("case_name"),
                        "objective": failure.get("objective"),
                        "objective_kind": failure.get("objective_kind"),
                        "metric": failure.get("metric"),
                        "op": failure.get("op"),
                        "threshold": failure.get("threshold"),
                        "value": failure.get("value"),
                        "reason": failure.get("reason"),
                        "failure_mode": failure.get("failure_mode"),
                        "diagnosis": failure.get("diagnosis"),
                        "suggestion": failure.get("suggestion"),
                        "margin": failure.get("margin"),
                        "termination_reason": failure_analysis.get("termination_reason"),
                        "output_dir": artifacts.get("output_dir"),
                    }
                )

    lines = [
        f"# {result.get('suite_name', 'controller_bench')}",
        "",
        str(result.get("description", "") or "").strip(),
        "",
        "## Variants",
        "",
        "| Variant | Pass Rate | Passed | Total |",
        "| --- | ---: | ---: | ---: |",
    ]
    for summary in list(result.get("variant_summaries", []) or []):
        lines.append(
            f"| {summary.get('variant_name')} | "
            f"{100.0 * float(summary.get('pass_rate', 0.0)):.1f}% | "
            f"{int(summary.get('passed_runs', 0))} | "
            f"{int(summary.get('run_count', 0))} |"
        )
    envelope = dict(result.get("envelope", {}) or {})
    if envelope.get("enabled"):
        lines.extend(
            [
                "",
                "## Envelope",
                "",
                f"- Generated cases: `{int(envelope.get('generated_case_count', 0) or 0)}`",
                f"- Base cases: `{', '.join(list(envelope.get('base_cases', []) or [])) or 'all'}`",
                "",
                "| Variation | Path | Values |",
                "| --- | --- | --- |",
            ]
        )
        for variation in list(envelope.get("variations", []) or []):
            row = dict(variation or {})
            values = ", ".join(str(value) for value in list(row.get("values", []) or []))
            lines.append(f"| {row.get('name', '')} | `{row.get('path', '')}` | {values} |")
    optimization = dict(result.get("optimization", {}) or {})
    if optimization.get("enabled"):
        best_params = dict(optimization.get("best_parameters", {}) or {})
        lines.extend(
            [
                "",
                "## Optimization",
                "",
                f"- Algorithm: `{optimization.get('algorithm', 'pso')}`",
                f"- Particles: `{optimization.get('particles')}`",
                f"- Iterations: `{optimization.get('iterations')}`",
                f"- Workers: `{optimization.get('workers')}`",
                f"- Best cost: `{float(optimization.get('best_cost', 0.0)):.6g}`",
                "",
                "| Parameter | Value |",
                "| --- | ---: |",
            ]
        )
        for name in sorted(best_params.keys()):
            value = best_params[name]
            if isinstance(value, (float, int)):
                value_text = f"{float(value):.12g}"
            else:
                value_text = json.dumps(value)
            lines.append(f"| {name} | {value_text} |")
        best_variant_yaml = str(optimization.get("best_variant_yaml", "") or "").strip()
        if best_variant_yaml:
            lines.extend(
                [
                    "",
                    "Best tuned variant:",
                    "",
                    "```yaml",
                    best_variant_yaml,
                    "```",
                ]
            )
    objective_names: list[str] = []
    for summary in list(result.get("variant_summaries", []) or []):
        for name in dict(summary.get("objective_pass_rates", {}) or {}).keys():
            if name not in objective_names:
                objective_names.append(str(name))
    if objective_names:
        lines.extend(
            [
                "",
                "## Objective Pass Rates",
                "",
                "| Variant | Objective | Pass Rate |",
                "| --- | --- | ---: |",
            ]
        )
        for summary in list(result.get("variant_summaries", []) or []):
            objective_pass_rates = dict(summary.get("objective_pass_rates", {}) or {})
            for name in objective_names:
                if name not in objective_pass_rates:
                    continue
                lines.append(
                    f"| {summary.get('variant_name')} | {name} | "
                    f"{100.0 * float(objective_pass_rates.get(name, 0.0)):.1f}% |"
                )
    if relative_rendezvous:
        lines.extend(
            [
                "",
                "## Rendezvous Leaderboards",
                "",
            ]
        )
        for objective in relative_rendezvous:
            objective_name = str(objective.get("objective_name", "") or "").strip()
            if not objective_name:
                continue
            lines.extend(
                [
                    f"### {objective_name}",
                    "",
                ]
            )
            for ranking in list(objective.get("rankings", []) or []):
                label = str(ranking.get("label", ranking.get("metric", "metric")) or "metric")
                lines.extend(
                    [
                        f"- {label}",
                        "",
                        "| Rank | Variant | Value | Samples |",
                        "| ---: | --- | ---: | ---: |",
                    ]
                )
                for entry in list(ranking.get("entries", []) or []):
                    value = entry.get("value")
                    if isinstance(value, float):
                        value_text = f"{value:.6g}"
                    else:
                        value_text = str(value)
                    sample_count = entry.get("sample_count", entry.get("run_count", ""))
                    lines.append(
                        f"| {entry.get('rank')} | {entry.get('variant_name')} | {value_text} | {sample_count} |"
                )
                lines.append("")
    if rocket_orbit_insertion:
        lines.extend(
            [
                "",
                "## Rocket Insertion Leaderboards",
                "",
            ]
        )
        for objective in rocket_orbit_insertion:
            objective_name = str(objective.get("objective_name", "") or "").strip()
            if not objective_name:
                continue
            lines.extend(
                [
                    f"### {objective_name}",
                    "",
                ]
            )
            for ranking in list(objective.get("rankings", []) or []):
                label = str(ranking.get("label", ranking.get("metric", "metric")) or "metric")
                lines.extend(
                    [
                        f"- {label}",
                        "",
                        "| Rank | Variant | Value | Samples |",
                        "| ---: | --- | ---: | ---: |",
                    ]
                )
                for entry in list(ranking.get("entries", []) or []):
                    value = entry.get("value")
                    if isinstance(value, float):
                        value_text = f"{value:.6g}"
                    else:
                        value_text = str(value)
                    sample_count = entry.get("sample_count", entry.get("run_count", ""))
                    lines.append(
                        f"| {entry.get('rank')} | {entry.get('variant_name')} | {value_text} | {sample_count} |"
                    )
                lines.append("")
    constraint_summary = dict(result.get("constraint_summary", {}) or {})
    if constraint_summary:
        failed_runs = list(constraint_summary.get("failed_runs", []) or [])
        worst = dict(constraint_summary.get("worst", {}) or {})
        failure_mode_counts = dict(constraint_summary.get("failure_mode_counts", {}) or {})
        suggestions_by_mode = dict(constraint_summary.get("suggestions_by_mode", {}) or {})
        lines.extend(
            [
                "",
                "## Constraint Summary",
                "",
                f"- Failed runs: `{int(constraint_summary.get('failed_run_count', 0) or 0)}` / `{int(constraint_summary.get('run_count', 0) or 0)}`",
                f"- Early terminations: `{int(constraint_summary.get('early_termination_count', 0) or 0)}`",
                "",
            ]
        )
        if worst:
            lines.extend(
                [
                    "| Constraint | Worst Value | Variant | Case | Context |",
                    "| --- | ---: | --- | --- | --- |",
                ]
            )
            for key in sorted(worst.keys()):
                row = dict(worst.get(key, {}) or {})
                context = row.get("object_id") or row.get("objective") or ""
                value = row.get("value")
                value_txt = f"{float(value):.6g}" if isinstance(value, (float, int)) else str(value)
                lines.append(
                    f"| {key} | {value_txt} | {row.get('variant_name', '')} | {row.get('case_name', '')} | {context} |"
                )
        if failure_mode_counts:
            lines.extend(
                [
                    "",
                    "## Failure Modes",
                    "",
                    "| Failure Mode | Count | Suggested Next Action |",
                    "| --- | ---: | --- |",
                ]
            )
            for mode, count in sorted(failure_mode_counts.items(), key=lambda item: (-int(item[1]), str(item[0]))):
                suggestion = str(suggestions_by_mode.get(mode, "") or "")
                lines.append(f"| {mode} | {int(count)} | {suggestion} |")
        if failed_runs:
            lines.extend(
                [
                    "",
                    "## Failed Runs",
                    "",
                    "| Variant | Case | Failures | Modes | Termination |",
                    "| --- | --- | ---: | --- | --- |",
                ]
            )
            for row in failed_runs[:20]:
                modes = ", ".join(str(mode) for mode in list(row.get("failure_modes", []) or []))
                lines.append(
                    f"| {row.get('variant_name', '')} | {row.get('case_name', '')} | "
                    f"{int(row.get('failure_count', 0) or 0)} | {modes} | {row.get('termination_reason') or ''} |"
                )
            if len(failed_runs) > 20:
                lines.append(f"| ... | ... | {len(failed_runs) - 20} more | ... | |")
    semantic_lint = dict(result.get("semantic_lint", {}) or {})
    if semantic_lint.get("enabled"):
        lines.extend(
            [
                "",
                "## Semantic Lint",
                "",
                f"- Findings: `{int(semantic_lint.get('finding_count', 0) or 0)}`",
                f"- Errors: `{int(semantic_lint.get('error_count', 0) or 0)}`",
                f"- Warnings: `{int(semantic_lint.get('warning_count', 0) or 0)}`",
            ]
        )
    engineering_runs = _controller_bench_engineering_rows(result)
    if engineering_runs:
        lines.extend(
            [
                "",
                "## Engineering Evidence",
                "",
                "| Variant | Case | Stack Conformance | Saturation Fraction | Spectral Radius | Task Timing Error (s) |",
                "| --- | --- | --- | ---: | ---: | ---: |",
            ]
        )
        for row in engineering_runs:
            lines.append(
                f"| {row.get('variant_name', '')} | {row.get('case_name', '')} | "
                f"{row.get('stack_conformance_passed', '')} | "
                f"{_report_number(row.get('actuator_saturation_fraction'))} | "
                f"{_report_number(row.get('identified_spectral_radius'))} | "
                f"{_report_number(row.get('max_nominal_release_timing_error_s'))} |"
            )
    narrative_lines = list(lines)

    review_outputs = write_workflow_review(
        output_dir=outdir,
        workflow_type="controller_bench",
        title=str(result.get("suite_name", "controller_bench")),
        scenario_name=str(result.get("suite_name", "controller_bench")),
        status="building",
        summary={
            "variant_count": len(list(result.get("variants", []) or [])),
            "case_count": len(list(result.get("cases", []) or [])),
            "run_count": len(runs),
            "passed_runs": sum(1 for run in runs if bool(run.get("passed", False))),
            "failed_runs": sum(1 for run in runs if not bool(run.get("passed", False))),
        },
        artifacts=result.get("artifacts", {}),
        recommended_queries=[
            {
                "name": "workflow_metadata",
                "description": "Controller-bench workflow metadata.",
                "sql": "SELECT * FROM workflow_metadata",
            },
            {
                "name": "controller_bench_runs",
                "description": "Pass/fail row for each variant/case run.",
                "sql": "SELECT variant_name, case_name, passed, failure_count, output_dir FROM bench_runs",
            },
            {
                "name": "controller_bench_leaderboard",
                "description": "Objective leaderboard rows.",
                "sql": "SELECT kind, objective, metric, rank, variant_name, value FROM bench_leaderboard",
            },
            {
                "name": "controller_bench_engineering",
                "description": "Run-level stack, cadence, authority, and stability evidence.",
                "sql": "SELECT * FROM bench_engineering",
            },
        ],
        recommended_review_order=[
            "Open controller_bench_summary.md for the narrative summary.",
            "Query bench_runs to confirm run-level pass/fail evidence.",
            "Query bench_leaderboard or bench_failures for objective-level diagnosis.",
        ],
        source_config=str(result.get("config_path", "") or ""),
        provenance={"writer": "sim.controller_lab.reporting.write_controller_bench_reports"},
        tables={
            "bench_runs": _controller_bench_run_rows(result),
            "bench_variant_summaries": _controller_bench_variant_rows(result),
            "bench_leaderboard": leaderboard_rows,
            "bench_failures": _controller_bench_failure_rows(result),
            "bench_engineering": engineering_runs,
        },
    )
    result.setdefault("artifacts", {}).update(review_outputs)
    index_path = write_output_index(
        outdir=outdir,
        workflow="controller_bench",
        title=str(result.get("suite_name", "controller_bench")),
        payload=result,
        artifacts=dict(result.get("artifacts", {}) or {}),
    )
    result.setdefault("artifacts", {})["output_index_md"] = str(index_path)
    summary_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    lines = [*narrative_lines, "", "## Artifacts", ""]
    for artifact_id, artifact_path in sorted(dict(result.get("artifacts", {}) or {}).items()):
        if isinstance(artifact_path, str):
            lines.append(f"- {artifact_id.replace('_', ' ').title()}: `{artifact_path}`")
    summary_md.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    try:
        from sim.controller_lab.engineering import write_evidence_manifest

        result.setdefault("artifacts", {}).update(write_evidence_manifest(result, outdir))
    except Exception as exc:
        result.setdefault("evidence_manifest_error", str(exc))
    try:
        from sim.controller_lab.agent import write_controller_bench_agent_packet

        packet_outputs = write_controller_bench_agent_packet(result, outdir)
        result.setdefault("artifacts", {}).update(packet_outputs)
    except Exception as exc:
        result.setdefault("agent_packet_error", str(exc))
    for review_key in ("workflow_manifest_json", "sqlite", "schema_json", "saved_views_json"):
        result.setdefault("artifacts", {}).pop(review_key, None)
    index_path = write_output_index(
        outdir=outdir,
        workflow="controller_bench",
        title=str(result.get("suite_name", "controller_bench")),
        payload=result,
        artifacts=dict(result.get("artifacts", {}) or {}),
    )
    result.setdefault("artifacts", {})["output_index_md"] = str(index_path)
    lines = [*narrative_lines, "", "## Artifacts", ""]
    for artifact_id, artifact_path in sorted(dict(result.get("artifacts", {}) or {}).items()):
        if isinstance(artifact_path, str):
            lines.append(f"- {artifact_id.replace('_', ' ').title()}: `{artifact_path}`")
    summary_md.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    summary_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    final_review_outputs = write_workflow_review(
        output_dir=outdir,
        workflow_type="controller_bench",
        title=str(result.get("suite_name", "controller_bench")),
        scenario_name=str(result.get("suite_name", "controller_bench")),
        status="complete",
        summary={
            "variant_count": len(list(result.get("variants", []) or [])),
            "case_count": len(list(result.get("cases", []) or [])),
            "run_count": len(runs),
            "passed_runs": sum(1 for run in runs if bool(run.get("passed", False))),
            "failed_runs": sum(1 for run in runs if not bool(run.get("passed", False))),
        },
        artifacts=dict(result.get("artifacts", {}) or {}),
        recommended_queries=[
            {"name": "workflow_metadata", "description": "Controller-bench workflow metadata.", "sql": "SELECT * FROM workflow_metadata"},
            {
                "name": "controller_bench_runs",
                "description": "Pass/fail row for each variant/case run.",
                "sql": "SELECT variant_name, case_name, passed, failure_count, output_dir FROM bench_runs",
            },
            {
                "name": "controller_bench_leaderboard",
                "description": "Objective leaderboard rows.",
                "sql": "SELECT kind, objective, metric, rank, variant_name, value FROM bench_leaderboard",
            },
            {
                "name": "controller_bench_engineering",
                "description": "Run-level stack, cadence, authority, and stability evidence.",
                "sql": "SELECT * FROM bench_engineering",
            },
        ],
        recommended_review_order=[
            "Open controller_bench_summary.md for the narrative summary.",
            "Query bench_runs to confirm run-level pass/fail evidence.",
            "Query bench_leaderboard or bench_failures for objective-level diagnosis.",
        ],
        source_config=str(result.get("config_path", "") or ""),
        provenance={"writer": "sim.controller_lab.reporting.write_controller_bench_reports"},
        tables={
            "bench_runs": _controller_bench_run_rows(result),
            "bench_variant_summaries": _controller_bench_variant_rows(result),
            "bench_leaderboard": leaderboard_rows,
            "bench_failures": _controller_bench_failure_rows(result),
            "bench_engineering": engineering_runs,
        },
    )
    result.setdefault("artifacts", {}).update(final_review_outputs)
    return artifact_paths


def _controller_bench_run_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in list(result.get("runs", []) or []):
        artifacts = dict(run.get("artifacts", {}) or {})
        failures = list(dict(run.get("failure_analysis", {}) or {}).get("failures", []) or [])
        rows.append(
            {
                "variant_name": run.get("variant_name"),
                "case_name": run.get("case_name"),
                "passed": bool(run.get("passed", False)),
                "failure_count": len(failures),
                "output_dir": artifacts.get("output_dir"),
            }
        )
    return rows


def _controller_bench_variant_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for summary in list(result.get("variant_summaries", []) or []):
        row = dict(summary or {})
        row["metric_means_json"] = json.dumps(row.pop("metric_means", {}) or {}, sort_keys=True)
        row["objective_pass_rates_json"] = json.dumps(row.pop("objective_pass_rates", {}) or {}, sort_keys=True)
        rows.append(row)
    return rows


def _controller_bench_failure_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in list(result.get("runs", []) or []):
        failure_analysis = dict(run.get("failure_analysis", {}) or {})
        artifacts = dict(run.get("artifacts", {}) or {})
        failures = list(failure_analysis.get("failures", []) or [])
        if not failures and not bool(run.get("passed", False)):
            failures = [{"reason": "unknown"}]
        for failure_raw in failures:
            failure = dict(failure_raw or {})
            rows.append(
                {
                    "variant_name": run.get("variant_name"),
                    "case_name": run.get("case_name"),
                    "objective": failure.get("objective"),
                    "objective_kind": failure.get("objective_kind"),
                    "metric": failure.get("metric"),
                    "op": failure.get("op"),
                    "threshold": failure.get("threshold"),
                    "value": failure.get("value"),
                    "reason": failure.get("reason"),
                    "failure_mode": failure.get("failure_mode"),
                    "diagnosis": failure.get("diagnosis"),
                    "suggestion": failure.get("suggestion"),
                    "margin": failure.get("margin"),
                    "termination_reason": failure_analysis.get("termination_reason"),
                    "output_dir": artifacts.get("output_dir"),
                }
            )
    return rows


def _controller_bench_engineering_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in list(result.get("runs", []) or []):
        engineering = dict(run.get("engineering", {}) or {})
        if not engineering:
            continue
        conformance = dict(engineering.get("stack_conformance", {}) or {})
        authority = dict(engineering.get("authority_analysis", {}) or {})
        cadence = dict(engineering.get("cadence_analysis", {}) or {})
        stability = dict(engineering.get("stability_response", {}) or {})
        identification = dict(stability.get("local_linear_identification", {}) or {})
        observed = dict(cadence.get("observed", {}) or {})
        rows.append(
            {
                "variant_name": run.get("variant_name"),
                "case_name": run.get("case_name"),
                "stack_conformance_passed": conformance.get("passed"),
                "actuator_saturation_fraction": authority.get("saturation_fraction"),
                "max_force_realization_error_n": authority.get("max_force_realization_error_n"),
                "identified_spectral_radius": identification.get("spectral_radius"),
                "max_nominal_release_timing_error_s": observed.get("max_nominal_release_timing_error_s"),
                "missed_nominal_release_count": observed.get("missed_nominal_release_count"),
            }
        )
    return rows


def _report_number(value: Any) -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{float(value):.6g}"
    return ""
