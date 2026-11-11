# ruff: noqa: F401,F403,F405,I001
from pathlib import Path

from .campaign_common import *


def _monte_carlo_run_output_dir(
    run: dict[str, Any],
    summary: dict[str, Any],
    artifacts: dict[str, Any],
) -> str | None:
    for value in (
        run.get("output_dir"),
        summary.get("output_dir"),
        artifacts.get("output_dir"),
    ):
        text = str(value or "").strip()
        if text:
            return text

    output_index = str(summary.get("output_index_md") or "").strip()
    if output_index:
        return str(Path(output_index).parent)

    review_sqlite = str(summary.get("review_sqlite_path") or "").strip()
    if review_sqlite:
        path = Path(review_sqlite)
        if path.parent.name == "review":
            return str(path.parent.parent)
    return None


def _monte_carlo_review_run_rows(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in runs:
        summary = dict(run.get("summary", {}) or {})
        assessment = dict(run.get("assessment", {}) or {})
        artifacts = dict(summary.get("artifacts", {}) or {})
        rows.append(
            {
                "iteration": run.get("iteration"),
                "passed": assessment.get("pass", run.get("pass", summary.get("pass"))),
                "closest_approach_km": assessment.get(
                    "closest_approach_km",
                    run.get("closest_approach_km", summary.get("closest_approach_km")),
                ),
                "duration_s": assessment.get("duration_s", run.get("duration_s", summary.get("duration_s"))),
                "total_dv_m_s": assessment.get(
                    "total_dv_m_s_total",
                    run.get("total_dv_m_s_total", summary.get("total_dv_m_s_total")),
                ),
                "output_dir": _monte_carlo_run_output_dir(run, summary, artifacts),
                "sampled_parameters_json": json.dumps(dict(run.get("sampled_parameters", {}) or {}), sort_keys=True),
            }
        )
    return rows


def _monte_carlo_review_metric_rows(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in runs:
        iteration = run.get("iteration")
        summary = dict(run.get("summary", {}) or {})
        assessment = dict(run.get("assessment", {}) or {})
        metrics = dict(summary.get("metrics", {}) or {})
        aliases = {
            "closest_approach_km": "closest_approach_km",
            "duration_s": "duration_s",
            "total_dv_m_s": "total_dv_m_s_total",
        }
        for key, assessment_key in aliases.items():
            if assessment_key in assessment:
                metrics.setdefault(key, assessment.get(assessment_key))
            elif assessment_key in run:
                metrics.setdefault(key, run.get(assessment_key))
            elif assessment_key in summary:
                metrics.setdefault(key, summary.get(assessment_key))
        for name, value in sorted(metrics.items()):
            if isinstance(value, (dict, list, tuple)):
                value = json.dumps(value, sort_keys=True)
            rows.append({"iteration": iteration, "metric_name": name, "metric_value": value})
    return rows

__all__ = [name for name in globals() if not name.startswith("__")]
