from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

from sim.review.plotting import (
    ReviewPlotArtifact,
    ReviewPlotSpec,
    _dry_run_from_result,
    _save_review_plot_from_result,
    record_generated_artifact,
)
from sim.review.workspace import ReviewWorkspace

REVIEW_PLOT_PLAN_SCHEMA_VERSION = 1


def plot_spec_from_mapping(arguments: Mapping[str, Any], *, source: str) -> ReviewPlotSpec:
    return ReviewPlotSpec(
        sql=str(arguments["sql"]),
        x_column=str(arguments.get("x_column", "") or ""),
        y_columns=[str(item) for item in list(arguments.get("y_columns", []) or [])],
        plot_type=str(arguments.get("plot_type", "line") or "line"),
        group_column=str(arguments.get("group_column", "") or ""),
        style_name=str(arguments.get("style", "oel_dark") or "oel_dark"),
        title=str(arguments.get("title", "") or ""),
        subtitle=str(arguments.get("subtitle", "") or ""),
        x_label=str(arguments.get("x_label", "") or ""),
        y_label=str(arguments.get("y_label", "") or ""),
        artifact_id=str(arguments.get("artifact_id", "") or ""),
        file_format=str(arguments.get("format", "png") or "png"),
        dpi=int(arguments.get("dpi", 150)),
        max_rows=int(arguments.get("max_rows", 5000)),
        renderer_id="generic",
        extra={"source": source, "plot_contract": "typed_review_plot_v2"},
    )


def plan_review_plot(
    output_dir: str | Path,
    spec: ReviewPlotSpec,
    *,
    protected: bool | None = None,
) -> dict[str, Any]:
    with (
        ReviewWorkspace.open(output_dir)
        if protected is None
        else ReviewWorkspace.open(output_dir, protected=protected)
    ) as workspace:
        dry_run = _dry_run(workspace, spec)
        plan_id = review_plot_plan_id(workspace, spec)
        warnings: list[str] = []
        if dry_run["truncated"]:
            warnings.append("The query reached max_rows; render evidence would be truncated.")
        return {
            "status": "planned",
            "output_dir": str(workspace.output_dir),
            "review_store": "review/run.sqlite",
            "plot_plan_id": plan_id,
            "spec": asdict(spec),
            "columns": dry_run["columns"],
            "numeric_columns": dry_run["numeric_columns"],
            "row_count": dry_run["row_count"],
            "truncated": dry_run["truncated"],
            "warnings": warnings,
            "render_authorized": False,
            "visual_review_required": True,
        }


def render_review_plot(
    output_dir: str | Path,
    spec: ReviewPlotSpec,
    *,
    plot_plan_id: str,
    path: str | Path,
    record: bool = True,
    writer: Any | None = None,
    logical_output_dir: str | Path | None = None,
    protected: bool | None = None,
) -> ReviewPlotArtifact:
    workspace = (
        ReviewWorkspace.open(output_dir)
        if protected is None
        else ReviewWorkspace.open(output_dir, protected=protected)
    )
    active_writer = writer or _current_output_writer()
    initial_identity = workspace.evidence_identity()
    current_id = review_plot_plan_id(workspace, spec, review_store_identity=initial_identity)
    if str(plot_plan_id) != current_id:
        raise ValueError("The plot_plan_id is stale or does not match the review store and plot specification.")
    result = workspace.query(spec.sql, max_rows=max(int(spec.max_rows), 1))
    dry_run = _dry_run_from_result(spec, result)
    if dry_run["truncated"]:
        raise ValueError(
            "The planned review query is truncated at max_rows; increase the bound or narrow the query."
        )
    artifact = _save_review_plot_from_result(
        workspace,
        spec,
        result=result,
        path=path,
        record=False,
        writer=writer,
        logical_output_dir=logical_output_dir,
    )
    final_identity = workspace.evidence_identity()
    if review_plot_plan_id(workspace, spec, review_store_identity=final_identity) != current_id:
        # Protected renders may already have been published through a
        # descriptor-bound writer.  A pathname unlink here could remove a
        # concurrent replacement at the same logical path; isolated staging
        # transactions own their cleanup instead.
        if active_writer is None:
            artifact.path.unlink(missing_ok=True)
        raise ValueError("The review store changed while the planned plot was rendering; no artifact was recorded.")
    if record:
        record_generated_artifact(
            workspace,
            artifact,
            review_store_identity=final_identity,
            writer=writer,
            logical_output_dir=logical_output_dir,
        )
    return artifact


def review_plot_plan_id(
    workspace: ReviewWorkspace,
    spec: ReviewPlotSpec,
    *,
    review_store_identity: dict[str, Any] | None = None,
) -> str:
    payload = {
        "schema_version": REVIEW_PLOT_PLAN_SCHEMA_VERSION,
        "review_store": review_store_identity or workspace.evidence_identity(),
        "spec": asdict(spec),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return "oel-review-plot-plan-v1:" + hashlib.sha256(encoded).hexdigest()


def _dry_run(workspace: ReviewWorkspace, spec: ReviewPlotSpec) -> dict[str, Any]:
    from sim.review.plotting import EvidencePlotter

    return EvidencePlotter(workspace).dry_run(spec)


def _current_output_writer() -> Any | None:
    try:
        from sim.security.output_writes import current_output_writer
    except ImportError:
        return None
    return current_output_writer()


__all__ = [
    "REVIEW_PLOT_PLAN_SCHEMA_VERSION",
    "plan_review_plot",
    "plot_spec_from_mapping",
    "render_review_plot",
    "review_plot_plan_id",
]
