from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from sim.plotting.quality import STRICT_AGENT_PLOT_QUALITY, PlotQualityReport, apply_plot_quality_policy
from sim.plotting.style import add_artifact_footer, artifact_metadata, oel_plot_context, save_oel_figure
from sim.review.plot_recipes import REVIEW_PLOT_RECIPES, ReviewPlotRecipe
from sim.review.queries import get_saved_review_query
from sim.review.workspace import ReviewQueryError, ReviewQueryResult, ReviewWorkspace
from sim.runtime_environment import configure_headless_runtime

PLOT_TYPES = ("line", "scatter", "bar", "histogram", "heatmap")
STYLE_NAMES = ("oel_dark", "oel_light")


@dataclass(frozen=True)
class ReviewPlotSpec:
    sql: str
    x_column: str
    y_columns: list[str]
    plot_type: str = "line"
    group_column: str = ""
    style_name: str = "oel_dark"
    title: str = ""
    subtitle: str = ""
    x_label: str = ""
    y_label: str = ""
    artifact_id: str = ""
    file_format: str = "png"
    dpi: int = 150
    max_rows: int = 5000
    renderer_id: str = "generic"
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ReviewPlotArtifact:
    artifact_id: str
    path: Path
    relative_path: str
    row_count: int
    truncated: bool
    spec: ReviewPlotSpec
    qa: dict[str, Any] = field(default_factory=dict)


EvidencePlotRecipe = ReviewPlotRecipe
EVIDENCE_PLOT_RECIPES = REVIEW_PLOT_RECIPES


class EvidencePlotter:
    """Agent-friendly OEL-styled plotting API for completed review stores."""

    def __init__(self, workspace: ReviewWorkspace | str | Path) -> None:
        self.workspace = workspace if isinstance(workspace, ReviewWorkspace) else ReviewWorkspace.open(workspace)

    def line(self, *, sql: str, x: str, y: str | Sequence[str], **kwargs: Any) -> ReviewPlotArtifact:
        return self.plot(sql=sql, x=x, y=y, plot_type="line", **kwargs)

    def scatter(self, *, sql: str, x: str, y: str | Sequence[str], **kwargs: Any) -> ReviewPlotArtifact:
        return self.plot(sql=sql, x=x, y=y, plot_type="scatter", **kwargs)

    def bar(self, *, sql: str, x: str, y: str | Sequence[str], **kwargs: Any) -> ReviewPlotArtifact:
        return self.plot(sql=sql, x=x, y=y, plot_type="bar", **kwargs)

    def histogram(self, *, sql: str, y: str | Sequence[str], x: str = "", **kwargs: Any) -> ReviewPlotArtifact:
        return self.plot(sql=sql, x=x, y=y, plot_type="histogram", **kwargs)

    def heatmap(self, *, sql: str, x: str, y: str, value: str, **kwargs: Any) -> ReviewPlotArtifact:
        y_columns = [value]
        return self.plot(sql=sql, x=x, y=y_columns, group=y, plot_type="heatmap", **kwargs)

    def table(
        self,
        table: str,
        *,
        x: str = "",
        y: str | Sequence[str] | None = None,
        limit: int = 1000,
        **kwargs: Any,
    ) -> ReviewPlotArtifact:
        sql = f"SELECT * FROM {_quote_identifier(table)} LIMIT {max(int(limit), 1)}"
        return self.auto(sql=sql, x=x, y=y, **kwargs)

    def saved_query(
        self,
        name: str,
        *,
        x: str = "",
        y: str | Sequence[str] | None = None,
        **kwargs: Any,
    ) -> ReviewPlotArtifact:
        saved = get_saved_review_query(name)
        if saved is None:
            raise ValueError(f"Unknown saved review query '{name}'.")
        return self.auto(sql=saved.sql, x=x, y=y, title=kwargs.pop("title", saved.description), **kwargs)

    def recipe(self, recipe_id: str, **kwargs: Any) -> ReviewPlotArtifact:
        recipe = EVIDENCE_PLOT_RECIPES.get(recipe_id)
        if recipe is None:
            raise ValueError(
                f"Unknown evidence plot recipe '{recipe_id}'. "
                f"Available recipes: {', '.join(sorted(EVIDENCE_PLOT_RECIPES))}."
            )
        if "planetary_environment" in self.workspace.tables() and "planetary_state" not in recipe.required_tables:
            raise ValueError(f"Recipe {recipe_id!r} cannot consume planetary evidence; select a planetary recipe or explicit neutral SQL")
        self._require_tables(recipe.required_tables, recipe_id=recipe.recipe_id)
        return self.plot(
            sql=recipe.sql,
            x=recipe.x_column,
            y=list(recipe.y_columns),
            group=kwargs.pop("group", recipe.group_column),
            plot_type=kwargs.pop("plot_type", recipe.plot_type),
            title=kwargs.pop("title", recipe.title),
            x_label=kwargs.pop("x_label", recipe.x_label),
            y_label=kwargs.pop("y_label", recipe.y_label),
            artifact_id=kwargs.pop("artifact_id", recipe.artifact_id),
            renderer_id=kwargs.pop("renderer_id", recipe.renderer_id),
            extra={
                **{"recipe_id": recipe.recipe_id, "recipe_version": recipe.recipe_version},
                **dict(kwargs.pop("extra", {}) or {}),
            },
            **kwargs,
        )

    def relative_range(self, **kwargs: Any) -> ReviewPlotArtifact:
        return self.recipe("relative_range", **kwargs)

    def relative_range_rate(self, **kwargs: Any) -> ReviewPlotArtifact:
        return self.recipe("relative_range_rate", **kwargs)

    def relative_velocity_components(self, **kwargs: Any) -> ReviewPlotArtifact:
        return self.recipe("relative_velocity_components", **kwargs)

    def relative_position_ric_2d(self, **kwargs: Any) -> ReviewPlotArtifact:
        return self.recipe("relative_position_ric_2d", **kwargs)

    def burn_activity(self, **kwargs: Any) -> ReviewPlotArtifact:
        return self.recipe("burn_activity", **kwargs)

    def ground_access(self, **kwargs: Any) -> ReviewPlotArtifact:
        return self.recipe("ground_access", **kwargs)

    def auto(
        self,
        *,
        sql: str,
        x: str = "",
        y: str | Sequence[str] | None = None,
        plot_type: str = "line",
        **kwargs: Any,
    ) -> ReviewPlotArtifact:
        result = self.workspace.query(sql, max_rows=max(int(kwargs.get("max_rows", 5000)), 1))
        spec = default_plot_spec(sql, result, artifact_id=str(kwargs.pop("artifact_id", "") or ""))
        if x:
            spec = _replace_spec(spec, x_column=x)
        if y is not None:
            spec = _replace_spec(spec, y_columns=_coerce_y_columns(y))
        output = kwargs.pop("output", None)
        # ``result`` already contains the bounded, authorized query output
        # used to infer the automatic specification.  Reuse it during the
        # render so automatic plotting does not execute an expensive source
        # query a second time.
        spec = _replace_spec(spec, plot_type=plot_type, **_spec_kwargs(kwargs))
        return _save_review_plot_from_result(
            self.workspace,
            spec,
            result=result,
            path=_resolve_output_path(self.workspace, output),
        )

    def plot(
        self,
        *,
        sql: str,
        x: str,
        y: str | Sequence[str],
        plot_type: str = "line",
        group: str = "",
        style: str = "oel_dark",
        title: str = "",
        subtitle: str = "",
        x_label: str = "",
        y_label: str = "",
        artifact_id: str = "",
        file_format: str = "png",
        dpi: int = 150,
        max_rows: int = 5000,
        output: str | Path | None = None,
        renderer_id: str = "generic",
        extra: dict[str, Any] | None = None,
    ) -> ReviewPlotArtifact:
        spec = ReviewPlotSpec(
            sql=sql,
            x_column=x,
            y_columns=_coerce_y_columns(y),
            plot_type=plot_type,
            group_column=group,
            style_name=_normalize_style_alias(style),
            title=title,
            subtitle=subtitle,
            x_label=x_label,
            y_label=y_label,
            artifact_id=artifact_id,
            file_format=file_format,
            dpi=dpi,
            max_rows=max_rows,
            renderer_id=renderer_id,
            extra={"source": "oel_review_plot_api", **dict(extra or {})},
        )
        return self.save(spec, output=output)

    def save(self, spec: ReviewPlotSpec, *, output: str | Path | None = None) -> ReviewPlotArtifact:
        return save_review_plot(self.workspace, spec, path=_resolve_output_path(self.workspace, output))

    def preview(self, spec: ReviewPlotSpec, *, path: str | Path) -> ReviewPlotArtifact:
        return save_review_plot(self.workspace, spec, path=path, record=False)

    def dry_run(self, spec: ReviewPlotSpec) -> dict[str, Any]:
        result = self.workspace.query(spec.sql, max_rows=max(int(spec.max_rows), 1))
        return _dry_run_from_result(spec, result)

    def _require_tables(self, tables: Sequence[str], *, recipe_id: str) -> None:
        available = set(self.workspace.tables())
        missing = [table for table in tables if table not in available]
        if missing:
            raise ValueError(
                f"Recipe '{recipe_id}' requires missing review table(s): {', '.join(missing)}. "
                f"Available tables: {', '.join(sorted(available)) or '(none)'}."
            )


def numeric_columns(result: ReviewQueryResult) -> list[str]:
    numeric: list[str] = []
    for column in result.columns:
        saw_value = False
        for row in result.rows:
            value = row.get(column)
            if value is None:
                continue
            try:
                float(value)
            except (TypeError, ValueError):
                break
            saw_value = True
        else:
            if saw_value:
                numeric.append(column)
    return numeric


def categorical_columns(result: ReviewQueryResult) -> list[str]:
    numeric = set(numeric_columns(result))
    return [column for column in result.columns if column not in numeric]


def default_plot_spec(sql: str, result: ReviewQueryResult, *, artifact_id: str = "") -> ReviewPlotSpec:
    numeric = numeric_columns(result)
    if len(numeric) >= 2:
        x_column = "time_s" if "time_s" in numeric else numeric[0]
        y_columns = [column for column in numeric if column != x_column][:1]
    elif len(numeric) == 1:
        x_column = result.columns[0] if result.columns else numeric[0]
        y_columns = numeric
    else:
        x_column = result.columns[0] if result.columns else ""
        y_columns = []
    return ReviewPlotSpec(
        sql=sql,
        x_column=x_column,
        y_columns=y_columns,
        title=_default_title(y_columns, x_column),
        artifact_id=artifact_id,
    )


def save_review_plot(
    workspace: ReviewWorkspace,
    spec: ReviewPlotSpec,
    *,
    path: str | Path | None = None,
    record: bool = True,
) -> ReviewPlotArtifact:
    result = workspace.query(spec.sql, max_rows=max(int(spec.max_rows), 1))
    return _save_review_plot_from_result(workspace, spec, result=result, path=path, record=record)


def _dry_run_from_result(spec: ReviewPlotSpec, result: ReviewQueryResult) -> dict[str, Any]:
    _validate_plot_spec(spec, result)
    return {
        "spec": _spec_to_dict(spec),
        "columns": result.columns,
        "row_count": result.row_count,
        "truncated": result.truncated,
        "numeric_columns": numeric_columns(result),
    }


def _save_review_plot_from_result(
    workspace: ReviewWorkspace,
    spec: ReviewPlotSpec,
    *,
    result: ReviewQueryResult,
    path: str | Path | None = None,
    record: bool = True,
) -> ReviewPlotArtifact:
    _validate_plot_spec(spec, result)
    spec = _replace_spec(spec, style_name=_normalize_style_alias(spec.style_name))
    artifact_id = _normalize_artifact_id(spec.artifact_id or _default_artifact_id(spec))
    extension = _normalize_extension(spec.file_format)
    if path is None:
        figures_dir = workspace.output_dir / "review" / "figures"
        out_path = _unique_path(figures_dir / f"{artifact_id}.{extension}")
        artifact_id = _normalize_artifact_id(out_path.stem)
        spec = _replace_spec(spec, artifact_id=artifact_id)
    else:
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
    scenario_name = _scenario_name(workspace)
    plot_metadata = artifact_metadata(scenario_name=scenario_name, artifact_id=artifact_id)

    _ensure_matplotlib_cache_env()
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    with oel_plot_context(
        style_name=_normalize_style(spec.style_name),
        metadata=plot_metadata,
    ):
        if spec.renderer_id == "ric_rectangular_2d":
            fig = _draw_ric_rectangular_2d(plt, result, spec)
        elif spec.renderer_id == "directed_link_margin":
            fig = _draw_directed_link_margin(plt, result, spec)
        elif spec.renderer_id == "generic":
            fig, ax = plt.subplots(figsize=(9.5, 5.25))
            _draw_plot(ax, result, spec)
            ax.set_title(spec.title or _default_title(spec.y_columns, spec.x_column))
            if spec.subtitle:
                fig.suptitle(spec.subtitle, y=0.975, fontsize=9)
            ax.set_xlabel(spec.x_label or spec.x_column)
            ax.set_ylabel(spec.y_label or _default_y_label(spec.y_columns))
            ax.grid(True, alpha=0.35)
            if _needs_legend(spec):
                ax.legend(loc="best")
            fig.tight_layout(rect=(0, 0.025, 1, 0.96 if spec.subtitle else 1))
            if str(spec.extra.get("recipe_id", "")).startswith("planetary_"):
                from matplotlib.ticker import MaxNLocator

                ax.yaxis.set_major_locator(MaxNLocator(nbins=7, prune="both"))
                fig.subplots_adjust(left=max(fig.subplotpars.left, 0.16))
        else:
            raise ValueError(f"Unsupported review plot renderer_id '{spec.renderer_id}'.")
        add_artifact_footer(fig, metadata=plot_metadata, artifact_id=artifact_id)
        presentation_qa = apply_plot_quality_policy(fig, policy=STRICT_AGENT_PLOT_QUALITY)
        save_oel_figure(
            fig,
            out_path,
            dpi=int(spec.dpi),
            metadata=plot_metadata,
            artifact_id=artifact_id,
            style_name=_normalize_style(spec.style_name),
            bbox_inches="tight",
        )
        plt.close(fig)

    qa = automated_plot_qa(out_path, result=result, spec=spec, presentation_qa=presentation_qa)
    artifact = ReviewPlotArtifact(
        artifact_id=artifact_id,
        path=out_path,
        relative_path=_relative_to_output(workspace, out_path),
        row_count=result.row_count,
        truncated=result.truncated,
        spec=spec,
        qa=qa,
    )
    if record:
        record_generated_artifact(workspace, artifact)
    return artifact


def record_generated_artifact(
    workspace: ReviewWorkspace,
    artifact: ReviewPlotArtifact,
    *,
    review_store_identity: dict[str, Any] | None = None,
) -> None:
    index_path = workspace.output_dir / "review" / "generated_artifacts.json"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    existing: list[dict[str, Any]] = []
    if index_path.exists():
        try:
            data = json.loads(index_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                existing = [dict(item) for item in list(data.get("artifacts", []) or []) if isinstance(item, dict)]
        except Exception:
            existing = []
    spec = artifact.spec
    try:
        metadata = workspace.query(
            "SELECT oel_version, review_schema_version FROM run_metadata LIMIT 1",
            max_rows=1,
        )
    except ReviewQueryError:
        version_row = {}
    else:
        version_row = metadata.rows[0] if metadata.rows else {}
    existing.append(
        {
            "artifact_id": artifact.artifact_id,
            "artifact_type": "figure",
            "path": artifact.relative_path,
            "created_utc": os.environ.get("OEL_GENERATED_UTC", "").strip()
            or datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "source": str(dict(spec.extra or {}).get("source", "oel_review_plot_api") or "oel_review_plot_api"),
            "source_query": spec.sql,
            "plot_type": spec.plot_type,
            "renderer_id": spec.renderer_id,
            "style_name": _normalize_style(spec.style_name),
            "x_column": spec.x_column,
            "y_columns": list(spec.y_columns),
            "group_column": spec.group_column,
            "title": spec.title,
            "subtitle": spec.subtitle,
            "x_label": spec.x_label,
            "y_label": spec.y_label,
            "row_count": artifact.row_count,
            "max_rows": spec.max_rows,
            "truncated": artifact.truncated,
            "oel_version": str(version_row.get("oel_version") or "unknown"),
            "review_schema_version": str(
                version_row.get("review_schema_version") or "unknown"
            ),
            "query_sha256": hashlib.sha256(spec.sql.encode("utf-8")).hexdigest(),
            "review_store": review_store_identity or _review_store_identity(workspace),
            "qa": dict(artifact.qa),
            "extra": dict(spec.extra or {}),
        }
    )
    index_path.write_text(json.dumps({"artifacts": existing}, indent=2) + "\n", encoding="utf-8")
    _refresh_output_index_generated_artifacts(workspace.output_dir, existing)


def _refresh_output_index_generated_artifacts(
    output_dir: Path,
    artifacts: list[dict[str, Any]],
) -> None:
    index_path = output_dir / "index.md"
    if not index_path.is_file():
        return
    start = "<!-- OEL_REVIEW_GENERATED_ARTIFACTS_START -->"
    end = "<!-- OEL_REVIEW_GENERATED_ARTIFACTS_END -->"
    rows = [start, "## Review-Generated Artifacts", ""]
    for artifact in artifacts:
        relative = str(artifact.get("path", "") or "").strip()
        if not relative:
            continue
        artifact_id = str(artifact.get("artifact_id", "review figure") or "review figure")
        rows.append(f"- `{artifact_id}`: [`{relative}`]({relative})")
    rows.extend([end, ""])
    section = "\n".join(rows)
    text = index_path.read_text(encoding="utf-8")
    if start in text and end in text:
        prefix, remainder = text.split(start, 1)
        _old, suffix = remainder.split(end, 1)
        updated = prefix.rstrip() + "\n\n" + section + suffix.lstrip("\n")
    else:
        updated = text.rstrip() + "\n\n" + section
    index_path.write_text(updated, encoding="utf-8")


def _review_store_identity(workspace: ReviewWorkspace) -> dict[str, Any]:
    return workspace.evidence_identity()


def automated_plot_qa(
    path: Path,
    *,
    result: ReviewQueryResult,
    spec: ReviewPlotSpec,
    presentation_qa: PlotQualityReport | None = None,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    exists = path.is_file()
    size_bytes = int(path.stat().st_size) if exists else 0
    checks.append({"check_id": "artifact_exists", "passed": exists, "value": exists})
    checks.append({"check_id": "artifact_nonempty", "passed": size_bytes >= 1000, "value": size_bytes})
    checks.append({"check_id": "query_has_rows", "passed": result.row_count > 0, "value": result.row_count})
    checks.append({"check_id": "query_not_truncated", "passed": not result.truncated, "value": result.truncated})

    if exists and path.suffix.lower() == ".png":
        try:
            from PIL import Image, ImageStat

            with Image.open(path) as image:
                width, height = image.size
                grayscale = image.convert("L")
                variation = float(ImageStat.Stat(grayscale).stddev[0])
            checks.append(
                {
                    "check_id": "image_dimensions",
                    "passed": width >= 600 and height >= 300,
                    "value": {"width": width, "height": height},
                }
            )
            checks.append({"check_id": "image_not_blank", "passed": variation >= 2.0, "value": variation})
        except Exception as exc:
            checks.append(
                {
                    "check_id": "image_decode",
                    "passed": False,
                    "value": type(exc).__name__,
                }
            )

    presentation = presentation_qa.to_dict() if presentation_qa is not None else {}
    if presentation:
        presentation_passed = presentation.get("automated_status") == "passed"
        checks.append(
            {
                "check_id": "presentation_quality",
                "passed": presentation_passed,
                "value": {
                    "policy_id": presentation.get("policy_id"),
                    "policy_version": presentation.get("policy_version"),
                    "failed_checks": list(presentation.get("failed_checks", []) or []),
                    "repairs": list(presentation.get("repairs", []) or []),
                },
            }
        )
    failed = [str(check["check_id"]) for check in checks if not bool(check["passed"])]
    if presentation.get("automated_status") == "failed":
        failed.extend(str(item) for item in list(presentation.get("failed_checks", []) or []))
    failed = list(dict.fromkeys(failed))
    return {
        "automated_status": "passed" if not failed else "failed",
        "checks": checks,
        "failed_checks": failed,
        "visual_qa_status": "pending_agent_review",
        "visual_review_required": True,
        "non_claim": "Automated checks do not replace agent visual inspection.",
        "renderer_id": spec.renderer_id,
        "presentation_quality": presentation,
    }


def _draw_ric_rectangular_2d(plt: Any, result: ReviewQueryResult, spec: ReviewPlotSpec) -> Any:
    required = {"r_radial_km", "i_intrack_km", "c_crosstrack_km"}
    missing = sorted(required - set(result.columns))
    if missing:
        raise ValueError(
            "The rectangular-RIC renderer requires query columns: " + ", ".join(sorted(required))
        )
    group_column = spec.group_column if spec.group_column in result.columns else ""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in result.rows:
        group = str(row.get(group_column) or "relative trajectory") if group_column else "relative trajectory"
        grouped.setdefault(group, []).append(row)

    planes = (
        ("i_intrack_km", "r_radial_km", "I-R Projection", "In-track, I (km)", "Radial, R (km)"),
        ("i_intrack_km", "c_crosstrack_km", "I-C Projection", "In-track, I (km)", "Cross-track, C (km)"),
        ("c_crosstrack_km", "r_radial_km", "C-R Projection", "Cross-track, C (km)", "Radial, R (km)"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.8))
    palette = ("#38BDF8", "#F97316", "#A78BFA", "#22C55E", "#EAB308", "#EC4899")
    for axis, (x_column, y_column, title, x_label, y_label) in zip(axes, planes, strict=True):
        for index, (group, rows) in enumerate(sorted(grouped.items())):
            points = [
                (float(row[x_column]), float(row[y_column]))
                for row in rows
                if row.get(x_column) is not None and row.get(y_column) is not None
            ]
            if not points:
                continue
            x_values = [point[0] for point in points]
            y_values = [point[1] for point in points]
            color = palette[index % len(palette)]
            axis.plot(x_values, y_values, linewidth=1.8, color=color, label=group, zorder=3)
            axis.scatter(
                [x_values[0]],
                [y_values[0]],
                marker="o",
                s=44,
                color="#22C55E",
                edgecolors="#0F172A",
                linewidths=0.7,
                label="start" if index == 0 else "_nolegend_",
                zorder=6,
            )
            axis.scatter(
                [x_values[-1]],
                [y_values[-1]],
                marker="X",
                s=56,
                color="#F97316",
                edgecolors="#0F172A",
                linewidths=0.7,
                label="end" if index == 0 else "_nolegend_",
                zorder=7,
            )
        axis.scatter(
            [0.0],
            [0.0],
            marker="*",
            s=95,
            color="#F8FAFC",
            edgecolors="#111827",
            linewidths=0.8,
            label="chief origin",
            zorder=8,
        )
        axis.axhline(0.0, color="#94A3B8", linewidth=0.8, alpha=0.5, zorder=1)
        axis.axvline(0.0, color="#94A3B8", linewidth=0.8, alpha=0.5, zorder=1)
        axis.set_title(title)
        axis.set_xlabel(x_label)
        axis.set_ylabel(y_label)
        axis.set_aspect("equal", adjustable="datalim")
        axis.margins(0.08)
        axis.grid(True, alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=min(5, len(handles)), frameon=False)
    fig.suptitle(spec.title or "Relative Trajectory in Rectangular RIC", y=0.98)
    if spec.subtitle:
        fig.text(0.5, 0.925, spec.subtitle, ha="center", va="top", fontsize=9)
    fig.tight_layout(rect=(0.0, 0.11, 1.0, 0.91 if spec.subtitle else 0.94))
    return fig


def _draw_directed_link_margin(plt: Any, result: ReviewQueryResult, spec: ReviewPlotSpec) -> Any:
    required = {"analysis_id", "time_s", "margin_db", "available"}
    missing = sorted(required - set(result.columns))
    if missing:
        raise ValueError(
            "The directed-link margin renderer requires query columns: "
            + ", ".join(sorted(required))
        )
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in result.rows:
        grouped.setdefault(str(row.get("analysis_id") or "directed link"), []).append(row)
    fig, ax = plt.subplots(figsize=(9.5, 5.25))
    palette = ("#38BDF8", "#F97316", "#A78BFA", "#22C55E", "#EAB308", "#EC4899")
    for index, (group, rows) in enumerate(sorted(grouped.items())):
        points = [
            (float(row["time_s"]), float(row["margin_db"]), bool(row["available"]))
            for row in rows
            if row.get("time_s") is not None and row.get("margin_db") is not None
        ]
        if not points:
            continue
        x_values = np.asarray([point[0] for point in points], dtype=float)
        y_values = np.asarray([point[1] for point in points], dtype=float)
        available = np.asarray([point[2] for point in points], dtype=bool)
        color = palette[index % len(palette)]
        ax.plot(x_values, y_values, color=color, linewidth=1.8, label=group, zorder=3)
        ax.fill_between(
            x_values,
            y_values,
            0.0,
            where=available,
            color=color,
            alpha=0.16,
            interpolate=False,
            label="RF-qualified samples" if index == 0 else "_nolegend_",
            zorder=2,
        )
    ax.axhline(0.0, color="#F59E0B", linewidth=1.2, linestyle="--", label="Closure threshold")
    ax.set_title(spec.title or "Directed-link margin")
    if spec.subtitle:
        fig.suptitle(spec.subtitle, y=0.975, fontsize=9)
    ax.set_xlabel(spec.x_label or "Analysis time (s)")
    ax.set_ylabel(spec.y_label or "Margin (dB)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout(rect=(0, 0.025, 1, 0.96 if spec.subtitle else 1))
    return fig


def _draw_plot(ax: Any, result: ReviewQueryResult, spec: ReviewPlotSpec) -> None:
    plot_type = _normalize_plot_type(spec.plot_type)
    rows = result.rows
    if plot_type == "heatmap":
        _draw_heatmap(ax, rows, spec)
        return
    if spec.group_column:
        groups: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            groups.setdefault(str(row.get(spec.group_column) or "n/a"), []).append(row)
        for group, group_rows in groups.items():
            for y_column in spec.y_columns:
                label = group if len(spec.y_columns) == 1 else f"{group}:{y_column}"
                _draw_series(ax, group_rows, spec.x_column, y_column, plot_type, label=label)
        return
    if plot_type == "bar":
        _draw_bar(ax, rows, spec)
        return
    if plot_type == "histogram":
        _draw_histogram(ax, rows, spec)
        return
    for y_column in spec.y_columns:
        _draw_series(ax, rows, spec.x_column, y_column, plot_type, label=y_column if len(spec.y_columns) > 1 else "")


def _draw_series(
    ax: Any,
    rows: list[dict[str, Any]],
    x_column: str,
    y_column: str,
    plot_type: str,
    *,
    label: str = "",
) -> None:
    pairs = [
        (row.get(x_column), row.get(y_column))
        for row in rows
        if row.get(x_column) is not None and row.get(y_column) is not None
    ]
    x_values = [item[0] for item in pairs]
    y_values = [float(item[1]) for item in pairs]
    if plot_type == "scatter":
        ax.scatter(x_values, y_values, s=20, label=label or None)
    else:
        ax.plot(x_values, y_values, marker="o", markersize=3, label=label or None)


def _draw_bar(ax: Any, rows: list[dict[str, Any]], spec: ReviewPlotSpec) -> None:
    labels = [str(row.get(spec.x_column)) for row in rows]
    series_count = len(spec.y_columns)
    width = 0.8 / max(series_count, 1)
    centers = np.arange(len(labels), dtype=float)
    for series_index, y_column in enumerate(spec.y_columns):
        values = [float(row[y_column]) if row.get(y_column) is not None else np.nan for row in rows]
        offsets = centers - 0.4 + width * (series_index + 0.5)
        ax.bar(offsets, values, width=width, label=y_column if series_count > 1 else None)
    ax.set_xticks(centers, labels)
    if len(labels) > 8:
        ax.tick_params(axis="x", labelrotation=35)


def _draw_histogram(ax: Any, rows: list[dict[str, Any]], spec: ReviewPlotSpec) -> None:
    for y_column in spec.y_columns:
        values = [float(row.get(y_column)) for row in rows if row.get(y_column) is not None]
        ax.hist(values, bins="auto", alpha=0.72, label=y_column if len(spec.y_columns) > 1 else None)


def _draw_heatmap(ax: Any, rows: list[dict[str, Any]], spec: ReviewPlotSpec) -> None:
    x_column = spec.x_column
    y_column = spec.group_column
    value_column = spec.y_columns[0]
    x_labels = _ordered_unique(str(row.get(x_column)) for row in rows if row.get(x_column) is not None)
    y_labels = _ordered_unique(str(row.get(y_column)) for row in rows if row.get(y_column) is not None)
    x_index = {value: idx for idx, value in enumerate(x_labels)}
    y_index = {value: idx for idx, value in enumerate(y_labels)}
    matrix = [[float("nan") for _ in x_labels] for _ in y_labels]
    for row in rows:
        x_value = row.get(x_column)
        y_value = row.get(y_column)
        z_value = row.get(value_column)
        if x_value is None or y_value is None or z_value is None:
            continue
        matrix[y_index[str(y_value)]][x_index[str(x_value)]] = float(z_value)
    image = ax.imshow(matrix, aspect="auto")
    ax.set_xticks(range(len(x_labels)), x_labels)
    ax.set_yticks(range(len(y_labels)), y_labels)
    if len(x_labels) > 6:
        ax.tick_params(axis="x", labelrotation=35)
    ax.figure.colorbar(image, ax=ax, label=spec.y_label or value_column)


def _validate_plot_spec(spec: ReviewPlotSpec, result: ReviewQueryResult) -> None:
    if result.row_count <= 0:
        raise ValueError("The review query returned no rows to plot.")
    columns = set(result.columns)
    if spec.x_column not in columns:
        if _normalize_plot_type(spec.plot_type) != "histogram" or spec.x_column:
            raise ValueError(
                f"x_column '{spec.x_column}' is not in the query result. "
                f"Available columns: {', '.join(result.columns) or '(none)'}."
            )
    if not spec.y_columns:
        raise ValueError("At least one y_column is required.")
    for column in spec.y_columns:
        if column not in columns:
            raise ValueError(
                f"y_column '{column}' is not in the query result. "
                f"Available columns: {', '.join(result.columns) or '(none)'}."
            )
    if spec.group_column and spec.group_column not in columns:
        raise ValueError(
            f"group_column '{spec.group_column}' is not in the query result. "
            f"Available columns: {', '.join(result.columns) or '(none)'}."
        )
    numeric = set(numeric_columns(result))
    for column in spec.y_columns:
        if column not in numeric:
            raise ValueError(
                f"y_column '{column}' must contain numeric values. "
                f"Numeric columns: {', '.join(numeric_columns(result)) or '(none)'}."
            )
    if _normalize_plot_type(spec.plot_type) == "heatmap" and not spec.group_column:
        raise ValueError("Heatmap plots require a group_column for the y-axis category.")
    if spec.renderer_id == "ric_rectangular_2d":
        required = {"r_radial_km", "i_intrack_km", "c_crosstrack_km"}
        missing = sorted(required - columns)
        if missing:
            raise ValueError(
                "The rectangular-RIC renderer requires query columns: " + ", ".join(sorted(required))
            )
        nonnumeric = sorted(required - numeric)
        if nonnumeric:
            raise ValueError("Rectangular-RIC columns must contain numeric values: " + ", ".join(nonnumeric))
    elif spec.renderer_id == "directed_link_margin":
        required = {"analysis_id", "time_s", "margin_db", "available"}
        missing = sorted(required - columns)
        if missing:
            raise ValueError(
                "The directed-link margin renderer requires query columns: "
                + ", ".join(sorted(required))
            )
        nonnumeric = sorted({"time_s", "margin_db", "available"} - numeric)
        if nonnumeric:
            raise ValueError(
                "Directed-link margin numeric columns must contain numeric values: "
                + ", ".join(nonnumeric)
            )
    elif spec.renderer_id != "generic":
        raise ValueError(f"Unsupported review plot renderer_id '{spec.renderer_id}'.")
    _normalize_plot_type(spec.plot_type)
    _normalize_style(spec.style_name)
    _normalize_extension(spec.file_format)


def _scenario_name(workspace: ReviewWorkspace) -> str:
    for table in ("run_metadata", "workflow_metadata"):
        try:
            result = workspace.query(f"SELECT scenario_name FROM {table}", max_rows=1)
        except Exception:
            continue
        if result.rows:
            return str(result.rows[0].get("scenario_name") or "")
    return ""


def _default_title(y_columns: list[str], x_column: str) -> str:
    if not y_columns:
        return "Review Plot"
    if len(y_columns) == 1:
        return f"{y_columns[0]} vs {x_column}"
    return f"{', '.join(y_columns[:3])} vs {x_column}"


def _default_y_label(y_columns: list[str]) -> str:
    if len(y_columns) == 1:
        return y_columns[0]
    return "value"


def _default_artifact_id(spec: ReviewPlotSpec) -> str:
    return f"evidence_{spec.plot_type}_{spec.y_columns[0] if spec.y_columns else 'plot'}"


def _coerce_y_columns(value: str | Sequence[str]) -> list[str]:
    if isinstance(value, str):
        out = [value]
    else:
        out = [str(item) for item in value]
    return [item.strip() for item in out if item and item.strip()]


def _replace_spec(spec: ReviewPlotSpec, **kwargs: Any) -> ReviewPlotSpec:
    return replace(spec, **{key: value for key, value in kwargs.items() if value is not None})


def _spec_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    mapping = {
        "group": "group_column",
        "style": "style_name",
        "title": "title",
        "subtitle": "subtitle",
        "x_label": "x_label",
        "y_label": "y_label",
        "file_format": "file_format",
        "dpi": "dpi",
        "max_rows": "max_rows",
        "renderer_id": "renderer_id",
    }
    out: dict[str, Any] = {}
    if "extra" in kwargs:
        out["extra"] = {"source": "oel_review_plot_api", **dict(kwargs["extra"] or {})}
    for source, target in mapping.items():
        if source not in kwargs:
            continue
        value = kwargs[source]
        if target == "style_name":
            value = _normalize_style_alias(str(value))
        out[target] = value
    return out


def _resolve_output_path(workspace: ReviewWorkspace, output: str | Path | None) -> Path | None:
    if output is None or str(output).strip() == "":
        return None
    path = Path(output)
    if path.is_absolute():
        return path
    return workspace.output_dir / path


def _spec_to_dict(spec: ReviewPlotSpec) -> dict[str, Any]:
    return {
        "sql": spec.sql,
        "x_column": spec.x_column,
        "y_columns": list(spec.y_columns),
        "plot_type": spec.plot_type,
        "group_column": spec.group_column,
        "style_name": spec.style_name,
        "title": spec.title,
        "subtitle": spec.subtitle,
        "x_label": spec.x_label,
        "y_label": spec.y_label,
        "artifact_id": spec.artifact_id,
        "file_format": spec.file_format,
        "dpi": spec.dpi,
        "max_rows": spec.max_rows,
        "renderer_id": spec.renderer_id,
        "extra": dict(spec.extra or {}),
    }


def _quote_identifier(value: str) -> str:
    return '"' + str(value).replace('"', '""') + '"'


def _ordered_unique(values: Sequence[str] | Any) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        text = str(value)
        if text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _normalize_artifact_id(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "").strip()).strip("._-")
    return cleaned[:80] or f"evidence_plot_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"


def _normalize_plot_type(value: str) -> str:
    plot_type = str(value or "line").strip().lower()
    if plot_type not in PLOT_TYPES:
        raise ValueError(f"Unsupported plot type '{value}'. Valid plot types: {', '.join(PLOT_TYPES)}.")
    return plot_type


def _normalize_style(value: str) -> str:
    style = str(value or "oel_dark").strip().lower()
    if style not in STYLE_NAMES:
        raise ValueError(f"Unsupported style '{value}'. Valid styles: {', '.join(STYLE_NAMES)}.")
    return style


def _normalize_style_alias(value: str) -> str:
    style = str(value or "oel_dark").strip().lower()
    if style == "dark":
        return "oel_dark"
    if style == "light":
        return "oel_light"
    return style


def _normalize_extension(value: str) -> str:
    extension = str(value or "png").strip().lower().lstrip(".")
    if extension not in {"png", "svg", "pdf"}:
        raise ValueError("Review plot exports support png, svg, or pdf.")
    return extension


def _unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    for index in range(2, 1000):
        candidate = path.with_name(f"{path.stem}_{index}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise ValueError(f"Could not find an available artifact filename for {path.name}.")


def _relative_to_output(workspace: ReviewWorkspace, path: Path) -> str:
    try:
        return path.relative_to(workspace.output_dir).as_posix()
    except ValueError:
        return str(path)


def _needs_legend(spec: ReviewPlotSpec) -> bool:
    return _normalize_plot_type(spec.plot_type) != "heatmap" and (
        bool(spec.group_column) or len(spec.y_columns) > 1
    )


def _ensure_matplotlib_cache_env() -> None:
    status = configure_headless_runtime(force=True)
    if not status.ok:
        raise RuntimeError("Could not prepare headless plotting caches: " + "; ".join(status.errors))
