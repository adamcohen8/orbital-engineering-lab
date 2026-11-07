"""Professional, evidence-bound rendering for supported review animations."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from sim.plotting.animation_quality import (
    animation_time_decimal_places,
    fixed_time_text_width,
    format_animation_time,
    save_animation_with_quality,
)
from sim.plotting.style import artifact_metadata, oel_plot_context
from sim.review.animation_planning import ReviewAnimationSpec, review_animation_frame_plan
from sim.review.animation_recipes import ReviewAnimationRecipe
from sim.review.plotting import _lexical_relative_path, _logical_artifact_path
from sim.review.workspace import ReviewQueryResult, ReviewWorkspace
from sim.runtime_environment import configure_headless_runtime
from sim.utils.figure_size import cap_figsize


@dataclass(frozen=True)
class ReviewAnimationArtifact:
    artifact_id: str
    path: Path
    relative_path: str
    contact_sheet_path: Path
    quality_receipt_path: Path
    row_count: int
    truncated: bool
    spec: ReviewAnimationSpec
    qa: dict[str, Any]
    # ``path`` and companion paths remain physical staging paths for encoder
    # QA and cleanup.  Logical paths are persisted in receipts and manifests.
    logical_path: Path | None = None
    logical_contact_sheet_path: Path | None = None
    logical_quality_receipt_path: Path | None = None


def render_review_animation_artifact(
    workspace: ReviewWorkspace,
    recipe: ReviewAnimationRecipe,
    result: ReviewQueryResult,
    spec: ReviewAnimationSpec,
    *,
    path: str | Path,
    record: bool = True,
    writer: Any | None = None,
    logical_output_dir: str | Path | None = None,
    _staging_call: bool = False,
) -> ReviewAnimationArtifact:
    if recipe.renderer_id != "ric_rectangular_2d":
        raise ValueError(f"Unsupported review animation renderer_id '{recipe.renderer_id}'.")
    active_writer = writer or _current_output_writer()
    if writer is None and active_writer is not None and not _staging_call:
        logical_target = Path(path)
        logical_contact_sheet = logical_target.with_suffix(".contact-sheet.png")
        logical_quality_receipt = logical_target.with_suffix(".quality.json")
        from sim.security.output_writes import staged_output_file

        with staged_output_file(logical_target) as staged_target:
            artifact = render_review_animation_artifact(
                workspace,
                recipe,
                result,
                spec,
                path=staged_target,
                record=False,
                logical_output_dir=logical_target.parent,
                _staging_call=True,
            )
            with staged_output_file(logical_contact_sheet) as staged_contact_sheet:
                shutil.copyfile(artifact.contact_sheet_path, staged_contact_sheet)
            with staged_output_file(logical_quality_receipt) as staged_quality_receipt:
                shutil.copyfile(artifact.quality_receipt_path, staged_quality_receipt)
        artifact = replace(
            artifact,
            path=logical_target,
            relative_path=_relative_to_output(workspace, logical_target),
            contact_sheet_path=logical_contact_sheet,
            quality_receipt_path=logical_quality_receipt,
            logical_path=logical_target,
            logical_contact_sheet_path=logical_contact_sheet,
            logical_quality_receipt_path=logical_quality_receipt,
        )
        if record:
            record_generated_animation(
                workspace,
                artifact,
                recipe=recipe,
                writer=active_writer,
                logical_output_dir=Path(active_writer.logical_root),
            )
        return artifact
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    logical_target = _logical_artifact_path(
        target,
        writer=writer,
        logical_output_dir=logical_output_dir,
    )
    logical_contact_sheet = logical_target.with_suffix(".contact-sheet.png")
    logical_quality_receipt = logical_target.with_suffix(".quality.json")
    frame_plan = review_animation_frame_plan(result, spec)
    frame_times = np.asarray(frame_plan["frame_times_s"], dtype=float)
    if frame_times.size < 2:
        raise ValueError("Review animation requires at least two distinct time samples.")
    grouped = _group_ric_evidence(result)
    if not grouped:
        raise ValueError("Review animation contains no finite rectangular-RIC evidence.")

    configure_headless_runtime()
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib import animation

    artifact_id = spec.artifact_id or recipe.artifact_id
    scenario_name = _scenario_name(workspace)
    metadata = artifact_metadata(scenario_name=scenario_name, artifact_id=artifact_id)
    palette = ("#38BDF8", "#F59E0B", "#A78BFA", "#34D399", "#FB7185")
    planes = (
        (1, 0, "I-R Projection", "In-track, I (km)", "Radial, R (km)"),
        (1, 2, "I-C Projection", "In-track, I (km)", "Cross-track, C (km)"),
        (2, 0, "C-R Projection", "Cross-track, C (km)", "Radial, R (km)"),
    )
    camera = _camera_plan(grouped, frame_times, planes=planes, camera_policy=spec.camera_policy)
    decimal_places = animation_time_decimal_places(frame_times)
    time_width = fixed_time_text_width(frame_times, decimal_places=decimal_places)

    with oel_plot_context(style_name=spec.style_name, metadata=metadata):
        fig, axes_array = plt.subplots(1, 3, figsize=cap_figsize(15.0, 4.8), dpi=spec.dpi)
        axes = list(np.asarray(axes_array, dtype=object).reshape(-1))
        line_by_plane_pair: dict[tuple[int, str], Any] = {}
        dot_by_plane_pair: dict[tuple[int, str], Any] = {}
        for plane_index, (_x_index, _y_index, title, x_label, y_label) in enumerate(planes):
            ax = axes[plane_index]
            ax.set_title(title)
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.grid(True, alpha=0.3)
            initial_limits = camera["limits"][plane_index][0]
            ax.set_xlim(*initial_limits[0])
            ax.set_ylim(*initial_limits[1])
            ax.set_aspect("equal", adjustable="box")
            for pair_index, pair_id in enumerate(sorted(grouped)):
                color = palette[pair_index % len(palette)]
                (line,) = ax.plot([], [], linewidth=1.4, color=color, label=pair_id)
                (dot,) = ax.plot([], [], marker="o", markersize=4.5, color=color)
                line_by_plane_pair[(plane_index, pair_id)] = line
                dot_by_plane_pair[(plane_index, pair_id)] = dot

        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(
                handles,
                labels,
                loc="lower center",
                bbox_to_anchor=(0.5, 0.035),
                ncol=min(len(labels), 5),
                frameon=True,
            )
        fig.suptitle(recipe.title, x=0.02, y=0.985, ha="left", va="top", fontsize=12)
        time_text = fig.text(
            0.98,
            0.985,
            "",
            ha="right",
            va="top",
            fontsize=9,
            family="monospace",
            gid="oel_animation_time",
        )
        fig.tight_layout(rect=(0.01, 0.105, 0.99, 0.92))

        def update(frame_ordinal: int) -> list[Any]:
            ordinal = int(frame_ordinal)
            time_now = float(frame_times[ordinal])
            artists: list[Any] = []
            for plane_index, (x_index, y_index, _title, _x_label, _y_label) in enumerate(planes):
                ax = axes[plane_index]
                x_limits, y_limits = camera["limits"][plane_index][ordinal]
                ax.set_xlim(*x_limits)
                ax.set_ylim(*y_limits)
                for pair_id, pair in grouped.items():
                    end = int(np.searchsorted(pair[:, 0], time_now, side="right"))
                    if end <= 0:
                        line_by_plane_pair[(plane_index, pair_id)].set_data([], [])
                        dot_by_plane_pair[(plane_index, pair_id)].set_data([], [])
                    else:
                        line_by_plane_pair[(plane_index, pair_id)].set_data(
                            pair[:end, 1 + x_index], pair[:end, 1 + y_index]
                        )
                        dot_by_plane_pair[(plane_index, pair_id)].set_data(
                            [pair[end - 1, 1 + x_index]], [pair[end - 1, 1 + y_index]]
                        )
                    artists.extend(
                        [line_by_plane_pair[(plane_index, pair_id)], dot_by_plane_pair[(plane_index, pair_id)]]
                    )
            time_value = format_animation_time(
                time_now,
                decimal_places=decimal_places,
                width=time_width,
            )
            time_text.set_text(f"Sim time: {time_value} s")
            artists.append(time_text)
            return artists

        animation_obj = animation.FuncAnimation(
            fig,
            update,
            frames=int(frame_times.size),
            interval=1000.0 / float(spec.fps),
            blit=False,
        )
        format_limits = {
            (plane_index, "x"): tuple(camera["format_limits"][plane_index][0])
            for plane_index in range(len(planes))
        }
        format_limits.update(
            {
                (plane_index, "y"): tuple(camera["format_limits"][plane_index][1])
                for plane_index in range(len(planes))
            }
        )
        source = {
            "review_store": _review_store_identity(workspace),
            "query_sha256": hashlib.sha256(recipe.sql.encode("utf-8")).hexdigest(),
            "recipe_id": recipe.recipe_id,
            "recipe_version": recipe.recipe_version,
            "renderer_id": recipe.renderer_id,
            "row_count": result.row_count,
            "truncated": result.truncated,
            "frame_plan": frame_plan,
        }
        try:
            report = save_animation_with_quality(
                animation_obj,
                fig,
                target,
                update=update,
                frame_values=tuple(range(int(frame_times.size))),
                frame_times_s=tuple(float(value) for value in frame_times),
                fps=spec.fps,
                camera_policy=spec.camera_policy,
                metadata=metadata,
                artifact_id=artifact_id,
                style_name=spec.style_name,
                format_limits=format_limits,
                key_frame_indices=_extrema_frame_indices(grouped, frame_times),
                source=source,
                logical_contact_sheet_path=logical_contact_sheet,
                logical_quality_receipt_path=logical_quality_receipt,
            )
        finally:
            plt.close(fig)

    artifact = ReviewAnimationArtifact(
        artifact_id=artifact_id,
        path=target,
        relative_path=_relative_to_output(workspace, logical_target),
        contact_sheet_path=target.with_suffix(".contact-sheet.png"),
        quality_receipt_path=target.with_suffix(".quality.json"),
        row_count=result.row_count,
        truncated=result.truncated,
        spec=spec,
        qa=report.to_dict(),
        logical_path=logical_target,
        logical_contact_sheet_path=logical_contact_sheet,
        logical_quality_receipt_path=logical_quality_receipt,
    )
    if record:
        record_generated_animation(
            workspace,
            artifact,
            recipe=recipe,
            writer=writer,
            logical_output_dir=logical_output_dir,
        )
    return artifact


def record_generated_animation(
    workspace: ReviewWorkspace,
    artifact: ReviewAnimationArtifact,
    *,
    recipe: ReviewAnimationRecipe,
    review_store_identity: dict[str, Any] | None = None,
    writer: Any | None = None,
    logical_output_dir: str | Path | None = None,
) -> None:
    writer = writer or _current_output_writer()
    index_path = workspace.review_dir / "generated_artifacts.json"
    index_relative: Path | None = None
    if writer is not None:
        logical_root = Path(logical_output_dir) if logical_output_dir is not None else Path(writer.logical_root)
        index_relative = _lexical_relative_path(index_path, logical_root)
        if index_relative is None:
            raise ValueError("A protected generated-artifact index must be below the logical output root.")
        try:
            # Keep the merge on the writer-owned inode even if the visible
            # output directory is replaced during the operation.
            writer.file_record(index_relative)
            payload = json.loads(writer.read_bytes(index_relative).decode("utf-8"))
            existing = list(payload.get("artifacts", []) or []) if isinstance(payload, dict) else []
        except FileNotFoundError:
            existing = []
        except Exception:
            existing = []
    else:
        if index_path.is_file():
            payload = json.loads(index_path.read_text(encoding="utf-8"))
            existing = list(payload.get("artifacts", []) or []) if isinstance(payload, dict) else []
        else:
            existing = []
    existing.append(
        {
            "artifact_id": artifact.artifact_id,
            "artifact_type": "animation",
            "path": artifact.relative_path,
            "contact_sheet_path": _relative_to_output(
                workspace,
                artifact.logical_contact_sheet_path or artifact.contact_sheet_path,
            ),
            "quality_receipt_path": _relative_to_output(
                workspace,
                artifact.logical_quality_receipt_path or artifact.quality_receipt_path,
            ),
            "created_utc": os.environ.get("OEL_GENERATED_UTC", "").strip()
            or datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "source": str(dict(artifact.spec.extra or {}).get("source", "oel_review_animation_api")),
            "source_query": recipe.sql,
            "query_sha256": hashlib.sha256(recipe.sql.encode("utf-8")).hexdigest(),
            "review_store": review_store_identity or _review_store_identity(workspace),
            "recipe_id": recipe.recipe_id,
            "recipe_version": recipe.recipe_version,
            "renderer_id": recipe.renderer_id,
            "style_name": artifact.spec.style_name,
            "file_format": artifact.spec.file_format,
            "fps": artifact.spec.fps,
            "camera_policy": artifact.spec.camera_policy,
            "row_count": artifact.row_count,
            "truncated": artifact.truncated,
            "qa": dict(artifact.qa),
            "spec": asdict(artifact.spec),
        }
    )
    encoded = (json.dumps({"artifacts": existing}, indent=2) + "\n").encode("utf-8")
    if writer is None:
        index_path.write_bytes(encoded)
    else:
        assert index_relative is not None
        try:
            writer.write_bytes(index_relative, encoded)
        except FileExistsError:
            writer.update_existing_bytes(index_relative, encoded)


def _group_ric_evidence(result: ReviewQueryResult) -> dict[str, np.ndarray]:
    grouped: dict[str, list[tuple[float, float, float, float]]] = {}
    for row in result.rows:
        try:
            values = (
                float(row["time_s"]),
                float(row["r_radial_km"]),
                float(row["i_intrack_km"]),
                float(row["c_crosstrack_km"]),
            )
        except (KeyError, TypeError, ValueError):
            continue
        if not all(math.isfinite(value) for value in values):
            continue
        grouped.setdefault(str(row.get("pair_id") or "relative trajectory"), []).append(values)
    return {
        pair_id: np.asarray(sorted(rows, key=lambda row: row[0]), dtype=float)
        for pair_id, rows in sorted(grouped.items())
        if rows
    }


def _camera_plan(
    grouped: dict[str, np.ndarray],
    frame_times: np.ndarray,
    *,
    planes: tuple[tuple[int, int, str, str, str], ...],
    camera_policy: str,
) -> dict[str, Any]:
    all_points = np.concatenate([values[:, 1:4] for values in grouped.values()], axis=0)
    format_limits: list[tuple[tuple[float, float], tuple[float, float]]] = []
    limits_by_plane: list[list[tuple[tuple[float, float], tuple[float, float]]]] = []
    for x_index, y_index, _title, _x_label, _y_label in planes:
        x_values = all_points[:, x_index]
        y_values = all_points[:, y_index]
        if camera_policy == "fixed":
            extent = max(float(np.max(np.abs(x_values))), float(np.max(np.abs(y_values))), 0.5) * 1.12
            base_x = (-extent, extent)
            base_y = (-extent, extent)
        else:
            base_x = _padded_limits(x_values)
            base_y = _padded_limits(y_values)
        format_limits.append((base_x, base_y))
        if camera_policy != "follow":
            limits_by_plane.append([(base_x, base_y) for _ in frame_times])
            continue

        current_by_frame: list[np.ndarray] = []
        for time_now in frame_times:
            points: list[tuple[float, float]] = []
            for pair in grouped.values():
                index = max(int(np.searchsorted(pair[:, 0], time_now, side="right")) - 1, 0)
                points.append((float(pair[index, 1 + x_index]), float(pair[index, 1 + y_index])))
            current_by_frame.append(np.asarray(points, dtype=float))
        x_span = max((float(np.ptp(points[:, 0])) for points in current_by_frame), default=0.0)
        y_span = max((float(np.ptp(points[:, 1])) for points in current_by_frame), default=0.0)
        x_span = max(x_span * 1.24, 1.0)
        y_span = max(y_span * 1.24, 1.0)
        format_limits[-1] = ((-0.5 * x_span, 0.5 * x_span), (-0.5 * y_span, 0.5 * y_span))
        dynamic: list[tuple[tuple[float, float], tuple[float, float]]] = []
        for points in current_by_frame:
            center_x = float(np.mean(points[:, 0]))
            center_y = float(np.mean(points[:, 1]))
            dynamic.append(
                (
                    (center_x - 0.5 * x_span, center_x + 0.5 * x_span),
                    (center_y - 0.5 * y_span, center_y + 0.5 * y_span),
                )
            )
        limits_by_plane.append(dynamic)
    return {"format_limits": format_limits, "limits": limits_by_plane}


def _padded_limits(values: np.ndarray, *, minimum_span: float = 1.0, margin: float = 0.12) -> tuple[float, float]:
    lower = float(np.min(values))
    upper = float(np.max(values))
    center = 0.5 * (lower + upper)
    span = max(upper - lower, minimum_span)
    half = 0.5 * span * (1.0 + 2.0 * margin)
    return center - half, center + half


def _extrema_frame_indices(grouped: dict[str, np.ndarray], frame_times: np.ndarray) -> tuple[int, ...]:
    rows = np.concatenate([values for values in grouped.values()], axis=0)
    magnitude = np.linalg.norm(rows[:, 1:4], axis=1)
    extrema_times = (float(rows[int(np.argmin(magnitude)), 0]), float(rows[int(np.argmax(magnitude)), 0]))
    return tuple(int(np.argmin(np.abs(frame_times - value))) for value in extrema_times)


def _scenario_name(workspace: ReviewWorkspace) -> str:
    try:
        result = workspace.query("SELECT scenario_name FROM run_metadata LIMIT 1", max_rows=1)
        if result.rows:
            value = str(result.rows[0].get("scenario_name") or "").strip()
            if value:
                return value
    except Exception:
        pass
    return workspace.output_dir.name


def _review_store_identity(workspace: ReviewWorkspace) -> dict[str, Any]:
    return workspace.evidence_identity()


def _relative_to_output(workspace: ReviewWorkspace, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(workspace.output_dir.resolve()))
    except ValueError:
        return str(path.resolve())


def _current_output_writer() -> Any | None:
    try:
        from sim.security.output_writes import current_output_writer
    except ImportError:
        return None
    return current_output_writer()


__all__ = ["ReviewAnimationArtifact", "record_generated_animation", "render_review_animation_artifact"]
