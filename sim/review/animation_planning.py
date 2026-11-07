"""Content-bound planning for agent-native review animations."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from sim.plotting.animation_quality import STRICT_AGENT_ANIMATION_QUALITY
from sim.review.animation_recipes import ReviewAnimationRecipe, get_review_animation_recipe
from sim.review.workspace import ReviewQueryResult, ReviewWorkspace

REVIEW_ANIMATION_PLAN_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ReviewAnimationSpec:
    recipe_id: str
    artifact_id: str
    style_name: str = "oel_dark"
    file_format: str = "mp4"
    fps: float = 20.0
    frame_stride: int = 1
    camera_policy: str = "fit_history"
    max_rows: int = 5000
    dpi: int = 120
    extra: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if get_review_animation_recipe(self.recipe_id) is None:
            raise ValueError(f"Unknown supported review animation recipe '{self.recipe_id}'.")
        if self.style_name not in {"oel_dark", "oel_light"}:
            raise ValueError("Review animation style must be 'oel_dark' or 'oel_light'.")
        if self.file_format not in {"mp4", "gif"}:
            raise ValueError("Review animation format must be 'mp4' or 'gif'.")
        if not 1.0 <= float(self.fps) <= 60.0:
            raise ValueError("Review animation fps must be between 1 and 60.")
        if not 1 <= int(self.frame_stride) <= 1000:
            raise ValueError("Review animation frame_stride must be between 1 and 1000.")
        if self.camera_policy not in {"fixed", "fit_history", "follow"}:
            raise ValueError("Review animation camera_policy must be 'fixed', 'fit_history', or 'follow'.")
        if not 1 <= int(self.max_rows) <= 5000:
            raise ValueError("Review animation max_rows must be between 1 and 5000.")
        if not 72 <= int(self.dpi) <= 200:
            raise ValueError("Review animation dpi must be between 72 and 200.")


def animation_spec_from_mapping(arguments: Mapping[str, Any], *, source: str) -> ReviewAnimationSpec:
    return ReviewAnimationSpec(
        recipe_id=str(arguments["recipe_id"]),
        artifact_id=str(arguments["artifact_id"]),
        style_name=str(arguments.get("style", "oel_dark") or "oel_dark"),
        file_format=str(arguments.get("format", "mp4") or "mp4"),
        fps=float(arguments.get("fps", 20.0)),
        frame_stride=int(arguments.get("frame_stride", 1)),
        camera_policy=str(arguments.get("camera_policy", "fit_history") or "fit_history"),
        max_rows=int(arguments.get("max_rows", 5000)),
        dpi=int(arguments.get("dpi", 120)),
        extra={"source": source, "animation_contract": "typed_review_animation_v1"},
    )


def plan_review_animation(
    output_dir: str | Path,
    spec: ReviewAnimationSpec,
    *,
    protected: bool | None = None,
) -> dict[str, Any]:
    with (
        ReviewWorkspace.open(output_dir)
        if protected is None
        else ReviewWorkspace.open(output_dir, protected=protected)
    ) as workspace:
        recipe, result = load_review_animation_evidence(workspace, spec)
        frame_plan = review_animation_frame_plan(result, spec)
        plan_id = _review_animation_plan_id_from_evidence(workspace, spec, recipe, result)
        warnings: list[str] = []
        if result.truncated:
            warnings.append("The review query reached max_rows; rendering is blocked because the movie would omit evidence.")
        if frame_plan["effective_frame_stride"] > spec.frame_stride:
            warnings.append(
                "The frame stride was increased deterministically to satisfy the animation frame and duration limits."
            )
        return {
            "status": "planned",
            "output_dir": str(workspace.output_dir),
            "review_store": "review/run.sqlite",
            "animation_plan_id": plan_id,
            "spec": asdict(spec),
            "recipe": recipe.to_dict(),
            "row_count": result.row_count,
            "truncated": result.truncated,
            "source_frame_count": frame_plan["source_frame_count"],
            "render_frame_count": frame_plan["render_frame_count"],
            "effective_frame_stride": frame_plan["effective_frame_stride"],
            "encoded_duration_s": frame_plan["encoded_duration_s"],
            "resource_estimate": {
                "width_px": int(round(15.0 * spec.dpi)),
                "height_px": int(round(4.8 * spec.dpi)),
                "maximum_file_bytes": STRICT_AGENT_ANIMATION_QUALITY.max_file_bytes,
                "maximum_frames": STRICT_AGENT_ANIMATION_QUALITY.max_frames,
            },
            "warnings": warnings,
            "render_ready": not result.truncated and frame_plan["render_frame_count"] >= 2,
            "render_authorized": False,
            "visual_review_required": True,
        }


def render_review_animation(
    output_dir: str | Path,
    spec: ReviewAnimationSpec,
    *,
    animation_plan_id: str,
    path: str | Path,
    record: bool = True,
    writer: Any | None = None,
    logical_output_dir: str | Path | None = None,
    protected: bool | None = None,
) -> Any:
    workspace = (
        ReviewWorkspace.open(output_dir)
        if protected is None
        else ReviewWorkspace.open(output_dir, protected=protected)
    )
    active_writer = writer or _current_output_writer()
    recipe, result = load_review_animation_evidence(workspace, spec)
    initial_identity = workspace.evidence_identity()
    current_id = _review_animation_plan_id_from_evidence(
        workspace,
        spec,
        recipe,
        result,
        review_store_identity=initial_identity,
    )
    if str(animation_plan_id) != current_id:
        raise ValueError(
            "The animation_plan_id is stale or does not match the review store and animation specification."
        )
    if result.truncated:
        raise ValueError("The review animation query is truncated; increase frame_stride or use a smaller study.")
    from sim.review.animation_rendering import (
        record_generated_animation,
        render_review_animation_artifact,
    )

    artifact = render_review_animation_artifact(
        workspace,
        recipe,
        result,
        spec,
        path=path,
        record=False,
        writer=writer,
        logical_output_dir=logical_output_dir,
    )
    final_recipe, final_result = load_review_animation_evidence(workspace, spec)
    final_identity = workspace.evidence_identity()
    if (
        _review_animation_plan_id_from_evidence(
            workspace,
            spec,
            final_recipe,
            final_result,
            review_store_identity=final_identity,
        )
        != current_id
    ):
        # Protected renders may already have been published through a
        # descriptor-bound writer.  A pathname unlink here could remove a
        # concurrent replacement at the same logical path; isolated staging
        # transactions own their cleanup instead.
        if active_writer is None:
            for generated_path in (
                artifact.path,
                artifact.contact_sheet_path,
                artifact.quality_receipt_path,
            ):
                generated_path.unlink(missing_ok=True)
        raise ValueError(
            "The review store changed while the planned animation was rendering; no artifact was recorded."
        )
    if record:
        record_generated_animation(
            workspace,
            artifact,
            recipe=recipe,
            review_store_identity=final_identity,
            writer=writer,
            logical_output_dir=logical_output_dir,
        )
    return artifact


def review_animation_plan_id(workspace: ReviewWorkspace, spec: ReviewAnimationSpec) -> str:
    recipe, result = load_review_animation_evidence(workspace, spec)
    return _review_animation_plan_id_from_evidence(workspace, spec, recipe, result)


def _current_output_writer() -> Any | None:
    try:
        from sim.security.output_writes import current_output_writer
    except ImportError:
        return None
    return current_output_writer()


def _review_animation_plan_id_from_evidence(
    workspace: ReviewWorkspace,
    spec: ReviewAnimationSpec,
    recipe: ReviewAnimationRecipe,
    result: ReviewQueryResult,
    *,
    review_store_identity: dict[str, Any] | None = None,
) -> str:
    payload = {
        "schema_version": REVIEW_ANIMATION_PLAN_SCHEMA_VERSION,
        "animation_quality_policy": {
            "policy_id": STRICT_AGENT_ANIMATION_QUALITY.policy_id,
            "version": STRICT_AGENT_ANIMATION_QUALITY.version,
        },
        "review_store": review_store_identity or workspace.evidence_identity(),
        "recipe": recipe.to_dict(),
        "spec": asdict(spec),
        "frame_plan": review_animation_frame_plan(result, spec),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return "oel-review-animation-plan-v1:" + hashlib.sha256(encoded).hexdigest()


def load_review_animation_evidence(
    workspace: ReviewWorkspace,
    spec: ReviewAnimationSpec,
) -> tuple[ReviewAnimationRecipe, ReviewQueryResult]:
    recipe = get_review_animation_recipe(spec.recipe_id)
    if recipe is None:
        raise ValueError(f"Unknown supported review animation recipe '{spec.recipe_id}'.")
    missing_tables = sorted(set(recipe.supported_tables) - set(workspace.tables()))
    if missing_tables:
        raise ValueError("Review animation requires missing tables: " + ", ".join(missing_tables))
    result = workspace.query(recipe.sql, max_rows=spec.max_rows)
    if result.row_count == 0:
        raise ValueError("Review animation query returned no evidence rows.")
    missing_columns = sorted(set(recipe.required_columns) - set(result.columns))
    if missing_columns:
        raise ValueError("Review animation query is missing columns: " + ", ".join(missing_columns))
    return recipe, result


def review_animation_frame_plan(result: ReviewQueryResult, spec: ReviewAnimationSpec) -> dict[str, Any]:
    times = sorted(
        {
            float(row["time_s"])
            for row in result.rows
            if row.get("time_s") is not None and math.isfinite(float(row["time_s"]))
        }
    )
    if not times:
        raise ValueError("Review animation evidence contains no finite time samples.")
    max_by_duration = max(2, int(math.floor(STRICT_AGENT_ANIMATION_QUALITY.max_duration_s * spec.fps)))
    maximum_frames = min(STRICT_AGENT_ANIMATION_QUALITY.max_frames, max_by_duration)
    effective_stride = max(int(spec.frame_stride), int(math.ceil(len(times) / maximum_frames)))
    selected = list(range(0, len(times), effective_stride))
    if selected[-1] != len(times) - 1:
        selected.append(len(times) - 1)
    return {
        "source_frame_count": len(times),
        "render_frame_count": len(selected),
        "effective_frame_stride": effective_stride,
        "source_frame_indices": selected,
        "frame_times_s": [times[index] for index in selected],
        "encoded_duration_s": len(selected) / float(spec.fps),
    }


__all__ = [
    "REVIEW_ANIMATION_PLAN_SCHEMA_VERSION",
    "ReviewAnimationSpec",
    "animation_spec_from_mapping",
    "load_review_animation_evidence",
    "plan_review_animation",
    "render_review_animation",
    "review_animation_frame_plan",
    "review_animation_plan_id",
]
