"""Stable, lazy review facade.

Table-only review-store writers are part of ordinary simulation execution and
must not import Matplotlib. Plot and animation owners are loaded only when a
consumer requests their public facade symbol.
"""

from __future__ import annotations

import importlib
from typing import Any

from sim.runtime_environment import configure_headless_runtime, configure_runtime_caches

configure_runtime_caches()
configure_headless_runtime()

_EXPORTS = {
    "validate_review_sql": "sim.review.workspace",
    "scientific_evidence": "sim.review.scientific",
    "read_scientific_evidence": "sim.review.scientific",
    "render_cr3bp_research": "sim.review.cr3bp",
    "load_cr3bp_history": "sim.review.cr3bp",
    "EVIDENCE_CAPSULE_MANIFEST": "sim.review.evidence_capsule",
    "EVIDENCE_CAPSULE_SCHEMA": "sim.review.evidence_capsule",
    "ANIMATION_RECIPE_SCHEMA_VERSION": "sim.review.animation_recipes",
    "REVIEW_ANIMATION_PLAN_SCHEMA_VERSION": "sim.review.animation_planning",
    "REVIEW_ANIMATION_RECIPES": "sim.review.animation_recipes",
    "ReviewAnimationArtifact": "sim.review.animation_rendering",
    "ReviewAnimationRecipe": "sim.review.animation_recipes",
    "ReviewAnimationSpec": "sim.review.animation_planning",
    "ReviewQueryError": "sim.review.workspace",
    "ReviewQueryResult": "sim.review.workspace",
    "ReviewStoreNotFoundError": "sim.review.workspace",
    "ReviewWorkspace": "sim.review.workspace",
    "SAVED_REVIEW_QUERIES": "sim.review.queries",
    "SAVED_QUERY_MATURITY_LEVELS": "sim.review.queries",
    "WORKFLOW_REVIEW_SCHEMA_VERSION": "sim.review.manifest",
    "EVIDENCE_PLOT_RECIPES": "sim.review.plotting",
    "EvidencePlotRecipe": "sim.review.plotting",
    "EvidencePlotter": "sim.review.plotting",
    "ReviewPlotArtifact": "sim.review.plotting",
    "ReviewPlotSpec": "sim.review.plotting",
    "ReviewPlotRecipe": "sim.review.plot_recipes",
    "SavedReviewQuery": "sim.review.queries",
    "animation_spec_from_mapping": "sim.review.animation_planning",
    "create_evidence_capsule": "sim.review.evidence_capsule",
    "categorical_columns": "sim.review.plotting",
    "default_plot_spec": "sim.review.plotting",
    "get_saved_review_query": "sim.review.queries",
    "get_review_animation_recipe": "sim.review.animation_recipes",
    "load_workflow_manifest": "sim.review.manifest",
    "list_saved_review_queries": "sim.review.queries",
    "list_review_animation_recipes": "sim.review.animation_recipes",
    "numeric_columns": "sim.review.plotting",
    "PLOT_RECIPE_SCHEMA_VERSION": "sim.review.plot_recipes",
    "REVIEW_PLOT_PLAN_SCHEMA_VERSION": "sim.review.plot_planning",
    "REVIEW_PLOT_RECIPES": "sim.review.plot_recipes",
    "get_review_plot_recipe": "sim.review.plot_recipes",
    "list_review_plot_recipes": "sim.review.plot_recipes",
    "evidence_file_exists": "sim.review.evidence_capsule",
    "evidence_file_sha256": "sim.review.evidence_capsule",
    "materialized_evidence_file": "sim.review.evidence_capsule",
    "plan_review_plot": "sim.review.plot_planning",
    "plan_review_animation": "sim.review.animation_planning",
    "plot_spec_from_mapping": "sim.review.plot_planning",
    "render_review_plot": "sim.review.plot_planning",
    "render_review_animation": "sim.review.animation_planning",
    "review_animation_plan_id": "sim.review.animation_planning",
    "review_plot_plan_id": "sim.review.plot_planning",
    "save_review_plot": "sim.review.plotting",
    "restore_evidence_capsule": "sim.review.evidence_capsule",
    "workflow_manifest_path": "sim.review.manifest",
    "write_workflow_review": "sim.review.manifest",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
