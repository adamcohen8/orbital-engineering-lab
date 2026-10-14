from __future__ import annotations

from typing import Any

from integrations.oel_mcp.contracts import (
    HANDLING_SCHEMA,
    MAX_MANIFEST_BYTES,
    MAX_REVIEW_RESULT_BYTES,
    MAX_REVIEW_VALUE_BYTES,
    MAX_ROWS,
    MAX_VM_STEPS,
    ToolContract,
    handling_properties,
    object_schema,
)
from integrations.oel_mcp.cr3bp_registry import CR3BP_TOOL_CONTRACTS
from integrations.oel_mcp.execution import M4_RESOURCE_PROFILES
from integrations.oel_mcp.fsw_authoring_registry import FSW_AUTHORING_TOOL_CONTRACTS, FSW_AUTHORING_TOOL_IDS
from sim.review.animation_recipes import list_review_animation_recipes
from sim.review.plot_recipes import list_review_plot_recipes

PUBLIC_PROFILES = ("public_local", "direct_frontier_restricted")
M4_LOCAL_PROFILES = ("public_local",)
M3_PUBLIC_TOOL_IDS = (
    "oel.describe_capabilities.v1",
    "oel.inspect_run.v1",
    "oel.query_review.v1",
)

APPROVAL_SCHEMA: dict[str, Any] = object_schema(
    {
        "approval_id": {"type": "string", "minLength": 1, "maxLength": 120},
        "scope": {"type": "string", "enum": ["trust", "write", "execute"]},
    },
    required=("approval_id", "scope"),
)

SUPPORTED_REVIEW_PLOT_RECIPE_IDS = tuple(
    recipe.recipe_id for recipe in list_review_plot_recipes() if recipe.maturity == "supported"
)
SUPPORTED_REVIEW_ANIMATION_RECIPE_IDS = tuple(
    recipe.recipe_id for recipe in list_review_animation_recipes() if recipe.maturity == "supported"
)

REVIEW_PLOT_SPEC_PROPERTIES: dict[str, Any] = {
    "output_dir": {"type": "string", "minLength": 1},
    "sql": {"type": "string", "minLength": 1, "maxLength": 20_000},
    "x_column": {"type": "string", "maxLength": 160},
    "y_columns": {
        "type": "array",
        "items": {"type": "string", "minLength": 1, "maxLength": 160},
        "minItems": 1,
        "maxItems": 12,
    },
    "plot_type": {"type": "string", "enum": ["line", "scatter", "bar", "histogram", "heatmap"]},
    "group_column": {"type": "string", "maxLength": 160},
    "style": {"type": "string", "enum": ["oel_dark", "oel_light"], "default": "oel_dark"},
    "title": {"type": "string", "maxLength": 240},
    "subtitle": {"type": "string", "maxLength": 320},
    "x_label": {"type": "string", "maxLength": 160},
    "y_label": {"type": "string", "maxLength": 160},
    "format": {"type": "string", "enum": ["png", "svg", "pdf"], "default": "png"},
    "dpi": {"type": "integer", "minimum": 72, "maximum": 600, "default": 150},
    "max_rows": {"type": "integer", "minimum": 1, "maximum": 5000, "default": 5000},
    "artifact_id": {"type": "string", "minLength": 1, "maxLength": 80},
}

REVIEW_ANIMATION_SPEC_PROPERTIES: dict[str, Any] = {
    "output_dir": {"type": "string", "minLength": 1},
    "recipe_id": {"type": "string", "enum": list(SUPPORTED_REVIEW_ANIMATION_RECIPE_IDS)},
    "artifact_id": {"type": "string", "minLength": 1, "maxLength": 80},
    "style": {"type": "string", "enum": ["oel_dark", "oel_light"], "default": "oel_dark"},
    "format": {"type": "string", "enum": ["mp4", "gif"], "default": "mp4"},
    "fps": {"type": "number", "minimum": 1, "maximum": 60, "default": 20},
    "frame_stride": {"type": "integer", "minimum": 1, "maximum": 1000, "default": 1},
    "camera_policy": {
        "type": "string",
        "enum": ["fixed", "fit_history", "follow"],
        "default": "fit_history",
    },
    "max_rows": {"type": "integer", "minimum": 1, "maximum": 5000, "default": 5000},
    "dpi": {"type": "integer", "minimum": 72, "maximum": 200, "default": 120},
}

SCENARIO_INPUT_PROPERTIES: dict[str, Any] = {
    "config_path": {"type": "string", "minLength": 1},
    "output_dir": {"type": "string", "minLength": 1},
    "resource_profile": {"type": "string", "enum": list(M4_RESOURCE_PROFILES), "default": "laptop-safe"},
}

TOOL_READINESS_SCHEMA = object_schema(
    {
        "tool_id": {"type": "string"}, "registered": {"const": True},
        "configuration_ready": {"type": "boolean"},
        "blockers": {"type": "array", "items": {"type": "string"}},
        "entitlement_status": {"enum": ["not_required", "available", "unavailable", "not_checked"]},
        "trust_approval_required": {"type": "boolean"},
        "conditional_trust_available": {"type": ["boolean", "null"]},
        "execution_authorized": {"const": False},
    },
    required=("tool_id", "registered", "configuration_ready", "blockers", "entitlement_status",
              "trust_approval_required", "conditional_trust_available", "execution_authorized"),
)

DESCRIBE_RESULT_SCHEMA: dict[str, Any] = object_schema(
    {
        "status": {"const": "available"},
        "integration": {"type": "string"},
        "transport": {"const": "stdio"},
        "deployment_profile": {"type": "string"},
        "capabilities": {"type": "array", "items": {"type": "object"}},
        "dependency_direction": {"const": "mcp_consumes_oel"},
        "compatibility": {"type": "object"},
        "non_claims": {"type": "array", "items": {"type": "string"}},
        "agent_guide_uri": {"type": "string"},
        "workflow_routes_uri": {"type": "string"},
        "readiness": object_schema({
            "semantics": {"type": "string"}, "execution_authorized": {"const": False},
            "tools": {"type": "array", "items": TOOL_READINESS_SCHEMA},
        }, required=("semantics", "execution_authorized", "tools")),
    },
    required=(
        "status",
        "integration",
        "transport",
        "deployment_profile",
        "capabilities",
        "dependency_direction",
        "compatibility",
        "non_claims",
    ),
)

INSPECT_RUN_RESULT_SCHEMA: dict[str, Any] = object_schema(
    {
        "output_dir": {"type": "string"},
        "status": {"enum": ["completed", "partial"]},
        "evidence_summary": {"type": "object"},
        "review": {"type": "object"},
        "artifact_summary": {"type": "object"},
        "execution_provenance": {"type": "object"},
        "failure_hints": {"type": "array", "items": {"type": "object"}},
        "caveats": {"type": "array", "items": {"type": "string"}},
        "freshness": {"type": "object"},
    },
    required=(
        "output_dir",
        "status",
        "evidence_summary",
        "review",
        "artifact_summary",
        "execution_provenance",
        "failure_hints",
        "caveats",
        "freshness",
    ),
)

QUERY_REVIEW_RESULT_SCHEMA: dict[str, Any] = object_schema(
    {
        "output_dir": {"type": "string"},
        "review_store": {"const": "review/run.sqlite"},
        "sql": {"type": "string"},
        "columns": {"type": "array", "items": {"type": "string"}},
        "rows": {"type": "array", "items": {"type": "object"}},
        "row_count": {"type": "integer", "minimum": 0, "maximum": MAX_ROWS},
        "empty_result": {"type": "boolean"},
        "empty_result_semantics": {"type": "string"},
        "truncated": {"type": "boolean"},
        "units_semantics": {"type": "string"},
    },
    required=(
        "output_dir",
        "review_store",
        "sql",
        "columns",
        "rows",
        "row_count",
        "empty_result",
        "empty_result_semantics",
        "truncated",
        "units_semantics",
    ),
)

INSPECT_STUDY_RESULT_SCHEMA: dict[str, Any] = object_schema(
    {
        "schema_version": {"type": "string"},
        "status": {"const": "verified"},
        "study_id": {"type": "string"},
        "bundle_semantic_sha256": {"type": "string"},
        "request_sha256": {"type": "string"},
        "plan_sha256": {"type": "string"},
        "run_sha256": {"type": "string"},
        "evidence_sha256": {"type": "string"},
        "claims_sha256": {"type": "string"},
        "step_count": {"type": "integer", "minimum": 0},
        "claim_count": {"type": "integer", "minimum": 0},
        "non_claim_count": {"type": "integer", "minimum": 0},
        "title": {"type": "string"},
        "question": {"type": "string"},
        "capabilities": {"type": "array", "items": {"type": "string"}},
        "steps": {"type": "array", "items": {"type": "object"}},
        "claims": {"type": "array", "items": {"type": "object"}},
        "non_claims": {"type": "array", "items": {"type": "string"}},
    },
    required=(
        "schema_version",
        "status",
        "study_id",
        "bundle_semantic_sha256",
        "request_sha256",
        "plan_sha256",
        "run_sha256",
        "evidence_sha256",
        "claims_sha256",
        "step_count",
        "claim_count",
        "non_claim_count",
        "title",
        "question",
        "capabilities",
        "steps",
        "claims",
        "non_claims",
    ),
)

REPLAY_STUDY_RESULT_SCHEMA: dict[str, Any] = object_schema(
    {
        key: value
        for key, value in INSPECT_STUDY_RESULT_SCHEMA["properties"].items()
        if key not in {"title", "question", "capabilities", "steps", "claims", "non_claims"}
    }
    | {"replay_status": {"const": "identity_verified"}},
    required=(
        "schema_version",
        "status",
        "study_id",
        "bundle_semantic_sha256",
        "request_sha256",
        "plan_sha256",
        "run_sha256",
        "evidence_sha256",
        "claims_sha256",
        "step_count",
        "claim_count",
        "non_claim_count",
        "replay_status",
    ),
)

COMPARE_STUDIES_RESULT_SCHEMA: dict[str, Any] = object_schema(
    {
        "schema_version": {"type": "string"},
        "status": {"enum": ["equivalent", "different"]},
        "same_bundle": {"type": "boolean"},
        "left_study_id": {"type": "string"},
        "right_study_id": {"type": "string"},
        "left_bundle_semantic_sha256": {"type": "string"},
        "right_bundle_semantic_sha256": {"type": "string"},
        "changed_records": {"type": "array", "items": {"type": "string"}},
        "changed_evidence_steps": {"type": "array", "items": {"type": "string"}},
    },
    required=(
        "schema_version",
        "status",
        "same_bundle",
        "left_study_id",
        "right_study_id",
        "left_bundle_semantic_sha256",
        "right_bundle_semantic_sha256",
        "changed_records",
        "changed_evidence_steps",
    ),
)

INSPECT_CCSDS_RESULT_SCHEMA: dict[str, Any] = object_schema(
    {
        "status": {"const": "inspected"},
        "product_kind": {"enum": ["oem", "odm", "tdm", "cdm"]},
        "source_path": {"type": "string"},
        "source_sha256": {"type": "string"},
        "inspection": {"type": "object"},
        "execution_occurred": {"const": False},
        "non_claims": {"type": "array", "items": {"type": "string"}},
    },
    required=(
        "status",
        "product_kind",
        "source_path",
        "source_sha256",
        "inspection",
        "execution_occurred",
        "non_claims",
    ),
)

FRAME_TIME_RESULT_SCHEMA: dict[str, Any] = object_schema(
    {
        "status": {"enum": ["converted", "inspected"]},
        "operation": {"enum": ["convert_epoch", "inspect_eop", "transform_state", "transform_covariance"]},
        "result": {"type": "object"},
        "receipt": {"type": ["object", "null"]},
        "eop_source": {"type": ["object", "null"]},
        "execution_occurred": {"const": False},
        "non_claims": {"type": "array", "items": {"type": "string"}},
    },
    required=(
        "status",
        "operation",
        "result",
        "receipt",
        "eop_source",
        "execution_occurred",
        "non_claims",
    ),
)

PLAN_RUN_RESULT_SCHEMA = object_schema(
    {
        "scenario": {"type": "object"},
        "identity": {"type": "object"},
        "safe_validation": {"type": "object"},
        "resource_estimate": {"type": "object"},
        "phases": {"type": "array", "items": {"type": "string"}},
        "approval": {"type": "object"},
        "execution_authorized": {"const": False},
    },
    required=(
        "scenario",
        "identity",
        "safe_validation",
        "resource_estimate",
        "phases",
        "approval",
        "execution_authorized",
    ),
)

VALIDATE_SCENARIO_RESULT_SCHEMA = object_schema(
    {
        "status": {"enum": ["validated", "safe_only", "failed"]},
        "scenario": {"type": "object"},
        "identity": {"type": "object"},
        "safe_validation": {"type": "object"},
        "trusted_validation": {"type": "object"},
        "resource_estimate": {"type": "object"},
        "execution_ready": {"type": "boolean"},
        "execution_authorized": {"const": False},
        "next_step": {"type": "string"},
    },
    required=(
        "status",
        "scenario",
        "identity",
        "safe_validation",
        "trusted_validation",
        "resource_estimate",
        "execution_ready",
        "execution_authorized",
        "next_step",
    ),
)

RUN_SCENARIO_RESULT_SCHEMA = object_schema(
    {
        "status": {"enum": ["completed", "partial", "cancelled"]},
        "scenario": {"type": "object"},
        "identity": {"type": "object"},
        "output_dir": {"type": "string"},
        "run": {"type": "object"},
        "artifacts": {"type": "array", "items": {"type": "string"}},
        "manifest_path": {"type": "string"},
    },
    required=("status", "scenario", "identity", "output_dir", "run", "artifacts", "manifest_path"),
)

COMPARE_RUNS_RESULT_SCHEMA = object_schema(
    {
        "status": {"enum": ["completed", "partial"]},
        "base_output_dir": {"type": "string"},
        "candidate_output_dir": {"type": "string"},
        "metric_names": {"type": "array", "items": {"type": "string"}},
        "metrics": {"type": "object"},
        "deltas": {"type": "object"},
        "metric_status": {"type": "array", "items": {"type": "object"}},
        "summary": {"type": "object"},
    },
    required=(
        "status",
        "base_output_dir",
        "candidate_output_dir",
        "metric_names",
        "metrics",
        "deltas",
        "metric_status",
        "summary",
    ),
)

PLOT_EVIDENCE_RESULT_SCHEMA = object_schema(
    {
        "status": {"enum": ["completed", "partial"]},
        "output_dir": {"type": "string"},
        "recipe_id": {"type": "string"},
        "artifact": {"type": "object"},
        "manifest_path": {"type": "string"},
    },
    required=("status", "output_dir", "recipe_id", "artifact", "manifest_path"),
)

PLAN_REVIEW_PLOT_RESULT_SCHEMA = object_schema(
    {
        "status": {"const": "planned"},
        "output_dir": {"type": "string"},
        "review_store": {"const": "review/run.sqlite"},
        "plot_plan_id": {"type": "string"},
        "spec": {"type": "object"},
        "columns": {"type": "array", "items": {"type": "string"}},
        "numeric_columns": {"type": "array", "items": {"type": "string"}},
        "row_count": {"type": "integer", "minimum": 1, "maximum": 5000},
        "truncated": {"type": "boolean"},
        "warnings": {"type": "array", "items": {"type": "string"}},
        "render_authorized": {"const": False},
        "visual_review_required": {"const": True},
    },
    required=(
        "status",
        "output_dir",
        "review_store",
        "plot_plan_id",
        "spec",
        "columns",
        "numeric_columns",
        "row_count",
        "truncated",
        "warnings",
        "render_authorized",
        "visual_review_required",
    ),
)

RENDER_REVIEW_PLOT_RESULT_SCHEMA = object_schema(
    {
        "status": {"enum": ["completed", "partial"]},
        "output_dir": {"type": "string"},
        "plot_plan_id": {"type": "string"},
        "artifact": {"type": "object"},
        "manifest_path": {"type": "string"},
    },
    required=("status", "output_dir", "plot_plan_id", "artifact", "manifest_path"),
)

PLAN_REVIEW_ANIMATION_RESULT_SCHEMA = object_schema(
    {
        "status": {"const": "planned"},
        "output_dir": {"type": "string"},
        "review_store": {"const": "review/run.sqlite"},
        "animation_plan_id": {"type": "string"},
        "spec": {"type": "object"},
        "recipe": {"type": "object"},
        "row_count": {"type": "integer", "minimum": 1, "maximum": 5000},
        "truncated": {"type": "boolean"},
        "source_frame_count": {"type": "integer", "minimum": 1},
        "render_frame_count": {"type": "integer", "minimum": 1, "maximum": 600},
        "effective_frame_stride": {"type": "integer", "minimum": 1},
        "encoded_duration_s": {"type": "number", "minimum": 0, "maximum": 30},
        "resource_estimate": {"type": "object"},
        "warnings": {"type": "array", "items": {"type": "string"}},
        "render_ready": {"type": "boolean"},
        "render_authorized": {"const": False},
        "visual_review_required": {"const": True},
    },
    required=(
        "status",
        "output_dir",
        "review_store",
        "animation_plan_id",
        "spec",
        "recipe",
        "row_count",
        "truncated",
        "source_frame_count",
        "render_frame_count",
        "effective_frame_stride",
        "encoded_duration_s",
        "resource_estimate",
        "warnings",
        "render_ready",
        "render_authorized",
        "visual_review_required",
    ),
)

RENDER_REVIEW_ANIMATION_RESULT_SCHEMA = object_schema(
    {
        "status": {"enum": ["completed", "partial"]},
        "output_dir": {"type": "string"},
        "animation_plan_id": {"type": "string"},
        "artifact": {"type": "object"},
        "manifest_path": {"type": "string"},
    },
    required=("status", "output_dir", "animation_plan_id", "artifact", "manifest_path"),
)

RUN_AGENT_TASK_RESULT_SCHEMA = object_schema(
    {
        "status": {"enum": ["completed", "partial", "failed", "cancelled"]},
        "recipe_id": {"type": "string"},
        "recipe_maturity": {"const": "supported"},
        "output_dir": {"type": "string"},
        "evidence_summary": {"type": "object"},
        "packet_path": {"type": "string"},
        "manifest_path": {"type": "string"},
        "artifacts": {"type": "array", "items": {"type": "object"}},
        "failure_hints": {"type": "array", "items": {"type": "object"}},
    },
    required=(
        "status",
        "recipe_id",
        "recipe_maturity",
        "output_dir",
        "evidence_summary",
        "packet_path",
        "manifest_path",
        "artifacts",
        "failure_hints",
    ),
)

PREPARE_REPORT_PACKET_RESULT_SCHEMA = object_schema(
    {
        "status": {"enum": ["completed", "partial"]},
        "packet_id": {"type": "string"},
        "source_output_dir": {"type": "string"},
        "packet_output_dir": {"type": "string"},
        "packet_path": {"type": "string"},
        "brief_path": {"type": "string"},
        "manifest_path": {"type": "string"},
        "packet_sha256": {"type": "string"},
        "artifact_count": {"type": "integer", "minimum": 0, "maximum": 100},
        "evidence_summary": {"type": "object"},
        "provider_call_made": {"const": False},
        "non_claims": {"type": "array", "items": {"type": "string"}},
    },
    required=(
        "status",
        "packet_id",
        "source_output_dir",
        "packet_output_dir",
        "packet_path",
        "brief_path",
        "manifest_path",
        "packet_sha256",
        "artifact_count",
        "evidence_summary",
        "provider_call_made",
        "non_claims",
    ),
)

AUDIT_REPORT_RESULT_SCHEMA = object_schema(
    {
        "status": {"enum": ["passed", "needs_review"]},
        "packet_id": {"type": "string"},
        "report_path": {"type": "string"},
        "packet_path": {"type": "string"},
        "audit_output_dir": {"type": "string"},
        "audit_json_path": {"type": "string"},
        "audit_markdown_path": {"type": "string"},
        "manifest_path": {"type": "string"},
        "checks": {"type": "object"},
        "unknown_evidence_references": {"type": "array", "items": {"type": "string"}},
        "unavailable_evidence_references": {"type": "array", "items": {"type": "string"}},
        "missing_required_sections": {"type": "array", "items": {"type": "string"}},
        "provider_call_made": {"const": False},
        "semantic_claim_review_performed": {"const": False},
        "non_claims": {"type": "array", "items": {"type": "string"}},
    },
    required=(
        "status",
        "packet_id",
        "report_path",
        "packet_path",
        "audit_output_dir",
        "audit_json_path",
        "audit_markdown_path",
        "manifest_path",
        "checks",
        "unknown_evidence_references",
        "unavailable_evidence_references",
        "missing_required_sections",
        "provider_call_made",
        "semantic_claim_review_performed",
        "non_claims",
    ),
)

INSPECT_HANDOFF_RESULT_SCHEMA = object_schema(
    {
        "document_type": {"enum": ["product", "manifest", "unknown"]},
        "schema_id": {"type": "string"},
        "schema_version": {},
        "identifier": {"type": "string"},
        "product_kind": {"type": "string"},
        "quality": {"type": "object"},
        "freshness": {"type": "object"},
        "validation": {"type": "object"},
        "supported_next_actions": {"type": "array", "items": {"type": "string"}},
        "source_path": {"type": "string"},
    },
    required=(
        "document_type",
        "schema_id",
        "schema_version",
        "identifier",
        "product_kind",
        "quality",
        "freshness",
        "validation",
        "supported_next_actions",
        "source_path",
    ),
)

EXPORT_RUN_PRODUCT_RESULT_SCHEMA = object_schema(
    {
        "status": {"const": "completed"},
        "product_kind": {
            "enum": ["completed_run_state", "completed_run_snapshot", "maneuver_detection"]
        },
        "product_path": {"type": "string"},
        "product_id": {"type": "string"},
        "selection": {"type": "object"},
        "object_ids": {"type": "array", "items": {"type": "string"}},
        "event_id": {"type": "string"},
        "operation_manifest_path": {"type": "string"},
        "execution_occurred": {"const": False},
    },
    required=(
        "status",
        "product_kind",
        "product_path",
        "product_id",
        "selection",
        "object_ids",
        "event_id",
        "operation_manifest_path",
        "execution_occurred",
    ),
)

EMIT_SCENARIO_OVERLAY_RESULT_SCHEMA = object_schema(
    {
        "status": {"const": "completed"},
        "product_path": {"type": "string"},
        "product_id": {"type": "string"},
        "overlay_id": {"type": "string"},
        "operation_count": {"type": "integer", "minimum": 1},
        "operation_manifest_path": {"type": "string"},
        "execution_occurred": {"const": False},
    },
    required=("status", "product_path", "product_id", "overlay_id", "operation_count", "operation_manifest_path", "execution_occurred"),
)

MATERIALIZE_HANDOFF_RESULT_SCHEMA = object_schema(
    {
        "status": {"enum": ["materialized", "blocked"]},
        "scenario_path": {"type": "string"},
        "manifest_path": {"type": "string"},
        "manifest_id": {"type": "string"},
        "source_product_id": {"type": "string"},
        "validation": {"type": "object"},
        "failures": {"type": "array", "items": {"type": "object"}},
        "recommended_next_action": {"type": "string"},
        "operation_manifest_path": {"type": "string"},
        "execution_occurred": {"const": False},
        "execution_authorized": {"const": False},
    },
    required=(
        "status",
        "scenario_path",
        "manifest_path",
        "manifest_id",
        "source_product_id",
        "validation",
        "failures",
        "recommended_next_action",
        "operation_manifest_path",
        "execution_occurred",
        "execution_authorized",
    ),
)

COMPARE_HANDOFF_RESULT_SCHEMA = object_schema(
    {
        "status": {"enum": ["equivalent", "failed"]},
        "comparison_id": {"type": "string"},
        "output_path": {"type": "string"},
        "source": {"type": "object"},
        "materialization": {"type": "object"},
        "summary": {"type": "object"},
        "execution_evidence": {"type": "object"},
        "non_claims": {"type": "array", "items": {"type": "string"}},
        "operation_manifest_path": {"type": "string"},
    },
    required=(
        "status",
        "comparison_id",
        "output_path",
        "source",
        "materialization",
        "summary",
        "execution_evidence",
        "non_claims",
        "operation_manifest_path",
    ),
)

MANEUVER_READINESS_RESULT_SCHEMA = object_schema(
    {
        "status": {"const": "completed"},
        "verdict": {"enum": ["ready", "not_ready", "unknown"]},
        "object_id": {"type": "string"},
        "chief_id": {"type": "string"},
        "thresholds": {"type": "object"},
        "metrics": {"type": "object"},
        "gates": {"type": "array", "items": {"type": "object"}},
        "packet_path": {"type": "string"},
        "operation_manifest_path": {"type": "string"},
        "non_claims": {"type": "array", "items": {"type": "string"}},
    },
    required=(
        "status",
        "verdict",
        "object_id",
        "chief_id",
        "thresholds",
        "metrics",
        "gates",
        "packet_path",
        "operation_manifest_path",
        "non_claims",
    ),
)

DYNAMICS_OD_RESULT_SCHEMA = object_schema(
    {
        "status": {"enum": ["completed", "partial", "failed", "cancelled"]},
        "method": {"type": "string"},
        "quality_gates": {"type": "object"},
        "verdict": {"type": "object"},
        "fit_metrics": {"type": "object"},
        "holdout_metrics": {"type": "object"},
        "maneuver_detection": {"type": "object"},
        "resource_plan": {"type": "object"},
        "state_product_path": {"type": "string"},
        "artifacts": {"type": "array", "items": {"type": "string"}},
        "operation_manifest_path": {"type": "string"},
        "execution_authorized_outputs": {"const": False},
        "non_claims": {"type": "array", "items": {"type": "string"}},
    },
    required=(
        "status", "method", "quality_gates", "verdict", "fit_metrics", "holdout_metrics",
        "maneuver_detection", "resource_plan", "state_product_path", "artifacts", "operation_manifest_path",
        "execution_authorized_outputs", "non_claims",
    ),
)

HANDOFF_SELECTOR_PROPERTIES: dict[str, Any] = {
    "selector": {"type": "string", "enum": ["final", "sample_index", "time_s", "event"]},
    "sample_index": {"type": "integer", "minimum": 0},
    "time_s": {"type": "number", "minimum": 0.0},
    "event_id": {"type": "string", "minLength": 1},
    "epoch_jd_utc": {"type": "number", "minimum": 0.0},
}

READINESS_THRESHOLDS_SCHEMA = object_schema(
    {
        "max_final_range_km": {"type": "number", "minimum": 0.0},
        "max_allocation_force_residual_n": {"type": "number", "minimum": 0.0},
        "max_allocation_saturated_duration_s": {"type": "number", "minimum": 0.0},
        "max_pointing_error_deg": {"type": "number", "minimum": 0.0},
        "min_final_propellant_kg": {"type": "number", "minimum": 0.0},
        "min_burn_samples": {"type": "number", "minimum": 0.0},
        "require_no_attitude_guardrail_events": {"type": "boolean"},
    },
    required=(),
)

PUBLIC_TOOL_CONTRACTS: tuple[ToolContract, ...] = (
    ToolContract(
        tool_id="oel.describe_capabilities.v1",
        title="Describe OEL MCP capabilities",
        description="List the OEL MCP tools available in the active deployment profile, including maturity and non-claims.",
        risk_class="R0_read",
        oel_api="integration-owned registry",
        maturity="supported",
        install_profile="mcp",
        deployment_profiles=PUBLIC_PROFILES,
        data_classes=("capability_metadata",),
        writes=False,
        input_schema=object_schema({}, required=()),
        result_schema=DESCRIBE_RESULT_SCHEMA,
    ),
    ToolContract(
        tool_id="oel.inspect_run.v1",
        title="Inspect completed OEL run",
        description="Inspect an existing OEL output directory without writing a new evidence packet.",
        risk_class="R0_read",
        oel_api="sim.agent_task.runner.inspect_output",
        maturity="supported",
        install_profile="mcp",
        deployment_profiles=PUBLIC_PROFILES,
        data_classes=("run_evidence",),
        writes=False,
        input_schema=object_schema(
            handling_properties(
                {
                    "output_dir": {"type": "string", "minLength": 1},
                    "query_names": {"type": "array", "items": {"type": "string"}, "maxItems": 20},
                    "max_rows": {"type": "integer", "minimum": 1, "maximum": MAX_ROWS, "default": 50},
                }
            ),
            required=("output_dir", "handling"),
        ),
        result_schema=INSPECT_RUN_RESULT_SCHEMA,
    ),
    ToolContract(
        tool_id="oel.query_review.v1",
        title="Query OEL review evidence",
        description="Run one bounded read-only SELECT or WITH query against an existing OEL review store.",
        risk_class="R0_read",
        oel_api="sim.review.ReviewWorkspace.query",
        maturity="supported",
        install_profile="mcp",
        deployment_profiles=PUBLIC_PROFILES,
        data_classes=("run_evidence",),
        writes=False,
        input_schema=object_schema(
            handling_properties(
                {
                    "output_dir": {"type": "string", "minLength": 1},
                    "sql": {"type": "string", "minLength": 1, "maxLength": 100_000},
                    "max_rows": {"type": "integer", "minimum": 1, "maximum": MAX_ROWS, "default": 100},
                    "max_vm_steps": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": MAX_VM_STEPS,
                        "default": 250_000,
                    },
                }
            ),
            required=("output_dir", "sql", "handling"),
        ),
        result_schema=QUERY_REVIEW_RESULT_SCHEMA,
        limits={
            "max_review_value_bytes": MAX_REVIEW_VALUE_BYTES,
            "max_review_result_bytes": MAX_REVIEW_RESULT_BYTES,
        },
    ),
    ToolContract(
        tool_id="oel.plan_run.v1",
        title="Plan a bounded OEL scenario run",
        description="Inspect a trusted local scenario structurally and report its normalized identity, phases, and resource estimate without executing it.",
        risk_class="R0_read",
        oel_api="sim.api.SimulationWorkspace.validate_safe; sim.resource_limits.estimate_resource_requirements",
        maturity="supported",
        install_profile="mcp",
        deployment_profiles=M4_LOCAL_PROFILES,
        data_classes=("scenario_config", "resource_estimate"),
        writes=False,
        input_schema=object_schema(
            handling_properties(dict(SCENARIO_INPUT_PROPERTIES)), required=("config_path", "output_dir", "handling")
        ),
        result_schema=PLAN_RUN_RESULT_SCHEMA,
    ),
    ToolContract(
        tool_id="oel.validate_scenario.v1",
        title="Validate an OEL scenario for a bounded run",
        description="Run safe validation first and optionally trusted plugin validation, returning a content-bound validation id without authorizing execution.",
        risk_class="R0_read",
        oel_api="sim.api.SimulationWorkspace.validate",
        maturity="supported",
        install_profile="mcp",
        deployment_profiles=M4_LOCAL_PROFILES,
        data_classes=("scenario_config", "validation_evidence"),
        writes=False,
        input_schema=object_schema(
            handling_properties(
                {
                    **SCENARIO_INPUT_PROPERTIES,
                    "trust_plugins": {"type": "boolean", "default": False},
                    "trust_approval": APPROVAL_SCHEMA,
                }
            ),
            required=("config_path", "output_dir", "trust_plugins", "handling"),
        ),
        result_schema=VALIDATE_SCENARIO_RESULT_SCHEMA,
    ),
    ToolContract(
        tool_id="oel.run_scenario.v1",
        title="Run one validated deterministic OEL scenario",
        description="Execute a content-bound, trusted, resource-preflighted public scenario into a new approved output directory.",
        risk_class="R2_execute",
        oel_api="sim.execution.run_simulation_config_file",
        maturity="supported",
        install_profile="mcp",
        deployment_profiles=M4_LOCAL_PROFILES,
        data_classes=("scenario_config", "run_evidence"),
        writes=True,
        executes=True,
        input_schema=object_schema(
            handling_properties(
                {
                    **SCENARIO_INPUT_PROPERTIES,
                    "validation_id": {"type": "string", "minLength": 1},
                    "trust_approval": APPROVAL_SCHEMA,
                    "approval": APPROVAL_SCHEMA,
                }
            ),
            required=("config_path", "output_dir", "resource_profile", "validation_id", "trust_approval", "approval", "handling"),
        ),
        result_schema=RUN_SCENARIO_RESULT_SCHEMA,
    ),
    ToolContract(
        tool_id="oel.compare_runs.v1",
        title="Compare two completed OEL runs",
        description="Compare allowlisted semantic metrics from two existing review stores without rerunning either scenario.",
        risk_class="R0_read",
        oel_api="sim.agent_task.runner.compare_outputs",
        maturity="supported",
        install_profile="mcp",
        deployment_profiles=M4_LOCAL_PROFILES,
        data_classes=("run_evidence",),
        writes=False,
        input_schema=object_schema(
            handling_properties(
                {
                    "base_output_dir": {"type": "string", "minLength": 1},
                    "candidate_output_dir": {"type": "string", "minLength": 1},
                    "metric_names": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 20},
                    "max_rows": {"type": "integer", "minimum": 1, "maximum": 200, "default": 50},
                }
            ),
            required=("base_output_dir", "candidate_output_dir", "metric_names", "handling"),
        ),
        result_schema=COMPARE_RUNS_RESULT_SCHEMA,
    ),
    ToolContract(
        tool_id="oel.plot_evidence.v1",
        title="Plot completed OEL review evidence",
        description=(
            "Generate one supported OEL plot recipe from completed review evidence. "
            "Use this before host-native visualization tools for OEL review-store data."
        ),
        risk_class="R1_write",
        oel_api="sim.agent_task.runner.create_plot",
        maturity="supported",
        install_profile="mcp",
        deployment_profiles=M4_LOCAL_PROFILES,
        data_classes=("run_evidence", "plot_artifact"),
        writes=True,
        input_schema=object_schema(
            handling_properties(
                {
                    "output_dir": {"type": "string", "minLength": 1},
                    "recipe_id": {"type": "string", "enum": list(SUPPORTED_REVIEW_PLOT_RECIPE_IDS)},
                    "style": {"type": "string", "enum": ["oel_dark", "oel_light"], "default": "oel_dark"},
                    "format": {"type": "string", "enum": ["png", "svg", "pdf"], "default": "png"},
                    "artifact_id": {"type": "string", "minLength": 1, "maxLength": 80},
                    "approval": APPROVAL_SCHEMA,
                }
            ),
            required=("output_dir", "recipe_id", "artifact_id", "approval", "handling"),
        ),
        result_schema=PLOT_EVIDENCE_RESULT_SCHEMA,
    ),
    ToolContract(
        tool_id="oel.plan_review_plot.v1",
        title="Plan a custom OEL review plot",
        description=(
            "Validate one typed, read-only review query and plot mapping without writing an artifact. "
            "Use this for OEL review-store data when no supported plot recipe matches."
        ),
        risk_class="R0_read",
        oel_api="sim.review.plan_review_plot",
        maturity="supported",
        install_profile="mcp",
        deployment_profiles=M4_LOCAL_PROFILES,
        data_classes=("run_evidence", "plot_specification"),
        writes=False,
        input_schema=object_schema(
            handling_properties(REVIEW_PLOT_SPEC_PROPERTIES),
            required=(
                "output_dir",
                "sql",
                "x_column",
                "y_columns",
                "plot_type",
                "artifact_id",
                "handling",
            ),
        ),
        result_schema=PLAN_REVIEW_PLOT_RESULT_SCHEMA,
        limits={"read_only_query": True, "render_authorized": False, "max_rows": 5000},
    ),
    ToolContract(
        tool_id="oel.render_review_plot.v2",
        title="Render a planned custom OEL review plot",
        description=(
            "Render one content-bound typed OEL review plot with style, provenance, and automated QA. "
            "Use this before host-native visualization tools for OEL review-store data."
        ),
        risk_class="R1_write",
        oel_api="sim.review.render_review_plot",
        maturity="supported",
        install_profile="mcp",
        deployment_profiles=M4_LOCAL_PROFILES,
        data_classes=("run_evidence", "plot_specification", "plot_artifact"),
        writes=True,
        input_schema=object_schema(
            handling_properties(
                {
                    **REVIEW_PLOT_SPEC_PROPERTIES,
                    "plot_plan_id": {"type": "string", "minLength": 1, "maxLength": 128},
                    "approval": APPROVAL_SCHEMA,
                }
            ),
            required=(
                "output_dir",
                "sql",
                "x_column",
                "y_columns",
                "plot_type",
                "artifact_id",
                "plot_plan_id",
                "approval",
                "handling",
            ),
        ),
        result_schema=RENDER_REVIEW_PLOT_RESULT_SCHEMA,
        limits={"one_plot_per_call": True, "operator_approval_required": True, "content_bound_plan": True},
    ),
    ToolContract(
        tool_id="oel.plan_review_animation.v1",
        title="Plan an OEL review animation",
        description=(
            "Validate one supported, read-only review animation recipe and compute a bounded frame plan "
            "without writing an artifact."
        ),
        risk_class="R0_read",
        oel_api="sim.review.plan_review_animation",
        maturity="supported",
        install_profile="mcp",
        deployment_profiles=M4_LOCAL_PROFILES,
        data_classes=("run_evidence", "animation_specification"),
        writes=False,
        input_schema=object_schema(
            handling_properties(REVIEW_ANIMATION_SPEC_PROPERTIES),
            required=("output_dir", "recipe_id", "artifact_id", "handling"),
        ),
        result_schema=PLAN_REVIEW_ANIMATION_RESULT_SCHEMA,
        limits={
            "read_only_query": True,
            "render_authorized": False,
            "max_rows": 5000,
            "max_frames": 600,
            "max_duration_s": 30,
        },
    ),
    ToolContract(
        tool_id="oel.render_review_animation.v1",
        title="Render a planned OEL review animation",
        description=(
            "Render one content-bound OEL review animation with stable formatting, temporal checks, "
            "encoding verification, a quality receipt, and a contact sheet."
        ),
        risk_class="R1_write",
        oel_api="sim.review.render_review_animation",
        maturity="supported",
        install_profile="mcp",
        deployment_profiles=M4_LOCAL_PROFILES,
        data_classes=("run_evidence", "animation_specification", "animation_artifact"),
        writes=True,
        input_schema=object_schema(
            handling_properties(
                {
                    **REVIEW_ANIMATION_SPEC_PROPERTIES,
                    "animation_plan_id": {"type": "string", "minLength": 1, "maxLength": 160},
                    "approval": APPROVAL_SCHEMA,
                }
            ),
            required=(
                "output_dir",
                "recipe_id",
                "artifact_id",
                "animation_plan_id",
                "approval",
                "handling",
            ),
        ),
        result_schema=RENDER_REVIEW_ANIMATION_RESULT_SCHEMA,
        limits={
            "one_animation_per_call": True,
            "operator_approval_required": True,
            "content_bound_plan": True,
            "max_frames": 600,
            "max_duration_s": 30,
            "max_file_bytes": 100_000_000,
        },
    ),
    ToolContract(
        tool_id="oel.run_agent_task.v1",
        title="Run one supported public OEL agent task",
        description="Execute one checked-in supported public task recipe with bounded review rows and optional allowlisted plots.",
        risk_class="R2_execute",
        oel_api="sim.agent_task.runner.run_recipe",
        maturity="supported",
        install_profile="mcp",
        deployment_profiles=M4_LOCAL_PROFILES,
        data_classes=("scenario_config", "run_evidence", "task_evidence_packet"),
        writes=True,
        executes=True,
        input_schema=object_schema(
            handling_properties(
                {
                    "recipe_id": {"type": "string", "minLength": 1},
                    "output_dir": {"type": "string", "minLength": 1},
                    "resource_profile": {
                        "type": "string",
                        "enum": list(M4_RESOURCE_PROFILES),
                        "default": "laptop-safe",
                    },
                    "make_plots": {"type": "boolean", "default": False},
                    "style": {"type": "string", "enum": ["oel_dark", "oel_light"], "default": "oel_dark"},
                    "max_rows": {"type": "integer", "minimum": 1, "maximum": 200, "default": 50},
                    "approval": APPROVAL_SCHEMA,
                }
            ),
            required=("recipe_id", "output_dir", "resource_profile", "approval", "handling"),
        ),
        result_schema=RUN_AGENT_TASK_RESULT_SCHEMA,
    ),
    ToolContract(
        tool_id="oel.prepare_report_packet.v1",
        title="Prepare a provider-neutral OEL report evidence packet",
        description="Write a bounded, hashed evidence packet and authoring brief from one completed local OEL run without calling a model provider.",
        risk_class="R1_write",
        oel_api="sim.agent_task.runner.inspect_output; integration-owned packet writer",
        maturity="supported",
        install_profile="mcp",
        deployment_profiles=M4_LOCAL_PROFILES,
        data_classes=("run_evidence", "report_evidence_packet"),
        writes=True,
        input_schema=object_schema(
            handling_properties(
                {
                    "source_output_dir": {"type": "string", "minLength": 1},
                    "packet_output_dir": {"type": "string", "minLength": 1},
                    "packet_id": {"type": "string", "minLength": 1, "maxLength": 80},
                    "query_names": {"type": "array", "items": {"type": "string"}, "maxItems": 20},
                    "max_rows": {"type": "integer", "minimum": 1, "maximum": 100, "default": 50},
                    "approval": APPROVAL_SCHEMA,
                }
            ),
            required=("source_output_dir", "packet_output_dir", "packet_id", "approval", "handling"),
        ),
        result_schema=PREPARE_REPORT_PACKET_RESULT_SCHEMA,
    ),
    ToolContract(
        tool_id="oel.audit_report.v1",
        title="Audit an agent-authored report against an OEL evidence packet",
        description="Verify packet and artifact hashes, report structure, and explicit evidence references without calling a model or judging narrative claims.",
        risk_class="R1_write",
        oel_api="integration-owned deterministic report audit",
        maturity="supported",
        install_profile="mcp",
        deployment_profiles=M4_LOCAL_PROFILES,
        data_classes=("report_evidence_packet", "agent_authored_report", "report_audit"),
        writes=True,
        input_schema=object_schema(
            handling_properties(
                {
                    "report_path": {"type": "string", "minLength": 1},
                    "packet_path": {"type": "string", "minLength": 1},
                    "max_packet_bytes": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 2_000_000,
                        "default": 2_000_000,
                    },
                    "audit_output_dir": {"type": "string", "minLength": 1},
                    "author": {"type": "string", "minLength": 1, "maxLength": 120},
                    "model": {"type": "string", "maxLength": 200, "default": ""},
                    "approval": APPROVAL_SCHEMA,
                }
            ),
            required=("report_path", "packet_path", "audit_output_dir", "author", "approval", "handling"),
        ),
        result_schema=AUDIT_REPORT_RESULT_SCHEMA,
    ),
    ToolContract(
        tool_id="oel.inspect_handoff.v1",
        title="Inspect OEL interchange evidence",
        description="Inspect and validate one bounded OEL product or handoff manifest without writing or executing.",
        risk_class="R0_read",
        oel_api="sim.handoff.inspect_path",
        maturity="supported",
        install_profile="mcp",
        deployment_profiles=M4_LOCAL_PROFILES,
        data_classes=("interchange_product", "handoff_manifest"),
        writes=False,
        input_schema=object_schema(
            handling_properties(
                {
                    "path": {"type": "string", "minLength": 1},
                    "verify_sources": {"type": "boolean", "default": True},
                }
            ),
            required=("path", "handling"),
        ),
        result_schema=INSPECT_HANDOFF_RESULT_SCHEMA,
        limits={"max_input_file_bytes": MAX_MANIFEST_BYTES},
    ),
    ToolContract(
        tool_id="oel.export_run_product.v1",
        title="Export a typed product from completed OEL evidence",
        description="Export one exact completed-run state, atomic snapshot, or maneuver-detection product without executing a scenario.",
        risk_class="R1_write",
        oel_api="sim.handoff completed-run exporters",
        maturity="supported",
        install_profile="mcp",
        deployment_profiles=M4_LOCAL_PROFILES,
        data_classes=("run_evidence", "interchange_product"),
        writes=True,
        input_schema=object_schema(
            handling_properties(
                {
                    "completed_run": {"type": "string", "minLength": 1},
                    "output_path": {"type": "string", "minLength": 1},
                    "product_kind": {
                        "type": "string",
                        "enum": ["completed_run_state", "completed_run_snapshot", "maneuver_detection"],
                    },
                    "object_id": {"type": "string", "minLength": 1},
                    "object_ids": {
                        "type": "array",
                        "items": {"type": "string", "minLength": 1},
                        "minItems": 2,
                        "maxItems": 32,
                    },
                    "observer_id": {"type": "string", "minLength": 1},
                    "target_id": {"type": "string", "minLength": 1},
                    **HANDOFF_SELECTOR_PROPERTIES,
                    "approval": APPROVAL_SCHEMA,
                }
            ),
            required=("completed_run", "output_path", "product_kind", "approval", "handling"),
        ),
        result_schema=EXPORT_RUN_PRODUCT_RESULT_SCHEMA,
        limits={"new_output_required": True, "max_objects": 32, "operator_approval_required": True},
    ),
    ToolContract(
        tool_id="oel.emit_scenario_overlay.v1",
        title="Emit a bounded OEL scenario overlay",
        description="Convert one closed scenario-capability overlay into a source-bound typed patch without materializing or executing it.",
        risk_class="R1_write",
        oel_api="sim.handoff.emit_scenario_overlay",
        maturity="supported",
        install_profile="mcp",
        deployment_profiles=M4_LOCAL_PROFILES,
        data_classes=("scenario_config", "scenario_overlay", "interchange_product"),
        writes=True,
        input_schema=object_schema(
            handling_properties(
                {
                    "source_scenario": {"type": "string", "minLength": 1},
                    "overlay_path": {"type": "string", "minLength": 1},
                    "overlay_id": {"type": "string", "minLength": 1, "maxLength": 120},
                    "rationale": {"type": "string", "minLength": 1, "maxLength": 1000},
                    "output_path": {"type": "string", "minLength": 1},
                    "approval": APPROVAL_SCHEMA,
                }
            ),
            required=(
                "source_scenario",
                "overlay_path",
                "overlay_id",
                "rationale",
                "output_path",
                "approval",
                "handling",
            ),
        ),
        result_schema=EMIT_SCENARIO_OVERLAY_RESULT_SCHEMA,
        limits={"new_output_required": True, "operator_approval_required": True},
    ),
    ToolContract(
        tool_id="oel.materialize_onp_handoff.v1",
        title="Materialize an ONP handoff scenario",
        description="Materialize and validate one passive ONP scenario from an accepted state or atomic snapshot product without executing it.",
        risk_class="R1_write",
        oel_api="sim.handoff.materialize_onp; sim.handoff.materialize_snapshot_onp",
        maturity="supported",
        install_profile="mcp",
        deployment_profiles=M4_LOCAL_PROFILES,
        data_classes=("interchange_product", "scenario_config", "handoff_manifest"),
        writes=True,
        input_schema=object_schema(
            handling_properties(
                {
                    "product_path": {"type": "string", "minLength": 1},
                    "scenario_name": {"type": "string", "minLength": 1, "maxLength": 160},
                    "scenario_path": {"type": "string", "minLength": 1},
                    "run_output_dir": {"type": "string", "minLength": 1},
                    "duration_s": {"type": "number", "minimum": 1e-9},
                    "dt_s": {"type": "number", "minimum": 1e-9},
                    "trust_plugins": {"type": "boolean", "default": False},
                    "trust_approval": APPROVAL_SCHEMA,
                    "approval": APPROVAL_SCHEMA,
                }
            ),
            required=(
                "product_path",
                "scenario_name",
                "scenario_path",
                "run_output_dir",
                "duration_s",
                "dt_s",
                "trust_plugins",
                "approval",
                "handling",
            ),
        ),
        result_schema=MATERIALIZE_HANDOFF_RESULT_SCHEMA,
        limits={"new_output_required": True, "execution_authorized": False, "operator_approval_required": True},
    ),
    ToolContract(
        tool_id="oel.materialize_scenario_patch.v1",
        title="Materialize a typed OEL scenario patch",
        description="Apply one accepted source-bound scenario patch and validate the generated scenario without executing it.",
        risk_class="R1_write",
        oel_api="sim.handoff.materialize_scenario_patch",
        maturity="supported",
        install_profile="mcp",
        deployment_profiles=M4_LOCAL_PROFILES,
        data_classes=("interchange_product", "scenario_config", "handoff_manifest"),
        writes=True,
        input_schema=object_schema(
            handling_properties(
                {
                    "patch_product": {"type": "string", "minLength": 1},
                    "source_scenario": {"type": "string", "minLength": 1},
                    "scenario_name": {"type": "string", "minLength": 1, "maxLength": 160},
                    "scenario_path": {"type": "string", "minLength": 1},
                    "run_output_dir": {"type": "string", "minLength": 1},
                    "trust_plugins": {"type": "boolean", "default": False},
                    "trust_approval": APPROVAL_SCHEMA,
                    "approval": APPROVAL_SCHEMA,
                }
            ),
            required=(
                "patch_product",
                "source_scenario",
                "scenario_name",
                "scenario_path",
                "run_output_dir",
                "trust_plugins",
                "approval",
                "handling",
            ),
        ),
        result_schema=MATERIALIZE_HANDOFF_RESULT_SCHEMA,
        limits={"new_output_required": True, "execution_authorized": False, "operator_approval_required": True},
    ),
    ToolContract(
        tool_id="oel.compare_handoff.v1",
        title="Compare OEL handoff semantics",
        description="Write a bounded semantic-parity packet for a product, scenario, manifest, and optional first consumer evidence row.",
        risk_class="R1_write",
        oel_api="sim.handoff.compare_handoff",
        maturity="supported",
        install_profile="mcp",
        deployment_profiles=M4_LOCAL_PROFILES,
        data_classes=("interchange_product", "scenario_config", "handoff_manifest", "run_evidence"),
        writes=True,
        input_schema=object_schema(
            handling_properties(
                {
                    "product_path": {"type": "string", "minLength": 1},
                    "scenario_path": {"type": "string", "minLength": 1},
                    "manifest_path": {"type": "string", "minLength": 1},
                    "run_output_dir": {"type": "string", "minLength": 1},
                    "output_path": {"type": "string", "minLength": 1},
                    "approval": APPROVAL_SCHEMA,
                }
            ),
            required=("product_path", "scenario_path", "output_path", "approval", "handling"),
        ),
        result_schema=COMPARE_HANDOFF_RESULT_SCHEMA,
        limits={"new_output_required": True, "operator_approval_required": True},
    ),
    ToolContract(
        tool_id="oel.assess_maneuver_readiness.v1",
        title="Assess bounded maneuver readiness evidence",
        description="Apply explicit engineering thresholds to one completed deterministic run and write a fail-closed readiness packet.",
        risk_class="R1_write",
        oel_api="sim.reporting.build_maneuver_readiness_packet",
        maturity="supported",
        install_profile="mcp",
        deployment_profiles=M4_LOCAL_PROFILES,
        data_classes=("run_evidence", "maneuver_readiness_packet"),
        writes=True,
        input_schema=object_schema(
            handling_properties(
                {
                    "completed_run": {"type": "string", "minLength": 1},
                    "object_id": {"type": "string", "minLength": 1},
                    "chief_id": {"type": "string", "minLength": 1},
                    "thresholds": READINESS_THRESHOLDS_SCHEMA,
                    "output_path": {"type": "string", "minLength": 1},
                    "approval": APPROVAL_SCHEMA,
                }
            ),
            required=("completed_run", "object_id", "chief_id", "thresholds", "output_path", "approval", "handling"),
        ),
        result_schema=MANEUVER_READINESS_RESULT_SCHEMA,
        limits={"new_output_required": True, "operator_approval_required": True},
    ),
    ToolContract(
        tool_id="oel.inspect_study.v1",
        title="Inspect a completed OEL study bundle",
        description="Verify and summarize one content-bound public OEL study bundle without executing analysis.",
        risk_class="R0_read",
        oel_api="sim.study.inspect_study_bundle",
        maturity="supported",
        install_profile="mcp",
        deployment_profiles=M4_LOCAL_PROFILES,
        data_classes=("study_bundle", "analysis_evidence"),
        writes=False,
        input_schema=object_schema(
            handling_properties({"bundle_dir": {"type": "string", "minLength": 1}}),
            required=("bundle_dir", "handling"),
        ),
        result_schema=INSPECT_STUDY_RESULT_SCHEMA,
        limits={"max_steps": 12, "max_evidence_file_bytes": 16 * 1024 * 1024},
    ),
    ToolContract(
        tool_id="oel.replay_study.v1",
        title="Replay OEL study bundle identity",
        description="Recompute the authoritative identity and citation bindings of one completed public OEL study bundle.",
        risk_class="R0_read",
        oel_api="sim.study.replay_study_bundle",
        maturity="supported",
        install_profile="mcp",
        deployment_profiles=M4_LOCAL_PROFILES,
        data_classes=("study_bundle", "analysis_evidence"),
        writes=False,
        input_schema=object_schema(
            handling_properties({"bundle_dir": {"type": "string", "minLength": 1}}),
            required=("bundle_dir", "handling"),
        ),
        result_schema=REPLAY_STUDY_RESULT_SCHEMA,
        limits={"identity_replay_only": True, "analysis_execution": False},
    ),
    ToolContract(
        tool_id="oel.compare_studies.v1",
        title="Compare two completed OEL study bundles",
        description="Compare content-bound study identities, root records, and evidence steps without rerunning analysis.",
        risk_class="R0_read",
        oel_api="sim.study.compare_study_bundles",
        maturity="supported",
        install_profile="mcp",
        deployment_profiles=M4_LOCAL_PROFILES,
        data_classes=("study_bundle", "analysis_evidence"),
        writes=False,
        input_schema=object_schema(
            handling_properties(
                {
                    "left_bundle_dir": {"type": "string", "minLength": 1},
                    "right_bundle_dir": {"type": "string", "minLength": 1},
                }
            ),
            required=("left_bundle_dir", "right_bundle_dir", "handling"),
        ),
        result_schema=COMPARE_STUDIES_RESULT_SCHEMA,
        limits={"analysis_execution": False},
    ),
    ToolContract(
        tool_id="oel.inspect_ccsds.v1",
        title="Inspect a bounded CCSDS navigation message",
        description="Parse and inspect one public OEM, ODM, TDM, or CDM file without converting or executing it.",
        risk_class="R0_read",
        oel_api="sim.ccsds.inspect_oem/inspect_odm/inspect_tdm/inspect_cdm",
        maturity="supported",
        install_profile="mcp",
        deployment_profiles=M4_LOCAL_PROFILES,
        data_classes=("ccsds_navigation_message",),
        writes=False,
        input_schema=object_schema(
            handling_properties(
                {
                    "path": {"type": "string", "minLength": 1},
                    "product_kind": {"type": "string", "enum": ["oem", "odm", "tdm", "cdm"]},
                }
            ),
            required=("path", "product_kind", "handling"),
        ),
        result_schema=INSPECT_CCSDS_RESULT_SCHEMA,
        limits={"max_input_file_bytes": MAX_MANIFEST_BYTES, "execution": False},
    ),
    ToolContract(
        tool_id="oel.convert_frame_time.v1",
        title="Inspect or convert bounded frame and time data",
        description="Convert one epoch, inspect one EOP source, or transform one Cartesian state or covariance under the public frame/time contract.",
        risk_class="R0_read",
        oel_api="sim.frame_time",
        maturity="supported",
        install_profile="mcp",
        deployment_profiles=M4_LOCAL_PROFILES,
        data_classes=("frame_time_input", "earth_orientation_data"),
        writes=False,
        input_schema=object_schema(
            handling_properties(
                {
                    "operation": {
                        "type": "string",
                        "enum": ["convert_epoch", "inspect_eop", "transform_state", "transform_covariance"],
                    },
                    "epoch": {"type": "string", "minLength": 1},
                    "from_scale": {"type": "string", "enum": ["UTC", "TAI", "TT", "UT1"]},
                    "to_scale": {"type": "string", "enum": ["UTC", "TAI", "TT", "UT1"]},
                    "time_scale": {"type": "string", "enum": ["UTC", "TAI", "TT", "UT1"]},
                    "dut1_s": {"type": "number", "minimum": -2.0, "maximum": 2.0},
                    "source_frame": {"type": "string", "enum": ["EME2000", "TEME", "ITRF", "GCRF"]},
                    "target_frame": {"type": "string", "enum": ["EME2000", "TEME", "ITRF", "GCRF"]},
                    "position_km": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3},
                    "velocity_km_s": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3},
                    "covariance": {
                        "type": "array",
                        "items": {"type": "array", "items": {"type": "number"}, "minItems": 6, "maxItems": 6},
                        "minItems": 6,
                        "maxItems": 6,
                    },
                    "eop_path": {"type": "string", "minLength": 1},
                    "eop_format": {"type": "string", "enum": ["auto", "finals2000a", "c04_csv"]},
                    "as_of": {"type": "string", "minLength": 1},
                    "max_observed_age_days": {"type": "number", "minimum": 0.0, "maximum": 3650.0},
                }
            ),
            required=("operation", "handling"),
        ),
        result_schema=FRAME_TIME_RESULT_SCHEMA,
        limits={"max_eop_file_bytes": MAX_MANIFEST_BYTES, "execution": False},
    ),
)

PUBLIC_TOOL_CONTRACTS = PUBLIC_TOOL_CONTRACTS + FSW_AUTHORING_TOOL_CONTRACTS + CR3BP_TOOL_CONTRACTS


def public_contracts_for_profile(profile: str) -> tuple[ToolContract, ...]:
    return tuple(contract for contract in PUBLIC_TOOL_CONTRACTS if profile in contract.deployment_profiles)


def public_contract_map(profile: str) -> dict[str, ToolContract]:
    return {contract.tool_id: contract for contract in public_contracts_for_profile(profile)}


def public_tool_definitions(profile: str) -> list[dict[str, Any]]:
    return [contract.mcp_definition() for contract in public_contracts_for_profile(profile)]


__all__ = [
    "HANDLING_SCHEMA",
    "M3_PUBLIC_TOOL_IDS",
    "FSW_AUTHORING_TOOL_IDS",
    "PUBLIC_TOOL_CONTRACTS",
    "public_contract_map",
    "public_contracts_for_profile",
    "public_tool_definitions",
]
