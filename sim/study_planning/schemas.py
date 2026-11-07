"""Versioned closed schemas for the transport-neutral OEL study lifecycle."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

SHA256_PATTERN = r"[0-9a-f]{64}"
IDENTIFIER_PATTERN = r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,159}"


def _object(properties: dict[str, Any], *, required: tuple[str, ...] = ()) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": list(required),
        "additionalProperties": False,
    }


def _text(*, maximum: int = 2000) -> dict[str, Any]:
    return {"type": "string", "minLength": 1, "maxLength": maximum}


def _identifier() -> dict[str, Any]:
    return {"type": "string", "pattern": IDENTIFIER_PATTERN}


def _sha256() -> dict[str, Any]:
    return {"type": "string", "pattern": SHA256_PATTERN}


def _string_array(*, maximum: int = 128) -> dict[str, Any]:
    return {"type": "array", "items": _text(maximum=500), "maxItems": maximum}


HANDLING_SCHEMA = _object(
    {
        "marking": _text(maximum=120),
        "release_scope": {"type": "string", "enum": ["public", "local_only", "frontier_eligible"]},
        "owner": {"type": "string", "maxLength": 200},
    },
    required=("marking", "release_scope"),
)

INPUT_REFERENCE_SCHEMA = _object(
    {
        "input_id": _identifier(),
        "kind": _text(maximum=80),
        "schema_id": _identifier(),
        "metadata": {"type": "object"},
        "content_sha256": {"type": ["string", "null"], "pattern": SHA256_PATTERN},
        "description": {"type": "string", "maxLength": 1000},
        "required": {"type": "boolean"},
    },
    required=("input_id", "kind", "required"),
)

CHOICE_SCHEMA = _object(
    {
        "choice_id": _identifier(),
        "description": _text(maximum=1000),
        "value": {},
        "source": {"type": "string", "enum": ["user", "host_default", "agent_proposal"]},
        "material": {"type": "boolean"},
    },
    required=("choice_id", "description", "value", "source", "material"),
)

CLARIFICATION_SCHEMA = _object(
    {
        "clarification_id": _identifier(),
        "question": _text(maximum=1000),
        "answer": {"type": ["string", "null"], "maxLength": 4000},
        "material": {"type": "boolean"},
    },
    required=("clarification_id", "question", "answer", "material"),
)

STUDY_REQUEST_SCHEMA = _object(
    {
        "schema": {"const": "oel.hosted_study_request.v1"},
        "request_id": _identifier(),
        "request_sha256": _sha256(),
        "question": _text(maximum=20_000),
        "inputs": {"type": "array", "items": INPUT_REFERENCE_SCHEMA, "maxItems": 128},
        "required_choices": {"type": "array", "items": CHOICE_SCHEMA, "maxItems": 256},
        "clarifications": {"type": "array", "items": CLARIFICATION_SCHEMA, "maxItems": 128},
        "handling": HANDLING_SCHEMA,
    },
    required=(
        "schema",
        "request_id",
        "request_sha256",
        "question",
        "inputs",
        "required_choices",
        "clarifications",
        "handling",
    ),
)

OPERATION_BOUNDS_SCHEMA = _object(
    {
        "max_iterations": {"type": "integer", "minimum": 1, "maximum": 1_000_000},
        "max_cases": {"type": "integer", "minimum": 1, "maximum": 10_000_000},
        "max_wall_time_s": {"type": "number", "minimum": 0.001, "maximum": 2_592_000},
        "max_cpu_time_s": {"type": "number", "minimum": 0.001, "maximum": 2_592_000},
        "max_work_units": {"type": "number", "minimum": 0.001, "maximum": 1_000_000_000_000_000},
        "max_peak_memory_mb": {"type": "number", "minimum": 0.001, "maximum": 1_048_576},
        "max_retained_output_bytes": {
            "type": "integer",
            "minimum": 0,
            "maximum": 10_995_116_277_760,
        },
        "max_input_bytes": {"type": "integer", "minimum": 1, "maximum": 17_179_869_184},
        "max_observations": {"type": "integer", "minimum": 1, "maximum": 10_000_000},
        "max_arc_duration_s": {"type": "number", "minimum": 0.001, "maximum": 31_536_000},
        "max_propagation_steps": {"type": "integer", "minimum": 1, "maximum": 1_000_000_000},
        "max_estimator_evaluations": {"type": "integer", "minimum": 1, "maximum": 1_000_000_000},
        "max_batch_evaluations": {"type": "integer", "minimum": 1, "maximum": 100_000_000},
        "max_stations": {"type": "integer", "minimum": 1, "maximum": 100_000},
    },
    required=("max_wall_time_s",),
)

ARTIFACT_REFERENCE_SCHEMA = _object(
    {
        "ref_id": _identifier(),
        "kind": _text(maximum=120),
        "port_id": _identifier(),
        "schema_id": _identifier(),
        "metadata": {"type": "object"},
    },
    required=("ref_id", "kind"),
)

OPERATION_SCHEMA = _object(
    {
        "operation_id": _identifier(),
        "capability_id": _identifier(),
        "depends_on": {"type": "array", "items": _identifier(), "maxItems": 64, "uniqueItems": True},
        "input_refs": {"type": "array", "items": ARTIFACT_REFERENCE_SCHEMA, "maxItems": 128},
        "output_refs": {"type": "array", "items": ARTIFACT_REFERENCE_SCHEMA, "maxItems": 128},
        "config_ref": {"type": ["string", "null"], "maxLength": 160},
        "parameters": {"type": "object"},
        "bounds": OPERATION_BOUNDS_SCHEMA,
    },
    required=(
        "operation_id",
        "capability_id",
        "depends_on",
        "input_refs",
        "output_refs",
        "config_ref",
        "parameters",
        "bounds",
    ),
)

CONFIG_CONSTRAINT_SCHEMA = _object(
    {
        "constraint_id": _identifier(),
        "config_ref": _identifier(),
        "path": {
            "type": "string",
            "minLength": 1,
            "maxLength": 500,
            "pattern": r"/(?:[^~/]|~[01])*(?:/(?:[^~/]|~[01])*)*",
            "description": (
                "RFC 6901 JSON Pointer into the normalized configuration. "
                "It must begin with '/'; use '/' between tokens, '~1' for '/', "
                "and '~0' for '~'. Dotted paths are invalid."
            ),
            "examples": ["/scenario_name", "/simulator/duration_s"],
        },
        "operator": {"type": "string", "enum": ["equals", "present", "absent", "one_of"]},
        "expected": {},
        "material": {"type": "boolean"},
        "rationale": _text(maximum=1000),
    },
    required=("constraint_id", "config_ref", "path", "operator", "expected", "material", "rationale"),
)

EVIDENCE_REQUIREMENT_SCHEMA = _object(
    {
        "evidence_id": _identifier(),
        "kind": _text(maximum=120),
        "description": _text(maximum=1000),
        "source_operation_ids": {
            "type": "array",
            "items": _identifier(),
            "minItems": 1,
            "maxItems": 64,
            "uniqueItems": True,
        },
    },
    required=("evidence_id", "kind", "description", "source_operation_ids"),
)

ACCEPTANCE_SCHEMA = _object(
    {
        "criterion_id": _identifier(),
        "description": _text(maximum=1000),
        "evidence_ids": {
            "type": "array",
            "items": _identifier(),
            "minItems": 1,
            "maxItems": 64,
            "uniqueItems": True,
        },
    },
    required=("criterion_id", "description", "evidence_ids"),
)

CLAIM_SCHEMA = _object(
    {
        "claim_id": _identifier(),
        "statement": _text(maximum=2000),
        "evidence_ids": {
            "type": "array",
            "items": _identifier(),
            "minItems": 1,
            "maxItems": 64,
            "uniqueItems": True,
        },
    },
    required=("claim_id", "statement", "evidence_ids"),
)

TRANSFER_SCHEMA = _object(
    {
        "input_id": _identifier(),
        "purpose": _text(maximum=1000),
        "required": {"type": "boolean"},
        "content_sha256": _sha256(),
    },
    required=("input_id", "purpose", "required", "content_sha256"),
)

RESOURCE_ENVELOPE_SCHEMA = _object(
    {
        "resource_profile": {"type": "string", "enum": ["laptop-safe", "standard"]},
        "max_operations": {"type": "integer", "minimum": 1, "maximum": 256},
        "max_total_cases": {"type": "integer", "minimum": 1, "maximum": 10_000_000},
        "max_wall_time_s": {"type": "number", "minimum": 0.001, "maximum": 2_592_000},
        "max_cpu_time_s": {"type": "number", "minimum": 0.001, "maximum": 2_592_000},
        "max_work_units": {"type": "number", "minimum": 0.001, "maximum": 1_000_000_000_000_000},
        "max_peak_memory_mb": {"type": "number", "minimum": 1, "maximum": 1_048_576},
        "max_retained_output_bytes": {
            "type": "integer",
            "minimum": 0,
            "maximum": 10_995_116_277_760,
        },
        "max_storage_mb": {"type": "number", "minimum": 0.001, "maximum": 10_485_760},
        "max_artifact_mb": {"type": "number", "minimum": 0.001, "maximum": 1_048_576},
        "max_parallel_workers": {"type": "integer", "minimum": 1, "maximum": 1024},
        "max_input_bytes": {"type": "integer", "minimum": 1, "maximum": 17_179_869_184},
        "max_observations": {"type": "integer", "minimum": 1, "maximum": 10_000_000},
        "max_arc_duration_s": {"type": "number", "minimum": 0.001, "maximum": 31_536_000},
        "max_propagation_steps": {"type": "integer", "minimum": 1, "maximum": 1_000_000_000},
        "max_estimator_evaluations": {"type": "integer", "minimum": 1, "maximum": 1_000_000_000},
        "max_batch_evaluations": {"type": "integer", "minimum": 1, "maximum": 100_000_000},
        "max_stations": {"type": "integer", "minimum": 1, "maximum": 100_000},
    },
    required=("resource_profile", "max_operations", "max_total_cases", "max_wall_time_s"),
)

RESOURCE_ACCOUNTING_METRIC_SCHEMA = _object(
    {
        "status": {"type": "string", "enum": ["assessed", "unassessed"]},
        "value": {"type": ["number", "null"]},
        "basis": _text(maximum=500),
        "issue_codes": _string_array(maximum=64),
    },
    required=("status", "value", "basis", "issue_codes"),
)

RESOURCE_ACCOUNTING_SCHEMA = _object(
    {
        "schema": {"const": "oel.study_resource_accounting.v1"},
        "metrics": _object(
            {
                "total_cases": RESOURCE_ACCOUNTING_METRIC_SCHEMA,
                "cpu_time_s": RESOURCE_ACCOUNTING_METRIC_SCHEMA,
                "work_units": RESOURCE_ACCOUNTING_METRIC_SCHEMA,
                "propagation_steps": RESOURCE_ACCOUNTING_METRIC_SCHEMA,
                "estimator_evaluations": RESOURCE_ACCOUNTING_METRIC_SCHEMA,
                "batch_evaluations": RESOURCE_ACCOUNTING_METRIC_SCHEMA,
                "arc_duration_s": RESOURCE_ACCOUNTING_METRIC_SCHEMA,
                "peak_memory_mb": RESOURCE_ACCOUNTING_METRIC_SCHEMA,
                "unique_input_bytes": RESOURCE_ACCOUNTING_METRIC_SCHEMA,
                "retained_output_bytes": RESOURCE_ACCOUNTING_METRIC_SCHEMA,
                "observation_count": RESOURCE_ACCOUNTING_METRIC_SCHEMA,
                "station_count": RESOURCE_ACCOUNTING_METRIC_SCHEMA,
            },
            required=(
                "total_cases",
                "cpu_time_s",
                "work_units",
                "propagation_steps",
                "estimator_evaluations",
                "batch_evaluations",
                "arc_duration_s",
                "peak_memory_mb",
                "unique_input_bytes",
                "retained_output_bytes",
            ),
        ),
        "worker_bound": {"type": ["integer", "null"]},
        "issue_codes": _string_array(maximum=128),
    },
    required=("schema", "metrics", "worker_bound", "issue_codes"),
)

STUDY_PLAN_SCHEMA = _object(
    {
        "schema": {"const": "oel.hosted_study_plan.v2"},
        "plan_id": _identifier(),
        "plan_sha256": _sha256(),
        "request_id": _identifier(),
        "request_sha256": _sha256(),
        "capability_manifest_sha256": _sha256(),
        "planner_version": _text(maximum=120),
        "question_answered": _text(maximum=20_000),
        "operations": {"type": "array", "items": OPERATION_SCHEMA, "minItems": 1, "maxItems": 128},
        "config_constraints": {
            "type": "array",
            "items": CONFIG_CONSTRAINT_SCHEMA,
            "maxItems": 1024,
        },
        "assumptions": _string_array(maximum=256),
        "allowed_defaults": _string_array(maximum=256),
        "forbidden_capabilities": {
            "type": "array",
            "items": _identifier(),
            "maxItems": 128,
            "uniqueItems": True,
        },
        "evidence_requirements": {
            "type": "array",
            "items": EVIDENCE_REQUIREMENT_SCHEMA,
            "minItems": 1,
            "maxItems": 256,
        },
        "acceptance_criteria": {
            "type": "array",
            "items": ACCEPTANCE_SCHEMA,
            "minItems": 1,
            "maxItems": 256,
        },
        "claims": {"type": "array", "items": CLAIM_SCHEMA, "maxItems": 256},
        "non_claims": _string_array(maximum=256),
        "transfers": {"type": "array", "items": TRANSFER_SCHEMA, "maxItems": 128},
        "resource_envelope": RESOURCE_ENVELOPE_SCHEMA,
    },
    required=(
        "schema",
        "plan_id",
        "plan_sha256",
        "request_id",
        "request_sha256",
        "capability_manifest_sha256",
        "planner_version",
        "question_answered",
        "operations",
        "config_constraints",
        "assumptions",
        "allowed_defaults",
        "forbidden_capabilities",
        "evidence_requirements",
        "acceptance_criteria",
        "claims",
        "non_claims",
        "transfers",
        "resource_envelope",
    ),
)

STUDY_RUN_SCHEMA = _object(
    {
        "schema": {"const": "oel.hosted_study_run.v1"},
        "run_id": _identifier(),
        "plan_id": _identifier(),
        "plan_sha256": _sha256(),
        "status": {"type": "string", "enum": ["accepted", "running", "completed", "failed", "cancelled", "interrupted"]},
        "operation_runs": {"type": "array", "items": {"type": "object"}, "maxItems": 256},
        "manifest_ref": _text(maximum=500),
    },
    required=("schema", "run_id", "plan_id", "plan_sha256", "status", "operation_runs", "manifest_ref"),
)

STUDY_EVIDENCE_SCHEMA = _object(
    {
        "schema": {"const": "oel.hosted_study_evidence.v1"},
        "evidence_id": _identifier(),
        "run_id": _identifier(),
        "plan_sha256": _sha256(),
        "complete": {"type": "boolean"},
        "artifacts": {"type": "array", "items": {"type": "object"}, "maxItems": 1024},
        "missing_evidence_ids": {"type": "array", "items": _identifier(), "maxItems": 256},
    },
    required=("schema", "evidence_id", "run_id", "plan_sha256", "complete", "artifacts", "missing_evidence_ids"),
)

STUDY_CLAIMS_SCHEMA = _object(
    {
        "schema": {"const": "oel.hosted_study_claims.v1"},
        "claims_id": _identifier(),
        "evidence_id": _identifier(),
        "supported_claims": {"type": "array", "items": CLAIM_SCHEMA, "maxItems": 256},
        "non_claims": _string_array(maximum=256),
        "ready_to_cite": {"type": "boolean"},
    },
    required=("schema", "claims_id", "evidence_id", "supported_claims", "non_claims", "ready_to_cite"),
)

STUDY_RECEIPT_SCHEMA = _object(
    {
        "schema": {"const": "oel.hosted_study_receipt.v1"},
        "receipt_id": _identifier(),
        "request_id": _identifier(),
        "plan_id": _identifier(),
        "plan_sha256": _sha256(),
        "run_id": {"type": ["string", "null"], "maxLength": 160},
        "disposition": {"type": "string", "enum": ["ready_for_approval", "unsupported", "completed", "failed", "cancelled", "interrupted"]},
        "price": _object(
            {
                "currency": {"const": "USD"},
                "oel_execution_amount": {"type": "number", "minimum": 0},
                "configuration_validation_amount": {"const": 0},
                "model_billing": {"const": "byo_provider_direct"},
            },
            required=("currency", "oel_execution_amount", "configuration_validation_amount", "model_billing"),
        ),
        "evidence_id": {"type": ["string", "null"], "maxLength": 160},
        "claims_id": {"type": ["string", "null"], "maxLength": 160},
    },
    required=("schema", "receipt_id", "request_id", "plan_id", "plan_sha256", "run_id", "disposition", "price", "evidence_id", "claims_id"),
)

CAPABILITY_REQUEST_SCHEMA = _object(
    {
        "schema": {"const": "oel.capability_request.v1"},
        "capability_request_id": _identifier(),
        "request_id": _identifier(),
        "refusal_category": {
            "type": "string",
            "enum": [
                "missing_physics",
                "missing_data",
                "unsupported_analysis",
                "insufficient_validation",
                "resource_limit",
                "deployment_limit",
                "planner_error",
            ],
        },
        "missing_capability": _text(maximum=500),
        "user_summary": _text(maximum=4000),
        "would_pay": {"type": ["boolean", "null"]},
        "contact_permission": {"type": "boolean"},
        "contact": {"type": ["string", "null"], "maxLength": 500},
        "versions": _object(
            {
                "planner": _text(maximum=120),
                "capability_manifest": _sha256(),
                "engine": _text(maximum=120),
            },
            required=("planner", "capability_manifest", "engine"),
        ),
        "attachments": {"type": "array", "items": INPUT_REFERENCE_SCHEMA, "maxItems": 16},
        "submission_authorized": {"const": False},
    },
    required=("schema", "capability_request_id", "request_id", "refusal_category", "missing_capability", "user_summary", "would_pay", "contact_permission", "contact", "versions", "attachments", "submission_authorized"),
)

EXECUTION_ROUTE_SCHEMA = _object(
    {
        "route": {
            "type": "string",
            "enum": ["LOCAL_FREE_AVAILABLE", "HOSTED_PRO_REQUIRED", "NOT_ELIGIBLE"],
        },
        "public_capability_ids": {
            "type": "array",
            "items": _identifier(),
            "maxItems": 128,
            "uniqueItems": True,
        },
        "pro_capability_ids": {
            "type": "array",
            "items": _identifier(),
            "maxItems": 128,
            "uniqueItems": True,
        },
        "unavailable_capability_ids": {
            "type": "array",
            "items": _identifier(),
            "maxItems": 128,
            "uniqueItems": True,
        },
        "local_free_available": {"type": "boolean"},
        "hosted_pro_required": {"type": "boolean"},
        "hosted_preflight_required": {"type": "boolean"},
    },
    required=(
        "route",
        "public_capability_ids",
        "pro_capability_ids",
        "unavailable_capability_ids",
        "local_free_available",
        "hosted_pro_required",
        "hosted_preflight_required",
    ),
)

COMMERCIAL_STATUS_SCHEMA = _object(
    {
        "currency": {"const": "USD"},
        "oel_execution_amount": {"const": 0},
        "configuration_validation_amount": {"const": 0},
        "model_billing": {"const": "byo_provider_direct"},
        "hosted_preflight_required": {"type": "boolean"},
        "payment_authorized": {"const": False},
        "policy_status": {"const": "local_discovery_preflight_only"},
    },
    required=(
        "currency",
        "oel_execution_amount",
        "configuration_validation_amount",
        "model_billing",
        "hosted_preflight_required",
        "payment_authorized",
        "policy_status",
    ),
)

STUDY_PLAN_REVIEW_SCHEMA = _object(
    {
        "schema": {"const": "oel.study_plan_review.v1"},
        "review_id": _identifier(),
        "planning_result_sha256": _sha256(),
        "request_id": _identifier(),
        "request_sha256": _sha256(),
        "plan_id": _identifier(),
        "plan_sha256": _sha256(),
        "capability_manifest_sha256": _sha256(),
        "planner_version": _text(maximum=120),
        "question_answered": _text(maximum=20_000),
        "operations": {"type": "array", "items": OPERATION_SCHEMA, "minItems": 1, "maxItems": 128},
        "transfer_manifest": {"type": "array", "items": TRANSFER_SCHEMA, "maxItems": 128},
        "evidence_requirements": {
            "type": "array", "items": EVIDENCE_REQUIREMENT_SCHEMA, "minItems": 1, "maxItems": 256,
        },
        "acceptance_criteria": {"type": "array", "items": ACCEPTANCE_SCHEMA, "minItems": 1, "maxItems": 256},
        "claims": {"type": "array", "items": CLAIM_SCHEMA, "maxItems": 256},
        "assumptions": _string_array(maximum=256),
        "non_claims": _string_array(maximum=256),
        "resource_envelope": RESOURCE_ENVELOPE_SCHEMA,
        "resource_accounting": RESOURCE_ACCOUNTING_SCHEMA,
        "execution_route": EXECUTION_ROUTE_SCHEMA,
        "commercial_status": COMMERCIAL_STATUS_SCHEMA,
        "user_review_required": {"const": True},
        "execution_authorized": {"const": False},
        "agent_may_authorize": {"const": False},
    },
    required=(
        "schema",
        "review_id",
        "planning_result_sha256",
        "request_id",
        "request_sha256",
        "plan_id",
        "plan_sha256",
        "capability_manifest_sha256",
        "planner_version",
        "question_answered",
        "operations",
        "transfer_manifest",
        "evidence_requirements",
        "acceptance_criteria",
        "claims",
        "assumptions",
        "non_claims",
        "resource_envelope",
        "resource_accounting",
        "execution_route",
        "commercial_status",
        "user_review_required",
        "execution_authorized",
        "agent_may_authorize",
    ),
)

STUDY_COMPLAINT_SCHEMA = _object(
    {
        "schema": {"const": "oel.study_complaint.v1"},
        "study_complaint_id": _identifier(),
        "request_id": _identifier(),
        "plan_id": _identifier(),
        "plan_sha256": _sha256(),
        "receipt_id": _identifier(),
        "alleged_failure": _text(maximum=4000),
        "violated_criterion_ids": {"type": "array", "items": _identifier(), "maxItems": 64},
        "selected_evidence_refs": {"type": "array", "items": _identifier(), "maxItems": 128},
        "product_feedback_permission": {"type": "boolean"},
        "submission_authorized": {"type": "boolean"},
    },
    required=("schema", "study_complaint_id", "request_id", "plan_id", "plan_sha256", "receipt_id", "alleged_failure", "violated_criterion_ids", "selected_evidence_refs", "product_feedback_permission", "submission_authorized"),
)

STUDY_COMPLAINT_RESOLUTION_SCHEMA = _object(
    {
        "schema": {"const": "oel.study_complaint_resolution.v1"},
        "resolution_id": _identifier(),
        "study_complaint_id": _identifier(),
        "disposition": {"type": "string", "enum": ["refund", "free_corrected_run", "declined"]},
        "reason": _text(maximum=4000),
        "charged_oel_amount_usd": {"type": "number", "minimum": 0, "maximum": 1_000_000},
        "refund_amount_usd": {"type": "number", "minimum": 0, "maximum": 1_000_000},
        "corrected_run_id": {"type": ["string", "null"], "maxLength": 160},
    },
    required=("schema", "resolution_id", "study_complaint_id", "disposition", "reason", "charged_oel_amount_usd", "refund_amount_usd", "corrected_run_id"),
)

CAPABILITY_PORT_SCHEMA = _object(
    {
        "port_id": _identifier(),
        "semantic_role": _identifier(),
        "artifact_kind": _text(maximum=120),
        "schema_ids": {
            "type": "array",
            "items": _identifier(),
            "minItems": 1,
            "maxItems": 16,
            "uniqueItems": True,
        },
        "cardinality": {
            "type": "string",
            "enum": ["exactly_one", "optional", "many"],
        },
        "required_metadata": _string_array(maximum=32),
    },
    required=(
        "port_id",
        "semantic_role",
        "artifact_kind",
        "schema_ids",
        "cardinality",
        "required_metadata",
    ),
)


STUDY_CAPABILITY_SCHEMA = _object(
    {
        "schema": {"const": "oel.hosted_study_capability.v2"},
        "capability_id": _identifier(),
        "title": _text(maximum=200),
        "description": _text(maximum=2000),
        "edition": {"type": "string", "enum": ["public", "pro"]},
        "maturity": {"type": "string", "enum": ["supported", "experimental", "prototype"]},
        "availability": {"type": "string", "enum": ["available", "unavailable"]},
        "executor_binding": _object(
            {
                "status": {"type": "string", "enum": ["bound", "unbound"]},
                "executor_contract_ids": {
                    "type": "array",
                    "items": _identifier(),
                    "maxItems": 16,
                    "uniqueItems": True,
                },
                "adapter_id": {"type": ["string", "null"], "maxLength": 160},
            },
            required=("status", "executor_contract_ids", "adapter_id"),
        ),
        "access": _object(
            {
                "discoverable_in_public": {"const": True},
                "available_in_public_install": {"type": "boolean"},
                "access_mode": {"type": "string", "enum": ["public_local", "hosted_pro_only"]},
                "hosted_pro_required": {"type": "boolean"},
            },
            required=(
                "discoverable_in_public",
                "available_in_public_install",
                "access_mode",
                "hosted_pro_required",
            ),
        ),
        "effects": _object(
            {
                "reads": {"type": "boolean"},
                "writes": {"type": "boolean"},
                "executes": {"type": "boolean"},
                "external_communication": {"const": False},
            },
            required=("reads", "writes", "executes", "external_communication"),
        ),
        "input_kinds": _string_array(maximum=64),
        "output_kinds": _string_array(maximum=64),
        "input_ports": {"type": "array", "items": CAPABILITY_PORT_SCHEMA, "maxItems": 64},
        "output_ports": {"type": "array", "items": CAPABILITY_PORT_SCHEMA, "maxItems": 64},
        "parameter_schema": {"type": "object"},
        "limitations": _string_array(maximum=64),
    },
    required=("schema", "capability_id", "title", "description", "edition", "maturity", "availability", "executor_binding", "access", "effects", "input_kinds", "output_kinds", "input_ports", "output_ports", "parameter_schema", "limitations"),
)

STUDY_CAPABILITY_MANIFEST_SCHEMA = _object(
    {
        "schema": {"const": "oel.hosted_study_capability_manifest.v2"},
        "planner_version": _text(maximum=120),
        "catalog_scope": {
            "type": "string", "enum": ["public_only", "public_agent_discovery"],
        },
        "manifest_sha256": _sha256(),
        "capabilities": {"type": "array", "items": STUDY_CAPABILITY_SCHEMA, "maxItems": 128},
        "access": _object(
            {
                "descriptor_visibility": {"const": "public"},
                "public_capabilities_available_locally": {"const": True},
                "pro_capabilities_available_locally": {"const": False},
                "pro_execution_tools_exposed": {"const": False},
                "hosted_preflight_required_for_pro": {"const": True},
            },
            required=(
                "descriptor_visibility",
                "public_capabilities_available_locally",
                "pro_capabilities_available_locally",
                "pro_execution_tools_exposed",
                "hosted_preflight_required_for_pro",
            ),
        ),
        "effects": _object(
            {
                "writes": {"const": False},
                "executes": {"const": False},
                "external_communication": {"const": False},
            },
            required=("writes", "executes", "external_communication"),
        ),
    },
    required=(
        "schema",
        "planner_version",
        "catalog_scope",
        "manifest_sha256",
        "capabilities",
        "access",
        "effects",
    ),
)

PLANNING_FINDING_SCHEMA = _object(
    {
        "code": _identifier(),
        "severity": {"type": "string", "enum": ["info", "warning", "error", "blocker"]},
        "category": _text(maximum=80),
        "message": _text(maximum=2000),
        "paths": {"type": "array", "items": {"type": "string", "maxLength": 500}, "maxItems": 32},
        "observed": {"type": "object"},
        "suggestion": {"type": "string", "maxLength": 2000},
    },
    required=("code", "severity", "category", "message", "paths", "observed", "suggestion"),
)

STUDY_CONFIG_VALIDATION_RECEIPT_SCHEMA = _object(
    {
        "schema": {"const": "oel.study_config_validation_receipt.v1"},
        "config_ref": _identifier(),
        "source_sha256": _sha256(),
        "normalized_config_sha256": {"type": ["string", "null"], "pattern": SHA256_PATTERN},
        "valid": {"type": "boolean"},
        "safe_validation_only": {"const": True},
        "plugins_imported": {"const": False},
        "execution_advanced": {"const": False},
        "charge_created": {"const": False},
        "errors": _string_array(maximum=256),
        "resource_estimate": {"type": "object"},
    },
    required=(
        "schema",
        "config_ref",
        "source_sha256",
        "normalized_config_sha256",
        "valid",
        "safe_validation_only",
        "plugins_imported",
        "execution_advanced",
        "charge_created",
        "errors",
        "resource_estimate",
    ),
)

NULLABLE_STUDY_PLAN_SCHEMA = deepcopy(STUDY_PLAN_SCHEMA)
NULLABLE_STUDY_PLAN_SCHEMA["type"] = ["object", "null"]

PLANNING_RESULT_SCHEMA = _object(
    {
        "schema": {"const": "oel.study_planning_result.v1"},
        "status": {"type": "string", "enum": ["PLAN_VALID", "CLARIFICATION_REQUIRED", "UNSUPPORTED"]},
        "request": STUDY_REQUEST_SCHEMA,
        "capability_manifest_sha256": _sha256(),
        "planner_version": _text(maximum=120),
        "plan": NULLABLE_STUDY_PLAN_SCHEMA,
        "plan_sha256": {"type": ["string", "null"], "pattern": SHA256_PATTERN},
        "findings": {"type": "array", "items": PLANNING_FINDING_SCHEMA, "maxItems": 2048},
        "resource_accounting": RESOURCE_ACCOUNTING_SCHEMA,
        "config_receipts": {
            "type": "array",
            "items": STUDY_CONFIG_VALIDATION_RECEIPT_SCHEMA,
            "maxItems": 128,
        },
        "execution_route": EXECUTION_ROUTE_SCHEMA,
        "commercial_status": COMMERCIAL_STATUS_SCHEMA,
        "review_status": _object(
            {
                "review_available": {"type": "boolean"},
                "user_review_required": {"type": "boolean"},
                "execution_authorized": {"const": False},
                "agent_may_authorize": {"const": False},
            },
            required=(
                "review_available",
                "user_review_required",
                "execution_authorized",
                "agent_may_authorize",
            ),
        ),
        "non_claims": _string_array(maximum=64),
    },
    required=(
        "schema",
        "status",
        "request",
        "capability_manifest_sha256",
        "planner_version",
        "plan",
        "plan_sha256",
        "findings",
        "resource_accounting",
        "config_receipts",
        "execution_route",
        "commercial_status",
        "review_status",
        "non_claims",
    ),
)

SCHEMAS: dict[str, dict[str, Any]] = {
    "oel.hosted_study_request.v1": STUDY_REQUEST_SCHEMA,
    "oel.hosted_study_plan.v2": STUDY_PLAN_SCHEMA,
    "oel.hosted_study_run.v1": STUDY_RUN_SCHEMA,
    "oel.hosted_study_evidence.v1": STUDY_EVIDENCE_SCHEMA,
    "oel.hosted_study_claims.v1": STUDY_CLAIMS_SCHEMA,
    "oel.hosted_study_receipt.v1": STUDY_RECEIPT_SCHEMA,
    "oel.capability_request.v1": CAPABILITY_REQUEST_SCHEMA,
    "oel.study_plan_review.v1": STUDY_PLAN_REVIEW_SCHEMA,
    "oel.study_complaint.v1": STUDY_COMPLAINT_SCHEMA,
    "oel.study_complaint_resolution.v1": STUDY_COMPLAINT_RESOLUTION_SCHEMA,
    "oel.hosted_study_capability.v2": STUDY_CAPABILITY_SCHEMA,
    "oel.hosted_study_capability_manifest.v2": STUDY_CAPABILITY_MANIFEST_SCHEMA,
    "oel.study_config_validation_receipt.v1": STUDY_CONFIG_VALIDATION_RECEIPT_SCHEMA,
    "oel.study_planning_result.v1": PLANNING_RESULT_SCHEMA,
}


def schema_for(schema_id: str) -> dict[str, Any]:
    try:
        return deepcopy(SCHEMAS[str(schema_id)])
    except KeyError as exc:
        raise ValueError(f"Unsupported study schema: {schema_id!r}") from exc


__all__ = [
    "CAPABILITY_REQUEST_SCHEMA",
    "PLANNING_RESULT_SCHEMA",
    "RESOURCE_ACCOUNTING_SCHEMA",
    "RESOURCE_ACCOUNTING_METRIC_SCHEMA",
    "SCHEMAS",
    "STUDY_CAPABILITY_SCHEMA",
    "STUDY_CAPABILITY_MANIFEST_SCHEMA",
    "STUDY_CLAIMS_SCHEMA",
    "STUDY_COMPLAINT_RESOLUTION_SCHEMA",
    "STUDY_COMPLAINT_SCHEMA",
    "STUDY_EVIDENCE_SCHEMA",
    "STUDY_PLAN_SCHEMA",
    "STUDY_PLAN_REVIEW_SCHEMA",
    "STUDY_RECEIPT_SCHEMA",
    "STUDY_REQUEST_SCHEMA",
    "STUDY_RUN_SCHEMA",
    "schema_for",
]
