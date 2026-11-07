"""Content-bound contracts for open-ended OEL study planning."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from .schema_validation import validate_schema
from .schemas import SCHEMAS, schema_for

STUDY_REQUEST_SCHEMA_ID = "oel.hosted_study_request.v1"
STUDY_PLAN_SCHEMA_ID = "oel.hosted_study_plan.v2"
STUDY_RUN_SCHEMA_ID = "oel.hosted_study_run.v1"
STUDY_EVIDENCE_SCHEMA_ID = "oel.hosted_study_evidence.v1"
STUDY_CLAIMS_SCHEMA_ID = "oel.hosted_study_claims.v1"
STUDY_RECEIPT_SCHEMA_ID = "oel.hosted_study_receipt.v1"
CAPABILITY_REQUEST_SCHEMA_ID = "oel.capability_request.v1"
STUDY_COMPLAINT_SCHEMA_ID = "oel.study_complaint.v1"
STUDY_COMPLAINT_RESOLUTION_SCHEMA_ID = "oel.study_complaint_resolution.v1"
STUDY_CAPABILITY_SCHEMA_ID = "oel.hosted_study_capability.v2"
STUDY_CAPABILITY_MANIFEST_SCHEMA_ID = "oel.hosted_study_capability_manifest.v2"
STUDY_PLAN_REVIEW_SCHEMA_ID = "oel.study_plan_review.v1"
PLANNING_RESULT_SCHEMA_ID = "oel.study_planning_result.v1"


class StudyContractError(ValueError):
    """Raised when a study document violates its closed contract."""


@dataclass(frozen=True, slots=True)
class ContractIssue:
    code: str
    path: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code, "path": self.path, "message": self.message}


@dataclass(frozen=True, slots=True)
class ContractReport:
    schema: str
    valid: bool
    document_sha256: str
    issues: tuple[ContractIssue, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "valid": self.valid,
            "document_sha256": self.document_sha256,
            "issues": [issue.to_dict() for issue in self.issues],
        }


def canonical_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): canonical_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [canonical_value(item) for item in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise StudyContractError("Study contract values must be finite.")
        return value
    raise StudyContractError(f"Unsupported study contract value: {type(value).__name__}")


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        canonical_value(value),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def validate_document(document: Mapping[str, Any], *, expected_schema: str | None = None) -> ContractReport:
    payload = deepcopy(dict(document))
    schema_id = str(payload.get("schema", "") or "")
    issues: list[ContractIssue] = []
    if expected_schema is not None and schema_id != expected_schema:
        issues.append(
            ContractIssue(
                code="schema.unexpected",
                path="$.schema",
                message=f"Expected {expected_schema!r}, received {schema_id!r}.",
            )
        )
    schema = SCHEMAS.get(schema_id)
    if schema is None:
        issues.append(
            ContractIssue(
                code="schema.unsupported",
                path="$.schema",
                message=f"Unsupported study schema {schema_id!r}.",
            )
        )
    else:
        issues.extend(
            ContractIssue(code="schema.invalid", path=item.path, message=item.message)
            for item in validate_schema(payload, schema)
        )
    issues.extend(_semantic_issues(payload, schema_id=schema_id))
    return ContractReport(
        schema=schema_id,
        valid=not issues,
        document_sha256=canonical_sha256(payload),
        issues=tuple(issues),
    )


def require_valid_document(document: Mapping[str, Any], *, expected_schema: str | None = None) -> dict[str, Any]:
    payload = deepcopy(dict(document))
    report = validate_document(payload, expected_schema=expected_schema)
    if not report.valid:
        detail = "; ".join(f"{issue.path}: {issue.message}" for issue in report.issues[:8])
        raise StudyContractError(f"Study contract validation failed: {detail}")
    return payload


def bind_study_request(document: Mapping[str, Any]) -> dict[str, Any]:
    payload = deepcopy(dict(document))
    payload.setdefault("schema", STUDY_REQUEST_SCHEMA_ID)
    payload.setdefault("inputs", [])
    payload.setdefault("required_choices", [])
    payload.setdefault("clarifications", [])
    payload.setdefault("handling", {"marking": "UNSPECIFIED", "release_scope": "local_only", "owner": ""})
    payload["request_id"] = str(payload.get("request_id", "") or "pending")
    payload.pop("request_sha256", None)
    digest = canonical_sha256(payload)
    if payload["request_id"] == "pending":
        payload["request_id"] = f"request:{digest[:24]}"
    payload["request_sha256"] = canonical_sha256(_without_identity(payload, "request_sha256"))
    return require_valid_document(payload, expected_schema=STUDY_REQUEST_SCHEMA_ID)


def bind_study_plan(
    document: Mapping[str, Any],
    *,
    request_id: str,
    request_sha256: str,
    capability_manifest_sha256: str,
    planner_version: str,
) -> dict[str, Any]:
    payload = deepcopy(dict(document))
    payload.setdefault("schema", STUDY_PLAN_SCHEMA_ID)
    payload["request_id"] = request_id
    payload["request_sha256"] = request_sha256
    payload["capability_manifest_sha256"] = capability_manifest_sha256
    payload["planner_version"] = planner_version
    payload.setdefault("operations", [])
    payload.setdefault("config_constraints", [])
    payload.setdefault("assumptions", [])
    payload.setdefault("allowed_defaults", [])
    payload.setdefault("forbidden_capabilities", [])
    payload.setdefault("evidence_requirements", [])
    payload.setdefault("acceptance_criteria", [])
    payload.setdefault("claims", [])
    payload.setdefault("non_claims", [])
    payload.setdefault("transfers", [])
    payload.setdefault(
        "resource_envelope",
        {
            "resource_profile": "laptop-safe",
            "max_operations": 32,
            "max_total_cases": 10_000,
            "max_wall_time_s": 3600.0,
            "max_cpu_time_s": 3600.0,
            "max_peak_memory_mb": 2048.0,
            "max_storage_mb": 512.0,
            "max_artifact_mb": 256.0,
            "max_parallel_workers": 1,
        },
    )
    envelope = payload["resource_envelope"]
    if isinstance(envelope, Mapping):
        envelope = dict(envelope)
        wall_time = envelope.get("max_wall_time_s")
        if (
            "max_wall_time_s" not in envelope
            or isinstance(wall_time, bool)
            or not isinstance(wall_time, (int, float))
            or not math.isfinite(float(wall_time))
        ):
            raise StudyContractError(
                "Study contract validation failed: "
                "$.resource_envelope.max_wall_time_s must be a finite number."
            )
        # CPU authorization is an independent ceiling.  Keep the historical
        # default for plans that omit it rather than deriving one from wall time.
        envelope.setdefault("max_cpu_time_s", 3600.0)
        envelope.setdefault("max_peak_memory_mb", 2048.0)
        envelope.setdefault("max_storage_mb", 512.0)
        envelope.setdefault("max_artifact_mb", 256.0)
        envelope.setdefault("max_parallel_workers", 1)
        payload["resource_envelope"] = envelope
    payload["plan_id"] = str(payload.get("plan_id", "") or "pending")
    payload.pop("plan_sha256", None)
    provisional = canonical_sha256(_without_identity(payload, "plan_id", "plan_sha256"))
    if payload["plan_id"] == "pending":
        payload["plan_id"] = f"plan:{provisional[:24]}"
    payload["plan_sha256"] = canonical_sha256(_without_identity(payload, "plan_sha256"))
    return require_valid_document(payload, expected_schema=STUDY_PLAN_SCHEMA_ID)


def verify_bound_plan(plan: Mapping[str, Any]) -> ContractReport:
    report = validate_document(plan, expected_schema=STUDY_PLAN_SCHEMA_ID)
    issues = list(report.issues)
    if report.valid:
        expected = canonical_sha256(_without_identity(plan, "plan_sha256"))
        if str(plan.get("plan_sha256", "")) != expected:
            issues.append(
                ContractIssue(
                    code="identity.plan_digest_mismatch",
                    path="$.plan_sha256",
                    message="The declared plan digest does not match the canonical plan content.",
                )
            )
    return ContractReport(
        schema=report.schema,
        valid=not issues,
        document_sha256=report.document_sha256,
        issues=tuple(issues),
    )


def classify_execution_route(
    plan: Mapping[str, Any] | None,
    *,
    catalog: Any,
    eligible: bool,
    allow_unbound_pro_planning: bool = False,
) -> dict[str, Any]:
    """Classify a plan from declared capability editions without model judgment."""

    public_ids: list[str] = []
    pro_ids: list[str] = []
    unavailable_ids: list[str] = []
    for operation in list((plan or {}).get("operations", []) or []):
        capability_id = str(operation.get("capability_id", "") or "")
        if (
            not capability_id
            or capability_id in public_ids
            or capability_id in pro_ids
            or capability_id in unavailable_ids
        ):
            continue
        capability = catalog.get(capability_id)
        if capability is None:
            unavailable_ids.append(capability_id)
        elif capability.edition == "pro" and allow_unbound_pro_planning:
            pro_ids.append(capability_id)
        elif capability.availability != "available":
            unavailable_ids.append(capability_id)
        elif capability.edition == "public":
            public_ids.append(capability_id)
        elif capability.edition == "pro":
            pro_ids.append(capability_id)
        else:
            unavailable_ids.append(capability_id)

    if eligible and not unavailable_ids and pro_ids:
        route = "HOSTED_PRO_REQUIRED"
    elif eligible and public_ids and not pro_ids and not unavailable_ids:
        route = "LOCAL_FREE_AVAILABLE"
    else:
        route = "NOT_ELIGIBLE"

    return {
        "route": route,
        "public_capability_ids": public_ids,
        "pro_capability_ids": pro_ids,
        "unavailable_capability_ids": unavailable_ids,
        "local_free_available": route == "LOCAL_FREE_AVAILABLE",
        "hosted_pro_required": route == "HOSTED_PRO_REQUIRED",
        "hosted_preflight_required": route == "HOSTED_PRO_REQUIRED",
    }


def build_plan_review(planning_result: Mapping[str, Any]) -> dict[str, Any]:
    """Render a review only from a successful, content-bound preflight result."""

    result = require_valid_document(
        planning_result,
        expected_schema=PLANNING_RESULT_SCHEMA_ID,
    )
    if result["status"] != "PLAN_VALID" or result["plan"] is None:
        raise StudyContractError("A plan review requires a PLAN_VALID preflight result.")
    if result["execution_route"]["route"] == "NOT_ELIGIBLE":
        raise StudyContractError("A plan review cannot be built for an ineligible execution route.")
    plan = require_valid_document(result["plan"], expected_schema=STUDY_PLAN_SCHEMA_ID)
    verification = verify_bound_plan(plan)
    if not verification.valid:
        raise StudyContractError("A plan review requires a plan with a valid content digest.")
    if plan["request_id"] != result["request"]["request_id"]:
        raise StudyContractError("The plan does not identify the preflight request.")
    if plan["request_sha256"] != result["request"]["request_sha256"]:
        raise StudyContractError("The plan does not bind the exact preflight request content.")
    if plan["capability_manifest_sha256"] != result["capability_manifest_sha256"]:
        raise StudyContractError("The plan does not bind the preflight capability manifest.")
    if plan["planner_version"] != result["planner_version"]:
        raise StudyContractError("The plan does not bind the preflight planner version.")

    result_sha256 = canonical_sha256(result)
    payload = {
        "schema": STUDY_PLAN_REVIEW_SCHEMA_ID,
        "review_id": f"review:{result_sha256[:24]}",
        "planning_result_sha256": result_sha256,
        "request_id": plan["request_id"],
        "request_sha256": plan["request_sha256"],
        "plan_id": plan["plan_id"],
        "plan_sha256": plan["plan_sha256"],
        "capability_manifest_sha256": plan["capability_manifest_sha256"],
        "planner_version": plan["planner_version"],
        "question_answered": plan["question_answered"],
        "operations": deepcopy(plan["operations"]),
        "transfer_manifest": deepcopy(plan["transfers"]),
        "evidence_requirements": deepcopy(plan["evidence_requirements"]),
        "acceptance_criteria": deepcopy(plan["acceptance_criteria"]),
        "claims": deepcopy(plan["claims"]),
        "assumptions": deepcopy(plan["assumptions"]),
        "non_claims": deepcopy(plan["non_claims"]),
        "resource_envelope": deepcopy(plan["resource_envelope"]),
        "resource_accounting": deepcopy(result["resource_accounting"]),
        "execution_route": deepcopy(result["execution_route"]),
        "commercial_status": deepcopy(result["commercial_status"]),
        "user_review_required": True,
        "execution_authorized": False,
        "agent_may_authorize": False,
    }
    return require_valid_document(payload, expected_schema=STUDY_PLAN_REVIEW_SCHEMA_ID)


def contract_schema(schema_id: str) -> dict[str, Any]:
    return schema_for(schema_id)


def _without_identity(document: Mapping[str, Any], *keys: str) -> dict[str, Any]:
    payload = deepcopy(dict(document))
    for key in keys:
        payload.pop(key, None)
    return payload


def _semantic_issues(document: Mapping[str, Any], *, schema_id: str) -> list[ContractIssue]:
    if schema_id == STUDY_REQUEST_SCHEMA_ID:
        return _request_semantic_issues(document)
    if schema_id == STUDY_PLAN_SCHEMA_ID:
        return _plan_semantic_issues(document)
    if schema_id == STUDY_CAPABILITY_MANIFEST_SCHEMA_ID:
        return _capability_manifest_semantic_issues(document)
    if schema_id == PLANNING_RESULT_SCHEMA_ID:
        return _planning_result_semantic_issues(document)
    if schema_id == STUDY_COMPLAINT_RESOLUTION_SCHEMA_ID:
        return _complaint_resolution_semantic_issues(document)
    return []


def _request_semantic_issues(document: Mapping[str, Any]) -> list[ContractIssue]:
    issues: list[ContractIssue] = []
    identifiers = [str(item.get("input_id", "")) for item in document.get("inputs", []) if isinstance(item, Mapping)]
    if len(set(identifiers)) != len(identifiers):
        issues.append(ContractIssue("request.duplicate_input", "$.inputs", "Input identifiers must be unique."))
    for index, item in enumerate(document.get("inputs", [])):
        if isinstance(item, Mapping) and item.get("required") and item.get("content_sha256") is None:
            issues.append(
                ContractIssue(
                    "request.required_input_digest_missing",
                    f"$.inputs[{index}].content_sha256",
                    "Every required request input must have a content SHA-256 identity.",
                )
            )
    choice_ids = [
        str(item.get("choice_id", ""))
        for item in document.get("required_choices", [])
        if isinstance(item, Mapping)
    ]
    if len(set(choice_ids)) != len(choice_ids):
        issues.append(
            ContractIssue("request.duplicate_choice", "$.required_choices", "Choice identifiers must be unique.")
        )
    return issues


def _capability_manifest_semantic_issues(document: Mapping[str, Any]) -> list[ContractIssue]:
    issues: list[ContractIssue] = []
    expected = canonical_sha256(_without_identity(document, "manifest_sha256"))
    if document.get("manifest_sha256") != expected:
        issues.append(
            ContractIssue(
                "identity.capability_manifest_digest_mismatch",
                "$.manifest_sha256",
                "The declared manifest digest does not match the complete manifest content.",
            )
        )
    for index, capability in enumerate(document.get("capabilities", [])):
        if not isinstance(capability, Mapping):
            continue
        binding = capability.get("executor_binding", {})
        bound = isinstance(binding, Mapping) and binding.get("status") == "bound"
        expected_availability = "available" if bound else "unavailable"
        if capability.get("availability") != expected_availability:
            issues.append(
                ContractIssue(
                    "capability.binding_availability_mismatch",
                    f"$.capabilities[{index}].availability",
                    "Capability availability must be derived from its executor binding status.",
                )
            )
        executor_ids = list(binding.get("executor_contract_ids", []) or []) if isinstance(binding, Mapping) else []
        adapter_id = binding.get("adapter_id") if isinstance(binding, Mapping) else None
        if bound and (not executor_ids or not adapter_id):
            issues.append(
                ContractIssue(
                    "capability.bound_executor_missing",
                    f"$.capabilities[{index}].executor_binding",
                    "A bound capability requires concrete executor contracts and an adapter identity.",
                )
            )
        if not bound and (executor_ids or adapter_id is not None):
            issues.append(
                ContractIssue(
                    "capability.unbound_executor_declared",
                    f"$.capabilities[{index}].executor_binding",
                    "An unbound capability cannot declare active executor contracts or an adapter identity.",
                )
            )
    return issues


def _plan_semantic_issues(document: Mapping[str, Any]) -> list[ContractIssue]:
    issues: list[ContractIssue] = []
    operations = [item for item in document.get("operations", []) if isinstance(item, Mapping)]
    operation_ids = [str(item.get("operation_id", "")) for item in operations]
    if len(set(operation_ids)) != len(operation_ids):
        issues.append(ContractIssue("plan.duplicate_operation", "$.operations", "Operation identifiers must be unique."))
    known = set(operation_ids)
    for index, operation in enumerate(operations):
        for field in ("input_refs", "output_refs"):
            refs = [
                str(item.get("ref_id", ""))
                for item in operation.get(field, [])
                if isinstance(item, Mapping)
            ]
            if len(set(refs)) != len(refs):
                issues.append(
                    ContractIssue(
                        "plan.duplicate_artifact_reference",
                        f"$.operations[{index}].{field}",
                        "Artifact reference identifiers must be unique within an operation.",
                    )
                )
    dependencies = {str(item.get("operation_id", "")): tuple(map(str, item.get("depends_on", []) or ())) for item in operations}
    for operation_id, required in dependencies.items():
        missing = sorted(set(required) - known)
        if missing:
            issues.append(
                ContractIssue(
                    "plan.unknown_dependency",
                    f"$.operations[{operation_id}].depends_on",
                    f"Unknown operation dependencies: {', '.join(missing)}.",
                )
            )
    if _has_cycle(dependencies):
        issues.append(ContractIssue("plan.dependency_cycle", "$.operations", "Operation dependencies must be acyclic."))

    transfer_ids = [
        str(item.get("input_id", ""))
        for item in document.get("transfers", [])
        if isinstance(item, Mapping)
    ]
    if len(set(transfer_ids)) != len(transfer_ids):
        issues.append(
            ContractIssue(
                "plan.duplicate_transfer",
                "$.transfers",
                "Transfer input identifiers must be unique.",
            )
        )

    evidence = [item for item in document.get("evidence_requirements", []) if isinstance(item, Mapping)]
    evidence_ids = [str(item.get("evidence_id", "")) for item in evidence]
    if len(set(evidence_ids)) != len(evidence_ids):
        issues.append(
            ContractIssue("plan.duplicate_evidence", "$.evidence_requirements", "Evidence identifiers must be unique.")
        )
    known_evidence = set(evidence_ids)
    for index, item in enumerate(evidence):
        missing = sorted(set(map(str, item.get("source_operation_ids", []) or ())) - known)
        if missing:
            issues.append(
                ContractIssue(
                    "plan.evidence_unknown_operation",
                    f"$.evidence_requirements[{index}].source_operation_ids",
                    f"Evidence references unknown operations: {', '.join(missing)}.",
                )
            )
    for collection in ("acceptance_criteria", "claims"):
        for index, item in enumerate(document.get(collection, []) or []):
            if not isinstance(item, Mapping):
                continue
            missing = sorted(set(map(str, item.get("evidence_ids", []) or ())) - known_evidence)
            if missing:
                issues.append(
                    ContractIssue(
                        "plan.unknown_evidence_reference",
                        f"$.{collection}[{index}].evidence_ids",
                        f"Unknown evidence references: {', '.join(missing)}.",
                    )
                )
    return issues


def _planning_result_semantic_issues(document: Mapping[str, Any]) -> list[ContractIssue]:
    issues: list[ContractIssue] = []
    status = document.get("status")
    plan = document.get("plan")
    route = dict(document.get("execution_route", {}) or {})
    review = dict(document.get("review_status", {}) or {})
    blockers = [
        item
        for item in document.get("findings", [])
        if isinstance(item, Mapping) and item.get("severity") in {"error", "blocker"}
    ]
    valid = status == "PLAN_VALID"
    if valid and (not isinstance(plan, Mapping) or blockers or route.get("route") == "NOT_ELIGIBLE"):
        issues.append(
            ContractIssue(
                "planning_result.invalid_success_state",
                "$.status",
                "PLAN_VALID requires a bound plan, no error findings, and an eligible execution route.",
            )
        )
    if bool(review.get("review_available")) != valid or bool(review.get("user_review_required")) != valid:
        issues.append(
            ContractIssue(
                "planning_result.review_state_mismatch",
                "$.review_status",
                "A plan review is available and required only for PLAN_VALID.",
            )
        )
    if isinstance(plan, Mapping):
        bindings = (
            ("request_id", dict(document.get("request", {}) or {}).get("request_id")),
            ("request_sha256", dict(document.get("request", {}) or {}).get("request_sha256")),
            ("capability_manifest_sha256", document.get("capability_manifest_sha256")),
            ("planner_version", document.get("planner_version")),
            ("plan_sha256", document.get("plan_sha256")),
        )
        for field, expected in bindings:
            if plan.get(field) != expected:
                issues.append(
                    ContractIssue(
                        "planning_result.plan_binding_mismatch",
                        f"$.plan.{field}",
                        f"The plan {field} does not match the planning result.",
                    )
                )
    elif document.get("plan_sha256") is not None:
        issues.append(
            ContractIssue(
                "planning_result.unexpected_plan_digest",
                "$.plan_sha256",
                "A planning result without a plan cannot declare a plan digest.",
            )
        )
    return issues


def _complaint_resolution_semantic_issues(document: Mapping[str, Any]) -> list[ContractIssue]:
    disposition = str(document.get("disposition", ""))
    charged_amount = float(document.get("charged_oel_amount_usd", 0.0) or 0.0)
    amount = float(document.get("refund_amount_usd", 0.0) or 0.0)
    corrected_run = document.get("corrected_run_id")
    issues: list[ContractIssue] = []
    if disposition == "refund" and (charged_amount <= 0.0 or amount != charged_amount):
        issues.append(
            ContractIssue(
                "complaint.full_refund_required",
                "$.refund_amount_usd",
                "A confirmed refund resolution must refund the full charged OEL amount recorded on the receipt.",
            )
        )
    if disposition != "refund" and amount != 0.0:
        issues.append(
            ContractIssue(
                "complaint.unexpected_refund_amount",
                "$.refund_amount_usd",
                "Non-refund resolutions must not declare a refund amount.",
            )
        )
    if disposition == "free_corrected_run" and not corrected_run:
        issues.append(
            ContractIssue(
                "complaint.corrected_run_required",
                "$.corrected_run_id",
                "A free corrected-run resolution must identify the new run.",
            )
        )
    return issues


def _has_cycle(graph: Mapping[str, tuple[str, ...]]) -> bool:
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str) -> bool:
        if node in visiting:
            return True
        if node in visited:
            return False
        visiting.add(node)
        for dependency in graph.get(node, ()):
            if dependency in graph and visit(dependency):
                return True
        visiting.remove(node)
        visited.add(node)
        return False

    return any(visit(node) for node in graph)


__all__ = [
    "CAPABILITY_REQUEST_SCHEMA_ID",
    "ContractIssue",
    "ContractReport",
    "PLANNING_RESULT_SCHEMA_ID",
    "STUDY_CAPABILITY_SCHEMA_ID",
    "STUDY_CAPABILITY_MANIFEST_SCHEMA_ID",
    "STUDY_CLAIMS_SCHEMA_ID",
    "STUDY_COMPLAINT_RESOLUTION_SCHEMA_ID",
    "STUDY_COMPLAINT_SCHEMA_ID",
    "STUDY_EVIDENCE_SCHEMA_ID",
    "STUDY_PLAN_SCHEMA_ID",
    "STUDY_PLAN_REVIEW_SCHEMA_ID",
    "STUDY_RECEIPT_SCHEMA_ID",
    "STUDY_REQUEST_SCHEMA_ID",
    "STUDY_RUN_SCHEMA_ID",
    "StudyContractError",
    "bind_study_plan",
    "bind_study_request",
    "build_plan_review",
    "classify_execution_route",
    "canonical_bytes",
    "canonical_sha256",
    "contract_schema",
    "require_valid_document",
    "validate_document",
    "verify_bound_plan",
]
