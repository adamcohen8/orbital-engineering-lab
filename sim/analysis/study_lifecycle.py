"""Transport-neutral, content-bound study records over completed OEL evidence."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from sim.utils.io import SafeReadError, read_regular_file_nofollow

STUDY_REQUEST_SCHEMA = "oel.study_request.v1"
STUDY_PLAN_SCHEMA = "oel.study_plan.v1"
STUDY_RUN_SCHEMA = "oel.study_run.v1"
STUDY_EVIDENCE_SCHEMA = "oel.study_evidence.v1"
STUDY_CLAIMS_SCHEMA = "oel.study_claims.v1"
STUDY_RECEIPT_SCHEMA = "oel.study_receipt.v1"
STUDY_VERIFICATION_SCHEMA = "oel.study_verification.v1"
STUDY_COMPARISON_SCHEMA = "oel.study_comparison.v1"

MAX_STUDY_STEPS = 12
MAX_STUDY_EVIDENCE_BYTES = 16 * 1024 * 1024
_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_.-]{2,127}$")
_STEP_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_VALIDATION_LEVELS = {f"VC-{index}" for index in range(5)}
_VALIDATION_RANK = {f"VC-{index}": index for index in range(5)}
_RESOURCE_PROFILES = {"off", "laptop-safe", "standard"}

CAPABILITY_CONTRACTS: dict[str, dict[str, Any]] = {
    "constellation_design": {
        "analysis_interface": "python -m sim.constellation_design solve",
        "evidence_schema": "oel.constellation_design_evidence.v1",
        "accepted_statuses": ("complete",),
        "max_validation_level": "VC-1",
    },
    "conjunction_assessment": {
        "analysis_interface": "python -m sim.conjunction assess",
        "evidence_schema": "oel.conjunction_assessment_evidence.v1",
        "accepted_statuses": ("completed",),
        "max_validation_level": "VC-1",
    },
    "mission_scheduling": {
        "analysis_interface": "python -m sim.mission_scheduling solve",
        "evidence_schema": "oel.mission_scheduling_evidence.v1",
        "accepted_statuses": ("complete",),
        "max_validation_level": "VC-1",
    },
    "orbit_lifetime": {
        "analysis_interface": "python -m sim.orbit_lifetime analyze",
        "evidence_schema": "oel.orbit_lifetime_evidence.v1",
        "accepted_statuses": ("completed",),
        "max_validation_level": "VC-1",
    },
    "spacecraft_power": {
        "analysis_interface": "python -m sim.spacecraft_power analyze",
        "evidence_schema": "oel.spacecraft_power_evidence.v1",
        "accepted_statuses": ("completed",),
        "max_validation_level": "VC-1",
    },
    "trajectory_targeting": {
        "analysis_interface": "python -m sim.trajectory_design solve",
        "evidence_schema": "oel.trajectory_targeting_evidence.v1",
        "accepted_statuses": ("converged",),
        "max_validation_level": "VC-1",
    },
}

_REQUEST_FILE = "study_request.json"
_PLAN_FILE = "study_plan.json"
_RUN_FILE = "study_run.json"
_EVIDENCE_FILE = "study_evidence.json"
_CLAIMS_FILE = "study_claims.json"
_RECEIPT_FILE = "study_receipt.json"
_ROOT_RECORD_FILES = (_REQUEST_FILE, _PLAN_FILE, _RUN_FILE, _EVIDENCE_FILE, _CLAIMS_FILE)
_ALL_ROOT_FILES = (*_ROOT_RECORD_FILES, _RECEIPT_FILE)


class StudyLifecycleError(ValueError):
    """Raised when a study record, bundle, citation, or receipt is invalid."""


def _reject_constant(value: str) -> None:
    raise StudyLifecycleError(f"Non-finite JSON constant is not allowed: {value}.")


def _clone(value: Any) -> Any:
    try:
        return json.loads(
            json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False),
            parse_constant=_reject_constant,
        )
    except (TypeError, ValueError) as exc:
        raise StudyLifecycleError(f"Study records must contain finite JSON-compatible values: {exc}") from exc


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise StudyLifecycleError(f"Study records must contain finite JSON-compatible values: {exc}") from exc


def _semantic_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _required_text(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise StudyLifecycleError(f"{field} must be a string.")
    normalized = value.strip()
    if not normalized:
        raise StudyLifecycleError(f"{field} must be a non-empty string.")
    return normalized


def _identifier(value: Any, field: str, *, step: bool = False) -> str:
    normalized = _required_text(value, field)
    pattern = _STEP_IDENTIFIER if step else _IDENTIFIER
    if pattern.fullmatch(normalized) is None:
        raise StudyLifecycleError(f"{field} has an invalid portable identifier: {normalized!r}.")
    return normalized


def _digest_or_auto(value: Any, field: str) -> str:
    normalized = _required_text(value, field).lower()
    if normalized != "auto" and _SHA256.fullmatch(normalized) is None:
        raise StudyLifecycleError(f"{field} must be 'auto' or a lowercase SHA-256 digest.")
    return normalized


def _object(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise StudyLifecycleError(f"{field} must be a JSON object.")
    return dict(value)


def _exact_keys(value: Mapping[str, Any], *, required: set[str], field: str) -> None:
    actual = set(value)
    missing = sorted(required - actual)
    unknown = sorted(actual - required)
    if missing:
        raise StudyLifecycleError(f"{field} is missing required fields: {', '.join(missing)}.")
    if unknown:
        raise StudyLifecycleError(f"{field} contains unknown fields: {', '.join(unknown)}.")


def _string_list(value: Any, field: str, *, allow_empty: bool = False) -> list[str]:
    if not isinstance(value, list):
        raise StudyLifecycleError(f"{field} must be a JSON array.")
    normalized = [_required_text(item, f"{field} item") for item in value]
    if not allow_empty and not normalized:
        raise StudyLifecycleError(f"{field} must not be empty.")
    if len(normalized) != len(set(normalized)):
        raise StudyLifecycleError(f"{field} must not contain duplicates.")
    return normalized


def _normalize_request(value: Mapping[str, Any]) -> dict[str, Any]:
    raw = _object(value, "study request")
    fields = {
        "schema_version",
        "study_id",
        "title",
        "question",
        "capabilities",
        "assumptions",
        "clarifications",
        "context",
        "fidelity",
        "acceptance_criteria",
    }
    _exact_keys(raw, required=fields, field="study request")
    if raw["schema_version"] != STUDY_REQUEST_SCHEMA:
        raise StudyLifecycleError(f"Unsupported study-request schema: {raw['schema_version']!r}.")
    capabilities = sorted(_string_list(raw["capabilities"], "capabilities"))
    unknown_capabilities = sorted(set(capabilities) - set(CAPABILITY_CONTRACTS))
    if unknown_capabilities:
        raise StudyLifecycleError(f"Unsupported study capabilities: {', '.join(unknown_capabilities)}.")
    assumptions = sorted(_string_list(raw["assumptions"], "assumptions"))
    clarifications_raw = raw["clarifications"]
    if not isinstance(clarifications_raw, list):
        raise StudyLifecycleError("clarifications must be a JSON array.")
    clarifications: list[dict[str, str]] = []
    for index, item in enumerate(clarifications_raw):
        clarification = _object(item, f"clarifications[{index}]")
        _exact_keys(clarification, required={"question", "resolution"}, field=f"clarifications[{index}]")
        clarifications.append(
            {
                "question": _required_text(clarification["question"], "clarification question"),
                "resolution": _required_text(clarification["resolution"], "clarification resolution"),
            }
        )
    context = _object(raw["context"], "context")
    _exact_keys(context, required={"epoch", "time_system", "frame", "units"}, field="context")
    normalized_context = {key: _required_text(context[key], f"context.{key}") for key in sorted(context)}
    fidelity = _object(raw["fidelity"], "fidelity")
    _exact_keys(fidelity, required={"level", "description"}, field="fidelity")
    normalized_fidelity = {
        "level": _required_text(fidelity["level"], "fidelity.level"),
        "description": _required_text(fidelity["description"], "fidelity.description"),
    }
    criteria_raw = raw["acceptance_criteria"]
    if not isinstance(criteria_raw, list) or not criteria_raw:
        raise StudyLifecycleError("acceptance_criteria must be a non-empty JSON array.")
    criteria: list[dict[str, str]] = []
    for index, item in enumerate(criteria_raw):
        criterion = _object(item, f"acceptance_criteria[{index}]")
        _exact_keys(
            criterion,
            required={"criterion_id", "description"},
            field=f"acceptance_criteria[{index}]",
        )
        criteria.append(
            {
                "criterion_id": _identifier(criterion["criterion_id"], "criterion_id", step=True),
                "description": _required_text(criterion["description"], "criterion description"),
            }
        )
    criterion_ids = [item["criterion_id"] for item in criteria]
    if len(criterion_ids) != len(set(criterion_ids)):
        raise StudyLifecycleError("acceptance_criteria criterion_id values must be unique.")
    return {
        "schema_version": STUDY_REQUEST_SCHEMA,
        "study_id": _identifier(raw["study_id"], "study_id"),
        "title": _required_text(raw["title"], "title"),
        "question": _required_text(raw["question"], "question"),
        "capabilities": capabilities,
        "assumptions": assumptions,
        "clarifications": sorted(clarifications, key=lambda item: (item["question"], item["resolution"])),
        "context": normalized_context,
        "fidelity": normalized_fidelity,
        "acceptance_criteria": sorted(criteria, key=lambda item: item["criterion_id"]),
    }


def _normalize_plan(value: Mapping[str, Any], request: Mapping[str, Any]) -> dict[str, Any]:
    raw = _object(value, "study plan")
    fields = {"schema_version", "study_id", "request_sha256", "resource_profile", "steps"}
    _exact_keys(raw, required=fields, field="study plan")
    if raw["schema_version"] != STUDY_PLAN_SCHEMA:
        raise StudyLifecycleError(f"Unsupported study-plan schema: {raw['schema_version']!r}.")
    if raw["study_id"] != request["study_id"]:
        raise StudyLifecycleError("Study plan study_id does not match the request.")
    request_sha256 = _semantic_sha256(request)
    declared_request_sha256 = _digest_or_auto(raw["request_sha256"], "request_sha256")
    if declared_request_sha256 not in {"auto", request_sha256}:
        raise StudyLifecycleError("Study plan request_sha256 does not match the normalized request.")
    profile = _required_text(raw["resource_profile"], "resource_profile")
    if profile not in _RESOURCE_PROFILES:
        raise StudyLifecycleError(f"resource_profile must be one of: {', '.join(sorted(_RESOURCE_PROFILES))}.")
    steps_raw = raw["steps"]
    if not isinstance(steps_raw, list) or not steps_raw:
        raise StudyLifecycleError("steps must be a non-empty JSON array.")
    if len(steps_raw) > MAX_STUDY_STEPS:
        raise StudyLifecycleError(f"A public study plan supports at most {MAX_STUDY_STEPS} steps.")
    criterion_ids = {item["criterion_id"] for item in request["acceptance_criteria"]}
    steps: list[dict[str, Any]] = []
    for index, item in enumerate(steps_raw):
        step = _object(item, f"steps[{index}]")
        step_fields = {
            "step_id",
            "capability",
            "analysis_interface",
            "expected_evidence_schema",
            "depends_on",
            "acceptance_criterion_ids",
        }
        _exact_keys(step, required=step_fields, field=f"steps[{index}]")
        capability = _required_text(step["capability"], "step capability")
        if capability not in CAPABILITY_CONTRACTS:
            raise StudyLifecycleError(f"Unsupported step capability: {capability!r}.")
        if capability not in request["capabilities"]:
            raise StudyLifecycleError(f"Plan step capability {capability!r} is not requested.")
        contract = CAPABILITY_CONTRACTS[capability]
        interface = _required_text(step["analysis_interface"], "analysis_interface")
        if interface != contract["analysis_interface"]:
            raise StudyLifecycleError(f"Plan step {step['step_id']!r} does not use the registered interface.")
        evidence_schema = _required_text(step["expected_evidence_schema"], "expected_evidence_schema")
        if evidence_schema != contract["evidence_schema"]:
            raise StudyLifecycleError(f"Plan step {step['step_id']!r} has the wrong evidence schema.")
        covered = sorted(_string_list(step["acceptance_criterion_ids"], "acceptance_criterion_ids"))
        unknown_criteria = sorted(set(covered) - criterion_ids)
        if unknown_criteria:
            raise StudyLifecycleError(f"Plan step cites unknown criteria: {', '.join(unknown_criteria)}.")
        steps.append(
            {
                "step_id": _identifier(step["step_id"], "step_id", step=True),
                "capability": capability,
                "analysis_interface": interface,
                "expected_evidence_schema": evidence_schema,
                "depends_on": sorted(_string_list(step["depends_on"], "depends_on", allow_empty=True)),
                "acceptance_criterion_ids": covered,
            }
        )
    step_ids = [item["step_id"] for item in steps]
    if len(step_ids) != len(set(step_ids)):
        raise StudyLifecycleError("Study plan step_id values must be unique.")
    known_steps = set(step_ids)
    for step in steps:
        unknown_dependencies = sorted(set(step["depends_on"]) - known_steps)
        if unknown_dependencies:
            raise StudyLifecycleError(
                f"Plan step {step['step_id']!r} has unknown dependencies: {', '.join(unknown_dependencies)}."
            )
        if step["step_id"] in step["depends_on"]:
            raise StudyLifecycleError(f"Plan step {step['step_id']!r} cannot depend on itself.")
    pending = {item["step_id"]: set(item["depends_on"]) for item in steps}
    resolved: set[str] = set()
    while pending:
        ready = sorted(step_id for step_id, dependencies in pending.items() if dependencies <= resolved)
        if not ready:
            raise StudyLifecycleError("Study plan dependencies contain a cycle.")
        for step_id in ready:
            resolved.add(step_id)
            pending.pop(step_id)
    planned_capabilities = {item["capability"] for item in steps}
    if planned_capabilities != set(request["capabilities"]):
        raise StudyLifecycleError("Every requested capability must appear in the study plan.")
    covered_criteria = {criterion for item in steps for criterion in item["acceptance_criterion_ids"]}
    if covered_criteria != criterion_ids:
        missing = sorted(criterion_ids - covered_criteria)
        raise StudyLifecycleError(f"Study plan does not cover every acceptance criterion: {', '.join(missing)}.")
    return {
        "schema_version": STUDY_PLAN_SCHEMA,
        "study_id": request["study_id"],
        "request_sha256": request_sha256,
        "resource_profile": profile,
        "steps": sorted(steps, key=lambda item: item["step_id"]),
    }


def _normalize_claims(value: Mapping[str, Any], request: Mapping[str, Any], plan: Mapping[str, Any]) -> dict[str, Any]:
    raw = _object(value, "study claims")
    fields = {"schema_version", "study_id", "plan_sha256", "claims", "non_claims"}
    _exact_keys(raw, required=fields, field="study claims")
    if raw["schema_version"] != STUDY_CLAIMS_SCHEMA:
        raise StudyLifecycleError(f"Unsupported study-claims schema: {raw['schema_version']!r}.")
    if raw["study_id"] != request["study_id"]:
        raise StudyLifecycleError("Study claims study_id does not match the request.")
    plan_sha256 = _semantic_sha256(plan)
    declared_plan_sha256 = _digest_or_auto(raw["plan_sha256"], "plan_sha256")
    if declared_plan_sha256 not in {"auto", plan_sha256}:
        raise StudyLifecycleError("Study claims plan_sha256 does not match the normalized plan.")
    steps_by_id = {item["step_id"]: item for item in plan["steps"]}
    step_ids = set(steps_by_id)
    criterion_ids = {item["criterion_id"] for item in request["acceptance_criteria"]}
    claims_raw = raw["claims"]
    if not isinstance(claims_raw, list):
        raise StudyLifecycleError("claims must be a JSON array.")
    claims: list[dict[str, Any]] = []
    for index, item in enumerate(claims_raw):
        claim = _object(item, f"claims[{index}]")
        claim_fields = {"claim_id", "statement", "validation_level", "criterion_ids", "evidence"}
        _exact_keys(claim, required=claim_fields, field=f"claims[{index}]")
        level = _required_text(claim["validation_level"], "validation_level").upper()
        if level not in _VALIDATION_LEVELS:
            raise StudyLifecycleError(f"validation_level must be one of: {', '.join(sorted(_VALIDATION_LEVELS))}.")
        cited_criteria = sorted(_string_list(claim["criterion_ids"], "claim criterion_ids"))
        unknown_criteria = sorted(set(cited_criteria) - criterion_ids)
        if unknown_criteria:
            raise StudyLifecycleError(f"Claim cites unknown criteria: {', '.join(unknown_criteria)}.")
        references_raw = claim["evidence"]
        if not isinstance(references_raw, list) or not references_raw:
            raise StudyLifecycleError("Each claim requires at least one evidence reference.")
        references: list[dict[str, str]] = []
        for reference_index, reference_value in enumerate(references_raw):
            reference = _object(reference_value, f"claims[{index}].evidence[{reference_index}]")
            _exact_keys(
                reference,
                required={"step_id", "json_pointer"},
                field=f"claims[{index}].evidence[{reference_index}]",
            )
            step_id = _identifier(reference["step_id"], "evidence step_id", step=True)
            if step_id not in step_ids:
                raise StudyLifecycleError(f"Claim cites unknown evidence step {step_id!r}.")
            pointer = _required_text(reference["json_pointer"], "json_pointer")
            if not pointer.startswith("/"):
                raise StudyLifecycleError("json_pointer must begin with '/'.")
            references.append({"step_id": step_id, "json_pointer": pointer})
        cited_steps = {reference["step_id"] for reference in references}
        cited_step_criteria = {
            criterion_id for step_id in cited_steps for criterion_id in steps_by_id[step_id]["acceptance_criterion_ids"]
        }
        unbound_criteria = sorted(set(cited_criteria) - cited_step_criteria)
        if unbound_criteria:
            raise StudyLifecycleError(
                "Claim criteria must be covered by a cited evidence step: " + ", ".join(unbound_criteria) + "."
            )
        maximum_supported_rank = min(
            _VALIDATION_RANK[str(CAPABILITY_CONTRACTS[steps_by_id[step_id]["capability"]]["max_validation_level"])]
            for step_id in cited_steps
        )
        if _VALIDATION_RANK[level] > maximum_supported_rank:
            supported = f"VC-{maximum_supported_rank}"
            raise StudyLifecycleError(
                f"Claim validation_level {level} exceeds the cited evidence-step maximum {supported}."
            )
        claims.append(
            {
                "claim_id": _identifier(claim["claim_id"], "claim_id", step=True),
                "statement": _required_text(claim["statement"], "claim statement"),
                "validation_level": level,
                "criterion_ids": cited_criteria,
                "evidence": sorted(references, key=lambda item: (item["step_id"], item["json_pointer"])),
            }
        )
    claim_ids = [item["claim_id"] for item in claims]
    if len(claim_ids) != len(set(claim_ids)):
        raise StudyLifecycleError("Study claim_id values must be unique.")
    non_claims = sorted(_string_list(raw["non_claims"], "non_claims"))
    return {
        "schema_version": STUDY_CLAIMS_SCHEMA,
        "study_id": request["study_id"],
        "plan_sha256": plan_sha256,
        "claims": sorted(claims, key=lambda item: item["claim_id"]),
        "non_claims": non_claims,
    }


@dataclass(frozen=True)
class StudyRequest:
    payload: dict[str, Any]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> StudyRequest:
        return cls(_normalize_request(value))

    def to_dict(self) -> dict[str, Any]:
        return _clone(self.payload)


@dataclass(frozen=True)
class StudyPlan:
    payload: dict[str, Any]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], request: StudyRequest | Mapping[str, Any]) -> StudyPlan:
        request_value = request.to_dict() if isinstance(request, StudyRequest) else request
        normalized_request = _normalize_request(request_value)
        return cls(_normalize_plan(value, normalized_request))

    def to_dict(self) -> dict[str, Any]:
        return _clone(self.payload)


@dataclass(frozen=True)
class StudyClaims:
    payload: dict[str, Any]

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        request: StudyRequest | Mapping[str, Any],
        plan: StudyPlan | Mapping[str, Any],
    ) -> StudyClaims:
        request_value = request.to_dict() if isinstance(request, StudyRequest) else request
        normalized_request = _normalize_request(request_value)
        plan_value = plan.to_dict() if isinstance(plan, StudyPlan) else plan
        normalized_plan = _normalize_plan(plan_value, normalized_request)
        return cls(_normalize_claims(value, normalized_request, normalized_plan))

    def to_dict(self) -> dict[str, Any]:
        return _clone(self.payload)


@dataclass(frozen=True)
class StudyBundleArtifacts:
    output_dir: Path
    request_json: Path
    plan_json: Path
    run_json: Path
    evidence_json: Path
    claims_json: Path
    receipt_json: Path


def _read_json_bytes(
    path: Path, field: str, *, byte_limit: int = MAX_STUDY_EVIDENCE_BYTES
) -> tuple[dict[str, Any], bytes]:
    try:
        content = read_regular_file_nofollow(path, min_bytes=1, max_bytes=byte_limit)
    except SafeReadError as exc:
        raise StudyLifecycleError(f"Could not safely read {field}: {exc}") from exc
    try:
        value = json.loads(content.decode("utf-8"), parse_constant=_reject_constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise StudyLifecycleError(f"Could not parse {field} as finite UTF-8 JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise StudyLifecycleError(f"{field} must contain a JSON object.")
    return value, content


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(dict(value), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _domain_fields(value: Mapping[str, Any], required: set[str], field: str) -> None:
    missing = sorted(required - set(value))
    if missing:
        raise StudyLifecycleError(f"{field} is missing required fields: {', '.join(missing)}.")


def _domain_object(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise StudyLifecycleError(f"{field} must be a JSON object.")
    return value


def _domain_list(value: Any, field: str, *, allow_empty: bool = False) -> list[Any]:
    if not isinstance(value, list) or (not allow_empty and not value):
        suffix = "" if allow_empty else " non-empty"
        raise StudyLifecycleError(f"{field} must be a{suffix} JSON array.")
    return value


def _domain_number(value: Any, field: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise StudyLifecycleError(f"{field} must be a finite number.")
    normalized = float(value)
    if minimum is not None and normalized < minimum:
        raise StudyLifecycleError(f"{field} must be at least {minimum}.")
    return normalized


def _domain_count(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise StudyLifecycleError(f"{field} must be a nonnegative integer.")
    return value


def _domain_digest(value: Any, field: str) -> str:
    normalized = _required_text(value, field)
    if normalized != normalized.lower() or _SHA256.fullmatch(normalized) is None:
        raise StudyLifecycleError(f"{field} must be a lowercase SHA-256 digest.")
    return normalized


def _domain_text_list(value: Any, field: str) -> list[str]:
    return _string_list(value, field)


def _validate_trajectory_targeting_evidence(value: Mapping[str, Any]) -> None:
    field = "trajectory-targeting evidence"
    _domain_fields(
        value,
        {
            "problem_name",
            "problem_sha256",
            "converged",
            "variables",
            "constraints",
            "decision_values",
            "solution_execution",
            "solution_constraint_evaluation",
            "authoritative_repropagation",
            "resources",
            "limitations",
        },
        field,
    )
    _required_text(value["problem_name"], f"{field} problem_name")
    _domain_digest(value["problem_sha256"], f"{field} problem_sha256")
    if value["converged"] is not True:
        raise StudyLifecycleError(f"{field} must declare converged=true for accepted evidence.")
    _domain_list(value["variables"], f"{field} variables")
    _domain_list(value["constraints"], f"{field} constraints")
    decisions = _domain_list(value["decision_values"], f"{field} decision_values")
    for index, decision in enumerate(decisions):
        _domain_number(decision, f"{field} decision_values[{index}]")
    _domain_object(value["solution_execution"], f"{field} solution_execution")
    constraint_evaluation = _domain_object(
        value["solution_constraint_evaluation"], f"{field} solution_constraint_evaluation"
    )
    if constraint_evaluation.get("all_satisfied") is not True:
        raise StudyLifecycleError(f"{field} solution constraints are not satisfied.")
    repropagation = _domain_object(value["authoritative_repropagation"], f"{field} authoritative_repropagation")
    _domain_fields(
        repropagation,
        {"status", "execution", "constraint_evaluation"},
        f"{field} authoritative_repropagation",
    )
    if repropagation["status"] != "verified":
        raise StudyLifecycleError(f"{field} authoritative repropagation is not verified.")
    _domain_object(repropagation["execution"], f"{field} authoritative repropagation execution")
    repropagation_constraints = _domain_object(
        repropagation["constraint_evaluation"],
        f"{field} authoritative repropagation constraint_evaluation",
    )
    if repropagation_constraints.get("all_satisfied") is not True:
        raise StudyLifecycleError(f"{field} authoritative repropagation constraints are not satisfied.")
    _domain_object(value["resources"], f"{field} resources")
    _domain_text_list(value["limitations"], f"{field} limitations")


def _validate_constellation_design_evidence(value: Mapping[str, Any]) -> None:
    field = "constellation-design evidence"
    _domain_fields(
        value,
        {
            "analysis_id",
            "input_semantic_sha256",
            "result_semantic_sha256",
            "ranking",
            "recommended_design_id",
            "candidate_results",
            "resource_estimate",
            "claim_limits",
        },
        field,
    )
    _required_text(value["analysis_id"], f"{field} analysis_id")
    _domain_digest(value["input_semantic_sha256"], f"{field} input_semantic_sha256")
    _domain_digest(value["result_semantic_sha256"], f"{field} result_semantic_sha256")
    ranking = _domain_list(value["ranking"], f"{field} ranking")
    ranking_ids = [_required_text(item, f"{field} ranking item") for item in ranking]
    if len(ranking_ids) != len(set(ranking_ids)):
        raise StudyLifecycleError(f"{field} ranking must contain unique design IDs.")
    candidates = _domain_list(value["candidate_results"], f"{field} candidate_results")
    if len(candidates) != len(ranking_ids):
        raise StudyLifecycleError(f"{field} candidate count must match ranking count.")
    for index, candidate_value in enumerate(candidates):
        candidate = _domain_object(candidate_value, f"{field} candidate_results[{index}]")
        _domain_fields(
            candidate,
            {"design_id", "rank", "feasible", "score", "score_components", "generated_members", "coverage", "network"},
            f"{field} candidate_results[{index}]",
        )
        if _required_text(candidate["design_id"], f"{field} candidate design_id") != ranking_ids[index]:
            raise StudyLifecycleError(f"{field} candidate ordering differs from ranking.")
        if candidate["rank"] != index + 1:
            raise StudyLifecycleError(f"{field} candidate rank is not canonical.")
        if not isinstance(candidate["feasible"], bool):
            raise StudyLifecycleError(f"{field} candidate feasible must be Boolean.")
        _domain_number(candidate["score"], f"{field} candidate score")
        _domain_object(candidate["score_components"], f"{field} candidate score_components")
        _domain_list(candidate["generated_members"], f"{field} candidate generated_members")
        coverage = _domain_object(candidate["coverage"], f"{field} candidate coverage")
        coverage_fraction = _domain_number(
            coverage.get("time_weighted_mean_covered_fraction"),
            f"{field} candidate coverage fraction",
            minimum=0.0,
        )
        if coverage_fraction > 1.0:
            raise StudyLifecycleError(f"{field} candidate coverage fraction must not exceed 1.")
        _domain_digest(coverage.get("interval_semantic_sha256"), f"{field} candidate coverage digest")
        network = _domain_object(candidate["network"], f"{field} candidate network")
        availability = _domain_number(
            network.get("union_sampled_available_fraction"),
            f"{field} candidate network availability",
            minimum=0.0,
        )
        if availability > 1.0:
            raise StudyLifecycleError(f"{field} candidate network availability must not exceed 1.")
    recommended_value = value["recommended_design_id"]
    if recommended_value is None:
        if any(bool(candidate.get("feasible")) for candidate in candidates):
            raise StudyLifecycleError(f"{field} must recommend the first feasible ranked design.")
    elif _required_text(recommended_value, f"{field} recommended_design_id") != ranking_ids[0]:
        raise StudyLifecycleError(f"{field} recommended design must be first in ranking.")
    elif not bool(candidates[0].get("feasible")):
        raise StudyLifecycleError(f"{field} must not recommend an infeasible design.")
    resources = _domain_object(value["resource_estimate"], f"{field} resource_estimate")
    for name in (
        "candidate_count",
        "sample_count",
        "total_satellite_candidates",
        "coverage_cell_time_comparisons",
        "link_samples",
    ):
        _domain_count(resources.get(name), f"{field} resource_estimate {name}")
    if resources["candidate_count"] != len(ranking_ids):
        raise StudyLifecycleError(f"{field} resource candidate count differs from ranking.")
    _domain_text_list(value["claim_limits"], f"{field} claim_limits")


def _validate_conjunction_assessment_evidence(value: Mapping[str, Any]) -> None:
    field = "conjunction-assessment evidence"
    _domain_fields(
        value,
        {"problem_name", "problem_sha256", "baseline", "avoidance_candidates", "resources", "limitations"},
        field,
    )
    _required_text(value["problem_name"], f"{field} problem_name")
    _domain_digest(value["problem_sha256"], f"{field} problem_sha256")
    baseline = _domain_object(value["baseline"], f"{field} baseline")
    _domain_fields(
        baseline,
        {"primary_id", "secondary_id", "closest_approach", "encounter_frame", "covariance_projection", "probability"},
        f"{field} baseline",
    )
    _required_text(baseline["primary_id"], f"{field} baseline primary_id")
    _required_text(baseline["secondary_id"], f"{field} baseline secondary_id")
    closest = _domain_object(baseline["closest_approach"], f"{field} baseline closest_approach")
    _domain_fields(closest, {"time_s", "miss_distance_km", "relative_speed_km_s"}, f"{field} closest_approach")
    _domain_number(closest["time_s"], f"{field} closest_approach time_s", minimum=0.0)
    _domain_number(closest["miss_distance_km"], f"{field} closest_approach miss_distance_km", minimum=0.0)
    _domain_number(closest["relative_speed_km_s"], f"{field} closest_approach relative_speed_km_s", minimum=0.0)
    probability = _domain_object(baseline["probability"], f"{field} baseline probability")
    collision_probability = _domain_number(
        probability.get("collision_probability"), f"{field} baseline collision_probability", minimum=0.0
    )
    if collision_probability > 1.0:
        raise StudyLifecycleError(f"{field} baseline collision_probability must not exceed 1.")
    _domain_object(baseline["encounter_frame"], f"{field} baseline encounter_frame")
    _domain_object(baseline["covariance_projection"], f"{field} baseline covariance_projection")
    _domain_list(value["avoidance_candidates"], f"{field} avoidance_candidates", allow_empty=True)
    resources = _domain_object(value["resources"], f"{field} resources")
    _domain_fields(
        resources,
        {"primary_samples", "secondary_samples", "screening_object_count", "candidate_count"},
        f"{field} resources",
    )
    for name in ("primary_samples", "secondary_samples", "screening_object_count", "candidate_count"):
        _domain_count(resources[name], f"{field} resources {name}")
    _domain_text_list(value["limitations"], f"{field} limitations")


def _validate_mission_scheduling_evidence(value: Mapping[str, Any]) -> None:
    field = "mission-scheduling evidence"
    _domain_fields(
        value,
        {
            "analysis_id",
            "solver",
            "candidate_count",
            "asset_count",
            "station_count",
            "evaluated_subset_count",
            "feasible_subset_count",
            "selected_count",
            "selected_observation_count",
            "objective_value",
            "input_semantic_sha256",
            "schedule_semantic_sha256",
            "source_product_sha256s",
            "claim_limits",
        },
        field,
    )
    _required_text(value["analysis_id"], f"{field} analysis_id")
    if value["solver"] != "deterministic_exact_exhaustive_enumeration":
        raise StudyLifecycleError(f"{field} has an unsupported solver declaration.")
    for name in (
        "candidate_count",
        "asset_count",
        "station_count",
        "evaluated_subset_count",
        "feasible_subset_count",
        "selected_count",
        "selected_observation_count",
    ):
        _domain_count(value[name], f"{field} {name}")
    if value["selected_count"] > value["candidate_count"]:
        raise StudyLifecycleError(f"{field} selected_count exceeds candidate_count.")
    if value["feasible_subset_count"] > value["evaluated_subset_count"]:
        raise StudyLifecycleError(f"{field} feasible_subset_count exceeds evaluated_subset_count.")
    _domain_number(value["objective_value"], f"{field} objective_value", minimum=0.0)
    _domain_digest(value["input_semantic_sha256"], f"{field} input_semantic_sha256")
    _domain_digest(value["schedule_semantic_sha256"], f"{field} schedule_semantic_sha256")
    source_digests = _domain_list(value["source_product_sha256s"], f"{field} source_product_sha256s", allow_empty=True)
    for index, digest in enumerate(source_digests):
        _domain_digest(digest, f"{field} source_product_sha256s[{index}]")
    if source_digests != sorted(set(source_digests)):
        raise StudyLifecycleError(f"{field} source_product_sha256s must be unique and sorted.")
    _domain_text_list(value["claim_limits"], f"{field} claim_limits")


def _validate_orbit_lifetime_evidence(value: Mapping[str, Any]) -> None:
    field = "orbit-lifetime evidence"
    _domain_fields(
        value,
        {
            "analysis_id",
            "asset_id",
            "outcome",
            "problem_semantic_sha256",
            "result_semantic_sha256",
            "propagator",
            "resource_use",
            "initial",
            "final",
            "thresholds",
            "claim_limits",
        },
        field,
    )
    _required_text(value["analysis_id"], f"{field} analysis_id")
    _required_text(value["asset_id"], f"{field} asset_id")
    _required_text(value["outcome"], f"{field} outcome")
    _domain_digest(value["problem_semantic_sha256"], f"{field} problem_semantic_sha256")
    _domain_digest(value["result_semantic_sha256"], f"{field} result_semantic_sha256")
    propagator = _domain_object(value["propagator"], f"{field} propagator")
    if propagator.get("family") != "ONP":
        raise StudyLifecycleError(f"{field} propagator family must be ONP.")
    resources = _domain_object(value["resource_use"], f"{field} resource_use")
    _domain_fields(
        resources,
        {"integration_steps", "output_samples", "event_count", "propagated_duration_s"},
        f"{field} resource_use",
    )
    for name in ("integration_steps", "output_samples", "event_count"):
        _domain_count(resources[name], f"{field} resource_use {name}")
    _domain_number(resources["propagated_duration_s"], f"{field} propagated_duration_s", minimum=0.0)
    _domain_object(value["initial"], f"{field} initial")
    final = _domain_object(value["final"], f"{field} final")
    _domain_number(final.get("time_s"), f"{field} final time_s", minimum=0.0)
    _domain_object(value["thresholds"], f"{field} thresholds")
    _domain_text_list(value["claim_limits"], f"{field} claim_limits")


def _validate_spacecraft_power_evidence(value: Mapping[str, Any]) -> None:
    field = "spacecraft-power evidence"
    _domain_fields(
        value,
        {
            "analysis_id",
            "asset_id",
            "feasibility",
            "model",
            "problem_semantic_sha256",
            "history_semantic_sha256",
            "result_semantic_sha256",
            "sample_count",
            "illumination_interval_count",
            "event_count",
            "totals",
            "battery",
            "conservation_residuals_wh",
            "source_product_sha256s",
            "claim_limits",
        },
        field,
    )
    _required_text(value["analysis_id"], f"{field} analysis_id")
    _required_text(value["asset_id"], f"{field} asset_id")
    if value["feasibility"] not in {"feasible", "infeasible"}:
        raise StudyLifecycleError(f"{field} feasibility must be feasible or infeasible.")
    if value["model"] != "deterministic_sampled_solar_array_battery_v1":
        raise StudyLifecycleError(f"{field} has an unsupported model declaration.")
    for name in ("problem_semantic_sha256", "history_semantic_sha256", "result_semantic_sha256"):
        _domain_digest(value[name], f"{field} {name}")
    for name in ("sample_count", "illumination_interval_count", "event_count"):
        _domain_count(value[name], f"{field} {name}")
    _domain_object(value["totals"], f"{field} totals")
    battery = _domain_object(value["battery"], f"{field} battery")
    for name in ("initial_soc_fraction", "final_soc_fraction", "minimum_soc_fraction", "maximum_soc_fraction"):
        amount = _domain_number(battery.get(name), f"{field} battery {name}", minimum=0.0)
        if amount > 1.0:
            raise StudyLifecycleError(f"{field} battery {name} must not exceed 1.")
    residuals = _domain_object(value["conservation_residuals_wh"], f"{field} conservation_residuals_wh")
    for name in ("battery_storage", "power_bus", "load_service"):
        _domain_number(residuals.get(name), f"{field} conservation residual {name}")
    source_digests = _domain_list(value["source_product_sha256s"], f"{field} source_product_sha256s", allow_empty=True)
    for index, digest in enumerate(source_digests):
        _domain_digest(digest, f"{field} source_product_sha256s[{index}]")
    if source_digests != sorted(set(source_digests)):
        raise StudyLifecycleError(f"{field} source_product_sha256s must be unique and sorted.")
    _domain_text_list(value["claim_limits"], f"{field} claim_limits")


_CAPABILITY_EVIDENCE_VALIDATORS = {
    "constellation_design": _validate_constellation_design_evidence,
    "conjunction_assessment": _validate_conjunction_assessment_evidence,
    "mission_scheduling": _validate_mission_scheduling_evidence,
    "orbit_lifetime": _validate_orbit_lifetime_evidence,
    "spacecraft_power": _validate_spacecraft_power_evidence,
    "trajectory_targeting": _validate_trajectory_targeting_evidence,
}


def _validate_domain_evidence(step: Mapping[str, Any], value: Mapping[str, Any]) -> str:
    contract = CAPABILITY_CONTRACTS[str(step["capability"])]
    if value.get("schema_version") != contract["evidence_schema"]:
        raise StudyLifecycleError(
            f"Evidence for step {step['step_id']!r} has schema {value.get('schema_version')!r}; "
            f"expected {contract['evidence_schema']!r}."
        )
    status = _required_text(value.get("status"), f"evidence status for {step['step_id']}")
    if status not in contract["accepted_statuses"]:
        raise StudyLifecycleError(
            f"Evidence for step {step['step_id']!r} has non-accepted status {status!r}; "
            f"expected one of {', '.join(contract['accepted_statuses'])}."
        )
    _CAPABILITY_EVIDENCE_VALIDATORS[str(step["capability"])](value)
    return status


def _resolve_json_pointer(value: Any, pointer: str, field: str) -> Any:
    current = value
    for token in pointer.split("/")[1:]:
        decoded = token.replace("~1", "/").replace("~0", "~")
        if isinstance(current, dict):
            if decoded not in current:
                raise StudyLifecycleError(f"{field} points to missing object member {decoded!r}.")
            current = current[decoded]
        elif isinstance(current, list):
            if not decoded.isdigit() or int(decoded) >= len(current):
                raise StudyLifecycleError(f"{field} points to invalid array index {decoded!r}.")
            current = current[int(decoded)]
        else:
            raise StudyLifecycleError(f"{field} traverses through a scalar value.")
    return current


def _evidence_record(
    study_id: str,
    plan: Mapping[str, Any],
    retained_root: Path,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    entries: list[dict[str, Any]] = []
    values: dict[str, dict[str, Any]] = {}
    for step in plan["steps"]:
        step_id = step["step_id"]
        relative = Path("evidence") / f"{step_id}.json"
        value, content = _read_json_bytes(retained_root / relative, f"retained evidence for {step_id}")
        status = _validate_domain_evidence(step, value)
        values[step_id] = value
        entries.append(
            {
                "step_id": step_id,
                "capability": step["capability"],
                "schema_version": value["schema_version"],
                "status": status,
                "retained_path": relative.as_posix(),
                "bytes": len(content),
                "sha256": _bytes_sha256(content),
                "semantic_sha256": _semantic_sha256(value),
            }
        )
    steps_by_id = {step["step_id"]: step for step in plan["steps"]}
    for step in plan["steps"]:
        if step["capability"] != "spacecraft_power":
            continue
        power_sources = set(values[step["step_id"]]["source_product_sha256s"])
        for dependency_id in step["depends_on"]:
            if steps_by_id[dependency_id]["capability"] != "mission_scheduling":
                continue
            schedule_digest = values[dependency_id]["schedule_semantic_sha256"]
            if schedule_digest not in power_sources:
                raise StudyLifecycleError(
                    f"Power step {step['step_id']!r} does not bind the retained schedule "
                    f"from dependency {dependency_id!r}."
                )
    return (
        {
            "schema_version": STUDY_EVIDENCE_SCHEMA,
            "study_id": study_id,
            "plan_sha256": _semantic_sha256(plan),
            "status": "complete",
            "steps": entries,
        },
        values,
    )


def _run_record(study_id: str, plan: Mapping[str, Any], evidence: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": STUDY_RUN_SCHEMA,
        "study_id": study_id,
        "plan_sha256": _semantic_sha256(plan),
        "status": "completed",
        "execution_mode": "bound_completed_evidence",
        "resource_profile": plan["resource_profile"],
        "steps": [
            {
                "step_id": item["step_id"],
                "status": "evidence_bound",
                "evidence_sha256": item["sha256"],
            }
            for item in evidence["steps"]
        ],
    }


def _validate_claim_citations(claims: Mapping[str, Any], evidence_values: Mapping[str, Mapping[str, Any]]) -> None:
    for claim in claims["claims"]:
        for reference in claim["evidence"]:
            _resolve_json_pointer(
                evidence_values[reference["step_id"]],
                reference["json_pointer"],
                f"claim {claim['claim_id']!r} evidence pointer",
            )


def _file_receipt(path: Path, root: Path) -> dict[str, Any]:
    value, content = _read_json_bytes(path, f"study record {path.name}")
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": len(content),
        "sha256": _bytes_sha256(content),
        "semantic_sha256": _semantic_sha256(value),
    }


def _receipt_record(study_id: str, root: Path) -> dict[str, Any]:
    records = [_file_receipt(root / name, root) for name in _ROOT_RECORD_FILES]
    semantic = _semantic_sha256(
        {
            "study_id": study_id,
            "records": [{"path": item["path"], "semantic_sha256": item["semantic_sha256"]} for item in records],
        }
    )
    return {
        "schema_version": STUDY_RECEIPT_SCHEMA,
        "study_id": study_id,
        "status": "verified",
        "records": records,
        "bundle_semantic_sha256": semantic,
        "limitations": [
            "Study replay verifies retained lifecycle records and evidence identity; it does not rerun domain physics.",
            "Scientific recomputation remains governed by each cited capability's authoritative replay contract.",
            "A verified study receipt is engineering provenance, not operational authorization or flight qualification.",
        ],
    }


def build_study_bundle(
    request: StudyRequest | Mapping[str, Any],
    plan: StudyPlan | Mapping[str, Any],
    claims: StudyClaims | Mapping[str, Any],
    evidence_sources: Mapping[str, str | Path],
    output_dir: str | Path,
) -> StudyBundleArtifacts:
    """Retain completed evidence and build one deterministic public study bundle."""

    request_value = request.to_dict() if isinstance(request, StudyRequest) else request
    normalized_request = _normalize_request(request_value)
    plan_value = plan.to_dict() if isinstance(plan, StudyPlan) else plan
    normalized_plan = _normalize_plan(plan_value, normalized_request)
    claims_value = claims.to_dict() if isinstance(claims, StudyClaims) else claims
    normalized_claims = _normalize_claims(claims_value, normalized_request, normalized_plan)
    expected_steps = {item["step_id"] for item in normalized_plan["steps"]}
    supplied_steps = set(evidence_sources)
    if supplied_steps != expected_steps:
        missing = sorted(expected_steps - supplied_steps)
        extra = sorted(supplied_steps - expected_steps)
        details = [*(f"missing={item}" for item in missing), *(f"extra={item}" for item in extra)]
        raise StudyLifecycleError("Evidence bindings must exactly match plan steps: " + ", ".join(details))
    destination_input = Path(output_dir).expanduser()
    if destination_input.is_symlink():
        raise StudyLifecycleError(f"output_dir must not be a symbolic link: {destination_input}.")
    destination = destination_input.resolve()
    if destination.exists():
        raise StudyLifecycleError(f"output_dir must not already exist: {destination}.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.building-", dir=destination.parent))
    try:
        evidence_root = temporary / "evidence"
        evidence_root.mkdir()
        by_step = {item["step_id"]: item for item in normalized_plan["steps"]}
        for step_id in sorted(expected_steps):
            source_input = Path(evidence_sources[step_id]).expanduser()
            if source_input.is_symlink():
                raise StudyLifecycleError(f"evidence source for {step_id} must not be a symbolic link: {source_input}.")
            source = source_input.resolve()
            value, content = _read_json_bytes(source, f"evidence source for {step_id}")
            _validate_domain_evidence(by_step[step_id], value)
            (evidence_root / f"{step_id}.json").write_bytes(content)
        evidence_record, evidence_values = _evidence_record(normalized_request["study_id"], normalized_plan, temporary)
        _validate_claim_citations(normalized_claims, evidence_values)
        run_record = _run_record(normalized_request["study_id"], normalized_plan, evidence_record)
        _write_json(temporary / _REQUEST_FILE, normalized_request)
        _write_json(temporary / _PLAN_FILE, normalized_plan)
        _write_json(temporary / _RUN_FILE, run_record)
        _write_json(temporary / _EVIDENCE_FILE, evidence_record)
        _write_json(temporary / _CLAIMS_FILE, normalized_claims)
        receipt = _receipt_record(normalized_request["study_id"], temporary)
        _write_json(temporary / _RECEIPT_FILE, receipt)
        os.replace(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return StudyBundleArtifacts(
        output_dir=destination,
        request_json=destination / _REQUEST_FILE,
        plan_json=destination / _PLAN_FILE,
        run_json=destination / _RUN_FILE,
        evidence_json=destination / _EVIDENCE_FILE,
        claims_json=destination / _CLAIMS_FILE,
        receipt_json=destination / _RECEIPT_FILE,
    )


def _read_bundle_records(root: Path) -> dict[str, dict[str, Any]]:
    if root.is_symlink() or not root.is_dir():
        raise StudyLifecycleError(f"Study bundle must be a regular directory: {root}.")
    actual_root_entries = {path.name for path in root.iterdir()}
    expected_root_entries = {*_ALL_ROOT_FILES, "evidence"}
    if actual_root_entries != expected_root_entries:
        missing = sorted(expected_root_entries - actual_root_entries)
        extra = sorted(actual_root_entries - expected_root_entries)
        details = [*(f"missing={item}" for item in missing), *(f"extra={item}" for item in extra)]
        raise StudyLifecycleError("Study bundle contains an unexpected root artifact set: " + ", ".join(details))
    return {name: _read_json_bytes(root / name, name)[0] for name in _ALL_ROOT_FILES}


def _validate_evidence_inventory(root: Path, plan: Mapping[str, Any]) -> None:
    expected_names = {f"{item['step_id']}.json" for item in plan["steps"]}
    evidence_root = root / "evidence"
    if evidence_root.is_symlink() or not evidence_root.is_dir():
        raise StudyLifecycleError("Study evidence path must be a regular directory.")
    entries = list(evidence_root.iterdir())
    if {path.name for path in entries} != expected_names:
        raise StudyLifecycleError("Study evidence directory contains an unexpected artifact set.")
    for path in entries:
        if path.is_symlink() or not path.is_file():
            raise StudyLifecycleError(f"Study evidence artifact must be a regular file: {path.name}.")


def verify_study_bundle(bundle_dir: str | Path) -> dict[str, Any]:
    """Verify every record, retained evidence file, citation, and content receipt."""

    root_input = Path(bundle_dir).expanduser()
    if root_input.is_symlink():
        raise StudyLifecycleError(f"Study bundle must not be a symbolic link: {root_input}.")
    root = root_input.resolve()
    records = _read_bundle_records(root)
    request = _normalize_request(records[_REQUEST_FILE])
    if records[_REQUEST_FILE] != request:
        raise StudyLifecycleError("Persisted study request is not normalized.")
    plan = _normalize_plan(records[_PLAN_FILE], request)
    if records[_PLAN_FILE] != plan:
        raise StudyLifecycleError("Persisted study plan is not normalized and content-bound.")
    claims = _normalize_claims(records[_CLAIMS_FILE], request, plan)
    if records[_CLAIMS_FILE] != claims:
        raise StudyLifecycleError("Persisted study claims are not normalized and content-bound.")
    _validate_evidence_inventory(root, plan)
    expected_evidence, evidence_values = _evidence_record(request["study_id"], plan, root)
    if records[_EVIDENCE_FILE] != expected_evidence:
        raise StudyLifecycleError("Study evidence record does not match the retained evidence files.")
    _validate_claim_citations(claims, evidence_values)
    expected_run = _run_record(request["study_id"], plan, expected_evidence)
    if records[_RUN_FILE] != expected_run:
        raise StudyLifecycleError("Study run record does not match the bound completed evidence.")
    receipt = records[_RECEIPT_FILE]
    if receipt.get("schema_version") != STUDY_RECEIPT_SCHEMA:
        raise StudyLifecycleError("Study receipt has an unsupported schema version.")
    expected_receipt = _receipt_record(request["study_id"], root)
    if receipt != expected_receipt:
        raise StudyLifecycleError("Study receipt does not match the authoritative bundle contents.")
    # The receipt binds the root lifecycle records, while the evidence record
    # binds the retained evidence files.  Re-read the retained evidence after
    # the root-record checks so a concurrent replacement cannot be accepted
    # solely because the first evidence read happened before the replacement.
    # Each read is still performed through _read_json_bytes, which gives us
    # the existing bounded, no-follow, stable-file read semantics.
    rechecked_evidence, rechecked_values = _evidence_record(request["study_id"], plan, root)
    if rechecked_evidence != expected_evidence:
        raise StudyLifecycleError("Study evidence changed during verification.")
    _validate_claim_citations(claims, rechecked_values)
    return {
        "schema_version": STUDY_VERIFICATION_SCHEMA,
        "status": "verified",
        "study_id": request["study_id"],
        "bundle_semantic_sha256": receipt["bundle_semantic_sha256"],
        "request_sha256": _semantic_sha256(request),
        "plan_sha256": _semantic_sha256(plan),
        "run_sha256": _semantic_sha256(expected_run),
        "evidence_sha256": _semantic_sha256(expected_evidence),
        "claims_sha256": _semantic_sha256(claims),
        "step_count": len(plan["steps"]),
        "claim_count": len(claims["claims"]),
        "non_claim_count": len(claims["non_claims"]),
    }


def inspect_study_bundle(bundle_dir: str | Path) -> dict[str, Any]:
    verification = verify_study_bundle(bundle_dir)
    root = Path(bundle_dir).expanduser().resolve()
    request = _read_json_bytes(root / _REQUEST_FILE, _REQUEST_FILE)[0]
    plan = _read_json_bytes(root / _PLAN_FILE, _PLAN_FILE)[0]
    claims = _read_json_bytes(root / _CLAIMS_FILE, _CLAIMS_FILE)[0]
    return {
        **verification,
        "title": request["title"],
        "question": request["question"],
        "capabilities": request["capabilities"],
        "steps": [{"step_id": item["step_id"], "capability": item["capability"]} for item in plan["steps"]],
        "claims": [
            {
                "claim_id": item["claim_id"],
                "statement": item["statement"],
                "validation_level": item["validation_level"],
            }
            for item in claims["claims"]
        ],
        "non_claims": claims["non_claims"],
    }


def replay_study_bundle(bundle_dir: str | Path) -> dict[str, Any]:
    verification = verify_study_bundle(bundle_dir)
    return {**verification, "replay_status": "identity_verified"}


def compare_study_bundles(left_dir: str | Path, right_dir: str | Path) -> dict[str, Any]:
    left = verify_study_bundle(left_dir)
    right = verify_study_bundle(right_dir)
    left_root = Path(left_dir).expanduser().resolve()
    right_root = Path(right_dir).expanduser().resolve()
    record_differences: list[str] = []
    for name in _ROOT_RECORD_FILES:
        left_value = _read_json_bytes(left_root / name, f"left {name}")[0]
        right_value = _read_json_bytes(right_root / name, f"right {name}")[0]
        if _semantic_sha256(left_value) != _semantic_sha256(right_value):
            record_differences.append(name)
    left_evidence = _read_json_bytes(left_root / _EVIDENCE_FILE, "left study evidence")[0]
    right_evidence = _read_json_bytes(right_root / _EVIDENCE_FILE, "right study evidence")[0]
    left_steps = {item["step_id"]: item["semantic_sha256"] for item in left_evidence["steps"]}
    right_steps = {item["step_id"]: item["semantic_sha256"] for item in right_evidence["steps"]}
    changed_steps = sorted(
        step_id for step_id in set(left_steps) | set(right_steps) if left_steps.get(step_id) != right_steps.get(step_id)
    )
    same_bundle = left["bundle_semantic_sha256"] == right["bundle_semantic_sha256"]
    return {
        "schema_version": STUDY_COMPARISON_SCHEMA,
        "status": "equivalent" if same_bundle else "different",
        "same_bundle": same_bundle,
        "left_study_id": left["study_id"],
        "right_study_id": right["study_id"],
        "left_bundle_semantic_sha256": left["bundle_semantic_sha256"],
        "right_bundle_semantic_sha256": right["bundle_semantic_sha256"],
        "changed_records": record_differences,
        "changed_evidence_steps": changed_steps,
    }


__all__ = [
    "CAPABILITY_CONTRACTS",
    "MAX_STUDY_EVIDENCE_BYTES",
    "MAX_STUDY_STEPS",
    "STUDY_CLAIMS_SCHEMA",
    "STUDY_COMPARISON_SCHEMA",
    "STUDY_EVIDENCE_SCHEMA",
    "STUDY_PLAN_SCHEMA",
    "STUDY_RECEIPT_SCHEMA",
    "STUDY_REQUEST_SCHEMA",
    "STUDY_RUN_SCHEMA",
    "STUDY_VERIFICATION_SCHEMA",
    "StudyBundleArtifacts",
    "StudyClaims",
    "StudyLifecycleError",
    "StudyPlan",
    "StudyRequest",
    "build_study_bundle",
    "compare_study_bundles",
    "inspect_study_bundle",
    "replay_study_bundle",
    "verify_study_bundle",
]
