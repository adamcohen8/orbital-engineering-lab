"""Deterministic compilation of a proposed study plan into an MVP disposition."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .capabilities import CapabilityCatalog
from .contracts import (
    PLANNING_RESULT_SCHEMA_ID,
    StudyContractError,
    bind_study_plan,
    bind_study_request,
    canonical_sha256,
    classify_execution_route,
    require_valid_document,
)
from .edge_compiler import StudyEdgeCompilationError, verify_typed_operation_edges
from .manifest import build_capability_manifest
from .resource_accounting import (
    _OBSERVATION_COUNT_KEYS,
    _STATION_COUNT_KEYS,
    account_plan_resources,
)
from .schema_validation import validate_schema

_RESOURCE_METRIC_NAMES = (
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
    "observation_count",
    "station_count",
)


@dataclass(frozen=True, slots=True)
class PlanningResourcePolicy:
    max_operations: int = 32
    max_total_cases: int = 10_000
    max_wall_time_s: float = 3600.0
    max_cpu_time_s: float = 3600.0
    max_peak_memory_mb: float = 32_768.0
    max_storage_mb: float = 10_240.0
    max_artifact_mb: float = 2048.0
    max_parallel_workers: int = 32
    max_work_units: float = 1_000_000_000_000_000.0
    max_propagation_steps: int = 1_000_000_000
    max_estimator_evaluations: int = 1_000_000_000
    max_batch_evaluations: int = 100_000_000
    max_arc_duration_s: float = 31_536_000.0
    max_input_bytes: int = 17_179_869_184
    max_retained_output_bytes: int = 10_995_116_277_760
    # Observation/station host ceilings are intentionally unset until an
    # authoritative host policy declares them.  Plan envelopes and typed
    # operation bounds still enforce their own selected ceilings.
    max_observations: int | None = None
    max_stations: int | None = None


@dataclass(frozen=True, slots=True)
class PlanningFinding:
    code: str
    severity: str
    category: str
    message: str
    paths: tuple[str, ...] = ()
    observed: Mapping[str, Any] | None = None
    suggestion: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity,
            "category": self.category,
            "message": self.message,
            "paths": list(self.paths),
            "observed": dict(self.observed or {}),
            "suggestion": self.suggestion,
        }


def preflight_study_plan(
    request: Mapping[str, Any],
    proposed_plan: Mapping[str, Any] | None,
    *,
    catalog: CapabilityCatalog,
    normalized_configs: Mapping[str, Mapping[str, Any]] | None = None,
    config_receipts: Sequence[Mapping[str, Any]] = (),
    policy: PlanningResourcePolicy | None = None,
    allow_unbound_pro_planning: bool = False,
) -> dict[str, Any]:
    """Validate a model-proposed plan without executing OEL or authorizing a quote.

    ``allow_unbound_pro_planning`` is reserved for the public Hosted package
    validator. It allows a known public-safe Pro contract to be planned locally
    while leaving execution availability and authoritative validation to the
    hosted service. The default remains fail-closed for ordinary local study
    preflight.
    """

    policy = policy or PlanningResourcePolicy()
    bound_request = bind_study_request(request)
    capability_manifest = build_capability_manifest(catalog)
    findings: list[PlanningFinding] = _request_findings(bound_request)
    bound_plan: dict[str, Any] | None = None
    if proposed_plan is None:
        findings.append(
            PlanningFinding(
                "plan.missing",
                "error",
                "clarification",
                "The agent has not proposed a complete typed study plan.",
                ("$.plan",),
                suggestion="Clarify the material study choices and submit a complete StudyPlan.",
            )
        )
    else:
        try:
            bound_plan = bind_study_plan(
                proposed_plan,
                request_id=str(bound_request["request_id"]),
                request_sha256=str(bound_request["request_sha256"]),
                capability_manifest_sha256=str(capability_manifest["manifest_sha256"]),
                planner_version=str(capability_manifest["planner_version"]),
            )
        except StudyContractError as exc:
            findings.append(
                PlanningFinding(
                    "plan.contract_invalid",
                    "error",
                    "clarification",
                    str(exc),
                    ("$.plan",),
                    suggestion="Repair the plan contract; do not execute or infer missing fields.",
                )
            )

    if bound_plan is not None:
        findings.extend(
            _compile_plan(
                bound_request,
                bound_plan,
                catalog=catalog,
                normalized_configs=dict(normalized_configs or {}),
                config_receipts=config_receipts,
                policy=policy,
                allow_unbound_pro_planning=allow_unbound_pro_planning,
            )
        )

    status = _status_for(findings, plan_present=bound_plan is not None)
    execution_route = classify_execution_route(
        bound_plan,
        catalog=catalog,
        eligible=status == "PLAN_VALID",
        allow_unbound_pro_planning=allow_unbound_pro_planning,
    )
    hosted_pro = execution_route["route"] == "HOSTED_PRO_REQUIRED"
    resource_accounting = (
        account_plan_resources(
            bound_plan["operations"],
            bound_plan["resource_envelope"],
            countable_input_refs=_countable_input_refs(bound_plan["operations"], catalog),
        ).to_dict()
        if bound_plan is not None
        else _unassessed_resource_accounting()
    )
    result = {
        "schema": PLANNING_RESULT_SCHEMA_ID,
        "status": status,
        "request": bound_request,
        "capability_manifest_sha256": capability_manifest["manifest_sha256"],
        "planner_version": capability_manifest["planner_version"],
        "plan": bound_plan,
        "plan_sha256": None if bound_plan is None else bound_plan["plan_sha256"],
        "findings": [finding.to_dict() for finding in findings],
        "resource_accounting": resource_accounting,
        "config_receipts": [deepcopy(dict(item)) for item in config_receipts],
        "execution_route": execution_route,
        "commercial_status": {
            "currency": "USD",
            "oel_execution_amount": 0,
            "configuration_validation_amount": 0,
            "model_billing": "byo_provider_direct",
            "hosted_preflight_required": hosted_pro,
            "payment_authorized": False,
            "policy_status": "local_discovery_preflight_only",
        },
        "review_status": {
            "review_available": status == "PLAN_VALID",
            "user_review_required": status == "PLAN_VALID",
            "execution_authorized": False,
            "agent_may_authorize": False,
        },
        "non_claims": [
            "Preflight does not execute a simulation or analysis study.",
            "Preflight does not authorize payment, approval, or hosted execution.",
            "Schema validity and resource eligibility are not scientific qualification.",
            "The BYO model provider bills its usage directly to the user.",
            *(
                [
                    "Local Hosted package validation does not establish remote availability, entitlement, pricing, or authoritative Pro payload validity."
                ]
                if allow_unbound_pro_planning
                else []
            ),
        ],
    }
    return require_valid_document(result, expected_schema=PLANNING_RESULT_SCHEMA_ID)


def _unassessed_resource_accounting() -> dict[str, Any]:
    issue_code = "resource.plan_unassessed"
    return {
        "schema": "oel.study_resource_accounting.v1",
        "metrics": {
            name: {
                "status": "unassessed",
                "value": None,
                "basis": "No content-bound study plan was available for accounting.",
                "issue_codes": [issue_code],
            }
            for name in _RESOURCE_METRIC_NAMES
        },
        "worker_bound": None,
        "issue_codes": [issue_code],
    }


def _request_findings(request: Mapping[str, Any]) -> list[PlanningFinding]:
    findings: list[PlanningFinding] = []
    unanswered = [
        str(item["clarification_id"])
        for item in request["clarifications"]
        if item["material"] and item["answer"] is None
    ]
    proposed = [
        str(item["choice_id"])
        for item in request["required_choices"]
        if item["material"] and item["source"] == "agent_proposal"
    ]
    if unanswered:
        findings.append(
            PlanningFinding(
                "request.material_clarification_unanswered",
                "error",
                "clarification",
                "Material clarification questions remain unanswered.",
                ("$.request.clarifications",),
                {"clarification_ids": unanswered},
                "Ask the user and record their answers before compiling a terminal plan.",
            )
        )
    if proposed:
        findings.append(
            PlanningFinding(
                "request.material_choice_not_user_selected",
                "error",
                "clarification",
                "One or more material choices are still agent proposals rather than user selections.",
                ("$.request.required_choices",),
                {"choice_ids": proposed},
                "Present the choices to the user and record the selected value with source=user.",
            )
        )
    return findings


def _compile_plan(
    request: Mapping[str, Any],
    plan: Mapping[str, Any],
    *,
    catalog: CapabilityCatalog,
    normalized_configs: Mapping[str, Mapping[str, Any]],
    config_receipts: Sequence[Mapping[str, Any]],
    policy: PlanningResourcePolicy,
    allow_unbound_pro_planning: bool,
) -> list[PlanningFinding]:
    findings: list[PlanningFinding] = []
    if plan["question_answered"] != request["question"]:
        findings.append(
            PlanningFinding(
                "plan.question_mismatch",
                "blocker",
                "intent_parity",
                "The plan must answer the exact content-bound request question.",
                ("$.request.question", "$.plan.question_answered"),
                suggestion="Record any user-approved narrowing as a new request, then preflight again.",
            )
        )
    operations = list(plan["operations"])
    try:
        verify_typed_operation_edges(
            operations,
            request_inputs=request["inputs"],
            catalog=catalog,
        )
    except StudyEdgeCompilationError as exc:
        findings.extend(
            PlanningFinding(
                item.code,
                "blocker",
                "planner_error",
                item.message,
                item.paths,
                item.observed,
                item.suggestion,
            )
            for item in exc.findings
        )
    forbidden = set(map(str, plan["forbidden_capabilities"]))
    request_inputs = {str(item["input_id"]): str(item["kind"]) for item in request["inputs"]}
    operation_by_id = {str(item["operation_id"]): item for item in operations}
    output_producers: dict[str, tuple[str, str]] = {}
    duplicate_outputs: set[str] = set()
    for operation in operations:
        operation_id = str(operation["operation_id"])
        capability = catalog.get(str(operation["capability_id"]))
        for output_ref in operation["output_refs"]:
            ref_id = str(output_ref["ref_id"])
            kind = str(output_ref["kind"])
            if ref_id in output_producers:
                duplicate_outputs.add(ref_id)
            output_producers[ref_id] = (operation_id, kind)
            if capability is not None and kind not in capability.output_kinds:
                findings.append(
                    PlanningFinding(
                        "plan.output_kind_not_declared",
                        "blocker",
                        "planner_error",
                        f"Capability {capability.capability_id!r} does not produce output kind {kind!r}.",
                        ("$.plan.operations",),
                        {"operation_id": operation_id, "ref_id": ref_id, "kind": kind},
                        "Use an output kind declared by the selected capability.",
                    )
                )
    if duplicate_outputs:
        findings.append(
            PlanningFinding(
                "plan.duplicate_output_reference",
                "blocker",
                "intent_parity",
                "Operation output references must have one authoritative producer.",
                ("$.plan.operations",),
                {"output_refs": sorted(duplicate_outputs)},
            )
        )

    for index, operation in enumerate(operations):
        capability_id = str(operation["capability_id"])
        capability = catalog.get(capability_id)
        capability_is_plannable = capability is not None and (
            capability.availability == "available"
            or (allow_unbound_pro_planning and capability.edition == "pro")
        )
        if not capability_is_plannable:
            findings.append(
                PlanningFinding(
                    "capability.unavailable",
                    "blocker",
                    "unsupported_analysis",
                    f"The plan requires unavailable capability {capability_id!r}.",
                    (f"$.plan.operations[{index}].capability_id",),
                    {"capability_id": capability_id},
                    "Remove the operation or submit an opt-in capability request after reviewing the refusal.",
                )
            )
        elif (
            {"scenario_config", "validated_scenario"} & set(capability.input_kinds)
            and operation["config_ref"] is None
        ):
            findings.append(
                PlanningFinding(
                    "config.reference_required",
                    "blocker",
                    "missing_data",
                    f"Capability {capability_id!r} requires a validated scenario configuration.",
                    (f"$.plan.operations[{index}].config_ref",),
                    suggestion="Bind the operation to a config_ref and run free validate-only preflight.",
                )
            )
        elif (
            not ({"scenario_config", "validated_scenario"} & set(capability.input_kinds))
            and operation["config_ref"] is not None
        ):
            findings.append(
                PlanningFinding(
                    "config.reference_not_applicable",
                    "blocker",
                    "planner_error",
                    f"Capability {capability_id!r} does not consume a scenario configuration.",
                    (f"$.plan.operations[{index}].config_ref",),
                    {"config_ref": operation["config_ref"]},
                    "Set config_ref to null for typed problem or evidence inputs that are not scenario YAML.",
                )
            )
        if capability is not None:
            parameter_issues = validate_schema(operation["parameters"], capability.parameter_schema)
            if parameter_issues:
                findings.append(
                    PlanningFinding(
                        "capability.parameters_invalid",
                        "blocker",
                        "planner_error",
                        f"Operation parameters do not satisfy {capability_id!r}'s typed contract.",
                        (f"$.plan.operations[{index}].parameters",),
                        {"issues": [item.to_dict() for item in parameter_issues]},
                        "Revise the proposal using the parameter_schema published with this capability.",
                    )
                )
            declared_cases = _declared_cases(capability_id, operation["parameters"])
            if declared_cases is not None and declared_cases > int(operation["bounds"].get("max_cases", 1)):
                findings.append(
                    PlanningFinding(
                        "resource.operation_bound_understates_parameters",
                        "blocker",
                        "resource_limit",
                        f"Operation {operation['operation_id']!r} declares more cases than its resource bound permits.",
                        (
                            f"$.plan.operations[{index}].parameters",
                            f"$.plan.operations[{index}].bounds.max_cases",
                        ),
                        {
                            "declared_cases": declared_cases,
                            "bounded_cases": int(operation["bounds"].get("max_cases", 1)),
                        },
                        "Increase the visible bound or reduce the typed operation size.",
                    )
                )
        if capability_id in forbidden:
            findings.append(
                PlanningFinding(
                    "capability.forbidden_by_plan",
                    "blocker",
                    "intent_parity",
                    f"The operation uses capability {capability_id!r}, which the plan itself forbids.",
                    (f"$.plan.operations[{index}].capability_id", "$.plan.forbidden_capabilities"),
                )
            )
        dependencies = _dependency_closure(str(operation["operation_id"]), operation_by_id)
        available_inputs = dict(request_inputs)
        available_inputs.update(
            {
                ref_id: kind
                for ref_id, (producer, kind) in output_producers.items()
                if producer in dependencies
            }
        )
        missing_inputs: list[str] = []
        for input_ref in operation["input_refs"]:
            ref_id = str(input_ref["ref_id"])
            declared_kind = str(input_ref["kind"])
            source_kind = available_inputs.get(ref_id)
            if source_kind is None:
                missing_inputs.append(ref_id)
                continue
            if source_kind != declared_kind:
                findings.append(
                    PlanningFinding(
                        "plan.input_kind_mismatch",
                        "blocker",
                        "intent_parity",
                        f"Input {ref_id!r} is {source_kind!r}, not {declared_kind!r}.",
                        (f"$.plan.operations[{index}].input_refs",),
                        {"ref_id": ref_id, "declared_kind": declared_kind, "source_kind": source_kind},
                        "Use the exact kind produced by the request input or dependency operation.",
                    )
                )
            if capability is not None and declared_kind not in capability.input_kinds:
                findings.append(
                    PlanningFinding(
                        "plan.input_kind_not_accepted",
                        "blocker",
                        "planner_error",
                        f"Capability {capability_id!r} does not accept input kind {declared_kind!r}.",
                        (f"$.plan.operations[{index}].input_refs",),
                        {"ref_id": ref_id, "kind": declared_kind},
                        "Use an input kind declared by the selected capability.",
                    )
                )
        if missing_inputs:
            findings.append(
                PlanningFinding(
                    "plan.input_reference_unavailable",
                    "blocker",
                    "missing_data",
                    "An operation references inputs that are not supplied by the request or earlier operations.",
                    (f"$.plan.operations[{index}].input_refs",),
                    {"input_refs": missing_inputs},
                    "Add content-bound request inputs or a dependency operation that produces each reference.",
                )
            )

    required_inputs = {str(item["input_id"]) for item in request["inputs"] if item["required"]}
    transfers = {str(item["input_id"]): item for item in plan["transfers"]}
    missing_transfers = sorted(required_inputs - set(transfers))
    if missing_transfers:
        findings.append(
            PlanningFinding(
                "transfer.required_input_missing",
                "blocker",
                "missing_data",
                "The transfer manifest omits required request inputs.",
                ("$.plan.transfers",),
                {"input_ids": missing_transfers},
                "Add each required input to the user-visible transfer manifest.",
            )
        )
    for input_id, transfer in transfers.items():
        request_input = next((item for item in request["inputs"] if item["input_id"] == input_id), None)
        if request_input is None:
            findings.append(
                PlanningFinding(
                    "transfer.unknown_input",
                    "blocker",
                    "intent_parity",
                    f"Transfer {input_id!r} does not identify a request input.",
                    ("$.plan.transfers",),
                )
            )
        elif request_input["content_sha256"] != transfer["content_sha256"]:
            findings.append(
                PlanningFinding(
                    "transfer.digest_mismatch",
                    "blocker",
                    "intent_parity",
                    f"Transfer {input_id!r} does not preserve the request input digest.",
                    ("$.request.inputs", "$.plan.transfers"),
                )
            )
        elif bool(request_input["required"]) != bool(transfer["required"]):
            findings.append(
                PlanningFinding(
                    "transfer.required_flag_mismatch",
                    "blocker",
                    "intent_parity",
                    f"Transfer {input_id!r} changes whether the request input is required.",
                    ("$.request.inputs", "$.plan.transfers"),
                )
            )

    findings.extend(_config_findings(request, plan, normalized_configs, config_receipts))
    findings.extend(_evidence_findings(plan, operation_by_id))
    findings.extend(_orphan_operation_findings(plan, operation_by_id))
    findings.extend(
        _resource_findings(
            plan,
            operations,
            policy,
            countable_input_refs=_countable_input_refs(operations, catalog),
        )
    )
    return findings


def _config_findings(
    request: Mapping[str, Any],
    plan: Mapping[str, Any],
    normalized_configs: Mapping[str, Mapping[str, Any]],
    receipts: Sequence[Mapping[str, Any]],
) -> list[PlanningFinding]:
    findings: list[PlanningFinding] = []
    required_refs = {
        str(item["config_ref"])
        for item in plan["operations"]
        if item["config_ref"] is not None
        and any(
            ref["kind"] in {"scenario_config", "validated_scenario"}
            for ref in item["input_refs"]
        )
    }
    receipt_by_ref = {str(item.get("config_ref", "")): item for item in receipts}
    request_input_by_id = {str(item["input_id"]): item for item in request["inputs"]}
    for config_ref in sorted(required_refs):
        receipt = receipt_by_ref.get(config_ref)
        config = normalized_configs.get(config_ref)
        if receipt is None or config is None:
            findings.append(
                PlanningFinding(
                    "config.validation_receipt_missing",
                    "blocker",
                    "missing_data",
                    f"Plan configuration {config_ref!r} has no normalized free-validation receipt.",
                    ("$.plan.operations",),
                    {"config_ref": config_ref},
                    "Run free validate-only preflight for this configuration.",
                )
            )
            continue
        if not bool(receipt.get("valid")):
            findings.append(
                PlanningFinding(
                    "config.validation_failed",
                    "blocker",
                    "unsupported_analysis",
                    f"Plan configuration {config_ref!r} failed deterministic validation.",
                    ("$.config_receipts",),
                    {"config_ref": config_ref},
                    "Correct the configuration and validate it again.",
                )
            )
        if str(dict(receipt.get("resource_estimate", {}) or {}).get("action", "")) == "refuse":
            findings.append(
                PlanningFinding(
                    "config.resource_refused",
                    "blocker",
                    "resource_limit",
                    f"Plan configuration {config_ref!r} exceeds its validation resource policy.",
                    ("$.config_receipts",),
                    {"config_ref": config_ref},
                    "Reduce or split the study; validation cannot silently relax the resource policy.",
                )
            )
        try:
            normalized_digest = canonical_sha256(config)
        except (StudyContractError, TypeError, ValueError) as exc:
            normalized_digest = None
            findings.append(
                PlanningFinding(
                    "config.normalized_document_invalid",
                    "blocker",
                    "planner_error",
                    f"Plan configuration {config_ref!r} is not a canonical normalized document: {exc}",
                    ("$.config_receipts",),
                    {"config_ref": config_ref},
                    "Use the exact normalized document returned by safe configuration validation.",
                )
            )
        if normalized_digest != receipt.get("normalized_config_sha256"):
            findings.append(
                PlanningFinding(
                    "config.normalized_digest_mismatch",
                    "blocker",
                    "intent_parity",
                    f"Normalized configuration {config_ref!r} does not match its validation receipt.",
                    ("$.config_receipts",),
                    {
                        "config_ref": config_ref,
                        "expected": receipt.get("normalized_config_sha256"),
                        "observed": normalized_digest,
                    },
                    "Use the exact normalized document returned by the matching safe validation receipt.",
                )
            )
        bound_input_ids = {
            str(input_ref["ref_id"])
            for operation in plan["operations"]
            if operation["config_ref"] == config_ref
            for input_ref in operation["input_refs"]
            if input_ref["kind"] == "scenario_config"
            and str(input_ref["ref_id"]) in request_input_by_id
        }
        if not bound_input_ids:
            findings.append(
                PlanningFinding(
                    "config.request_input_not_bound",
                    "blocker",
                    "intent_parity",
                    f"Configuration {config_ref!r} is not bound to a scenario_config request input.",
                    ("$.request.inputs", "$.plan.operations"),
                    suggestion="Reference the user-granted scenario input from each operation using this config.",
                )
            )
        for input_id in sorted(bound_input_ids):
            requested_digest = request_input_by_id[input_id]["content_sha256"]
            if requested_digest is None or requested_digest != receipt.get("source_sha256"):
                findings.append(
                    PlanningFinding(
                        "config.source_digest_mismatch",
                        "blocker",
                        "intent_parity",
                        f"Validated configuration {config_ref!r} does not match request input {input_id!r}.",
                        ("$.request.inputs", "$.config_receipts"),
                        {
                            "input_id": input_id,
                            "request_sha256": requested_digest,
                            "validated_source_sha256": receipt.get("source_sha256"),
                        },
                        "Refresh the request input digest or validate the exact user-selected file.",
                    )
                )

    for index, constraint in enumerate(plan["config_constraints"]):
        config_ref = str(constraint["config_ref"])
        config = normalized_configs.get(config_ref)
        if config is None:
            if config_ref not in required_refs:
                findings.append(
                    PlanningFinding(
                        "config.constraint_target_missing",
                        "blocker",
                        "intent_parity",
                        f"Constraint references unknown configuration {config_ref!r}.",
                        (f"$.plan.config_constraints[{index}].config_ref",),
                    )
                )
            continue
        present, actual = _json_pointer(config, str(constraint["path"]))
        operator = str(constraint["operator"])
        expected = constraint["expected"]
        matched = (
            (operator == "present" and present)
            or (operator == "absent" and not present)
            or (operator == "equals" and present and actual == expected)
            or (operator == "one_of" and present and isinstance(expected, list) and actual in expected)
        )
        if not matched:
            severity = "blocker" if constraint["material"] else "warning"
            findings.append(
                PlanningFinding(
                    "config.intent_constraint_mismatch",
                    severity,
                    "intent_parity",
                    f"Normalized configuration {config_ref!r} does not satisfy an approved plan constraint.",
                    (f"$.plan.config_constraints[{index}]",),
                    {"present": present, "actual": actual, "expected": expected, "operator": operator},
                    "Revise the plan or configuration and obtain a new content-bound plan digest.",
                )
            )
    return findings


def _evidence_findings(
    plan: Mapping[str, Any],
    operation_by_id: Mapping[str, Mapping[str, Any]],
) -> list[PlanningFinding]:
    findings: list[PlanningFinding] = []
    for index, evidence in enumerate(plan["evidence_requirements"]):
        producible: set[str] = set()
        for operation_id in evidence["source_operation_ids"]:
            operation = operation_by_id.get(str(operation_id))
            if operation is not None:
                producible.update(str(item["kind"]) for item in operation["output_refs"])
        if str(evidence["kind"]) not in producible:
            findings.append(
                PlanningFinding(
                    "evidence.not_producible",
                    "blocker",
                    "insufficient_validation",
                    f"The selected operations do not declare output kind {evidence['kind']!r}.",
                    (f"$.plan.evidence_requirements[{index}]",),
                    {"producible_kinds": sorted(producible)},
                    "Select a capability that explicitly produces the required evidence or narrow the claim.",
                )
            )
    return findings


def _orphan_operation_findings(
    plan: Mapping[str, Any],
    operation_by_id: Mapping[str, Mapping[str, Any]],
) -> list[PlanningFinding]:
    """Reject work that cannot contribute to a user-visible promised result."""

    promised_evidence_ids = {
        str(evidence_id)
        for item in (*plan["acceptance_criteria"], *plan["claims"])
        for evidence_id in item["evidence_ids"]
    }
    terminal_operation_ids = {
        str(operation_id)
        for evidence in plan["evidence_requirements"]
        if str(evidence["evidence_id"]) in promised_evidence_ids
        for operation_id in evidence["source_operation_ids"]
    }
    relevant_operation_ids = set(terminal_operation_ids)
    for operation_id in terminal_operation_ids:
        if operation_id in operation_by_id:
            relevant_operation_ids.update(_dependency_closure(operation_id, operation_by_id))
    orphan_ids = sorted(set(operation_by_id) - relevant_operation_ids)
    if not orphan_ids:
        return []
    return [
        PlanningFinding(
            "plan.operation_not_terminally_relevant",
            "blocker",
            "resource_limit",
            "Every operation must contribute through dependencies to evidence promised by an acceptance criterion or claim.",
            ("$.plan.operations", "$.plan.acceptance_criteria", "$.plan.claims"),
            {"operation_ids": orphan_ids},
            "Remove the orphan operations or bind their evidence to an explicit user-visible promise.",
        )
    ]


def _countable_input_refs(
    operations: Sequence[Mapping[str, Any]],
    catalog: CapabilityCatalog,
) -> dict[str, set[tuple[int, int]]]:
    """Classify count metrics from authoritative typed input-port metadata.

    A typed problem/configuration port that does not require an observation or
    station count is not itself a dataset for that metric.  Count metadata on
    such a port still opts it into accounting.  Unknown or ambiguous ports are
    retained for both metrics so missing applicability cannot make a plan look
    assessed.
    """

    countable: dict[str, set[tuple[int, int]]] = {
        "observation_count": set(),
        "station_count": set(),
    }
    metric_keys = {
        "observation_count": _OBSERVATION_COUNT_KEYS,
        "station_count": _STATION_COUNT_KEYS,
    }
    for operation_index, operation in enumerate(operations):
        capability = catalog.get(str(operation.get("capability_id", "")))
        for ref_index, ref in enumerate(list(operation.get("input_refs", []) or [])):
            metadata = ref.get("metadata", {})
            metadata = dict(metadata) if isinstance(metadata, Mapping) else {}
            matching_ports = []
            if capability is not None:
                for port in capability.compiled_input_ports:
                    if ref.get("port_id") is not None and str(ref.get("port_id")) != port.port_id:
                        continue
                    if str(ref.get("kind")) != port.artifact_kind:
                        continue
                    if str(ref.get("schema_id")) not in port.schema_ids:
                        continue
                    matching_ports.append(port)

            # A missing or ambiguous authoritative port is an unclassified
            # reference. Keep it countable so the accounting remains fail-closed.
            if len(matching_ports) != 1:
                for metric_name in metric_keys:
                    countable[metric_name].add((operation_index, ref_index))
                continue

            required_metadata = set(matching_ports[0].required_metadata)
            for metric_name, accepted_keys in metric_keys.items():
                if any(key in required_metadata or key in metadata for key in accepted_keys):
                    countable[metric_name].add((operation_index, ref_index))
    return countable


def _resource_findings(
    plan: Mapping[str, Any],
    operations: Sequence[Mapping[str, Any]],
    policy: PlanningResourcePolicy,
    *,
    countable_input_refs: Mapping[str, set[tuple[int, int]]] | None = None,
) -> list[PlanningFinding]:
    envelope = plan["resource_envelope"]
    accounting = account_plan_resources(
        operations,
        envelope,
        countable_input_refs=countable_input_refs,
    )
    findings = [
        PlanningFinding(
            issue.code,
            "blocker",
            "resource_limit",
            issue.message,
            issue.paths,
            issue.observed,
            issue.suggestion,
        )
        for issue in accounting.issues
    ]
    total_cases = accounting.total_cases
    wall_time = sum(float(item["bounds"].get("max_wall_time_s", 0.0)) for item in operations)
    limits = {
        "operations": (len(operations), min(int(envelope["max_operations"]), policy.max_operations)),
        "total_cases": (total_cases, min(int(envelope["max_total_cases"]), policy.max_total_cases)),
        "wall_time_s": (wall_time, min(float(envelope["max_wall_time_s"]), policy.max_wall_time_s)),
        "cpu_time_s": (float(envelope["max_cpu_time_s"]), policy.max_cpu_time_s),
        "work_units": (
            float(envelope["max_work_units"]),
            policy.max_work_units,
        ) if "max_work_units" in envelope else None,
        "propagation_steps": (
            int(envelope["max_propagation_steps"]),
            policy.max_propagation_steps,
        ) if "max_propagation_steps" in envelope else None,
        "estimator_evaluations": (
            int(envelope["max_estimator_evaluations"]),
            policy.max_estimator_evaluations,
        ) if "max_estimator_evaluations" in envelope else None,
        "batch_evaluations": (
            int(envelope["max_batch_evaluations"]),
            policy.max_batch_evaluations,
        ) if "max_batch_evaluations" in envelope else None,
        "arc_duration_s": (
            float(envelope["max_arc_duration_s"]),
            policy.max_arc_duration_s,
        ) if "max_arc_duration_s" in envelope else None,
        "peak_memory_mb": (float(envelope["max_peak_memory_mb"]), policy.max_peak_memory_mb),
        "storage_mb": (float(envelope["max_storage_mb"]), policy.max_storage_mb),
        "artifact_mb": (float(envelope["max_artifact_mb"]), policy.max_artifact_mb),
        "parallel_workers": (int(envelope["max_parallel_workers"]), policy.max_parallel_workers),
        "input_bytes": (
            int(envelope["max_input_bytes"]),
            policy.max_input_bytes,
        ) if "max_input_bytes" in envelope else None,
    }
    if "max_observations" in envelope and policy.max_observations is not None:
        limits["observation_count"] = (
            int(envelope["max_observations"]),
            min(int(envelope["max_observations"]), policy.max_observations),
        )
    if "max_stations" in envelope and policy.max_stations is not None:
        limits["station_count"] = (
            int(envelope["max_stations"]),
            min(int(envelope["max_stations"]), policy.max_stations),
        )
    if "max_retained_output_bytes" in envelope:
        limits["retained_output_bytes_cap"] = (
            float(envelope["max_retained_output_bytes"]),
            min(
                float(policy.max_retained_output_bytes),
                float(policy.max_storage_mb) * 1024.0 * 1024.0,
            ),
        )
    for name, limit in limits.items():
        if limit is None:
            continue
        observed, maximum = limit
        if observed > maximum:
            findings.append(
                PlanningFinding(
                    f"resource.{name}_exceeded",
                    "blocker",
                    "resource_limit",
                    f"The plan declares {name}={observed}, above the local planning resource limit {maximum}.",
                    ("$.plan.resource_envelope", "$.plan.operations"),
                    {"observed": observed, "maximum": maximum},
                    "Reduce, bound, or split the study before seeking approval.",
                )
            )

    def add_accounted_finding(
        metric_name: str,
        observed: float | int,
        maximum: float | int,
        suffix: str,
        *,
        label: str,
    ) -> None:
        if observed > maximum:
            findings.append(
                PlanningFinding(
                    f"resource.{metric_name}_{suffix}",
                    "blocker",
                    "resource_limit",
                    f"The accounted {label}={observed} exceeds the applicable limit {maximum}.",
                    ("$.plan.resource_envelope", "$.plan.operations"),
                    {"observed": observed, "maximum": maximum, "metric": metric_name},
                    "Reduce, bound, or split the study before seeking approval.",
                )
            )

    def check_accounted_metric(
        metric_name: str,
        observed: float | int | None,
        envelope_limit: float | int | None,
        policy_limit: float | int | None,
        *,
        label: str,
    ) -> None:
        if observed is None:
            return
        if envelope_limit is not None:
            add_accounted_finding(
                metric_name,
                observed,
                envelope_limit,
                "accounted_exceeded",
                label=label,
            )
        if policy_limit is not None and observed > policy_limit and envelope_limit != policy_limit:
            add_accounted_finding(
                metric_name,
                observed,
                policy_limit,
                "policy_exceeded",
                label=label,
            )

    check_accounted_metric(
        "cpu_time_s",
        accounting.cpu_time_s,
        float(envelope["max_cpu_time_s"]),
        policy.max_cpu_time_s,
        label="CPU time",
    )
    check_accounted_metric(
        "work_units",
        accounting.work_units,
        float(envelope["max_work_units"]) if "max_work_units" in envelope else None,
        policy.max_work_units,
        label="work units",
    )
    check_accounted_metric(
        "propagation_steps",
        accounting.propagation_steps,
        int(envelope["max_propagation_steps"]) if "max_propagation_steps" in envelope else None,
        policy.max_propagation_steps,
        label="propagation steps",
    )
    check_accounted_metric(
        "estimator_evaluations",
        accounting.estimator_evaluations,
        int(envelope["max_estimator_evaluations"])
        if "max_estimator_evaluations" in envelope
        else None,
        policy.max_estimator_evaluations,
        label="estimator evaluations",
    )
    check_accounted_metric(
        "batch_evaluations",
        accounting.batch_evaluations,
        int(envelope["max_batch_evaluations"]) if "max_batch_evaluations" in envelope else None,
        policy.max_batch_evaluations,
        label="batch evaluations",
    )
    check_accounted_metric(
        "arc_duration_s",
        accounting.arc_duration_s,
        float(envelope["max_arc_duration_s"]) if "max_arc_duration_s" in envelope else None,
        policy.max_arc_duration_s,
        label="per-operation arc duration",
    )
    check_accounted_metric(
        "peak_memory_mb",
        accounting.peak_memory_mb,
        float(envelope["max_peak_memory_mb"]),
        policy.max_peak_memory_mb,
        label="peak memory",
    )
    check_accounted_metric(
        "unique_input_bytes",
        accounting.unique_input_bytes,
        int(envelope["max_input_bytes"]) if "max_input_bytes" in envelope else None,
        policy.max_input_bytes,
        label="unique input bytes",
    )
    check_accounted_metric(
        "observation_count",
        accounting.observation_count,
        int(envelope["max_observations"]) if "max_observations" in envelope else None,
        policy.max_observations,
        label="observations",
    )
    check_accounted_metric(
        "station_count",
        accounting.station_count,
        int(envelope["max_stations"]) if "max_stations" in envelope else None,
        policy.max_stations,
        label="stations",
    )
    retained_envelope_limit = min(
        float(envelope["max_storage_mb"]) * 1024.0 * 1024.0,
        float(envelope["max_retained_output_bytes"])
        if "max_retained_output_bytes" in envelope
        else float("inf"),
    )
    retained_policy_limit = min(
        float(policy.max_storage_mb) * 1024.0 * 1024.0,
        float(policy.max_retained_output_bytes),
    )
    check_accounted_metric(
        "retained_output_bytes",
        accounting.retained_output_bytes,
        retained_envelope_limit,
        retained_policy_limit,
        label="retained output bytes",
    )
    return findings


def _declared_cases(capability_id: str, parameters: Mapping[str, Any]) -> int | None:
    parameter_name = {
        "oel.pro.campaign.monte_carlo.v1": "samples",
        "oel.pro.campaign.sensitivity.v1": "cases",
        "oel.pro.scale.screening.v1": "maximum_candidates",
    }.get(capability_id)
    if parameter_name is not None and parameter_name in parameters:
        return int(parameters[parameter_name])
    if capability_id == "oel.pro.controller.benchmark.v1":
        return len(list(parameters.get("controller_variants", []) or []))
    return None


def _json_pointer(document: Mapping[str, Any], pointer: str) -> tuple[bool, Any]:
    if pointer == "":
        return True, document
    if not pointer.startswith("/"):
        return False, None
    current: Any = document
    for raw in pointer[1:].split("/"):
        token = raw.replace("~1", "/").replace("~0", "~")
        if isinstance(current, Mapping) and token in current:
            current = current[token]
        elif isinstance(current, list) and token.isdigit() and int(token) < len(current):
            current = current[int(token)]
        else:
            return False, None
    return True, current


def _dependency_closure(operation_id: str, operation_by_id: Mapping[str, Mapping[str, Any]]) -> set[str]:
    found: set[str] = set()
    pending = list(map(str, operation_by_id[operation_id]["depends_on"]))
    while pending:
        dependency = pending.pop()
        if dependency in found:
            continue
        found.add(dependency)
        operation = operation_by_id.get(dependency)
        if operation is not None:
            pending.extend(map(str, operation["depends_on"]))
    return found


def _status_for(findings: Sequence[PlanningFinding], *, plan_present: bool) -> str:
    if any(item.severity == "blocker" for item in findings):
        return "UNSUPPORTED"
    if not plan_present or any(item.category == "clarification" and item.severity in {"error", "blocker"} for item in findings):
        return "CLARIFICATION_REQUIRED"
    return "PLAN_VALID"


__all__ = ["PlanningFinding", "PlanningResourcePolicy", "preflight_study_plan"]
