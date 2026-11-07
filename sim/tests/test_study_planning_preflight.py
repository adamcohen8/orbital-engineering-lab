from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from sim.study_planning import (
    StudyContractError,
    compile_typed_operation_edges,
    discovery_capability_catalog,
    preflight_study_plan,
)
from sim.study_planning.config_validation import validate_config_path
from sim.study_planning.edge_compiler import StudyEdgeCompilationError, verify_typed_operation_edges
from sim.study_planning.feedback import prepare_capability_request

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = Path(__file__).parent / "fixtures" / "study_planning"


def _load(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def _ready_result() -> dict:
    receipt, normalized = validate_config_path(
        "scenario_main", ROOT / "configs" / "automation_smoke.yaml", workspace_root=ROOT
    )
    assert normalized is not None
    return preflight_study_plan(
        _load("passive_request.json"),
        _load("passive_plan.json"),
        catalog=discovery_capability_catalog(),
        normalized_configs={"scenario_main": normalized},
        config_receipts=[receipt],
    )


def test_valid_preflight_is_free_nonexecuting_and_requires_user_review() -> None:
    result = _ready_result()
    assert result["status"] == "PLAN_VALID"
    assert result["execution_route"]["route"] == "LOCAL_FREE_AVAILABLE"
    assert result["execution_route"]["public_capability_ids"] == [
        "oel.config.validate.v1",
        "oel.scenario.execute.v1",
    ]
    assert result["execution_route"]["pro_capability_ids"] == []
    assert result["commercial_status"]["oel_execution_amount"] == 0
    assert result["commercial_status"]["payment_authorized"] is False
    assert result["review_status"] == {
        "review_available": True,
        "user_review_required": True,
        "execution_authorized": False,
        "agent_may_authorize": False,
    }
    assert result["plan"]["request_sha256"] == result["request"]["request_sha256"]
    assert result["plan"]["capability_manifest_sha256"] == result["capability_manifest_sha256"]
    assert result["config_receipts"][0]["execution_advanced"] is False
    assert result["config_receipts"][0]["charge_created"] is False
    assert result["config_receipts"][0]["plugins_imported"] is False
    assert result["resource_accounting"]["metrics"]["total_cases"]["status"] == "assessed"
    assert result["resource_accounting"]["metrics"]["cpu_time_s"]["status"] == "unassessed"


@pytest.mark.parametrize(
    ("field", "value", "finding_code"),
    [
        ("port_id", "not-a-real-port", "edge.port_unknown"),
        ("schema_id", "oel.artifact.wrong.v1", "edge.port_contract_mismatch"),
    ],
)
def test_authoritative_preflight_rejects_typed_port_drift(
    field: str,
    value: str,
    finding_code: str,
) -> None:
    receipt, normalized = validate_config_path(
        "scenario_main", ROOT / "configs" / "automation_smoke.yaml", workspace_root=ROOT
    )
    assert normalized is not None
    plan = _load("passive_plan.json")
    plan["operations"][1]["input_refs"][0][field] = value
    result = preflight_study_plan(
        _load("passive_request.json"),
        plan,
        catalog=discovery_capability_catalog(),
        normalized_configs={"scenario_main": normalized},
        config_receipts=[receipt],
    )
    assert result["status"] != "PLAN_VALID"
    assert finding_code in {item["code"] for item in result["findings"]}


def test_non_scenario_operation_rejects_config_reference_without_requesting_scenario_validation() -> None:
    digest = "2" * 64
    request = {
        "question": "Inspect this exact completed run.",
        "inputs": [
            {
                "input_id": "completed_run_input",
                "kind": "completed_run",
                "schema_id": "oel.artifact.completed_run.v1",
                "metadata": {"content_sha256": digest},
                "content_sha256": digest,
                "description": "Exact completed run.",
                "required": True,
            }
        ],
        "required_choices": [],
        "clarifications": [],
        "handling": {"marking": "TEST", "release_scope": "local_only", "owner": "OEL"},
    }
    operations = compile_typed_operation_edges(
        [
            {
                "operation_id": "inspect",
                "capability_id": "oel.run.inspect.v1",
                "depends_on": [],
                "config_ref": "not_a_scenario",
                "parameters": {},
                "bounds": {"max_cases": 1, "max_wall_time_s": 30.0},
            }
        ],
        request_inputs=request["inputs"],
        catalog=discovery_capability_catalog(),
    )["operations"]
    plan = {
        "question_answered": request["question"],
        "operations": operations,
        "config_constraints": [],
        "assumptions": [],
        "allowed_defaults": [],
        "forbidden_capabilities": [],
        "evidence_requirements": [
            {
                "evidence_id": "inspection",
                "kind": "run_inspection",
                "description": "Inspection evidence.",
                "source_operation_ids": ["inspect"],
            }
        ],
        "acceptance_criteria": [
            {
                "criterion_id": "inspection_returned",
                "description": "Return the inspection.",
                "evidence_ids": ["inspection"],
            }
        ],
        "claims": [],
        "non_claims": [],
        "transfers": [
            {
                "input_id": "completed_run_input",
                "purpose": "Inspect it.",
                "required": True,
                "content_sha256": digest,
            }
        ],
        "resource_envelope": {
            "resource_profile": "standard",
            "max_operations": 1,
            "max_total_cases": 1,
            "max_wall_time_s": 30.0,
            "max_cpu_time_s": 30.0,
            "max_peak_memory_mb": 512.0,
            "max_storage_mb": 16.0,
            "max_artifact_mb": 8.0,
            "max_parallel_workers": 1,
        },
    }
    result = preflight_study_plan(request, plan, catalog=discovery_capability_catalog())
    codes = {item["code"] for item in result["findings"]}
    assert "config.reference_not_applicable" in codes
    assert "config.validation_receipt_missing" not in codes


def test_request_content_digest_overrides_untrusted_metadata_digest() -> None:
    digest = "2" * 64
    request_input = {
        "input_id": "completed_run_input",
        "kind": "completed_run",
        "content_sha256": digest,
        "description": "Exact completed run.",
        "required": True,
    }
    operations = compile_typed_operation_edges(
        [
            {
                "operation_id": "inspect",
                "capability_id": "oel.run.inspect.v1",
                "depends_on": [],
                "config_ref": None,
                "parameters": {},
                "bounds": {"max_cases": 1, "max_wall_time_s": 30.0},
            }
        ],
        request_inputs=[request_input],
        catalog=discovery_capability_catalog(),
    )["operations"]
    operations[0]["input_refs"][0]["metadata"]["content_sha256"] = "f" * 64

    with pytest.raises(StudyEdgeCompilationError, match="edge.input_source_identity_mismatch"):
        verify_typed_operation_edges(
            operations,
            request_inputs=[request_input],
            catalog=discovery_capability_catalog(),
        )


def test_validated_scenario_consumer_cannot_skip_config_receipt_binding() -> None:
    receipt, normalized = validate_config_path(
        "scenario_main", ROOT / "configs" / "automation_smoke.yaml", workspace_root=ROOT
    )
    assert normalized is not None
    plan = _load("passive_plan.json")
    plan["operations"][1]["config_ref"] = "unvalidated_config"
    result = preflight_study_plan(
        _load("passive_request.json"),
        plan,
        catalog=discovery_capability_catalog(),
        normalized_configs={"scenario_main": normalized},
        config_receipts=[receipt],
    )
    codes = {item["code"] for item in result["findings"]}
    assert "config.validation_receipt_missing" in codes


def test_normalized_config_must_match_its_validation_receipt() -> None:
    receipt, normalized = validate_config_path(
        "scenario_main", ROOT / "configs" / "automation_smoke.yaml", workspace_root=ROOT
    )
    assert normalized is not None
    tampered = deepcopy(normalized)
    tampered["scenario_name"] = "tampered_after_validation"
    result = preflight_study_plan(
        _load("passive_request.json"),
        _load("passive_plan.json"),
        catalog=discovery_capability_catalog(),
        normalized_configs={"scenario_main": tampered},
        config_receipts=[receipt],
    )
    assert "config.normalized_digest_mismatch" in {item["code"] for item in result["findings"]}


@pytest.mark.parametrize("resource_value", ["unknown", True, -1, 10**1000])
def test_invalid_resource_metadata_is_reported_as_a_planning_finding(resource_value) -> None:
    receipt, normalized = validate_config_path(
        "scenario_main", ROOT / "configs" / "automation_smoke.yaml", workspace_root=ROOT
    )
    assert normalized is not None
    plan = _load("passive_plan.json")
    plan["operations"][1]["input_refs"][0]["metadata"]["input_bytes"] = resource_value
    result = preflight_study_plan(
        _load("passive_request.json"),
        plan,
        catalog=discovery_capability_catalog(),
        normalized_configs={"scenario_main": normalized},
        config_receipts=[receipt],
    )
    assert "edge.resource_metadata_invalid" in {item["code"] for item in result["findings"]}


def test_preflight_rejects_an_operation_that_does_not_contribute_to_promised_evidence() -> None:
    receipt, normalized = validate_config_path(
        "scenario_main", ROOT / "configs" / "automation_smoke.yaml", workspace_root=ROOT
    )
    assert normalized is not None
    plan = _load("passive_plan.json")
    orphan = deepcopy(plan["operations"][1])
    orphan.update(
        {
            "operation_id": "unpromised_second_run",
            "depends_on": ["validate"],
            "output_refs": [
                {"ref_id": "unpromised_completed_run", "kind": "completed_run"},
                {"ref_id": "unpromised_review_store", "kind": "review_store"},
                {"ref_id": "unpromised_artifact_manifest", "kind": "artifact_manifest"},
            ],
        }
    )
    plan["operations"].append(orphan)
    result = preflight_study_plan(
        _load("passive_request.json"),
        plan,
        catalog=discovery_capability_catalog(),
        normalized_configs={"scenario_main": normalized},
        config_receipts=[receipt],
    )
    finding = next(
        item for item in result["findings"]
        if item["code"] == "plan.operation_not_terminally_relevant"
    )
    assert result["status"] == "UNSUPPORTED"
    assert finding["observed"]["operation_ids"] == ["unpromised_second_run"]


def test_material_unanswered_question_returns_clarification_state() -> None:
    result = preflight_study_plan(
        _load("clarification_request.json"),
        None,
        catalog=discovery_capability_catalog(),
    )
    assert result["status"] == "CLARIFICATION_REQUIRED"
    assert result["execution_route"]["route"] == "NOT_ELIGIBLE"
    assert result["commercial_status"]["oel_execution_amount"] == 0
    assert result["review_status"]["review_available"] is False


def test_unknown_capability_and_config_drift_are_honest_refusals() -> None:
    receipt, normalized = validate_config_path(
        "scenario_main", ROOT / "configs" / "automation_smoke.yaml", workspace_root=ROOT
    )
    assert normalized is not None
    unknown = _load("passive_plan.json")
    unknown["operations"][1]["capability_id"] = "oel.pro.magic.unbounded.v1"
    result = preflight_study_plan(
        _load("passive_request.json"),
        unknown,
        catalog=discovery_capability_catalog(),
        normalized_configs={"scenario_main": normalized},
        config_receipts=[receipt],
    )
    assert result["status"] == "UNSUPPORTED"
    assert "capability.unavailable" in {item["code"] for item in result["findings"]}
    assert result["execution_route"]["route"] == "NOT_ELIGIBLE"
    assert result["commercial_status"]["oel_execution_amount"] == 0
    assert result["commercial_status"]["payment_authorized"] is False

    drifted = deepcopy(normalized)
    drifted["scenario_name"] = "different_study"
    result = preflight_study_plan(
        _load("passive_request.json"),
        _load("passive_plan.json"),
        catalog=discovery_capability_catalog(),
        normalized_configs={"scenario_main": drifted},
        config_receipts=[receipt],
    )
    assert result["status"] == "UNSUPPORTED"
    assert "config.intent_constraint_mismatch" in {item["code"] for item in result["findings"]}
    assert result["execution_route"]["route"] == "NOT_ELIGIBLE"
    assert result["commercial_status"]["oel_execution_amount"] == 0


def test_refusal_feedback_is_a_local_sanitized_preview() -> None:
    receipt, normalized = validate_config_path(
        "scenario_main", ROOT / "configs" / "automation_smoke.yaml", workspace_root=ROOT
    )
    assert normalized is not None
    plan = _load("passive_plan.json")
    plan["operations"][1]["capability_id"] = "oel.pro.magic.unbounded.v1"
    result = preflight_study_plan(
        _load("passive_request.json"),
        plan,
        catalog=discovery_capability_catalog(),
        normalized_configs={"scenario_main": normalized},
        config_receipts=[receipt],
    )
    preview = prepare_capability_request(
        result,
        user_summary="I would like OEL to support this bounded analysis.",
        engine_version="test",
    )
    assert preview["versions"]["planner"] == result["planner_version"]
    assert preview["versions"]["capability_manifest"] == result["capability_manifest_sha256"]
    assert preview["submission_authorized"] is False
    assert preview["attachments"] == []
    assert "question" not in preview
    assert "plan" not in preview
    assert "transcript" not in preview


def test_unbound_monte_carlo_descriptor_is_not_advertised_as_executable() -> None:
    receipt, normalized = validate_config_path(
        "scenario_main", ROOT / "configs" / "automation_smoke.yaml", workspace_root=ROOT
    )
    assert normalized is not None
    plan = _load("passive_plan.json")
    operation = plan["operations"][1]
    operation.update(
        {
            "operation_id": "uncertainty_campaign",
            "capability_id": "oel.pro.campaign.monte_carlo.v1",
            "input_refs": [{"ref_id": "validated_scenario", "kind": "validated_scenario"}],
            "output_refs": [{"ref_id": "campaign_evidence", "kind": "campaign_evidence"}],
            "parameters": {
                "samples": 100,
                "seed": 42,
                "distributions": [{"path": "/metadata/test_uncertainty", "kind": "normal"}],
            },
            "bounds": {"max_cases": 100, "max_wall_time_s": 600.0},
        }
    )
    plan["evidence_requirements"][0].update(
        {
            "kind": "campaign_evidence",
            "source_operation_ids": ["uncertainty_campaign"],
        }
    )
    plan["forbidden_capabilities"] = []
    plan["resource_envelope"].update({"max_total_cases": 200, "max_wall_time_s": 900.0})
    result = preflight_study_plan(
        _load("passive_request.json"),
        plan,
        catalog=discovery_capability_catalog(),
        normalized_configs={"scenario_main": normalized},
        config_receipts=[receipt],
    )
    assert result["status"] == "UNSUPPORTED"
    assert result["execution_route"]["route"] == "NOT_ELIGIBLE"
    assert result["execution_route"]["public_capability_ids"] == ["oel.config.validate.v1"]
    assert result["execution_route"]["pro_capability_ids"] == []
    assert result["execution_route"]["unavailable_capability_ids"] == [
        "oel.pro.campaign.monte_carlo.v1"
    ]
    assert result["execution_route"]["hosted_preflight_required"] is False
    assert result["commercial_status"]["oel_execution_amount"] == 0
    assert result["commercial_status"]["payment_authorized"] is False
    assert "capability.unavailable" in {item["code"] for item in result["findings"]}
    assert [item["capability_id"] for item in result["plan"]["operations"]] == [
        "oel.config.validate.v1",
        "oel.pro.campaign.monte_carlo.v1",
    ]


def test_completed_run_inspection_is_a_distinct_ready_operation_graph() -> None:
    digest = "1" * 64
    request = {
        "question": "Inspect the selected completed run and summarize its recorded run evidence.",
        "inputs": [
            {
                "input_id": "completed_run_input",
                "kind": "completed_run",
                "content_sha256": digest,
                "description": "User-selected completed run manifest.",
                "required": True,
            }
        ],
        "required_choices": [],
        "clarifications": [],
        "handling": {"marking": "TEST", "release_scope": "public", "owner": "OEL"},
    }
    plan = {
        "question_answered": request["question"],
        "operations": [
            {
                "operation_id": "inspect",
                "capability_id": "oel.run.inspect.v1",
                "depends_on": [],
                "input_refs": [{"ref_id": "completed_run_input", "kind": "completed_run"}],
                "output_refs": [{"ref_id": "run_inspection", "kind": "run_inspection"}],
                "config_ref": None,
                "parameters": {},
                "bounds": {"max_cases": 1, "max_wall_time_s": 30.0},
            }
        ],
        "config_constraints": [],
        "assumptions": [],
        "allowed_defaults": [],
        "forbidden_capabilities": [],
        "evidence_requirements": [
            {
                "evidence_id": "inspection_evidence",
                "kind": "run_inspection",
                "description": "Recorded run status and provenance.",
                "source_operation_ids": ["inspect"],
            }
        ],
        "acceptance_criteria": [
            {
                "criterion_id": "inspection_returned",
                "description": "The selected run can be inspected without execution.",
                "evidence_ids": ["inspection_evidence"],
            }
        ],
        "claims": [],
        "non_claims": ["Inspection does not rerun or scientifically qualify the scenario."],
        "transfers": [
            {
                "input_id": "completed_run_input",
                "purpose": "Read the selected run evidence.",
                "required": True,
                "content_sha256": digest,
            }
        ],
        "resource_envelope": {
            "resource_profile": "laptop-safe",
            "max_operations": 2,
            "max_total_cases": 2,
            "max_wall_time_s": 60.0,
        },
    }
    plan["operations"] = compile_typed_operation_edges(
        [
            {key: value for key, value in operation.items() if key not in {"input_refs", "output_refs"}}
            for operation in plan["operations"]
        ],
        request_inputs=request["inputs"],
        catalog=discovery_capability_catalog(),
    )["operations"]
    result = preflight_study_plan(request, plan, catalog=discovery_capability_catalog())
    assert result["status"] == "PLAN_VALID"
    assert result["execution_route"]["route"] == "LOCAL_FREE_AVAILABLE"
    assert result["commercial_status"]["oel_execution_amount"] == 0
    assert result["plan"]["operations"][0]["capability_id"] == "oel.run.inspect.v1"


def test_capability_parameter_contract_rejects_agent_invented_fields() -> None:
    receipt, normalized = validate_config_path(
        "scenario_main", ROOT / "configs" / "automation_smoke.yaml", workspace_root=ROOT
    )
    assert normalized is not None
    plan = _load("passive_plan.json")
    plan["operations"][1]["parameters"]["unrestricted_python"] = "print('bypass')"
    result = preflight_study_plan(
        _load("passive_request.json"),
        plan,
        catalog=discovery_capability_catalog(),
        normalized_configs={"scenario_main": normalized},
        config_receipts=[receipt],
    )
    assert result["status"] == "UNSUPPORTED"
    assert "capability.parameters_invalid" in {item["code"] for item in result["findings"]}


def test_question_drift_and_artifact_kind_confusion_fail_closed() -> None:
    receipt, normalized = validate_config_path(
        "scenario_main", ROOT / "configs" / "automation_smoke.yaml", workspace_root=ROOT
    )
    assert normalized is not None

    drifted = _load("passive_plan.json")
    drifted["question_answered"] = "Answer a materially different orbital-analysis question."
    result = preflight_study_plan(
        _load("passive_request.json"),
        drifted,
        catalog=discovery_capability_catalog(),
        normalized_configs={"scenario_main": normalized},
        config_receipts=[receipt],
    )
    assert result["status"] == "UNSUPPORTED"
    assert "plan.question_mismatch" in {item["code"] for item in result["findings"]}

    mistyped = _load("passive_plan.json")
    mistyped["operations"][1]["input_refs"][0]["kind"] = "review_store"
    result = preflight_study_plan(
        _load("passive_request.json"),
        mistyped,
        catalog=discovery_capability_catalog(),
        normalized_configs={"scenario_main": normalized},
        config_receipts=[receipt],
    )
    assert result["status"] == "UNSUPPORTED"
    codes = {item["code"] for item in result["findings"]}
    assert "plan.input_kind_mismatch" in codes
    assert "plan.input_kind_not_accepted" in codes


def test_required_input_and_transfer_cannot_omit_content_identity() -> None:
    request = _load("passive_request.json")
    request["inputs"][0]["content_sha256"] = None
    with pytest.raises(StudyContractError, match="required request input"):
        preflight_study_plan(
            request,
            _load("passive_plan.json"),
            catalog=discovery_capability_catalog(),
        )

    plan = _load("passive_plan.json")
    plan["transfers"][0]["content_sha256"] = None
    result = preflight_study_plan(
        _load("passive_request.json"),
        plan,
        catalog=discovery_capability_catalog(),
    )
    assert result["status"] == "CLARIFICATION_REQUIRED"
    assert "plan.contract_invalid" in {item["code"] for item in result["findings"]}
