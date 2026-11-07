from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from sim.study_planning import (
    StudyContractError,
    bind_study_plan,
    bind_study_request,
    build_capability_manifest,
    build_plan_review,
    contract_schema,
    discovery_capability_catalog,
    preflight_study_plan,
    verify_bound_plan,
)
from sim.study_planning.cli import _load_object
from sim.study_planning.config_validation import validate_config_path
from sim.study_planning.contracts import (
    STUDY_COMPLAINT_RESOLUTION_SCHEMA_ID,
    canonical_sha256,
    require_valid_document,
)

FIXTURES = Path(__file__).parent / "fixtures" / "study_planning"
ROOT = Path(__file__).resolve().parents[2]


def _load(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def _bind_plan(request: dict, proposal: dict) -> dict:
    manifest = build_capability_manifest(discovery_capability_catalog())
    return bind_study_plan(
        proposal,
        request_id=request["request_id"],
        request_sha256=request["request_sha256"],
        capability_manifest_sha256=manifest["manifest_sha256"],
        planner_version=manifest["planner_version"],
    )


def _valid_result() -> dict:
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


def test_request_plan_and_review_are_content_bound_and_not_authorized() -> None:
    request = bind_study_request(_load("passive_request.json"))
    plan = _bind_plan(request, _load("passive_plan.json"))

    assert verify_bound_plan(plan).valid
    assert plan["request_sha256"] == request["request_sha256"]
    assert len(plan["capability_manifest_sha256"]) == 64
    modified = deepcopy(plan)
    modified["question_answered"] = "A different material question"
    assert not verify_bound_plan(modified).valid

    with pytest.raises(StudyContractError, match="study_planning_result"):
        build_plan_review(plan)

    result = _valid_result()
    review = build_plan_review(result)
    assert review["plan_sha256"] == result["plan_sha256"]
    assert review["request_sha256"] == result["request"]["request_sha256"]
    assert review["capability_manifest_sha256"] == result["capability_manifest_sha256"]
    assert review["operations"] == result["plan"]["operations"]
    assert review["acceptance_criteria"] == result["plan"]["acceptance_criteria"]
    assert review["claims"] == result["plan"]["claims"]
    assert review["resource_accounting"] == result["resource_accounting"]
    assert review["commercial_status"]["oel_execution_amount"] == 0
    assert review["user_review_required"] is True
    assert review["execution_authorized"] is False
    assert review["agent_may_authorize"] is False


def test_closed_contract_rejects_unknown_fields_and_dependency_cycles() -> None:
    request = bind_study_request(_load("passive_request.json"))
    plan = _load("passive_plan.json")
    plan["surprise"] = True
    with pytest.raises(StudyContractError, match="not an allowed field"):
        _bind_plan(request, plan)

    cyclic = _load("passive_plan.json")
    cyclic["operations"][0]["depends_on"] = ["execute"]
    with pytest.raises(StudyContractError, match="acyclic"):
        _bind_plan(request, cyclic)


def test_malformed_resource_envelope_returns_a_contract_error() -> None:
    request = bind_study_request(_load("passive_request.json"))
    malformed = _load("passive_plan.json")
    malformed["resource_envelope"] = None
    with pytest.raises(StudyContractError, match="resource_envelope"):
        _bind_plan(request, malformed)


def test_planning_cli_rejects_symlinked_json_documents(tmp_path: Path) -> None:
    target = tmp_path / "request.json"
    target.write_text(json.dumps(_load("passive_request.json")), encoding="utf-8")
    alias = tmp_path / "alias.json"
    alias.symlink_to(target)

    with pytest.raises(ValueError, match="Could not safely read planning document"):
        _load_object(alias)


def test_manifest_digest_binds_access_policy_and_executor_bindings() -> None:
    manifest = build_capability_manifest(discovery_capability_catalog())
    assert manifest["manifest_sha256"] == canonical_sha256(
        {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    )
    capabilities = {item["capability_id"]: item for item in manifest["capabilities"]}
    assert capabilities["oel.scenario.execute.v1"]["executor_binding"] == {
        "status": "bound",
        "executor_contract_ids": ["oel.run_scenario.v1"],
        "adapter_id": "oel.study.adapter.public_scenario_execute.v1",
    }
    assert capabilities["oel.pro.campaign.monte_carlo.v1"]["availability"] == "unavailable"

    tampered = deepcopy(manifest)
    tampered["access"]["pro_execution_tools_exposed"] = True
    with pytest.raises(StudyContractError, match="manifest digest"):
        require_valid_document(tampered, expected_schema="oel.hosted_study_capability_manifest.v2")


def test_required_request_input_requires_a_digest() -> None:
    request = _load("passive_request.json")
    request["inputs"][0]["content_sha256"] = None
    with pytest.raises(StudyContractError, match="required request input"):
        bind_study_request(request)


def test_config_constraint_paths_teach_and_enforce_json_pointer_syntax() -> None:
    schema = contract_schema("oel.hosted_study_plan.v2")
    path_schema = schema["properties"]["config_constraints"]["items"]["properties"]["path"]
    assert "RFC 6901 JSON Pointer" in path_schema["description"]
    assert path_schema["examples"] == ["/scenario_name", "/simulator/duration_s"]

    request = bind_study_request(_load("passive_request.json"))
    dotted = _load("passive_plan.json")
    dotted["config_constraints"][0]["path"] = "simulator.duration_s"
    with pytest.raises(StudyContractError, match="has an invalid format"):
        _bind_plan(request, dotted)

    escaped = _load("passive_plan.json")
    escaped["config_constraints"][0]["path"] = "/metadata/key~1with~1slashes/~0literal"
    assert _bind_plan(request, escaped)["config_constraints"][0][
        "path"
    ] == "/metadata/key~1with~1slashes/~0literal"


def test_confirmed_refund_resolution_must_refund_full_oel_charge() -> None:
    resolution = {
        "schema": STUDY_COMPLAINT_RESOLUTION_SCHEMA_ID,
        "resolution_id": "resolution:one",
        "study_complaint_id": "complaint:one",
        "disposition": "refund",
        "reason": "The completed run omitted promised evidence.",
        "charged_oel_amount_usd": 275,
        "refund_amount_usd": 99,
        "corrected_run_id": None,
    }
    with pytest.raises(StudyContractError, match="full charged OEL amount"):
        require_valid_document(resolution, expected_schema=STUDY_COMPLAINT_RESOLUTION_SCHEMA_ID)

    resolution["refund_amount_usd"] = 275
    assert require_valid_document(
        resolution, expected_schema=STUDY_COMPLAINT_RESOLUTION_SCHEMA_ID
    )["refund_amount_usd"] == 275
