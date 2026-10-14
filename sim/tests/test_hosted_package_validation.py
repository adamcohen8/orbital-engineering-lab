from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from integrations.oel_mcp.study_handlers import StudyPlanningMCPHandlers
from sim.hosted_client import (
    hosted_execution_package_schema,
    hosted_pro_package_schema,
    validate_hosted_execution_package,
    validate_hosted_pro_package,
)

ROOT = Path(__file__).resolve().parents[2]


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _package(tmp_path: Path) -> Path:
    root = tmp_path / "hosted-pro-package"
    problem = {
        "schema_version": "oel.pro_tracking_od_problem.v1",
        "problem_name": "synthetic_public_safe_tracking_problem",
        "observations": "selected reduced optical observations are stored separately in a future package version",
    }
    request = {
        "question": "Fit the selected reduced optical observations with the Hosted Pro sequential OD workflow and return replayable evidence.",
        "inputs": [
            {
                "input_id": "tracking_problem",
                "kind": "pro_tracking_od_problem",
                "schema_id": "oel.pro_tracking_od_problem.v1",
                "metadata": {
                    "measurement_model": "inertial_ra_dec",
                    "observation_count": 24,
                    "arc_duration_s": 1800.0,
                    "station_count": 1,
                },
                "content_sha256": None,
                "description": "Synthetic public-safe reduced optical tracking problem.",
                "required": True,
            }
        ],
        "required_choices": [
            {
                "choice_id": "od_workflow",
                "description": "Use the Hosted Pro reduced-tracking workflow with an untouched holdout.",
                "value": "sequential_fit_with_holdout",
                "source": "user",
                "material": True,
            }
        ],
        "clarifications": [],
        "handling": {
            "marking": "PUBLIC TEST FIXTURE",
            "release_scope": "public",
            "owner": "OEL",
        },
    }
    plan = {
        "question_answered": request["question"],
        "operations": [
            {
                "operation_id": "fit_tracking",
                "capability_id": "orbit_determination.reduced_tracking",
                "depends_on": [],
                "input_refs": [],
                "output_refs": [],
                "config_ref": None,
                "parameters": {"authoritative_replay_required": True},
                "bounds": {
                    "max_cases": 1,
                    "max_wall_time_s": 600.0,
                    "max_input_bytes": 1_000_000,
                    "max_observations": 24,
                    "max_arc_duration_s": 1800.0,
                    "max_stations": 1,
                },
            }
        ],
        "config_constraints": [],
        "assumptions": ["The selected observations are already reduced inertial optical angles."],
        "allowed_defaults": [],
        "forbidden_capabilities": [],
        "evidence_requirements": [
            {
                "evidence_id": "od_evidence",
                "kind": "pro_tracking_od_evidence",
                "description": "Content-bound fit, holdout, and replay evidence.",
                "source_operation_ids": ["fit_tracking"],
            }
        ],
        "acceptance_criteria": [
            {
                "criterion_id": "evidence_complete",
                "description": "The requested OD evidence and authoritative replay receipt are complete.",
                "evidence_ids": ["od_evidence"],
            }
        ],
        "claims": [],
        "non_claims": ["The result is not calibrated predicted accuracy or operational authority."],
        "transfers": [
            {
                "input_id": "tracking_problem",
                "purpose": "Run the exact user-selected reduced-tracking problem.",
                "required": True,
            }
        ],
        "resource_envelope": {
            "resource_profile": "laptop-safe",
            "max_operations": 2,
            "max_total_cases": 1,
            "max_wall_time_s": 900.0,
        },
    }
    manifest = {
        "schema": "oel.hosted_pro_package.v1",
        "package_name": "synthetic-reduced-tracking",
        "request_path": "request.json",
        "plan_path": "plan.json",
        "inputs": [
            {
                "input_id": "tracking_problem",
                "relative_path": "inputs/problem.json",
                "media_type": "application/json",
                "config_ref": None,
            }
        ],
        "retention": {
            "policy_id": "oel.retention.standard-30d.v1",
            "delete_incomplete_uploads": True,
            "local_originals_retained": True,
        },
    }
    _write_json(root / "inputs/problem.json", problem)
    _write_json(root / "request.json", request)
    _write_json(root / "plan.json", plan)
    _write_json(root / "oel-hosted-package.json", manifest)
    return root


def _public_package(tmp_path: Path) -> Path:
    root = tmp_path / "hosted-public-package"
    plan = json.loads((ROOT / "sim/tests/fixtures/study_planning/passive_plan.json").read_text(encoding="utf-8"))
    for operation in plan["operations"]:
        operation["bounds"]["max_input_bytes"] = 1_000_000
    _write_json(
        root / "request.json",
        json.loads((ROOT / "sim/tests/fixtures/study_planning/passive_request.json").read_text(encoding="utf-8")),
    )
    _write_json(root / "plan.json", plan)
    scenario_path = root / "inputs/scenario.yaml"
    scenario_path.parent.mkdir(parents=True, exist_ok=True)
    scenario_path.write_bytes((ROOT / "configs/automation_smoke.yaml").read_bytes())
    _write_json(
        root / "oel-hosted-package.json",
        {
            "schema": "oel.hosted_execution_package.v1",
            "package_name": "public-automation-smoke",
            "request_path": "request.json",
            "plan_path": "plan.json",
            "inputs": [
                {
                    "input_id": "scenario_input",
                    "relative_path": "inputs/scenario.yaml",
                    "media_type": "application/yaml",
                    "config_ref": "scenario_main",
                }
            ],
            "retention": {
                "policy_id": "oel.retention.ephemeral-7d.v1",
                "delete_incomplete_uploads": True,
                "local_originals_retained": True,
            },
        },
    )
    return root


def test_public_package_validator_plans_pro_without_importing_or_executing(tmp_path: Path) -> None:
    root = _package(tmp_path)

    result = validate_hosted_pro_package(root, workspace_root=tmp_path)

    assert result["status"] == "CLIENT_VALID"
    assert result["planning_result"]["status"] == "PLAN_VALID"
    assert result["planning_result"]["execution_route"] == {
        "route": "HOSTED_PRO_REQUIRED",
        "public_capability_ids": [],
        "pro_capability_ids": ["orbit_determination.reduced_tracking"],
        "unavailable_capability_ids": [],
        "local_free_available": False,
        "hosted_pro_required": True,
        "hosted_preflight_required": True,
    }
    assert result["hosted_preflight_required"] is True
    assert result["local_execution_available"] is False
    assert result["execution_options"]["hosted"]["available"] is False
    assert result["recommendation"]["option"] is None
    from sim.hosted_client.contracts import route_planning_result
    route = route_planning_result(result["planning_result"])
    assert route["execution_options"]["hosted"]["available"] is False
    assert route["next_action"] == "use_public_fallback_or_revise_plan"
    assert route["recommendation"]["option"] is None
    assert result["execution_authorized"] is False
    assert result["payment_authorized"] is False
    assert result["pro_code_imported"] is False
    assert result["files_uploaded"] is False
    assert result["inventory"][0]["content_sha256"] == result["bound_request"]["inputs"][0]["content_sha256"]
    assert result["bound_plan"]["transfers"][0]["content_sha256"] == result["inventory"][0]["content_sha256"]
    assert result["bound_plan"]["operations"][0]["input_refs"][0]["port_id"] == "problem"


def test_generic_package_validator_keeps_public_local_and_hosted_closed(tmp_path: Path) -> None:
    result = validate_hosted_execution_package(_public_package(tmp_path), workspace_root=tmp_path)

    assert result["status"] == "CLIENT_VALID"
    assert result["planning_result"]["execution_route"]["route"] == "LOCAL_FREE_AVAILABLE"
    assert result["execution_options"] == {
        "local": {"available": True, "oel_execution_amount_usd": 0},
        "hosted": {"available": False, "quote_required": True},
    }
    assert result["recommendation"] == {"option": "local", "requirement": False}
    assert result["local_execution_available"] is True
    assert result["hosted_preflight_required"] is True
    assert result["execution_authorized"] is False
    assert result["payment_authorized"] is False
    assert result["files_uploaded"] is False


def test_package_validator_rejects_unqualified_planetary_origin(tmp_path: Path) -> None:
    root = _public_package(tmp_path)
    (root / "inputs/scenario.yaml").write_bytes((ROOT / "configs/mars_reference_drag.yaml").read_bytes())
    result = validate_hosted_execution_package(root, workspace_root=tmp_path)
    assert result["status"] == "CLIENT_INVALID"
    assert "package.planetary_unsupported" in {item["code"] for item in result["issues"]}
    assert result["execution_authorized"] is False


def test_package_validator_rejects_unqualified_additional_gravity(tmp_path: Path) -> None:
    import hashlib

    import yaml

    from sim.tests.test_additional_gravity import raw_scenario

    root = _public_package(tmp_path)
    path = root / "inputs/scenario.yaml"
    document = raw_scenario(root / "inputs")
    document["outputs"]["output_dir"] = str(root / "outputs")
    path.write_text(yaml.safe_dump(document))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    def bind_digest(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if key == "content_sha256":
                    value[key] = digest
                else:
                    bind_digest(child)
        elif isinstance(value, list):
            for child in value:
                bind_digest(child)
    for name in ("request.json", "plan.json"):
        control = root / name
        value = json.loads(control.read_text())
        bind_digest(value)
        _write_json(control, value)
    result = validate_hosted_execution_package(root, workspace_root=tmp_path)
    assert result["status"] == "CLIENT_INVALID"
    assert "package.additional_gravity_unsupported" in {item["code"] for item in result["issues"]}
    assert result["execution_authorized"] is False


def test_package_validator_rejects_tampering_and_missing_port_metadata(tmp_path: Path) -> None:
    root = _package(tmp_path)
    request_path = root / "request.json"
    request = json.loads(request_path.read_text(encoding="utf-8"))
    request["inputs"][0]["content_sha256"] = "0" * 64
    request["inputs"][0]["metadata"].pop("observation_count")
    _write_json(request_path, request)

    result = validate_hosted_pro_package(root, workspace_root=tmp_path)

    assert result["status"] == "CLIENT_INVALID"
    codes = {item["code"] for item in result["issues"]}
    assert "package.input_digest_mismatch" in codes
    assert result["hosted_preflight_required"] is False


def test_package_digest_binds_exact_control_document_bytes(tmp_path: Path) -> None:
    root = _package(tmp_path)
    first = validate_hosted_pro_package(root, workspace_root=tmp_path)
    request_path = root / "request.json"
    request = json.loads(request_path.read_text(encoding="utf-8"))
    request_path.write_text(json.dumps(request, separators=(",", ":")) + "\n", encoding="utf-8")

    second = validate_hosted_pro_package(root, workspace_root=tmp_path)

    assert first["status"] == second["status"] == "CLIENT_VALID"
    assert first["bound_request"]["request_sha256"] == second["bound_request"]["request_sha256"]
    assert first["control_documents"]["request"]["source_sha256"] != second["control_documents"]["request"]["source_sha256"]
    assert first["package_sha256"] != second["package_sha256"]


def test_package_validator_rejects_paths_and_symlinks(tmp_path: Path) -> None:
    root = _package(tmp_path)
    manifest_path = root / "oel-hosted-package.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["inputs"][0]["relative_path"] = "../outside.json"
    _write_json(manifest_path, manifest)
    result = validate_hosted_pro_package(root, workspace_root=tmp_path)
    assert "package.path_unsafe" in {item["code"] for item in result["issues"]}

    if not hasattr(os, "symlink"):
        return
    root = _package(tmp_path / "linked")
    original = root / "inputs/problem.json"
    outside = tmp_path / "outside.json"
    outside.write_text(original.read_text(encoding="utf-8"), encoding="utf-8")
    original.unlink()
    original.symlink_to(outside)
    linked = validate_hosted_pro_package(root, workspace_root=tmp_path)
    assert linked["status"] == "CLIENT_INVALID"
    assert "package.input_unreadable" in {item["code"] for item in linked["issues"]}


def test_package_validator_rejects_public_only_plan(tmp_path: Path) -> None:
    root = _package(tmp_path)
    plan_path = root / "plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan["operations"][0]["capability_id"] = "oel.run.inspect.v1"
    _write_json(plan_path, plan)

    result = validate_hosted_pro_package(root, workspace_root=tmp_path)

    assert result["status"] == "CLIENT_INVALID"
    assert result["files_uploaded"] is False


def test_package_schema_and_cli_are_public_and_machine_readable(tmp_path: Path) -> None:
    schema = hosted_execution_package_schema()
    assert schema["properties"]["schema"]["const"] == "oel.hosted_execution_package.v1"
    documented = json.loads(
        (ROOT / "docs/contracts/schemas/oel-hosted-execution-package-v1.schema.json").read_text(encoding="utf-8")
    )
    assert documented == schema
    legacy_schema = hosted_pro_package_schema()
    legacy_documented = json.loads(
        (ROOT / "docs/contracts/schemas/oel-hosted-pro-package-v1.schema.json").read_text(encoding="utf-8")
    )
    assert legacy_documented == legacy_schema
    root = _package(tmp_path)

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "sim.hosted_client",
            "validate-package",
            str(root),
            "--workspace-root",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["status"] == "CLIENT_VALID"


def test_restricted_mcp_can_validate_package_but_cannot_upload_or_execute(tmp_path: Path) -> None:
    root = _package(tmp_path)
    handlers = StudyPlanningMCPHandlers(read_roots=(tmp_path,))

    response = handlers.call(
        "oel.hosted.validate_package.v1",
        {
            "package_path": str(root),
            "handling": {
                "marking": "PUBLIC TEST FIXTURE",
                "release_scope": "public",
                "owner": "OEL",
            },
        },
    )

    assert response["status"] == "completed", response["error"]
    assert response["result"]["status"] == "CLIENT_VALID"
    assert response["result"]["files_uploaded"] is False
    assert response["result"]["execution_authorized"] is False
    assert response["effects"] == {
        "reads": True,
        "writes": False,
        "executes": False,
        "external_communication": False,
    }
