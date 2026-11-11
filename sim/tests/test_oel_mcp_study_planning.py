from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

from integrations.oel_mcp.public_registry import public_contract_map
from integrations.oel_mcp.study_handlers import StudyPlanningMCPHandlers
from integrations.oel_mcp.study_server import main as study_server_main
from sim.study_planning import PUBLIC_DOCUMENTED_EXECUTOR_MODULES

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = Path(__file__).parent / "fixtures" / "study_planning"
HANDLING = {"marking": "PUBLIC TEST FIXTURE", "release_scope": "public", "owner": "OEL"}
MCP_SDK_AVAILABLE = importlib.util.find_spec("mcp") is not None


def _load(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def test_restricted_byo_mcp_profile_exposes_planning_without_execution() -> None:
    from integrations.oel_mcp.resources import build_public_resource_catalog

    handlers = StudyPlanningMCPHandlers(read_roots=(ROOT,))
    resources = {r.contract.uri: r.text for r in build_public_resource_catalog(
        profile=handlers.profile, tool_contracts=handlers.contracts.values())}
    tool_catalog = json.loads(resources["oel://capabilities/tools/v1"])
    assert [tool["name"] for tool in tool_catalog["tools"]] == list(handlers.contracts)
    routes = json.loads(resources["oel://agent/workflows/v1"])
    assert routes["study_planner"]["registered_here"] is True
    assert next(r for r in routes["routes"] if r["route_id"] == "scenario")["all_tools_registered_here"] is False
    assert set(handlers.contracts) == {
        "oel.describe_capabilities.v1",
        "oel.study.capabilities.v1",
        "oel.study.preflight.v1",
        "oel.study.plan_review.v1",
        "oel.hosted.validate_package.v1",
        "oel.study.prepare_capability_request.v1",
    }
    assert all(not contract.writes and not contract.executes for contract in handlers.contracts.values())

    response = handlers.call("oel.study.capabilities.v1", {"handling": HANDLING})
    assert response["status"] == "completed"
    manifest = response["result"]
    assert manifest["catalog_scope"] == "public_agent_discovery"
    assert manifest["access"] == {
        "descriptor_visibility": "public",
        "public_capabilities_available_locally": True,
        "pro_capabilities_available_locally": False,
        "pro_execution_tools_exposed": False,
        "hosted_preflight_required_for_pro": True,
    }
    capabilities = {item["capability_id"]: item for item in manifest["capabilities"]}
    public = capabilities["oel.scenario.execute.v1"]
    pro = capabilities["oel.pro.campaign.monte_carlo.v1"]
    assert public["access"] == {
        "discoverable_in_public": True,
        "available_in_public_install": True,
        "access_mode": "public_local",
        "hosted_pro_required": False,
    }
    assert public["availability"] == "available"
    assert public["executor_binding"]["status"] == "bound"
    assert public["executor_binding"]["executor_contract_ids"] == ["oel.run_scenario.v1"]
    assert pro["access"] == {
        "discoverable_in_public": True,
        "available_in_public_install": False,
        "access_mode": "hosted_pro_only",
        "hosted_pro_required": True,
    }
    assert pro["availability"] == "unavailable"
    assert pro["executor_binding"]["status"] == "unbound"
    public_executor_ids = set(public_contract_map("public_local")) | set(
        PUBLIC_DOCUMENTED_EXECUTOR_MODULES
    )
    for contract_id, module_name in PUBLIC_DOCUMENTED_EXECUTOR_MODULES.items():
        assert contract_id.startswith("python.module.")
        assert importlib.util.find_spec(module_name) is not None
    for capability in capabilities.values():
        binding = capability["executor_binding"]
        if capability["availability"] == "available":
            assert set(binding["executor_contract_ids"]) <= public_executor_ids
    assert not any(name.startswith("oel.pro.") for name in handlers.contracts)
    serialized = json.dumps(response)
    assert "source_path" not in serialized
    assert "provider_key" not in serialized


def test_study_server_host_config_launches_the_study_profile(capsys: pytest.CaptureFixture[str]) -> None:
    assert study_server_main(["--print-host-config", "codex", "--cwd", str(ROOT)]) == 0
    config = capsys.readouterr().out
    assert "integrations.oel_mcp.study_server" in config or "oel-study-mcp" in config
    assert "direct_frontier_restricted" in config


def test_restricted_byo_mcp_preflight_returns_ready_but_never_authorizes() -> None:
    handlers = StudyPlanningMCPHandlers(read_roots=(ROOT,))
    response = handlers.call(
        "oel.study.preflight.v1",
        {
            "request": _load("passive_request.json"),
            "plan": _load("passive_plan.json"),
            "configs": [
                {
                    "config_ref": "scenario_main",
                    "path": str(ROOT / "configs" / "automation_smoke.yaml"),
                }
            ],
            "handling": HANDLING,
        },
    )
    assert response["status"] == "completed", response["error"]
    result = response["result"]
    assert result["status"] == "PLAN_VALID"
    assert result["execution_route"]["route"] == "LOCAL_FREE_AVAILABLE"
    assert result["commercial_status"]["oel_execution_amount"] == 0
    assert result["commercial_status"]["payment_authorized"] is False
    assert result["review_status"]["execution_authorized"] is False
    assert result["plan"]["config_constraints"][0]["path"] == "/scenario_name"
    assert response["effects"] == {
        "reads": True,
        "writes": False,
        "executes": False,
        "external_communication": False,
    }

    review_response = handlers.call(
        "oel.study.plan_review.v1",
        {"planning_result": result, "handling": HANDLING},
    )
    assert review_response["status"] == "completed", review_response["error"]
    review = review_response["result"]
    assert review["execution_route"]["route"] == "LOCAL_FREE_AVAILABLE"
    assert review["commercial_status"]["oel_execution_amount"] == 0
    assert review["execution_authorized"] is False


def test_restricted_byo_mcp_declines_unbound_pro_capability() -> None:
    handlers = StudyPlanningMCPHandlers(read_roots=(ROOT,))
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
        {"kind": "campaign_evidence", "source_operation_ids": ["uncertainty_campaign"]}
    )
    plan["forbidden_capabilities"] = []
    plan["resource_envelope"].update({"max_total_cases": 200, "max_wall_time_s": 900.0})
    response = handlers.call(
        "oel.study.preflight.v1",
        {
            "request": _load("passive_request.json"),
            "plan": plan,
            "configs": [
                {
                    "config_ref": "scenario_main",
                    "path": str(ROOT / "configs" / "automation_smoke.yaml"),
                }
            ],
            "handling": HANDLING,
        },
    )
    assert response["status"] == "completed", response["error"]
    result = response["result"]
    assert result["status"] == "UNSUPPORTED"
    assert result["execution_route"]["route"] == "NOT_ELIGIBLE"
    assert result["execution_route"]["hosted_preflight_required"] is False
    assert result["commercial_status"]["oel_execution_amount"] == 0
    assert result["commercial_status"]["payment_authorized"] is False


@pytest.mark.skipif(not MCP_SDK_AVAILABLE, reason="optional MCP SDK profile is not installed")
def test_byo_study_planner_is_discoverable_by_an_official_mcp_client() -> None:
    import anyio
    from mcp import Client, StdioServerParameters, stdio_client

    parameters = StdioServerParameters(
        command=sys.executable,
        args=["-m", "integrations.oel_mcp.study_server"],
        cwd=ROOT,
        env={
            **os.environ,
            "OEL_MCP_ADAPTER": "sdk",
            "OEL_MCP_READ_ROOTS": str(ROOT),
        },
    )

    async def exercise() -> tuple[tuple[str, ...], dict[str, object]]:
        async with Client(stdio_client(parameters), mode="auto", cache=None) as client:
            tools = await client.list_tools(cache_mode="reload")
            called = await client.call_tool("oel.study.capabilities.v1", {"handling": HANDLING})
            return tuple(tool.name for tool in tools.tools), dict(called.structured_content or {})

    names, payload = anyio.run(exercise)
    assert "oel.study.preflight.v1" in names
    assert "oel.hosted.validate_package.v1" in names
    assert "oel.study.prepare_capability_request.v1" in names
    assert payload["status"] == "completed"
    assert payload["effects"]["executes"] is False
