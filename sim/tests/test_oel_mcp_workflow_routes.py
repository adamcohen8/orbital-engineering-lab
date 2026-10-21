from __future__ import annotations

import json
from pathlib import Path

import pytest

from integrations.oel_mcp.agent_guidance import workflow_routes
from integrations.oel_mcp.public_registry import PUBLIC_TOOL_CONTRACTS, public_contracts_for_profile
from integrations.oel_mcp.resources import MAX_RESOURCE_BYTES, build_public_resource_catalog

PRO_AVAILABLE = (Path(__file__).resolve().parents[2] / "integrations/oel_mcp/pro_registry.py").is_file()


def _route_map(tool_ids: list[str] | tuple[str, ...]) -> dict[str, dict]:
    return {route["route_id"]: route for route in workflow_routes(tool_ids)["routes"]}


def test_public_workflow_routes_cover_registered_public_workflows() -> None:
    tool_ids = tuple(contract.tool_id for contract in PUBLIC_TOOL_CONTRACTS)
    routes = _route_map(tool_ids)

    assert set(routes) == {
        "scenario", "study_plan", "review", "comparison", "plot", "animation", "tasks",
        "report_evidence", "handoff", "maneuver_readiness", "public_fsw", "cr3bp_render",
        "study_evidence", "interchange",
    }
    assert routes["plot"]["tools"] == [
        "oel.plot_evidence.v1", "oel.plan_review_plot.v1", "oel.render_review_plot.v2",
    ]
    assert all(route["all_tools_registered_here"] for name, route in routes.items() if name != "study_plan")
    assert routes["study_plan"]["all_tools_registered_here"] is False
    routed = {tool for route in routes.values() for tool in route["tools"]}
    assert set(tool_ids) - {"oel.describe_capabilities.v1"} <= routed
    assert all(
        not tool_id.startswith("oel.pro.")
        for route in workflow_routes(tool_ids)["routes"]
        for tool_id in route["tools"]
    )


@pytest.mark.skipif(not PRO_AVAILABLE, reason="private MCP registry is not exported")
def test_pro_workflow_routes_cover_active_pro_registry() -> None:
    from integrations.oel_mcp.pro_registry import PRO_TOOL_CONTRACTS

    tool_ids = tuple(contract.tool_id for contract in PUBLIC_TOOL_CONTRACTS + PRO_TOOL_CONTRACTS)
    routes = _route_map(tool_ids)

    assert {"pro_analytics", "orbit_determination", "pro_fsw", "spacecraft_twin", "trajectory_optimization"} <= set(routes)
    assert routes["orbit_determination"]["tools"][0] == "oel.run_dynamics_od.v1"
    assert all(tool_id in tool_ids for tool_id in routes["orbit_determination"]["tools"])
    assert all(route["all_tools_registered_here"] for name, route in routes.items() if name != "study_plan")
    routed = {tool for route in routes.values() for tool in route["tools"]}
    assert set(tool_ids) - {"oel.describe_capabilities.v1"} <= routed


def test_new_cr3bp_and_od_routes_follow_active_ids() -> None:
    routes = _route_map([
        "oel.plan_cr3bp_render.v1",
        "oel.render_cr3bp.v1",
        "oel.pro.run_ground_sensor_od.v1",
        "oel.pro.run_relative_od.v1",
    ])

    assert routes["cr3bp_render"]["tools"] == [
        "oel.plan_cr3bp_render.v1", "oel.render_cr3bp.v1",
    ]
    assert routes["cr3bp_render"]["all_tools_registered_here"] is True
    assert routes["orbit_determination"]["tools"] == [
        "oel.pro.run_ground_sensor_od.v1", "oel.pro.run_relative_od.v1",
    ]


def test_partial_new_route_reports_missing_pair_without_advertising_pro_tools() -> None:
    routes = _route_map(["oel.plan_cr3bp_render.v1"])

    assert routes["cr3bp_render"]["all_tools_registered_here"] is False
    assert routes["cr3bp_render"]["missing_tools"] == ["oel.render_cr3bp.v1"]
    assert "orbit_determination" not in routes
    assert "pro_analytics" not in routes
    assert workflow_routes(["oel.plan_cr3bp_render.v1"])["study_planner"]["registered_here"] is False


@pytest.mark.parametrize("profile", ["public_local", "direct_frontier_restricted", "mendicant_sealed"])
def test_local_workflow_discovery_does_not_advertise_unregistered_tools_or_authorize_effects(profile: str) -> None:
    contracts = public_contracts_for_profile(profile)
    resources = build_public_resource_catalog(profile=profile, tool_contracts=contracts)
    resource = next(item for item in resources if item.contract.uri == "oel://agent/workflows/v1")
    payload = json.loads(resource.text)
    local = payload["local_workflows"]

    assert resource.size <= MAX_RESOURCE_BYTES
    assert local["host_authorization_required"] is True
    assert local["discovery_authorizes_effects"] is False
    assert {item["workflow_id"] for item in local["workflows"]} == {
        "modular_data_io", "full_state_innovation_recovery",
    }
    for item in local["workflows"]:
        assert item["dedicated_mcp_tools"] == []
        assert (Path(__file__).resolve().parents[2] / item["documentation"]).is_file()
    assert "sim.pro_" not in resource.text
    assert "agents/pro" not in resource.text
    assert "/Users/" not in resource.text
    active_ids = {contract.tool_id for contract in contracts}
    for route in payload["routes"]:
        assert set(route["tools"]) - set(route["missing_tools"]) <= active_ids
