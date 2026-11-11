from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from integrations.oel_mcp.acceptance import run_public_workflow_acceptance
from integrations.oel_mcp.interop import (
    CAPABILITY_TOOL_ID,
    CLAUDE_CAPABILITY_TOOL_ID,
    INSPECTOR_NPM_INTEGRITY,
    INSPECTOR_NPM_SPEC,
    PUBLIC_TOOL_IDS,
    _git_commit,
    _git_source_state,
    _parse_claude_payload,
    _parse_claude_plot_selection,
    _parse_codex_payload,
    _parse_codex_plot_selection,
    _resolve_executable,
    plot_selection_prompt,
    run_sdk_stdio,
)
from integrations.oel_mcp.public_registry import PUBLIC_TOOL_CONTRACTS, public_contracts_for_profile
from integrations.oel_mcp.resources import PUBLIC_RESOURCE_URIS, PUBLIC_SAVED_QUERY_NAMES, build_public_resource_catalog

ROOT = Path(__file__).resolve().parents[2]
MCP_SDK_AVAILABLE = importlib.util.find_spec("mcp") is not None


def _capability_payload() -> dict[str, object]:
    return {
        "tool_id": CAPABILITY_TOOL_ID,
        "status": "completed",
        "effects": {
            "reads": True,
            "writes": False,
            "executes": False,
            "external_communication": False,
        },
        "evidence": {"complete": True, "empty": False, "truncated": False},
        "result": {
            "transport": "stdio",
            "capabilities": [{"tool_id": tool_id} for tool_id in PUBLIC_TOOL_IDS],
        },
    }


def test_public_tool_annotations_match_declared_effects() -> None:
    for contract in PUBLIC_TOOL_CONTRACTS:
        assert contract.mcp_definition()["annotations"] == {
            "readOnlyHint": not contract.writes and not contract.executes,
            "destructiveHint": contract.writes or contract.executes,
            "idempotentHint": not contract.writes and not contract.executes,
            "openWorldHint": False,
        }


def test_supported_public_resources_are_bounded_and_exclude_private_metadata() -> None:
    catalog = build_public_resource_catalog(profile="public_local", tool_contracts=public_contracts_for_profile("public_local"))

    assert tuple(resource.contract.uri for resource in catalog) == PUBLIC_RESOURCE_URIS
    assert all(resource.size <= 500_000 for resource in catalog)
    tool_catalog = json.loads(catalog[0].text)
    query_catalog = json.loads(catalog[1].text)
    task_catalog = json.loads(catalog[2].text)
    plot_catalog = json.loads(catalog[PUBLIC_RESOURCE_URIS.index("oel://review/plot-recipes/v1")].text)
    animation_catalog = json.loads(
        catalog[PUBLIC_RESOURCE_URIS.index("oel://review/animation-recipes/v1")].text
    )
    workflow_catalog = json.loads(catalog[PUBLIC_RESOURCE_URIS.index("oel://analysis/workflows/v1")].text)
    assert [tool["name"] for tool in tool_catalog["tools"]] == list(PUBLIC_TOOL_IDS)
    assert {query["name"] for query in query_catalog["queries"]} == set(PUBLIC_SAVED_QUERY_NAMES)
    assert all("public" in task["tags"] and "pro" not in task["tags"] for task in task_catalog["tasks"])
    assert {recipe["recipe_id"] for recipe in plot_catalog["recipes"]} >= {
        "relative_range",
        "relative_position_ric_2d",
    }
    assert {recipe["recipe_id"] for recipe in animation_catalog["recipes"]} == {
        "relative_position_ric_2d"
    }
    assert animation_catalog["quality_policy"]["contact_sheet_required"] is True
    assert plot_catalog["routing"]["oel_review_evidence_plotter_is_authoritative"] is True
    local_routes = {workflow["workflow_id"]: workflow for workflow in workflow_catalog["workflows"]}
    assert {"trajectory_optimization", "constellation_optimization", "scale_catalog",
            "local_tracking_and_slr", "local_campaigns", "controller_bench"} <= set(local_routes)
    assert not workflow_catalog["cross_cutting_pro_escalations"]
    for workflow in local_routes.values():
        assert set(workflow["mcp_tools"]) <= set(PUBLIC_TOOL_IDS)
    public_resource_text = "".join(resource.text for resource in catalog)
    assert "sim.intent_hypothesis" not in public_resource_text
    assert "agents/pro" not in public_resource_text
    assert "/Users/" not in public_resource_text


def test_sdk_resource_reader_rejects_unlisted_uri_without_path_details() -> None:
    import anyio
    from mcp import Client, MCPError

    from integrations.oel_mcp.public_handlers import PublicOELMCPHandlers
    from integrations.oel_mcp.sdk_protocol import build_sdk_server

    async def exercise() -> MCPError:
        async with Client(build_sdk_server(PublicOELMCPHandlers()), mode="auto", cache=None) as client:
            try:
                await client.read_resource("file:///private/customer/secret.txt", cache_mode="reload")
            except MCPError as exc:
                return exc
        raise AssertionError("Unknown resource unexpectedly completed.")

    error = anyio.run(exercise)

    assert error.code == -32602
    assert error.message == "Resource is not available in this deployment profile."
    assert "private/customer" not in str(error)


def test_inspector_pin_records_exact_stable_artifact() -> None:
    assert INSPECTOR_NPM_SPEC == "@modelcontextprotocol/inspector@2.0.0"
    assert INSPECTOR_NPM_INTEGRITY.startswith("sha512-")


def test_executable_resolution_preserves_virtual_environment_symlink(tmp_path: Path) -> None:
    base = tmp_path / "base-python"
    base.touch()
    launcher = tmp_path / "venv-python"
    launcher.symlink_to(base)

    assert _resolve_executable(str(launcher), "python") == launcher


def test_interop_commit_marker_tolerates_generated_export_without_git(tmp_path: Path) -> None:
    assert _git_commit(tmp_path) == "unavailable_public_export"
    assert _git_source_state(tmp_path) == {
        "commit": "unavailable_public_export",
        "clean": None,
        "release_evidence_eligible": False,
    }


def test_interop_export_does_not_inherit_parent_repository_provenance(tmp_path: Path) -> None:
    git = shutil.which("git")
    if git is None:
        pytest.skip("Git is unavailable")
    repository = tmp_path / "source"
    repository.mkdir()
    subprocess.run([git, "init", "-q"], cwd=repository, check=True)
    subprocess.run([git, "-c", "user.name=OEL Test", "-c", "user.email=test@example.invalid",
                    "commit", "--allow-empty", "-qm", "fixture"], cwd=repository, check=True)
    commit = subprocess.check_output([git, "rev-parse", "HEAD"], cwd=repository, text=True).strip()
    assert _git_commit(repository) == commit
    assert _git_source_state(repository) == {"commit": commit, "clean": True, "release_evidence_eligible": True}
    export = repository / "export"
    export.mkdir()
    (export / "README.md").write_text("Generated export without Git metadata")
    assert _git_commit(export) == "unavailable_public_export"
    assert _git_source_state(export) == {
        "commit": "unavailable_public_export", "clean": None, "release_evidence_eligible": False,
    }
    assert _git_source_state(repository) == {"commit": commit, "clean": False, "release_evidence_eligible": False}


def test_codex_parser_requires_one_successful_structured_oel_call() -> None:
    payload = _capability_payload()
    stdout = json.dumps(
        {
            "type": "item.completed",
            "item": {
                "type": "mcp_tool_call",
                "server": "oel",
                "tool": CAPABILITY_TOOL_ID,
                "status": "completed",
                "error": None,
                "result": {"structured_content": payload},
            },
        }
    )

    assert _parse_codex_payload(stdout) == payload


def test_claude_parser_requires_normalized_tool_name_and_no_permission_denial() -> None:
    payload = _capability_payload()
    stdout = "\n".join(
        (
            json.dumps(
                {
                    "type": "assistant",
                    "message": {
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "toolu_test",
                                "name": CLAUDE_CAPABILITY_TOOL_ID,
                                "input": {},
                            }
                        ]
                    },
                }
            ),
            json.dumps({"type": "user", "tool_use_result": {"structuredContent": payload}}),
            json.dumps({"type": "result", "permission_denials": []}),
        )
    )

    assert _parse_claude_payload(stdout) == payload


def _plot_selection_payload() -> dict[str, object]:
    return {
        "tool_id": "oel.plot_evidence.v1",
        "status": "completed",
        "result": {
            "recipe_id": "relative_position_ric_2d",
            "artifact": {
                "artifact_id": "host_ric",
                "path_exists": True,
                "qa": {"visual_qa_status": "pending_agent_review"},
            },
        },
    }


def test_cross_host_natural_language_plot_selection_parsers_require_oel_recipe(tmp_path: Path) -> None:
    payload = _plot_selection_payload()
    codex_stdout = json.dumps(
        {
            "type": "item.completed",
            "item": {
                "type": "mcp_tool_call",
                "server": "oel",
                "tool": "oel.plot_evidence.v1",
                "status": "completed",
                "result": {"structured_content": payload},
            },
        }
    )
    claude_stdout = "\n".join(
        (
            json.dumps(
                {
                    "type": "assistant",
                    "message": {
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "toolu_plot",
                                "name": "mcp__oel__oel_plot_evidence_v1",
                            }
                        ]
                    },
                }
            ),
            json.dumps({"type": "user", "tool_use_result": {"structuredContent": payload}}),
            json.dumps({"type": "result", "permission_denials": []}),
        )
    )

    assert _parse_codex_plot_selection(codex_stdout) == payload
    assert _parse_claude_plot_selection(claude_stdout) == payload
    prompt = plot_selection_prompt(tmp_path, artifact_id="host_ric")
    assert "two-dimensional relative trajectory in RIC" in prompt
    assert '"recipe_id": "relative_position_ric_2d"' in prompt
    assert "host-native visualization tools" in prompt


@pytest.mark.external
@pytest.mark.skipif(not MCP_SDK_AVAILABLE, reason="optional MCP SDK profile is not installed")
def test_official_sdk_conformance_uses_real_repeated_stdio_processes() -> None:
    report = run_sdk_stdio(ROOT, Path(sys.executable))

    assert report["status"] == "passed"
    assert report["protocol_revision"] == "2026-07-28"
    assert report["tool_ids"] == list(PUBLIC_TOOL_IDS)
    assert report["error_handling"] == {
        "passed": True,
        "code": -32602,
        "message": "Handling metadata is required for this operation.",
        "data": {"recovery": {
            "code": "handling.review_required",
            "actor": "operator",
            "action": "Establish authoritative handling metadata for this input; do not relabel data to bypass policy.",
        }},
    }
    assert report["resources"]["passed"] is True
    assert report["resources"]["resource_uris"] == list(PUBLIC_RESOURCE_URIS)
    assert report["lifecycle"]["passed"] is True
    assert report["lifecycle"]["clean_shutdowns"] == 2
    assert all(row["tool_ids"] == list(PUBLIC_TOOL_IDS) for row in report["lifecycle"]["sessions"])
    assert all(row["resource_uris"] == list(PUBLIC_RESOURCE_URIS) for row in report["lifecycle"]["sessions"])


@pytest.mark.external
@pytest.mark.skipif(not MCP_SDK_AVAILABLE, reason="optional MCP SDK profile is not installed")
def test_public_workflow_acceptance_uses_real_stdio_and_no_provider_calls(tmp_path: Path) -> None:
    report = run_public_workflow_acceptance(
        root=ROOT,
        python_executable=Path(sys.executable),
        work_root=tmp_path / "acceptance",
    )

    assert report["status"] == "passed"
    assert report["transport"] == "stdio"
    assert report["provider_calls"] == 0
    assert [row["check_id"] for row in report["checks"]] == [
        "restricted_profile",
        "plan",
        "validate",
        "execute",
        "inspect",
        "query",
        "inspect_study",
        "replay_study",
        "compare_studies",
        "inspect_ccsds",
        "convert_frame_time",
        "fsw_describe",
        "fsw_scaffold",
        "fsw_inspect",
        "fsw_plan",
        "fsw_validate",
        "fsw_tests",
        "fsw_smoke",
        "fsw_verify_receipt",
        "export_run_product",
        "inspect_handoff",
        "materialize_onp_handoff",
        "emit_scenario_overlay",
        "materialize_scenario_patch",
        "compare_handoff",
        "agent_task",
        "assess_maneuver_readiness",
        "compare",
        "plot",
        "plan_review_plot",
        "render_review_plot",
        "plan_review_animation",
        "render_review_animation",
        "prepare_report_packet",
        "audit_report",
    ]
