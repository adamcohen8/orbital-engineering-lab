from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Event
from unittest.mock import patch

import pytest
import yaml

from integrations.oel_mcp.approval_grants import ApprovalGrant, ApprovalGrantStore, canonical_request_digest
from integrations.oel_mcp.execution import ExecutionApprovalPolicy, bounded_config_path_policy
from integrations.oel_mcp.policy import MCPPathPolicy
from integrations.oel_mcp.public_handlers import PublicOELMCPHandlers
from integrations.oel_mcp.public_registry import PUBLIC_TOOL_CONTRACTS
from integrations.oel_mcp.sdk_protocol import build_sdk_server
from sim.security.config_paths import ConfigPathSecurityError

ROOT = Path(__file__).resolve().parents[2]
HANDLING = {"marking": "PUBLIC", "release_scope": "public"}
EXECUTION_APPROVAL = {"approval_id": "test-execution", "scope": "execute"}
WRITE_APPROVAL = {"approval_id": "test-write", "scope": "write"}
TRUST_APPROVAL = {"approval_id": "test-trust", "scope": "trust"}
M4_IDS = (
    "oel.plan_run.v1",
    "oel.validate_scenario.v1",
    "oel.run_scenario.v1",
    "oel.compare_runs.v1",
    "oel.plot_evidence.v1",
    "oel.plan_review_plot.v1",
    "oel.render_review_plot.v2",
    "oel.run_agent_task.v1",
)
MCP_SDK_AVAILABLE = importlib.util.find_spec("mcp") is not None
PRO_HANDLERS_AVAILABLE = (ROOT / "integrations" / "oel_mcp" / "pro_handlers.py").is_file()
if PRO_HANDLERS_AVAILABLE:
    from integrations.oel_mcp.pro_handlers import ProOELMCPHandlers


def _handlers(tmp_path: Path) -> PublicOELMCPHandlers:
    return PublicOELMCPHandlers(
        read_roots=(ROOT, tmp_path),
        write_roots=(tmp_path,),
        approval_policy=ExecutionApprovalPolicy(
            write_approval_ids=frozenset({"test-write"}),
            execution_approval_ids=frozenset({"test-execution"}),
            trust_approval_ids=frozenset({"test-trust"}),
        ),
    )


def _scenario_arguments(tmp_path: Path, name: str) -> dict[str, object]:
    return {
        "config_path": str(ROOT / "configs" / "automation_smoke.yaml"),
        "output_dir": str(tmp_path / name),
        "resource_profile": "laptop-safe",
        "handling": HANDLING,
    }


def test_m4_registry_effects_and_deployment_views(tmp_path: Path) -> None:
    contracts = {contract.tool_id: contract for contract in PUBLIC_TOOL_CONTRACTS}
    public = _handlers(tmp_path)
    frontier = PublicOELMCPHandlers(
        profile="direct_frontier_restricted",
        read_roots=(ROOT, tmp_path),
        write_roots=(tmp_path,),
    )

    assert tuple(tool_id for tool_id in public.contracts if tool_id in M4_IDS) == M4_IDS
    assert not set(M4_IDS) & set(frontier.contracts)
    if PRO_HANDLERS_AVAILABLE:
        pro = ProOELMCPHandlers(
            profile="pro_local",
            read_roots=(ROOT, tmp_path),
            write_roots=(tmp_path,),
        )
        assert set(M4_IDS) <= set(pro.contracts)
    assert contracts["oel.run_scenario.v1"].capability()["effects"] == {
        "reads": True,
        "writes": True,
        "executes": True,
        "external_communication": False,
    }
    assert contracts["oel.plot_evidence.v1"].mcp_definition()["annotations"] == {
        "readOnlyHint": False,
        "destructiveHint": True,
        "idempotentHint": False,
        "openWorldHint": False,
    }


def test_mcp_config_policy_does_not_union_repository_or_config_roots(tmp_path: Path) -> None:
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    source = allowed / "scenario.yaml"
    source.write_text("simulator: {}\n", encoding="utf-8")
    policy = MCPPathPolicy.configured(read_roots=(allowed,), write_roots=(tmp_path,))
    bounded = bounded_config_path_policy(config_path=source, path_policy=policy)

    assert bounded.read_roots == (allowed.resolve(),)
    with pytest.raises(ConfigPathSecurityError):
        bounded.resolve_input_file(ROOT / "README.md", purpose="nested input")
    prompt = tmp_path / "outside-prompt.txt"
    prompt.write_text("prompt", encoding="utf-8")
    with pytest.raises(ConfigPathSecurityError):
        bounded.resolve_ai_prompt_file(prompt, purpose="AI prompt")


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS compatibility aliases are platform-specific")
@pytest.mark.parametrize(
    ("alias_root", "canonical_root"),
    ((Path("/tmp"), Path("/private/tmp")), (Path("/var"), Path("/private/var"))),
)
def test_mcp_write_policy_accepts_mac_compatibility_alias(
    alias_root: Path,
    canonical_root: Path,
) -> None:
    configured_root = alias_root / "oel-mcp-policy-alias-regression"
    policy = MCPPathPolicy.configured(write_roots=(configured_root,))

    assert policy.resolve_write(configured_root / "new-output") == (
        canonical_root / "oel-mcp-policy-alias-regression" / "new-output"
    )


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS compatibility aliases are platform-specific")
def test_mcp_write_policy_accepts_mac_compatibility_alias_without_following_descendants(
    tmp_path: Path,
) -> None:
    """Configured /private roots and lexical /var aliases must agree safely."""

    resolved_root = tmp_path.resolve()
    private_var = Path("/private/var")
    try:
        relative_root = resolved_root.relative_to(private_var)
    except ValueError:
        pytest.skip("pytest temporary roots are not under the macOS /private/var alias")

    alias_root = Path("/var") / relative_root
    policy = MCPPathPolicy.configured(write_roots=(alias_root,))
    assert policy.resolve_write(alias_root / "new-output") == resolved_root / "new-output"

    outside = tmp_path.parent / "mcp-alias-outside"
    outside.mkdir()
    redirected = tmp_path / "redirected"
    redirected.symlink_to(outside, target_is_directory=True)
    with pytest.raises(PermissionError):
        policy.resolve_write(alias_root / "redirected" / "artifact.json")


def test_plan_validate_and_run_require_external_approval_and_bound_identity(tmp_path: Path) -> None:
    handlers = _handlers(tmp_path)
    arguments = _scenario_arguments(tmp_path, "run")

    plan = handlers.plan_run(**arguments)
    safe_only = handlers.validate_scenario(**arguments, trust_plugins=False)
    with pytest.raises(PermissionError, match="trust approval"):
        handlers.validate_scenario(**arguments, trust_plugins=True)
    trusted = handlers.validate_scenario(**arguments, trust_plugins=True, trust_approval=TRUST_APPROVAL)

    assert plan["status"] == "completed"
    assert plan["result"]["execution_authorized"] is False
    assert safe_only["status"] == "partial"
    assert safe_only["result"]["identity"]["validation_id"] == ""
    assert "trust approval" in safe_only["result"]["next_step"]
    assert "resource envelope" not in safe_only["result"]["next_step"]
    assert trusted["status"] == "completed"
    validation_id = trusted["result"]["identity"]["validation_id"]
    assert validation_id.startswith("oel-m4-validation-v1:")
    assert trusted["result"]["execution_authorized"] is False


    disabled = PublicOELMCPHandlers(read_roots=(ROOT, tmp_path), write_roots=(tmp_path,))
    try:
        disabled.run_scenario(
            **arguments,
            validation_id=validation_id,
            trust_approval=TRUST_APPROVAL,
            approval=EXECUTION_APPROVAL,
        )
    except PermissionError as exc:
        assert "operator approval policy" in str(exc)
    else:  # pragma: no cover - explicit safety assertion
        raise AssertionError("Execution unexpectedly bypassed the server approval policy.")

    with pytest.raises(ValueError, match="trust_approval"):
        handlers.run_scenario(
            **arguments,
            validation_id=validation_id,
            approval=EXECUTION_APPROVAL,
        )
    assert not Path(str(arguments["output_dir"])).exists()

    mismatch = handlers.run_scenario(
        **arguments,
        validation_id="oel-m4-validation-v1:wrong",
        trust_approval=TRUST_APPROVAL,
        approval=EXECUTION_APPROVAL,
    )
    assert mismatch["status"] == "failed"
    assert not Path(str(arguments["output_dir"])).exists()

    completed = handlers.run_scenario(
        **arguments,
        validation_id=validation_id,
        trust_approval=TRUST_APPROVAL,
        approval=EXECUTION_APPROVAL,
    )
    manifest = json.loads((Path(str(arguments["output_dir"])) / "mcp_execution_manifest.json").read_text())
    assert completed["status"] == "completed"
    assert completed["effects"]["executes"] is True
    assert manifest["status"] == "completed"
    assert manifest["validation_id"] == validation_id
    assert manifest["artifacts_complete"] is True
    assert (Path(str(arguments["output_dir"])) / "review" / "run.sqlite").is_file()

    inspected = handlers.inspect_run(output_dir=arguments["output_dir"], handling=HANDLING)
    provenance = inspected["result"]["execution_provenance"]
    assert provenance["available"] is True
    assert provenance["status"] == "completed"
    assert provenance["validation_id"] == validation_id
    assert provenance["artifact_count"] == len(manifest["artifacts"])
    assert provenance["artifacts_complete"] is True
    assert inspected["result"]["evidence_summary"]["mcp_execution_complete"] is True
    assert inspected["result"]["freshness"]["content_bound_execution_recorded"] is True

    # A source mutation after inspection must be caught by the second
    # content-bound manifest check before packet serialization.
    from integrations.oel_mcp import reporting as reporting_module

    original_inspect_output = reporting_module.inspect_output

    def inspect_then_tamper(*args, **kwargs):
        result = original_inspect_output(*args, **kwargs)
        summary_path = Path(str(arguments["output_dir"])) / "master_run_summary.json"
        summary_path.write_text(summary_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        return result

    with patch.object(reporting_module, "inspect_output", side_effect=inspect_then_tamper):
        raced_packet = handlers.prepare_report_packet(
            source_output_dir=arguments["output_dir"],
            packet_output_dir=tmp_path / "raced-packet",
            packet_id="raced-source",
            query_names=[],
            max_rows=10,
            handling=HANDLING,
            approval=WRITE_APPROVAL,
        )
    assert raced_packet["status"] == "failed"
    assert "artifact records do not match" in raced_packet["error"]["message"]

    # Report preparation must revalidate the original execution manifest;
    # changing a source artifact after inspection cannot produce a trusted
    # packet merely because the manifest still says it is complete.
    summary_path = Path(str(arguments["output_dir"])) / "master_run_summary.json"
    summary_path.write_text(summary_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    tampered_packet = handlers.prepare_report_packet(
        source_output_dir=arguments["output_dir"],
        packet_output_dir=tmp_path / "tampered-packet",
        packet_id="tampered-source",
        query_names=[],
        max_rows=10,
        handling=HANDLING,
        approval=WRITE_APPROVAL,
    )
    assert tampered_packet["status"] == "failed"
    assert "artifact records do not match" in tampered_packet["error"]["message"]


def test_bounded_resource_ceiling_rejects_before_public_run_output_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A host-computed ceiling must stop an effect before its output root exists."""

    validation_handlers = _handlers(tmp_path)
    scenario = _scenario_arguments(tmp_path, "bounded-run")
    validated = validation_handlers.validate_scenario(
        **scenario,
        trust_plugins=True,
        trust_approval=TRUST_APPROVAL,
    )
    validation_id = validated["result"]["identity"]["validation_id"]
    assert validated["status"] == "completed"

    approval = {"approval_id": "bounded-execute", "scope": "execute"}
    run_arguments = {
        **scenario,
        "validation_id": validation_id,
        "trust_approval": TRUST_APPROVAL,
        "approval": approval,
    }
    grant = ApprovalGrant(
        grant_id="bounded-execute",
        scopes=frozenset({"execute"}),
        tool_ids=frozenset({"oel.run_scenario.v1"}),
        reviewed_arguments_sha256=canonical_request_digest("oel.run_scenario.v1", run_arguments),
        resource_ceiling={"runs": 0},
        expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
    )
    ledger_root = tmp_path.parent / "bounded-resource-private-ledger"
    monkeypatch.setenv("OEL_MCP_OPERATION_LEDGER_ROOT", str(ledger_root))
    bounded_handlers = PublicOELMCPHandlers(
        read_roots=(ROOT, tmp_path),
        write_roots=(tmp_path,),
        approval_policy=ExecutionApprovalPolicy(
            trust_approval_ids=frozenset({TRUST_APPROVAL["approval_id"]}),
            grant_store=ApprovalGrantStore([grant]),
        ),
    )

    result = bounded_handlers.run_scenario(**run_arguments)

    assert result["status"] == "failed"
    assert result["error"]["type"] == "ApprovalGrantMismatch"
    assert "runs ceiling" in result["error"]["message"]
    assert not Path(str(scenario["output_dir"])).exists()


def test_m5_2_completed_run_product_export_and_inspection(tmp_path: Path) -> None:
    handlers = _handlers(tmp_path)
    arguments = _scenario_arguments(tmp_path, "glue-source")
    validated = handlers.validate_scenario(
        **arguments, trust_plugins=True, trust_approval=TRUST_APPROVAL
    )
    run = handlers.run_scenario(
        **arguments,
        validation_id=validated["result"]["identity"]["validation_id"],
        trust_approval=TRUST_APPROVAL,
        approval=EXECUTION_APPROVAL,
    )
    assert run["status"] == "completed"

    product_path = tmp_path / "products" / "final-state.json"
    exported = handlers.export_run_product(
        completed_run=tmp_path / "glue-source",
        product_kind="completed_run_state",
        object_id="target",
        selector="final",
        epoch_jd_utc=2461254.5,
        output_path=product_path,
        handling=HANDLING,
        approval=WRITE_APPROVAL,
    )
    assert exported["status"] == "completed"
    assert exported["result"]["execution_occurred"] is False
    assert Path(exported["result"]["operation_manifest_path"]).is_file()

    inspected = handlers.inspect_handoff(path=product_path, handling=HANDLING)
    assert inspected["status"] == "completed"
    assert inspected["result"]["product_kind"] == "oel.completed_run_state"
    assert "materialize_onp_handoff" in inspected["result"]["supported_next_actions"]
@pytest.mark.parametrize(
    "unsafe_fragment",
    [
        {"outputs": {"ai_report": {"enabled": True, "provider": "ollama"}}},
        {
            "simulator": {
                "dynamics": {
                    "orbit": {
                        "spherical_harmonics": {
                            "enabled": True,
                            "degree": 8,
                            "order": 8,
                            "source": "egm96",
                            "allow_download": True,
                        }
                    }
                }
            }
        },
    ],
)
def test_public_mcp_rejects_provider_and_download_capable_scenarios(
    tmp_path: Path, unsafe_fragment: dict[str, object]
) -> None:
    source = yaml.safe_load((ROOT / "configs" / "automation_smoke.yaml").read_text(encoding="utf-8"))
    for key, value in unsafe_fragment.items():
        if isinstance(value, dict) and isinstance(source.get(key), dict):
            source[key].update(value)
        else:
            source[key] = value
    config = tmp_path / "unsafe.yaml"
    config.write_text(yaml.safe_dump(source, sort_keys=False), encoding="utf-8")
    arguments = {
        **_scenario_arguments(tmp_path, "unsafe_run"),
        "config_path": str(config),
    }

    result = _handlers(tmp_path).plan_run(**arguments)

    assert result["status"] == "failed"
    assert "MCP sealed execution policy failed" in result["error"]["message"]


def test_unset_write_roots_do_not_inherit_read_authority(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("OEL_MCP_READ_ROOTS", str(tmp_path))
    monkeypatch.delenv("OEL_MCP_WRITE_ROOTS", raising=False)
    monkeypatch.delenv("OEL_MCP_ALLOWED_ROOTS", raising=False)

    policy = MCPPathPolicy.configured()

    assert policy.read_roots == (tmp_path.resolve(),)
    assert policy.write_roots == ()
    with pytest.raises(PermissionError, match="not authorized"):
        policy.resolve_write(tmp_path / "output")

    monkeypatch.setenv("OEL_MCP_ALLOWED_ROOTS", str(tmp_path))
    legacy_policy = MCPPathPolicy.configured()
    assert legacy_policy.read_roots == (tmp_path.resolve(),)
    assert legacy_policy.write_roots == (tmp_path.resolve(),)


def test_cancelled_run_is_partial_and_writes_unambiguous_manifest(tmp_path: Path) -> None:
    handlers = _handlers(tmp_path)
    arguments = _scenario_arguments(tmp_path, "cancelled")
    validation = handlers.validate_scenario(**arguments, trust_plugins=True, trust_approval=TRUST_APPROVAL)
    cancel_event = Event()
    cancel_event.set()

    result = handlers.call(
        "oel.run_scenario.v1",
        {
            **arguments,
            "validation_id": validation["result"]["identity"]["validation_id"],
            "trust_approval": TRUST_APPROVAL,
            "approval": EXECUTION_APPROVAL,
        },
        cancel_event=cancel_event,
    )
    manifest = json.loads((Path(str(arguments["output_dir"])) / "mcp_execution_manifest.json").read_text())

    assert result["status"] == "partial"
    assert result["result"]["status"] == "cancelled"
    assert result["evidence"]["complete"] is False
    assert manifest["status"] == "cancelled"
    assert manifest["cancelled"] is True
    assert manifest["artifacts_complete"] is False


def test_supported_task_comparison_and_plot_flow(tmp_path: Path) -> None:
    handlers = _handlers(tmp_path)
    output = tmp_path / "task"
    task = handlers.run_agent_task(
        recipe_id="quickstart_review",
        output_dir=output,
        resource_profile="laptop-safe",
        make_plots=False,
        max_rows=25,
        approval=EXECUTION_APPROVAL,
        handling=HANDLING,
    )
    comparison = handlers.compare_runs(
        base_output_dir=output,
        candidate_output_dir=output,
        metric_names=["final_range_km", "closest_approach_km"],
        max_rows=25,
        handling=HANDLING,
    )
    plot = handlers.plot_evidence(
        output_dir=output,
        recipe_id="relative_range",
        artifact_id="m4_relative_range",
        style="oel_light",
        format="png",
        approval=WRITE_APPROVAL,
        handling=HANDLING,
    )
    plot_arguments = {
        "output_dir": output,
        "sql": "SELECT time_s, range_km FROM relative_state ORDER BY time_s",
        "x_column": "time_s",
        "y_columns": ["range_km"],
        "plot_type": "line",
        "artifact_id": "m4_typed_relative_range",
        "style": "oel_light",
        "format": "png",
        "handling": HANDLING,
    }
    plan = handlers.plan_review_plot(**plot_arguments)
    rendered = handlers.render_review_plot(
        **plot_arguments,
        plot_plan_id=plan["result"]["plot_plan_id"],
        approval=WRITE_APPROVAL,
    )

    assert task["status"] == "completed"
    assert task["result"]["evidence_summary"]["ready_to_cite"] is True
    assert comparison["status"] == "completed"
    assert comparison["result"]["deltas"] == {"closest_approach_km": 0.0, "final_range_km": 0.0}
    assert plot["status"] == "completed"
    assert Path(plot["result"]["artifact"]["path"]).is_file()
    assert Path(plot["result"]["manifest_path"]).is_file()
    assert plan["status"] == "completed"
    assert plan["result"]["render_authorized"] is False
    assert rendered["status"] == "completed"
    assert rendered["result"]["artifact"]["qa"]["automated_status"] == "passed"
    assert rendered["result"]["artifact"]["qa"]["visual_qa_status"] == "pending_agent_review"


@pytest.mark.skipif(not MCP_SDK_AVAILABLE, reason="optional MCP SDK profile is not installed")
def test_sdk_plot_call_returns_image_content(tmp_path: Path) -> None:
    import anyio
    from mcp import Client

    handlers = _handlers(tmp_path)
    output = tmp_path / "task"
    handlers.run_agent_task(
        recipe_id="quickstart_review",
        output_dir=output,
        resource_profile="laptop-safe",
        make_plots=False,
        max_rows=25,
        approval=EXECUTION_APPROVAL,
        handling=HANDLING,
    )

    async def exercise() -> list[str]:
        async with Client(build_sdk_server(handlers), mode="auto", cache=None) as client:
            result = await client.call_tool(
                "oel.plot_evidence.v1",
                {
                    "output_dir": str(output),
                    "recipe_id": "relative_position_ric_2d",
                    "artifact_id": "sdk_ric_plot",
                    "style": "oel_light",
                    "format": "png",
                    "approval": WRITE_APPROVAL,
                    "handling": HANDLING,
                },
            )
            return [str(getattr(item, "type", "")) for item in result.content]

    assert anyio.run(exercise) == ["text", "image"]


def test_partial_plot_writes_incomplete_manifest(tmp_path: Path) -> None:
    output = tmp_path / "run"
    review = output / "review"
    review.mkdir(parents=True)
    (review / "run.sqlite").write_bytes(b"")
    handlers = _handlers(tmp_path)
    failed_artifact = {
        "status": "error",
        "path_exists": False,
        "truncated": False,
    }

    with patch("integrations.oel_mcp.public_handlers.create_plot", return_value=failed_artifact):
        plot = handlers.plot_evidence(
            output_dir=output,
            recipe_id="relative_range",
            artifact_id="failed_plot",
            approval=WRITE_APPROVAL,
            handling=HANDLING,
        )

    assert plot["status"] == "partial", json.dumps(plot, indent=2)
    manifest = json.loads((review / "mcp_plots" / "failed_plot.manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "partial"
    assert manifest["artifacts"] == []
    assert manifest["artifacts_complete"] is False


def test_unapproved_or_private_task_recipe_is_rejected(tmp_path: Path) -> None:
    handlers = _handlers(tmp_path)
    result = handlers.run_agent_task(
        recipe_id="dynamics_od_smoke",
        output_dir=tmp_path / "private-task",
        resource_profile="laptop-safe",
        approval=EXECUTION_APPROVAL,
        handling=HANDLING,
    )

    assert result["status"] == "failed"
    assert result["result"] is None
    assert "Only supported public scenario-run" in result["error"]["message"]


def test_failed_agent_task_packet_remains_failed_with_diagnostics(tmp_path: Path) -> None:
    handlers = _handlers(tmp_path)
    output = tmp_path / "failed-task"
    failed_packet = {
        "status": "failed",
        "evidence_summary": {"ready_to_cite": False},
        "packet_path": "",
        "artifacts": [],
        "failure_hints": [{"code": "simulation_failed", "next_step": "Inspect the deterministic error."}],
    }

    with patch("integrations.oel_mcp.public_handlers.run_recipe", return_value=failed_packet):
        result = handlers.run_agent_task(
            recipe_id="quickstart_review",
            output_dir=output,
            resource_profile="laptop-safe",
            approval=EXECUTION_APPROVAL,
            handling=HANDLING,
        )

    assert result["status"] == "failed"
    assert result["error"] is None
    assert result["result"]["status"] == "failed"
    assert result["result"]["failure_hints"] == failed_packet["failure_hints"]
    manifest = json.loads((output / "mcp_execution_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["artifacts_complete"] is False


@pytest.mark.skipif(not MCP_SDK_AVAILABLE, reason="optional MCP SDK profile is not installed")
def test_sdk_cancellation_reaches_execution_callback_and_finalizes_manifest(tmp_path: Path) -> None:
    import anyio
    from mcp import Client

    source = yaml.safe_load((ROOT / "configs" / "automation_smoke.yaml").read_text())
    source["simulator"]["duration_s"] = 100_000.0
    source["simulator"]["dt_s"] = 1.0
    config_path = tmp_path / "long.yaml"
    config_path.write_text(yaml.safe_dump(source, sort_keys=False), encoding="utf-8")
    handlers = _handlers(tmp_path)
    output = tmp_path / "sdk-cancelled"
    arguments = {
        "config_path": str(config_path),
        "output_dir": str(output),
        "resource_profile": "laptop-safe",
        "handling": HANDLING,
    }
    validation = handlers.validate_scenario(**arguments, trust_plugins=True, trust_approval=TRUST_APPROVAL)
    call_arguments = {
        **arguments,
        "validation_id": validation["result"]["identity"]["validation_id"],
        "trust_approval": TRUST_APPROVAL,
        "approval": EXECUTION_APPROVAL,
    }

    async def exercise() -> None:
        async with Client(build_sdk_server(handlers), mode="auto", cache=None) as client:
            scope = anyio.CancelScope()
            done = anyio.Event()

            async def invoke() -> None:
                try:
                    with scope:
                        await client.call_tool("oel.run_scenario.v1", call_arguments)
                finally:
                    done.set()

            async with anyio.create_task_group() as task_group:
                task_group.start_soon(invoke)
                with anyio.fail_after(5):
                    while not (output / "mcp_execution_manifest.json").is_file():
                        await anyio.sleep(0.01)
                scope.cancel()
                await done.wait()
                with anyio.fail_after(5):
                    while True:
                        manifest = json.loads((output / "mcp_execution_manifest.json").read_text())
                        if manifest["status"] != "running":
                            break
                        await anyio.sleep(0.01)

    anyio.run(exercise)
    manifest = json.loads((output / "mcp_execution_manifest.json").read_text())
    assert manifest["status"] == "cancelled"
    assert manifest["cancelled"] is True
    assert manifest["artifacts_complete"] is False
