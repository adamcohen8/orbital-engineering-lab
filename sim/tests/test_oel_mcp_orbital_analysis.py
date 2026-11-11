from __future__ import annotations

import json
from pathlib import Path

from integrations.oel_mcp.public_handlers import PublicOELMCPHandlers
from integrations.oel_mcp.public_registry import public_contracts_for_profile
from integrations.oel_mcp.resources import build_public_resource_catalog
from sim.tests.test_study_lifecycle import _build

ROOT = Path(__file__).resolve().parents[2]
PUBLIC_HANDLING = {"marking": "PUBLIC_TEST", "release_scope": "public"}
OEM_REFERENCE = ROOT / "sim/interchange/examples/oel_earth_eme2000_utc_v3.oem"


def _handlers(tmp_path: Path) -> PublicOELMCPHandlers:
    return PublicOELMCPHandlers(
        read_roots=(ROOT, tmp_path),
        write_roots=(tmp_path,),
    )


def test_analysis_workflow_resource_routes_typed_problems_without_expanding_execution_surface() -> None:
    resources = build_public_resource_catalog(
        profile="public_local",
        tool_contracts=public_contracts_for_profile("public_local"),
    )
    workflow = next(resource for resource in resources if resource.contract.uri == "oel://analysis/workflows/v1")

    assert '"workflow_id": "study_lifecycle"' in workflow.text
    payload = json.loads(workflow.text)
    source_study = next(
        item for item in payload["workflows"]
        if item["workflow_id"] == "source_schedule_power_study"
    )
    assert source_study["evidence"].startswith("oel.source_schedule_power_study.v1")
    assert source_study["mcp_tools"] == []
    orbit_bound = next(
        item for item in payload["workflows"] if item["workflow_id"] == "orbit_bound_source_study"
    )
    assert orbit_bound["evidence"].startswith("oel.orbit_bound_source_study.v1")
    assert orbit_bound["mcp_tools"] == []
    constellation = next(item for item in payload["workflows"] if item["workflow_id"] == "constellation_design")
    assert constellation["evidence"] == "oel.constellation_design_evidence.v1"
    assert constellation["authoritative_replay"] == "python -m sim.constellation_design replay"
    assert constellation["mcp_tools"] == []
    by_id = {item["workflow_id"]: item for item in payload["workflows"]}
    assert by_id["constellation_optimization"]["interfaces"] == ["python -m sim.pro_constellation_optimization"]
    assert "python -m sim.pro_slr_od" in by_id["local_tracking_and_slr"]["interfaces"]
    assert "oel.run_scale_refresh_to_onp.v1" in by_id["scale_catalog"]["mcp_tools"]
    assert "pro_escalation" not in by_id["collection_analysis"]
    assert payload["routing"]["hosted_pro_package_validator"] == "oel.hosted.validate_package.v1"
    assert payload["routing"]["hosted_pro_package_validation_is_not_execution"] is True
    assert payload["cross_cutting_pro_escalations"] == []
    assert '"mcp_tools": []' in workflow.text
    assert "scenario YAML or typed orbital-analysis problem" in workflow.text
    assert "sim.intent_hypothesis" not in workflow.text
    assert "agents/pro" not in workflow.text
    assert "/Users/" not in workflow.text


def test_study_inspect_replay_and_compare_are_read_only_and_content_bound(tmp_path: Path) -> None:
    artifacts = _build(tmp_path / "study")
    handlers = _handlers(tmp_path)

    inspected = handlers.inspect_study(bundle_dir=artifacts.output_dir, handling=PUBLIC_HANDLING)
    replayed = handlers.replay_study(bundle_dir=artifacts.output_dir, handling=PUBLIC_HANDLING)
    compared = handlers.compare_studies(
        left_bundle_dir=artifacts.output_dir,
        right_bundle_dir=artifacts.output_dir,
        handling=PUBLIC_HANDLING,
    )

    assert inspected["status"] == "completed"
    assert inspected["result"]["status"] == "verified"
    assert replayed["result"]["replay_status"] == "identity_verified"
    assert compared["result"]["same_bundle"] is True
    assert all(not payload["effects"]["writes"] for payload in (inspected, replayed, compared))


def test_ccsds_and_frame_time_adapters_return_bounded_nonexecuting_receipts(tmp_path: Path) -> None:
    handlers = _handlers(tmp_path)

    ccsds = handlers.inspect_ccsds(
        path=OEM_REFERENCE,
        product_kind="oem",
        handling=PUBLIC_HANDLING,
    )
    epoch = handlers.convert_frame_time(
        operation="convert_epoch",
        epoch="2024-01-01T00:00:00Z",
        from_scale="UTC",
        to_scale="TAI",
        handling=PUBLIC_HANDLING,
    )
    state = handlers.convert_frame_time(
        operation="transform_state",
        epoch="2024-01-01T00:00:00Z",
        time_scale="UTC",
        source_frame="EME2000",
        target_frame="EME2000",
        position_km=[7000.0, 0.0, 0.0],
        velocity_km_s=[0.0, 7.5, 0.0],
        handling=PUBLIC_HANDLING,
    )

    assert ccsds["result"]["inspection"]["valid_oem"] is True
    assert ccsds["result"]["execution_occurred"] is False
    assert epoch["result"]["result"]["scale"] == "TAI"
    assert state["result"]["result"]["position_km"] == [7000.0, 0.0, 0.0]
    assert epoch["effects"]["executes"] is False
    assert state["effects"]["writes"] is False
