from __future__ import annotations

import importlib.util
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier, Event

import numpy as np
import pytest

from integrations.oel_mcp.base_handlers import BaseOELMCPHandlers
from integrations.oel_mcp.cr3bp_handlers import call_cr3bp_tool
from integrations.oel_mcp.cr3bp_registry import (
    CR3BP_PLAN_TOOL_ID,
    CR3BP_RENDER_TOOL_ID,
    CR3BP_TOOL_CONTRACTS,
    cr3bp_contract_map,
)
from integrations.oel_mcp.execution import ExecutionApprovalPolicy
from integrations.oel_mcp.public_handlers import PublicOELMCPHandlers
from integrations.oel_mcp.sdk_protocol import build_sdk_server
from sim.dynamics.orbit.cr3bp import cr3bp_halo_seed_state_km_s, propagate_cr3bp_state
from sim.review.cr3bp import STATE_COLUMNS
from sim.review.evidence_capsule import create_evidence_capsule

ROOT = Path(__file__).resolve().parents[2]
HANDLING = {"marking": "PUBLIC", "release_scope": "public"}
WRITE_APPROVAL = {"approval_id": "cr3bp-test-write", "scope": "write"}


class _CR3BPHarness(BaseOELMCPHandlers):
    def __init__(self, tmp_path: Path) -> None:
        super().__init__(
            profile="public_local",
            contracts=cr3bp_contract_map("public_local"),
            read_roots=(ROOT, tmp_path),
            write_roots=(tmp_path,),
            approval_policy=ExecutionApprovalPolicy(
                write_approval_ids=frozenset({"cr3bp-test-write"}),
            ),
        )

    def _call_contract(self, contract, arguments, *, cancel_event=None, progress=None):
        result = call_cr3bp_tool(self, contract, arguments, cancel_event=cancel_event, progress=progress)
        if result is None:
            raise PermissionError("CR3BP harness received an unknown contract.")
        return result


@pytest.fixture
def cr3bp_output(tmp_path: Path) -> Path:
    output = tmp_path / "cr3bp"
    review = output / "review"
    review.mkdir(parents=True)
    (output / "index.md").write_text("# CR3BP review\n", encoding="utf-8")
    db = review / "run.sqlite"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE object_state_frame (object_id TEXT, state_frame TEXT)")
        conn.execute("INSERT INTO object_state_frame VALUES ('vehicle', 'cr3bp_rotating')")
        conn.execute(
            "CREATE TABLE object_propagation (object_id TEXT, propagator_family TEXT, propagator_name TEXT)"
        )
        conn.execute("INSERT INTO object_propagation VALUES ('vehicle', 'CR3BP', 'earth_moon CR3BP')")
        conn.execute(
            "CREATE TABLE object_state (object_id TEXT, time_s REAL, "
            + ", ".join(column + " REAL" for column in STATE_COLUMNS)
            + ")"
        )
        state = cr3bp_halo_seed_state_km_s(family="l1_northern_large")
        for time_s in np.arange(0.0, 1800.0, 300.0):
            conn.execute(
                "INSERT INTO object_state VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                ["vehicle", float(time_s), *state],
            )
            state = propagate_cr3bp_state(state, 300.0, float(time_s))
    return output


def _arguments(output: Path, **overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "output_dir": str(output),
        "object_id": "vehicle",
        "kind": "trajectory",
        "axes": "rotating",
        "origin": "barycenter",
        "plane": "xy",
        "resolution": 31,
        "frames": 6,
        "fps": 3,
        "style": "oel_dark",
        "handling": HANDLING,
    }
    values.update(overrides)
    return values


def test_cr3bp_registry_is_public_local_experimental_and_bounded() -> None:
    assert tuple(contract.tool_id for contract in CR3BP_TOOL_CONTRACTS) == (
        CR3BP_PLAN_TOOL_ID,
        CR3BP_RENDER_TOOL_ID,
    )
    assert set(cr3bp_contract_map("direct_frontier_restricted")) == set()
    plan, render = CR3BP_TOOL_CONTRACTS
    assert plan.maturity == render.maturity == "experimental"
    assert plan.deployment_profiles == render.deployment_profiles == ("public_local",)
    assert render.writes is True
    assert render.limits["max_rows"] == 100_000
    assert render.limits["max_grid_resolution"] == 301
    assert render.limits["max_frames"] == 600
    assert render.input_schema["properties"]["cr3bp_plan_id"]["minLength"] == 1


def test_plan_is_bound_to_evidence_identity_and_spec(cr3bp_output: Path, tmp_path: Path) -> None:
    harness = _CR3BPHarness(tmp_path)
    response = harness.call(CR3BP_PLAN_TOOL_ID, _arguments(cr3bp_output))

    assert response["status"] == "completed"
    result = response["result"]
    assert result["status"] == "planned"
    assert result["render_authorized"] is False
    assert result["row_count"] == 6
    assert result["cr3bp_plan_id"].startswith("oel-cr3bp-render-plan-v1:")
    assert result["evidence_identity"]["sha256"]
    assert result["warnings"][-1].endswith("not J2000.")


def test_plan_accepts_noncanonical_content_bound_capsule(cr3bp_output: Path, tmp_path: Path) -> None:
    database = cr3bp_output / "review" / "run.sqlite"
    entry = create_evidence_capsule(database, remove_original=True)
    compressed = database.parent / str(entry["compressed_path"])
    remapped = database.parent / "cr3bp-review-payload.sqlite.gz"
    compressed.rename(remapped)
    manifest_path = database.parent / "evidence_capsule.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"][0]["compressed_path"] = remapped.name
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")

    harness = _CR3BPHarness(tmp_path)
    response = harness.call(CR3BP_PLAN_TOOL_ID, _arguments(cr3bp_output))

    assert response["status"] == "completed"
    assert response["result"]["row_count"] == 6
    assert response["result"]["evidence_identity"]["storage"] == "sqlite_gzip_capsule"


def test_plan_rejects_oversized_noncanonical_capsule_payload(
    monkeypatch, cr3bp_output: Path, tmp_path: Path
) -> None:
    import integrations.oel_mcp.cr3bp_handlers as cr3bp_handlers

    database = cr3bp_output / "review" / "run.sqlite"
    entry = create_evidence_capsule(database, remove_original=True)
    compressed = database.parent / str(entry["compressed_path"])
    remapped = database.parent / "oversized-cr3bp-payload.gz"
    compressed.rename(remapped)
    manifest_path = database.parent / "evidence_capsule.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"][0]["compressed_path"] = remapped.name
    # Keep the declared hydrated size within the mock budget so this test
    # exercises the actual noncanonical compressed file-size guard.
    manifest["artifacts"][0]["original_bytes"] = 1
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")
    monkeypatch.setattr(cr3bp_handlers, "MAX_CR3BP_REVIEW_STORE_BYTES", remapped.stat().st_size - 1)

    response = _CR3BPHarness(tmp_path).call(CR3BP_PLAN_TOOL_ID, _arguments(cr3bp_output))

    assert response["status"] == "failed"
    assert "file-size budget" in response["error"]["message"]


def test_plan_rejects_escaping_noncanonical_capsule_reference(cr3bp_output: Path, tmp_path: Path) -> None:
    database = cr3bp_output / "review" / "run.sqlite"
    create_evidence_capsule(database, remove_original=True)
    manifest_path = database.parent / "evidence_capsule.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"][0]["compressed_path"] = "../escape.sqlite.gz"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")

    response = _CR3BPHarness(tmp_path).call(CR3BP_PLAN_TOOL_ID, _arguments(cr3bp_output))

    assert response["status"] == "failed"
    assert "bounded compressed-size declaration" in response["error"]["message"]


def test_public_handler_central_hook_exposes_cr3bp_plan(cr3bp_output: Path, tmp_path: Path) -> None:
    handlers = PublicOELMCPHandlers(
        read_roots=(ROOT, tmp_path),
        write_roots=(tmp_path,),
        approval_policy=ExecutionApprovalPolicy(write_approval_ids=frozenset({"cr3bp-test-write"})),
    )
    response = handlers.plan_cr3bp_render(**_arguments(cr3bp_output))
    assert response["status"] == "completed"
    assert response["tool_id"] == CR3BP_PLAN_TOOL_ID
    assert response["result"]["render_authorized"] is False


def test_render_writes_static_artifact_receipt_manifest_and_index(cr3bp_output: Path, tmp_path: Path) -> None:
    harness = _CR3BPHarness(tmp_path)
    arguments = _arguments(cr3bp_output)
    plan = harness.call(CR3BP_PLAN_TOOL_ID, arguments)
    rendered = harness.call(
        CR3BP_RENDER_TOOL_ID,
        {
            **arguments,
            "cr3bp_plan_id": plan["result"]["cr3bp_plan_id"],
            "approval": WRITE_APPROVAL,
        },
    )

    assert rendered["status"] == "completed"
    result = rendered["result"]
    figure = Path(result["artifact"]["path"])
    receipt = Path(result["artifact"]["files"]["receipt"])
    manifest = Path(result["manifest_path"])
    assert figure.is_file() and figure.parent.name == "figures"
    assert receipt.is_file() and receipt.parent == figure.parent
    assert manifest.is_file() and manifest.parent.name == "mcp_cr3bp"
    assert (cr3bp_output / "review" / "generated_artifacts.json").is_file()
    index = (cr3bp_output / "index.md").read_text(encoding="utf-8")
    assert "OEL_REVIEW_GENERATED_ARTIFACTS_START" in index
    receipt_payload = json.loads(receipt.read_text(encoding="utf-8"))
    assert receipt_payload["review_store"]["sha256"] == plan["result"]["evidence_identity"]["sha256"]
    assert receipt_payload["inertial_alignment"] == "CR3BP model axes, not J2000"


def test_render_writes_gif_and_contact_sheet(cr3bp_output: Path, tmp_path: Path) -> None:
    harness = _CR3BPHarness(tmp_path)
    arguments = _arguments(cr3bp_output, movie_format="gif", frames=6, fps=3, axes="inertial", origin="p2")
    plan = harness.call(CR3BP_PLAN_TOOL_ID, arguments)
    rendered = harness.call(
        CR3BP_RENDER_TOOL_ID,
        {
            **arguments,
            "cr3bp_plan_id": plan["result"]["cr3bp_plan_id"],
            "approval": WRITE_APPROVAL,
        },
    )

    assert rendered["status"] == "completed"
    files = rendered["result"]["artifact"]["files"]
    movie = Path(files["movie"])
    contact_sheet = Path(files["contact_sheet"])
    quality = Path(files["movie_quality_receipt"])
    assert movie.is_file() and movie.suffix == ".gif"
    assert contact_sheet.is_file() and contact_sheet.stat().st_size > 0
    assert quality.is_file()
    from PIL import Image

    durations_ms: list[int] = []
    decoded_frames = 0
    with Image.open(movie) as animated:
        for frame_index in range(animated.n_frames):
            animated.seek(frame_index)
            decoded = animated.convert("RGBA")
            assert decoded.tobytes()
            decoded_frames += 1
            durations_ms.append(int(animated.info.get("duration", 0)))
        assert animated.n_frames == 6
    assert decoded_frames == 6
    assert all(duration > 0 for duration in durations_ms)
    duration_s = sum(durations_ms) / 1000.0
    assert duration_s == pytest.approx(6 / 3, abs=0.5)
    movie.with_name(movie.stem + ".gif_validation.json").write_text(
        json.dumps(
            {
                "movie": str(movie),
                "decoded_frames": decoded_frames,
                "expected_frames": 6,
                "durations_ms": durations_ms,
                "duration_s": duration_s,
                "expected_duration_s": 6 / 3,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    receipt = json.loads(Path(files["receipt"]).read_text(encoding="utf-8"))
    assert receipt["movie_qa"]["automated_status"] == "passed"
    assert receipt["inertial_alignment"] == "CR3BP model axes, not J2000"


def test_concurrent_render_reservation_keeps_one_terminal_receipt(
    monkeypatch, cr3bp_output: Path, tmp_path: Path
) -> None:
    import integrations.oel_mcp.cr3bp_handlers as cr3bp_handlers

    harness = _CR3BPHarness(tmp_path)
    arguments = _arguments(cr3bp_output)
    plan = harness.call(CR3BP_PLAN_TOOL_ID, arguments)
    render_arguments = {
        **arguments,
        "cr3bp_plan_id": plan["result"]["cr3bp_plan_id"],
        "approval": WRITE_APPROVAL,
    }
    barrier = Barrier(2)
    original_reservation = cr3bp_handlers._operation_manifest

    def synchronized_reservation(*args, **kwargs):
        barrier.wait(timeout=10)
        return original_reservation(*args, **kwargs)

    monkeypatch.setattr(cr3bp_handlers, "_operation_manifest", synchronized_reservation)

    def render_once() -> dict[str, object]:
        return harness.call(CR3BP_RENDER_TOOL_ID, render_arguments)

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _index: render_once(), range(2)))

    assert sorted(result["status"] for result in results) == ["completed", "failed"]
    failed = next(result for result in results if result["status"] == "failed")
    assert failed["error"]["type"] == "FileExistsError"
    manifests = list((cr3bp_output / "review" / "mcp_cr3bp").glob("*.manifest.json"))
    assert len(manifests) == 1
    receipt = json.loads(manifests[0].read_text(encoding="utf-8"))
    assert receipt["status"] == "completed"


def test_render_rejects_stale_plan_and_spec_mismatch(cr3bp_output: Path, tmp_path: Path) -> None:
    harness = _CR3BPHarness(tmp_path)
    arguments = _arguments(cr3bp_output)
    plan = harness.call(CR3BP_PLAN_TOOL_ID, arguments)
    stale = harness.call(
        CR3BP_RENDER_TOOL_ID,
        {
            **arguments,
            "resolution": 32,
            "cr3bp_plan_id": plan["result"]["cr3bp_plan_id"],
            "approval": WRITE_APPROVAL,
        },
    )
    assert stale["status"] == "failed"
    assert stale["error"]["type"] == "ValueError"
    assert "stale" in stale["error"]["message"]

    with sqlite3.connect(cr3bp_output / "review" / "run.sqlite") as conn:
        conn.execute("UPDATE object_state SET pos_x_eci_km = pos_x_eci_km + 0.001 WHERE object_id = 'vehicle'")
    changed = harness.call(
        CR3BP_RENDER_TOOL_ID,
        {
            **arguments,
            "cr3bp_plan_id": plan["result"]["cr3bp_plan_id"],
            "approval": WRITE_APPROVAL,
        },
    )
    assert changed["status"] == "failed"
    assert "stale" in changed["error"]["message"]


def test_cancelled_render_leaves_terminal_cancelled_receipt(cr3bp_output: Path, tmp_path: Path) -> None:
    harness = _CR3BPHarness(tmp_path)
    arguments = _arguments(cr3bp_output)
    plan = harness.call(CR3BP_PLAN_TOOL_ID, arguments)
    cancel_event = Event()
    cancel_event.set()
    response = harness.call(
        CR3BP_RENDER_TOOL_ID,
        {
            **arguments,
            "cr3bp_plan_id": plan["result"]["cr3bp_plan_id"],
            "approval": WRITE_APPROVAL,
        },
        cancel_event=cancel_event,
    )
    assert response["status"] == "failed"
    assert response["error"]["type"] == "MCPExecutionCancelled"
    manifests = list((cr3bp_output / "review" / "mcp_cr3bp").glob("*.manifest.json"))
    assert len(manifests) == 1
    payload = json.loads(manifests[0].read_text(encoding="utf-8"))
    assert payload["status"] == "cancelled"
    assert payload["cancelled"] is True
    assert payload["artifacts_complete"] is False


def test_source_drift_terminalizes_receipt_as_failed(monkeypatch, cr3bp_output: Path, tmp_path: Path) -> None:
    import integrations.oel_mcp.cr3bp_handlers as cr3bp_handlers

    harness = _CR3BPHarness(tmp_path)
    arguments = _arguments(cr3bp_output)
    plan = harness.call(CR3BP_PLAN_TOOL_ID, arguments)

    def drifted_renderer(workspace, object_id, **_kwargs):
        folder = workspace.review_dir / "figures"
        folder.mkdir(parents=True, exist_ok=True)
        figure = folder / "cr3bp_drift.png"
        receipt = folder / "cr3bp_drift.json"
        figure.write_bytes(b"placeholder")
        receipt.write_text("{}\n", encoding="utf-8")
        with sqlite3.connect(workspace.logical_db_path) as conn:
            conn.execute("UPDATE object_state SET pos_x_eci_km = pos_x_eci_km + 0.001 WHERE object_id = 'vehicle'")
        return {"figure": str(figure), "receipt": str(receipt), "diagnostics": {}}

    monkeypatch.setattr(cr3bp_handlers, "render_cr3bp_research", drifted_renderer)
    response = harness.call(
        CR3BP_RENDER_TOOL_ID,
        {
            **arguments,
            "cr3bp_plan_id": plan["result"]["cr3bp_plan_id"],
            "approval": WRITE_APPROVAL,
        },
    )
    assert response["status"] == "failed"
    assert response["error"]["type"] == "ValueError"
    manifests = list((cr3bp_output / "review" / "mcp_cr3bp").glob("*.manifest.json"))
    assert len(manifests) == 1
    payload = json.loads(manifests[0].read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert payload["artifacts_complete"] is False
    assert payload["error_type"] == "EvidenceChanged"


def test_render_rejects_artifact_directory_symlink_escape(cr3bp_output: Path, tmp_path: Path) -> None:
    outside = tmp_path.parent / "cr3bp-mcp-outside"
    outside.mkdir()
    figures = cr3bp_output / "review" / "figures"
    figures.symlink_to(outside, target_is_directory=True)
    try:
        harness = _CR3BPHarness(tmp_path)
        arguments = _arguments(cr3bp_output)
        plan = harness.call(CR3BP_PLAN_TOOL_ID, arguments)
        response = harness.call(
            CR3BP_RENDER_TOOL_ID,
            {
                **arguments,
                "cr3bp_plan_id": plan["result"]["cr3bp_plan_id"],
                "approval": WRITE_APPROVAL,
            },
        )
        assert response["status"] == "failed"
        assert "not authorized" in response["error"]["message"]
        assert not list(outside.iterdir())
    finally:
        figures.unlink(missing_ok=True)
        outside.rmdir()


def test_plan_rejects_oversized_generated_artifact_sidecar(
    monkeypatch, cr3bp_output: Path, tmp_path: Path
) -> None:
    import integrations.oel_mcp.cr3bp_handlers as cr3bp_handlers

    generated = cr3bp_output / "review" / "generated_artifacts.json"
    generated.write_bytes(b"012345678")
    monkeypatch.setattr(cr3bp_handlers, "MAX_MANIFEST_BYTES", 8)

    response = _CR3BPHarness(tmp_path).call(CR3BP_PLAN_TOOL_ID, _arguments(cr3bp_output))

    assert response["status"] == "failed"
    assert "file-size budget" in response["error"]["message"]


@pytest.mark.skipif(importlib.util.find_spec("mcp") is None, reason="optional MCP SDK is not installed")
def test_official_sdk_lists_cr3bp_and_preserves_structured_content(cr3bp_output: Path, tmp_path: Path) -> None:
    import anyio
    from mcp import Client

    harness = _CR3BPHarness(tmp_path)
    arguments = _arguments(cr3bp_output)
    plan = harness.call(CR3BP_PLAN_TOOL_ID, arguments)
    render_args = {
        **arguments,
        "cr3bp_plan_id": plan["result"]["cr3bp_plan_id"],
        "approval": WRITE_APPROVAL,
    }

    async def exercise() -> tuple[list[str], dict[str, object], list[str]]:
        async with Client(build_sdk_server(harness), mode="auto", cache=None) as client:
            tools = await client.list_tools()
            names = [str(tool.name) for tool in tools.tools]
            result = await client.call_tool(CR3BP_RENDER_TOOL_ID, render_args)
            text_blocks = [block.text for block in result.content if getattr(block, "type", "") == "text"]
            assert text_blocks
            return names, json.loads(text_blocks[0]), [str(block.type) for block in result.content]

    names, payload, content_types = anyio.run(exercise)
    assert CR3BP_PLAN_TOOL_ID in names
    assert CR3BP_RENDER_TOOL_ID in names
    assert content_types == ["text", "image"]
    assert payload["tool_id"] == CR3BP_RENDER_TOOL_ID
    assert payload["status"] == "completed"
    assert Path(payload["result"]["artifact"]["path"]).is_file()
