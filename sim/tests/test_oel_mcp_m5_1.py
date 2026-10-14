from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

from integrations.oel_mcp.diagnostics import default_host_launch, doctor_report, host_config
from integrations.oel_mcp.execution import ExecutionApprovalPolicy
from integrations.oel_mcp.public_handlers import PublicOELMCPHandlers
from integrations.oel_mcp.public_registry import public_contract_map
from integrations.oel_mcp.reporting import MAX_REPORT_SOURCE_BYTES

HANDLING = {"marking": "PUBLIC_TEST", "release_scope": "public"}
WRITE_APPROVAL = {"approval_id": "write-ok", "scope": "write"}


def _handlers(tmp_path: Path) -> PublicOELMCPHandlers:
    return PublicOELMCPHandlers(
        read_roots=(tmp_path,),
        write_roots=(tmp_path,),
        approval_policy=ExecutionApprovalPolicy(write_approval_ids=frozenset({"write-ok"})),
    )


def _completed_output(tmp_path: Path) -> Path:
    output = tmp_path / "run"
    review = output / "review"
    review.mkdir(parents=True)
    with sqlite3.connect(review / "run.sqlite") as conn:
        conn.execute(
            "CREATE TABLE run_metadata "
            "(scenario_name TEXT, duration_s REAL, dt_s REAL, samples INTEGER, oel_version TEXT, "
            "review_schema_version INTEGER)"
        )
        conn.execute("INSERT INTO run_metadata VALUES (?, ?, ?, ?, ?, ?)", ("report_fixture", 2.0, 1.0, 3, "test", 1))
    (output / "master_run_summary.json").write_text(
        json.dumps({"scenario_name": "report_fixture", "status": "completed"}) + "\n",
        encoding="utf-8",
    )
    return output


def test_report_packet_and_audit_are_provider_neutral_and_hash_bound(tmp_path: Path) -> None:
    output = _completed_output(tmp_path)
    handlers = _handlers(tmp_path)
    prepared = handlers.prepare_report_packet(
        source_output_dir=output,
        packet_output_dir=tmp_path / "packet",
        packet_id="report-fixture",
        query_names=["run_metadata"],
        max_rows=10,
        approval=WRITE_APPROVAL,
        handling=HANDLING,
    )

    assert prepared["status"] == "completed"
    assert prepared["result"]["provider_call_made"] is False
    packet_path = Path(prepared["result"]["packet_path"])
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    assert packet["artifact_summary"]["total"] == len(packet["artifacts"])
    assert packet["artifact_summary"]["existing"] == len(packet["artifacts"])
    assert packet["evidence_summary"]["ready_to_cite"] is True
    assert packet["query_evidence"]
    assert packet["execution_provenance"]["available"] is False
    artifact_id = packet["artifacts"][0]["artifact_id"]
    report = tmp_path / "agent_report.md"
    report.write_text(
        f"# Report\n\n## Evidence\n\nCompleted evidence. [evidence:{artifact_id}]\n\n"
        "## Limitations\n\nNo operational claim.\n",
        encoding="utf-8",
    )
    audited = handlers.audit_report(
        report_path=report,
        packet_path=packet_path,
        audit_output_dir=tmp_path / "audit",
        author="test_agent",
        model="none",
        approval=WRITE_APPROVAL,
        handling=HANDLING,
    )

    assert audited["status"] == "completed"
    assert audited["result"]["status"] == "passed"
    assert audited["result"]["semantic_claim_review_performed"] is False
    assert Path(audited["result"]["audit_json_path"]).is_file()

    (output / "master_run_summary.json").write_text('{"tampered": true}\n', encoding="utf-8")
    tampered = handlers.audit_report(
        report_path=report,
        packet_path=packet_path,
        audit_output_dir=tmp_path / "tampered-audit",
        author="test_agent",
        approval=WRITE_APPROVAL,
        handling=HANDLING,
    )
    assert tampered["status"] == "partial"
    assert tampered["result"]["status"] == "needs_review"
    assert tampered["result"]["checks"]["artifact_integrity_valid"] is False


def test_report_packet_marks_unknown_queries_and_missing_required_artifacts_partial(tmp_path: Path) -> None:
    handlers = _handlers(tmp_path)
    output = _completed_output(tmp_path)
    unknown = handlers.prepare_report_packet(
        source_output_dir=output,
        packet_output_dir=tmp_path / "unknown-packet",
        packet_id="unknown-query",
        query_names=["run_metadata", "not_a_saved_query"],
        max_rows=10,
        approval=WRITE_APPROVAL,
        handling=HANDLING,
    )
    unknown_packet = json.loads(Path(unknown["result"]["packet_path"]).read_text(encoding="utf-8"))

    assert unknown["status"] == "partial"
    assert unknown_packet["evidence_summary"]["review_queries_complete"] is False
    assert any(item["code"] == "review_queries_incomplete" for item in unknown_packet["failure_hints"])

    (output / "master_run_summary.json").unlink()
    missing = handlers.prepare_report_packet(
        source_output_dir=output,
        packet_output_dir=tmp_path / "missing-packet",
        packet_id="missing-summary",
        max_rows=10,
        approval=WRITE_APPROVAL,
        handling=HANDLING,
    )
    missing_packet = json.loads(Path(missing["result"]["packet_path"]).read_text(encoding="utf-8"))
    summary_row = next(row for row in missing_packet["artifacts"] if row["artifact_id"] == "run_summary")

    assert missing["status"] == "partial"
    assert missing_packet["required_artifacts_missing"] is True
    assert summary_row["exists"] is False
    assert "run_summary" in missing_packet["artifact_summary"]["required_missing"]


def test_report_packet_projects_oversized_source_summary_and_remains_auditable(tmp_path: Path) -> None:
    output = _completed_output(tmp_path)
    (output / "agent_evidence_packet.json").write_text(
        json.dumps({"evidence_summary": {"ready_to_cite": True, "unbounded": "x" * 1_900_000}}),
        encoding="utf-8",
    )
    prepared = _handlers(tmp_path).prepare_report_packet(
        source_output_dir=output,
        packet_output_dir=tmp_path / "bounded-packet",
        packet_id="bounded-source-summary",
        query_names=["run_metadata"],
        max_rows=10,
        approval=WRITE_APPROVAL,
        handling=HANDLING,
    )

    packet_path = Path(prepared["result"]["packet_path"])
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    assert packet_path.stat().st_size <= MAX_REPORT_SOURCE_BYTES
    assert "unbounded" not in packet["evidence_summary"]
    report = tmp_path / "bounded-report.md"
    report.write_text(
        f"# Report\n\n## Evidence\n\nBounded packet. [evidence:{packet['artifacts'][0]['artifact_id']}]\n\n"
        "## Limitations\n\nNo operational claim.\n",
        encoding="utf-8",
    )
    audited = _handlers(tmp_path).audit_report(
        report_path=report,
        packet_path=packet_path,
        audit_output_dir=tmp_path / "bounded-audit",
        author="test_agent",
        approval=WRITE_APPROVAL,
        handling=HANDLING,
    )
    assert audited["status"] == "completed"
    assert audited["result"]["status"] == "passed"


def test_report_audit_rejects_packet_over_call_budget_before_creating_output(tmp_path: Path) -> None:
    packet = tmp_path / "oversized-packet.json"
    packet.write_text('{"source_output_dir":"' + "x" * 200 + '"}', encoding="utf-8")

    audited = _handlers(tmp_path).audit_report(
        report_path=packet,
        packet_path=packet,
        max_packet_bytes=100,
        audit_output_dir=tmp_path / "oversized-audit",
        author="test_agent",
        approval=WRITE_APPROVAL,
        handling=HANDLING,
    )

    assert audited["status"] == "failed"
    assert audited["error"]["type"] == "ValueError"
    assert "file-size budget" in audited["error"]["message"]
    assert not (tmp_path / "oversized-audit").exists()
    contract = public_contract_map("public_local")["oel.audit_report.v1"]
    assert contract.input_schema["properties"]["max_packet_bytes"]["maximum"] == MAX_REPORT_SOURCE_BYTES
    assert contract.capability()["limits"]["max_packet_bytes"] == MAX_REPORT_SOURCE_BYTES


def test_report_audit_rejects_self_rehashed_packet_without_prepare_binding(tmp_path: Path) -> None:
    output = _completed_output(tmp_path)
    handlers = _handlers(tmp_path)
    prepared = handlers.prepare_report_packet(
        source_output_dir=output,
        packet_output_dir=tmp_path / "packet",
        packet_id="duplicate-fixture",
        query_names=["run_metadata"],
        max_rows=10,
        approval=WRITE_APPROVAL,
        handling=HANDLING,
    )
    original = json.loads(Path(prepared["result"]["packet_path"]).read_text(encoding="utf-8"))
    original["artifacts"].append(dict(original["artifacts"][0]))
    original["packet_sha256"] = hashlib.sha256(
        json.dumps(
            {key: value for key, value in original.items() if key != "packet_sha256"},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    duplicate_packet = tmp_path / "duplicate_packet.json"
    duplicate_packet.write_text(json.dumps(original, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = tmp_path / "report.md"
    report.write_text(
        f"# Report\n\n## Evidence\n\nCited. [evidence:{original['artifacts'][0]['artifact_id']}]\n\n"
        "## Limitations\n\nNo operational claim.\n",
        encoding="utf-8",
    )

    audited = handlers.audit_report(
        report_path=report,
        packet_path=duplicate_packet,
        audit_output_dir=tmp_path / "duplicate-audit",
        author="test_agent",
        approval=WRITE_APPROVAL,
        handling=HANDLING,
    )

    assert audited["status"] == "failed"
    assert audited["error"]["type"] == "ValueError"
    assert "preparation manifest" in audited["error"]["message"]
    assert not (tmp_path / "duplicate-audit").exists()


def test_report_tools_are_local_only_and_absent_from_frontier_profile(tmp_path: Path) -> None:
    public = _handlers(tmp_path)
    frontier = PublicOELMCPHandlers(
        profile="direct_frontier_restricted",
        read_roots=(tmp_path,),
        write_roots=(tmp_path,),
    )

    assert "oel.prepare_report_packet.v1" in public.contracts
    assert "oel.audit_report.v1" in public.contracts
    assert "oel.prepare_report_packet.v1" not in frontier.contracts
    assert "oel.audit_report.v1" not in frontier.contracts


def test_doctor_and_host_config_make_disabled_effects_explicit(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("OEL_MCP_READ_ROOTS", str(tmp_path))
    monkeypatch.setenv("OEL_MCP_WRITE_ROOTS", str(tmp_path))
    monkeypatch.delenv("OEL_MCP_WRITE_APPROVAL_IDS", raising=False)
    monkeypatch.delenv("OEL_MCP_EXECUTION_APPROVAL_IDS", raising=False)
    monkeypatch.delenv("OEL_MCP_TRUST_APPROVAL_IDS", raising=False)

    report = doctor_report(profile="public_local", adapter="sdk")
    registry = next(row for row in report["checks"] if row["check_id"] == "registry")
    approvals = next(row for row in report["checks"] if row["check_id"] == "operator_approvals")

    assert report["status"] == "ready_with_disabled_effects"
    assert registry["detail"]["tool_count"] == 37
    assert approvals["passed"] is False
    assert approvals["required"] is False
    assert report["network_listener"] is False
    from sim.project_version import source_project_version

    assert report["oel_version"] == source_project_version()
    assert report["oel_version_source"] == "source_pyproject"
    launch = next(row for row in report["checks"] if row["check_id"] == "host_launch")
    assert launch["passed"] is True

    codex = host_config(
        host="codex",
        command="python",
        command_args=("-m", "integrations.oel_mcp"),
        cwd=tmp_path,
        profile="public_local",
    )
    claude = json.loads(
        host_config(
            host="claude",
            command="python",
            command_args=("-m", "integrations.oel_mcp"),
            cwd=tmp_path,
            profile="public_local",
        )
    )
    assert "[mcp_servers.oel]" in codex
    assert 'args = ["-m","integrations.oel_mcp"]' in codex
    assert claude["mcpServers"]["oel"]["args"] == ["-m", "integrations.oel_mcp"]

    command, args, source = default_host_launch()
    assert command
    assert source in {"installed_console_entrypoint", "python_module_fallback"}
    if source == "python_module_fallback":
        assert args == ("-m", "integrations.oel_mcp")
