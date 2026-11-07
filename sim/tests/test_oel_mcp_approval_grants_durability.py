from __future__ import annotations

import json
from pathlib import Path

from integrations.oel_mcp.approval_grants import DurableOperationLedger


def _finish(ledger: DurableOperationLedger, operation_id: str, artifact: Path) -> None:
    digest = "a" * 64
    ledger.begin(
        operation_id=operation_id,
        tool_id="tool.test.v1",
        request_digest=digest,
        authority_id="grant-1",
    )
    ledger.finish(
        operation_id=operation_id,
        tool_id="tool.test.v1",
        request_digest=digest,
        authority_id="grant-1",
        envelope={"status": "completed", "result": {"artifacts": [str(artifact)]}},
        status="completed",
    )


def test_ledger_creates_nested_private_root_and_persists_terminal_record(tmp_path: Path) -> None:
    ledger = DurableOperationLedger(tmp_path / "private" / "ledger")

    _finish(ledger, "nested-record", tmp_path / "missing-artifact.json")

    record = ledger.inspect("nested-record")
    assert record["status"] == "completed"
    assert record["artifact_verification"] == "unverified"
    assert json.loads((ledger._record_path("nested-record")).read_text())["status"] == "completed"


def test_artifact_hash_requires_an_explicit_authorized_root(tmp_path: Path) -> None:
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    artifact = tmp_path / "outside.json"
    artifact.write_text("stable\n")
    ledger = DurableOperationLedger(tmp_path / "private-ledger", artifact_roots=(allowed,))

    _finish(ledger, "outside-artifact", artifact)

    record = ledger.inspect("outside-artifact")
    assert record["artifact_verification"] == "unverified"
    assert record["artifact_identities"] == []
    replay = ledger.begin(
        operation_id="outside-artifact",
        tool_id="tool.test.v1",
        request_digest="a" * 64,
        authority_id="grant-1",
    ).replay_envelope
    assert replay is not None
    assert replay["status"] == "partial"


def test_artifact_hash_rejects_symlinked_ancestor_even_inside_authorized_root(tmp_path: Path) -> None:
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    artifact = outside / "artifact.json"
    artifact.write_text("stable\n")
    redirected = allowed / "redirected"
    redirected.symlink_to(outside, target_is_directory=True)
    ledger = DurableOperationLedger(tmp_path / "private-ledger", artifact_roots=(allowed,))

    _finish(ledger, "symlink-artifact", redirected / artifact.name)

    record = ledger.inspect("symlink-artifact")
    assert record["artifact_verification"] == "unverified"
    assert record["artifact_identities"] == []
