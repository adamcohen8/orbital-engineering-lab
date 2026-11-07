from __future__ import annotations

import json
from pathlib import Path

import pytest

from integrations.oel_mcp.approval_grants import (
    ApprovalGrantStore,
    DurableOperationLedger,
    OperationConflictError,
    UnknownOperationOutcome,
)
from integrations.oel_mcp.execution import ExecutionApprovalPolicy
from integrations.oel_mcp.protocol import OELMCPServer
from integrations.oel_mcp.recovery import reconciliation_error_data, recovery_details


def test_catalog_rejects_unbounded_entries_but_legacy_standing_ids_remain_valid(monkeypatch) -> None:
    entry = {"grant_id": "legacy", "scopes": ["write"]}
    with pytest.raises(ValueError, match="bounded"):
        ApprovalGrantStore([entry])

    monkeypatch.setenv("OEL_MCP_APPROVAL_GRANTS_JSON", json.dumps([entry]))
    with pytest.raises(ValueError, match="bounded"):
        ApprovalGrantStore.from_environment()

    policy = ExecutionApprovalPolicy(
        write_approval_ids=frozenset({"legacy"}),
        grant_store=ApprovalGrantStore(),
    )
    decision = policy.authorize(
        {"approval_id": "legacy", "scope": "write"},
        executes=False,
        tool_id="tool.test.v1",
        arguments={},
    )
    assert decision.grant_id is None
    assert decision.bounded is False


@pytest.mark.parametrize(
    ("error_type", "recovery_code"),
    [
        (OperationConflictError, "operation.conflict"),
        (UnknownOperationOutcome, "operation.unknown_outcome"),
    ],
)
def test_reconciliation_errors_are_structured_and_non_retryable(error_type, recovery_code) -> None:
    error = error_type("safe diagnostic")
    assert recovery_details(error) == {
        "code": recovery_code,
        "actor": "operator",
        "action": (
            "Review the recorded operation identity and reconcile it before retrying; do not select a new authority."
            if recovery_code == "operation.conflict"
            else "Inspect and reconcile the reserved operation before retrying; do not rerun it under a new authority."
        ),
    }
    assert reconciliation_error_data(error) == {
        "reconciliation_required": True,
        "retryable": False,
        "recovery": recovery_details(error),
    }


class _RaisingHandlers:
    contracts = {}

    def __init__(self, error: Exception) -> None:
        self.error = error

    def call(self, *_args, **_kwargs):
        raise self.error


@pytest.mark.parametrize("error_type", [OperationConflictError, UnknownOperationOutcome])
def test_stdio_transport_exposes_reconciliation_required_error(error_type) -> None:
    response = OELMCPServer(_RaisingHandlers(error_type("safe diagnostic"))).dispatch(
        {
            "jsonrpc": "2.0",
            "id": "request-1",
            "method": "tools/call",
            "params": {"name": "tool.test.v1", "arguments": {}},
        }
    )
    assert response is not None
    assert response["error"]["code"] == -32602
    assert response["error"]["data"]["reconciliation_required"] is True
    assert response["error"]["data"]["retryable"] is False


@pytest.mark.parametrize(
    "artifact_key",
    [
        "brief_path",
        "audit_json_path",
        "audit_markdown_path",
        "mcp_manifest_path",
        "state_product_path",
        "report_path",
    ],
)
def test_ledger_replay_verifies_each_exposed_result_artifact_path(
    tmp_path: Path,
    artifact_key: str,
) -> None:
    artifact = tmp_path / f"{artifact_key}.json"
    artifact.write_text("stable\n")
    ledger = DurableOperationLedger(
        tmp_path / "ledger",
        artifact_roots=(tmp_path,),
    )
    request_digest = "a" * 64
    operation_id = f"op-{artifact_key}"
    ledger.begin(
        operation_id=operation_id,
        tool_id="tool.test.v1",
        request_digest=request_digest,
        authority_id="grant-1",
    )
    ledger.finish(
        operation_id=operation_id,
        tool_id="tool.test.v1",
        request_digest=request_digest,
        authority_id="grant-1",
        envelope={"status": "completed", "result": {artifact_key: str(artifact)}},
        status="completed",
    )
    record = ledger.inspect(operation_id)
    assert [item["path"] for item in record["artifact_identities"]] == [str(artifact)]

    artifact.write_text("changed\n")
    replay = ledger.begin(
        operation_id=operation_id,
        tool_id="tool.test.v1",
        request_digest=request_digest,
        authority_id="grant-1",
    ).replay_envelope
    assert replay is not None
    assert replay["status"] == "partial"
    assert replay["evidence"]["complete"] is False
    assert replay["error"]["type"] == "StoredArtifactStale"
