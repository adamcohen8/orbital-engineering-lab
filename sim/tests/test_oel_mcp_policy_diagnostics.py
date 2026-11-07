"""Configuration diagnostics must describe grants without disclosing bindings."""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone

from integrations.oel_mcp.diagnostics import doctor_report


def _configure(monkeypatch, tmp_path, *, expires_at: datetime) -> None:
    data = tmp_path / "data"
    data.mkdir()
    for key in (
        "OEL_MCP_WRITE_APPROVAL_IDS", "OEL_MCP_EXECUTION_APPROVAL_IDS",
        "OEL_MCP_TRUST_APPROVAL_IDS", "OEL_MCP_APPROVAL_GRANTS",
        "OEL_MCP_REVOKED_APPROVAL_GRANTS", "OEL_MCP_OPERATION_LEDGER_ROOT",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("OEL_MCP_READ_ROOTS", str(data))
    monkeypatch.setenv("OEL_MCP_WRITE_ROOTS", str(data))
    monkeypatch.setenv("OEL_MCP_APPROVAL_GRANTS_JSON", json.dumps([{
        "grant_id": "reviewed-binding-keep-private",
        "scopes": ["write"],
        "tool_ids": ["oel.plot_evidence.v1"],
        "reviewed_arguments_sha256": "a" * 64,
        "expires_at": expires_at.isoformat(),
    }]))


def test_doctor_counts_bounded_grants_without_disclosing_bindings(monkeypatch, tmp_path):
    _configure(monkeypatch, tmp_path, expires_at=datetime.now(timezone.utc) + timedelta(hours=1))
    report = doctor_report(profile="public_local", adapter="legacy")
    checks = {row["check_id"]: row for row in report["checks"]}
    assert checks["operator_approvals"]["passed"] is True
    detail = checks["operator_approvals"]["detail"]
    assert detail["bounded_grants"] == detail["active_bounded_grants"] == 1
    assert checks["bounded_operation_ledger"]["passed"] is False
    serialized = json.dumps(report)
    assert "reviewed-binding-keep-private" not in serialized
    assert "a" * 64 not in serialized


def test_expired_and_revoked_grants_do_not_report_active_authority(monkeypatch, tmp_path):
    _configure(monkeypatch, tmp_path, expires_at=datetime.now(timezone.utc) - timedelta(seconds=1))
    report = doctor_report(profile="public_local", adapter="legacy")
    checks = {row["check_id"]: row for row in report["checks"]}
    assert checks["operator_approvals"]["passed"] is False
    assert checks["operator_approvals"]["detail"]["active_bounded_grants"] == 0
    configured = json.loads(os.environ["OEL_MCP_APPROVAL_GRANTS_JSON"])
    configured[0]["expires_at"] = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
    monkeypatch.setenv("OEL_MCP_APPROVAL_GRANTS_JSON", json.dumps(configured))
    monkeypatch.setenv("OEL_MCP_REVOKED_APPROVAL_GRANTS", "reviewed-binding-keep-private")
    report = doctor_report(profile="public_local", adapter="legacy")
    check = next(row for row in report["checks"] if row["check_id"] == "operator_approvals")
    assert check["passed"] is False
    assert check["detail"]["active_bounded_grants"] == 0
