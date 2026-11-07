from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from integrations.oel_mcp.approval_grants import (
    DurableOperationLedger,
    OperationLedgerError,
    UnknownOperationOutcome,
)


def _request(operation_id: str = "approved-op") -> dict[str, str]:
    return {
        "operation_id": operation_id,
        "tool_id": "test.tool.v1",
        "request_digest": "a" * 64,
        "authority_id": "reviewed-authority",
    }


def _record_name(operation_id: str = "approved-op") -> str:
    return hashlib.sha256(operation_id.encode()).hexdigest() + ".json"


def test_replaced_ledger_root_cannot_erase_unknown_operation_outcome(tmp_path: Path) -> None:
    root = tmp_path / "ledger"
    ledger = DurableOperationLedger(root)
    try:
        ledger.begin(**_request())
        original = tmp_path / "retained-ledger"
        root.rename(original)
        root.mkdir()
        fake = json.loads((original / _record_name()).read_text())
        fake.update(status="completed", envelope={"attacker": True}, artifact_verification="none")
        (root / _record_name()).write_text(json.dumps(fake))

        with pytest.raises(UnknownOperationOutcome):
            ledger.begin(**_request())
        assert ledger.inspect("approved-op")["status"] == "begun"

        ledger.finish(**_request(), envelope={"result": "verified"}, status="completed")
        assert json.loads((original / _record_name()).read_text())["envelope"] == {"result": "verified"}
        assert json.loads((root / _record_name()).read_text())["envelope"] == {"attacker": True}
        replay = ledger.begin(**_request())
        assert replay.replay_envelope == {"result": "verified"}
    finally:
        ledger.close()


def test_ledger_root_symlink_replacement_cannot_redirect_new_reservation(tmp_path: Path) -> None:
    root = tmp_path / "ledger"
    ledger = DurableOperationLedger(root)
    try:
        original = tmp_path / "original"
        root.rename(original)
        outside = tmp_path / "outside"
        outside.mkdir()
        root.symlink_to(outside, target_is_directory=True)

        ledger.begin(**_request())
        assert (original / _record_name()).is_file()
        assert list(outside.iterdir()) == []
    finally:
        ledger.close()


def test_ledger_rejects_oversized_and_non_object_records(tmp_path: Path) -> None:
    root = tmp_path / "ledger"
    ledger = DurableOperationLedger(root)
    try:
        path = root / _record_name()
        path.write_bytes(b" " * 4_000_001)
        with pytest.raises(OperationLedgerError):
            ledger.inspect("approved-op")
        path.write_text("[]")
        with pytest.raises(OperationLedgerError):
            ledger.inspect("approved-op")
    finally:
        ledger.close()


def test_closing_ledger_revokes_its_read_capability(tmp_path: Path) -> None:
    ledger = DurableOperationLedger(tmp_path / "ledger")
    ledger.begin(**_request())
    ledger.close()
    ledger.close()
    with pytest.raises(OperationLedgerError):
        ledger.inspect("approved-op")
    with pytest.raises(OperationLedgerError):
        ledger.begin(**_request())
    with pytest.raises(OperationLedgerError):
        ledger.finish(**_request(), envelope={"result": "never-written"}, status="completed")
