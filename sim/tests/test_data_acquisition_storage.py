from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from sim.data_acquisition.contracts import AcquisitionError, Request, strict_json
from sim.data_acquisition.files import read_snapshot
from sim.data_acquisition.store import Store
from sim.tests.data_acquisition_helpers import GRAVITY, service


@pytest.mark.parametrize("change", [
    {"provider": "unknown"}, {"product": "upload"}, {"url": "https://example.com"}, {"max_bytes": True},
    {"max_records": 0}, {"max_requests": 17}, {"object_ids": [True]}, {"object_ids": [1000000000]},
    {"start": "now"}, {"start": "2024-01-01T00:00:00"}, {"published_before": "now"},
])
def test_requests_reject_unbounded_unknown_and_ambiguous_fields(change):
    from sim.tests.data_acquisition_helpers import history
    with pytest.raises(AcquisitionError):
        Request.from_dict(history(**change))


def test_request_resolves_timezone_and_normalizes_ids():
    from sim.tests.data_acquisition_helpers import history
    request = Request.from_dict(history(start="2024-01-09T17:00:00-07:00", object_ids=[2, 1, 2]))
    assert request.start == "2024-01-10T00:00:00Z" and request.object_ids == (1, 2)


@pytest.mark.parametrize("text", ['{"a":1,"a":2}', '{"x":NaN}', '{"x":Infinity}'])
def test_duplicate_and_nonfinite_json_rejected(text):
    with pytest.raises(AcquisitionError):
        strict_json(text)


def test_registration_is_an_immutable_snapshot_with_terms(tmp_path):
    svc = service(tmp_path)
    source = tmp_path / "licensed.gfc"
    source.write_bytes(GRAVITY)
    artifact = svc.register(source, kind="gravity", format="icgem", terms={"license": "local-use-only", "redistribution": "restricted"})
    source.write_text("changed")
    value, raw = svc.store.get(artifact["artifact_id"])
    assert raw == GRAVITY and value["terms"]["license"] == "local-use-only"
    assert value["inspection"]["compatible"]


@pytest.mark.parametrize("corrupt", ["artifact", "receipt"])
def test_corrupt_retained_evidence_fails_closed(tmp_path, corrupt):
    svc = service(tmp_path)
    path = tmp_path / "input.bin"
    path.write_bytes(b"original")
    value = svc.register(path, kind="opaque", format="opaque")
    if corrupt == "artifact":
        (svc.store.root / value["stored_path"]).write_bytes(b"modified")
    else:
        with sqlite3.connect(svc.store.root / "index.sqlite") as db:
            db.execute("UPDATE artifacts SET receipt=?", (b"{}",))
    with pytest.raises(AcquisitionError, match="identity changed"):
        svc.inspect(value["artifact_id"])


def test_symlinks_and_nonregular_inputs_are_rejected(tmp_path):
    target = tmp_path / "target"
    target.write_bytes(b"hi")
    link = tmp_path / "link"
    link.symlink_to(target)
    with pytest.raises(AcquisitionError):
        read_snapshot(link, 100)
    with pytest.raises(AcquisitionError):
        read_snapshot(tmp_path, 100)


def test_small_registration_budget_is_enforced(tmp_path):
    source = tmp_path / "large"
    source.write_bytes(b"12345")
    with pytest.raises(AcquisitionError, match="max_bytes"):
        service(tmp_path).register(source, kind="opaque", format="opaque", max_bytes=4)


def test_materialize_never_overwrites_and_retention_lists_references(tmp_path):
    svc = service(tmp_path)
    source = tmp_path / "raw"
    source.write_bytes(b"raw")
    value = svc.register(source, kind="opaque", format="opaque")
    destination = tmp_path / "inputs" / "raw"
    result = svc.materialize(value["artifact_id"], destination)
    assert destination.read_bytes() == b"raw" and not result["execution_authorized"]
    with pytest.raises(AcquisitionError, match="new destination"):
        svc.materialize(value["artifact_id"], destination)
    retention = svc.store.retention()
    assert retention["artifacts"][0]["references"] == [str(destination)]
    assert json.loads(Path(result["receipt_path"]).read_text())["source_sha256"] == value["sha256"]


def test_cross_process_lock_and_rate_ledger(tmp_path):
    store = Store(tmp_path / "data")
    other = Store(tmp_path / "data")
    with store.operation("account"):
        with pytest.raises(AcquisitionError, match="Another OEL"):
            with other.operation("account"):
                pass
    for i in range(20):
        store.reserve("account", "login", now=10000 + i * 0.1)
    with pytest.raises(AcquisitionError) as error:
        other.reserve("account", "login", now=10003)
    assert error.value.code == "RATE_LIMIT" and error.value.retry_after_s > 0
    other.reserve("different-account", "login", now=10003)


def test_retrieval_interval_and_provider_retry_after_are_persistent(tmp_path):
    store = Store(tmp_path)
    store.reserve("account", "gp", now=10000, retrieval_interval=3600)
    with pytest.raises(AcquisitionError):
        Store(tmp_path).reserve("account", "gp", now=10001, retrieval_interval=3600)
    store.backoff("account", 20000)
    with pytest.raises(AcquisitionError):
        store.reserve("account", "login", now=10002)


def test_connections_do_not_expose_secrets_and_account_aliases_share_scope(tmp_path):
    svc = service(tmp_path)
    svc.connections.add("one", "synthetic@example.invalid", "fake-password")
    svc.connections.add("two", "SYNTHETIC@example.invalid", "another-fake")
    assert svc.connections.describe("one")["scope"] == svc.connections.describe("two")["scope"]
    for path in (tmp_path / "connections").glob("*.json"):
        assert "fake-password" not in path.read_text() and "synthetic@" not in path.read_text()
    assert svc.connections.status("one")["network_used"] is False
    svc.connections.remove("one")
    assert not svc.connections.path("one").exists()


def test_no_plaintext_backend_fallback(monkeypatch):
    import sys
    from types import SimpleNamespace

    from sim.data_acquisition.credentials import SystemSecrets
    monkeypatch.setitem(sys.modules, "keyring", SimpleNamespace(get_keyring=lambda: object()))
    with pytest.raises(AcquisitionError) as error:
        SystemSecrets()
    assert error.value.code == "CREDENTIAL_BACKEND"
