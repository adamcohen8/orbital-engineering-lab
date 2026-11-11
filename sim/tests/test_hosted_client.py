from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from sim.hosted_client import (
    HostedClient,
    append_event,
    import_result_transfer,
    link_profile,
    read_ledger,
    route_planning_result,
)
from sim.hosted_client.result_import import _safe_relative_path
from sim.hosted_client.transport import HostedServiceError
from sim.study_planning import canonical_sha256, discovery_capability_catalog, preflight_study_plan
from sim.study_planning.config_validation import validate_config_path

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("path", ["C:/escape", "C:escape", "dir/C:/escape", "report.json:stream", "../escape", "\\\\host\\share", "a//b", "report.json/"])
def test_result_paths_are_safe_on_windows_and_posix(path: str) -> None:
    with pytest.raises(ValueError, match="Unsafe path"):
        _safe_relative_path(path)


@pytest.mark.parametrize("authorization", ["false", "true", 1, [], {"approved": True}, None])
def test_client_requires_literal_authorization(authorization) -> None:
    client = object.__new__(HostedClient)
    with pytest.raises(ValueError, match="Explicit user authorization"):
        client.approve_and_submit(
            {}, idempotency_key="key", authorized_by="user", user_authorized=authorization, confirmation="unused"
        )


@pytest.mark.parametrize("job,tenant", [("other-job", "tenant"), ("job", "other-tenant")])
def test_pull_results_rejects_mismatched_identity_before_import(tmp_path: Path, job: str, tenant: str) -> None:
    from types import SimpleNamespace

    client = object.__new__(HostedClient)
    client.token = {}
    client.profile = {"tenant_id": "tenant"}
    client.transport = SimpleNamespace(call=lambda *_: {"job_id": job, "tenant_id": tenant})
    destination = tmp_path / "result"
    with pytest.raises(ValueError, match="requested job and tenant"):
        client.pull_results("job", destination)
    assert not destination.exists()


def _fixture(name: str) -> dict:
    path = Path(__file__).parent / "fixtures" / "study_planning" / name
    return json.loads(path.read_text(encoding="utf-8"))


def _local_planning_result() -> dict:
    receipt, normalized = validate_config_path(
        "scenario_main",
        ROOT / "configs" / "automation_smoke.yaml",
        workspace_root=ROOT,
    )
    assert normalized is not None
    return preflight_study_plan(
        _fixture("passive_request.json"),
        _fixture("passive_plan.json"),
        catalog=discovery_capability_catalog(),
        normalized_configs={"scenario_main": normalized},
        config_receipts=[receipt],
    )


def test_route_decision_distinguishes_free_local_and_unsupported() -> None:
    local = route_planning_result(_local_planning_result())
    assert local["disposition"] == "execution_options_available"
    assert local["oel_execution_amount_usd"] == 0
    assert local["execution_options"] == {
        "local": {
            "available": True,
            "oel_execution_amount_usd": 0,
            "quote_required": False,
        },
        "hosted": {
            "available": False,
            "oel_execution_amount_usd": None,
            "quote_required": True,
        },
    }
    assert local["recommendation"]["option"] == "local"
    assert local["recommendation"]["requirement"] is False
    assert local["user_approval_required"] is False
    assert local["next_action"] == "execute_locally"
    assert local["hosted_access"]["public_registration_available"] is False
    assert "access is not publicly available" in local["message"]
    invited = route_planning_result(_local_planning_result(), hosted_profile_configured=True)
    assert invited["execution_options"]["hosted"]["available"] is True
    assert invited["user_approval_required"] is True
    assert invited["hosted_access"]["execution_authorized"] is False

    unsupported_plan = _fixture("passive_plan.json")
    unsupported_plan["operations"][0]["capability_id"] = "oel.pro.not-implemented.v1"
    unsupported = preflight_study_plan(
        _fixture("passive_request.json"),
        unsupported_plan,
        catalog=discovery_capability_catalog(),
    )
    decision = route_planning_result(unsupported)
    assert decision["disposition"] == "not_eligible"
    assert decision["feedback_preview_available"] is True


def test_profile_transport_and_ledger_are_content_bound(tmp_path: Path) -> None:
    service = tmp_path / "fake_service.py"
    service.write_text(
        """import json, sys
request = json.loads(sys.stdin.read())
if request[\"action\"] == \"capabilities\":
    result = {\"schema\": \"oel.hosted_capability_offer.v1\", \"offer_sha256\": \"a\" * 64, \"capabilities\": []}
    print(json.dumps({\"outcome\": \"ok\", \"result\": result}))
else:
    print(json.dumps({\"outcome\": \"error\", \"error\": {\"type\": \"ValueError\", \"message\": \"/private/internal/path\"}}))
    raise SystemExit(1)
""",
        encoding="utf-8",
    )
    token = {
        "schema": "oel.hosted_session_token.v1",
        "tenant_id": "tenant:public-test",
        "signature": {"alg": "RS256", "key_id": "test", "value": "opaque"},
    }
    profile_path = tmp_path / "profile.json"
    profile = link_profile(
        profile_path,
        service_label="local black-box",
        transport_command=[sys.executable, str(service)],
        session_token=token,
    )
    assert profile_path.stat().st_mode & 0o777 == 0o600
    assert profile["model_credentials_stored"] is False
    ledger = tmp_path / "transactions.jsonl"
    client = HostedClient(profile_path, ledger_path=ledger)
    capabilities = client.capabilities()
    assert capabilities["capabilities"] == []
    assert read_ledger(ledger)[0]["event_type"] == "capabilities_observed"
    with pytest.raises(HostedServiceError) as error:
        client.get_job("job")
    assert "/private/internal/path" not in str(error.value)

    ledger.write_text(ledger.read_text(encoding="utf-8").replace("capabilities_observed", "tampered"), encoding="utf-8")
    with pytest.raises(ValueError, match="chain verification"):
        read_ledger(ledger)

    profile_path.chmod(0o644)
    with pytest.raises(PermissionError, match="group or others"):
        HostedClient(profile_path, ledger_path=ledger)


def test_ledger_rejects_symlinks_and_reuses_an_abandoned_lock_inode(tmp_path: Path) -> None:
    unrelated = tmp_path / "unrelated.txt"
    unrelated.write_text("keep me", encoding="utf-8")
    alias = tmp_path / "transactions.jsonl"
    alias.symlink_to(unrelated)
    with pytest.raises(ValueError, match="symbolic link"):
        append_event(alias, "route_classified", {})
    assert unrelated.read_text(encoding="utf-8") == "keep me"
    with pytest.raises(ValueError, match="symbolic link"):
        read_ledger(alias)

    alias.unlink()
    linked_parent = tmp_path / "linked"
    linked_parent.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="symbolic link"):
        append_event(linked_parent / "transactions.jsonl", "route_classified", {})

    # A previous process can leave the lock inode behind; the OS lock is gone.
    lock = alias.with_suffix(alias.suffix + ".lock")
    lock.touch()
    assert append_event(alias, "route_classified", {})["sequence"] == 1
    assert lock.exists()


def test_result_import_rejects_traversal_and_byte_drift(tmp_path: Path) -> None:
    payload = b"evidence"
    transfer = {
        "schema": "oel.hosted_result_transfer.v1",
        "job_id": "job",
        "tenant_id": "tenant",
        "capsule_sha256": "a" * 64,
        "worker_image_sha256": "b" * 64,
        "artifact_bundle_sha256": "c" * 64,
        "artifacts": [
            {
                "artifact_id": "report",
                "kind": "report_packet",
                "content_sha256": "d" * 64,
                "files": [
                    {
                        "relative_path": "report.json",
                        "bytes": len(payload),
                        "content_sha256": hashlib.sha256(payload).hexdigest(),
                    }
                ],
            }
        ],
    }
    transfer["transfer_sha256"] = canonical_sha256(transfer)
    destination = tmp_path / "result"
    with pytest.raises(ValueError, match="did not match"):
        import_result_transfer(transfer, destination, read_file=lambda *_args: b"changed")
    assert not destination.exists()

    traversal = json.loads(json.dumps(transfer))
    traversal["artifacts"][0]["files"][0]["relative_path"] = "../escape"
    traversal["transfer_sha256"] = canonical_sha256(
        {key: value for key, value in traversal.items() if key != "transfer_sha256"}
    )
    with pytest.raises(ValueError, match="Unsafe path"):
        import_result_transfer(traversal, destination, read_file=lambda *_args: payload)
    assert not (tmp_path / "escape").exists()


def test_hosted_cli_requires_explicit_confirmation(tmp_path: Path) -> None:
    offer = tmp_path / "offer.json"
    offer.write_text("{}", encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "sim.hosted_client",
            "approve",
            "--profile",
            str(tmp_path / "missing-profile.json"),
            "--offer",
            str(offer),
            "--idempotency-key",
            "proof",
            "--authorized-by",
            "user",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "--confirm" in result.stderr
    assert "Traceback" not in result.stderr


def test_ledger_appends_a_verified_hash_chain(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.jsonl"
    first = append_event(ledger, "route_classified", {"route": "LOCAL_FREE_AVAILABLE"})
    second = append_event(ledger, "results_imported", {"artifact_bytes": 10})
    assert second["previous_event_sha256"] == first["event_sha256"]
    assert [item["sequence"] for item in read_ledger(ledger)] == [1, 2]


@pytest.mark.parametrize('paths', [
    ('report.json', 'REPORT.json'),
    ('Dir/one.json', 'dir/two.json'),
    ('caf\u00e9.json', 'cafe\u0301.json'),
    ('report', 'report/child.json'),
    ('report/child.json', 'report'),
    ('report.json', 'report.json'),
])
def test_result_import_rejects_portable_path_collisions_before_download(tmp_path, paths):
    transfer = _result_transfer_for_paths(paths)
    reads = []
    with pytest.raises(ValueError, match='duplicate or aliasing'):
        import_result_transfer(transfer, tmp_path / 'result', read_file=lambda *args: reads.append(args))
    assert reads == []
    assert not (tmp_path / 'result').exists()


def test_result_import_preserves_distinct_files_and_receipt(tmp_path):
    paths = ('reports/one.json', 'reports/two.json')
    transfer = _result_transfer_for_paths(paths)
    receipt = import_result_transfer(transfer, tmp_path / 'result', read_file=lambda _, name: name.encode())
    artifact = receipt['artifacts'][0]
    for row in artifact['files']:
        output = tmp_path / 'result' / artifact['local_directory'] / row['relative_path']
        assert hashlib.sha256(output.read_bytes()).hexdigest() == row['content_sha256']
    assert (tmp_path / 'result' / 'result_import_receipt.json').is_file()


def _result_transfer_for_paths(paths):
    transfer = {
        'schema': 'oel.hosted_result_transfer.v1', 'job_id': 'job', 'tenant_id': 'tenant',
        'capsule_sha256': 'a' * 64, 'worker_image_sha256': 'b' * 64, 'artifact_bundle_sha256': 'c' * 64,
        'artifacts': [{'artifact_id': 'report', 'kind': 'report', 'files': [
            {'relative_path': path, 'bytes': len(path.encode()), 'content_sha256': hashlib.sha256(path.encode()).hexdigest()}
            for path in paths
        ]}],
    }
    transfer['transfer_sha256'] = canonical_sha256(transfer)
    return transfer
