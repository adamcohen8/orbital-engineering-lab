from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from sim.hosted_client.result_import import import_result_transfer, verify_result_import
from sim.study_planning import canonical_sha256


def _transfer(
    payloads: dict[str, bytes],
    *,
    aggregate: str | None = None,
) -> tuple[dict[str, object], dict[str, bytes]]:
    rows = [
        {
            "relative_path": relative,
            "bytes": len(payload),
            "content_sha256": hashlib.sha256(payload).hexdigest(),
        }
        for relative, payload in payloads.items()
    ]
    canonical_aggregate = canonical_sha256(
        sorted((row["relative_path"], row["content_sha256"]) for row in rows)
    )
    transfer: dict[str, object] = {
        "schema": "oel.hosted_result_transfer.v1",
        "job_id": "job:importer-regression",
        "tenant_id": "tenant:importer-regression",
        "capsule_sha256": "a" * 64,
        "worker_image_sha256": "b" * 64,
        "artifact_bundle_sha256": "c" * 64,
        "artifacts": [
            {
                "artifact_id": "result",
                "kind": "report_packet",
                "content_sha256": canonical_aggregate if aggregate is None else aggregate,
                "files": rows,
            }
        ],
    }
    transfer["transfer_sha256"] = canonical_sha256(
        {key: value for key, value in transfer.items() if key != "transfer_sha256"}
    )
    return transfer, payloads


@pytest.mark.parametrize(
    ("payloads", "aggregate"),
    [
        ({"report.json": b"file artifact"}, "raw"),
        ({"report.json": b"directory artifact"}, "canonical"),
        (
            {"reports/one.json": b"one", "reports/two.json": b"two"},
            "canonical",
        ),
    ],
)
def test_result_import_accepts_file_and_directory_aggregate_forms(
    tmp_path: Path,
    payloads: dict[str, bytes],
    aggregate: str,
) -> None:
    transfer, payload_map = _transfer(payloads)
    rows = transfer["artifacts"][0]["files"]  # type: ignore[index]
    if aggregate == "raw":
        declared = hashlib.sha256(payloads["report.json"]).hexdigest()
    else:
        declared = transfer["artifacts"][0]["content_sha256"]  # type: ignore[index]
    transfer["artifacts"][0]["content_sha256"] = declared  # type: ignore[index]
    transfer["transfer_sha256"] = canonical_sha256(
        {key: value for key, value in transfer.items() if key != "transfer_sha256"}
    )

    destination = tmp_path / "result"
    receipt = import_result_transfer(
        transfer,
        destination,
        read_file=lambda _artifact_id, relative: payload_map[relative],
    )

    assert receipt["artifacts"][0]["content_sha256"] == declared  # type: ignore[index]
    assert verify_result_import(transfer, destination) == receipt
    assert rows


def test_result_import_rejects_wrong_aggregate_without_committing(tmp_path: Path) -> None:
    transfer, payload_map = _transfer({"report.json": b"evidence"}, aggregate="0" * 64)
    destination = tmp_path / "result"

    with pytest.raises(ValueError, match="aggregate content"):
        import_result_transfer(
            transfer,
            destination,
            read_file=lambda _artifact_id, relative: payload_map[relative],
        )

    assert not destination.exists()


def test_result_import_rejects_tampered_cached_identity(tmp_path: Path) -> None:
    transfer, payload_map = _transfer({"report.json": b"legacy file"})
    raw_digest = hashlib.sha256(payload_map["report.json"]).hexdigest()
    transfer["artifacts"][0]["content_sha256"] = raw_digest  # type: ignore[index]
    transfer["transfer_sha256"] = canonical_sha256(
        {key: value for key, value in transfer.items() if key != "transfer_sha256"}
    )
    destination = tmp_path / "result"
    import_result_transfer(
        transfer,
        destination,
        read_file=lambda _artifact_id, relative: payload_map[relative],
    )

    receipt_path = destination / "result_import_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["artifacts"][0]["content_sha256"] = canonical_sha256(
        [("report.json", raw_digest)]
    )
    receipt["receipt_sha256"] = canonical_sha256(
        {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    )
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    with pytest.raises(ValueError, match="aggregate content"):
        verify_result_import(transfer, destination)
