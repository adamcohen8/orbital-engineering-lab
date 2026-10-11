"""Public Hosted OEL client that sees only JSON contracts and returned bytes."""

from __future__ import annotations

import base64
import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

from .ledger import append_event
from .profile import load_profile
from .result_import import import_result_transfer
from .transport import SubprocessJsonTransport


class HostedClient:
    def __init__(self, profile_path: str | Path, *, ledger_path: str | Path) -> None:
        self.profile = load_profile(profile_path)
        self.ledger_path = Path(ledger_path)
        if self.profile["transport"]["kind"] == "https_json_v1":
            from .https_transport import HttpsJsonTransport
            self.transport = HttpsJsonTransport(self.profile)
        else:
            self.transport = SubprocessJsonTransport(self.profile)
        self.token = dict(self.profile["session_token"])

    def capabilities(self) -> dict[str, Any]:
        result = self.transport.call("capabilities", {"token": self.token})
        append_event(
            self.ledger_path,
            "capabilities_observed",
            {
                "offer_sha256": result["offer_sha256"],
                "capability_count": len(result["capabilities"]),
            },
        )
        return result

    def stage_input(
        self,
        source_path: str | Path,
        *,
        input_id: str,
        kind: str,
        expected_sha256: str,
    ) -> dict[str, Any]:
        result = self.transport.call(
            "stage_input",
            {
                "token": self.token,
                "source_path": str(Path(source_path).expanduser().absolute()),
                "input_id": str(input_id),
                "kind": str(kind),
                "expected_sha256": str(expected_sha256),
            },
        )
        append_event(
            self.ledger_path,
            "input_staged",
            {
                key: result[key]
                for key in ("grant_id", "input_id", "kind", "content_sha256", "bytes")
            },
        )
        return result

    def prepare_offer(
        self,
        request: Mapping[str, Any],
        proposed_plan: Mapping[str, Any] | None,
        *,
        upload_grant_ids: Sequence[str],
        worker_image_sha256: str,
    ) -> dict[str, Any]:
        result = self.transport.call(
            "prepare_offer",
            {
                "token": self.token,
                "request": dict(request),
                "proposed_plan": None if proposed_plan is None else dict(proposed_plan),
                "upload_grant_ids": list(map(str, upload_grant_ids)),
                "worker_image_sha256": str(worker_image_sha256),
            },
        )
        quote = result.get("quote") or {}
        append_event(
            self.ledger_path,
            "offer_prepared",
            {
                "offer_sha256": result["offer_sha256"],
                "status": result["status"],
                "execution_route": result["planning_result"]["execution_route"]["route"],
                "pricing_policy_sha256": quote.get("pricing_policy_sha256"),
                "estimated_price_microusd": quote.get("estimated_price_microusd"),
                "maximum_authorized_microusd": quote.get(
                    "maximum_authorized_microusd"
                ),
            },
        )
        return result

    def approve_and_submit(
        self,
        offer: Mapping[str, Any],
        *,
        idempotency_key: str,
        authorized_by: str,
        user_authorized: bool,
        confirmation: str,
    ) -> dict[str, Any]:
        if user_authorized is not True:
            raise ValueError("Explicit user authorization is required before hosted submission.")
        result = self.transport.call(
            "approve_and_submit",
            {
                "token": self.token,
                "offer": dict(offer),
                "idempotency_key": str(idempotency_key),
                "authorized_by": str(authorized_by),
                "user_authorized": True,
                "confirmation": str(confirmation),
            },
        )
        append_event(
            self.ledger_path,
            "study_approved_and_submitted",
            {
                "offer_sha256": result["offer_sha256"],
                "approval_sha256": result["approval_sha256"],
                "job_id": result["job"]["job_id"],
                "capsule_sha256": result["job"]["capsule_sha256"],
                "pricing_policy_sha256": result["job"]["billing"][
                    "pricing_policy_sha256"
                ],
                "maximum_authorized_microusd": result["job"]["billing"][
                    "maximum_authorized_microusd"
                ],
            },
        )
        return result

    def get_job(self, job_id: str) -> dict[str, Any]:
        result = self.transport.call(
            "get_job",
            {"token": self.token, "job_id": str(job_id)},
        )
        append_event(
            self.ledger_path,
            "job_status_observed",
            {
                "job_id": result["job_id"],
                "status_sha256": result["status_sha256"],
                "state": result["state"],
                "terminal": result["terminal"],
            },
        )
        return result

    def cancel(self, job_id: str) -> dict[str, Any]:
        result = self.transport.call(
            "request_cancel",
            {"token": self.token, "job_id": str(job_id)},
        )
        append_event(
            self.ledger_path,
            "cancellation_requested",
            {
                "job_id": result["job_id"],
                "status_sha256": result["status_sha256"],
                "state": result["state"],
            },
        )
        return result

    def pull_results(self, job_id: str, destination: str | Path) -> dict[str, Any]:
        transfer = self.transport.call(
            "prepare_result_transfer",
            {"token": self.token, "job_id": str(job_id)},
        )
        if (
            not isinstance(transfer, Mapping)
            or transfer.get("job_id") != str(job_id)
            or transfer.get("tenant_id") != self.profile["tenant_id"]
        ):
            raise ValueError("Hosted result transfer does not match the requested job and tenant.")

        def read_file(artifact_id: str, relative_path: str) -> bytes:
            response = self.transport.call(
                "read_result_file",
                {
                    "token": self.token,
                    "job_id": str(job_id),
                    "artifact_id": artifact_id,
                    "relative_path": relative_path,
                },
            )
            payload = base64.b64decode(str(response["content_base64"]), validate=True)
            if (
                response.get("artifact_id") != artifact_id
                or response.get("relative_path") != relative_path
                or response.get("bytes") != len(payload)
                or response.get("content_sha256") != hashlib.sha256(payload).hexdigest()
            ):
                raise ValueError("Hosted result-file response failed identity verification.")
            return payload

        receipt = import_result_transfer(transfer, destination, read_file=read_file)
        total_bytes = sum(
            int(row["bytes"])
            for artifact in receipt["artifacts"]
            for row in artifact["files"]
        )
        append_event(
            self.ledger_path,
            "results_imported",
            {
                "job_id": str(job_id),
                "transfer_sha256": receipt["transfer_sha256"],
                "receipt_sha256": receipt["receipt_sha256"],
                "artifact_count": len(receipt["artifacts"]),
                "artifact_bytes": total_bytes,
            },
        )
        return receipt

    def pull_review_slice(
        self,
        job_id: str,
        destination: str | Path,
        *,
        start_s: float,
        end_s: float,
        object_ids: Sequence[str] = (),
        families: Sequence[str] = ("state", "relative", "control", "events"),
    ) -> dict[str, Any]:
        selection = {
            "start_s": float(start_s), "end_s": float(end_s),
            "object_ids": sorted(set(map(str, object_ids))),
            "families": sorted(set(map(str, families))),
        }
        transfer = self.transport.call(
            "prepare_review_slice_transfer",
            {"token": self.token, "job_id": str(job_id), **selection},
        )
        if (
            not isinstance(transfer, Mapping)
            or transfer.get("job_id") != str(job_id)
            or transfer.get("tenant_id") != self.profile["tenant_id"]
            or transfer.get("selection") != selection
            or len(transfer.get("artifacts", [])) != 1
            or transfer["artifacts"][0].get("kind") != "review_slice"
        ):
            raise ValueError("Hosted review slice transfer does not match the requested job, tenant, or selection.")
        slice_id = str(transfer["slice_id"])

        def read_file(artifact_id: str, relative_path: str) -> bytes:
            if artifact_id != f"review_slice:{slice_id}":
                raise ValueError("Hosted review slice artifact identity changed.")
            response = self.transport.call(
                "read_review_slice_file",
                {"token": self.token, "job_id": str(job_id), "slice_id": slice_id,
                 "relative_path": relative_path},
            )
            payload = base64.b64decode(str(response["content_base64"]), validate=True)
            if (
                response.get("artifact_id") != artifact_id
                or response.get("relative_path") != relative_path
                or response.get("bytes") != len(payload)
                or response.get("content_sha256") != hashlib.sha256(payload).hexdigest()
            ):
                raise ValueError("Hosted review slice file failed transport verification.")
            return payload

        receipt = import_result_transfer(transfer, destination, read_file=read_file)
        append_event(
            self.ledger_path,
            "review_slice_imported",
            {"job_id": str(job_id), "slice_id": slice_id,
             "transfer_sha256": receipt["transfer_sha256"],
             "receipt_sha256": receipt["receipt_sha256"],
             "artifact_bytes": sum(int(row["bytes"]) for row in receipt["artifacts"][0]["files"])},
        )
        return receipt


__all__ = ["HostedClient"]
