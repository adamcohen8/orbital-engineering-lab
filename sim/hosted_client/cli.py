"""Public Hosted OEL bridge for routing, approval, job control, and result import."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .client import HostedClient
from .contracts import route_planning_result
from .ledger import append_event
from .package_contract import hosted_execution_package_schema, validate_hosted_execution_package
from .profile import link_https_profile, link_profile, load_profile
from .transport import HostedServiceError

DEFAULT_LEDGER = Path(".oel/hosted/transactions.jsonl")


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.expanduser().read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path.name!r}.")
    return value


def _print(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def _client(args: argparse.Namespace) -> HostedClient:
    return HostedClient(args.profile, ledger_path=args.ledger)


def _add_connection(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, epilog="Hosted OEL is a closed alpha; access is not publicly available. Local public workflows remain free.")
    commands = parser.add_subparsers(dest="command", required=True)

    profile = commands.add_parser("profile", help="Manage the local scoped Hosted OEL profile.")
    profile_commands = profile.add_subparsers(dest="profile_command", required=True)
    link = profile_commands.add_parser("link")
    link.add_argument("--profile", type=Path, required=True)
    link.add_argument("--service-label", required=True)
    link.add_argument("--transport-command", type=Path, required=True)
    link.add_argument("--session-token", type=Path, required=True)
    link.add_argument("--timeout", type=float, default=30.0)
    https = profile_commands.add_parser("link-https", help="Exchange an operator account key for a scoped HTTPS session.")
    https.add_argument("--profile", type=Path, required=True)
    https.add_argument("--service-label", required=True)
    https.add_argument("--service-url", required=True)
    https.add_argument("--account-key-file", type=Path, required=True)
    https.add_argument("--scope", action="append", required=True)
    https.add_argument("--ca-file", type=Path)
    https.add_argument("--timeout", type=float, default=30.0)

    route = commands.add_parser("route", help="Classify a content-bound planning result.")
    route.add_argument("--planning-result", type=Path, required=True)
    route.add_argument("--profile", type=Path, help="Optional operator-issued alpha profile; configuration does not authorize execution.")
    route.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)

    commands.add_parser("package-schema", help="Print the public Hosted execution package schema.")

    validate_package = commands.add_parser(
        "validate-package",
        help="Validate and content-bind a local Hosted execution package without uploading or executing it.",
    )
    validate_package.add_argument("package", type=Path)
    validate_package.add_argument("--workspace-root", type=Path)

    capabilities = commands.add_parser("capabilities", help="Discover tenant-visible execution offers.")
    _add_connection(capabilities)

    stage = commands.add_parser("stage", help="Grant one exact local input to Hosted OEL.")
    _add_connection(stage)
    stage.add_argument("--source", type=Path, required=True)
    stage.add_argument("--input-id", required=True)
    stage.add_argument("--kind", required=True)
    stage.add_argument("--expected-sha256", required=True)

    offer = commands.add_parser("offer", help="Request authoritative hosted preflight and quote.")
    _add_connection(offer)
    offer.add_argument("--request", type=Path, required=True)
    offer.add_argument("--plan", type=Path)
    offer.add_argument("--grant", action="append", default=[])
    offer.add_argument("--worker-image-sha256", required=True)

    approve = commands.add_parser("approve", help="Explicitly approve one exact offer and submit it.")
    _add_connection(approve)
    approve.add_argument("--offer", type=Path, required=True)
    approve.add_argument("--idempotency-key", required=True)
    approve.add_argument("--authorized-by", required=True)
    approve.add_argument(
        "--confirmation",
        required=True,
        help="Exact text APPROVE <offer_sha256> shown with the reviewed offer.",
    )

    status = commands.add_parser("status", help="Inspect tenant-safe durable job state.")
    _add_connection(status)
    status.add_argument("job_id")

    cancel = commands.add_parser("cancel", help="Request cancellation of one owned job.")
    _add_connection(cancel)
    cancel.add_argument("job_id")

    pull = commands.add_parser("pull", help="Import verified terminal evidence into a new directory.")
    _add_connection(pull)
    pull.add_argument("job_id")
    pull.add_argument("--destination", type=Path, required=True)
    review_slice = commands.add_parser(
        "pull-review-slice", help="Export only a bounded review slice from a completed hosted job."
    )
    _add_connection(review_slice)
    review_slice.add_argument("job_id")
    review_slice.add_argument("--destination", type=Path, required=True)
    review_slice.add_argument("--start-s", type=float, required=True)
    review_slice.add_argument("--end-s", type=float, required=True)
    review_slice.add_argument("--object", action="append", default=[])
    review_slice.add_argument("--family", action="append", default=[])
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "profile":
            if args.profile_command == "link-https":
                key_path = args.account_key_file.expanduser().absolute()
                if any(p.is_symlink() for p in (key_path, *key_path.parents)) or key_path.stat().st_mode & 0o077:
                    raise PermissionError("Account key file must be owner-only and contain no symlinks")
                linked_profile = link_https_profile(args.profile, service_label=args.service_label,
                    service_url=args.service_url, account_key=_load_object(key_path)["account_key"], scopes=args.scope,
                    ca_file=args.ca_file, timeout_s=args.timeout)
            else:
                command = _load_object(args.transport_command)
                argv_value = command.get("command")
                if not isinstance(argv_value, list):
                    raise ValueError("Transport-command file must contain a 'command' argv array.")
                linked_profile = link_profile(
                    args.profile,
                    service_label=args.service_label,
                    transport_command=argv_value,
                    session_token=_load_object(args.session_token),
                    timeout_s=args.timeout,
                )
            # The profile must retain the opaque local transport binding, but the
            # public receipt must not echo command arguments or scoped credentials.
            result = {
                "schema": "oel.hosted_profile_link_receipt.v1",
                "status": "linked",
                "profile_id": linked_profile["profile_id"],
                "profile_sha256": linked_profile["profile_sha256"],
                "service_label": linked_profile["service_label"],
                "tenant_id": linked_profile["tenant_id"],
                "linked_at": linked_profile["linked_at"],
                "transport_kind": linked_profile["transport"]["kind"],
                "model_credentials_stored": False,
            }
        elif args.command == "package-schema":
            result = hosted_execution_package_schema()
        elif args.command == "validate-package":
            result = validate_hosted_execution_package(args.package, workspace_root=args.workspace_root)
            _print(result)
            return 0 if result["status"] == "CLIENT_VALID" else 2
        elif args.command == "route":
            if args.profile is not None:
                load_profile(args.profile)
            result = route_planning_result(
                _load_object(args.planning_result), hosted_profile_configured=args.profile is not None,
            )
            append_event(
                args.ledger,
                "route_classified",
                {
                    key: result[key]
                    for key in (
                        "planning_result_sha256",
                        "execution_route",
                        "disposition",
                        "oel_execution_amount_usd",
                    )
                },
            )
        elif args.command == "capabilities":
            result = _client(args).capabilities()
        elif args.command == "stage":
            result = _client(args).stage_input(
                args.source,
                input_id=args.input_id,
                kind=args.kind,
                expected_sha256=args.expected_sha256,
            )
        elif args.command == "offer":
            result = _client(args).prepare_offer(
                _load_object(args.request),
                None if args.plan is None else _load_object(args.plan),
                upload_grant_ids=args.grant,
                worker_image_sha256=args.worker_image_sha256,
            )
        elif args.command == "approve":
            result = _client(args).approve_and_submit(
                _load_object(args.offer),
                idempotency_key=args.idempotency_key,
                authorized_by=args.authorized_by,
                user_authorized=True,
                confirmation=args.confirmation,
            )
        elif args.command == "status":
            result = _client(args).get_job(args.job_id)
        elif args.command == "cancel":
            result = _client(args).cancel(args.job_id)
        elif args.command == "pull-review-slice":
            result = _client(args).pull_review_slice(
                args.job_id, args.destination,
                start_s=args.start_s, end_s=args.end_s,
                object_ids=args.object,
                families=args.family or ("state", "relative", "control", "events"),
            )
        else:
            result = _client(args).pull_results(args.job_id, args.destination)
        _print(result)
        return 0
    except (HostedServiceError, OSError, TypeError, ValueError) as exc:
        print(
            json.dumps(
                {
                    "status": "failed",
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2


__all__ = ["DEFAULT_LEDGER", "main"]
