"""Local BYO-agent study planning CLI. This surface never executes a study."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

from sim.project_version import installed_project_version, source_project_version
from sim.utils.io import SafeReadError, read_regular_file_nofollow

from .capabilities import discovery_capability_catalog, public_capability_catalog
from .config_validation import validate_config_path
from .contracts import build_plan_review, contract_schema
from .feedback import prepare_capability_request
from .manifest import PLANNER_VERSION, build_capability_manifest
from .preflight import preflight_study_plan
from .schemas import SCHEMAS

_MAX_PLANNING_DOCUMENT_BYTES = 16 * 1024 * 1024


def _load_object(path: Path) -> dict[str, Any]:
    try:
        content = read_regular_file_nofollow(
            path,
            min_bytes=1,
            max_bytes=_MAX_PLANNING_DOCUMENT_BYTES,
        )
    except SafeReadError as exc:
        raise ValueError(f"Could not safely read planning document {path}: {exc}") from exc
    value = json.loads(content)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path.name!r}.")
    return value


def _print(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def _catalog(scope: str):
    return discovery_capability_catalog() if scope == "discovery" else public_capability_catalog()


def _config_mapping(values: list[str], *, workspace_root: Path) -> tuple[dict[str, Mapping[str, Any]], list[dict[str, Any]]]:
    normalized: dict[str, Mapping[str, Any]] = {}
    receipts: list[dict[str, Any]] = []
    for value in values:
        if "=" not in value:
            raise ValueError("Each --config value must use CONFIG_REF=PATH syntax.")
        config_ref, raw_path = value.split("=", 1)
        if not config_ref or not raw_path or config_ref in normalized:
            raise ValueError("Configuration references and paths must be non-empty and unique.")
        receipt, document = validate_config_path(config_ref, raw_path, workspace_root=workspace_root)
        receipts.append(receipt)
        if document is not None:
            normalized[config_ref] = document
    return normalized, receipts


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create and preflight content-bound OEL study plans without execution or payment."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    schema = commands.add_parser("schema", help="Print one closed study schema.")
    schema.add_argument("schema_id", choices=sorted(SCHEMAS))
    capabilities = commands.add_parser("capabilities", help="Print the explicit planning capability vocabulary.")
    capabilities.add_argument("--scope", choices=("public", "discovery"), default="discovery")
    preflight = commands.add_parser("preflight", help="Compile a proposed plan without executing it.")
    preflight.add_argument("--request", type=Path, required=True)
    preflight.add_argument("--plan", type=Path)
    preflight.add_argument("--config", action="append", default=[], metavar="CONFIG_REF=PATH")
    preflight.add_argument("--workspace-root", type=Path, default=Path.cwd())
    preflight.add_argument("--scope", choices=("public", "discovery"), default="discovery")
    review = commands.add_parser("plan-review", help="Render a review from a PLAN_VALID preflight result.")
    review.add_argument("--planning-result", type=Path, required=True)
    feedback = commands.add_parser("prepare-feedback", help="Prepare an UNSUPPORTED feedback preview; never submit it.")
    feedback.add_argument("--planning-result", type=Path, required=True)
    feedback.add_argument("--summary", required=True)
    feedback.add_argument("--would-pay", choices=("yes", "no", "unspecified"), default="unspecified")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "schema":
            _print(contract_schema(args.schema_id))
        elif args.command == "capabilities":
            _print(build_capability_manifest(_catalog(args.scope)))
        elif args.command == "preflight":
            normalized, receipts = _config_mapping(args.config, workspace_root=args.workspace_root.resolve())
            result = preflight_study_plan(
                _load_object(args.request),
                None if args.plan is None else _load_object(args.plan),
                catalog=_catalog(args.scope),
                normalized_configs=normalized,
                config_receipts=receipts,
            )
            _print(result)
        elif args.command == "plan-review":
            _print(build_plan_review(_load_object(args.planning_result)))
        elif args.command == "prepare-feedback":
            would_pay = None if args.would_pay == "unspecified" else args.would_pay == "yes"
            _print(
                prepare_capability_request(
                    _load_object(args.planning_result),
                    user_summary=args.summary,
                    engine_version=source_project_version() or installed_project_version() or "unknown",
                    would_pay=would_pay,
                )
            )
        return 0
    except (OSError, TypeError, ValueError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 2


__all__ = ["PLANNER_VERSION", "main"]
