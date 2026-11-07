"""CLI for the public OEL FSW Authoring Kit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .candidate import ROOT, CandidateValidationError, inspect_candidate
from .native_transport import NativeBridgeFailure
from .services import (
    describe_capabilities,
    doctor,
    init_candidate,
    plan_workflow,
    run_contract_tests,
    run_smoke,
    validate_candidate_service,
    verify_receipt,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="OEL Public Flight Software Authoring Kit")
    parser.add_argument("--workspace-root", default=str(ROOT))
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("describe")
    sub.add_parser("doctor")
    init = sub.add_parser("init")
    init.add_argument("name")
    init.add_argument("--template", choices=("adcs", "rpo"), default="adcs")
    init.add_argument("--output-dir")
    init.add_argument("--class-name")
    init.add_argument("--force", action="store_true")
    init.add_argument("--language", choices=("python", "c", "cpp", "rust"), default="python")
    inspect = sub.add_parser("inspect")
    inspect.add_argument("manifest")
    plan = sub.add_parser("plan")
    plan.add_argument("manifest")
    plan.add_argument("operation", choices=("validate", "test", "smoke", "build"))
    plan.add_argument("--output-dir")
    validate = sub.add_parser("validate")
    validate.add_argument("manifest")
    validate.add_argument("--trusted-import", action="store_true")
    validate.add_argument("--receipt-dir")
    build = sub.add_parser("build")
    build.add_argument("manifest")
    build.add_argument("--trusted-build", action="store_true")
    build.add_argument("--output-dir")
    for name in ("test", "smoke"):
        command = sub.add_parser(name)
        command.add_argument("manifest")
        command.add_argument("--output-dir")
        command.add_argument("--validation-id")
    verify = sub.add_parser("verify-receipt")
    verify.add_argument("receipt")
    return parser


def _print(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    root = Path(args.workspace_root).expanduser().resolve()
    try:
        if args.command == "describe":
            payload = describe_capabilities()
        elif args.command == "doctor":
            payload = doctor(workspace_root=root)
        elif args.command == "init":
            payload = init_candidate(
                args.name,
                template=args.template,
                workspace_root=root,
                output_dir=args.output_dir,
                class_name=args.class_name,
                force=args.force,
                language=args.language,
            )
        elif args.command == "inspect":
            payload = inspect_candidate(args.manifest, workspace_root=root)
        elif args.command == "plan":
            payload = plan_workflow(
                args.manifest,
                args.operation,
                workspace_root=root,
                output_dir=args.output_dir,
            )
        elif args.command == "build":
            from .native import build_candidate

            payload = build_candidate(
                args.manifest, workspace_root=root, output_dir=args.output_dir, trusted_build=args.trusted_build
            )
        elif args.command == "validate":
            payload = validate_candidate_service(
                args.manifest,
                workspace_root=root,
                trusted_import=args.trusted_import,
                receipt_dir=args.receipt_dir,
            )
        elif args.command == "test":
            payload = run_contract_tests(
                args.manifest,
                workspace_root=root,
                output_dir=args.output_dir,
                validation_id=args.validation_id,
            )
        elif args.command == "smoke":
            payload = run_smoke(
                args.manifest,
                workspace_root=root,
                output_dir=args.output_dir,
                validation_id=args.validation_id,
            )
        elif args.command == "verify-receipt":
            payload = verify_receipt(args.receipt, workspace_root=root)
        else:
            parser.error(f"Unknown command: {args.command}")
            return 2
    except (CandidateValidationError, OSError, ValueError, NativeBridgeFailure) as exc:
        _print({"status": "failed", "error": {"type": type(exc).__name__, "message": str(exc)}})
        return 2
    _print(payload)
    return 0 if payload.get("status") in {"ready", "passed"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
