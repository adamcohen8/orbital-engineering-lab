"""Command-line interface for the OEL FSWDK."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from sim.pro_features import (
    FEATURE_CONTROLLER_BENCH,
    FEATURE_CUSTOMER_PACKAGES,
    FEATURE_GNC_WORKBENCH,
    FEATURE_VALIDATION_EVIDENCE,
    require_pro_feature,
)

from .candidate import ROOT, CandidateValidationError, inspect_candidate
from .external import run_external_smoke, validate_external_compatibility
from .services import (
    compare_evidence,
    describe_capabilities,
    doctor,
    init_candidate,
    inspect_evidence,
    plan_workflow,
    prepare_package,
    promote_baseline,
    run_bench,
    run_contract_tests,
    run_qualification,
    run_smoke,
    validate_candidate_service,
    verify_package,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="OEL Flight Software Development and Verification Kit")
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
    inspect = sub.add_parser("inspect")
    inspect.add_argument("manifest")
    plan = sub.add_parser("plan")
    plan.add_argument("manifest")
    plan.add_argument("operation", choices=("validate", "test", "simulate", "bench", "qualify", "package"))
    plan.add_argument("--output-dir")
    validate = sub.add_parser("validate")
    validate.add_argument("manifest")
    validate.add_argument("--trusted-import", action="store_true")
    validate_external = sub.add_parser("validate-external")
    validate_external.add_argument("manifest")
    simulate_external = sub.add_parser("simulate-external")
    simulate_external.add_argument("manifest")
    simulate_external.add_argument("--output-dir")
    simulate_external.add_argument("--validation-id")
    simulate_external.add_argument("--startup-wait-s", type=float, default=2.0)
    for name in ("test", "simulate", "bench", "qualify"):
        command = sub.add_parser(name)
        command.add_argument("manifest")
        command.add_argument("--output-dir")
        command.add_argument("--validation-id")
    sub.choices["bench"].add_argument("--show-progress", action="store_true")
    sub.choices["qualify"].add_argument("--skip-bench", action="store_true")
    evidence = sub.add_parser("inspect-evidence")
    evidence.add_argument("path")
    promote = sub.add_parser("promote-baseline")
    promote.add_argument("manifest")
    promote.add_argument("evidence_packet")
    promote.add_argument("name")
    promote.add_argument("--registry-dir")
    promote.add_argument("--approve", action="store_true")
    compare = sub.add_parser("compare")
    compare.add_argument("baseline")
    compare.add_argument("candidate_packet")
    compare.add_argument("--output-path")
    package = sub.add_parser("package")
    package.add_argument("manifest")
    package.add_argument("evidence_dir")
    package.add_argument("--output-dir")
    verify = sub.add_parser("verify-package")
    verify.add_argument("package_manifest")
    return parser


def _print(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def _require_command_entitlement(command: str) -> None:
    if command in {"describe", "doctor"}:
        return
    feature = (
        FEATURE_CONTROLLER_BENCH
        if command in {"bench", "compare", "promote-baseline"}
        else FEATURE_VALIDATION_EVIDENCE
        if command in {"qualify", "inspect-evidence"}
        else FEATURE_CUSTOMER_PACKAGES
        if command in {"package", "verify-package"}
        else FEATURE_GNC_WORKBENCH
    )
    require_pro_feature(feature)


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    root = Path(args.workspace_root).expanduser().resolve()
    try:
        _require_command_entitlement(args.command)
        if args.command == "describe":
            payload = describe_capabilities()
        elif args.command == "doctor":
            payload = doctor(workspace_root=root)
        elif args.command == "init":
            payload = init_candidate(
                args.name,
                template=args.template,
                project_root=root,
                output_dir=args.output_dir,
                class_name=args.class_name,
                force=args.force,
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
        elif args.command == "validate":
            payload = validate_candidate_service(
                args.manifest,
                workspace_root=root,
                trusted_import=args.trusted_import,
            )
        elif args.command == "validate-external":
            payload = validate_external_compatibility(args.manifest, workspace_root=root)
        elif args.command == "simulate-external":
            payload = run_external_smoke(
                args.manifest,
                workspace_root=root,
                output_dir=args.output_dir,
                validation_id=args.validation_id,
                startup_wait_s=args.startup_wait_s,
            )
        elif args.command == "test":
            payload = run_contract_tests(
                args.manifest,
                workspace_root=root,
                output_dir=args.output_dir,
                validation_id=args.validation_id,
            )
        elif args.command == "simulate":
            payload = run_smoke(
                args.manifest,
                workspace_root=root,
                output_dir=args.output_dir,
                validation_id=args.validation_id,
            )
        elif args.command == "bench":
            payload = run_bench(
                args.manifest,
                workspace_root=root,
                output_dir=args.output_dir,
                validation_id=args.validation_id,
                show_progress=args.show_progress,
            )
        elif args.command == "qualify":
            payload = run_qualification(
                args.manifest,
                workspace_root=root,
                output_dir=args.output_dir,
                validation_id=args.validation_id,
                include_bench=not args.skip_bench,
            )
        elif args.command == "inspect-evidence":
            payload = inspect_evidence(args.path)
        elif args.command == "promote-baseline":
            payload = promote_baseline(
                args.manifest,
                args.evidence_packet,
                args.name,
                workspace_root=root,
                registry_dir=args.registry_dir,
                approved=args.approve,
            )
        elif args.command == "compare":
            payload = compare_evidence(args.baseline, args.candidate_packet, output_path=args.output_path)
        elif args.command == "package":
            payload = prepare_package(
                args.manifest,
                args.evidence_dir,
                workspace_root=root,
                output_dir=args.output_dir,
            )
        elif args.command == "verify-package":
            payload = verify_package(args.package_manifest)
        else:
            parser.error(f"Unknown command: {args.command}")
            return 2
    except (CandidateValidationError, FileNotFoundError, FileExistsError, PermissionError, ValueError) as exc:
        _print({"status": "failed", "error": {"type": type(exc).__name__, "message": str(exc)}})
        return 2
    _print(payload)
    return 0 if payload.get("status") in {"ready", "passed"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
