"""Command-line discovery for complete stacks and use-case profiles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .catalog import resolve_stack, stack_catalog
from .profiles import (
    materialize_use_case_profile,
    resolve_use_case_profile,
    use_case_profiles,
    validate_use_case_profile_catalog,
)


def _stack_row(entry: object) -> dict[str, object]:
    return {
        "kind": "stack",
        "id": entry.stack_id,
        "version": entry.version,
        "maturity": entry.maturity.value,
        "summary": entry.summary,
        "compatible_hardware_profiles": list(entry.compatible_hardware_profiles),
        "capabilities": list(entry.capabilities),
    }


def _profile_row(entry: object) -> dict[str, object]:
    payload = entry.to_dict()
    declared = str(payload.get("maturity", "experimental"))
    effective = "experimental"
    evidence_status = "qualification_evidence_unavailable"
    try:
        from .qualification import profile_qualification_status

        status = profile_qualification_status(entry.profile_id)
        effective = str(status["effective_maturity"])
        evidence_status = "promotion_ready" if status["promotion_ready"] else "not_promotion_ready"
    except (ModuleNotFoundError, FileNotFoundError, ValueError):
        pass
    return {
        "kind": "profile",
        "id": entry.profile_id,
        **payload,
        "declared_maturity": declared,
        "effective_maturity": effective,
        "evidence_status": evidence_status,
        "maturity": effective,
    }


def _print(payload: Any, *, json_mode: bool) -> None:
    if json_mode:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    if isinstance(payload, list):
        for row in payload:
            domain = f"/{row['domain']}" if row.get("domain") else ""
            print(f"{row['id']:<52} {row['maturity']:<13} {row['kind']}{domain}")
        return
    for key, value in payload.items():
        print(f"{key}: {value}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect OEL complete FSW stacks and use-case profiles.")
    sub = parser.add_subparsers(dest="command", required=True)

    list_parser = sub.add_parser("list", help="List complete stacks and/or use-case profiles.")
    list_parser.add_argument("--kind", choices=("all", "stack", "profile"), default="all")
    list_parser.add_argument("--domain")
    list_parser.add_argument("--json", action="store_true")

    show_parser = sub.add_parser("show", help="Show one stack or profile.")
    show_parser.add_argument("item_id")
    show_parser.add_argument("--json", action="store_true")

    materialize_parser = sub.add_parser("materialize", help="Resolve one profile into a YAML-ready FSW block.")
    materialize_parser.add_argument("profile_id")
    materialize_parser.add_argument("--params-json", default="{}")
    materialize_parser.add_argument("--hardware-profile")
    materialize_parser.add_argument("--task-period-s", type=float)
    materialize_parser.add_argument("--json", action="store_true")

    author_parser = sub.add_parser(
        "author",
        help="Run the public FSW Authoring Kit.",
    )
    author_parser.add_argument("author_args", nargs=argparse.REMAINDER)

    if (Path(__file__).resolve().parents[1] / "fswdk" / "__main__.py").is_file():
        fswdk_parser = sub.add_parser(
            "fswdk",
            help="Run the private agent-native Flight Software Development and Verification Kit.",
        )
        fswdk_parser.add_argument("fswdk_args", nargs=argparse.REMAINDER)

    validate_parser = sub.add_parser("validate", help="Validate profile identities and stack compatibility.")
    validate_parser.add_argument("--json", action="store_true")
    qualification_available = Path(__file__).with_name("qualification.py").is_file()
    if qualification_available:
        validate_parser.add_argument("--require-promotion-ready", action="store_true")
        status_parser = sub.add_parser("status", help="Inspect exact-profile qualification and evidence status.")
        status_parser.add_argument("profile_id")
        status_parser.add_argument("--json", action="store_true")

        qualify_parser = sub.add_parser("qualify", help="Run a trusted exact-profile qualification specification.")
        qualify_parser.add_argument("profile_id")
        qualify_parser.add_argument("--output-dir")
        qualify_parser.add_argument("--validate-only", action="store_true")
        qualify_parser.add_argument(
            "--overwrite-output",
            action="store_true",
            help="Archive existing scenario outputs before qualification execution.",
        )
        qualify_parser.add_argument(
            "--refresh-manifest",
            action="store_true",
            help="Refresh dependency hashes only after all qualification gates pass on stable source.",
        )
        qualify_parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    if args.command == "author":
        from sim.fsw_authoring.__main__ import main as authoring_main

        return authoring_main(list(args.author_args))

    if args.command == "fswdk":
        try:
            from sim.fswdk.__main__ import main as fswdk_main
        except ModuleNotFoundError as exc:
            parser.error("The private OEL FSWDK is unavailable in this installation.")
            raise AssertionError from exc
        return fswdk_main(list(args.fswdk_args))

    if args.command == "list":
        rows: list[dict[str, object]] = []
        if args.kind in {"all", "stack"} and args.domain is None:
            rows.extend(_stack_row(entry) for entry in stack_catalog())
        if args.kind in {"all", "profile"}:
            rows.extend(_profile_row(entry) for entry in use_case_profiles(domain=args.domain))
        _print(rows, json_mode=args.json)
        return 0
    if args.command == "show":
        if args.item_id.startswith("fsw.profile."):
            payload = _profile_row(resolve_use_case_profile(args.item_id))
        else:
            payload = _stack_row(resolve_stack(args.item_id))
        _print(payload, json_mode=args.json)
        return 0
    if args.command == "materialize":
        try:
            params = json.loads(args.params_json)
        except json.JSONDecodeError as exc:
            parser.error(f"--params-json is not valid JSON: {exc}")
        if not isinstance(params, dict):
            parser.error("--params-json must decode to an object")
        selection = materialize_use_case_profile(
            args.profile_id,
            params=params,
            hardware_profile=args.hardware_profile,
            task_period_s=args.task_period_s,
        )
        _print(selection.to_config(), json_mode=args.json)
        return 0
    if args.command in {"status", "qualify"}:
        try:
            from .qualification import profile_qualification_status, run_profile_qualification
        except ModuleNotFoundError as exc:
            parser.error(
                "profile qualification orchestration is unavailable in this installation; "
                "it is included with the private Controller Bench workflow"
            )
            raise AssertionError from exc
        if args.command == "status":
            payload = profile_qualification_status(args.profile_id)
            _print(payload, json_mode=args.json)
            return 0
        payload = run_profile_qualification(
            args.profile_id,
            output_dir=args.output_dir,
            validate_only=args.validate_only,
            refresh_manifest=args.refresh_manifest,
            overwrite_output=args.overwrite_output,
        )
        _print(payload, json_mode=args.json)
        if args.validate_only:
            return 0 if bool(payload.get("valid", False)) else 2
        return 0 if bool(payload.get("qualification_passed", False)) else 2
    errors = list(validate_use_case_profile_catalog())
    try:
        from .qualification import profile_qualification_status, validate_profile_qualification_catalog

        errors.extend(validate_profile_qualification_catalog())
        if getattr(args, "require_promotion_ready", False):
            for profile in use_case_profiles():
                status = profile_qualification_status(profile.profile_id)
                if not status["promotion_ready"]:
                    errors.append(f"{profile.profile_id}: not promotion-ready: " + "; ".join(status["blockers"]))
    except ModuleNotFoundError:
        if getattr(args, "require_promotion_ready", False):
            errors.append("profile qualification orchestration is unavailable")
    payload = {
        "valid": not errors,
        "profiles": len(use_case_profiles()),
        "stacks": len(stack_catalog()),
        "errors": errors,
    }
    _print(payload, json_mode=args.json)
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
