"""Stable public facade and CLI for content-bound OEL study bundles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from sim.analysis.study_lifecycle import (
    CAPABILITY_CONTRACTS,
    MAX_STUDY_EVIDENCE_BYTES,
    MAX_STUDY_STEPS,
    STUDY_CLAIMS_SCHEMA,
    STUDY_COMPARISON_SCHEMA,
    STUDY_EVIDENCE_SCHEMA,
    STUDY_PLAN_SCHEMA,
    STUDY_RECEIPT_SCHEMA,
    STUDY_REQUEST_SCHEMA,
    STUDY_RUN_SCHEMA,
    STUDY_VERIFICATION_SCHEMA,
    StudyBundleArtifacts,
    StudyClaims,
    StudyLifecycleError,
    StudyPlan,
    StudyRequest,
    build_study_bundle,
    compare_study_bundles,
    inspect_study_bundle,
    replay_study_bundle,
    verify_study_bundle,
)
from sim.utils.io import SafeReadError, read_regular_file_nofollow

__all__ = [
    "CAPABILITY_CONTRACTS",
    "MAX_STUDY_EVIDENCE_BYTES",
    "MAX_STUDY_STEPS",
    "STUDY_CLAIMS_SCHEMA",
    "STUDY_COMPARISON_SCHEMA",
    "STUDY_EVIDENCE_SCHEMA",
    "STUDY_PLAN_SCHEMA",
    "STUDY_RECEIPT_SCHEMA",
    "STUDY_REQUEST_SCHEMA",
    "STUDY_RUN_SCHEMA",
    "STUDY_VERIFICATION_SCHEMA",
    "StudyBundleArtifacts",
    "StudyClaims",
    "StudyLifecycleError",
    "StudyPlan",
    "StudyRequest",
    "build_study_bundle",
    "compare_study_bundles",
    "inspect_study_bundle",
    "replay_study_bundle",
    "verify_study_bundle",
]


def _read_json_object(path: Path, field: str) -> dict[str, Any]:
    try:
        value = json.loads(
            read_regular_file_nofollow(
                path,
                min_bytes=1,
                max_bytes=MAX_STUDY_EVIDENCE_BYTES,
            )
        )
    except (OSError, SafeReadError, json.JSONDecodeError) as exc:
        raise StudyLifecycleError(f"Could not read {field} from {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise StudyLifecycleError(f"{field} must contain a JSON object.")
    return value


def _evidence_bindings(values: list[str]) -> dict[str, Path]:
    bindings: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise StudyLifecycleError("--evidence must use STEP_ID=PATH syntax.")
        step_id, path = value.split("=", 1)
        step_id = step_id.strip()
        if not step_id or not path.strip():
            raise StudyLifecycleError("--evidence must use non-empty STEP_ID=PATH values.")
        if step_id in bindings:
            raise StudyLifecycleError(f"Duplicate --evidence binding for {step_id!r}.")
        bindings[step_id] = Path(path)
    return bindings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m sim.study",
        description="Build, inspect, replay, and compare content-bound OEL study bundles.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    request = commands.add_parser("validate-request", help="Validate and normalize one study request.")
    request.add_argument("request", type=Path)
    plan = commands.add_parser("validate-plan", help="Validate and bind one study plan to its request.")
    plan.add_argument("request", type=Path)
    plan.add_argument("plan", type=Path)
    claims = commands.add_parser("validate-claims", help="Validate claims against a bound request and plan.")
    claims.add_argument("request", type=Path)
    claims.add_argument("plan", type=Path)
    claims.add_argument("claims", type=Path)
    build = commands.add_parser("build", help="Build a new study bundle from completed evidence JSON files.")
    build.add_argument("request", type=Path)
    build.add_argument("plan", type=Path)
    build.add_argument("claims", type=Path)
    build.add_argument("--evidence", action="append", default=[], metavar="STEP_ID=PATH", required=True)
    build.add_argument("--output-dir", type=Path, required=True)
    inspect = commands.add_parser("inspect", help="Verify and summarize one study bundle.")
    inspect.add_argument("bundle", type=Path)
    replay = commands.add_parser("replay", help="Rebuild and verify the study identity graph.")
    replay.add_argument("bundle", type=Path)
    compare = commands.add_parser("compare", help="Compare two verified study bundles.")
    compare.add_argument("left", type=Path)
    compare.add_argument("right", type=Path)
    schedule_power = commands.add_parser(
        "build-schedule-power", help="Build a verified study from one completed schedule and power inputs."
    )
    schedule_power.add_argument("--schedule-dir", type=Path, required=True)
    schedule_power.add_argument("--problem", type=Path, required=True)
    schedule_power.add_argument("--history", type=Path, required=True)
    schedule_power.add_argument("--schedule-epoch-jd-utc", type=float, required=True)
    schedule_power.add_argument("--observation-load-w", type=float, required=True)
    schedule_power.add_argument("--downlink-load-w", type=float, required=True)
    schedule_power.add_argument("--output-dir", type=Path, required=True)
    schedule_power.add_argument("--study-id")
    schedule_power.add_argument("--title")
    inspect_schedule_power = commands.add_parser(
        "inspect-schedule-power", help="Replay both domains and verify a schedule-power study."
    )
    inspect_schedule_power.add_argument("output_dir", type=Path)
    source_power = commands.add_parser(
        "build-source-schedule-power",
        help="Build a verified study from collection/link source products and power inputs.",
    )
    source_power.add_argument("--source-plan", type=Path, required=True)
    source_power.add_argument("--base-dir", type=Path)
    source_power.add_argument("--problem", type=Path, required=True)
    source_power.add_argument("--history", type=Path, required=True)
    source_power.add_argument("--observation-load-w", type=float, required=True)
    source_power.add_argument("--downlink-load-w", type=float, required=True)
    source_power.add_argument("--output-dir", type=Path, required=True)
    source_power.add_argument("--study-id")
    source_power.add_argument("--title")
    inspect_source_power = commands.add_parser(
        "inspect-source-schedule-power",
        help="Replay source products, scheduling, power, and study evidence.",
    )
    inspect_source_power.add_argument("output_dir", type=Path)
    export_orbit = commands.add_parser(
        "export-orbit-history", help="Retain one completed-run ECI history with source receipts."
    )
    export_orbit.add_argument("completed_run", type=Path)
    export_orbit.add_argument("--object-id", required=True)
    export_orbit.add_argument("--output-dir", type=Path, required=True)
    bound_power = commands.add_parser(
        "build-orbit-bound-source-study",
        help="Build a study whose collection, link, and power share one retained orbit.",
    )
    bound_power.add_argument("--orbit-history-dir", type=Path, required=True)
    bound_power.add_argument("--source-plan", type=Path, required=True)
    bound_power.add_argument("--base-dir", type=Path)
    bound_power.add_argument("--problem", type=Path, required=True)
    bound_power.add_argument("--observation-load-w", type=float, required=True)
    bound_power.add_argument("--downlink-load-w", type=float, required=True)
    bound_power.add_argument("--output-dir", type=Path, required=True)
    bound_power.add_argument("--study-id")
    bound_power.add_argument("--title")
    inspect_bound = commands.add_parser(
        "inspect-orbit-bound-source-study", help="Replay all shared-orbit domain bindings."
    )
    inspect_bound.add_argument("output_dir", type=Path)
    bound_link = commands.add_parser(
        "build-orbit-bound-link", help="Generate one link product from exact orbit-history samples."
    )
    bound_link.add_argument("--orbit-history-dir", type=Path, required=True)
    bound_link.add_argument("--config", type=Path, required=True)
    bound_link.add_argument("--station-latitude-deg", type=float, required=True)
    bound_link.add_argument("--station-longitude-deg", type=float, required=True)
    bound_link.add_argument("--station-height-km", type=float, required=True)
    bound_link.add_argument("--first-sample-index", type=int, required=True)
    bound_link.add_argument("--last-sample-index", type=int, required=True)
    bound_link.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "validate-request":
            request = StudyRequest.from_mapping(_read_json_object(args.request, "study request"))
            payload: Any = {
                "status": "valid",
                "request": request.to_dict(),
            }
        elif args.command == "validate-plan":
            request = StudyRequest.from_mapping(_read_json_object(args.request, "study request"))
            plan = StudyPlan.from_mapping(_read_json_object(args.plan, "study plan"), request)
            payload = {"status": "valid", "plan": plan.to_dict()}
        elif args.command == "validate-claims":
            request = StudyRequest.from_mapping(_read_json_object(args.request, "study request"))
            plan = StudyPlan.from_mapping(_read_json_object(args.plan, "study plan"), request)
            claims = StudyClaims.from_mapping(
                _read_json_object(args.claims, "study claims"), request, plan
            )
            payload = {"status": "valid", "claims": claims.to_dict()}
        elif args.command == "build":
            artifacts = build_study_bundle(
                _read_json_object(args.request, "study request"),
                _read_json_object(args.plan, "study plan"),
                _read_json_object(args.claims, "study claims"),
                _evidence_bindings(args.evidence),
                args.output_dir,
            )
            payload = inspect_study_bundle(artifacts.output_dir)
        elif args.command == "inspect":
            payload = inspect_study_bundle(args.bundle)
        elif args.command == "replay":
            payload = replay_study_bundle(args.bundle)
        elif args.command == "build-schedule-power":
            from sim.analysis.schedule_power_study import build_schedule_power_study

            problem = json.loads(read_regular_file_nofollow(args.problem, min_bytes=1, max_bytes=16 * 1024 * 1024))
            history = json.loads(read_regular_file_nofollow(args.history, min_bytes=1, max_bytes=16 * 1024 * 1024))
            payload = build_schedule_power_study(
                schedule_dir=args.schedule_dir,
                problem=problem,
                history=history,
                schedule_epoch_jd_utc=args.schedule_epoch_jd_utc,
                observation_load_w=args.observation_load_w,
                downlink_load_w=args.downlink_load_w,
                output_dir=args.output_dir,
                study_id=args.study_id,
                title=args.title,
            )
        elif args.command == "inspect-schedule-power":
            from sim.analysis.schedule_power_study import inspect_schedule_power_study

            payload = inspect_schedule_power_study(args.output_dir)
        elif args.command == "build-source-schedule-power":
            from sim.analysis.schedule_power_study import build_source_schedule_power_study

            source_plan = json.loads(read_regular_file_nofollow(
                args.source_plan, min_bytes=1, max_bytes=16 * 1024 * 1024
            ))
            problem = json.loads(read_regular_file_nofollow(
                args.problem, min_bytes=1, max_bytes=16 * 1024 * 1024
            ))
            history = json.loads(read_regular_file_nofollow(
                args.history, min_bytes=1, max_bytes=16 * 1024 * 1024
            ))
            payload = build_source_schedule_power_study(
                source_plan=source_plan,
                base_dir=args.base_dir or args.source_plan.parent,
                problem=problem,
                history=history,
                observation_load_w=args.observation_load_w,
                downlink_load_w=args.downlink_load_w,
                output_dir=args.output_dir,
                study_id=args.study_id,
                title=args.title,
            )
        elif args.command == "inspect-source-schedule-power":
            from sim.analysis.schedule_power_study import inspect_source_schedule_power_study

            payload = inspect_source_schedule_power_study(args.output_dir)
        elif args.command == "export-orbit-history":
            from sim.analysis.orbit_history_product import export_orbit_history_product

            payload = export_orbit_history_product(
                args.completed_run, object_id=args.object_id, output_dir=args.output_dir
            )
        elif args.command == "build-orbit-bound-source-study":
            from sim.analysis.schedule_power_study import build_orbit_bound_source_study

            source_plan = json.loads(read_regular_file_nofollow(
                args.source_plan, min_bytes=1, max_bytes=16 * 1024 * 1024
            ))
            problem = json.loads(read_regular_file_nofollow(
                args.problem, min_bytes=1, max_bytes=16 * 1024 * 1024
            ))
            payload = build_orbit_bound_source_study(
                orbit_history_dir=args.orbit_history_dir,
                source_plan=source_plan,
                base_dir=args.base_dir or args.source_plan.parent,
                problem=problem,
                observation_load_w=args.observation_load_w,
                downlink_load_w=args.downlink_load_w,
                output_dir=args.output_dir,
                study_id=args.study_id,
                title=args.title,
            )
        elif args.command == "inspect-orbit-bound-source-study":
            from sim.analysis.schedule_power_study import inspect_orbit_bound_source_study

            payload = inspect_orbit_bound_source_study(args.output_dir)
        elif args.command == "build-orbit-bound-link":
            from sim.analysis.orbit_bound_sources import (
                directed_link_config_from_mapping,
                write_orbit_bound_link,
            )
            from sim.analysis.orbit_history_product import verify_orbit_history_product

            config = json.loads(read_regular_file_nofollow(
                args.config, min_bytes=1, max_bytes=16 * 1024 * 1024
            ))
            _, history = verify_orbit_history_product(args.orbit_history_dir)
            if not (0 <= args.first_sample_index < args.last_sample_index < history.times_s.size):
                raise ValueError("Link sample bounds must select at least two retained history rows.")
            payload = write_orbit_bound_link(
                directed_link_config_from_mapping(config), history,
                station_latitude_deg=args.station_latitude_deg,
                station_longitude_deg=args.station_longitude_deg,
                station_height_km=args.station_height_km,
                sample_indices=list(range(args.first_sample_index, args.last_sample_index + 1)),
                output_dir=args.output_dir,
            )
        else:
            payload = compare_study_bundles(args.left, args.right)
        print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
        return 0
    except (OSError, SafeReadError, StudyLifecycleError, ValueError) as exc:
        print(json.dumps({"status": "error", "message": str(exc)}, indent=2, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
