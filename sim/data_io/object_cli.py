"""User-facing object composition/export; only routes focused owners."""

from __future__ import annotations

import argparse
import json
import sys

from .common import read_text, source_receipt, strict_json
from .contracts import DataIOError


def build_parser():
    parser = argparse.ArgumentParser(
        prog="oel io object",
        description="Shared object import/export over existing OEL owners; no simulation execution.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser(
        "describe-import", help="Inspect requirements, promotion gates, defaults and field disposition."
    )
    apply = commands.add_parser(
        "import", help="Create a new scenario or add/replace a selected object in a derived scenario."
    )
    for command in (inspect, apply):
        command.add_argument("source")
        command.add_argument("--format", required=True)
        command.add_argument("--object-id", help="Input object selector or ID for formats without one.")
        command.add_argument("--sample-index", type=int, default=0)
        command.add_argument("--options", help="Advanced format options as JSON.")
        command.add_argument("--frame")
        command.add_argument("--target-frame")
        command.add_argument("--eop-path")
    apply.add_argument("output", help="New scenario YAML; existing inputs are never replaced.")
    apply.add_argument("--into", help="Existing scenario to copy and compose; original remains unchanged.")
    apply.add_argument("--as", dest="target_object_id", help="Object ID in the new/derived scenario.")
    apply.add_argument(
        "--replace",
        action="store_true",
        help="Explicitly replace an existing object's orbit while retaining other features.",
    )
    apply.add_argument(
        "--allow-model-change",
        action="store_true",
        help="Explicitly retain destination model assumptions when they differ from the input.",
    )
    apply.add_argument("--scenario-name", required=True)
    apply.add_argument("--run-output", required=True)
    apply.add_argument("--duration-s", type=float)
    apply.add_argument("--step-s", type=float)
    describe = commands.add_parser(
        "describe-export", help="Discover orbital, estimate, snapshot/checkpoint and power export routes."
    )
    export = commands.add_parser("export", help="Export one completed object or an explicit atomic snapshot.")
    for command in (describe, export):
        command.add_argument("source")
        command.add_argument("--object-id")
        command.add_argument("--source-kind", choices=("auto", "review", "tracking-od"), default="auto")
    export.add_argument("output")
    export.add_argument("--format", required=True)
    export.add_argument("--object-ids", nargs="+")
    selection = export.add_mutually_exclusive_group()
    selection.add_argument("--sample-index", type=int)
    selection.add_argument("--final", action="store_true")
    export.add_argument("--options")
    export.add_argument("--target-frame")
    export.add_argument("--eop-path")
    export.add_argument("--satellite-id")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        options = strict_json(read_text(args.options)) if getattr(args, "options", None) else {}
        if not isinstance(options, dict):
            raise DataIOError("Object IO options must be a mapping.")
        for key in ("frame", "target_frame", "eop_path", "satellite_id"):
            if getattr(args, key, None) is not None:
                options[key] = getattr(args, key)
        from .objects import (
            create_scenario,
            describe_export,
            describe_import,
            export_object,
            import_object,
            materialize_continuation,
            write_composed_scenario,
        )

        if args.command == "describe-import":
            result = describe_import(
                args.source,
                format=args.format,
                object_id=args.object_id,
                sample_index=args.sample_index,
                options=options,
            )
        elif args.command == "describe-export":
            result = describe_export(args.source, object_id=args.object_id, source_kind=args.source_kind)
        elif args.command == "export":
            result = export_object(
                args.source,
                args.output,
                format=args.format,
                object_id=args.object_id,
                object_ids=args.object_ids,
                sample_index=args.sample_index,
                final=args.final,
                source_kind=args.source_kind,
                options=options,
            )
        elif args.format in {"satellite-checkpoint", "snapshot"}:
            if args.into or args.replace or args.target_object_id or options:
                raise DataIOError(
                    "Complete continuation uses its exact native context; use the dedicated checkpoint/snapshot options."
                )
            if args.duration_s is None or args.step_s is None:
                raise DataIOError("Continuation needs --duration-s and --step-s.")
            result = materialize_continuation(
                args.source,
                args.output,
                format=args.format,
                scenario_name=args.scenario_name,
                output_dir=args.run_output,
                duration_s=args.duration_s,
                step_s=args.step_s,
            )
        else:
            value = import_object(
                args.source,
                format=args.format,
                object_id=args.object_id,
                sample_index=args.sample_index,
                options=options,
            )
            if args.into:
                from sim.api import SimulationWorkspace
                from sim.scenarios import ScenarioArtifact

                identity = source_receipt(args.into)
                artifact = ScenarioArtifact.from_yaml(args.into)
                safe = SimulationWorkspace().validate_safe(artifact)
                if not safe.get("ok"):
                    raise DataIOError(f"Destination scenario failed safe validation: {safe}.")
                if source_receipt(args.into) != identity:
                    raise DataIOError("Destination scenario changed while reading.")
                if args.duration_s is not None or args.step_s is not None:
                    raise DataIOError(
                        "Composition retains the destination timing; edit its scenario explicitly to change duration/timestep."
                    )
                result = write_composed_scenario(
                    artifact,
                    value,
                    args.output,
                    scenario_name=args.scenario_name,
                    output_dir=args.run_output,
                    object_id=args.target_object_id,
                    replace=args.replace,
                    allow_model_change=args.allow_model_change,
                    additional_sources=(args.into,),
                )
            else:
                if args.replace or args.allow_model_change:
                    raise DataIOError("replace/allow-model-change apply only with --into.")
                if args.duration_s is None or args.step_s is None:
                    raise DataIOError("A new scenario needs --duration-s and --step-s.")
                result = create_scenario(
                    value,
                    args.output,
                    scenario_name=args.scenario_name,
                    output_dir=args.run_output,
                    duration_s=args.duration_s,
                    step_s=args.step_s,
                    object_id=args.target_object_id,
                )
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
        return 2 if result.get("status") in {"blocked", "input_required"} else 0
    except (ValueError, KeyError, TypeError, OSError, RuntimeError, ImportError) as exc:
        print(json.dumps({"status": "error", "message": str(exc), "simulation_executed": False}), file=sys.stderr)
        return 2
