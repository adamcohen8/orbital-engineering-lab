"""The offline ``oel io`` interface; explicit formats and separate operations."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, is_dataclass

from .common import publish, read_text, source_receipt, strict_json
from .contracts import AttitudeHistory, DataIOError, Trajectory


def build_parser():
    parser = argparse.ArgumentParser(
        prog="oel io", description="Offline modular data IO; derived exports consume saved evidence."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    objects = commands.add_parser("object", help="Shared object import, composition, export and preflight.")
    objects.add_argument("arguments", nargs=argparse.REMAINDER)
    commands.add_parser("formats", help="Discover format profiles, directions, limits, dependencies and operations.")
    inspect = commands.add_parser("inspect", help="Inspect a file without executing a study.")
    inspect.add_argument("source")
    inspect.add_argument("--format", required=True)
    inspect.add_argument("--options", help="JSON file of explicit adapter options.")
    convert = commands.add_parser("convert", help="Convert supported data; no estimation or simulation.")
    convert.add_argument("source")
    convert.add_argument("output")
    convert.add_argument("--from", dest="input_format", required=True)
    convert.add_argument("--to", dest="output_format", required=True)
    convert.add_argument("--options")
    export = commands.add_parser("export-review", help="Create a new output from a completed review store.")
    export.add_argument("completed_run")
    export.add_argument("output")
    export.add_argument("--format", required=True)
    export.add_argument("--object-id")
    export.add_argument("--start-s", type=float)
    export.add_argument("--stop-s", type=float)
    export.add_argument("--options")
    table = commands.add_parser(
        "export-table", help="Export a bounded review SELECT, including access/coverage/link/schedule evidence."
    )
    table.add_argument("completed_run")
    table.add_argument("output")
    table.add_argument("--query", required=True)
    table.add_argument("--format", choices=("csv", "json", "parquet"), default="json")
    table.add_argument("--max-rows", type=int, default=200_000)
    materialize = commands.add_parser(
        "materialize", help="Write and validate an ordinary ONP or OGP scenario; never execute it."
    )
    materialize.add_argument("source")
    materialize.add_argument("output")
    materialize.add_argument("--format", required=True)
    materialize.add_argument("--scenario-name", required=True)
    materialize.add_argument("--run-output", required=True)
    materialize.add_argument("--duration-s", type=float, required=True)
    materialize.add_argument("--step-s", type=float, required=True)
    materialize.add_argument("--start-epoch", help="Explicit common UTC epoch for a catalog scenario.")
    materialize.add_argument("--options")
    fit = commands.add_parser(
        "fit-review", help="Pro: fit a completed review arc with OGP and check encoded TLE/OMM residuals."
    )
    fit.add_argument("completed_run")
    fit.add_argument("output_dir")
    fit.add_argument("--template", required=True)
    fit.add_argument("--template-format", default="omm-kvn")
    fit.add_argument("--object-id")
    fit.add_argument("--fit-until-s", type=float, required=True)
    fit.add_argument("--max-position-rms-m", type=float, required=True)
    fit.add_argument("--max-velocity-rms-m-s", type=float, required=True)
    fit.add_argument("--formats", nargs="+", default=["omm-kvn", "tle"])
    fit.add_argument("--max-nfev", type=int, default=120)
    fit.add_argument("--estimate-bstar", action="store_true")
    fit.add_argument("--options")
    sample = commands.add_parser("sample", help="Sample a prescribed PV arc with bounded Hermite interpolation.")
    sample.add_argument("source")
    sample.add_argument("--format", required=True)
    sample.add_argument("--time-s", type=float, required=True)
    sample.add_argument("--max-gap-s", type=float, required=True)
    sample.add_argument("--options")
    assessment = commands.add_parser(
        "export-assessment", help="Produce a CDM from saved, problem-bound conjunction assessment evidence."
    )
    assessment.add_argument("evidence")
    assessment.add_argument("problem")
    assessment.add_argument("output")
    assessment.add_argument("--initial-epoch", required=True)
    assessment.add_argument("--format", choices=("cdm-kvn", "cdm-xml"), default="cdm-kvn")
    measurements = commands.add_parser(
        "export-measurements", help="Produce a TDM from saved reduced-geometric measurement rows."
    )
    measurements.add_argument("source")
    measurements.add_argument("output")
    measurements.add_argument("--station-id", required=True)
    measurements.add_argument("--object-id", required=True)
    measurements.add_argument("--format", choices=("tdm-kvn", "tdm-xml"), default="tdm-kvn")
    maneuvers = commands.add_parser(
        "export-maneuvers", help="Export historical review-store impulsive maneuvers as OPM."
    )
    maneuvers.add_argument("completed_run")
    maneuvers.add_argument("output")
    maneuvers.add_argument("--object-id", required=True)
    maneuvers.add_argument("--sample-index", type=int, default=0)
    maneuvers.add_argument("--format", choices=("opm-kvn", "opm-xml"), default="opm-kvn")
    for command in (inspect, convert, export, materialize, sample):
        if command is not export:
            command.add_argument("--object-id")
        command.add_argument("--sample-index", type=int)
        command.add_argument("--segment-index", type=int)
        command.add_argument("--frame", help="Explicit input frame for SP3/Sentinel/binary kernels.")
        command.add_argument("--target-frame", help="Explicit Earth frame transform before export.")
        command.add_argument("--eop-path", help="Epoch-covering IERS EOP file for Earth-fixed conversions.")
        command.add_argument("--satellite-id", help="Three-character SP3 output satellite identifier.")
        command.add_argument("--derive-velocity", action="store_true", default=None)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "object":
        from .object_cli import main as object_main

        return object_main(args.arguments)
    try:
        options = strict_json(read_text(args.options)) if getattr(args, "options", None) else {}
        if not isinstance(options, dict):
            raise DataIOError("Adapter options must be a JSON mapping.")
        for key in (
            "object_id",
            "sample_index",
            "segment_index",
            "frame",
            "target_frame",
            "eop_path",
            "satellite_id",
            "derive_velocity",
        ):
            value = getattr(args, key, None)
            if value is not None:
                options[key] = value
        from .api import as_trajectory, convert, encode_data, export_review, read_data

        if args.command == "formats":
            from .registry import capability_catalog

            result = capability_catalog()
        elif args.command == "inspect":
            source = source_receipt(args.source)
            value = read_data(args.source, format=args.format, options=options)
            if isinstance(value, Trajectory):
                summary = {
                    "object_id": value.object_id,
                    "frame": value.frame,
                    "center": value.center,
                    "samples": len(value.times_s),
                    "coverage_s": [float(value.times_s[0]), float(value.times_s[-1])],
                    "covariance_samples": 0 if value.covariances is None else len(value.covariances),
                }
            elif isinstance(value, AttitudeHistory):
                summary = {
                    "object_id": value.object_id,
                    "reference_frame": value.reference_frame,
                    "body_frame": value.body_frame,
                    "samples": len(value.times_s),
                    "angular_rates": value.rates_rad_s is not None,
                }
            elif isinstance(value, list):
                summary = {"objects": [r.object_id for r in value], "records": len(value)}
            elif hasattr(value, "segments"):
                previews = []
                total = 0
                for segment in value.segments:
                    if isinstance(segment, dict):
                        rows = segment["observations"]
                        metadata = segment["metadata"]
                    else:
                        rows = getattr(segment, "states", getattr(segment, "observations", ()))
                        metadata = asdict(segment.metadata)
                    total += len(rows)
                    if len(previews) < 5:
                        previews.append({"metadata": metadata, "rows": len(rows)})
                summary = {"segment_count": len(value.segments), "row_count": total, "segment_preview": previews}
            elif hasattr(value, "records"):
                summary = {
                    "kind": value.kind,
                    "version": value.version,
                    "record_count": len(value.records),
                    "observation_types": value.observation_types,
                    "header_line_count": len(value.header),
                }
            elif isinstance(value, dict):
                summary = {
                    "schema": value.get("schema"),
                    "object_id": value.get("object_id"),
                    "maneuver_count": len(value.get("maneuvers", [])),
                }
            elif is_dataclass(value):
                summary = asdict(value)
            else:
                raise DataIOError("Unknown data model.")
            if source_receipt(args.source) != source:
                raise DataIOError("Input changed during inspection.")
            result = {
                "status": "inspected",
                "format": args.format,
                "source": source,
                "summary": summary,
                "simulation_executed": False,
            }
        elif args.command == "convert":
            result = convert(
                args.source,
                args.output,
                input_format=args.input_format,
                output_format=args.output_format,
                options=options,
            )
        elif args.command == "export-review":
            result = export_review(
                args.completed_run,
                args.output,
                format=args.format,
                object_id=args.object_id,
                start_s=args.start_s,
                stop_s=args.stop_s,
                options=options,
            )
        elif args.command == "export-table":
            from .tables import export_review_table

            result = export_review_table(
                args.completed_run, args.output, query=args.query, format=args.format, max_rows=args.max_rows
            )
        elif args.command == "materialize":
            from .catalog import read_catalog
            from .materialize import materialize_scenario
            from .time import parse_time

            identity = source_receipt(args.source)
            materialization_sources = (args.source,)
            materialization_identities = [identity]

            if args.format in {"tle", "gp-json", "gp-csv", "omm-kvn", "omm-xml"}:
                if any(options.get(key) for key in ("target_frame", "target_center", "spice_target_frame")):
                    raise DataIOError(
                        "Catalog materialization uses native TEME OGP states; frame conversion is a separate operation."
                    )
                value = read_catalog(
                    args.source, format=args.format, provider_profile=options.get("provider_profile", "celestrak-gp")
                )
            else:
                from .objects.importers import import_object

                imported = import_object(args.source, format=args.format, options=options)
                value = imported.value
                materialization_sources = tuple(row["path"] for row in imported.sources)
                materialization_identities = list(imported.sources)
            result = materialize_scenario(
                value,
                args.output,
                scenario_name=args.scenario_name,
                output_dir=args.run_output,
                duration_s=args.duration_s,
                step_s=args.step_s,
                start_epoch=None if args.start_epoch is None else parse_time(args.start_epoch),
                source_artifacts=materialization_sources,
                expected_source_identities=materialization_identities,
            )
        elif args.command == "fit-review":
            import importlib.util

            if importlib.util.find_spec("sim.data_io.fit") is None:
                raise DataIOError(
                    "Review mean-element fitting requires the Pro implementation and tracking_od entitlement."
                )
            from .api import _eop
            from .catalog import read_catalog
            from .fit import fit_review_catalog

            template = read_catalog(args.template, format=args.template_format)
            if len(template) != 1:
                raise DataIOError("Select exactly one template mean-element record.")
            result = fit_review_catalog(
                args.completed_run,
                template=template[0],
                output_dir=args.output_dir,
                object_id=args.object_id,
                fit_until_s=args.fit_until_s,
                max_position_rms_m=args.max_position_rms_m,
                max_velocity_rms_m_s=args.max_velocity_rms_m_s,
                formats=tuple(args.formats),
                eop_series=_eop(options),
                max_nfev=args.max_nfev,
                estimate_bstar=args.estimate_bstar,
            )
        elif args.command == "sample":
            from .trajectory import TrajectorySampler

            value = as_trajectory(read_data(args.source, format=args.format, options=options), options=options)
            state = TrajectorySampler(value, max_gap_s=args.max_gap_s).state_at(args.time_s)
            result = {
                "position_km": state[:3].tolist(),
                "velocity_km_s": state[3:].tolist(),
                "frame": value.frame,
                "center": value.center,
                "method": "piecewise_cubic_Hermite",
                "simulation_executed": False,
            }
        elif args.command == "export-assessment":
            from .products import cdm_from_assessment

            identities = [source_receipt(p) for p in (args.evidence, args.problem)]
            message = cdm_from_assessment(args.evidence, args.problem, initial_epoch=args.initial_epoch)
            data, extra = encode_data(message, format=args.format)
            if [source_receipt(p) for p in (args.evidence, args.problem)] != identities:
                raise DataIOError("Assessment inputs changed during export.")
            result = publish(
                args.output,
                data,
                receipt={
                    "operation": "export-assessment",
                    "format": args.format,
                    "sources": identities,
                    "initial_epoch": args.initial_epoch,
                    **extra,
                },
                sources=(args.evidence, args.problem),
            )
        elif args.command == "export-measurements":
            from .navigation import encode_navigation
            from .products import tdm_from_measurements

            identity = source_receipt(args.source)
            rows = strict_json(read_text(args.source))
            message = tdm_from_measurements(rows, station_id=args.station_id, object_id=args.object_id)
            data = encode_navigation(message, format=args.format)
            if source_receipt(args.source) != identity:
                raise DataIOError("Measurement input changed during export.")
            result = publish(
                args.output,
                data,
                receipt={
                    "operation": "export-measurements",
                    "format": args.format,
                    "source": identity,
                    "station_id": args.station_id,
                    "object_id": args.object_id,
                },
                sources=(args.source,),
            )
        elif args.command == "export-maneuvers":
            from .navigation import encode_navigation
            from .products import opm_from_review_maneuvers

            message, selection = opm_from_review_maneuvers(
                args.completed_run, object_id=args.object_id, sample_index=args.sample_index
            )
            result = publish(
                args.output,
                encode_navigation(message, format=args.format),
                receipt={"operation": "export-maneuvers", "format": args.format, "source": selection},
            )
        else:
            raise DataIOError("Unknown IO operation.")
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
        return 1 if result.get("status") == "rejected" else 0
    except (ValueError, TypeError, KeyError, OSError, ImportError, RuntimeError) as exc:
        print(json.dumps({"status": "error", "message": str(exc), "simulation_executed": False}), file=sys.stderr)
        return 2
