"""Public orbit determination API and bounded CCSDS TDM CLI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from sim.analysis.tracking_od import (
    TRACKING_OD_EVIDENCE_SCHEMA,
    TRACKING_OD_PROBLEM_SCHEMA,
    TrackingOdError,
    TrackingOdProblem,
    TrackingStation,
    assess_tdm_orbit_determination,
)
from sim.interchange.ccsds_tdm import (
    CCSDS_TDM_PROFILE,
    CcsdsTdmError,
    TdmHeader,
    TdmMessage,
    TdmMetadata,
    TdmObservation,
    TdmSegment,
    compare_tdm,
    inspect_tdm,
    parse_tdm_kvn,
    read_tdm_kvn,
    serialize_tdm_kvn,
    validate_tdm,
    write_tdm_kvn,
)
from sim.pro_tracking_od.precision import (
    PrecisionOrbitModel,
    PrecisionOrbitPredictor,
    fit_precision_orbit,
    predict_precision_orbit,
)
from sim.tracking_data import NORMALIZED_TRACKING_DATASET_SCHEMA, normalize_tdm_tracking_dataset

__all__ = [
    "CCSDS_TDM_PROFILE",
    "NORMALIZED_TRACKING_DATASET_SCHEMA",
    "TRACKING_OD_EVIDENCE_SCHEMA",
    "TRACKING_OD_PROBLEM_SCHEMA",
    "CcsdsTdmError",
    "TdmHeader",
    "TdmMessage",
    "TdmMetadata",
    "TdmObservation",
    "TdmSegment",
    "PrecisionOrbitModel",
    "PrecisionOrbitPredictor",
    "TrackingOdError",
    "TrackingOdProblem",
    "TrackingStation",
    "assess_tdm_orbit_determination",
    "compare_tdm",
    "fit_precision_orbit",
    "inspect_tdm",
    "normalize_tdm_tracking_dataset",
    "parse_tdm_kvn",
    "predict_precision_orbit",
    "read_tdm_kvn",
    "serialize_tdm_kvn",
    "validate_tdm",
    "write_tdm_kvn",
]


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TrackingOdError(f"Could not read tracking-OD problem {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise TrackingOdError("A tracking-OD problem must be a JSON object.")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m sim.tracking_od",
        description="Inspect bounded CCSDS TDM KVN and run public fit/holdout orbit determination.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    inspect_parser = commands.add_parser("inspect-tdm", help="Validate and inspect one bounded TDM 2.0 KVN file.")
    inspect_parser.add_argument("path", type=Path)
    roundtrip = commands.add_parser("roundtrip-tdm", help="Parse and canonically serialize one bounded TDM.")
    roundtrip.add_argument("source", type=Path)
    roundtrip.add_argument("destination", type=Path)
    fit = commands.add_parser("fit", help="Fit one bounded TDM using a tracking-OD problem JSON file.")
    fit.add_argument("tdm", type=Path)
    fit.add_argument("problem", type=Path)
    fit.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "inspect-tdm":
            payload: Any = inspect_tdm(args.path)
        elif args.command == "roundtrip-tdm":
            message = read_tdm_kvn(args.source)
            target = write_tdm_kvn(message, args.destination)
            payload = {"status": "written", "path": str(target), "comparison": compare_tdm(message, target)}
        else:
            message = read_tdm_kvn(args.tdm)
            problem = TrackingOdProblem.from_mapping(_read_json_object(args.problem))
            payload = assess_tdm_orbit_determination(message, problem, output_dir=args.output_dir)
        print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
        return 0
    except (CcsdsTdmError, TrackingOdError, OSError, ValueError) as exc:
        print(json.dumps({"status": "error", "message": str(exc)}, indent=2, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
