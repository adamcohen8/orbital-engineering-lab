from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from sim.pro_tracking_od import (
    ProTrackingOdError,
    ProTrackingOdProblem,
    replay_evidence,
    run_pro_tracking_od,
    validate_problem,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate, run, or replay a local tracking OD study.")
    parser.add_argument("--native-dir", type=Path)
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate")
    validate.add_argument("problem", type=Path)
    run = commands.add_parser("run")
    run.add_argument("problem", type=Path)
    run.add_argument("--output-dir", type=Path, required=True)
    replay = commands.add_parser("replay")
    replay.add_argument("evidence_dir", type=Path)
    for name in ("validate-continuous", "run-continuous"):
        command = commands.add_parser(name)
        command.add_argument("problem", type=Path)
        if name == "run-continuous":
            command.add_argument("--output-dir", type=Path, required=True)
            command.add_argument("--append", action="store_true")
    batch = commands.add_parser("batch-continuous")
    batch.add_argument("problem", type=Path)
    batch.add_argument("--output", type=Path, required=True)
    continuous_replay = commands.add_parser("replay-continuous")
    continuous_replay.add_argument("evidence_dir", type=Path)
    continuous_replay.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.native_dir:
        sys.path.insert(0, str(args.native_dir.resolve()))
    try:
        if args.command.endswith("continuous"):
            from sim.pro_tracking_od.continuous import replay_continuous_tracking_od, run_continuous_tracking_od
            from sim.pro_tracking_od.continuous import validate_problem as validate_continuous

            if args.command == "replay-continuous":
                payload = replay_continuous_tracking_od(args.evidence_dir, args.output_dir)
            else:
                problem = json.loads(args.problem.read_text())
                if args.command == "batch-continuous":
                    from sim.pro_tracking_od.batch import fit_tracking_batch

                    payload = fit_tracking_batch(problem, output_path=args.output)
                elif args.command == "validate-continuous":
                    rows, groups, _d, _m, duplicates = validate_continuous(problem)
                    payload = {
                        "status": "valid",
                        "measurements": len(rows),
                        "epochs": len(groups),
                        "duplicates": duplicates,
                    }
                else:
                    payload = run_continuous_tracking_od(problem, args.output_dir, append=args.append)
        elif args.command == "validate":
            payload = validate_problem(ProTrackingOdProblem.from_json(args.problem))
        elif args.command == "run":
            payload = run_pro_tracking_od(ProTrackingOdProblem.from_json(args.problem), args.output_dir)
        else:
            payload = replay_evidence(args.evidence_dir)
        print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
        return 0
    except (OSError, ValueError, ProTrackingOdError) as exc:
        print(json.dumps({"status": "error", "message": str(exc)}, indent=2, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
