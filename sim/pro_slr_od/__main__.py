from __future__ import annotations

import argparse
import json
from pathlib import Path

from sim.pro_slr_od import (
    ProSlrOdError,
    ProSlrOdProblem,
    replay_evidence,
    run_pro_slr_od,
    validate_problem,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate, run, or replay a bounded ILRS/SLR orbit determination study.")
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate")
    validate.add_argument("problem", type=Path)
    run = commands.add_parser("run")
    run.add_argument("problem", type=Path)
    run.add_argument("--output-dir", type=Path, required=True)
    replay = commands.add_parser("replay")
    replay.add_argument("evidence_dir", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "validate":
            payload = validate_problem(ProSlrOdProblem.from_json(args.problem))
        elif args.command == "run":
            payload = run_pro_slr_od(ProSlrOdProblem.from_json(args.problem), args.output_dir)
        else:
            payload = replay_evidence(args.evidence_dir)
        print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
        return 0
    except (OSError, ValueError, ProSlrOdError) as exc:
        print(json.dumps({"status": "error", "message": str(exc)}, indent=2, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
