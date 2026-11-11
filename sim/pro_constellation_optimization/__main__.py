"""Command-line interface for bounded local OEL constellation optimization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sim.pro_constellation_optimization import (
    ProConstellationOptimizationError,
    ProConstellationOptimizationProblem,
    replay_evidence,
    run_pro_constellation_optimization,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate, solve, resume, or replay a bounded constellation-design search."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate", help="Strictly validate and normalize a problem JSON file.")
    validate.add_argument("problem", type=Path)
    for command in ("solve", "resume"):
        operation = commands.add_parser(command, help=f"{command.title()} a bounded design search.")
        operation.add_argument("problem", type=Path)
        operation.add_argument("--output", type=Path, required=True)
        operation.add_argument(
            "--max-new-evaluations",
            type=int,
            help="Stop after this many new authoritative evaluations and retain a resumable checkpoint.",
        )
    replay = commands.add_parser("replay", help="Verify receipts and rerun a completed evidence packet.")
    replay.add_argument("output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "validate":
            problem = ProConstellationOptimizationProblem.from_json(args.problem)
            payload = {"status": "valid", "problem": problem.to_dict()}
        elif args.command == "replay":
            payload = replay_evidence(args.output)
        else:
            problem = ProConstellationOptimizationProblem.from_json(args.problem)
            payload = run_pro_constellation_optimization(
                problem,
                args.output,
                resume=args.command == "resume",
                max_new_evaluations=args.max_new_evaluations,
            )
    except (OSError, ProConstellationOptimizationError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
