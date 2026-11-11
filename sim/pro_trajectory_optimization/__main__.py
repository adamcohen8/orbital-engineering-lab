"""Command-line entry point for bounded local trajectory optimization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sim.pro_trajectory_optimization import (
    TrajectoryOptimizationProblem,
    replay_evidence,
    solve_trajectory_optimization,
    write_evidence,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate, solve, or replay a bounded OEL trajectory optimization.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate", help="Strictly validate and normalize a problem JSON file.")
    validate.add_argument("problem", type=Path)
    solve = subparsers.add_parser("solve", help="Solve a problem and write a complete evidence packet.")
    solve.add_argument("problem", type=Path)
    solve.add_argument("--output", type=Path, required=True)
    replay = subparsers.add_parser("replay", help="Verify packet integrity and authoritatively replay its solution.")
    replay.add_argument("output", type=Path)
    args = parser.parse_args(argv)
    if args.command == "validate":
        problem = TrajectoryOptimizationProblem.from_json(args.problem)
        print(json.dumps({"status": "valid", "problem": problem.to_dict()}, indent=2, sort_keys=True))
        return 0
    if args.command == "solve":
        result = solve_trajectory_optimization(TrajectoryOptimizationProblem.from_json(args.problem))
        manifest = write_evidence(result, args.output)
        print(
            json.dumps(
                {"status": result["status"], "output": str(args.output.resolve()), "manifest": manifest},
                indent=2,
                sort_keys=True,
            )
        )
        return 0 if result["converged"] else 2
    print(json.dumps(replay_evidence(args.output), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
