"""Versioned local Pro conjunction CLI; never executes a flight command."""

from __future__ import annotations

import argparse
import json

from sim.pro_conjunction.catalog import assess_catalog, catalog_status, optimize_catalog, snapshot
from sim.pro_conjunction.contracts import validate_problem
from sim.pro_conjunction.evidence import execute, inspect_bundle, read_json, replay_bundle, write_bundle


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m sim.pro_conjunction")
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate")
    validate.add_argument("problem")
    for name in ("assess", "sensitivity", "optimize"):
        sub = commands.add_parser(name)
        sub.add_argument("problem")
        sub.add_argument("--output", required=True)
        if name == "optimize":
            sub.add_argument("--trade", required=True)
        if name == "sensitivity":
            sub.add_argument("--scales", nargs="+", type=float, default=[0.5, 1, 2])
            sub.add_argument("--execution-scales", nargs="+", type=float, default=[0, 1, 2])
    for name in ("inspect", "replay", "catalog-status"):
        sub = commands.add_parser(name)
        sub.add_argument("bundle")
        if name == "catalog-status":
            sub.add_argument("--store", required=True)
    for name in ("catalog-snapshot", "catalog-assess", "catalog-optimize"):
        sub = commands.add_parser(name)
        sub.add_argument("--store", required=True)
        sub.add_argument("--candidate-id", required=True)
        if name != "catalog-snapshot":
            sub.add_argument("problem")
            sub.add_argument("--state-bindings", required=True)
            sub.add_argument("--ledger", required=True)
            if name == "catalog-optimize":
                sub.add_argument("--trade", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "validate":
            result = {"status": "valid", "problem": validate_problem(read_json(args.problem))}
        elif args.command in {"assess", "sensitivity", "optimize"}:
            inputs = {"problem": validate_problem(read_json(args.problem))}
            if args.command == "optimize":
                inputs["trade"] = read_json(args.trade)
            if args.command == "sensitivity":
                inputs.update(scales=args.scales, execution_scales=args.execution_scales)
            result = execute(args.command, inputs)
            manifest = write_bundle(args.output, args.command, inputs, result)
            result = {
                "status": result.get("status", "completed"),
                "bundle": args.output,
                "manifest_sha256": manifest["manifest_sha256"],
            }
        elif args.command == "inspect":
            verified = inspect_bundle(args.bundle)
            result = {"status": verified["status"], "manifest": verified["manifest"]}
        elif args.command == "replay":
            result = replay_bundle(args.bundle)
        elif args.command == "catalog-snapshot":
            result = snapshot(args.store, args.candidate_id)
        elif args.command == "catalog-status":
            result = catalog_status(args.bundle, args.store)
        elif args.command == "catalog-optimize":
            result = optimize_catalog(
                read_json(args.problem),
                read_json(args.trade),
                args.store,
                args.candidate_id,
                read_json(args.state_bindings),
                args.ledger,
            )
        else:
            result = assess_catalog(
                read_json(args.problem), args.store, args.candidate_id, read_json(args.state_bindings), args.ledger
            )
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
        return 0
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        print(json.dumps({"status": "error", "message": str(exc)}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
