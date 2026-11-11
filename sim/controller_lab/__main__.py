from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from sim.controller_lab import run_controller_bench, validate_controller_bench_config
from sim.controller_lab.agent import (
    inspect_controller_bench_output,
    list_controller_bench_catalog,
    plan_controller_bench_from_intent,
    scaffold_controller_bench_suite,
)
from sim.controller_lab.engineering import analyze_completed_controller_bench, reduce_controller_bench_failure


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Agent-native controller-bench helper commands.")
    sub = parser.add_subparsers(dest="command", required=True)

    list_parser = sub.add_parser("list", help="List controller-bench catalog entries.")
    list_parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")

    plan_parser = sub.add_parser("plan", help="Rank catalog suites for a plain-language intent.")
    plan_parser.add_argument("intent", help="Plain-language controller-bench intent.")
    plan_parser.add_argument("--limit", type=int, default=5)
    plan_parser.add_argument("--json", action="store_true")

    scaffold_parser = sub.add_parser("scaffold", help="Create a new suite from a catalog entry.")
    scaffold_parser.add_argument("suite_id", help="Catalog suite id.")
    scaffold_parser.add_argument("--out", required=True, help="Destination suite YAML.")
    scaffold_parser.add_argument("--suite-name", default="", help="Optional replacement suite_name.")
    scaffold_parser.add_argument("--output-dir", default="", help="Optional replacement output_dir.")
    scaffold_parser.add_argument("--json", action="store_true")

    validate_parser = sub.add_parser("validate", help="Validate a controller-bench suite.")
    validate_parser.add_argument("suite", help="Suite YAML.")
    validate_parser.add_argument("--compare", nargs="*", default=None, help="Optional variant names.")
    validate_parser.add_argument("--json", action="store_true")

    run_parser = sub.add_parser("run", help="Run a controller-bench suite and emit an evidence packet.")
    run_parser.add_argument("suite", help="Suite YAML.")
    run_parser.add_argument("--compare", nargs="*", default=None, help="Optional variant names.")
    run_parser.add_argument("--json", action="store_true")

    inspect_parser = sub.add_parser("inspect", help="Inspect completed controller-bench output.")
    inspect_parser.add_argument("output_dir", help="Completed controller-bench output directory.")
    inspect_parser.add_argument("--query", action="append", dest="queries", help="Saved review query to include.")
    inspect_parser.add_argument("--max-rows", type=int, default=50)
    inspect_parser.add_argument("--json", action="store_true")

    diagnose_parser = sub.add_parser("diagnose", help="Validate and semantically lint a controller-bench suite.")
    diagnose_parser.add_argument("suite", help="Suite YAML.")
    diagnose_parser.add_argument("--compare", nargs="*", default=None, help="Optional variant names.")
    diagnose_parser.add_argument("--json", action="store_true")

    analyze_parser = sub.add_parser(
        "analyze",
        help="Regenerate failure-reduction and evidence-manifest artifacts for completed output.",
    )
    analyze_parser.add_argument("output_dir", help="Completed controller-bench output directory.")
    analyze_parser.add_argument("--json", action="store_true")

    evidence_parser = sub.add_parser("evidence", help="Export a reproducible evidence manifest for completed output.")
    evidence_parser.add_argument("output_dir", help="Completed controller-bench output directory.")
    evidence_parser.add_argument("--json", action="store_true")

    reduce_parser = sub.add_parser("reduce", help="Bisect a numeric initial condition to find the smallest observed failing case.")
    reduce_parser.add_argument("suite", help="Suite YAML.")
    reduce_parser.add_argument("--case", required=True, dest="case_name", help="Failing case name.")
    reduce_parser.add_argument("--variant", required=True, dest="variant_name", help="Failing variant name.")
    reduce_parser.add_argument("--path", default="", help="Numeric scenario path to scale; defaults to target relative_ric_rect.")
    reduce_parser.add_argument("--iterations", type=int, default=8)
    reduce_parser.add_argument("--json", action="store_true")

    args = parser.parse_args(argv)
    try:
        if args.command == "list":
            payload = list_controller_bench_catalog()
            _print(payload, json_mode=args.json)
            return 0
        if args.command == "plan":
            payload = plan_controller_bench_from_intent(args.intent, limit=args.limit)
            _print(payload, json_mode=args.json)
            return 0
        if args.command == "scaffold":
            payload = scaffold_controller_bench_suite(
                args.suite_id,
                args.out,
                suite_name=args.suite_name,
                output_dir=args.output_dir or None,
            )
            _print(payload, json_mode=args.json)
            return 0
        if args.command == "validate":
            payload = validate_controller_bench_config(args.suite, compare_names=list(args.compare or []))
            _print(payload, json_mode=args.json)
            return 0 if bool(payload.get("passed", False)) else 2
        if args.command == "run":
            payload = run_controller_bench(
                args.suite,
                compare_names=list(args.compare or []),
                show_progress=not bool(args.json),
            )
            if args.json:
                _print(payload, json_mode=True)
            else:
                artifacts = dict(payload.get("artifacts", {}) or {})
                print(f"completed: {payload.get('suite_name')}")
                print(f"summary: {artifacts.get('summary_md', '')}")
                print(f"evidence_packet: {artifacts.get('agent_evidence_packet_json', '')}")
            failed_runs = int(dict(payload.get("constraint_summary", {}) or {}).get("failed_run_count", 0) or 0)
            return 0 if failed_runs == 0 else 2
        if args.command == "inspect":
            payload = inspect_controller_bench_output(
                args.output_dir,
                query_names=tuple(args.queries or ()),
                max_rows=args.max_rows,
            )
            _print(payload, json_mode=args.json)
            return 0 if payload.get("status") in {"completed", "partial"} else 2
        if args.command == "diagnose":
            payload = validate_controller_bench_config(args.suite, compare_names=list(args.compare or []))
            _print(payload, json_mode=args.json)
            return 0 if bool(payload.get("passed", False)) else 2
        if args.command in {"analyze", "evidence"}:
            payload = analyze_completed_controller_bench(args.output_dir)
            _print(payload, json_mode=args.json)
            return 0
        if args.command == "reduce":
            payload = reduce_controller_bench_failure(
                args.suite,
                case_name=args.case_name,
                variant_name=args.variant_name,
                parameter_path=args.path,
                iterations=args.iterations,
            )
            _print(payload, json_mode=args.json)
            return 0
    except Exception as exc:
        print(f"controller bench helper failed: {exc}", file=sys.stderr)
        return 2
    return 2


def _print(payload: dict[str, Any], *, json_mode: bool) -> None:
    if json_mode:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    if "items" in payload:
        for item in payload["items"]:
            tags = ", ".join(item.get("intent_tags") or ())
            print(f"{item.get('suite_id')}: {item.get('title')} [{tags}]")
        return
    if "matches" in payload:
        for item in payload["matches"]:
            entry = dict(item.get("entry", {}) or {})
            print(f"{entry.get('suite_id')}: score={item.get('score')} path={entry.get('suite_path')}")
        return
    if "agent_evidence_packet" in payload or payload.get("task_type") == "controller_bench":
        evidence = dict(payload.get("evidence_summary", {}) or {})
        print(f"status: {payload.get('status')}")
        print(f"suite: {evidence.get('suite_name', '')}")
        print(f"top_variant_by_pass_rate: {evidence.get('top_variant_by_pass_rate', '')}")
        if payload.get("agent_evidence_packet"):
            print(f"packet: {payload.get('agent_evidence_packet')}")
        return
    if "passed" in payload and "suite_name" in payload:
        print(f"status: {'OK' if payload.get('passed') else 'FAILED'}")
        print(f"suite: {payload.get('suite_name')}")
        print(f"scheduled_runs: {payload.get('scheduled_runs')}")
        for error in list(payload.get("errors", []) or []):
            print(f"error: {error}")
        semantic_lint = dict(payload.get("semantic_lint", {}) or {})
        for finding in list(semantic_lint.get("findings", []) or []):
            print(f"{finding.get('severity', 'info')}: [{finding.get('code')}] {finding.get('message')}")
        return
    if "output_path" in payload:
        print(f"wrote: {payload.get('output_path')}")
        print(f"source: {payload.get('source_suite')}")
        return
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    raise SystemExit(main())
