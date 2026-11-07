from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path
from typing import Any

from sim.agent_task.plot_recipes import get_plot_recipe, list_plot_recipes
from sim.agent_task.recipes import get_recipe, list_recipes
from sim.agent_task.runner import compare_configs, create_plot, inspect_output, run_recipe
from sim.agent_task.semantics import get_semantic_metric, list_semantic_metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run agent-safe OEL workflows and emit evidence packets.")
    sub = parser.add_subparsers(dest="command", required=True)

    list_parser = sub.add_parser("list", help="List bundled agent task recipes.")
    list_parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    list_parser.add_argument("--plots", action="store_true", help="List plot recipes instead of task recipes.")

    sem_parser = sub.add_parser("semantics", help="List or inspect semantic review metrics.")
    sem_parser.add_argument("metric", nargs="?", help="Metric name to inspect.")
    sem_parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")

    run_parser = sub.add_parser("run", help="Validate/run a bundled recipe and write an evidence packet.")
    run_parser.add_argument("recipe_id")
    run_parser.add_argument("--output-dir", help="Exact output directory for the run.")
    run_parser.add_argument("--output-root", help="Root directory; recipe id is appended.")
    run_parser.add_argument("--dry-run", action="store_true", help="Validate and write a packet without executing.")
    run_parser.add_argument("--plot", action="store_true", help="Generate recipe plots after a successful run.")
    run_parser.add_argument("--style", default="oel_dark", choices=("oel_dark", "oel_light"), help="Plot style.")
    run_parser.add_argument("--max-rows", type=int, default=50, help="Maximum rows per packet query.")
    run_parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")

    inspect_parser = sub.add_parser("inspect", help="Inspect an existing completed output directory.")
    inspect_parser.add_argument("output_dir")
    inspect_parser.add_argument("--query", action="append", dest="queries", help="Saved query name to include.")
    inspect_parser.add_argument("--max-rows", type=int, default=50)
    inspect_parser.add_argument("--json", action="store_true")

    compare_parser = sub.add_parser("compare", help="Run two configs and write an evidence-backed comparison packet.")
    compare_parser.add_argument("--base", required=True, help="Base scenario YAML.")
    compare_parser.add_argument("--candidate", required=True, help="Candidate scenario YAML.")
    compare_parser.add_argument("--output-dir", required=True, help="Comparison output directory.")
    compare_parser.add_argument("--metric", action="append", dest="metrics", help="Semantic metric to compare.")
    compare_parser.add_argument("--max-rows", type=int, default=50)
    compare_parser.add_argument("--json", action="store_true")

    plot_parser = sub.add_parser("plot", help="Generate a named plot from a review store.")
    plot_parser.add_argument("output_dir")
    plot_parser.add_argument("--recipe", required=True, help="Plot recipe id.")
    plot_parser.add_argument("--style", default="oel_dark", choices=("oel_dark", "oel_light"))
    plot_parser.add_argument("--format", default="png", choices=("png", "svg", "pdf"))
    plot_parser.add_argument("--artifact-id", default="")
    plot_parser.add_argument("--path", help="Optional exact output path.")
    plot_parser.add_argument("--json", action="store_true")

    args = parser.parse_args(argv)
    try:
        if args.command == "list":
            items = list_plot_recipes() if args.plots else list_recipes()
            payload = {"items": [item.to_dict() for item in items]}
            _print(payload, json_mode=args.json)
            return 0
        if args.command == "semantics":
            if args.metric:
                metric = get_semantic_metric(args.metric)
                if metric is None:
                    print(f"unknown semantic metric: {args.metric}", file=sys.stderr)
                    return 2
                payload = metric.to_dict()
            else:
                payload = {"metrics": [item.to_dict() for item in list_semantic_metrics()]}
            _print(payload, json_mode=args.json)
            return 0
        if args.command == "run":
            if get_recipe(args.recipe_id) is None:
                print(f"unknown recipe: {args.recipe_id}", file=sys.stderr)
                return 2
            with _operation_stdout(args.json):
                payload = run_recipe(
                    args.recipe_id,
                    output_dir=args.output_dir,
                    output_root=args.output_root,
                    dry_run=args.dry_run,
                    make_plots=args.plot,
                    style_name=args.style,
                    max_rows=args.max_rows,
                )
            _print_packet(payload, json_mode=args.json)
            return 0 if payload.get("status") in {"completed", "validated"} else 2
        if args.command == "inspect":
            with _operation_stdout(args.json):
                payload = inspect_output(
                    args.output_dir,
                    query_names=None if args.queries is None else tuple(args.queries),
                    max_rows=args.max_rows,
                )
            _print_packet(payload, json_mode=args.json)
            return 0 if payload.get("status") in {"completed", "partial"} else 2
        if args.command == "compare":
            with _operation_stdout(args.json):
                payload = compare_configs(
                    args.base,
                    args.candidate,
                    output_dir=args.output_dir,
                    metric_names=tuple(args.metrics or ()),
                    max_rows=args.max_rows,
                )
            _print_packet(payload, json_mode=args.json)
            return 0 if payload.get("status") == "completed" else 2
        if args.command == "plot":
            if get_plot_recipe(args.recipe) is None:
                print(f"unknown plot recipe: {args.recipe}", file=sys.stderr)
                return 2
            with _operation_stdout(args.json):
                payload = create_plot(
                    args.output_dir,
                    args.recipe,
                    style_name=args.style,
                    file_format=args.format,
                    artifact_id=args.artifact_id,
                    path=Path(args.path) if args.path else None,
                )
            _print(payload, json_mode=args.json)
            return 0
    except Exception as exc:
        print(f"agent task failed: {exc}", file=sys.stderr)
        return 2
    return 2


def _print(payload: dict[str, Any], *, json_mode: bool) -> None:
    if json_mode:
        print(json.dumps(payload, indent=2))
        return
    if "items" in payload:
        for item in payload["items"]:
            maturity = f" [{item.get('maturity')}]" if item.get("maturity") else ""
            print(f"{item.get('recipe_id')}: {item.get('title')}{maturity}")
        return
    if "metrics" in payload:
        for item in payload["metrics"]:
            units = f" ({item.get('units')})" if item.get("units") else ""
            maturity = f" [{item.get('maturity')}]" if item.get("maturity") else ""
            tables = ", ".join(item.get("source_tables") or ())
            source = f" tables={tables}" if tables else ""
            print(f"{item.get('name')}{units}{maturity}{source}: {item.get('description')}")
        return
    print(json.dumps(payload, indent=2))


def _operation_stdout(json_mode: bool):
    """Keep stdout machine-parseable while preserving simulation progress on stderr."""

    return contextlib.redirect_stdout(sys.stderr) if json_mode else contextlib.nullcontext()


def _print_packet(payload: dict[str, Any], *, json_mode: bool) -> None:
    if json_mode:
        print(json.dumps(payload, indent=2))
        return
    print(f"status: {payload.get('status')}")
    print(f"packet_path: {payload.get('packet_path')}")
    review = dict(payload.get("review", {}) or {})
    if review.get("db_path"):
        print(f"review_db: {review.get('db_path')}")
    if payload.get("failure_hints"):
        print("failure_hints:")
        for item in payload.get("failure_hints", []):
            print(f"- {item.get('code')}: {item.get('next_step')}")


if __name__ == "__main__":
    raise SystemExit(main())
