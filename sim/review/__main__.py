from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from sim.review import (
    ReviewQueryError,
    ReviewStoreNotFoundError,
    ReviewWorkspace,
    get_saved_review_query,
    list_saved_review_queries,
    load_workflow_manifest,
)


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if argv and argv[0] == "cr3bp":
        from sim.review.cr3bp import main as cr3bp_main

        return cr3bp_main(argv[1:])
    if argv and argv[0] == "plot":
        from sim.review.plot import main as plot_main

        return plot_main(argv[1:])
    if argv and argv[0] == "slice":
        from sim.review.slice import main as slice_main

        return slice_main(argv[1:])
    parser = argparse.ArgumentParser(description="Query an OEL output review store.")
    parser.add_argument("output_dir", nargs="?", help="Output directory or review/run.sqlite path.")
    query_group = parser.add_mutually_exclusive_group()
    query_group.add_argument("--query", "-q", help="Read-only SELECT query to run.")
    query_group.add_argument("--saved-query", help="Named built-in review query to run.")
    query_group.add_argument("--manifest", action="store_true", help="Print workflow review manifest summary.")
    query_group.add_argument("--list-artifacts", action="store_true", help="List workflow review artifacts.")
    parser.add_argument("--list-saved-queries", action="store_true", help="List built-in saved review queries and exit.")
    parser.add_argument("--max-rows", type=int, default=50, help="Maximum rows to print.")
    parser.add_argument(
        "--max-vm-steps",
        type=int,
        help="SQLite virtual-machine step budget (default: 250000, or the vetted saved-query budget).",
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    args = parser.parse_args(argv)

    if args.list_saved_queries:
        for item in list_saved_review_queries():
            tables = ",".join(item.source_tables)
            print(f"{item.name}: {item.description} [{item.maturity}; tables={tables}]")
        return 0

    if args.manifest or args.list_artifacts:
        if not args.output_dir:
            parser.error("output_dir is required when reading workflow review metadata")
        try:
            manifest = load_workflow_manifest(Path(args.output_dir))
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            print(f"review manifest failed: {exc}", file=sys.stderr)
            return 2
        if args.json:
            if args.list_artifacts:
                print(json.dumps({"artifacts": list(manifest.get("artifacts", []) or [])}, indent=2))
            else:
                print(json.dumps(manifest, indent=2))
        elif args.list_artifacts:
            _print_artifacts(list(manifest.get("artifacts", []) or []))
        else:
            _print_manifest(manifest)
        return 0

    sql = args.query
    saved_query = None
    if args.saved_query:
        saved_query = get_saved_review_query(args.saved_query)
        if saved_query is None:
            print(
                f"unknown saved review query: {args.saved_query}; "
                "run `python -m sim.review --list-saved-queries` to discover supported names",
                file=sys.stderr,
            )
            return 2
        sql = saved_query.sql
    if not sql:
        parser.error("one of --query, --saved-query, or --list-saved-queries is required")
    if not args.output_dir:
        parser.error("output_dir is required when running a query")

    try:
        with ReviewWorkspace.open(Path(args.output_dir)) as workspace:
            max_vm_steps = (
                max(int(args.max_vm_steps), 1)
                if args.max_vm_steps is not None
                else (saved_query.max_vm_steps if saved_query is not None else 250_000)
            )
            result = workspace.query(
                sql,
                max_rows=max(int(args.max_rows), 1),
                max_vm_steps=max_vm_steps,
            )
    except (ReviewStoreNotFoundError, ReviewQueryError) as exc:
        print(f"review query failed: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(
            json.dumps(
                {
                    "columns": result.columns,
                    "rows": _json_safe(result.rows),
                    "row_count": result.row_count,
                    "truncated": result.truncated,
                },
                indent=2,
            )
        )
    else:
        _print_table(result.columns, result.rows, truncated=result.truncated)
    return 0


def _print_table(columns: list[str], rows: list[dict[str, Any]], *, truncated: bool) -> None:
    if not columns:
        print("(no columns)")
        return
    widths = {col: len(col) for col in columns}
    for row in rows:
        for col in columns:
            widths[col] = min(max(widths[col], len(_fmt(row.get(col)))), 80)
    header = "  ".join(col.ljust(widths[col]) for col in columns)
    print(header)
    print("  ".join("-" * widths[col] for col in columns))
    for row in rows:
        print("  ".join(_fmt(row.get(col))[: widths[col]].ljust(widths[col]) for col in columns))
    if truncated:
        print("(truncated)")


def _print_manifest(manifest: dict[str, Any]) -> None:
    print(f"workflow_type: {manifest.get('workflow_type', '')}")
    print(f"scenario_name: {manifest.get('scenario_name', '')}")
    print(f"status: {manifest.get('status', '')}")
    print(f"generated_utc: {manifest.get('generated_utc', '')}")
    sqlite = str(manifest.get("sqlite", "") or "")
    if sqlite:
        print(f"sqlite: {sqlite}")
    queries = list(manifest.get("recommended_queries", []) or [])
    if queries:
        print("recommended_queries:")
        for item in queries:
            row = dict(item or {})
            print(f"- {row.get('name', '')}: {row.get('description', '')}")


def _print_artifacts(artifacts: list[Any]) -> None:
    if not artifacts:
        print("(no artifacts)")
        return
    for item in artifacts:
        row = dict(item or {})
        print(f"{row.get('artifact_key', '')}\t{row.get('artifact_type', '')}\t{row.get('path', '')}")


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.9g}"
    return str(value)


def _json_safe(value: Any) -> Any:
    """Represent opaque SQLite BLOBs without crashing or exposing their contents."""

    if isinstance(value, bytes):
        return {
            "type": "sqlite_blob",
            "byte_count": len(value),
            "sha256": hashlib.sha256(value).hexdigest(),
        }
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


if __name__ == "__main__":
    raise SystemExit(main())
