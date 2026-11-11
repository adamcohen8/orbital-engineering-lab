from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sim.controller_lab.catalog import (
    list_catalog_entries,
    plan_controller_bench,
    scaffold_from_catalog,
)
from sim.review import ReviewQueryError, ReviewStoreNotFoundError, ReviewWorkspace, get_saved_review_query
from sim.review.manifest import load_workflow_manifest

CONTROLLER_BENCH_AGENT_PACKET_VERSION = 1
DEFAULT_CONTROLLER_BENCH_QUERIES = (
    "workflow_metadata",
    "controller_bench_runs",
    "controller_bench_leaderboard",
    "controller_bench_failures",
    "workflow_artifacts",
)


def list_controller_bench_catalog() -> dict[str, Any]:
    return {"items": [entry.to_dict() for entry in list_catalog_entries()]}


def plan_controller_bench_from_intent(intent: str, *, limit: int = 5) -> dict[str, Any]:
    return plan_controller_bench(intent, limit=limit)


def scaffold_controller_bench_suite(
    suite_id: str,
    output_path: str | Path,
    *,
    suite_name: str = "",
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    return scaffold_from_catalog(suite_id, output_path, suite_name=suite_name, output_dir=output_dir)


def inspect_controller_bench_output(
    output_dir: str | Path,
    *,
    query_names: tuple[str, ...] | list[str] | None = None,
    max_rows: int = 50,
    write_packet: bool = True,
) -> dict[str, Any]:
    outdir = Path(output_dir).expanduser().resolve()
    packet = _base_packet(outdir)
    query_names = tuple(query_names or DEFAULT_CONTROLLER_BENCH_QUERIES)
    manifest = _load_manifest(outdir)
    if manifest:
        packet["manifest"] = manifest
        packet["configs"].append({"source_config": manifest.get("source_config", ""), "output_dir": str(outdir)})
    summary = _load_summary(outdir)
    if summary:
        packet["summary"] = _summarize_controller_bench(summary)
        packet["artifacts"] = _artifact_rows(summary, outdir)
        packet["caveats"].extend(_catalog_caveats(summary))
    try:
        workspace = ReviewWorkspace.open(outdir)
        packet["review"] = {
            "db_path": str(workspace.db_path),
            "tables": workspace.tables(),
            "queries": _run_saved_queries(workspace, query_names, max_rows=max_rows),
        }
        packet["review"]["query_summary"] = _query_summary(packet["review"]["queries"])
    except (ReviewStoreNotFoundError, ReviewQueryError, ValueError) as exc:
        packet["status"] = "partial"
        packet["review"] = {"error": str(exc), "queries": []}
        packet["failure_hints"].append(
            {
                "code": "controller_bench_review_unavailable",
                "severity": "warning",
                "message": str(exc),
                "next_step": "Run the controller bench normally and inspect its review/run.sqlite output.",
            }
        )
    if not summary and not manifest:
        packet["status"] = "partial"
        packet["failure_hints"].append(
            {
                "code": "controller_bench_summary_missing",
                "severity": "warning",
                "message": f"No controller_bench_summary.json or workflow manifest found under {outdir}.",
                "next_step": "Check the output directory or rerun the controller bench.",
            }
        )
    packet["evidence_summary"] = _evidence_summary(packet)
    if write_packet:
        packet_path = outdir / "agent_evidence_packet.json"
        outdir.mkdir(parents=True, exist_ok=True)
        packet_path.write_text(json.dumps(packet, indent=2, sort_keys=True), encoding="utf-8")
        packet["agent_evidence_packet"] = str(packet_path)
    return packet


def write_controller_bench_agent_packet(result: dict[str, Any], outdir: str | Path) -> dict[str, str]:
    packet = inspect_controller_bench_output(outdir, write_packet=True)
    return {"agent_evidence_packet_json": str(packet.get("agent_evidence_packet", ""))}


def _base_packet(outdir: Path) -> dict[str, Any]:
    return {
        "schema_version": CONTROLLER_BENCH_AGENT_PACKET_VERSION,
        "task_type": "controller_bench",
        "status": "completed",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(outdir),
        "configs": [],
        "summary": {},
        "manifest": {},
        "review": {},
        "artifacts": [],
        "evidence_summary": {},
        "failure_hints": [],
        "caveats": [
            "Controller-bench evidence is comparative evidence for the configured variants, cases, objectives, and thresholds.",
            "Do not claim mission qualification, operational safety, or global optimality without additional validation evidence.",
        ],
    }


def _load_summary(outdir: Path) -> dict[str, Any]:
    path = outdir / "controller_bench_summary.json"
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return dict(data) if isinstance(data, dict) else {}


def _load_manifest(outdir: Path) -> dict[str, Any]:
    try:
        return load_workflow_manifest(outdir)
    except (FileNotFoundError, json.JSONDecodeError, ValueError):
        return {}


def _run_saved_queries(workspace: ReviewWorkspace, query_names: tuple[str, ...], *, max_rows: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for name in query_names:
        saved = get_saved_review_query(name)
        if saved is None:
            out.append({"name": name, "status": "unknown_query"})
            continue
        try:
            result = workspace.query(saved.sql, max_rows=max_rows)
            out.append(
                {
                    "name": saved.name,
                    "status": "ok",
                    "description": saved.description,
                    "sql": saved.sql,
                    "columns": result.columns,
                    "rows": result.rows,
                    "row_count": result.row_count,
                    "truncated": result.truncated,
                    "allow_empty": saved.allow_empty,
                    "empty_result": result.row_count == 0,
                }
            )
        except ReviewQueryError as exc:
            out.append({"name": saved.name, "status": "failed", "error": str(exc), "sql": saved.sql})
    return out


def _query_summary(queries: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "query_count": len(queries),
        "ok_count": sum(1 for item in queries if item.get("status") == "ok"),
        "failed_count": sum(1 for item in queries if item.get("status") == "failed"),
        "empty_unexpected_count": sum(
            1 for item in queries if item.get("status") == "ok" and item.get("empty_result") and not item.get("allow_empty")
        ),
    }


def _summarize_controller_bench(summary: dict[str, Any]) -> dict[str, Any]:
    variants = list(summary.get("variant_summaries", []) or [])
    best_by_pass = sorted(
        [dict(item) for item in variants],
        key=lambda item: (-float(item.get("pass_rate", 0.0) or 0.0), str(item.get("variant_name", ""))),
    )
    constraint = dict(summary.get("constraint_summary", {}) or {})
    return {
        "suite_name": summary.get("suite_name"),
        "description": summary.get("description"),
        "config_path": summary.get("config_path"),
        "controller_target": summary.get("controller_target"),
        "case_count": len(list(summary.get("cases", []) or [])),
        "variant_count": len(list(summary.get("variants", []) or [])),
        "run_count": len(list(summary.get("runs", []) or [])),
        "best_variants_by_pass_rate": best_by_pass[:5],
        "failure_mode_counts": dict(constraint.get("failure_mode_counts", {}) or {}),
        "suggestions_by_mode": dict(constraint.get("suggestions_by_mode", {}) or {}),
        "failed_run_count": int(constraint.get("failed_run_count", 0) or 0),
        "leaderboard_kinds": sorted(str(key) for key in dict(summary.get("leaderboards", {}) or {}).keys()),
        "optimization": dict(summary.get("optimization", {}) or {}),
        "semantic_lint": dict(summary.get("semantic_lint", {}) or {}),
        "engineering_run_count": sum(bool(dict(run or {}).get("engineering")) for run in list(summary.get("runs", []) or [])),
    }


def _artifact_rows(summary: dict[str, Any], outdir: Path) -> list[dict[str, Any]]:
    artifacts = dict(summary.get("artifacts", {}) or {})
    rows: list[dict[str, Any]] = []
    for artifact_id, path in sorted(artifacts.items()):
        path_txt = str(path or "")
        if not path_txt:
            continue
        artifact_path = Path(path_txt)
        if not artifact_path.is_absolute():
            artifact_path = outdir / artifact_path
        rows.append(
            {
                "artifact_id": artifact_id,
                "path": str(artifact_path),
                "exists": artifact_path.exists(),
            }
        )
    return rows


def _catalog_caveats(summary: dict[str, Any]) -> list[str]:
    cfg_path = str(summary.get("config_path") or "")
    root = Path(__file__).resolve().parents[2]
    caveats: list[str] = []
    for entry in list_catalog_entries():
        if cfg_path and Path(cfg_path).resolve() == (root / entry.suite_path).resolve():
            caveats.extend(entry.does_not_prove)
            break
    return caveats


def _evidence_summary(packet: dict[str, Any]) -> dict[str, Any]:
    summary = dict(packet.get("summary", {}) or {})
    best = list(summary.get("best_variants_by_pass_rate", []) or [])
    return {
        "status": packet.get("status"),
        "suite_name": summary.get("suite_name"),
        "run_count": summary.get("run_count"),
        "failed_run_count": summary.get("failed_run_count"),
        "top_variant_by_pass_rate": best[0].get("variant_name") if best else "",
        "top_pass_rate": best[0].get("pass_rate") if best else None,
        "failure_modes": summary.get("failure_mode_counts", {}),
        "query_summary": dict(packet.get("review", {}).get("query_summary", {}) or {}),
    }
