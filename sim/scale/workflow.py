from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Callable

from sim.scale._output import bind_output_writer, store_read_path
from sim.scale.handoff import run_onp_handoff
from sim.scale.onp_execution import run_onp_refinement_queue
from sim.scale.onp_refinement import run_onp_refinement_materialization
from sim.scale.planner import enforce_scale_planner_limits, load_scale_planner_config
from sim.scale.propagation import run_ogp_propagation
from sim.scale.recompute import run_recompute_plan, run_recompute_propagation, run_recompute_screening
from sim.scale.refinement import run_candidate_refinement
from sim.scale.screening import run_pair_screening
from sim.security.output_writes import OutputWriter


@enforce_scale_planner_limits("catalog workflow")
def run_catalog_workflow(
    config_path: str | Path,
    *,
    writer: OutputWriter | None = None,
) -> dict[str, Any]:
    cfg = load_scale_planner_config(config_path)
    if cfg.mode != "run":
        raise ValueError("Catalog workflow execution requires scale_catalog.mode: run.")

    steps: list[dict[str, Any]] = []
    propagation = run_ogp_propagation(config_path, writer=writer)
    steps.append(_step("propagate_ogp", propagation))
    screening = run_pair_screening(config_path, writer=writer)
    steps.append(_step("screen_pairs", screening))
    refinement = run_candidate_refinement(config_path, writer=writer)
    steps.append(_step("refine_candidates", refinement))
    handoff = run_onp_handoff(config_path, writer=writer)
    steps.append(_step("handoff_hpop", handoff))
    final_plan = run_recompute_plan(config_path, writer=writer)
    steps.append(_step("final_recompute_health", final_plan))

    payload = _workflow_payload(
        cfg,
        workflow_type="catalog",
        status=_overall_status(steps),
        steps=steps,
        final_plan=final_plan,
        non_claims=[
            "Catalog workflow runs ordered OGP propagation, pair screening, candidate refinement, and ONP handoff packet writing.",
            "It does not run Monte Carlo or sensitivity campaigns; those are separate selected-object campaign workflows.",
            "ONP handoff remains manifest-only and does not execute ONP propagation.",
        ],
    )
    _write_workflow_artifacts(cfg, payload, stem="scale_catalog_workflow", writer=writer)
    return payload


@enforce_scale_planner_limits("recompute workflow")
def run_recompute_workflow(
    config_path: str | Path,
    *,
    writer: OutputWriter | None = None,
) -> dict[str, Any]:
    cfg = load_scale_planner_config(config_path)
    if cfg.mode != "run":
        raise ValueError("Recompute workflow execution requires scale_catalog.mode: run.")

    steps: list[dict[str, Any]] = []
    initial_plan = run_recompute_plan(config_path, writer=writer)
    steps.append(_step("initial_recompute_plan", initial_plan))

    propagation_tasks = list(dict(initial_plan.get("propagation", {}) or {}).get("tasks", []) or [])
    if propagation_tasks:
        propagation = run_recompute_propagation(config_path, writer=writer)
    else:
        propagation = {"status": "completed", "execution_status": "not_needed", "reason": "no propagation tasks"}
    steps.append(_step("run_recompute_propagation", propagation))

    post_propagation_plan = run_recompute_plan(config_path, writer=writer)
    steps.append(_step("post_propagation_plan", post_propagation_plan))
    post_screening = dict(post_propagation_plan.get("screening", {}) or {})
    if int(post_screening.get("total_rescreen_work_count", 0) or 0) > 0:
        screening = run_recompute_screening(config_path, writer=writer)
    else:
        screening = {"status": "completed", "execution_status": "not_needed", "reason": "no rescreening work"}
    steps.append(_step("run_recompute_screening", screening))

    post_screening_plan = run_recompute_plan(config_path, writer=writer)
    steps.append(_step("post_screening_plan", post_screening_plan))
    post_candidates = dict(post_screening_plan.get("candidates", {}) or {})
    if int(post_candidates.get("pending_refinement_count", 0) or 0) > 0:
        refinement = run_candidate_refinement(config_path, writer=writer)
    else:
        refinement = {
            "status": "completed",
            "execution_status": "not_needed",
            "reason": "no pending current-candidate refinements",
        }
    steps.append(_step("refine_current_candidates", refinement))

    post_refinement_plan = run_recompute_plan(config_path, writer=writer)
    steps.append(_step("post_refinement_plan", post_refinement_plan))
    post_handoff = dict(
        post_refinement_plan.get("onp_handoff", post_refinement_plan.get("hpop_handoff", {})) or {}
    )
    if cfg.onp_handoff_enabled and int(post_handoff.get("pending_packet_count", 0) or 0) > 0:
        handoff = run_onp_handoff(config_path, writer=writer)
    else:
        handoff = {
            "status": "completed",
            "execution_status": "not_needed",
            "reason": "no pending current ONP handoff packets",
        }
    steps.append(_step("write_current_hpop_handoff", handoff))

    final_plan = run_recompute_plan(config_path, writer=writer)
    steps.append(_step("final_recompute_health", final_plan))

    payload = _workflow_payload(
        cfg,
        workflow_type="recompute",
        status=_overall_status(steps),
        steps=steps,
        final_plan=final_plan,
        non_claims=[
            "Recompute workflow runs only targeted propagation and screening work selected by the scale store.",
            "It does not perform full-catalog all-pairs rescreening unless the planner reports that work.",
            "Maneuver flags remain advisory recompute triggers, not authoritative maneuver attribution.",
        ],
    )
    _write_workflow_artifacts(cfg, payload, stem="scale_recompute_workflow", writer=writer)
    return payload


@enforce_scale_planner_limits("refresh through ONP workflow")
def run_refresh_to_onp_workflow(
    config_path: str | Path,
    *,
    workers: int = 1,
    backend: str | None = None,
    step_callback: Callable[[int, int, str], None] | None = None,
    writer: OutputWriter | None = None,
) -> dict[str, Any]:
    """Run targeted refresh through materialized/executed ONP evidence in one command."""

    cfg = load_scale_planner_config(config_path)
    if cfg.mode != "run":
        raise ValueError("Refresh-to-ONP execution requires scale_catalog.mode: run.")
    _workflow_progress(step_callback, 0, 4, "Starting content-bound Scale refresh.")
    before = _scale_evidence_snapshot(cfg.store_path, writer=writer)
    recompute = run_recompute_workflow(config_path, writer=writer)
    _workflow_progress(step_callback, 1, 4, "Completed targeted recompute and ONP handoff planning.")
    handoff_waves: list[dict[str, Any]] = []
    for _ in range(1000):
        plan = run_recompute_plan(config_path, writer=writer)
        handoff = dict(plan.get("onp_handoff", plan.get("hpop_handoff", {})) or {})
        if int(handoff.get("pending_packet_count", 0) or 0) <= 0:
            break
        handoff_waves.append(run_onp_handoff(config_path, writer=writer))
        _workflow_progress(step_callback, 1, 4, "Drained one bounded ONP handoff wave.")
    else:
        raise ValueError("Refresh-to-ONP handoff draining exceeded 1000 bounded waves.")
    materialization_waves: list[dict[str, Any]] = []
    execution_waves: list[dict[str, Any]] = []
    for _ in range(1000):
        materialized = run_onp_refinement_materialization(config_path, writer=writer)
        if int(materialized.get("packets_seen", 0) or 0) <= 0:
            break
        materialization_waves.append(materialized)
        _workflow_progress(step_callback, 2, 4, "Materialized one bounded ONP refinement wave.")
        execution_waves.append(
            run_onp_refinement_queue(config_path, workers=workers, backend=backend, writer=writer)
        )
        _workflow_progress(step_callback, 3, 4, "Executed one bounded ONP refinement wave.")
    else:
        raise ValueError("Refresh-to-ONP materialization draining exceeded 1000 bounded waves.")
    materialization = _aggregate_waves(
        materialization_waves,
        count_keys=("packets_seen", "cases_materialized", "cases_inserted", "cases_updated", "validated_cases"),
    )
    execution = _aggregate_waves(
        execution_waves,
        count_keys=("cases_claimed", "cases_completed", "cases_failed", "incomplete_claims_recovered"),
    )
    final_plan = run_recompute_plan(config_path, writer=writer)
    _workflow_progress(step_callback, 4, 4, "Completed final Scale refresh health inspection.")
    after = _scale_evidence_snapshot(cfg.store_path, writer=writer)
    comparison = _compare_scale_snapshots(before, after)
    comparison_path = cfg.output_dir / "scale_refresh_to_onp_comparison.json"
    comparison_binding = bind_output_writer(writer, cfg.output_dir)
    comparison_text = json.dumps(comparison, indent=2, sort_keys=True) + "\n"
    if comparison_binding is None:
        comparison_path.parent.mkdir(parents=True, exist_ok=True)
        comparison_path.write_text(comparison_text, encoding="utf-8")
    else:
        comparison_binding.write_text(comparison_path, comparison_text)
    steps = [
        _step("recompute_to_onp_handoff", recompute),
        _step("materialize_onp_refinement", materialization),
        _step("execute_onp_refinement", execution),
        _step("final_recompute_health", final_plan),
    ]
    payload = _workflow_payload(
        cfg,
        workflow_type="refresh_to_onp",
        status=_overall_status(steps),
        steps=steps,
        final_plan=final_plan,
        non_claims=[
            "This workflow executes only planner-selected targeted refresh and current ONP refinement cases.",
            "OGP-vs-ONP differences are model comparisons, not collision probability or flight-safety claims.",
        ],
    )
    payload["before"] = before
    payload["after"] = after
    payload["comparison"] = comparison
    payload["handoff_drain_waves"] = handoff_waves
    payload["materialization_waves"] = materialization_waves
    payload["execution_waves"] = execution_waves
    _write_workflow_artifacts(
        cfg,
        payload,
        stem="scale_refresh_to_onp_workflow",
        writer=writer,
        extra_artifacts={"comparison_json": str(comparison_path)},
    )
    return payload


def _workflow_progress(
    callback: Callable[[int, int, str], None] | None,
    completed: int,
    total: int,
    message: str,
) -> None:
    if callback is not None:
        callback(completed, total, message)


def _step(name: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": name,
        "canonical_name": {
            "write_current_hpop_handoff": "write_current_onp_handoff",
        }.get(name, name),
        "status": str(payload.get("status", "")),
        "execution_status": str(payload.get("execution_status", "")),
        "summary": _step_summary(payload),
    }


def _step_summary(payload: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "status",
        "execution_status",
        "products_inserted",
        "samples_inserted",
        "pair_windows_inserted",
        "candidate_promotions_inserted",
        "refinements_inserted",
        "packets_inserted",
        "reason",
    )
    summary = {key: payload[key] for key in keys if key in payload}
    if "propagation" in payload:
        propagation = dict(payload.get("propagation", {}) or {})
        for key in ("planned_task_count", "products_inserted", "samples_inserted", "recompute_chunk_count"):
            if key in propagation:
                summary[f"propagation_{key}"] = propagation[key]
    ogp_payload = payload.get("ogp")
    if not ogp_payload and "propagation" in payload:
        ogp_payload = dict(payload.get("propagation", {}) or {}).get("ogp")
    if ogp_payload:
        ogp = dict(ogp_payload or {})
        counts = dict(ogp.get("object_counts", {}) or {})
        for key in ("total", "sgp4", "sdp4"):
            if key in counts:
                summary[f"ogp_object_count_{key}"] = counts[key]
        if ogp.get("sgp4_backend"):
            summary["ogp_sgp4_backend"] = ogp["sgp4_backend"]
        if ogp.get("sdp4_backend"):
            summary["ogp_sdp4_backend"] = ogp["sdp4_backend"]
    if "screening" in payload:
        screening = dict(payload.get("screening", {}) or {})
        for key in ("total_rescreen_work_count", "pending_rescreen_product_pair_count", "affected_pair_window_count"):
            if key in screening:
                summary[f"screening_{key}"] = screening[key]
    if "candidates" in payload:
        candidates = dict(payload.get("candidates", {}) or {})
        for key in ("pending_refinement_count", "current_candidate_count"):
            if key in candidates:
                summary[f"candidates_{key}"] = candidates[key]
    if "hpop_handoff" in payload:
        handoff = dict(payload.get("hpop_handoff", {}) or {})
        for key in ("pending_packet_count", "stale_packet_count"):
            if key in handoff:
                summary[f"hpop_handoff_{key}"] = handoff[key]
                summary[f"onp_handoff_{key}"] = handoff[key]
    return summary


def _overall_status(steps: list[dict[str, Any]]) -> str:
    warning_statuses = {"warning"}
    good_statuses = {"completed", "dry_run"}
    statuses = {str(step.get("status", "")) for step in steps}
    if statuses - good_statuses - warning_statuses:
        return "warning"
    if statuses & warning_statuses:
        return "warning"
    return "completed"


def _workflow_payload(
    cfg: Any,
    *,
    workflow_type: str,
    status: str,
    steps: list[dict[str, Any]],
    final_plan: dict[str, Any],
    non_claims: list[str],
) -> dict[str, Any]:
    final_invalidation = dict(final_plan.get("invalidation", {}) or {})
    final_screening = dict(final_plan.get("screening", {}) or {})
    final_candidates = dict(final_plan.get("candidates", {}) or {})
    final_handoff = dict(final_plan.get("onp_handoff", final_plan.get("hpop_handoff", {})) or {})
    return {
        "schema_version": 1,
        "status": status,
        "workflow_type": workflow_type,
        "workflow": {
            "name": cfg.name,
            "config_path": str(cfg.config_path),
            "mode": cfg.mode,
        },
        "store_path": str(cfg.store_path),
        "steps": steps,
        "final_health": {
            "open_invalidations": int(final_invalidation.get("open_items", 0) or 0),
            "recompute_chunks": int(dict(final_plan.get("propagation", {}) or {}).get("recompute_chunk_count", 0) or 0),
            "rescreen_work": int(final_screening.get("total_rescreen_work_count", 0) or 0),
            "pending_refinements": int(final_candidates.get("pending_refinement_count", 0) or 0),
            "pending_hpop_packets": int(final_handoff.get("pending_packet_count", 0) or 0),
            "stale_hpop_packets": int(final_handoff.get("stale_packet_count", 0) or 0),
            "current_candidates": int(final_candidates.get("current_candidate_count", 0) or 0),
            "pending_onp_packets": int(final_handoff.get("pending_packet_count", 0) or 0),
            "stale_onp_packets": int(final_handoff.get("stale_packet_count", 0) or 0),
        },
        "non_claims": non_claims,
        "artifacts": {},
    }


def _write_workflow_artifacts(
    cfg: Any,
    payload: dict[str, Any],
    *,
    stem: str,
    writer: OutputWriter | None = None,
    extra_artifacts: dict[str, str] | None = None,
) -> None:
    json_path = cfg.output_dir / f"{stem}.json"
    md_path = cfg.output_dir / f"{stem}.md"
    artifacts = {
        "workflow_json": str(json_path),
        "workflow_markdown": str(md_path),
    }
    artifacts.update(dict(extra_artifacts or {}))
    payload["artifacts"] = artifacts
    serialized_json = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    serialized_markdown = _render_workflow_markdown(payload)
    binding = bind_output_writer(writer, cfg.output_dir)
    if binding is None:
        cfg.output_dir.mkdir(parents=True, exist_ok=True)
        json_path.write_text(serialized_json, encoding="utf-8")
        md_path.write_text(serialized_markdown, encoding="utf-8")
    else:
        binding.write_text(json_path, serialized_json)
        binding.write_text(md_path, serialized_markdown)


def _scale_evidence_snapshot(
    store_path: Path,
    *,
    writer: OutputWriter | None = None,
) -> dict[str, Any]:
    read_path = store_read_path(store_path, writer=writer)
    if not read_path.is_file():
        return {"store_exists": False, "tables": {}}
    queries = {
        "candidates": (
            "SELECT candidate_id, deputy_id, chief_id, tca_jd_utc, miss_distance_km, "
            "promotion_status, supersedes_candidate_id FROM conjunction_candidates ORDER BY created_utc, candidate_id"
        ),
        "refinements": (
            "SELECT refinement_id, candidate_id, refined_tca_jd_utc, refined_miss_distance_km, status "
            "FROM candidate_refinements ORDER BY created_utc, refinement_id"
        ),
        "onp_handoffs": (
            "SELECT packet_id, candidate_id, refinement_id, miss_distance_km, status "
            "FROM hpop_handoff_packets ORDER BY created_utc, packet_id"
        ),
        "onp_runs": (
            "SELECT execution_id, case_id, attempt, status, comparison_json "
            "FROM onp_refinement_runs ORDER BY claimed_utc, execution_id"
        ),
    }
    with sqlite3.connect(read_path) as conn:
        conn.row_factory = sqlite3.Row
        tables = {
            name: [dict(row) for row in conn.execute(sql).fetchall()]
            for name, sql in queries.items()
        }
    for row in tables["onp_runs"]:
        raw = row.get("comparison_json")
        row["comparison"] = json.loads(raw) if raw else None
        row.pop("comparison_json", None)
    return {"store_exists": True, "tables": tables}


def _compare_scale_snapshots(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {"schema_version": 1, "status": "completed", "tables": {}}
    before_tables = dict(before.get("tables", {}) or {})
    after_tables = dict(after.get("tables", {}) or {})
    identifiers = {
        "candidates": "candidate_id",
        "refinements": "refinement_id",
        "onp_handoffs": "packet_id",
        "onp_runs": "execution_id",
    }
    for name, identifier in identifiers.items():
        old_rows = list(before_tables.get(name, []) or [])
        new_rows = list(after_tables.get(name, []) or [])
        old_by_id = {str(row[identifier]): row for row in old_rows}
        new_by_id = {str(row[identifier]): row for row in new_rows}
        result["tables"][name] = {
            "before_count": len(old_rows),
            "after_count": len(new_rows),
            "added_ids": sorted(set(new_by_id) - set(old_by_id)),
            "removed_ids": sorted(set(old_by_id) - set(new_by_id)),
            "changed_ids": sorted(
                key for key in set(old_by_id) & set(new_by_id) if old_by_id[key] != new_by_id[key]
            ),
        }
    latest_runs = list(after_tables.get("onp_runs", []) or [])
    result["latest_onp_comparison"] = latest_runs[-1].get("comparison") if latest_runs else None
    return result


def _aggregate_waves(waves: list[dict[str, Any]], *, count_keys: tuple[str, ...]) -> dict[str, Any]:
    statuses = {str(row.get("status", "")) for row in waves}
    payload: dict[str, Any] = {
        "status": "completed" if not (statuses - {"completed"}) else "completed_with_errors",
        "wave_count": len(waves),
    }
    for key in count_keys:
        payload[key] = sum(int(row.get(key, 0) or 0) for row in waves)
    backends = {
        str(row.get("backend", "")).strip()
        for row in waves
        if str(row.get("backend", "")).strip()
    }
    if backends:
        payload["backends"] = sorted(backends)
    actual_backends = {
        str(backend).strip()
        for row in waves
        for backend in list(row.get("actual_backends", []) or [])
        if str(backend).strip()
    }
    if actual_backends:
        payload["actual_backends"] = sorted(actual_backends)
    return payload


def _render_workflow_markdown(payload: dict[str, Any]) -> str:
    workflow = dict(payload.get("workflow", {}) or {})
    health = dict(payload.get("final_health", {}) or {})
    lines = [
        f"# OEL Scale {str(payload.get('workflow_type', 'workflow')).title()} Workflow: {workflow.get('name', 'scale_workflow')}",
        "",
        f"- Status: `{payload.get('status')}`",
        f"- Config: `{workflow.get('config_path')}`",
        f"- Store: `{payload.get('store_path')}`",
        f"- Open invalidations: `{health.get('open_invalidations')}`",
        f"- Rescreen work: `{health.get('rescreen_work')}`",
        f"- Pending refinements: `{health.get('pending_refinements')}`",
        f"- Pending ONP packets: `{health.get('pending_hpop_packets')}`",
        f"- Stale ONP packets: `{health.get('stale_hpop_packets')}`",
        "",
        "## Steps",
        "",
    ]
    for step in payload.get("steps", []) or []:
        lines.append(f"- `{step.get('name')}`: `{step.get('status')}`")
    lines.extend(["", "## Non-Claims", ""])
    lines.extend(f"- {item}" for item in payload.get("non_claims", []) or [])
    return "\n".join(lines) + "\n"
