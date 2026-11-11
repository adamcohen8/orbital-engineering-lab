from __future__ import annotations

import json
import sqlite3
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

from sim.scale._output import bind_output_writer, staged_sqlite_store, store_read_path
from sim.scale.planner import (
    _expected_propagation_windows,
    enforce_scale_planner_limits,
    load_scale_planner_config,
    resolve_selected_object_ids,
)
from sim.scale.propagation import build_sgp4_products_for_windows
from sim.scale.screening import build_pair_screening_results_for_product_pairs
from sim.scale.store import (
    PropagationProductSummary,
    complete_propagation_invalidations,
    count_current_conjunction_candidates,
    count_stale_hpop_handoff_packets,
    load_current_tle_states,
    pending_candidate_refinement_count,
    pending_hpop_handoff_count,
    summarize_store,
    write_propagation_products,
    write_screening_results,
)
from sim.security.output_writes import OutputWriter

OPEN_INVALIDATION_STATUSES = ("open", "planned", "claimed")


def run_recompute_plan(
    config_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    writer: OutputWriter | None = None,
) -> dict[str, Any]:
    cfg = load_scale_planner_config(
        config_path,
        output_dir=None if output_dir is None else Path(output_dir),
    )
    if cfg.mode != "run":
        raise ValueError("Recompute planning requires scale_catalog.mode: run.")
    logical_store_path = cfg.store_path
    cfg = replace(cfg, store_path=store_read_path(cfg.store_path, writer=writer))
    store_summary = summarize_store(cfg.store_path)
    configured_object_ids = resolve_selected_object_ids(cfg)
    current_states = load_current_tle_states(cfg.store_path, object_ids=configured_object_ids)
    selected_object_ids = tuple(state.object_id for state in current_states)
    invalidations = _load_open_invalidations(cfg.store_path, object_ids=selected_object_ids)
    changed_object_ids = tuple(sorted({item["object_id"] for item in invalidations if item.get("object_id")}))
    maneuver_object_ids = tuple(
        sorted({item["object_id"] for item in invalidations if item.get("reason") == "maneuver_detected" and item.get("object_id")})
    )
    expected_windows = _expected_propagation_windows(cfg)
    propagation_tasks = _build_propagation_tasks(
        cfg.store_path,
        object_ids=changed_object_ids,
        windows=expected_windows,
        model=cfg.propagation_model,
        backend=cfg.propagation_backend,
        output_frame=cfg.output_frame,
    )
    affected_windows = _load_affected_screening_windows(
        cfg.store_path,
        object_ids=changed_object_ids,
        windows=expected_windows,
    )
    affected_candidates = _load_affected_candidates(
        cfg.store_path,
        screening_window_ids=tuple(item["screening_window_id"] for item in affected_windows),
        object_ids=maneuver_object_ids,
    )
    affected_handoff = _load_affected_hpop_packets(
        cfg.store_path,
        candidate_ids=tuple(item["candidate_id"] for item in affected_candidates),
    )
    recompute_screening_pairs = _load_recompute_screening_product_pairs(cfg)
    invalidation_reasons = Counter(str(item["reason"]) for item in invalidations)
    affected_pairs = tuple(
        sorted({tuple(sorted((str(item["deputy_id"]), str(item["chief_id"])))) for item in affected_windows})
    )
    rescreen_pair_count = len(affected_windows) + len(recompute_screening_pairs)
    pending_refinement_count = pending_candidate_refinement_count(
        cfg.store_path, object_ids=configured_object_ids
    )
    pending_handoff_count = pending_hpop_handoff_count(
        cfg.store_path, object_ids=configured_object_ids
    )
    stale_hpop_packet_count = count_stale_hpop_handoff_packets(
        cfg.store_path, object_ids=configured_object_ids
    )
    planned_jobs = [
        {
            "job_type": "recompute_ogp_chunk",
            "aliases": ["recompute_sgp4_chunk"],
            "count": len(propagation_tasks),
            "status": "planned" if propagation_tasks else "not_needed",
            "reason": "current object state changed",
        },
        {
            "job_type": "review_maneuver_advisory",
            "count": len(maneuver_object_ids),
            "status": "planned" if maneuver_object_ids else "not_needed",
            "reason": "maneuver_detected invalidations require analyst review and targeted recompute",
        },
        {
            "job_type": "recompute_screen_pair_window",
            "count": rescreen_pair_count,
            "status": "planned" if rescreen_pair_count else "not_needed",
            "reason": "existing screened pair/window depends on a changed object or superseded propagation product",
        },
        {
            "job_type": "refresh_candidate_rankings",
            "count": 1 if affected_candidates else 0,
            "status": "planned" if affected_candidates else "not_needed",
            "reason": "candidate rows depend on affected screening windows",
        },
        {
            "job_type": "refine_current_candidate_tca",
            "count": pending_refinement_count,
            "status": "planned" if pending_refinement_count else "not_needed",
            "reason": "current candidates without completed refinement need refreshed TCA evidence",
        },
        {
            "job_type": "write_current_hpop_handoff",
            "count": pending_handoff_count,
            "status": "planned" if pending_handoff_count else "not_needed",
            "reason": "current refined candidates without active ONP handoff packets need new handoff evidence",
        },
        {
            "job_type": "review_stale_hpop_handoff",
            "count": stale_hpop_packet_count,
            "status": "planned" if stale_hpop_packet_count else "not_needed",
            "reason": "stale handoff packets remain as historical evidence and should not be used as current work",
        },
    ]
    payload = {
        "schema_version": 1,
        "status": "completed",
        "workflow": {
            "name": cfg.name,
            "config_path": str(cfg.config_path),
            "mode": cfg.mode,
        },
        "store": {
            "path": str(cfg.store_path),
            "schema_version": store_summary.schema_version,
            "table_counts": dict(store_summary.table_counts),
        },
        "selection": {
            "object_ids": list(configured_object_ids),
            "selected_object_ids": list(selected_object_ids),
            "selected_object_count": len(selected_object_ids),
        },
        "invalidation": {
            "open_items": len(invalidations),
            "by_reason": dict(sorted(invalidation_reasons.items())),
            "changed_object_ids": list(changed_object_ids),
            "maneuver_object_ids": list(maneuver_object_ids),
            "items": invalidations,
        },
        "propagation": {
            "expected_window_count": len(expected_windows),
            "recompute_chunk_count": len(propagation_tasks),
            "stale_existing_product_count": sum(len(item["stale_product_ids"]) for item in propagation_tasks),
            "tasks": propagation_tasks,
        },
        "screening": {
            "affected_pair_count": len(affected_pairs),
            "affected_pairs": [list(pair) for pair in affected_pairs],
            "affected_pair_window_count": len(affected_windows),
            "affected_windows": affected_windows,
            "pending_rescreen_product_pair_count": len(recompute_screening_pairs),
            "pending_rescreen_product_pairs": [
                [left.propagation_product_id, right.propagation_product_id]
                for left, right in recompute_screening_pairs
            ],
            "total_rescreen_work_count": rescreen_pair_count,
        },
        "candidates": {
            "affected_candidate_count": len(affected_candidates),
            "current_candidate_count": count_current_conjunction_candidates(
                cfg.store_path, object_ids=configured_object_ids
            ),
            "pending_refinement_count": pending_refinement_count,
            "affected_candidates": affected_candidates,
        },
        "hpop_handoff": {
            "affected_packet_count": len(affected_handoff),
            "pending_packet_count": pending_handoff_count,
            "stale_packet_count": stale_hpop_packet_count,
            "affected_packets": affected_handoff,
        },
        "planned_jobs": planned_jobs,
        "non_claims": [
            "This recompute planner does not execute propagation, screening, refinement, or ONP handoff.",
            "Affected pair windows are derived from prior screening evidence; this is not a full-catalog all-pairs rerun.",
            "Maneuver flags are advisory recompute triggers unless an explicit operations policy says otherwise.",
        ],
        "artifacts": {},
    }
    json_path = cfg.output_dir / "scale_recompute_plan.json"
    md_path = cfg.output_dir / "scale_recompute_plan.md"
    artifact_paths = {
        "planner_json": str(json_path),
        "planner_markdown": str(md_path),
    }
    payload["artifacts"] = artifact_paths
    payload["store"]["path"] = str(logical_store_path)
    serialized_json = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    serialized_markdown = _render_recompute_markdown(payload).encode("utf-8")
    if writer is None:
        cfg.output_dir.mkdir(parents=True, exist_ok=True)
        json_path.write_bytes(serialized_json)
        md_path.write_bytes(serialized_markdown)
    else:
        # MCP callers pass a descriptor-bound writer rooted at the approved
        # final output directory.  Keep the logical final paths in the plan
        # payload while the writer protects the actual producer writes.
        binding = bind_output_writer(writer, cfg.output_dir)
        assert binding is not None
        binding.write_bytes(json_path, serialized_json)
        binding.write_bytes(md_path, serialized_markdown)
    return payload


def run_observation_recompute_plan(
    store_path: str | Path,
    *,
    object_ids: tuple[str, ...] = (),
) -> dict[str, Any]:
    db_path = Path(store_path).expanduser().resolve()
    store_summary = summarize_store(db_path)
    invalidations = [
        item
        for item in _load_open_invalidations(db_path, object_ids=tuple(str(item) for item in object_ids))
        if item.get("reason") == "new_observation"
    ]
    ground_measurement_invalidations = [
        item for item in invalidations if item.get("stale_product_type") == "ground_station_measurement"
    ]
    measurement_ids = tuple(str(item["stale_product_id"]) for item in ground_measurement_invalidations)
    quality_by_measurement = _load_latest_ground_measurement_quality(db_path, measurement_ids=measurement_ids)
    objects: dict[str, dict[str, Any]] = {}
    measurement_items: list[dict[str, Any]] = []
    for item in ground_measurement_invalidations:
        object_id = str(item.get("object_id", "") or "")
        measurement_id = str(item.get("stale_product_id", "") or "")
        quality = dict(quality_by_measurement.get(measurement_id, {}) or {})
        quality_status = str(quality.get("status", "unassessed") or "unassessed")
        recommendation = _observation_quality_recommendation(quality_status)
        row = {
            "invalidation_id": item["invalidation_id"],
            "object_id": object_id,
            "measurement_id": measurement_id,
            "quality_status": quality_status,
            "quality_assessment_id": str(quality.get("assessment_id", "") or ""),
            "residual_kind": str(quality.get("residual_kind", "") or ""),
            "position_error_m": quality.get("position_error_m"),
            "normalized_position_error": quality.get("normalized_position_error"),
            "recommendation": recommendation,
        }
        measurement_items.append(row)
        bucket = objects.setdefault(
            object_id,
            {
                "object_id": object_id,
                "measurement_count": 0,
                "status_counts": {},
                "measurement_ids": [],
                "recommendation": "",
                "next_recommended_command": "",
            },
        )
        bucket["measurement_count"] += 1
        bucket["measurement_ids"].append(measurement_id)
        bucket["status_counts"][quality_status] = int(bucket["status_counts"].get(quality_status, 0) or 0) + 1
    object_rows = []
    for object_id, row in sorted(objects.items()):
        status_counts = dict(row["status_counts"])
        if status_counts.get("accepted", 0) or status_counts.get("warning", 0):
            recommendation = "quality_allows_recompute"
            next_command = "run or promote OD, then recompute affected propagation/screening products"
        elif status_counts.get("unassessed", 0):
            recommendation = "run_ground_measurement_od_then_assess_quality"
            next_command = (
                ".venv/bin/python -m sim.scale fit-ground-measurement-od --store <store.sqlite> "
                f"--station-id <station> --object-id {object_id} --output-dir <od_dir> --json"
            )
        elif status_counts.get("rejected", 0):
            recommendation = "skip_or_quarantine_rejected_measurements"
            next_command = "review rejected measurement quality rows; do not recompute from these rows by default"
        else:
            recommendation = "review_observation_state"
            next_command = "inspect measurement and quality rows"
        row["recommendation"] = recommendation
        row["next_recommended_command"] = next_command
        object_rows.append(row)
    summary_counts = {
        "objects_total": len(object_rows),
        "objects_quality_allows_recompute": sum(
            1 for row in object_rows if row["recommendation"] == "quality_allows_recompute"
        ),
        "objects_need_quality_assessment": sum(
            1 for row in object_rows if row["recommendation"] == "run_ground_measurement_od_then_assess_quality"
        ),
        "objects_only_rejected": sum(
            1 for row in object_rows if row["recommendation"] == "skip_or_quarantine_rejected_measurements"
        ),
        "ground_measurement_invalidations": len(ground_measurement_invalidations),
        "other_new_observation_invalidations": len(invalidations) - len(ground_measurement_invalidations),
    }
    planned_jobs = [
        {
            "job_type": "fit_ground_measurement_od_and_assess_quality",
            "count": summary_counts["objects_need_quality_assessment"],
            "status": "planned" if summary_counts["objects_need_quality_assessment"] else "not_needed",
            "reason": "open ground-station measurement invalidations have no advisory quality rows yet",
        },
        {
            "job_type": "quality_allowed_observation_recompute",
            "count": summary_counts["objects_quality_allows_recompute"],
            "status": "planned" if summary_counts["objects_quality_allows_recompute"] else "not_needed",
            "reason": "accepted or warning measurement quality can drive downstream OD/recompute policy",
        },
        {
            "job_type": "review_rejected_measurements",
            "count": summary_counts["objects_only_rejected"],
            "status": "planned" if summary_counts["objects_only_rejected"] else "not_needed",
            "reason": "all open measurements for these objects are rejected by the latest advisory quality assessment",
        },
    ]
    return {
        "schema_version": 1,
        "status": "completed",
        "store": {
            "path": str(db_path),
            "schema_version": store_summary.schema_version,
            "table_counts": dict(store_summary.table_counts),
        },
        "policy": {
            "accepted_statuses": ["accepted", "warning"],
            "rejected_statuses": ["rejected"],
            "unassessed_action": "run_ground_measurement_od_then_assess_quality",
            "mutation": "none",
        },
        "summary": summary_counts,
        "objects": object_rows,
        "measurements": measurement_items,
        "planned_jobs": planned_jobs,
        "non_claims": [
            "This planner is advisory and does not mutate invalidation status.",
            "Rejected measurement rows remain append-only evidence; they are not deleted.",
            "Quality-allowed recompute still requires an explicit OD/current-state promotion policy before propagation products are refreshed.",
        ],
    }


@enforce_scale_planner_limits("recompute propagation")
def run_recompute_propagation(
    config_path: str | Path,
    *,
    writer: OutputWriter | None = None,
) -> dict[str, Any]:
    cfg = load_scale_planner_config(config_path)
    if cfg.mode != "run":
        raise ValueError("Recompute propagation requires scale_catalog.mode: run.")
    if cfg.propagation_model != "ogp":
        raise ValueError("Recompute propagation requires scale_catalog.propagation.model: ogp.")
    if cfg.propagation_backend not in {"scalar", "rust"}:
        raise ValueError("Recompute propagation supports propagation.backend: scalar or rust.")
    plan = run_recompute_plan(config_path, writer=writer)
    tasks = list(dict(plan.get("propagation", {}) or {}).get("tasks", []) or [])
    if not tasks:
        payload = _recompute_propagation_payload(
            cfg,
            plan=plan,
            tasks=[],
            products_seen=0,
            products_inserted=0,
            samples_inserted=0,
            invalidations_completed=0,
            status="completed",
            execution_status="not_needed",
        )
        _write_recompute_propagation_artifacts(cfg, payload, writer=writer)
        return payload

    object_ids = tuple(sorted({str(task["object_id"]) for task in tasks}))
    with staged_sqlite_store(cfg.store_path, writer=writer) as store_path:
        states = load_current_tle_states(store_path, object_ids=object_ids)
        state_ids = {state.object_id for state in states}
        missing_state_ids = tuple(object_id for object_id in object_ids if object_id not in state_ids)
        if missing_state_ids:
            raise ValueError(f"Current TLE state is missing for recompute objects: {', '.join(missing_state_ids)}")
        windows = tuple(
            sorted(
                {
                    (float(task["start_jd_utc"]), float(task["end_jd_utc"]), float(task["step_s"]))
                    for task in tasks
                }
            )
        )
        products, telemetry = build_sgp4_products_for_windows(cfg, states=states, windows=windows)
        write_summary = write_propagation_products(store_path, products=products, complete_invalidations=False)
        all_products_completed = len(products) == len(tasks) and all(product.status == "completed" for product in products)
        invalidations_completed = 0
        execution_status = "completed"
        status = "completed"
        if all_products_completed:
            invalidations_completed = complete_propagation_invalidations(store_path, object_ids=object_ids)
        else:
            execution_status = "partial"
            status = "warning"
    payload = _recompute_propagation_payload(
        cfg,
        plan=plan,
        tasks=tasks,
        products_seen=write_summary["products_seen"],
        products_inserted=write_summary["products_inserted"],
        samples_inserted=write_summary["samples_inserted"],
        invalidations_completed=invalidations_completed,
        status=status,
        execution_status=execution_status,
    )
    payload["propagation"]["product_ids"] = [product.propagation_product_id for product in products]
    payload["propagation"]["error_product_count"] = sum(1 for product in products if product.status != "completed")
    payload["propagation"]["ogp"] = telemetry
    _write_recompute_propagation_artifacts(cfg, payload, writer=writer)
    return payload


@enforce_scale_planner_limits("recompute screening")
def run_recompute_screening(
    config_path: str | Path,
    *,
    writer: OutputWriter | None = None,
) -> dict[str, Any]:
    cfg = load_scale_planner_config(config_path)
    if cfg.mode != "run":
        raise ValueError("Recompute screening requires scale_catalog.mode: run.")
    if cfg.screening_mode != "sampled_distance":
        raise ValueError("Recompute screening currently supports only screening.mode: sampled_distance.")
    logical_store_path = cfg.store_path
    read_cfg = replace(cfg, store_path=store_read_path(cfg.store_path, writer=writer))
    product_pairs = _load_recompute_screening_product_pairs(read_cfg)
    windows, candidate_pool = build_pair_screening_results_for_product_pairs(read_cfg, product_pairs=product_pairs)
    candidate_pool.sort(key=lambda item: (item.miss_distance_km, item.relative_speed_km_s, item.candidate_id))
    candidates = candidate_pool[: cfg.promotion_limit] if cfg.promotion_limit else []
    with staged_sqlite_store(logical_store_path, writer=writer) as store_path:
        write_summary = write_screening_results(
            store_path,
            windows=windows,
            candidates=candidates,
            supersede_prior_candidates=True,
        )
    payload = {
        "schema_version": 1,
        "status": "completed",
        "workflow": {
            "name": cfg.name,
            "config_path": str(cfg.config_path),
            "mode": cfg.mode,
        },
        "store_path": str(cfg.store_path),
        "screening_mode": cfg.screening_mode,
        "targeting": {
            "method": "superseded_propagation_products",
            "product_pair_count": len(product_pairs),
            "product_pairs": [
                [left.propagation_product_id, right.propagation_product_id] for left, right in product_pairs
            ],
        },
        "pair_windows_seen": write_summary["windows_seen"],
        "pair_windows_inserted": write_summary["windows_inserted"],
        "candidate_promotions_seen": write_summary["candidates_seen"],
        "candidate_promotions_inserted": write_summary["candidates_inserted"],
        "candidates_superseded": write_summary["candidates_superseded"],
        "hpop_packets_marked_stale": write_summary["hpop_packets_marked_stale"],
        "current_candidate_count": count_current_conjunction_candidates(
            store_path, object_ids=resolve_selected_object_ids(cfg)
        ),
        "stale_hpop_packet_count": count_stale_hpop_handoff_packets(
            store_path, object_ids=resolve_selected_object_ids(cfg)
        ),
        "distance_threshold_km": cfg.distance_threshold_km,
        "promotion_limit": cfg.promotion_limit,
        "non_claims": [
            "This command rescreens only pair/windows whose prior evidence used a superseded propagation product.",
            "It does not perform full-catalog all-pairs screening.",
            "It does not refine candidates or regenerate ONP handoff packets.",
        ],
        "artifacts": {},
    }
    if cfg.screening_numeric_backend == "rust":
        payload["screening_numeric_backend"] = "rust"
    payload["store_path"] = str(logical_store_path)
    _write_recompute_screening_artifacts(cfg, payload, writer=writer)
    return payload


def _recompute_propagation_payload(
    cfg: Any,
    *,
    plan: dict[str, Any],
    tasks: list[dict[str, Any]],
    products_seen: int,
    products_inserted: int,
    samples_inserted: int,
    invalidations_completed: int,
    status: str,
    execution_status: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": status,
        "execution_status": execution_status,
        "workflow": dict(plan.get("workflow", {}) or {}),
        "store_path": str(cfg.store_path),
        "plan_artifacts": dict(plan.get("artifacts", {}) or {}),
        "invalidation": {
            "open_items_before": dict(plan.get("invalidation", {}) or {}).get("open_items", 0),
            "changed_object_ids": list(dict(plan.get("invalidation", {}) or {}).get("changed_object_ids", []) or []),
            "invalidations_completed": int(invalidations_completed),
        },
        "propagation": {
            "planned_task_count": len(tasks),
            "products_seen": int(products_seen),
            "products_inserted": int(products_inserted),
            "samples_inserted": int(samples_inserted),
            "tasks": tasks,
        },
        "downstream": {
            "affected_pair_window_count": dict(plan.get("screening", {}) or {}).get("total_rescreen_work_count", 0),
            "pending_rescreen_product_pair_count": dict(plan.get("screening", {}) or {}).get(
                "pending_rescreen_product_pair_count", 0
            ),
            "affected_candidate_count": dict(plan.get("candidates", {}) or {}).get("affected_candidate_count", 0),
            "next_recommended_command": ".venv/bin/python -m sim.scale plan-recompute <config> --json",
            "note": "Run the recompute planner again to confirm object invalidations are closed and downstream screening work remains.",
        },
        "non_claims": [
            "This command recomputes only planned OGP propagation chunks.",
            "It does not rerun pair screening, candidate ranking, refinement, or ONP handoff.",
            "Object invalidations are completed when all planned replacement propagation products are completed or already exist.",
        ],
        "artifacts": {},
    }


def _write_recompute_propagation_artifacts(
    cfg: Any,
    payload: dict[str, Any],
    *,
    writer: OutputWriter | None = None,
) -> None:
    json_path = cfg.output_dir / "scale_recompute_propagation.json"
    md_path = cfg.output_dir / "scale_recompute_propagation.md"
    payload["artifacts"] = {
        "execution_json": str(json_path),
        "execution_markdown": str(md_path),
    }
    serialized_json = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    serialized_markdown = _render_recompute_propagation_markdown(payload).encode("utf-8")
    if writer is None:
        cfg.output_dir.mkdir(parents=True, exist_ok=True)
        json_path.write_bytes(serialized_json)
        md_path.write_bytes(serialized_markdown)
        return
    binding = bind_output_writer(writer, cfg.output_dir)
    assert binding is not None
    binding.write_bytes(json_path, serialized_json)
    binding.write_bytes(md_path, serialized_markdown)


def _write_recompute_screening_artifacts(
    cfg: Any,
    payload: dict[str, Any],
    *,
    writer: OutputWriter | None = None,
) -> None:
    json_path = cfg.output_dir / "scale_recompute_screening.json"
    md_path = cfg.output_dir / "scale_recompute_screening.md"
    payload["artifacts"] = {
        "execution_json": str(json_path),
        "execution_markdown": str(md_path),
    }
    serialized_json = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    serialized_markdown = _render_recompute_screening_markdown(payload).encode("utf-8")
    if writer is None:
        cfg.output_dir.mkdir(parents=True, exist_ok=True)
        json_path.write_bytes(serialized_json)
        md_path.write_bytes(serialized_markdown)
        return
    binding = bind_output_writer(writer, cfg.output_dir)
    assert binding is not None
    binding.write_bytes(json_path, serialized_json)
    binding.write_bytes(md_path, serialized_markdown)


def _load_open_invalidations(path: Path, *, object_ids: tuple[str, ...]) -> list[dict[str, Any]]:
    params: list[Any] = list(OPEN_INVALIDATION_STATUSES)
    where = f"WHERE status IN ({', '.join('?' for _ in OPEN_INVALIDATION_STATUSES)})"
    if object_ids:
        where += f" AND object_id IN ({', '.join('?' for _ in object_ids)})"
        params.extend(object_ids)
    with sqlite3.connect(path) as conn:
        rows = conn.execute(
            f"""
            SELECT
                invalidation_id, object_id, deputy_id, chief_id,
                time_start_jd_utc, time_end_jd_utc, stale_product_id,
                stale_product_type, reason, priority, status, created_utc
            FROM invalidation_queue
            {where}
            ORDER BY priority DESC, created_utc ASC, invalidation_id ASC
            """,
            params,
        ).fetchall()
    return [
        {
            "invalidation_id": str(row[0]),
            "object_id": str(row[1] or ""),
            "deputy_id": str(row[2] or ""),
            "chief_id": str(row[3] or ""),
            "time_start_jd_utc": _float_or_none(row[4]),
            "time_end_jd_utc": _float_or_none(row[5]),
            "stale_product_id": str(row[6] or ""),
            "stale_product_type": str(row[7] or ""),
            "reason": str(row[8] or ""),
            "priority": float(row[9] or 0.0),
            "status": str(row[10] or ""),
            "created_utc": str(row[11] or ""),
        }
        for row in rows
    ]


def _load_latest_ground_measurement_quality(
    path: Path,
    *,
    measurement_ids: tuple[str, ...],
) -> dict[str, dict[str, Any]]:
    if not measurement_ids:
        return {}
    placeholders = ", ".join("?" for _ in measurement_ids)
    with sqlite3.connect(path) as conn:
        rows = conn.execute(
            f"""
            SELECT
                rowid, quality_id, assessment_id, measurement_id, residual_kind,
                position_error_m, normalized_position_error, status, flags_json,
                metrics_json, thresholds_json, created_utc
            FROM ground_station_measurement_quality
            WHERE measurement_id IN ({placeholders})
            ORDER BY measurement_id, created_utc ASC, rowid ASC
            """,
            list(measurement_ids),
        ).fetchall()
    latest: dict[str, dict[str, Any]] = {}
    for row in rows:
        latest[str(row[3])] = {
            "quality_id": str(row[1]),
            "assessment_id": str(row[2]),
            "measurement_id": str(row[3]),
            "residual_kind": str(row[4] or ""),
            "position_error_m": _float_or_none(row[5]),
            "normalized_position_error": _float_or_none(row[6]),
            "status": str(row[7] or ""),
            "flags": json.loads(str(row[8] or "[]")),
            "metrics": json.loads(str(row[9] or "{}")),
            "thresholds": json.loads(str(row[10] or "{}")),
            "created_utc": str(row[11] or ""),
        }
    return latest


def _observation_quality_recommendation(quality_status: str) -> str:
    if quality_status in {"accepted", "warning"}:
        return "quality_allows_recompute"
    if quality_status == "rejected":
        return "skip_or_quarantine"
    return "needs_quality_assessment"


def _build_propagation_tasks(
    path: Path,
    *,
    object_ids: tuple[str, ...],
    windows: tuple[tuple[float, float, float], ...],
    model: str,
    backend: str,
    output_frame: str,
) -> list[dict[str, Any]]:
    if not object_ids or not windows:
        return []
    products = _load_propagation_products_for_objects(
        path,
        object_ids=object_ids,
        model=model,
        backend=backend,
        output_frame=output_frame,
    )
    by_key: dict[tuple[str, float, float, float], list[dict[str, Any]]] = {}
    for product in products:
        key = (
            str(product["object_id"]),
            float(product["start_jd_utc"]),
            float(product["end_jd_utc"]),
            float(product["step_s"]),
        )
        by_key.setdefault(key, []).append(product)
    tasks: list[dict[str, Any]] = []
    for object_id in object_ids:
        for start_jd, end_jd, step_s in windows:
            stale_products = by_key.get((object_id, start_jd, end_jd, step_s), [])
            tasks.append(
                {
                    "object_id": object_id,
                    "start_jd_utc": float(start_jd),
                    "end_jd_utc": float(end_jd),
                    "step_s": float(step_s),
                    "stale_product_ids": [str(item["propagation_product_id"]) for item in stale_products],
                    "reason": "object_state_changed",
                }
            )
    return tasks


def _load_propagation_products_for_objects(
    path: Path,
    *,
    object_ids: tuple[str, ...],
    model: str,
    backend: str,
    output_frame: str,
) -> list[dict[str, Any]]:
    params: list[Any] = [model, backend, output_frame, *object_ids]
    with sqlite3.connect(path) as conn:
        rows = conn.execute(
            f"""
            SELECT
                propagation_product_id, object_id, source_od_solution_id,
                start_jd_utc, end_jd_utc, step_s, status, input_digest
            FROM propagation_products
            WHERE model = ?
              AND backend = ?
              AND output_frame = ?
              AND object_id IN ({', '.join('?' for _ in object_ids)})
            ORDER BY object_id, start_jd_utc, end_jd_utc, propagation_product_id
            """,
            params,
        ).fetchall()
    return [
        {
            "propagation_product_id": str(row[0]),
            "object_id": str(row[1]),
            "source_od_solution_id": str(row[2] or ""),
            "start_jd_utc": float(row[3]),
            "end_jd_utc": float(row[4]),
            "step_s": float(row[5]),
            "status": str(row[6] or ""),
            "input_digest": str(row[7] or ""),
        }
        for row in rows
    ]


def _load_recompute_screening_product_pairs(cfg: Any) -> list[tuple[PropagationProductSummary, PropagationProductSummary]]:
    replacements = _load_current_replacement_products(cfg)
    if not replacements:
        return []
    current_by_object_window = _load_current_products_by_object_window(cfg)
    pairs: dict[tuple[str, str], tuple[PropagationProductSummary, PropagationProductSummary]] = {}
    for replacement in replacements:
        superseded_product_id = str(replacement["supersedes_product_id"])
        old_windows = _load_screening_windows_for_source_product(
            cfg.store_path,
            source_product_id=superseded_product_id,
            start_jd_utc=float(replacement["start_jd_utc"]),
            end_jd_utc=float(replacement["end_jd_utc"]),
        )
        left = _product_summary_from_row(replacement)
        for window in old_windows:
            other_object_id = (
                str(window["chief_id"])
                if str(window["deputy_id"]) == replacement["object_id"]
                else str(window["deputy_id"])
            )
            other = current_by_object_window.get(
                (
                    other_object_id,
                    float(replacement["start_jd_utc"]),
                    float(replacement["end_jd_utc"]),
                    float(replacement["step_s"]),
                )
            )
            if other is None:
                continue
            key = tuple(sorted((left.propagation_product_id, other.propagation_product_id)))
            if _screening_window_exists_for_product_pair(
                cfg.store_path,
                product_ids=key,
                start_jd_utc=float(replacement["start_jd_utc"]),
                end_jd_utc=float(replacement["end_jd_utc"]),
                method=cfg.screening_mode,
            ):
                continue
            pairs[key] = (left, other)
    return list(pairs.values())


def _load_current_replacement_products(cfg: Any) -> list[dict[str, Any]]:
    object_ids = resolve_selected_object_ids(cfg)
    if not object_ids:
        return []
    expected_windows = {(float(start), float(end), float(step)) for start, end, step in _expected_propagation_windows(cfg)}
    params: list[Any] = [cfg.propagation_model, cfg.propagation_backend, cfg.output_frame, *object_ids]
    with sqlite3.connect(cfg.store_path) as conn:
        rows = conn.execute(
            f"""
            SELECT
                pp.propagation_product_id, pp.object_id, pp.model, pp.backend,
                pp.output_frame, pp.start_jd_utc, pp.end_jd_utc, pp.step_s,
                pp.sample_count, pp.input_digest, pp.supersedes_product_id,
                pp.ogp_regime, pp.runtime_backend, pp.metadata_json
            FROM propagation_products pp
            JOIN current_object_state cos ON cos.object_id = pp.object_id
            WHERE pp.status = 'completed'
              AND pp.source_od_solution_id = cos.od_solution_id
              AND pp.supersedes_product_id IS NOT NULL
              AND pp.model = ?
              AND pp.backend = ?
              AND pp.output_frame = ?
              AND pp.object_id IN ({', '.join('?' for _ in object_ids)})
            ORDER BY pp.object_id, pp.start_jd_utc, pp.end_jd_utc, pp.propagation_product_id
            """,
            params,
        ).fetchall()
    products = [_product_row(row) for row in rows]
    return [
        product
        for product in products
        if (product["start_jd_utc"], product["end_jd_utc"], product["step_s"]) in expected_windows
    ]


def _load_current_products_by_object_window(
    cfg: Any,
) -> dict[tuple[str, float, float, float], PropagationProductSummary]:
    object_ids = resolve_selected_object_ids(cfg)
    if not object_ids:
        return {}
    params: list[Any] = [cfg.propagation_model, cfg.propagation_backend, cfg.output_frame, *object_ids]
    with sqlite3.connect(cfg.store_path) as conn:
        rows = conn.execute(
            f"""
            SELECT
                pp.propagation_product_id, pp.object_id, pp.model, pp.backend,
                pp.output_frame, pp.start_jd_utc, pp.end_jd_utc, pp.step_s,
                pp.sample_count, pp.input_digest, pp.supersedes_product_id,
                pp.ogp_regime, pp.runtime_backend, pp.metadata_json
            FROM propagation_products pp
            JOIN current_object_state cos ON cos.object_id = pp.object_id
            WHERE pp.status = 'completed'
              AND pp.source_od_solution_id = cos.od_solution_id
              AND pp.model = ?
              AND pp.backend = ?
              AND pp.output_frame = ?
              AND pp.object_id IN ({', '.join('?' for _ in object_ids)})
            ORDER BY pp.object_id, pp.start_jd_utc, pp.end_jd_utc, pp.created_utc DESC
            """,
            params,
        ).fetchall()
    out: dict[tuple[str, float, float, float], PropagationProductSummary] = {}
    for row in rows:
        product = _product_summary_from_row(_product_row(row))
        key = (product.object_id, product.start_jd_utc, product.end_jd_utc, product.step_s)
        out.setdefault(key, product)
    return out


def _load_screening_windows_for_source_product(
    path: Path,
    *,
    source_product_id: str,
    start_jd_utc: float,
    end_jd_utc: float,
) -> list[dict[str, Any]]:
    with sqlite3.connect(path) as conn:
        rows = conn.execute(
            """
            SELECT screening_window_id, deputy_id, chief_id, window_start_jd_utc,
                   window_end_jd_utc, source_product_ids_json
            FROM screening_windows
            WHERE window_start_jd_utc = ?
              AND window_end_jd_utc = ?
            ORDER BY screening_window_id
            """,
            (start_jd_utc, end_jd_utc),
        ).fetchall()
    out: list[dict[str, Any]] = []
    for row in rows:
        source_product_ids = [str(item) for item in _loads(row[5], default=[])]
        if source_product_id not in source_product_ids:
            continue
        out.append(
            {
                "screening_window_id": str(row[0]),
                "deputy_id": str(row[1]),
                "chief_id": str(row[2]),
                "window_start_jd_utc": float(row[3]),
                "window_end_jd_utc": float(row[4]),
                "source_product_ids": source_product_ids,
            }
        )
    return out


def _screening_window_exists_for_product_pair(
    path: Path,
    *,
    product_ids: tuple[str, str],
    start_jd_utc: float,
    end_jd_utc: float,
    method: str,
) -> bool:
    expected = tuple(sorted(str(item) for item in product_ids))
    with sqlite3.connect(path) as conn:
        rows = conn.execute(
            """
            SELECT source_product_ids_json
            FROM screening_windows
            WHERE window_start_jd_utc = ?
              AND window_end_jd_utc = ?
              AND method = ?
            """,
            (start_jd_utc, end_jd_utc, method),
        ).fetchall()
    for row in rows:
        observed = tuple(sorted(str(item) for item in _loads(row[0], default=[])))
        if observed == expected:
            return True
    return False


def _product_row(row: Any) -> dict[str, Any]:
    return {
        "propagation_product_id": str(row[0]),
        "object_id": str(row[1]),
        "model": str(row[2] or ""),
        "backend": str(row[3] or ""),
        "output_frame": str(row[4] or ""),
        "start_jd_utc": float(row[5]),
        "end_jd_utc": float(row[6]),
        "step_s": float(row[7]),
        "sample_count": int(row[8] or 0),
        "input_digest": str(row[9] or ""),
        "supersedes_product_id": str(row[10] or ""),
        "ogp_regime": str(row[11] or "") if len(row) > 11 else "",
        "runtime_backend": str(row[12] or "") if len(row) > 12 else "",
        "metadata": dict(json.loads(row[13] or "{}")) if len(row) > 13 else {},
    }


def _product_summary_from_row(row: dict[str, Any]) -> PropagationProductSummary:
    return PropagationProductSummary(
        propagation_product_id=str(row["propagation_product_id"]),
        object_id=str(row["object_id"]),
        model=str(row["model"]),
        backend=str(row["backend"]),
        output_frame=str(row["output_frame"]),
        start_jd_utc=float(row["start_jd_utc"]),
        end_jd_utc=float(row["end_jd_utc"]),
        step_s=float(row["step_s"]),
        sample_count=int(row["sample_count"]),
        input_digest=str(row["input_digest"]),
        ogp_regime=str(row.get("ogp_regime", "")),
        runtime_backend=str(row.get("runtime_backend", "")),
        metadata=dict(row.get("metadata", {}) or {}),
    )


def _load_affected_screening_windows(
    path: Path,
    *,
    object_ids: tuple[str, ...],
    windows: tuple[tuple[float, float, float], ...],
) -> list[dict[str, Any]]:
    if not object_ids:
        return []
    expected_windows = {(float(start), float(end)) for start, end, _ in windows}
    params: list[Any] = [*object_ids, *object_ids]
    with sqlite3.connect(path) as conn:
        rows = conn.execute(
            f"""
            SELECT
                screening_window_id, screening_run_id, deputy_id, chief_id,
                window_start_jd_utc, window_end_jd_utc, source_product_ids_json,
                min_sample_distance_km, min_sample_time_jd_utc,
                relative_speed_km_s, pruning_reason, status
            FROM screening_windows
            WHERE deputy_id IN ({', '.join('?' for _ in object_ids)})
               OR chief_id IN ({', '.join('?' for _ in object_ids)})
            ORDER BY window_start_jd_utc, window_end_jd_utc, deputy_id, chief_id
            """,
            params,
        ).fetchall()
    out: list[dict[str, Any]] = []
    for row in rows:
        start = float(row[4])
        end = float(row[5])
        if expected_windows and (start, end) not in expected_windows:
            continue
        out.append(
            {
                "screening_window_id": str(row[0]),
                "screening_run_id": str(row[1] or ""),
                "deputy_id": str(row[2]),
                "chief_id": str(row[3]),
                "window_start_jd_utc": start,
                "window_end_jd_utc": end,
                "source_product_ids": list(_loads(row[6], default=[])),
                "min_sample_distance_km": float(row[7]),
                "min_sample_time_jd_utc": float(row[8]),
                "relative_speed_km_s": float(row[9]),
                "pruning_reason": str(row[10] or ""),
                "status": str(row[11] or ""),
            }
        )
    return out


def _load_affected_candidates(
    path: Path,
    *,
    screening_window_ids: tuple[str, ...],
    object_ids: tuple[str, ...],
) -> list[dict[str, Any]]:
    clauses: list[str] = []
    params: list[Any] = []
    if screening_window_ids:
        clauses.append(f"screening_window_id IN ({', '.join('?' for _ in screening_window_ids)})")
        params.extend(screening_window_ids)
    if object_ids:
        clauses.append(
            f"(deputy_id IN ({', '.join('?' for _ in object_ids)}) OR chief_id IN ({', '.join('?' for _ in object_ids)}))"
        )
        params.extend(object_ids)
        params.extend(object_ids)
    if not clauses:
        return []
    with sqlite3.connect(path) as conn:
        rows = conn.execute(
            f"""
            SELECT
                candidate_id, screening_window_id, deputy_id, chief_id,
                tca_jd_utc, miss_distance_km, relative_speed_km_s,
                ranking_score, promotion_status
            FROM conjunction_candidates
            WHERE {' OR '.join(clauses)}
            ORDER BY miss_distance_km ASC, relative_speed_km_s ASC, candidate_id ASC
            """,
            params,
        ).fetchall()
    return [
        {
            "candidate_id": str(row[0]),
            "screening_window_id": str(row[1]),
            "deputy_id": str(row[2]),
            "chief_id": str(row[3]),
            "tca_jd_utc": float(row[4]),
            "miss_distance_km": float(row[5]),
            "relative_speed_km_s": float(row[6]),
            "ranking_score": float(row[7]),
            "promotion_status": str(row[8] or ""),
        }
        for row in rows
    ]


def _load_affected_hpop_packets(path: Path, *, candidate_ids: tuple[str, ...]) -> list[dict[str, Any]]:
    if not candidate_ids:
        return []
    with sqlite3.connect(path) as conn:
        rows = conn.execute(
            f"""
            SELECT packet_id, candidate_id, refinement_id, status, packet_path
            FROM hpop_handoff_packets
            WHERE candidate_id IN ({', '.join('?' for _ in candidate_ids)})
            ORDER BY candidate_id, packet_id
            """,
            list(candidate_ids),
        ).fetchall()
    return [
        {
            "packet_id": str(row[0]),
            "candidate_id": str(row[1]),
            "refinement_id": str(row[2] or ""),
            "status": str(row[3] or ""),
            "packet_path": str(row[4] or ""),
        }
        for row in rows
    ]


def _render_recompute_markdown(payload: dict[str, Any]) -> str:
    workflow = dict(payload.get("workflow", {}) or {})
    invalidation = dict(payload.get("invalidation", {}) or {})
    propagation = dict(payload.get("propagation", {}) or {})
    screening = dict(payload.get("screening", {}) or {})
    candidates = dict(payload.get("candidates", {}) or {})
    hpop_handoff = dict(payload.get("hpop_handoff", {}) or {})
    lines = [
        f"# OEL Scale Recompute Plan: {workflow.get('name', 'scale_workflow')}",
        "",
        f"- Status: `{payload.get('status')}`",
        f"- Config: `{workflow.get('config_path')}`",
        f"- Store: `{dict(payload.get('store', {}) or {}).get('path')}`",
        f"- Open invalidations: `{invalidation.get('open_items')}`",
        f"- Changed objects: `{len(invalidation.get('changed_object_ids', []) or [])}`",
        f"- Propagation chunks to recompute: `{propagation.get('recompute_chunk_count')}`",
        f"- Affected pair windows: `{screening.get('affected_pair_window_count')}`",
        f"- Pending superseded-product rescreen pairs: `{screening.get('pending_rescreen_product_pair_count')}`",
        f"- Total rescreen work: `{screening.get('total_rescreen_work_count')}`",
        f"- Affected candidates: `{candidates.get('affected_candidate_count')}`",
        f"- Affected ONP packets: `{hpop_handoff.get('affected_packet_count')}`",
        f"- Current candidates: `{candidates.get('current_candidate_count')}`",
        f"- Pending current-candidate refinements: `{candidates.get('pending_refinement_count')}`",
        f"- Pending current ONP packets: `{hpop_handoff.get('pending_packet_count')}`",
        f"- Stale ONP packets: `{hpop_handoff.get('stale_packet_count')}`",
        "",
        "## Planned Jobs",
        "",
    ]
    for job in payload.get("planned_jobs", []) or []:
        lines.append(f"- `{job.get('job_type')}`: `{job.get('count')}` {job.get('status')}")
    reasons = dict(invalidation.get("by_reason", {}) or {})
    if reasons:
        lines.extend(["", "## Invalidation Reasons", ""])
        for reason, count in reasons.items():
            lines.append(f"- `{reason}`: `{count}`")
    lines.extend(["", "## Non-Claims", ""])
    lines.extend(f"- {item}" for item in payload.get("non_claims", []) or [])
    return "\n".join(lines) + "\n"


def _render_recompute_propagation_markdown(payload: dict[str, Any]) -> str:
    workflow = dict(payload.get("workflow", {}) or {})
    invalidation = dict(payload.get("invalidation", {}) or {})
    propagation = dict(payload.get("propagation", {}) or {})
    downstream = dict(payload.get("downstream", {}) or {})
    lines = [
        f"# OEL Scale Recompute Propagation: {workflow.get('name', 'scale_workflow')}",
        "",
        f"- Status: `{payload.get('status')}`",
        f"- Execution status: `{payload.get('execution_status')}`",
        f"- Config: `{workflow.get('config_path')}`",
        f"- Store: `{payload.get('store_path')}`",
        f"- Open invalidations before run: `{invalidation.get('open_items_before')}`",
        f"- Changed objects: `{len(invalidation.get('changed_object_ids', []) or [])}`",
        f"- Planned propagation tasks: `{propagation.get('planned_task_count')}`",
        f"- Products inserted: `{propagation.get('products_inserted')}`",
        f"- Samples inserted: `{propagation.get('samples_inserted')}`",
        f"- Invalidations completed: `{invalidation.get('invalidations_completed')}`",
        f"- Remaining rescreen work from source plan: `{downstream.get('affected_pair_window_count')}`",
        f"- Pending superseded-product rescreen pairs from source plan: `{downstream.get('pending_rescreen_product_pair_count')}`",
        f"- Remaining affected candidates from source plan: `{downstream.get('affected_candidate_count')}`",
        "",
        "## Non-Claims",
        "",
    ]
    lines.extend(f"- {item}" for item in payload.get("non_claims", []) or [])
    return "\n".join(lines) + "\n"


def _render_recompute_screening_markdown(payload: dict[str, Any]) -> str:
    workflow = dict(payload.get("workflow", {}) or {})
    targeting = dict(payload.get("targeting", {}) or {})
    lines = [
        f"# OEL Scale Recompute Screening: {workflow.get('name', 'scale_workflow')}",
        "",
        f"- Status: `{payload.get('status')}`",
        f"- Config: `{workflow.get('config_path')}`",
        f"- Store: `{payload.get('store_path')}`",
        f"- Targeting method: `{targeting.get('method')}`",
        f"- Product pairs: `{targeting.get('product_pair_count')}`",
        f"- Pair windows inserted: `{payload.get('pair_windows_inserted')}`",
        f"- Candidate promotions inserted: `{payload.get('candidate_promotions_inserted')}`",
        f"- Candidates superseded: `{payload.get('candidates_superseded')}`",
        f"- ONP packets marked stale: `{payload.get('hpop_packets_marked_stale')}`",
        f"- Current candidates: `{payload.get('current_candidate_count')}`",
        f"- Stale ONP packets: `{payload.get('stale_hpop_packet_count')}`",
        "",
        "## Non-Claims",
        "",
    ]
    lines.extend(f"- {item}" for item in payload.get("non_claims", []) or [])
    return "\n".join(lines) + "\n"


def _loads(value: Any, *, default: Any) -> Any:
    if value in (None, ""):
        return default
    try:
        return json.loads(str(value))
    except json.JSONDecodeError:
        return default


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)
