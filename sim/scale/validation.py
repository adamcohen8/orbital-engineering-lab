from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
import subprocess
import time
from collections import defaultdict
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from sim.data_generation.workflow import generate_synthetic_data, plan_synthetic_data, score_synthetic_data
from sim.evidence import artifact_reference, build_workflow_evidence, write_workflow_evidence
from sim.platform_compat import max_rss_mb
from sim.resource_limits import current_resource_snapshot
from sim.scale.planner import (
    _expected_propagation_windows,
    load_scale_planner_config,
    resolve_selected_object_ids,
)
from sim.scale.propagation import run_ogp_propagation
from sim.scale.screening import run_pair_screening
from sim.scale.store import (
    PropagationProductSummary,
    load_matching_current_propagation_products,
    load_product_samples,
    summarize_store,
)
from sim.security import ConfigPathPolicy


def run_scale_validation(config_path: str | Path) -> dict[str, Any]:
    source = Path(config_path).expanduser().resolve()
    raw = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    validation = dict(raw.get("scale_validation", {}) or {})
    if not validation:
        raise ValueError("Scale validation config requires a scale_validation mapping.")
    tier = str(validation.get("tier") or raw.get("name") or source.stem).strip()
    policy = ConfigPathPolicy.default(config_path=source)
    output_root = policy.resolve_output_dir(
        validation.get("output_root", "outputs/scale_validation"),
        purpose="scale_validation.output_root",
    )
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_root = output_root / tier / run_id
    runtime_config = _runtime_config(raw, source=source, run_root=run_root, policy=policy)
    runtime_config_path = run_root / "scale_validation_runtime.yaml"
    run_root.mkdir(parents=True, exist_ok=True)
    runtime_config_path.write_text(yaml.safe_dump(runtime_config, sort_keys=False), encoding="utf-8")

    phase_times: dict[str, float] = {}
    started = time.perf_counter()
    max_rss_before_mb = _max_rss_mb()

    plan, phase_times["generation_plan_s"] = _timed(plan_synthetic_data, runtime_config_path)
    generation, phase_times["generation_s"] = _timed(generate_synthetic_data, runtime_config_path)
    generation_score, phase_times["generation_score_s"] = _timed(score_synthetic_data, runtime_config_path)
    propagation, phase_times["propagation_s"] = _timed(run_ogp_propagation, runtime_config_path)
    screening, phase_times["spatial_screening_s"] = _timed(run_pair_screening, runtime_config_path)

    cfg = load_scale_planner_config(runtime_config_path)
    reference, phase_times["brute_force_reference_s"] = _timed(_brute_force_reference, cfg)
    actual = _actual_screening_evidence(cfg.store_path)
    expected = json.loads(Path(generation["artifacts"]["expected_results"]).read_text(encoding="utf-8"))
    parity = _parity_payload(reference, actual, promotion_limit=cfg.promotion_limit)
    truth_scoring = _truth_scoring(expected, actual)
    total_wall_s = time.perf_counter() - started
    store_summary = summarize_store(cfg.store_path)
    resource_snapshot = asdict(current_resource_snapshot())
    max_rss_after_mb = _max_rss_mb()
    quality = _quality_payload(
        validation,
        parity=parity,
        truth_scoring=truth_scoring,
        total_wall_s=total_wall_s,
    )
    evidence_json = run_root / "scale_validation_evidence_packet.json"
    evidence_md = run_root / "scale_validation_evidence_packet.md"
    payload = {
        "schema_version": 1,
        "status": "passed" if quality["passed"] else "failed",
        "tier": tier,
        "run_id": run_id,
        "source_config_path": str(source),
        "source_config_digest": hashlib.sha256(source.read_bytes()).hexdigest(),
        "runtime_config_path": str(runtime_config_path),
        "runtime_config_digest": hashlib.sha256(runtime_config_path.read_bytes()).hexdigest(),
        "repository": _repository_state(source.parent),
        "workload": {
            "object_count": generation["object_count"],
            "propagation_product_count": propagation["products_inserted"],
            "propagation_sample_count": propagation["samples_inserted"],
            "possible_pair_windows": screening["candidate_discovery"]["possible_pair_windows"],
            "spatial_candidate_pair_windows": screening["candidate_discovery"]["candidate_pair_windows"],
            "pruned_pair_windows": screening["candidate_discovery"]["pruned_pair_windows"],
            "distance_threshold_km": screening["distance_threshold_km"],
        },
        "parity": parity,
        "truth_scoring": truth_scoring,
        "quality": quality,
        "telemetry": {
            "wall_time_s": total_wall_s,
            "phase_durations_s": phase_times,
            "objects_per_second": generation["object_count"] / total_wall_s if total_wall_s else None,
            "state_samples_per_second": propagation["samples_inserted"] / total_wall_s if total_wall_s else None,
            "max_rss_before_mb": max_rss_before_mb,
            "max_rss_after_mb": max_rss_after_mb,
            "max_rss_growth_mb": (
                max(max_rss_after_mb - max_rss_before_mb, 0.0)
                if max_rss_before_mb is not None and max_rss_after_mb is not None
                else None
            ),
            "store_size_mb": cfg.store_path.stat().st_size / (1024.0 * 1024.0),
            "resource_snapshot": resource_snapshot,
        },
        "store": {
            "path": str(cfg.store_path),
            "schema_version": store_summary.schema_version,
            "table_counts": store_summary.table_counts,
        },
        "artifacts": {
            "evidence_json": str(evidence_json),
            "evidence_markdown": str(evidence_md),
            "generation_plan": plan["outputs"]["plan"],
            "truth_manifest": generation["artifacts"]["truth_manifest"],
            "expected_results": generation["artifacts"]["expected_results"],
            "generation_score": generation["artifacts"]["score_report"],
        },
        "non_claims": [
            "Parity applies to sampled-distance screening at configured sample epochs only.",
            "Synthetic fixtures do not establish operational conjunction assessment or collision probability.",
            "Runtime and memory evidence applies only to the recorded machine, config, and repository state.",
        ],
        "generation_status": generation_score["status"],
    }
    evidence_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    evidence_md.write_text(_render_markdown(payload), encoding="utf-8")
    workflow_evidence_path = write_workflow_evidence(
        run_root / "oel_workflow_evidence.json",
        build_workflow_evidence(
            workflow_id="oel_scale_validation",
            status=str(payload["status"]),
            disposition="validation_passed" if quality["passed"] else "validation_failed",
            inputs=[
                {
                    "input_id": "source_config",
                    "path": str(source),
                    "sha256": payload["source_config_digest"],
                }
            ],
            quality_gates=quality,
            warnings=[],
            failures=[] if quality["passed"] else [{"phase": "validation", "quality": quality}],
            artifacts=[
                artifact_reference(
                    evidence_json,
                    artifact_id="scale_validation_evidence",
                    media_type="application/json",
                ),
                artifact_reference(
                    evidence_md,
                    artifact_id="scale_validation_evidence_markdown",
                    media_type="text/markdown",
                ),
            ],
            provenance={"repository": payload["repository"], "telemetry": payload["telemetry"]},
            data_markings={"scope": "public_local", "synthetic": True, "hidden_truth_included": False},
            non_claims=payload["non_claims"],
            domain_summary={"tier": tier, "workload": payload["workload"]},
        ),
    )
    payload["artifacts"]["workflow_evidence"] = str(workflow_evidence_path)
    return payload


def _runtime_config(
    raw: dict[str, Any],
    *,
    source: Path,
    run_root: Path,
    policy: ConfigPathPolicy,
) -> dict[str, Any]:
    runtime = deepcopy(raw)
    synthetic = dict(runtime.get("synthetic_data", {}) or {})
    truth = dict(synthetic.get("truth", {}) or {})
    template_source = policy.resolve_input_file(
        truth.get("template_tle_file", ""),
        purpose="synthetic_data.truth.template_tle_file",
        base_dir=source.parent,
    )
    run_root.mkdir(parents=True, exist_ok=True)
    template_copy = run_root / "synthetic_template.tle"
    shutil.copy2(template_source, template_copy)
    truth["template_tle_file"] = str(template_copy)
    synthetic["truth"] = truth
    synthetic["output_dir"] = str(run_root / "data_generation")
    synthetic["store"] = {"path": str(run_root / "catalog_store.sqlite"), "backend": "sqlite"}
    runtime["synthetic_data"] = synthetic
    scale = dict(runtime.get("scale_catalog", {}) or {})
    scale["mode"] = "run"
    scale["store"] = {"path": str(run_root / "catalog_store.sqlite"), "backend": "sqlite"}
    planner = dict(scale.get("planner", {}) or {})
    planner["output_dir"] = str(run_root / "planner")
    scale["planner"] = planner
    handoff_key = "onp_handoff" if "onp_handoff" in scale else "hpop_handoff"
    handoff = dict(scale.get(handoff_key, {}) or {})
    handoff["enabled"] = False
    handoff["output_dir"] = str(run_root / "handoff")
    scale[handoff_key] = handoff
    runtime["scale_catalog"] = scale
    return runtime


def _brute_force_reference(cfg: Any) -> dict[str, Any]:
    products = load_matching_current_propagation_products(
        cfg.store_path,
        object_ids=resolve_selected_object_ids(cfg),
        windows=_expected_propagation_windows(cfg),
        model=cfg.propagation_model,
        backend=cfg.propagation_backend,
        output_frame=cfg.output_frame,
    )
    grouped: dict[tuple[float, float, float], list[PropagationProductSummary]] = defaultdict(list)
    for item in products:
        grouped[(item.start_jd_utc, item.end_jd_utc, item.step_s)].append(item)
    pair_metrics: dict[tuple[str, str, float, float], dict[str, Any]] = {}
    distance_evaluations = 0
    for (window_start, window_end, _step), window_products in sorted(grouped.items()):
        ordered = sorted(window_products, key=lambda item: item.object_id)
        samples = {
            item.object_id: load_product_samples(
                cfg.store_path, propagation_product_id=item.propagation_product_id
            )
            for item in ordered
        }
        sample_indices = sorted(
            set.intersection(*(set(row.sample_index for row in values) for values in samples.values()))
        )
        by_object_index = {
            object_id: {row.sample_index: row for row in values} for object_id, values in samples.items()
        }
        object_ids = [item.object_id for item in ordered]
        for sample_index in sample_indices:
            rows = [by_object_index[object_id][sample_index] for object_id in object_ids]
            valid_indices = [
                index
                for index, row in enumerate(rows)
                if not row.error
                and all(value is not None for value in (row.pos_x_km, row.pos_y_km, row.pos_z_km))
            ]
            if len(valid_indices) < 2:
                continue
            positions = np.array(
                [[rows[index].pos_x_km, rows[index].pos_y_km, rows[index].pos_z_km] for index in valid_indices],
                dtype=float,
            )
            velocities = np.array(
                [[rows[index].vel_x_km_s, rows[index].vel_y_km_s, rows[index].vel_z_km_s] for index in valid_indices],
                dtype=float,
            )
            ids = [object_ids[index] for index in valid_indices]
            jd_utc = float(rows[valid_indices[0]].jd_utc)
            for block_start in range(0, len(ids), 256):
                block_end = min(block_start + 256, len(ids))
                deltas = positions[block_start:block_end, None, :] - positions[None, :, :]
                distances = np.linalg.norm(deltas, axis=2)
                distance_evaluations += sum(len(ids) - (block_start + local_index) - 1 for local_index in range(block_end - block_start))
                for local_index, global_index in enumerate(range(block_start, block_end)):
                    close_indices = np.flatnonzero(distances[local_index] <= cfg.distance_threshold_km)
                    for right_index in close_indices:
                        if int(right_index) <= global_index:
                            continue
                        left_id, right_id = sorted((ids[global_index], ids[int(right_index)]))
                        relative_speed = float(
                            np.linalg.norm(velocities[global_index] - velocities[int(right_index)])
                        )
                        key = (left_id, right_id, window_start, window_end)
                        distance = float(distances[local_index, int(right_index)])
                        current = pair_metrics.get(key)
                        if current is None or distance < current["miss_distance_km"]:
                            pair_metrics[key] = {
                                "deputy_id": left_id,
                                "chief_id": right_id,
                                "window_start_jd_utc": window_start,
                                "window_end_jd_utc": window_end,
                                "tca_jd_utc": jd_utc,
                                "miss_distance_km": distance,
                                "relative_speed_km_s": relative_speed,
                            }
    ordered_pairs = sorted(
        pair_metrics.values(),
        key=lambda row: (
            row["miss_distance_km"],
            row["relative_speed_km_s"],
            row["deputy_id"],
            row["chief_id"],
            row["window_start_jd_utc"],
        ),
    )
    return {
        "method": "vectorized_blocked_all_pairs_sample_reference",
        "distance_evaluations": distance_evaluations,
        "pair_windows": ordered_pairs,
    }


def _actual_screening_evidence(store_path: Path) -> dict[str, Any]:
    with sqlite3.connect(store_path) as conn:
        windows = conn.execute(
            """
            SELECT deputy_id, chief_id, window_start_jd_utc, window_end_jd_utc,
                   min_sample_time_jd_utc, min_sample_distance_km, relative_speed_km_s
            FROM screening_windows
            ORDER BY min_sample_distance_km, relative_speed_km_s, deputy_id, chief_id, window_start_jd_utc
            """
        ).fetchall()
        candidates = conn.execute(
            """
            SELECT deputy_id, chief_id, tca_jd_utc, miss_distance_km,
                   relative_speed_km_s, candidate_id
            FROM conjunction_candidates
            WHERE promotion_status != 'superseded'
            ORDER BY miss_distance_km, relative_speed_km_s, candidate_id
            """
        ).fetchall()
    return {
        "pair_windows": [
            {
                "deputy_id": str(row[0]),
                "chief_id": str(row[1]),
                "window_start_jd_utc": float(row[2]),
                "window_end_jd_utc": float(row[3]),
                "tca_jd_utc": float(row[4]),
                "miss_distance_km": float(row[5]),
                "relative_speed_km_s": float(row[6]),
            }
            for row in windows
        ],
        "candidates": [
            {
                "deputy_id": str(row[0]),
                "chief_id": str(row[1]),
                "tca_jd_utc": float(row[2]),
                "miss_distance_km": float(row[3]),
                "relative_speed_km_s": float(row[4]),
                "candidate_id": str(row[5]),
            }
            for row in candidates
        ],
    }


def _parity_payload(reference: dict[str, Any], actual: dict[str, Any], *, promotion_limit: int) -> dict[str, Any]:
    def key(row: dict[str, Any]) -> tuple[str, str, float, float]:
        return (
            str(row["deputy_id"]),
            str(row["chief_id"]),
            float(row["window_start_jd_utc"]),
            float(row["window_end_jd_utc"]),
        )

    reference_keys = {key(row) for row in reference["pair_windows"]}
    actual_keys = {key(row) for row in actual["pair_windows"]}
    reference_order = [
        (row["deputy_id"], row["chief_id"]) for row in reference["pair_windows"][:promotion_limit]
    ]
    actual_order = [(row["deputy_id"], row["chief_id"]) for row in actual["candidates"]]
    false_negatives = sorted(reference_keys - actual_keys)
    false_positives = sorted(actual_keys - reference_keys)
    return {
        "reference_method": reference["method"],
        "distance_evaluations": reference["distance_evaluations"],
        "reference_pair_window_count": len(reference_keys),
        "spatial_pair_window_count": len(actual_keys),
        "false_negative_count": len(false_negatives),
        "false_positive_count": len(false_positives),
        "false_negatives": false_negatives,
        "false_positives": false_positives,
        "candidate_order_matches": actual_order == reference_order,
        "reference_candidate_order": reference_order,
        "actual_candidate_order": actual_order,
    }


def _truth_scoring(expected: dict[str, Any], actual: dict[str, Any]) -> dict[str, Any]:
    expected_pairs = {tuple(sorted(pair)) for pair in expected.get("expected_candidate_pairs", [])}
    actual_pairs = {
        tuple(sorted((row["deputy_id"], row["chief_id"]))) for row in actual.get("candidates", [])
    }
    actual_by_pair = {
        tuple(sorted((row["deputy_id"], row["chief_id"]))): row
        for row in actual.get("candidates", [])
    }
    tolerance_rows: list[dict[str, Any]] = []
    for expected_candidate in expected.get("expected_candidates", []) or []:
        pair = tuple(sorted(str(item) for item in expected_candidate.get("object_ids", [])))
        actual_candidate = actual_by_pair.get(pair)
        expected_tca = expected_candidate.get("expected_tca_jd_utc")
        expected_miss = expected_candidate.get("expected_miss_distance_km")
        tca_error_s = (
            abs(float(actual_candidate["tca_jd_utc"]) - float(expected_tca)) * 86400.0
            if actual_candidate is not None and expected_tca is not None
            else None
        )
        miss_error_km = (
            abs(float(actual_candidate["miss_distance_km"]) - float(expected_miss))
            if actual_candidate is not None and expected_miss is not None
            else None
        )
        tca_tolerance_s = float(expected_candidate.get("tca_tolerance_s", 0.0) or 0.0)
        miss_tolerance_km = float(
            expected_candidate.get("miss_distance_tolerance_km", 0.0) or 0.0
        )
        tolerance_rows.append(
            {
                "event_id": expected_candidate.get("event_id", ""),
                "object_ids": pair,
                "tca_error_s": tca_error_s,
                "tca_tolerance_s": tca_tolerance_s,
                "miss_distance_error_km": miss_error_km,
                "miss_distance_tolerance_km": miss_tolerance_km,
                "passed": actual_candidate is not None
                and (tca_error_s is None or tca_error_s <= tca_tolerance_s)
                and (miss_error_km is None or miss_error_km <= miss_tolerance_km),
            }
        )
    return {
        "expected_pairs": sorted(expected_pairs),
        "actual_pairs": sorted(actual_pairs),
        "missed_expected_pairs": sorted(expected_pairs - actual_pairs),
        "unexpected_candidate_pairs": sorted(actual_pairs - expected_pairs),
        "expected_pairs_detected": expected_pairs.issubset(actual_pairs),
        "negative_controls_clean": not (actual_pairs - expected_pairs),
        "candidate_tolerances_passed": all(row["passed"] for row in tolerance_rows),
        "candidate_tolerance_rows": tolerance_rows,
    }


def _quality_payload(
    validation: dict[str, Any],
    *,
    parity: dict[str, Any],
    truth_scoring: dict[str, Any],
    total_wall_s: float,
) -> dict[str, Any]:
    max_runtime_s = validation.get("max_runtime_s")
    checks = {
        "zero_spatial_false_negatives": parity["false_negative_count"] == 0,
        "zero_spatial_false_positives": parity["false_positive_count"] == 0,
        "deterministic_candidate_order": bool(parity["candidate_order_matches"]),
        "expected_close_approaches_detected": bool(truth_scoring["expected_pairs_detected"]),
        "negative_controls_clean": bool(truth_scoring["negative_controls_clean"]),
        "candidate_tca_and_miss_tolerances": bool(truth_scoring["candidate_tolerances_passed"]),
        "runtime_within_limit": max_runtime_s is None or total_wall_s <= float(max_runtime_s),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "thresholds": {"max_runtime_s": max_runtime_s},
    }


def _repository_state(start: Path) -> dict[str, Any]:
    try:
        root = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=start,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"], cwd=root, check=True, capture_output=True, text=True
            ).stdout.strip()
        )
        return {"root": root, "commit": commit, "dirty": dirty}
    except (OSError, subprocess.SubprocessError):
        return {"root": "", "commit": "", "dirty": None}


def _timed(function: Any, *args: Any) -> tuple[Any, float]:
    started = time.perf_counter()
    result = function(*args)
    return result, time.perf_counter() - started


def _max_rss_mb() -> float | None:
    return max_rss_mb()


def _render_markdown(payload: dict[str, Any]) -> str:
    workload = payload["workload"]
    parity = payload["parity"]
    truth = payload["truth_scoring"]
    telemetry = payload["telemetry"]
    lines = [
        f"# OEL Scale Validation: {payload['tier']}",
        "",
        f"- Status: `{payload['status']}`",
        f"- Objects: `{workload['object_count']}`",
        f"- State samples: `{workload['propagation_sample_count']}`",
        f"- Possible pair windows: `{workload['possible_pair_windows']}`",
        f"- Spatial candidates: `{workload['spatial_candidate_pair_windows']}`",
        f"- Pruned pair windows: `{workload['pruned_pair_windows']}`",
        f"- Brute-force false negatives: `{parity['false_negative_count']}`",
        f"- Brute-force false positives: `{parity['false_positive_count']}`",
        f"- Candidate order matches: `{parity['candidate_order_matches']}`",
        f"- Expected pairs detected: `{truth['expected_pairs_detected']}`",
        f"- Negative controls clean: `{truth['negative_controls_clean']}`",
        f"- Candidate TCA/miss tolerances: `{truth['candidate_tolerances_passed']}`",
        f"- Wall time: `{telemetry['wall_time_s']:.3f}` s",
        f"- Max RSS growth: `{_format_optional_float(telemetry['max_rss_growth_mb'])}` MB",
        f"- Store size: `{telemetry['store_size_mb']:.3f}` MB",
        "",
        "## Quality Gates",
        "",
    ]
    lines.extend(
        f"- {name}: `{'PASS' if passed else 'FAIL'}`"
        for name, passed in payload["quality"]["checks"].items()
    )
    lines.extend(["", "## Non-Claims", ""])
    lines.extend(f"- {item}" for item in payload["non_claims"])
    return "\n".join(lines) + "\n"


def _format_optional_float(value: Any) -> str:
    return "unavailable" if value is None else f"{float(value):.3f}"
