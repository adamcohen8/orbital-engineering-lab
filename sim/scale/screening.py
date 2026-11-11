from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from dataclasses import replace
from itertools import product as cartesian_product
from pathlib import Path
from typing import Any

from sim.scale._output import staged_sqlite_store
from sim.scale.planner import (
    ScalePlannerConfig,
    _expected_propagation_windows,
    enforce_scale_planner_limits,
    load_scale_planner_config,
    resolve_selected_object_ids,
)
from sim.scale.store import (
    ConjunctionCandidateRecord,
    PropagationProductSummary,
    PropagationSampleRecord,
    ScreeningWindowRecord,
    load_matching_current_propagation_products,
    load_product_samples,
    load_product_samples_by_id,
    write_screening_results,
)
from sim.security.output_writes import OutputWriter


@enforce_scale_planner_limits("pair screening")
def run_pair_screening(
    config_path: str | Path,
    *,
    writer: OutputWriter | None = None,
) -> dict[str, Any]:
    cfg = load_scale_planner_config(config_path)
    if cfg.mode != "run":
        raise ValueError("Pair screening requires scale_catalog.mode: run.")
    if cfg.screening_mode != "sampled_distance":
        raise ValueError("Phase 1C implements only screening.mode: sampled_distance.")
    with staged_sqlite_store(cfg.store_path, writer=writer) as store_path:
        return _run_pair_screening(cfg, store_path=store_path)


def _run_pair_screening(cfg: ScalePlannerConfig, *, store_path: Path) -> dict[str, Any]:
    logical_store_path = cfg.store_path
    cfg = replace(cfg, store_path=store_path)
    products = load_matching_current_propagation_products(
        cfg.store_path,
        object_ids=resolve_selected_object_ids(cfg),
        windows=_expected_propagation_windows(cfg),
        model=cfg.propagation_model,
        backend=cfg.propagation_backend,
        output_frame=cfg.output_frame,
    )
    windows, candidate_pool = build_pair_screening_results(cfg, products=products)
    possible_pair_windows = _possible_pair_window_count(products)
    candidate_pool.sort(key=lambda item: (item.miss_distance_km, item.relative_speed_km_s, item.candidate_id))
    candidates = candidate_pool[: cfg.promotion_limit] if cfg.promotion_limit else []
    windows = _apply_promotion_limit_to_windows(windows, candidates=candidates)
    write_summary = write_screening_results(cfg.store_path, windows=windows, candidates=candidates)
    result = {
        "schema_version": 1,
        "status": "completed",
        "workflow": {
            "name": cfg.name,
            "config_path": str(cfg.config_path),
            "mode": cfg.mode,
        },
        "store_path": str(cfg.store_path),
        "screening_mode": cfg.screening_mode,
        "propagation_products_seen": len(products),
        "pair_windows_seen": write_summary["windows_seen"],
        "pair_windows_inserted": write_summary["windows_inserted"],
        "candidate_promotions_seen": write_summary["candidates_seen"],
        "candidate_promotions_inserted": write_summary["candidates_inserted"],
        "distance_threshold_km": cfg.distance_threshold_km,
        "promotion_limit": cfg.promotion_limit,
        "candidate_discovery": {
            "method": "sample_grid_spatial_hash",
            "possible_pair_windows": possible_pair_windows,
            "candidate_pair_windows": len(windows),
            "pruned_pair_windows": max(possible_pair_windows - len(windows), 0),
        },
        "non_claims": [
            "Pair screening used a sampled-state spatial hash and does not interpolate between samples.",
            "No closest-approach interpolation was performed.",
            "No covariance, collision probability, or ONP handoff was generated.",
        ],
    }
    if cfg.screening_numeric_backend == "rust":
        result["screening_numeric_backend"] = "rust"
    if cfg.screening_distance_comparison != "exact_sqrt":
        result["candidate_discovery"]["distance_comparison"] = {
            "mode": cfg.screening_distance_comparison,
            "equivalence": "rounding_level",
            "provenance": "explicit_scale_screening_config",
        }
    result["store_path"] = str(logical_store_path)
    return result


def build_pair_screening_results(
    cfg: ScalePlannerConfig,
    *,
    products: list[PropagationProductSummary],
) -> tuple[list[ScreeningWindowRecord], list[ConjunctionCandidateRecord]]:
    windows: list[ScreeningWindowRecord] = []
    candidates: list[ConjunctionCandidateRecord] = []
    if cfg.screening_numeric_backend == "rust":
        from sim.scale.rust_screening import rust_spatial_candidate_pairs

        discover = rust_spatial_candidate_pairs
    else:
        discover = _spatial_candidate_pairs
    for _, window_products in sorted(_products_by_common_window(products).items()):
        samples_by_product = load_product_samples_by_id(
            cfg.store_path,
            propagation_product_ids=tuple(
                item.propagation_product_id for item in window_products
            ),
        )
        for left, right, best in discover(
            window_products,
            samples_by_product=samples_by_product,
            distance_threshold_km=cfg.distance_threshold_km,
            distance_comparison=cfg.screening_distance_comparison,
        ):
            result = _screen_product_pair(
                cfg,
                left=left,
                right=right,
                samples_by_product=samples_by_product,
                precomputed_best=best,
            )
            if result is None:
                continue
            window, candidate = result
            windows.append(window)
            if candidate is not None:
                candidates.append(candidate)
    return windows, candidates


def build_pair_screening_results_for_product_pairs(
    cfg: ScalePlannerConfig,
    *,
    product_pairs: list[tuple[PropagationProductSummary, PropagationProductSummary]],
) -> tuple[list[ScreeningWindowRecord], list[ConjunctionCandidateRecord]]:
    windows: list[ScreeningWindowRecord] = []
    candidates: list[ConjunctionCandidateRecord] = []
    samples_by_product: dict[str, list[PropagationSampleRecord]] = {}
    sample_indices_by_product: dict[str, dict[int, PropagationSampleRecord]] = {}
    for left, right in product_pairs:
        if left.object_id == right.object_id:
            continue
        result = _screen_product_pair(
            cfg,
            left=left,
            right=right,
            samples_by_product=samples_by_product,
            sample_indices_by_product=sample_indices_by_product,
        )
        if result is None:
            continue
        window, candidate = result
        windows.append(window)
        if candidate is not None:
            candidates.append(candidate)
    return windows, candidates


def _screen_product_pair(
    cfg: ScalePlannerConfig,
    *,
    left: PropagationProductSummary,
    right: PropagationProductSummary,
    samples_by_product: dict[str, list[PropagationSampleRecord]] | None = None,
    sample_indices_by_product: dict[str, dict[int, PropagationSampleRecord]] | None = None,
    precomputed_best: tuple[float, float, float] | None = None,
) -> tuple[ScreeningWindowRecord, ConjunctionCandidateRecord | None] | None:
    sample_cache = samples_by_product or {}
    left_samples = sample_cache.get(left.propagation_product_id)
    if left_samples is None:
        left_samples = load_product_samples(cfg.store_path, propagation_product_id=left.propagation_product_id)
        sample_cache[left.propagation_product_id] = left_samples
    right_samples = sample_cache.get(right.propagation_product_id)
    if right_samples is None:
        right_samples = load_product_samples(cfg.store_path, propagation_product_id=right.propagation_product_id)
        sample_cache[right.propagation_product_id] = right_samples
    index_cache = sample_indices_by_product if sample_indices_by_product is not None else {}
    right_by_index = index_cache.get(right.propagation_product_id)
    if right_by_index is None:
        right_by_index = {sample.sample_index: sample for sample in right_samples}
        index_cache[right.propagation_product_id] = right_by_index
    best = precomputed_best
    if best is None:
        if cfg.screening_numeric_backend == "rust":
            best = _rust_pair_closest_approach(left_samples, right_by_index)
        else:
            for left_sample in left_samples:
                right_sample = right_by_index.get(left_sample.sample_index)
                if right_sample is None:
                    continue
                sample = _sampled_relative_state(left_sample, right_sample)
                if sample is None:
                    continue
                distance_km, relative_speed_km_s = sample
                current = (distance_km, left_sample.jd_utc, relative_speed_km_s)
                if best is None or current[0] < best[0]:
                    best = current
    if best is None:
        return None

    deputy_id, chief_id = sorted((left.object_id, right.object_id))
    product_ids = tuple(sorted((left.propagation_product_id, right.propagation_product_id)))
    digest_payload = {
            "method": cfg.screening_mode,
            "distance_threshold_km": cfg.distance_threshold_km,
            "deputy_id": deputy_id,
            "chief_id": chief_id,
            "window_start_jd_utc": left.start_jd_utc,
            "window_end_jd_utc": left.end_jd_utc,
            "step_s": left.step_s,
            "product_ids": product_ids,
            "product_digests": sorted((left.input_digest, right.input_digest)),
        }
    if cfg.screening_distance_comparison != "exact_sqrt":
        digest_payload["distance_comparison"] = cfg.screening_distance_comparison
    digest = _digest(digest_payload)
    min_distance_km, min_time_jd_utc, relative_speed_km_s = best
    under_threshold = min_distance_km <= cfg.distance_threshold_km
    window = ScreeningWindowRecord(
        screening_window_id=f"screen:{digest}",
        screening_run_id=f"screen_run:{cfg.name}",
        deputy_id=deputy_id,
        chief_id=chief_id,
        window_start_jd_utc=float(left.start_jd_utc),
        window_end_jd_utc=float(left.end_jd_utc),
        method=cfg.screening_mode,
        source_product_ids=product_ids,
        input_digest=digest,
        min_sample_distance_km=float(min_distance_km),
        min_sample_time_jd_utc=float(min_time_jd_utc),
        relative_speed_km_s=float(relative_speed_km_s),
        pruning_reason="within_threshold" if under_threshold else "outside_threshold",
        status="promoted_for_hpop" if under_threshold else "screened",
    )
    candidate = None
    if under_threshold:
        candidate_digest = _digest({"screening_window_digest": digest, "kind": "sampled_candidate"})
        ranking_features = {
            "method": cfg.screening_mode,
            "distance_threshold_km": cfg.distance_threshold_km,
            "min_sample_distance_km": min_distance_km,
            "source_product_ids": list(product_ids),
        }
        if cfg.screening_distance_comparison != "exact_sqrt":
            ranking_features["distance_comparison"] = cfg.screening_distance_comparison
            ranking_features["distance_comparison_equivalence"] = "rounding_level"
        candidate = ConjunctionCandidateRecord(
            candidate_id=f"candidate:{candidate_digest}",
            screening_run_id=window.screening_run_id,
            screening_window_id=window.screening_window_id,
            deputy_id=deputy_id,
            chief_id=chief_id,
            tca_jd_utc=float(min_time_jd_utc),
            miss_distance_km=float(min_distance_km),
            relative_speed_km_s=float(relative_speed_km_s),
            ranking_score=_ranking_score(min_distance_km, cfg.distance_threshold_km),
            ranking_features=ranking_features,
            promotion_status="candidate_for_hpop",
        )
    return window, candidate


def _rust_pair_closest_approach(
    left_samples: list[PropagationSampleRecord],
    right_by_index: dict[int, PropagationSampleRecord],
) -> tuple[float, float, float] | None:
    """Compute one explicit pair's sampled minimum through the Rust kernel."""

    import numpy as np

    from sim.rust_scale_backend import closest_sampled_approach

    left_rows: list[list[float]] = []
    right_rows: list[list[float]] = []
    times: list[float] = []
    for left_sample in left_samples:
        right_sample = right_by_index.get(left_sample.sample_index)
        if right_sample is None:
            continue
        if left_sample.error or right_sample.error:
            continue
        values = (
            left_sample.pos_x_km,
            left_sample.pos_y_km,
            left_sample.pos_z_km,
            left_sample.vel_x_km_s,
            left_sample.vel_y_km_s,
            left_sample.vel_z_km_s,
            right_sample.pos_x_km,
            right_sample.pos_y_km,
            right_sample.pos_z_km,
            right_sample.vel_x_km_s,
            right_sample.vel_y_km_s,
            right_sample.vel_z_km_s,
        )
        if any(value is None for value in values):
            continue
        if any(not math.isfinite(float(value)) for value in values):
            continue
        if not math.isfinite(float(left_sample.jd_utc)):
            continue
        if _sampled_relative_state(left_sample, right_sample) is None:
            continue
        left_rows.append(
            [
                float(left_sample.pos_x_km),
                float(left_sample.pos_y_km),
                float(left_sample.pos_z_km),
                float(left_sample.vel_x_km_s),
                float(left_sample.vel_y_km_s),
                float(left_sample.vel_z_km_s),
            ]
        )
        right_rows.append(
            [
                float(right_sample.pos_x_km),
                float(right_sample.pos_y_km),
                float(right_sample.pos_z_km),
                float(right_sample.vel_x_km_s),
                float(right_sample.vel_y_km_s),
                float(right_sample.vel_z_km_s),
            ]
        )
        times.append(float(left_sample.jd_utc))
    if not times:
        return None
    result = closest_sampled_approach(
        np.asarray(left_rows, dtype=np.float64),
        np.asarray(right_rows, dtype=np.float64),
        np.asarray(times, dtype=np.float64),
    )
    if result is None:
        return None
    _sample_index, distance_km, jd_utc, relative_speed_km_s = result
    return distance_km, jd_utc, relative_speed_km_s


def _apply_promotion_limit_to_windows(
    windows: list[ScreeningWindowRecord],
    *,
    candidates: list[ConjunctionCandidateRecord],
) -> list[ScreeningWindowRecord]:
    promoted_window_ids = {candidate.screening_window_id for candidate in candidates}
    updated: list[ScreeningWindowRecord] = []
    for window in windows:
        if window.status != "promoted_for_hpop":
            updated.append(window)
            continue
        if window.screening_window_id in promoted_window_ids:
            updated.append(window)
            continue
        updated.append(replace(window, status="screened", pruning_reason="candidate_limit_deferred"))
    return updated


def _products_by_common_window(
    products: list[PropagationProductSummary],
) -> dict[tuple[float, float, float], list[PropagationProductSummary]]:
    grouped: dict[tuple[float, float, float], list[PropagationProductSummary]] = defaultdict(list)
    for product in products:
        key = (product.start_jd_utc, product.end_jd_utc, product.step_s)
        grouped[key].append(product)
    return {key: value for key, value in grouped.items() if len({item.object_id for item in value}) >= 2}


def _possible_pair_window_count(products: list[PropagationProductSummary]) -> int:
    return sum(
        (len(window_products) * (len(window_products) - 1)) // 2
        for window_products in _products_by_common_window(products).values()
    )


def _spatial_candidate_pairs(
    products: list[PropagationProductSummary],
    *,
    samples_by_product: dict[str, list[PropagationSampleRecord]],
    distance_threshold_km: float,
    distance_comparison: str = "exact_sqrt",
) -> list[
    tuple[
        PropagationProductSummary,
        PropagationProductSummary,
        tuple[float, float, float],
    ]
]:
    """Discover sampled close pairs without materializing the all-pairs catalog."""

    threshold = float(distance_threshold_km)
    products_by_id = {item.propagation_product_id: item for item in products}
    samples_by_index: dict[int, list[tuple[str, PropagationSampleRecord]]] = defaultdict(list)
    for product_id, samples in samples_by_product.items():
        for sample in samples:
            if _sample_position(sample) is not None:
                samples_by_index[sample.sample_index].append((product_id, sample))

    if distance_comparison not in {"exact_sqrt", "squared_distance"}:
        raise ValueError("distance_comparison must be exact_sqrt or squared_distance.")
    threshold_comparison = threshold if distance_comparison == "exact_sqrt" else threshold * threshold
    best_by_pair: dict[
        tuple[str, str],
        tuple[float, int, float, PropagationSampleRecord, PropagationSampleRecord],
    ] = {}
    neighbor_offsets = tuple(cartesian_product((-1, 0, 1), repeat=3))
    for sample_rows in samples_by_index.values():
        buckets: dict[tuple[int, int, int], list[tuple[str, PropagationSampleRecord]]] = defaultdict(list)
        for product_id, sample in sample_rows:
            position = _sample_position(sample)
            if position is None:
                continue
            cell = tuple(math.floor(value / threshold) for value in position)
            buckets[cell].append((product_id, sample))
        for cell, rows in buckets.items():
            for offset in neighbor_offsets:
                neighbor = tuple(cell[index] + offset[index] for index in range(3))
                if neighbor not in buckets or neighbor < cell:
                    continue
                for left_id, left_sample in rows:
                    for right_id, right_sample in buckets[neighbor]:
                        if left_id >= right_id and cell == neighbor:
                            continue
                        if left_id == right_id:
                            continue
                        sampled = _sampled_distance(
                            left_sample,
                            right_sample,
                            squared=distance_comparison == "squared_distance",
                        )
                        if sampled is None or sampled[0] > threshold_comparison:
                            continue
                        comparison, distance_km = sampled
                        pair_id = tuple(sorted((left_id, right_id)))
                        ordered_left, ordered_right = (
                            (left_sample, right_sample)
                            if left_id == pair_id[0]
                            else (right_sample, left_sample)
                        )
                        current = (
                            comparison,
                            int(ordered_left.sample_index),
                            distance_km,
                            ordered_left,
                            ordered_right,
                        )
                        previous = best_by_pair.get(pair_id)
                        if previous is None or current[:2] < previous[:2]:
                            best_by_pair[pair_id] = current
    result = []
    for left_id, right_id in sorted(best_by_pair):
        _, _, stored_distance, left_sample, right_sample = best_by_pair[(left_id, right_id)]
        distance_km = (
            math.sqrt(stored_distance)
            if distance_comparison == "squared_distance"
            else stored_distance
        )
        relative = _sampled_relative_state(left_sample, right_sample)
        if relative is None:
            continue
        result.append(
            (
                products_by_id[left_id],
                products_by_id[right_id],
                (distance_km, left_sample.jd_utc, relative[1]),
            )
        )
    return result


def _sampled_distance(
    left: PropagationSampleRecord,
    right: PropagationSampleRecord,
    *,
    squared: bool,
) -> tuple[float, float] | None:
    if left.error or right.error:
        return None
    left_position = (left.pos_x_km, left.pos_y_km, left.pos_z_km)
    right_position = (right.pos_x_km, right.pos_y_km, right.pos_z_km)
    left_velocity = (left.vel_x_km_s, left.vel_y_km_s, left.vel_z_km_s)
    right_velocity = (right.vel_x_km_s, right.vel_y_km_s, right.vel_z_km_s)
    values = (*left_position, *right_position, *left_velocity, *right_velocity)
    if any(value is None or not math.isfinite(float(value)) for value in values):
        return None
    if not math.isfinite(float(left.jd_utc)) or not math.isfinite(float(right.jd_utc)):
        return None
    dr = [float(b) - float(a) for a, b in zip(left_position, right_position, strict=True)]
    if any(not math.isfinite(value) for value in dr):
        return None
    squared_distance = dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2]
    if not math.isfinite(squared_distance):
        return None
    if squared:
        return squared_distance, squared_distance
    distance = math.sqrt(squared_distance)
    return distance, distance


def _sample_position(sample: PropagationSampleRecord) -> tuple[float, float, float] | None:
    if sample.error:
        return None
    values = (sample.pos_x_km, sample.pos_y_km, sample.pos_z_km)
    if any(value is None or not math.isfinite(float(value)) for value in values):
        return None
    return tuple(float(value) for value in values)  # type: ignore[return-value]


def _sampled_relative_state(
    left: PropagationSampleRecord,
    right: PropagationSampleRecord,
) -> tuple[float, float] | None:
    if left.error or right.error:
        return None
    left_position = (left.pos_x_km, left.pos_y_km, left.pos_z_km)
    right_position = (right.pos_x_km, right.pos_y_km, right.pos_z_km)
    left_velocity = (left.vel_x_km_s, left.vel_y_km_s, left.vel_z_km_s)
    right_velocity = (right.vel_x_km_s, right.vel_y_km_s, right.vel_z_km_s)
    values = (*left_position, *right_position, *left_velocity, *right_velocity)
    if any(value is None or not math.isfinite(float(value)) for value in values):
        return None
    if not math.isfinite(float(left.jd_utc)) or not math.isfinite(float(right.jd_utc)):
        return None
    dr = [float(b) - float(a) for a, b in zip(left_position, right_position, strict=True)]
    dv = [float(b) - float(a) for a, b in zip(left_velocity, right_velocity, strict=True)]
    if any(not math.isfinite(value) for value in (*dr, *dv)):
        return None
    # Fixed-order scalar sums preserve exact parity with native screening
    # across Python's older and compensated built-in sum implementations.
    distance_squared = dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2]
    speed_squared = dv[0] * dv[0] + dv[1] * dv[1] + dv[2] * dv[2]
    if not math.isfinite(distance_squared) or not math.isfinite(speed_squared):
        return None
    return math.sqrt(distance_squared), math.sqrt(speed_squared)


def _ranking_score(miss_distance_km: float, threshold_km: float) -> float:
    if threshold_km <= 0.0:
        return 0.0
    return max(0.0, (threshold_km - miss_distance_km) / threshold_km)


def _digest(payload: dict[str, Any]) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
