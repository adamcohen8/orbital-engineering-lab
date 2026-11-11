from __future__ import annotations

import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any

from sim.dynamics.orbit.sgp4 import sgp4_propagate_teme, transform_teme_to_output_frame
from sim.scale._output import staged_sqlite_store
from sim.scale.planner import (
    ScalePlannerConfig,
    enforce_scale_planner_limits,
    load_scale_planner_config,
    resolve_selected_object_ids,
)
from sim.scale.propagation import _frame_transform
from sim.scale.store import (
    CandidateForRefinement,
    CandidateRefinementRecord,
    CurrentTLEState,
    current_ogp_elements,
    load_current_tle_states,
    load_pending_candidates_for_refinement,
    write_candidate_refinements,
)
from sim.security.output_writes import OutputWriter


@enforce_scale_planner_limits("candidate refinement")
def run_candidate_refinement(
    config_path: str | Path,
    *,
    writer: OutputWriter | None = None,
) -> dict[str, Any]:
    cfg = load_scale_planner_config(config_path)
    if cfg.mode != "run":
        raise ValueError("Candidate refinement requires scale_catalog.mode: run.")
    if cfg.propagation_model != "ogp" or cfg.propagation_backend not in {"scalar", "rust"}:
        raise ValueError("Candidate refinement requires OGP with scalar or rust propagation.")
    with staged_sqlite_store(cfg.store_path, writer=writer) as store_path:
        return _run_candidate_refinement(cfg, store_path=store_path)


def _run_candidate_refinement(cfg: ScalePlannerConfig, *, store_path: Path) -> dict[str, Any]:
    logical_store_path = cfg.store_path
    cfg = replace(cfg, store_path=store_path)
    candidates = load_pending_candidates_for_refinement(
        cfg.store_path,
        object_ids=resolve_selected_object_ids(cfg),
        limit=cfg.refinement_limit,
    )
    object_ids = tuple(sorted({item.deputy_id for item in candidates} | {item.chief_id for item in candidates}))
    states = {state.object_id: state for state in load_current_tle_states(cfg.store_path, object_ids=object_ids)}
    refinements = build_candidate_refinements(cfg, candidates=candidates, states=states)
    refinements.sort(key=lambda item: (item.refined_miss_distance_km, item.refined_relative_speed_km_s, item.refinement_id))
    write_summary = write_candidate_refinements(cfg.store_path, refinements=refinements)
    payload = {
        "schema_version": 1,
        "status": "completed",
        "workflow": {
            "name": cfg.name,
            "config_path": str(cfg.config_path),
            "mode": cfg.mode,
        },
        "store_path": str(cfg.store_path),
        "method": "sgp4_tighter_sampled_distance",
        "candidates_seen": len(candidates),
        "refinements_seen": write_summary["refinements_seen"],
        "refinements_inserted": write_summary["refinements_inserted"],
        "refinement_step_s": cfg.refinement_step_s,
        "refinement_half_window_s": cfg.refinement_half_window_s,
        "ranking_policy": cfg.ranking_policy,
        "non_claims": [
            "Candidate refinement used scalar SGP4 only.",
            "Closest approach is still selected from a sampled grid, not an analytic interpolation.",
            "No covariance, collision probability, or ONP propagation was generated.",
        ],
    }
    if cfg.propagation_backend == "rust":
        payload["numeric_backend"] = "rust"
        payload["non_claims"][0] = "Candidate refinement used rust OGP sampling."
    payload["store_path"] = str(logical_store_path)
    return payload


def build_candidate_refinements(
    cfg: ScalePlannerConfig,
    *,
    candidates: list[CandidateForRefinement],
    states: dict[str, CurrentTLEState],
) -> list[CandidateRefinementRecord]:
    refinements: list[CandidateRefinementRecord] = []
    parsed = {object_id: current_ogp_elements(state) for object_id, state in states.items()}
    for candidate in candidates:
        deputy_elements = parsed.get(candidate.deputy_id)
        chief_elements = parsed.get(candidate.chief_id)
        deputy_state = states.get(candidate.deputy_id)
        chief_state = states.get(candidate.chief_id)
        if deputy_elements is None or chief_elements is None or deputy_state is None or chief_state is None:
            continue
        result = _refine_candidate(cfg, candidate=candidate, deputy=deputy_elements, chief=chief_elements)
        if result is None:
            continue
        refined_tca_jd, refined_miss_km, refined_speed_km_s, start_jd, end_jd, sample_count = result
        digest = _digest(
            {
                "candidate_id": candidate.candidate_id,
                "source_product_ids": list(candidate.source_product_ids),
                "deputy_tle_digest": deputy_state.input_digest,
                "chief_tle_digest": chief_state.input_digest,
                "method": "sgp4_tighter_sampled_distance",
                "output_frame": cfg.output_frame,
                "sampled_tca_jd_utc": candidate.sampled_tca_jd_utc,
                "refinement_start_jd_utc": start_jd,
                "refinement_end_jd_utc": end_jd,
                "refinement_step_s": cfg.refinement_step_s,
            }
        )
        features = {
            "method": "sgp4_tighter_sampled_distance",
            "ranking_policy": cfg.ranking_policy,
            "distance_threshold_km": cfg.distance_threshold_km,
            "sample_count": sample_count,
            "source_product_ids": list(candidate.source_product_ids),
            "sampled_tca_jd_utc": candidate.sampled_tca_jd_utc,
            "sampled_miss_distance_km": candidate.sampled_miss_distance_km,
            "sampled_relative_speed_km_s": candidate.sampled_relative_speed_km_s,
            "refined_tca_jd_utc": refined_tca_jd,
            "refined_miss_distance_km": refined_miss_km,
            "refined_relative_speed_km_s": refined_speed_km_s,
            "delta_tca_s": (refined_tca_jd - candidate.sampled_tca_jd_utc) * 86400.0,
            "delta_miss_distance_km": refined_miss_km - candidate.sampled_miss_distance_km,
        }
        if cfg.propagation_backend == "rust":
            features["numeric_backend"] = "rust"
        refinements.append(
            CandidateRefinementRecord(
                refinement_id=f"refine:{digest}",
                candidate_id=candidate.candidate_id,
                screening_run_id=candidate.screening_run_id,
                screening_window_id=candidate.screening_window_id,
                deputy_id=candidate.deputy_id,
                chief_id=candidate.chief_id,
                method="sgp4_tighter_sampled_distance",
                source_product_ids=candidate.source_product_ids,
                input_digest=digest,
                sampled_tca_jd_utc=candidate.sampled_tca_jd_utc,
                sampled_miss_distance_km=candidate.sampled_miss_distance_km,
                sampled_relative_speed_km_s=candidate.sampled_relative_speed_km_s,
                refinement_start_jd_utc=start_jd,
                refinement_end_jd_utc=end_jd,
                refinement_step_s=cfg.refinement_step_s,
                refined_tca_jd_utc=refined_tca_jd,
                refined_miss_distance_km=refined_miss_km,
                refined_relative_speed_km_s=refined_speed_km_s,
                ranking_score=_ranking_score(refined_miss_km, cfg.distance_threshold_km),
                ranking_features=features,
                status="completed",
            )
        )
    return refinements


def _refine_candidate(
    cfg: ScalePlannerConfig,
    *,
    candidate: CandidateForRefinement,
    deputy: Any,
    chief: Any,
) -> tuple[float, float, float, float, float, int] | None:
    half_window_days = cfg.refinement_half_window_s / 86400.0
    start_jd = max(candidate.window_start_jd_utc, candidate.sampled_tca_jd_utc - half_window_days)
    end_jd = min(candidate.window_end_jd_utc, candidate.sampled_tca_jd_utc + half_window_days)
    if end_jd < start_jd:
        start_jd = candidate.sampled_tca_jd_utc
        end_jd = candidate.sampled_tca_jd_utc
    times = _sample_times(start_jd=start_jd, end_jd=end_jd, step_s=cfg.refinement_step_s)
    if cfg.propagation_backend == "rust":
        return _refine_candidate_rust_batch(
            cfg,
            deputy=deputy,
            chief=chief,
            times=times,
            start_jd=start_jd,
            end_jd=end_jd,
        )
    best: tuple[float, float, float] | None = None
    for jd_utc in times:
        deputy_state = _propagate(
            deputy, jd_utc=jd_utc, output_frame=cfg.output_frame,
        )
        chief_state = _propagate(
            chief, jd_utc=jd_utc, output_frame=cfg.output_frame,
        )
        if deputy_state is None or chief_state is None:
            continue
        distance_km, speed_km_s = _relative_distance_speed(deputy_state, chief_state)
        current = (distance_km, jd_utc, speed_km_s)
        if best is None or current[0] < best[0]:
            best = current
    if best is None:
        return None
    distance_km, tca_jd_utc, speed_km_s = best
    return tca_jd_utc, distance_km, speed_km_s, start_jd, end_jd, len(times)


def _refine_candidate_rust_batch(
    cfg: ScalePlannerConfig,
    *,
    deputy: Any,
    chief: Any,
    times: list[float],
    start_jd: float,
    end_jd: float,
) -> tuple[float, float, float, float, float, int] | None:
    """Refine one pair with one native OGP call per object, not per sample."""

    import numpy as np

    from sim.rust_scale_backend import propagate_pair_grid, relative_metrics

    grid = np.asarray(
        [
            [(float(jd) - float(deputy.epoch_jd_utc)) * 1440.0 for jd in times],
            [(float(jd) - float(chief.epoch_jd_utc)) * 1440.0 for jd in times],
        ],
        dtype=np.float64,
    )
    positions, velocities, errors = propagate_pair_grid([deputy, chief], grid)
    frame_transform = _frame_transform(cfg.output_frame)
    deputy_states = np.full((len(times), 6), np.nan, dtype=np.float64)
    chief_states = np.full((len(times), 6), np.nan, dtype=np.float64)
    for sample_index, jd_utc in enumerate(times):
        if errors[0, sample_index] or errors[1, sample_index]:
            continue
        deputy_pos, deputy_vel = transform_teme_to_output_frame(
            positions[0, sample_index],
            velocities[0, sample_index],
            jd_utc=float(jd_utc),
            output_frame=cfg.output_frame,
            frame_transform=frame_transform,
        )
        chief_pos, chief_vel = transform_teme_to_output_frame(
            positions[1, sample_index],
            velocities[1, sample_index],
            jd_utc=float(jd_utc),
            output_frame=cfg.output_frame,
            frame_transform=frame_transform,
        )
        deputy_states[sample_index, :3] = deputy_pos
        deputy_states[sample_index, 3:] = deputy_vel
        chief_states[sample_index, :3] = chief_pos
        chief_states[sample_index, 3:] = chief_vel
    metrics = relative_metrics(deputy_states, chief_states)
    valid = np.flatnonzero(np.isfinite(metrics[:, 0]))
    if valid.size == 0:
        return None
    sample_index = int(valid[int(np.argmin(metrics[valid, 0]))])
    distance_km = float(metrics[sample_index, 0])
    tca_jd_utc = float(times[sample_index])
    speed_km_s = float(metrics[sample_index, 1])
    return tca_jd_utc, distance_km, speed_km_s, start_jd, end_jd, len(times)


def _propagate(
    elements: Any, *, jd_utc: float, output_frame: str, rust_context: Any = None,
) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
    tsince_min = (float(jd_utc) - float(elements.epoch_jd_utc)) * 1440.0
    if rust_context is None:
        propagated = sgp4_propagate_teme(elements, tsince_min)
        if propagated.error:
            return None
        position_teme_km, velocity_teme_km_s = propagated.position_teme_km, propagated.velocity_teme_km_s
    else:
        try:
            position_teme_km, velocity_teme_km_s = rust_context.propagate(tsince_min)
        except (RuntimeError, ValueError):
            return None
    pos, vel = transform_teme_to_output_frame(
        position_teme_km,
        velocity_teme_km_s,
        jd_utc=jd_utc,
        output_frame=output_frame,
        frame_transform=_frame_transform(output_frame),
    )
    return (
        (float(pos[0]), float(pos[1]), float(pos[2])),
        (float(vel[0]), float(vel[1]), float(vel[2])),
    )


def _relative_distance_speed(
    deputy: tuple[tuple[float, float, float], tuple[float, float, float]],
    chief: tuple[tuple[float, float, float], tuple[float, float, float]],
) -> tuple[float, float]:
    deputy_pos, deputy_vel = deputy
    chief_pos, chief_vel = chief
    dr = [b - a for a, b in zip(deputy_pos, chief_pos, strict=True)]
    dv = [b - a for a, b in zip(deputy_vel, chief_vel, strict=True)]
    return math.sqrt(sum(item * item for item in dr)), math.sqrt(sum(item * item for item in dv))


def _sample_times(*, start_jd: float, end_jd: float, step_s: float) -> list[float]:
    duration_s = max(0.0, (float(end_jd) - float(start_jd)) * 86400.0)
    sample_count = int(math.floor(duration_s / step_s)) + 1
    times = [float(start_jd) + (index * step_s) / 86400.0 for index in range(sample_count)]
    if not times or times[-1] < float(end_jd) - 1.0e-12:
        times.append(float(end_jd))
    return times


def _ranking_score(miss_distance_km: float, threshold_km: float) -> float:
    if threshold_km <= 0.0:
        return 0.0
    return max(0.0, (threshold_km - miss_distance_km) / threshold_km)


def _digest(payload: dict[str, Any]) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
