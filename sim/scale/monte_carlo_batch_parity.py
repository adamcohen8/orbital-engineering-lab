from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from sim.dynamics.orbit.sgp4 import sgp4_propagate_teme_batch_numba, sgp4_propagate_teme_batch_reference
from sim.dynamics.orbit.tle import parse_tle_lines
from sim.scale.monte_carlo_planning import (
    _digest,
    _run_artifact_stem,
)
from sim.scale.planner import ScalePlannerConfig
from sim.scale.store import (
    CampaignTrialResultRecord,
    load_current_tle_states,
)
from sim.scale.trial_builders import fit_payload_to_elements


def _attach_numba_chunked_propagation_summaries(
    cfg: ScalePlannerConfig,
    *,
    results: list[CampaignTrialResultRecord],
    duration_s: float | None,
    step_s: float | None,
    chunk_duration_s: float,
) -> tuple[list[CampaignTrialResultRecord], dict[str, Any]]:
    wall_start = time.perf_counter()
    eligible = [_chunked_propagation_payload(result) for result in results]
    eligible = [item for item in eligible if item is not None]
    if not eligible:
        return results, {
            "backend": "numba_chunked",
            "status": "blocked",
            "reason": "No completed fitted SGP4 trial products were eligible for chunked propagation.",
            "wall_time_s": time.perf_counter() - wall_start,
        }
    target_ids = tuple({str(item["target_object_id"]) for item in eligible if str(item.get("target_object_id", ""))})
    object_ids = target_ids or cfg.object_ids
    states = load_current_tle_states(cfg.store_path, object_ids=object_ids)
    if len(states) != 1:
        raise ValueError("Numba chunked MC propagation currently requires exactly one deployer TLE state.")

    deployer_elements = parse_tle_lines(states[0].line1, states[0].line2)
    epoch_jd_utc = float(cfg.start_jd_utc)
    duration = max(float(cfg.duration_s if duration_s is None else duration_s), 0.0)
    step = max(float(cfg.step_s if step_s is None else step_s), 1.0)
    times_s = _propagation_time_grid(duration_s=duration, step_s=step)
    max_chunk_samples = max(1, int(np.floor(float(chunk_duration_s) / step)) + 1)

    element_rows = []
    for item in eligible:
        deployed_elements = fit_payload_to_elements(deployer_elements, dict(item["fit"]["estimated_elements"]))
        element_rows.extend([deployer_elements, deployed_elements])

    accumulator = [
        {
            "initial_separation_km": None,
            "final_separation_km": None,
            "min_separation_km": float("inf"),
            "max_separation_km": float("-inf"),
            "time_of_min_separation_s": None,
            "time_of_max_separation_s": None,
            "final_relative_position_teme_km": None,
            "error_count": 0,
            "errors": [],
        }
        for _ in eligible
    ]
    chunk_count = 0
    observed_max_chunk_samples = 0
    batch_kernel_s = 0.0
    for start in range(0, int(len(times_s)), max_chunk_samples):
        chunk_count += 1
        chunk_times_s = times_s[start : min(start + max_chunk_samples, int(len(times_s)))]
        observed_max_chunk_samples = max(observed_max_chunk_samples, int(len(chunk_times_s)))
        jd_grid = epoch_jd_utc + chunk_times_s / 86400.0
        time_rows = [
            (jd_grid - float(elements.epoch_jd_utc)) * 1440.0
            for elements in element_rows
        ]
        batch_start = time.perf_counter()
        batch = sgp4_propagate_teme_batch_numba(element_rows, np.asarray(time_rows, dtype=float))
        batch_kernel_s += time.perf_counter() - batch_start
        for index, state in enumerate(accumulator):
            deployer_row = 2 * index
            deployed_row = deployer_row + 1
            row_errors = [
                str(error)
                for error in [*batch.errors[deployer_row].tolist(), *batch.errors[deployed_row].tolist()]
                if str(error)
            ]
            if row_errors:
                state["error_count"] = int(state["error_count"]) + len(row_errors)
                state["errors"].extend(row_errors[:5])
            rel = batch.position_teme_km[deployed_row] - batch.position_teme_km[deployer_row]
            separations = np.linalg.norm(rel, axis=1)
            if state["initial_separation_km"] is None and separations.size:
                state["initial_separation_km"] = float(separations[0])
            if separations.size:
                local_min_index = int(np.argmin(separations))
                local_max_index = int(np.argmax(separations))
                local_min = float(separations[local_min_index])
                local_max = float(separations[local_max_index])
                if local_min < float(state["min_separation_km"]):
                    state["min_separation_km"] = local_min
                    state["time_of_min_separation_s"] = float(chunk_times_s[local_min_index])
                if local_max > float(state["max_separation_km"]):
                    state["max_separation_km"] = local_max
                    state["time_of_max_separation_s"] = float(chunk_times_s[local_max_index])
                if start + len(chunk_times_s) >= len(times_s):
                    state["final_separation_km"] = float(separations[-1])
                    state["final_relative_position_teme_km"] = [float(value) for value in rel[-1]]

    updated_by_trial: dict[str, CampaignTrialResultRecord] = {}
    for item, state in zip(eligible, accumulator, strict=True):
        result = item["result"]
        if int(state["error_count"]):
            updated_by_trial[result.trial_id] = _with_numba_chunked_error(
                result,
                error_count=int(state["error_count"]),
                errors=[str(error) for error in state["errors"]],
            )
            continue
        summary = {
            "schema_version": 1,
            "product_type": "sgp4_trial_propagation_summary",
            "model": "sgp4",
            "backend": "numba_chunked",
            "batch_backend": "numba_cpu",
            "frame": "teme",
            "epoch_jd_utc": epoch_jd_utc,
            "duration_s": duration,
            "step_s": step,
            "sample_count": int(len(times_s)),
            "chunk_duration_s": float(chunk_duration_s),
            "chunk_count": int(chunk_count),
            "max_chunk_sample_count": int(observed_max_chunk_samples),
            "initial_separation_km": float(state["initial_separation_km"]),
            "final_separation_km": float(state["final_separation_km"]),
            "min_separation_km": float(state["min_separation_km"]),
            "max_separation_km": float(state["max_separation_km"]),
            "time_of_min_separation_s": float(state["time_of_min_separation_s"]),
            "time_of_max_separation_s": float(state["time_of_max_separation_s"]),
            "final_relative_position_teme_km": list(state["final_relative_position_teme_km"]),
            "error_count": 0,
            "non_claims": [
                "Propagation uses synthetic SGP4 mean elements generated from analyst-defined deployment assumptions.",
                "This is not an observed catalog TLE propagation.",
                "Chunked propagation retains summary statistics only; dense trajectories are not stored.",
            ],
        }
        updated_by_trial[result.trial_id] = _with_propagation_summary(
            result,
            propagation_summary=summary,
            propagation_summary_backend="numba_chunked",
        )
    element_row_count = int(len(element_rows))
    state_evaluations = int(element_row_count * len(times_s))
    peak_position_velocity_bytes = int(element_row_count * observed_max_chunk_samples * 3 * 8 * 2)
    telemetry = {
        "backend": "numba_chunked",
        "batch_backend": "numba_cpu",
        "status": "completed",
        "eligible_trial_count": int(len(eligible)),
        "element_row_count": element_row_count,
        "sample_count": int(len(times_s)),
        "chunk_count": int(chunk_count),
        "chunk_duration_s": float(chunk_duration_s),
        "max_chunk_sample_count": int(observed_max_chunk_samples),
        "state_evaluations": state_evaluations,
        "batch_kernel_s": float(batch_kernel_s),
        "wall_time_s": time.perf_counter() - wall_start,
        "state_evaluations_per_batch_kernel_s": (
            float(state_evaluations / batch_kernel_s) if batch_kernel_s > 0.0 else None
        ),
        "peak_position_shape": [element_row_count, int(observed_max_chunk_samples), 3],
        "peak_velocity_shape": [element_row_count, int(observed_max_chunk_samples), 3],
        "estimated_peak_position_velocity_bytes": peak_position_velocity_bytes,
        "estimated_peak_position_velocity_mib": float(peak_position_velocity_bytes / (1024.0 * 1024.0)),
    }
    return [updated_by_trial.get(result.trial_id, result) for result in results], telemetry


def _chunked_propagation_payload(result: CampaignTrialResultRecord) -> dict[str, Any] | None:
    if result.status != "completed":
        return None
    product = result.metrics.get("builder_product")
    if not isinstance(product, dict):
        return None
    fit = product.get("sgp4_fit")
    if not isinstance(fit, dict):
        return None
    if fit.get("status") != "completed" or not isinstance(fit.get("estimated_elements"), dict):
        return None
    deployer = product.get("deployer")
    target_object_id = ""
    if isinstance(deployer, dict):
        target_object_id = str(deployer.get("object_id", "") or "")
    return {
        "result": result,
        "target_object_id": target_object_id,
        "fit": fit,
    }


def _propagation_time_grid(*, duration_s: float, step_s: float) -> np.ndarray:
    duration = max(float(duration_s), 0.0)
    step = max(float(step_s), 1.0)
    times_s = np.arange(int(np.floor(duration / step)) + 1, dtype=float) * step
    if times_s[-1] < duration - 1.0e-9:
        times_s = np.append(times_s, duration)
    return times_s


def _with_numba_chunked_error(
    result: CampaignTrialResultRecord,
    *,
    error_count: int,
    errors: list[str],
) -> CampaignTrialResultRecord:
    metrics = dict(result.metrics)
    metrics["propagation_summary_backend"] = "numba_chunked"
    metrics["propagation_error_count"] = int(error_count)
    digest = _digest(
        {
            "run_id": result.run_id,
            "campaign_id": result.campaign_id,
            "trial_id": result.trial_id,
            "trial_index": result.trial_index,
            "backend": result.backend,
            "status": "error",
            "builder_status": result.builder_status,
            "metrics": metrics,
            "product_refs": list(result.product_refs),
            "error": errors,
        }
    )
    result_prefix = str(result.result_id).split(":", 1)[0] if ":" in result.result_id else "campaign_result"
    message = f"Numba chunked SGP4 propagation returned {error_count} sample errors."
    if errors:
        message += f" First errors: {', '.join(errors[:3])}"
    return replace(result, result_id=f"{result_prefix}:{digest}", status="error", metrics=metrics, input_digest=digest, error=message)


def _with_propagation_summary(
    result: CampaignTrialResultRecord,
    *,
    propagation_summary: dict[str, Any],
    propagation_summary_backend: str,
) -> CampaignTrialResultRecord:
    metrics = dict(result.metrics)
    product = dict(metrics.get("builder_product", {}) or {})
    product["sgp4_trial_propagation"] = propagation_summary
    product_refs = _updated_product_refs(product=product, existing_refs=result.product_refs, propagation_summary=propagation_summary)
    metrics.update(
        {
            "builder_product": product,
            "propagation_summary_backend": propagation_summary_backend,
            "propagation_duration_s": float(propagation_summary["duration_s"]),
            "propagation_step_s": float(propagation_summary["step_s"]),
            "propagation_sample_count": int(propagation_summary["sample_count"]),
            "final_separation_km": float(propagation_summary["final_separation_km"]),
            "max_separation_km": float(propagation_summary["max_separation_km"]),
            "min_separation_km": float(propagation_summary["min_separation_km"]),
            "time_of_min_separation_s": float(propagation_summary["time_of_min_separation_s"]),
            "time_of_max_separation_s": float(propagation_summary["time_of_max_separation_s"]),
            "propagation_chunk_duration_s": float(propagation_summary["chunk_duration_s"]),
            "propagation_chunk_count": int(propagation_summary["chunk_count"]),
        }
    )
    digest = _digest(
        {
            "run_id": result.run_id,
            "campaign_id": result.campaign_id,
            "trial_id": result.trial_id,
            "trial_index": result.trial_index,
            "backend": result.backend,
            "status": result.status,
            "builder_status": result.builder_status,
            "metrics": metrics,
            "product_refs": list(product_refs),
            "propagation_summary_backend": propagation_summary_backend,
        }
    )
    result_prefix = str(result.result_id).split(":", 1)[0] if ":" in result.result_id else "campaign_result"
    return replace(
        result,
        result_id=f"{result_prefix}:{digest}",
        metrics=metrics,
        product_refs=product_refs,
        input_digest=digest,
    )


def _updated_product_refs(
    *,
    product: dict[str, Any],
    existing_refs: tuple[str, ...],
    propagation_summary: dict[str, Any],
) -> tuple[str, ...]:
    refs = [f"trial_product:{_digest(product)}"]
    refs.extend(ref for ref in existing_refs if ref.startswith("sgp4_fit:"))
    refs.append(f"sgp4_trial_propagation:{_digest(propagation_summary)}")
    return tuple(dict.fromkeys(refs))
def _write_batch_propagation_parity_artifact(
    cfg: ScalePlannerConfig,
    *,
    run_id: str,
    results: list[CampaignTrialResultRecord],
    batch_backend: str,
    execution_backend: str,
) -> tuple[Path | None, dict[str, Any] | None]:
    if not batch_backend:
        return None, None
    parity_path = (
        cfg.monte_carlo_output_dir
        / f"{_run_artifact_stem(backend=execution_backend, run_id=run_id)}_{batch_backend}_batch_parity.json"
    )
    parity = _build_batch_propagation_parity(cfg, run_id=run_id, results=results, batch_backend=batch_backend)
    parity_path.write_text(json.dumps(parity, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return parity_path, parity


def _build_batch_propagation_parity(
    cfg: ScalePlannerConfig,
    *,
    run_id: str,
    results: list[CampaignTrialResultRecord],
    batch_backend: str,
) -> dict[str, Any]:
    if batch_backend not in {"scalar_reference", "numba"}:
        raise ValueError("Batch propagation parity currently supports only scalar_reference or numba.")
    eligible = [_batch_parity_trial_payload(result) for result in results]
    eligible = [item for item in eligible if item is not None]
    if not eligible:
        return {
            "schema_version": 1,
            "status": "blocked",
            "backend": batch_backend,
            "run_id": run_id,
            "trial_count": 0,
            "reason": "No completed propagated SGP4 trial products were available for batch parity.",
        }
    target_ids = tuple({str(item["target_object_id"]) for item in eligible if str(item.get("target_object_id", ""))})
    object_ids = target_ids or cfg.object_ids
    states = load_current_tle_states(cfg.store_path, object_ids=object_ids)
    if len(states) != 1:
        return {
            "schema_version": 1,
            "status": "blocked",
            "backend": batch_backend,
            "run_id": run_id,
            "trial_count": len(eligible),
            "reason": "Batch propagation parity currently requires exactly one deployer TLE state.",
            "target_object_ids": list(object_ids),
        }
    deployer_elements = parse_tle_lines(states[0].line1, states[0].line2)
    epoch_jd_utc = float(eligible[0]["scalar_summary"]["epoch_jd_utc"])
    duration_s = float(eligible[0]["scalar_summary"]["duration_s"])
    step_s = float(eligible[0]["scalar_summary"]["step_s"])
    sample_count = int(eligible[0]["scalar_summary"]["sample_count"])
    times_s = np.arange(int(np.floor(duration_s / step_s)) + 1, dtype=float) * step_s
    if times_s[-1] < duration_s - 1.0e-9:
        times_s = np.append(times_s, duration_s)
    if len(times_s) != sample_count:
        return {
            "schema_version": 1,
            "status": "blocked",
            "backend": batch_backend,
            "run_id": run_id,
            "trial_count": len(eligible),
            "reason": "Batch propagation parity could not reconstruct the scalar propagation sample grid.",
            "scalar_sample_count": sample_count,
            "batch_sample_count": int(len(times_s)),
        }
    element_rows = []
    time_rows = []
    for item in eligible:
        scalar_summary = dict(item["scalar_summary"])
        if (
            int(scalar_summary["sample_count"]) != int(len(times_s))
            or abs(float(scalar_summary["duration_s"]) - duration_s) > 1.0e-9
            or abs(float(scalar_summary["step_s"]) - step_s) > 1.0e-9
        ):
            return {
                "schema_version": 1,
                "status": "blocked",
                "backend": batch_backend,
                "run_id": run_id,
                "trial_count": len(eligible),
                "reason": "Batch propagation parity requires all eligible trials to share duration, step, and sample count.",
            }
        deployed_elements = fit_payload_to_elements(deployer_elements, dict(item["fit"]["estimated_elements"]))
        element_rows.extend([deployer_elements, deployed_elements])
        jd_grid = epoch_jd_utc + times_s / 86400.0
        time_rows.append((jd_grid - float(deployer_elements.epoch_jd_utc)) * 1440.0)
        time_rows.append((jd_grid - float(deployed_elements.epoch_jd_utc)) * 1440.0)
    if batch_backend == "numba":
        batch = sgp4_propagate_teme_batch_numba(element_rows, np.asarray(time_rows, dtype=float))
    else:
        batch = sgp4_propagate_teme_batch_reference(element_rows, np.asarray(time_rows, dtype=float))
    trial_summaries: list[dict[str, Any]] = []
    diff_values = {
        "initial_separation_km": [],
        "final_separation_km": [],
        "min_separation_km": [],
        "max_separation_km": [],
        "final_relative_position_teme_km": [],
        "separation_profile_km": [],
    }
    errors = []
    for index, item in enumerate(eligible):
        deployer_row = 2 * index
        deployed_row = deployer_row + 1
        row_errors = [
            str(error)
            for error in [*batch.errors[deployer_row].tolist(), *batch.errors[deployed_row].tolist()]
            if str(error)
        ]
        errors.extend(row_errors)
        rel = batch.position_teme_km[deployed_row] - batch.position_teme_km[deployer_row]
        separations = np.linalg.norm(rel, axis=1)
        scalar = dict(item["scalar_summary"])
        batch_summary = {
            "initial_separation_km": float(separations[0]),
            "final_separation_km": float(separations[-1]),
            "min_separation_km": float(np.min(separations)),
            "max_separation_km": float(np.max(separations)),
            "final_relative_position_teme_km": [float(value) for value in rel[-1]],
        }
        scalar_final_rel = np.asarray(scalar.get("final_relative_position_teme_km", [np.nan, np.nan, np.nan]), dtype=float)
        diffs = {
            "initial_separation_km": abs(batch_summary["initial_separation_km"] - float(scalar["initial_separation_km"])),
            "final_separation_km": abs(batch_summary["final_separation_km"] - float(scalar["final_separation_km"])),
            "min_separation_km": abs(batch_summary["min_separation_km"] - float(scalar["min_separation_km"])),
            "max_separation_km": abs(batch_summary["max_separation_km"] - float(scalar["max_separation_km"])),
            "final_relative_position_teme_km": float(
                np.max(np.abs(np.asarray(batch_summary["final_relative_position_teme_km"], dtype=float) - scalar_final_rel))
            ),
            "separation_profile_km": max(
                abs(batch_summary["initial_separation_km"] - float(scalar["initial_separation_km"])),
                abs(batch_summary["final_separation_km"] - float(scalar["final_separation_km"])),
                abs(batch_summary["min_separation_km"] - float(scalar["min_separation_km"])),
                abs(batch_summary["max_separation_km"] - float(scalar["max_separation_km"])),
            ),
        }
        for name, value in diffs.items():
            diff_values[name].append(float(value))
        trial_summaries.append(
            {
                "trial_index": int(item["result"].trial_index),
                "trial_id": item["result"].trial_id,
                "status": "error" if row_errors else "completed",
                "batch_summary": batch_summary,
                "absolute_differences": diffs,
                "errors": row_errors,
            }
        )
    max_abs = {name: float(max(values)) if values else None for name, values in diff_values.items()}
    status = "error" if errors else "completed"
    return {
        "schema_version": 1,
        "status": status,
        "backend": batch_backend,
        "run_id": run_id,
        "model": "sgp4",
        "frame": "teme",
        "trial_count": len(eligible),
        "sample_count": int(len(times_s)),
        "duration_s": duration_s,
        "step_s": step_s,
        "position_shape": list(batch.position_teme_km.shape),
        "velocity_shape": list(batch.velocity_teme_km_s.shape),
        "max_absolute_differences": max_abs,
        "error_count": len(errors),
        "trial_summaries": trial_summaries,
        "non_claims": [
            "Batch parity compares summary statistics from the requested batched SGP4 backend.",
            "Compared trial summaries use synthetic fitted SGP4 mean elements, not observed catalog TLEs.",
        ],
    }


def _batch_parity_trial_payload(result: CampaignTrialResultRecord) -> dict[str, Any] | None:
    if result.status != "completed":
        return None
    product = result.metrics.get("builder_product")
    if not isinstance(product, dict):
        return None
    scalar_summary = product.get("sgp4_trial_propagation")
    fit = product.get("sgp4_fit")
    if not isinstance(scalar_summary, dict) or not isinstance(fit, dict):
        return None
    if fit.get("status") != "completed" or not isinstance(fit.get("estimated_elements"), dict):
        return None
    deployer = product.get("deployer")
    target_object_id = ""
    if isinstance(deployer, dict):
        target_object_id = str(deployer.get("object_id", "") or "")
    return {
        "result": result,
        "target_object_id": target_object_id,
        "scalar_summary": scalar_summary,
        "fit": fit,
    }
