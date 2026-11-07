from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from sim.interchange.manifests import bind_output_path, write_json_document
from sim.interchange.provenance import sha256_file
from sim.review import ReviewWorkspace

SUPPORTED_THRESHOLDS = frozenset(
    {
        "max_final_range_km",
        "max_allocation_force_residual_n",
        "max_allocation_saturated_duration_s",
        "max_pointing_error_deg",
        "min_final_propellant_kg",
        "min_burn_samples",
        "require_no_attitude_guardrail_events",
    }
)


def build_maneuver_readiness_packet(
    completed_run: str | Path,
    *,
    object_id: str,
    chief_id: str,
    thresholds: Mapping[str, Any],
    output_path: str | Path | None = None,
    logical_output_path: str | Path | None = None,
    writer: Any | None = None,
) -> dict[str, Any]:
    """Evaluate explicit evidence gates; never infer operational readiness silently."""

    unknown = sorted(set(thresholds) - SUPPORTED_THRESHOLDS)
    if unknown:
        raise ValueError(f"Unsupported maneuver-readiness thresholds: {unknown}")
    if not thresholds:
        raise ValueError("At least one explicit maneuver-readiness threshold is required.")
    workspace = ReviewWorkspace.open(completed_run)
    review_identity = workspace.evidence_identity()
    summary_path = workspace.output_dir / "master_run_summary.json"
    if not summary_path.is_file():
        raise ValueError("Completed run has no master_run_summary.json.")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    object_key = str(object_id)
    chief_key = str(chief_id)
    actuator = dict(
        dict(summary.get("actuator_diagnostics_summary", {}) or {}).get(object_key, {}) or {}
    )
    thrust = dict(dict(summary.get("thrust_stats", {}) or {}).get(object_key, {}) or {})
    guardrails = dict(summary.get("attitude_guardrail_stats", {}) or {})
    final_relative = workspace.query(
        "SELECT sample_index, time_s, range_km, range_rate_km_s FROM relative_state "
        "WHERE deputy_id = ? AND chief_id = ? ORDER BY sample_index DESC LIMIT 1",
        (object_key, chief_key),
        max_rows=2,
    )
    relative = dict(final_relative.rows[0]) if final_relative.row_count == 1 else {}
    realization = workspace.query(
        "SELECT interval_start_ns, interval_end_ns, saturated, "
        "requested_force_x_n, requested_force_y_n, requested_force_z_n, "
        "realized_force_x_n, realized_force_y_n, realized_force_z_n "
        "FROM actuator_realization WHERE object_id = ? ORDER BY interval_start_ns, actuator_id",
        (object_key,),
        max_rows=1_000_000,
    )
    maximum_residual_n: float | None = None
    saturated_duration_s = 0.0
    for row in realization.rows:
        requested = np.asarray(
            [
                row.get("requested_force_x_n"),
                row.get("requested_force_y_n"),
                row.get("requested_force_z_n"),
            ],
            dtype=float,
        )
        realized = np.asarray(
            [
                row.get("realized_force_x_n"),
                row.get("realized_force_y_n"),
                row.get("realized_force_z_n"),
            ],
            dtype=float,
        )
        if np.all(np.isfinite(requested)) and np.all(np.isfinite(realized)):
            residual = float(np.linalg.norm(requested - realized))
            maximum_residual_n = residual if maximum_residual_n is None else max(maximum_residual_n, residual)
        if bool(row.get("saturated")):
            saturated_duration_s += max(
                int(row.get("interval_end_ns") or 0) - int(row.get("interval_start_ns") or 0),
                0,
            ) / 1.0e9
    metrics = {
        "final_range_km": _finite_or_none(relative.get("range_km")),
        "final_range_rate_km_s": _finite_or_none(relative.get("range_rate_km_s")),
        "burn_samples": _finite_or_none(thrust.get("burn_samples")),
        "total_delta_v_m_s": _finite_or_none(thrust.get("total_dv_m_s")),
        "max_allocation_force_residual_n": (
            maximum_residual_n
            if maximum_residual_n is not None
            else _finite_or_none(actuator.get("max_rcs_force_residual_n"))
        ),
        "allocation_saturated_duration_s": (
            saturated_duration_s
            if realization.row_count > 0
            else _finite_or_none(actuator.get("rcs_allocation_saturated_duration_s"))
        ),
        "max_pointing_error_deg": _finite_or_none(actuator.get("max_attitude_error_deg")),
        "final_propellant_remaining_kg": _finite_or_none(
            actuator.get("final_propellant_remaining_kg")
        ),
        "attitude_guardrail_event_count": sum(
            int(value or 0) for value in guardrails.values() if isinstance(value, (int, float))
        ),
    }
    gates = []
    specs = (
        ("max_final_range_km", "final_range_km", "max"),
        ("max_allocation_force_residual_n", "max_allocation_force_residual_n", "max"),
        ("max_allocation_saturated_duration_s", "allocation_saturated_duration_s", "max"),
        ("max_pointing_error_deg", "max_pointing_error_deg", "max"),
        ("min_final_propellant_kg", "final_propellant_remaining_kg", "min"),
        ("min_burn_samples", "burn_samples", "min"),
    )
    for threshold_name, metric_name, comparison in specs:
        if threshold_name not in thresholds:
            continue
        threshold = _finite_threshold(thresholds[threshold_name], threshold_name)
        observed = metrics[metric_name]
        passed = None if observed is None else (
            observed <= threshold if comparison == "max" else observed >= threshold
        )
        gates.append(
            {
                "gate_id": threshold_name,
                "metric": metric_name,
                "comparison": comparison,
                "threshold": threshold,
                "observed": observed,
                "status": "unknown" if passed is None else ("pass" if passed else "fail"),
            }
        )
    if "require_no_attitude_guardrail_events" in thresholds:
        required = thresholds["require_no_attitude_guardrail_events"]
        if not isinstance(required, bool):
            raise ValueError("require_no_attitude_guardrail_events must be a boolean.")
        observed = metrics["attitude_guardrail_event_count"]
        passed = (observed == 0) if required else True
        gates.append(
            {
                "gate_id": "require_no_attitude_guardrail_events",
                "metric": "attitude_guardrail_event_count",
                "comparison": "zero_if_required",
                "threshold": required,
                "observed": observed,
                "status": "pass" if passed else "fail",
            }
        )
    statuses = {str(item["status"]) for item in gates}
    verdict = "not_ready" if "fail" in statuses else ("unknown" if "unknown" in statuses else "ready")
    packet = {
        "schema_id": "oel-maneuver-readiness-packet-v1",
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "status": "completed",
        "verdict": verdict,
        "object_id": object_key,
        "chief_id": chief_key,
        "thresholds": dict(thresholds),
        "metrics": metrics,
        "gates": gates,
        "evidence": {
            "review_store_path": str(workspace.logical_db_path),
            "review_store_sha256": str(review_identity["sha256"]),
            "summary_path": str(summary_path),
            "summary_sha256": sha256_file(summary_path),
            "final_relative_state_query": (
                "SELECT sample_index, time_s, range_km, range_rate_km_s FROM relative_state "
                "WHERE deputy_id = ? AND chief_id = ? ORDER BY sample_index DESC LIMIT 1"
            ),
            "actuator_realization_query": (
                "SELECT interval_start_ns, interval_end_ns, saturated, "
                "requested_force_x_n, requested_force_y_n, requested_force_z_n, "
                "realized_force_x_n, realized_force_y_n, realized_force_z_n FROM actuator_realization "
                "WHERE object_id = ? ORDER BY interval_start_ns, actuator_id"
            ),
        },
        "non_claims": [
            "This verdict applies only to the explicit thresholds and completed deterministic scenario.",
            "It is not an operational flight-readiness certification.",
            "Unknown evidence fails closed to an unknown verdict unless another gate already fails.",
        ],
    }
    if output_path is not None:
        target, logical_target = bind_output_path(
            output_path,
            logical_output_path,
            writer=writer,
        )
        packet["packet_path"] = str(logical_target)
        write_json_document(packet, target, writer=writer, overwrite=True)
    return packet


def _finite_or_none(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _finite_threshold(value: Any, name: str) -> float:
    result = _finite_or_none(value)
    if result is None or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative.")
    return result


__all__ = ["SUPPORTED_THRESHOLDS", "build_maneuver_readiness_packet"]
