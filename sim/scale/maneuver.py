from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from sim.scale.store import ManeuverFlagRecord, OdSolutionSummary, load_recent_od_solutions, write_maneuver_flags

DEFAULT_THRESHOLDS = {
    "mean_motion_delta_rev_per_day": 0.005,
    "eccentricity_delta": 0.001,
    "inclination_delta_deg": 0.05,
}


def run_maneuver_detection(
    store_path: str | Path,
    *,
    object_ids: tuple[str, ...] = (),
    thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    resolved_thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    recent_by_object = load_recent_od_solutions(store_path, object_ids=object_ids, limit_per_object=2)
    flags = build_maneuver_flags(recent_by_object, thresholds=resolved_thresholds)
    write_summary = write_maneuver_flags(store_path, flags=flags)
    return {
        "schema_version": 1,
        "status": "completed",
        "store_path": str(Path(store_path).expanduser().resolve()),
        "objects_seen": len(recent_by_object),
        "objects_with_comparable_od": sum(1 for items in recent_by_object.values() if len(items) >= 2),
        "thresholds": resolved_thresholds,
        "flags_seen": write_summary["flags_seen"],
        "flags_inserted": write_summary["flags_inserted"],
        "invalidations_inserted": write_summary["invalidations_inserted"],
        "non_claims": [
            "Maneuver detection V0 is advisory change detection over successive OD/TLE-derived products.",
            "Flags are recompute triggers, not authoritative maneuver attribution.",
            "No covariance, force-model, or operator intent evidence is inferred.",
        ],
    }


def build_maneuver_flags(
    recent_by_object: dict[str, list[OdSolutionSummary]],
    *,
    thresholds: dict[str, float],
) -> list[ManeuverFlagRecord]:
    flags: list[ManeuverFlagRecord] = []
    for object_id, items in sorted(recent_by_object.items()):
        if len(items) < 2:
            continue
        latest, previous = items[0], items[1]
        metrics = _change_metrics(previous=previous, latest=latest)
        exceeded = {
            name: value
            for name, value in metrics.items()
            if name in thresholds and abs(float(value)) >= float(thresholds[name])
        }
        if not exceeded:
            continue
        confidence = min(1.0, max(abs(float(value)) / max(float(thresholds[name]), 1.0e-12) for name, value in exceeded.items()))
        summary = {
            "previous_od_solution_id": previous.od_solution_id,
            "latest_od_solution_id": latest.od_solution_id,
            "previous_epoch_jd_utc": previous.epoch_jd_utc,
            "latest_epoch_jd_utc": latest.epoch_jd_utc,
            "metrics": metrics,
            "thresholds": thresholds,
            "exceeded_thresholds": exceeded,
        }
        digest = _digest(
            {
                "object_id": object_id,
                "previous_od_solution_id": previous.od_solution_id,
                "latest_od_solution_id": latest.od_solution_id,
                "method": "tle_mean_element_delta_v0",
                "exceeded_thresholds": exceeded,
            }
        )
        flags.append(
            ManeuverFlagRecord(
                maneuver_flag_id=f"maneuver:{digest}",
                object_id=object_id,
                source_od_solution_id=latest.od_solution_id,
                start_jd_utc=min(previous.epoch_jd_utc, latest.epoch_jd_utc),
                end_jd_utc=max(previous.epoch_jd_utc, latest.epoch_jd_utc),
                method="tle_mean_element_delta_v0",
                confidence=float(confidence),
                status="advisory",
                summary=summary,
            )
        )
    return flags


def _change_metrics(*, previous: OdSolutionSummary, latest: OdSolutionSummary) -> dict[str, float]:
    prev = previous.mean_elements
    curr = latest.mean_elements
    return {
        "mean_motion_delta_rev_per_day": _delta(curr, prev, "mean_motion_rev_per_day"),
        "eccentricity_delta": _delta(curr, prev, "eccentricity"),
        "inclination_delta_deg": _delta(curr, prev, "inclination_deg"),
        "raan_delta_deg": _wrapped_angle_delta(
            _float(curr.get("raan_deg")),
            _float(prev.get("raan_deg")),
        ),
        "argp_delta_deg": _wrapped_angle_delta(
            _float(curr.get("argp_deg")),
            _float(prev.get("argp_deg")),
        ),
        "epoch_delta_days": float(latest.epoch_jd_utc - previous.epoch_jd_utc),
    }


def _delta(curr: dict[str, Any], prev: dict[str, Any], key: str) -> float:
    return _float(curr.get(key)) - _float(prev.get(key))


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _wrapped_angle_delta(current_deg: float, previous_deg: float) -> float:
    return ((current_deg - previous_deg + 180.0) % 360.0) - 180.0


def _digest(payload: dict[str, Any]) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
