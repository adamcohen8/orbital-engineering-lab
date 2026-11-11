from __future__ import annotations

from typing import Any

import numpy as np

from sim.scale.planner import ScalePlannerConfig
from sim.scale.store import (
    CampaignTrialResultRecord,
)


def _evaluate_quality_gates(
    cfg: ScalePlannerConfig,
    *,
    results: list[CampaignTrialResultRecord],
    status_counts: dict[str, int],
    metric_stats: dict[str, dict[str, Any]],
    batch_parity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    configured = dict(cfg.monte_carlo_outputs.get("quality_gates", {}) or {})
    gates = {
        "max_error_trials": 0,
        "max_blocked_trials": 0,
        **configured,
    }
    checks: list[dict[str, Any]] = []
    checks.append(
        _quality_check(
            name="max_error_trials",
            observed=float(status_counts.get("error", 0)),
            limit=gates.get("max_error_trials"),
            comparator="<=",
            units="trials",
        )
    )
    checks.append(
        _quality_check(
            name="max_blocked_trials",
            observed=float(status_counts.get("blocked", 0)),
            limit=gates.get("max_blocked_trials"),
            comparator="<=",
            units="trials",
        )
    )

    metric_gate_map = {
        "max_sgp4_fit_position_rms_m": ("sgp4_fit_position_rms_m", "max", "<=", "m"),
        "max_sgp4_fit_velocity_rms_mm_s": ("sgp4_fit_velocity_rms_mm_s", "max", "<=", "mm/s"),
        "max_final_separation_km": ("final_separation_km", "max", "<=", "km"),
        "min_final_separation_km": ("final_separation_km", "min", ">=", "km"),
    }
    for gate_name, (metric_name, statistic, comparator, units) in metric_gate_map.items():
        if gate_name not in gates:
            continue
        stats = metric_stats.get(metric_name, {})
        observed = _coerce_float(stats.get(statistic))
        checks.append(
            _quality_check(
                name=gate_name,
                observed=observed,
                limit=gates.get(gate_name),
                comparator=comparator,
                units=units,
                metric=metric_name,
                statistic=statistic,
            )
        )
    if batch_parity is not None:
        max_diff = _coerce_float(
            dict(batch_parity.get("max_absolute_differences", {}) or {}).get("separation_profile_km")
        )
        checks.append(
            _quality_check(
                name="max_sgp4_batch_parity_separation_diff_km",
                observed=max_diff,
                limit=gates.get("max_sgp4_batch_parity_separation_diff_km", 1.0e-9),
                comparator="<=",
                units="km",
                metric="batch_propagation_parity.separation_profile_km",
                statistic="max",
            )
        )
        checks.append(
            _quality_check(
                name="max_sgp4_batch_parity_error_count",
                observed=_coerce_float(batch_parity.get("error_count")),
                limit=gates.get("max_sgp4_batch_parity_error_count", 0),
                comparator="<=",
                units="errors",
                metric="batch_propagation_parity.error_count",
                statistic="count",
            )
        )

    failed = [check for check in checks if check["status"] == "failed"]
    warnings = [check for check in checks if check["status"] == "warning"]
    if failed:
        quality_status = "failed"
    elif warnings:
        quality_status = "warning"
    else:
        quality_status = "passed"
    return {
        "status": quality_status,
        "configured_gates": gates,
        "checks": checks,
        "trial_count": len(results),
    }


def _quality_check(
    *,
    name: str,
    observed: float | None,
    limit: Any,
    comparator: str,
    units: str,
    metric: str | None = None,
    statistic: str | None = None,
) -> dict[str, Any]:
    threshold = _coerce_float(limit)
    if threshold is None:
        status = "warning"
        passed = False
        message = "Gate threshold is not numeric."
    elif observed is None:
        status = "warning"
        passed = False
        message = "Observed value is unavailable."
    elif comparator == "<=":
        passed = observed <= threshold
        status = "passed" if passed else "failed"
        message = f"{observed:g} <= {threshold:g} {units}"
    elif comparator == ">=":
        passed = observed >= threshold
        status = "passed" if passed else "failed"
        message = f"{observed:g} >= {threshold:g} {units}"
    else:
        status = "warning"
        passed = False
        message = f"Unsupported comparator: {comparator}"
    check = {
        "name": name,
        "status": status,
        "passed": passed,
        "observed": observed,
        "threshold": threshold,
        "comparator": comparator,
        "units": units,
        "message": message,
    }
    if metric is not None:
        check["metric"] = metric
    if statistic is not None:
        check["statistic"] = statistic
    return check


def _coerce_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric):
        return None
    return numeric
