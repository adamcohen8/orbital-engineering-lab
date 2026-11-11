from __future__ import annotations

from typing import Any

import numpy as np

from sim.controller_lab.models import ControllerBenchMetric
from sim.dynamics.orbit.elements import rv_to_coe_eci
from sim.utils.frames import eci_relative_to_ric_rect


def _deep_get(root: dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = root
    for token in str(path).split("."):
        if not token:
            continue
        if not isinstance(cur, dict) or token not in cur:
            return default
        cur = cur[token]
    return cur


def _command_disposition(value: Any) -> str:
    raw = getattr(value, "value", value)
    return str(raw or "").rsplit(".", 1)[-1].lower()


def _packet_key(value: Any) -> tuple[str, str, int]:
    row = dict(value or {}) if isinstance(value, dict) else {}
    return (
        str(row.get("source_id", "") or ""),
        str(row.get("boot_id", "") or ""),
        int(row.get("sequence", -1) if row.get("sequence") is not None else -1),
    )


def _clock_ns(value: Any) -> int:
    row = dict(value or {}) if isinstance(value, dict) else {}
    return int(row.get("ticks", 0) or 0) * int(row.get("tick_period_ns", 1) or 1)


def _safe_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out


def _finite_values(values: Any) -> np.ndarray:
    arr = np.array(values if values is not None else [], dtype=float).reshape(-1)
    return arr[np.isfinite(arr)]


def _normalize_quaternion(q: np.ndarray) -> np.ndarray:
    arr = np.array(q, dtype=float).reshape(-1)
    if arr.size != 4:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    nrm = float(np.linalg.norm(arr))
    if nrm <= 0.0:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    return arr / nrm


def _relative_series(
    payload: dict[str, Any],
    object_id: str,
    reference_object_id: str,
) -> tuple[np.ndarray, np.ndarray] | None:
    truth_map = dict(payload.get("truth_by_object", {}) or {})
    truth = np.array(truth_map.get(object_id, []), dtype=float)
    ref = np.array(truth_map.get(reference_object_id, []), dtype=float)
    if truth.ndim != 2 or ref.ndim != 2 or truth.shape[0] == 0 or ref.shape[0] == 0:
        return None
    n = int(min(truth.shape[0], ref.shape[0]))
    if truth.shape[1] < 6 or ref.shape[1] < 6 or n <= 0:
        return None
    dr = truth[:n, :3] - ref[:n, :3]
    dv = truth[:n, 3:6] - ref[:n, 3:6]
    return dr, dv


def _relative_distance_series(payload: dict[str, Any], object_id: str, reference_object_id: str) -> np.ndarray:
    rel = _relative_series(payload, object_id, reference_object_id)
    if rel is None:
        return np.array([], dtype=float)
    dr, _ = rel
    return np.linalg.norm(dr, axis=1)


def _relative_speed_series(payload: dict[str, Any], object_id: str, reference_object_id: str) -> np.ndarray:
    rel = _relative_series(payload, object_id, reference_object_id)
    if rel is None:
        return np.array([], dtype=float)
    _, dv = rel
    return np.linalg.norm(dv, axis=1)


def _relative_ric_rect_series(payload: dict[str, Any], object_id: str, reference_object_id: str) -> np.ndarray:
    truth_map = dict(payload.get("truth_by_object", {}) or {})
    truth = np.array(truth_map.get(object_id, []), dtype=float)
    ref = np.array(truth_map.get(reference_object_id, []), dtype=float)
    if truth.ndim != 2 or ref.ndim != 2 or truth.shape[0] == 0 or ref.shape[0] == 0:
        return np.empty((0, 6), dtype=float)
    n = int(min(truth.shape[0], ref.shape[0]))
    if truth.shape[1] < 6 or ref.shape[1] < 6 or n <= 0:
        return np.empty((0, 6), dtype=float)
    out = np.full((n, 6), np.nan, dtype=float)
    for i in range(n):
        try:
            out[i] = eci_relative_to_ric_rect(truth[i, :6], ref[i, :6])
        except (TypeError, ValueError, FloatingPointError):
            continue
    return out


def _time_s(payload: dict[str, Any]) -> np.ndarray:
    return np.array(payload.get("time_s", []), dtype=float).reshape(-1)


def _sampled_dwell_time_s(mask: np.ndarray, time_s: np.ndarray) -> float:
    inside = np.array(mask, dtype=bool).reshape(-1)
    t = np.array(time_s, dtype=float).reshape(-1)
    n = min(inside.size, t.size)
    if n < 2:
        return 0.0
    dt = np.diff(t[:n])
    valid = np.isfinite(dt) & (dt > 0.0)
    if not np.any(valid):
        return 0.0
    return float(np.sum(dt[valid] * inside[: n - 1][valid]))


def _sampled_integral_norm_s(series: np.ndarray, time_s: np.ndarray) -> float:
    values = np.array(series, dtype=float)
    t = np.array(time_s, dtype=float).reshape(-1)
    if values.ndim != 2 or values.shape[0] < 2 or t.size < 2:
        return float("nan")
    n = min(values.shape[0], t.size)
    dt = np.diff(t[:n])
    norms = np.linalg.norm(values[: n - 1], axis=1)
    valid = np.isfinite(dt) & (dt > 0.0) & np.isfinite(norms)
    if not np.any(valid):
        return float("nan")
    return float(np.sum(norms[valid] * dt[valid]))


def _finite_tail(series: np.ndarray) -> np.ndarray:
    arr = np.array(series, dtype=float).reshape(-1)
    return arr[np.isfinite(arr)]


def _fsw_evidence(payload: dict[str, Any], object_id: str) -> dict[str, Any]:
    return dict(dict(payload.get("flight_software_evidence_by_object", {}) or {}).get(object_id, {}) or {})


def _diagnostic_fields(payload: dict[str, Any], object_id: str) -> list[dict[str, Any]]:
    fields: list[dict[str, Any]] = []
    invocations = {
        int(dict(item or {}).get("invocation_id", -1)): _safe_float(dict(item or {}).get("invocation_time_ns")) * 1.0e-9
        for item in list(_fsw_evidence(payload, object_id).get("invocations", []) or [])
    }
    for output_raw in list(_fsw_evidence(payload, object_id).get("outputs", []) or []):
        output = dict(output_raw or {})
        row: dict[str, Any] = {"invocation_id": output.get("invocation_id")}
        invocation_id = int(output.get("invocation_id", -1) or -1)
        if invocation_id in invocations:
            row["invocation_time_s"] = invocations[invocation_id]
        for telemetry_raw in list(output.get("telemetry", []) or []):
            telemetry = dict(telemetry_raw or {})
            for item_raw in list(telemetry.get("fields", []) or []):
                item = dict(item_raw or {})
                if item.get("name") is not None:
                    row[str(item["name"])] = item.get("value")
        fields.append(row)
    return fields


def _diagnostic_series(payload: dict[str, Any], object_id: str, field: str) -> list[Any]:
    return [row[field] for row in _diagnostic_fields(payload, object_id) if row.get(field) is not None]


def _numeric_diagnostic_series(payload: dict[str, Any], object_id: str, field: str) -> np.ndarray:
    return _finite_values([_safe_float(value) for value in _diagnostic_series(payload, object_id, field)])


def _first_sustained_time_s(time_s: np.ndarray, condition: np.ndarray, hold_s: float) -> float:
    t = np.asarray(time_s, dtype=float).reshape(-1)
    ok = np.asarray(condition, dtype=bool).reshape(-1)
    n = min(t.size, ok.size)
    if n == 0:
        return float("nan")
    for index in range(n):
        if not ok[index] or not np.isfinite(t[index]):
            continue
        if hold_s <= 0.0:
            return float(t[index])
        end = int(np.searchsorted(t[:n], float(t[index]) + hold_s, side="left"))
        if end < n and bool(np.all(ok[index : end + 1])):
            return float(t[index])
    return float("inf")


def _command_realization_statistics(payload: dict[str, Any], object_id: str) -> dict[str, float]:
    evidence = _fsw_evidence(payload, object_id)
    outputs = [dict(item or {}) for item in list(evidence.get("outputs", []) or [])]
    commands = [
        dict(command or {})
        for output in outputs
        for command in list(output.get("commands", []) or [])
    ]
    command_keys = {_packet_key(command.get("command_id")) for command in commands}
    receipts = [dict(item or {}) for item in list(evidence.get("receipts", []) or [])]
    accepted = {
        _packet_key(item.get("command_id"))
        for item in receipts
        if _command_disposition(item.get("disposition")) in {"accepted", "duplicate"}
    }
    realizations = [dict(item or {}) for item in list(evidence.get("realizations", []) or [])]
    realized = {
        _packet_key(dict(item or {}).get("source_command_id"))
        for item in realizations
        if dict(item or {}).get("source_command_id")
    }
    invocations = [dict(item or {}) for item in list(evidence.get("invocations", []) or [])]
    latest_evidence_ns = max(
        max((int(item.get("interval_end_ns", -1) or -1) for item in realizations), default=-1),
        max((int(item.get("invocation_time_ns", -1) or -1) for item in invocations), default=-1),
    )
    realizable_keys = {
        _packet_key(command.get("command_id"))
        for command in commands
        if _clock_ns(dict(command.get("validity", {}) or {}).get("not_before")) < latest_evidence_ns
    }
    return {
        "accepted_fraction": len(command_keys & accepted) / len(command_keys) if command_keys else 1.0,
        "realized_fraction": (
            len(realizable_keys & realized) / len(realizable_keys) if realizable_keys else 1.0
        ),
    }


def _fsw_audit_completeness_fraction(payload: dict[str, Any], object_id: str) -> float:
    evidence = _fsw_evidence(payload, object_id)
    invocations = {
        int(dict(row or {}).get("invocation_id", -1)): dict(row or {})
        for row in list(evidence.get("invocations", []) or [])
    }
    outputs = {
        int(dict(row or {}).get("invocation_id", -1)): dict(row or {})
        for row in list(evidence.get("outputs", []) or [])
    }
    ids = set(invocations) | set(outputs)
    if not ids:
        return 0.0
    complete = 0
    for invocation_id in ids:
        invocation = invocations.get(invocation_id, {})
        output = outputs.get(invocation_id, {})
        if (
            invocation_id >= 0
            and bool(invocation.get("stack_id"))
            and bool(invocation.get("stack_version"))
            # Stack-level controller-bench variants intentionally have no
            # exact profile identity. The audit contract requires the field
            # to be present and typed, not a forged non-empty profile label.
            and "profile_id" in invocation
            and (
                invocation.get("profile_id") is None
                or isinstance(invocation.get("profile_id"), str)
            )
            and isinstance(invocation.get("input_packet_ids"), list)
            and isinstance(invocation.get("command_ids"), list)
            and isinstance(output.get("commands"), list)
            and isinstance(output.get("telemetry"), list)
        ):
            complete += 1
    return complete / len(ids)


def _thrust_duty_cycle(payload: dict[str, Any], object_id: str) -> float:
    realizations = [dict(item or {}) for item in list(_fsw_evidence(payload, object_id).get("realizations", []) or [])]
    active_ns = 0
    total_ns = 0
    for row in realizations:
        start = int(row.get("interval_start_ns", 0) or 0)
        end = int(row.get("interval_end_ns", start) or start)
        duration = max(0, end - start)
        total_ns += duration
        force = np.asarray(row.get("realized_force_n", (0.0, 0.0, 0.0)), dtype=float)
        if force.shape == (3,) and np.all(np.isfinite(force)) and float(np.linalg.norm(force)) > 0.0:
            active_ns += duration
    return active_ns / total_ns if total_ns > 0 else 0.0


def _thrust_activity_runs_s(payload: dict[str, Any], object_id: str) -> list[tuple[bool, float]]:
    rows = sorted(
        (dict(item or {}) for item in list(_fsw_evidence(payload, object_id).get("realizations", []) or [])),
        key=lambda row: (int(row.get("interval_start_ns", 0) or 0), int(row.get("interval_end_ns", 0) or 0)),
    )
    runs: list[tuple[bool, float]] = []
    for row in rows:
        start = int(row.get("interval_start_ns", 0) or 0)
        end = int(row.get("interval_end_ns", start) or start)
        duration_s = max(0, end - start) * 1.0e-9
        if duration_s <= 0.0:
            continue
        force = np.asarray(row.get("realized_force_n", (0.0, 0.0, 0.0)), dtype=float)
        active = bool(
            force.shape == (3,) and np.all(np.isfinite(force)) and float(np.linalg.norm(force)) > 0.0
        )
        if runs and runs[-1][0] == active:
            runs[-1] = (active, runs[-1][1] + duration_s)
        else:
            runs.append((active, duration_s))
    return runs


def _actuator_realization_statistics(payload: dict[str, Any], object_id: str) -> dict[str, float]:
    rows = [dict(item or {}) for item in list(_fsw_evidence(payload, object_id).get("realizations", []) or [])]
    force_residuals: list[float] = []
    torque_residuals: list[float] = []
    for row in rows:
        requested_force = np.asarray(row.get("requested_force_n", (0.0, 0.0, 0.0)), dtype=float)
        realized_force = np.asarray(row.get("realized_force_n", (0.0, 0.0, 0.0)), dtype=float)
        requested_torque = np.asarray(row.get("requested_torque_n_m", (0.0, 0.0, 0.0)), dtype=float)
        realized_torque = np.asarray(row.get("realized_torque_n_m", (0.0, 0.0, 0.0)), dtype=float)
        if requested_force.shape == realized_force.shape == (3,):
            force_residuals.append(float(np.linalg.norm(requested_force - realized_force)))
        if requested_torque.shape == realized_torque.shape == (3,):
            torque_residuals.append(float(np.linalg.norm(requested_torque - realized_torque)))
    return {
        "count": float(len(rows)),
        "saturation_fraction": (sum(bool(row.get("saturated", False)) for row in rows) / len(rows)) if rows else 0.0,
        "max_force_residual_n": max(force_residuals, default=0.0),
        "max_torque_residual_n_m": max(torque_residuals, default=0.0),
    }


def _navigation_error_series(
    payload: dict[str, Any], object_id: str, reference_object_id: str
) -> tuple[np.ndarray, np.ndarray]:
    truth_map = dict(payload.get("truth_by_object", {}) or {})
    target_id = reference_object_id or object_id
    truth = np.asarray(truth_map.get(target_id, []), dtype=float)
    if reference_object_id:
        estimate = np.asarray(
            dict(dict(payload.get("knowledge_by_observer", {}) or {}).get(object_id, {}) or {}).get(target_id, []),
            dtype=float,
        )
    else:
        estimate = np.asarray(dict(payload.get("belief_by_object", {}) or {}).get(object_id, []), dtype=float)
    if truth.ndim != 2 or estimate.ndim != 2 or truth.shape[1] < 6 or estimate.shape[1] < 6:
        return np.array([], dtype=float), np.array([], dtype=float)
    n = min(len(truth), len(estimate))
    position_error_m = np.linalg.norm(estimate[:n, :3] - truth[:n, :3], axis=1) * 1.0e3
    velocity_error_m_s = np.linalg.norm(estimate[:n, 3:6] - truth[:n, 3:6], axis=1) * 1.0e3
    return position_error_m, velocity_error_m_s


def _quat_error_deg_series(truth_hist: np.ndarray, desired_q: np.ndarray) -> np.ndarray:
    q_des = _normalize_quaternion(desired_q)
    q_hist = np.array(truth_hist[:, 6:10], dtype=float)
    out = np.full(q_hist.shape[0], np.nan)
    for i in range(q_hist.shape[0]):
        q_cur = _normalize_quaternion(q_hist[i])
        dot = float(np.clip(np.dot(q_des, q_cur), -1.0, 1.0))
        out[i] = float(np.degrees(2.0 * np.arccos(abs(dot))))
    return out


def _attitude_settle_masks(
    payload: dict[str, Any],
    metric: ControllerBenchMetric,
) -> tuple[np.ndarray, np.ndarray]:
    if metric.desired_quat_bn is None or metric.settle_error_deg is None or metric.settle_rate_norm_rad_s is None:
        return np.array([], dtype=float), np.array([], dtype=bool)
    truth_map = dict(payload.get("truth_by_object", {}) or {})
    truth = np.array(truth_map.get(metric.object_id, []), dtype=float)
    if truth.ndim != 2 or truth.shape[1] < 13 or truth.shape[0] == 0:
        return np.array([], dtype=float), np.array([], dtype=bool)
    err = _quat_error_deg_series(truth, np.array(metric.desired_quat_bn, dtype=float))
    rates = np.linalg.norm(truth[:, 10:13], axis=1)
    t_s = _time_s(payload)
    n = min(err.size, rates.size, t_s.size)
    if n <= 0:
        return np.array([], dtype=float), np.array([], dtype=bool)
    ok = (
        np.isfinite(t_s[:n])
        & np.isfinite(err[:n])
        & np.isfinite(rates[:n])
        & (err[:n] <= float(metric.settle_error_deg))
        & (rates[:n] <= float(metric.settle_rate_norm_rad_s))
    )
    return t_s[:n], ok


def _attitude_settle_time_s(payload: dict[str, Any], metric: ControllerBenchMetric) -> float:
    t_s, ok = _attitude_settle_masks(payload, metric)
    if t_s.size == 0:
        return float("nan")
    hold_s = max(0.0, float(metric.settle_hold_s or 0.0))
    for i in range(t_s.size):
        if not bool(ok[i]):
            continue
        if hold_s <= 0.0:
            return float(t_s[i])
        end_idx = int(np.searchsorted(t_s, float(t_s[i]) + hold_s, side="left"))
        if end_idx >= t_s.size:
            continue
        if bool(np.all(ok[i : end_idx + 1])):
            return float(t_s[i])
    return float("inf")


def _settled_hold_duration_s(payload: dict[str, Any], metric: ControllerBenchMetric) -> float:
    t_s, ok = _attitude_settle_masks(payload, metric)
    n = min(t_s.size, ok.size)
    if n < 2:
        return 0.0
    dt = np.diff(t_s[:n])
    best = 0.0
    current = 0.0
    for idx, step_s in enumerate(dt):
        if not np.isfinite(step_s) or step_s <= 0.0:
            current = 0.0
            continue
        if bool(ok[idx]) and bool(ok[idx + 1]):
            current += float(step_s)
            best = max(best, current)
        else:
            current = 0.0
    return float(best)


def evaluate_metric(metric: ControllerBenchMetric, payload: dict[str, Any]) -> Any:
    if metric.source_path:
        return _deep_get(payload, metric.source_path)

    kind = str(metric.kind or "").strip().lower()
    object_id = str(metric.object_id or "").strip()
    ref_id = str(metric.reference_object_id or "").strip()

    truth_map = dict(payload.get("truth_by_object", {}) or {})
    debug_map = dict(payload.get("controller_debug_by_object", {}) or {})
    summary = dict(payload.get("summary", {}) or {})

    rocket_summary_map = {
        "rocket_insertion_achieved": "rocket_insertion_achieved",
        "rocket_insertion_time_s": "rocket_insertion_time_s",
        "rocket_final_altitude_km": "rocket_metrics_summary.final_altitude_km",
        "rocket_final_speed_km_s": "rocket_metrics_summary.final_speed_km_s",
        "rocket_final_apoapsis_alt_km": "rocket_metrics_summary.final_apoapsis_alt_km",
        "rocket_final_periapsis_alt_km": "rocket_metrics_summary.final_periapsis_alt_km",
        "rocket_final_eccentricity": "rocket_metrics_summary.final_eccentricity",
        "rocket_final_propellant_remaining_fraction": ("rocket_metrics_summary.final_propellant_remaining_fraction"),
        "rocket_max_dynamic_pressure_pa": "rocket_metrics_summary.max_dynamic_pressure_pa",
        "rocket_max_mach": "rocket_metrics_summary.max_mach",
        "rocket_max_abs_alpha_deg": "rocket_metrics_summary.max_abs_alpha_deg",
        "rocket_max_tvc_gimbal_deg": "rocket_metrics_summary.max_tvc_gimbal_deg",
        "rocket_max_aero_force_n": "rocket_metrics_summary.max_aero_force_n",
        "rocket_max_aero_moment_nm": "rocket_metrics_summary.max_aero_moment_nm",
    }
    if kind in rocket_summary_map:
        return _deep_get(summary, rocket_summary_map[kind])

    if kind in {"rocket_total_dv_m_s", "rocket_burn_samples"}:
        thrust_stats = dict(summary.get("thrust_stats", {}) or {})
        key = "total_dv_m_s" if kind == "rocket_total_dv_m_s" else "burn_samples"
        if object_id:
            return _deep_get({"thrust_stats": thrust_stats}, f"thrust_stats.{object_id}.{key}")
        for stats in thrust_stats.values():
            value = dict(stats or {}).get(key)
            if value is not None:
                return value
        return None

    if kind in {
        "rocket_max_abs_beta_deg",
        "rocket_min_propellant_remaining_fraction",
        "rocket_max_thrust_to_weight",
    }:
        rocket_metrics = dict(payload.get("rocket_metrics", {}) or {})
        source_key = {
            "rocket_max_abs_beta_deg": "beta_deg",
            "rocket_min_propellant_remaining_fraction": "propellant_remaining_fraction",
            "rocket_max_thrust_to_weight": "thrust_to_weight",
        }[kind]
        finite = _finite_values(rocket_metrics.get(source_key))
        if finite.size == 0:
            return float("nan")
        if kind == "rocket_min_propellant_remaining_fraction":
            return float(np.min(finite))
        if kind == "rocket_max_abs_beta_deg":
            return float(np.max(np.abs(finite)))
        return float(np.max(finite))

    if kind in {"total_dv_m_s", "burn_samples", "max_accel_km_s2"}:
        thrust_stats = dict(summary.get("thrust_stats", {}) or {})
        return _deep_get({"thrust_stats": thrust_stats}, f"thrust_stats.{object_id}.{kind}")

    if kind == "scheduled_burn_delta_v_error_m_s":
        if metric.target_value is None:
            return float("nan")
        thrust_stats = dict(summary.get("thrust_stats", {}) or {})
        delivered = _deep_get({"thrust_stats": thrust_stats}, f"thrust_stats.{object_id}.total_dv_m_s")
        return abs(_safe_float(delivered) - float(metric.target_value))

    if kind in {"final_semimajor_axis_error_km", "final_eccentricity_error"}:
        truth = np.asarray(truth_map.get(object_id, []), dtype=float)
        if truth.ndim != 2 or truth.shape[0] == 0 or truth.shape[1] < 6 or metric.target_value is None:
            return float("nan")
        try:
            elements = rv_to_coe_eci(truth[-1, :3], truth[-1, 3:6])
        except (ValueError, FloatingPointError):
            return float("nan")
        actual = elements.a_km if kind == "final_semimajor_axis_error_km" else elements.ecc
        return abs(float(actual) - float(metric.target_value))

    if kind in {
        "max_semimajor_axis_error_km",
        "rms_semimajor_axis_error_km",
        "max_eccentricity_error",
        "rms_eccentricity_error",
    }:
        truth = np.asarray(truth_map.get(object_id, []), dtype=float)
        if truth.ndim != 2 or truth.shape[0] == 0 or truth.shape[1] < 6 or metric.target_value is None:
            return float("nan")
        values: list[float] = []
        use_a = "semimajor_axis" in kind
        for row in truth:
            try:
                elements = rv_to_coe_eci(row[:3], row[3:6])
            except (ValueError, FloatingPointError):
                continue
            actual = elements.a_km if use_a else elements.ecc
            values.append(abs(float(actual) - float(metric.target_value)))
        finite = _finite_values(values)
        if finite.size == 0:
            return float("nan")
        return float(np.max(finite)) if kind.startswith("max_") else float(np.sqrt(np.mean(finite**2)))

    if kind in {"navigation_rms_position_error_m", "navigation_rms_velocity_error_m_s"}:
        position_error, velocity_error = _navigation_error_series(payload, object_id, ref_id)
        series = position_error if kind.endswith("position_error_m") else velocity_error
        finite = series[np.isfinite(series)]
        return float(np.sqrt(np.mean(finite**2))) if finite.size else float("nan")

    if kind in {
        "actuator_saturation_fraction",
        "max_actuator_force_residual_n",
        "max_actuator_torque_residual_n_m",
    }:
        stats = _actuator_realization_statistics(payload, object_id)
        return stats[
            {
                "actuator_saturation_fraction": "saturation_fraction",
                "max_actuator_force_residual_n": "max_force_residual_n",
                "max_actuator_torque_residual_n_m": "max_torque_residual_n_m",
            }[kind]
        ]

    if kind in {"fsw_invocation_count", "fsw_command_count", "fsw_boundary_error_count"}:
        evidence = _fsw_evidence(payload, object_id)
        invocations = list(evidence.get("invocations", []) or [])
        outputs = [dict(item or {}) for item in list(evidence.get("outputs", []) or [])]
        if kind == "fsw_invocation_count":
            return len(invocations)
        if kind == "fsw_command_count":
            return sum(len(list(item.get("commands", []) or [])) for item in outputs)
        errors = abs(len(invocations) - len(outputs))
        invocation_ids = {int(dict(item or {}).get("invocation_id", -1)) for item in invocations}
        output_ids = {int(item.get("invocation_id", -1)) for item in outputs}
        errors += len(invocation_ids.symmetric_difference(output_ids))
        errors += sum(
            1
            for item in list(evidence.get("input_events", []) or [])
            if _deep_get(dict(item or {}), "delivery_time.ticks", 0)
            < _deep_get(dict(item or {}), "source_time.ticks", 0)
        )
        errors += sum(
            1
            for item in list(evidence.get("receipts", []) or [])
            if _command_disposition(dict(item or {}).get("disposition")) not in {"accepted", "duplicate"}
        )
        commands = [dict(command or {}) for output in outputs for command in list(output.get("commands", []) or [])]
        command_keys = {_packet_key(item.get("command_id")) for item in commands}
        receipt_keys = {
            _packet_key(dict(item or {}).get("command_id")) for item in list(evidence.get("receipts", []) or [])
        }
        accepted_receipt_keys = {
            _packet_key(dict(item or {}).get("command_id"))
            for item in list(evidence.get("receipts", []) or [])
            if _command_disposition(dict(item or {}).get("disposition")) in {"accepted", "duplicate"}
        }
        realizations = [dict(item or {}) for item in list(evidence.get("realizations", []) or [])]
        latest_realization_end_ns = max(
            (int(item.get("interval_end_ns", -1) or -1) for item in realizations),
            default=-1,
        )
        latest_evidence_ns = max(
            latest_realization_end_ns,
            max(
                (int(dict(item or {}).get("invocation_time_ns", -1) or -1) for item in invocations),
                default=-1,
            ),
        )
        realizable_keys = {
            _packet_key(item.get("command_id"))
            for item in commands
            if _clock_ns(dict(item.get("validity", {}) or {}).get("not_before")) < latest_evidence_ns
        }
        realized_keys = {
            _packet_key(item.get("source_command_id")) for item in realizations if item.get("source_command_id")
        }
        errors += len(command_keys - receipt_keys)
        errors += len(command_keys - accepted_receipt_keys)
        errors += len(realizable_keys - realized_keys)
        return errors

    if kind in {"accepted_command_fraction", "physical_realization_fraction"}:
        stats = _command_realization_statistics(payload, object_id)
        key = "accepted_fraction" if kind == "accepted_command_fraction" else "realized_fraction"
        return stats[key]

    if kind == "fsw_audit_completeness_fraction":
        return _fsw_audit_completeness_fraction(payload, object_id)

    if kind in {"fsw_profile_identity_consistent", "fsw_stack_identity_consistent"}:
        rows = [dict(item or {}) for item in list(_fsw_evidence(payload, object_id).get("invocations", []) or [])]
        key = "profile_id" if kind == "fsw_profile_identity_consistent" else "stack_id"
        identities = [str(row.get(key, "") or "") for row in rows]
        if not identities or len(set(identities)) != 1:
            return False
        context_key = "profile_id" if key == "profile_id" else "stack_id"
        expected = str(dict(payload.get("controller_bench_context", {}) or {}).get(context_key, "") or "")
        return identities[0] == expected if expected else not identities[0]

    if kind in {"fsw_missed_task_release_count", "fsw_missed_sensor_release_count"}:
        rows = [dict(item or {}) for item in list(_fsw_evidence(payload, object_id).get("invocations", []) or [])]
        key = "missed_task_releases" if kind == "fsw_missed_task_release_count" else "missed_sensor_releases"
        return max((int(row.get(key, 0) or 0) for row in rows), default=0)

    if kind == "thrust_duty_cycle":
        return _thrust_duty_cycle(payload, object_id)

    if kind in {"thrust_transition_count", "minimum_thrust_dwell_s"}:
        runs = _thrust_activity_runs_s(payload, object_id)
        if kind == "thrust_transition_count":
            return max(len(runs) - 1, 0)
        return min((duration_s for _active, duration_s in runs), default=float("nan"))

    if kind == "max_task_cadence_jitter_s":
        rows = [dict(item or {}) for item in list(_fsw_evidence(payload, object_id).get("invocations", []) or [])]
        times = np.asarray([_safe_float(item.get("invocation_time_ns")) * 1.0e-9 for item in rows], dtype=float)
        times = times[np.isfinite(times)]
        configured = _safe_float(dict(payload.get("controller_bench_context", {}) or {}).get("task_period_s"))
        if times.size < 2 or not np.isfinite(configured) or configured <= 0.0:
            return 0.0
        duration = float(np.max(times))
        expected = np.arange(int(np.floor(duration / configured + 1.0e-10)) + 1, dtype=float) * configured
        errors = [float(np.min(np.abs(times - release))) for release in expected]
        return max(errors, default=0.0)

    if kind in {"executive_preemption_seen", "executive_recovery_seen", "executive_primary_resume_seen"}:
        fields = _diagnostic_fields(payload, object_id)
        phases = [str(item.get("executive_phase", "") or "") for item in fields]
        recovery_indices = [idx for idx, phase in enumerate(phases) if phase == "recovery"]
        if kind == "executive_recovery_seen":
            return bool(recovery_indices)
        if kind == "executive_preemption_seen":
            return bool(recovery_indices) or any(str(item.get("goal_state", "")) == "preempted" for item in fields)
        return bool(recovery_indices) and any(phase == "primary" for phase in phases[max(recovery_indices) + 1 :])

    if kind == "recovery_duration_s":
        fields = _diagnostic_fields(payload, object_id)
        time_values = np.asarray([_safe_float(item.get("invocation_time_s")) for item in fields], dtype=float)
        in_recovery = np.asarray(
            [str(item.get("executive_phase", "") or "") == "recovery" for item in fields],
            dtype=bool,
        )
        return _sampled_dwell_time_s(in_recovery, time_values)

    if kind in {"selected_mode_matches", "control_law_matches"}:
        expected = metric.expected_value
        key = "selected_mode" if kind == "selected_mode_matches" else "control_law"
        actual = [
            str(item.get(key, "") or "") for item in _diagnostic_fields(payload, object_id) if item.get(key) is not None
        ]
        return bool(actual) and all(value == expected for value in actual)

    if kind in {"mean_controller_runtime_ms", "max_controller_runtime_ms", "controller_trace_samples"}:
        debug_hist = list(debug_map.get(object_id, []) or [])
        runtimes = np.array([_safe_float(d.get("controller_runtime_ms")) for d in debug_hist], dtype=float)
        runtimes = runtimes[np.isfinite(runtimes)]
        if kind == "mean_controller_runtime_ms":
            return float(np.mean(runtimes)) if runtimes.size else float("nan")
        if kind == "max_controller_runtime_ms":
            return float(np.max(runtimes)) if runtimes.size else float("nan")
        return int(len(debug_hist))

    if kind in {"final_body_rate_norm_rad_s", "mean_body_rate_norm_rad_s", "max_body_rate_norm_rad_s"}:
        truth = np.array(truth_map.get(object_id, []), dtype=float)
        if truth.ndim != 2 or truth.shape[1] < 13 or truth.shape[0] == 0:
            return float("nan")
        rates = np.linalg.norm(truth[:, 10:13], axis=1)
        if kind == "final_body_rate_norm_rad_s":
            return float(rates[-1])
        if kind == "max_body_rate_norm_rad_s":
            return float(np.max(rates))
        return float(np.mean(rates))

    if kind == "detumble_time_s":
        truth = np.array(truth_map.get(object_id, []), dtype=float)
        if truth.ndim != 2 or truth.shape[1] < 13 or truth.shape[0] == 0 or metric.target_value is None:
            return float("nan")
        rates = np.linalg.norm(truth[:, 10:13], axis=1)
        t_s = _time_s(payload)
        n = min(rates.size, t_s.size)
        return _first_sustained_time_s(
            t_s[:n],
            np.isfinite(rates[:n]) & (rates[:n] <= float(metric.target_value)),
            max(0.0, float(metric.settle_hold_s or 0.0)),
        )

    if kind in {"max_pointing_error_deg", "rms_pointing_error_deg"}:
        radians = _numeric_diagnostic_series(payload, object_id, "pointing_error_rad")
        degrees = np.degrees(radians)
        if degrees.size == 0:
            return float("nan")
        if kind == "max_pointing_error_deg":
            return float(np.max(np.abs(degrees)))
        return float(np.sqrt(np.mean(degrees**2)))

    if kind == "pointing_compliance_fraction":
        values = _diagnostic_series(payload, object_id, "pointing_compliant")
        return sum(bool(value) for value in values) / len(values) if values else float("nan")

    if kind in {"max_wheel_momentum_fraction", "final_wheel_momentum_fraction"}:
        values = _numeric_diagnostic_series(payload, object_id, "wheel_momentum_fraction")
        if values.size == 0:
            return float("nan")
        return float(np.max(values)) if kind.startswith("max_") else float(values[-1])

    if kind in {"final_attitude_error_deg", "rms_attitude_error_deg", "max_attitude_error_deg"}:
        truth = np.array(truth_map.get(object_id, []), dtype=float)
        if truth.ndim != 2 or truth.shape[1] < 10 or truth.shape[0] == 0 or metric.desired_quat_bn is None:
            return float("nan")
        err = _quat_error_deg_series(truth, np.array(metric.desired_quat_bn, dtype=float))
        finite = err[np.isfinite(err)]
        if finite.size == 0:
            return float("nan")
        if kind == "final_attitude_error_deg":
            return float(finite[-1])
        if kind == "rms_attitude_error_deg":
            return float(np.sqrt(np.mean(finite**2)))
        return float(np.max(finite))

    if kind in {
        "final_relative_distance_km",
        "min_relative_distance_km",
        "max_relative_distance_km",
        "mean_relative_distance_km",
        "rms_relative_distance_km",
        "closest_approach_km",
    }:
        dist = _finite_tail(_relative_distance_series(payload, object_id, ref_id))
        if dist.size == 0:
            return float("nan")
        if kind == "final_relative_distance_km":
            return float(dist[-1])
        if kind in {"min_relative_distance_km", "closest_approach_km"}:
            return float(np.min(dist))
        if kind == "max_relative_distance_km":
            return float(np.max(dist))
        if kind == "mean_relative_distance_km":
            return float(np.mean(dist))
        return float(np.sqrt(np.mean(dist**2)))

    if kind in {
        "final_relative_speed_km_s",
        "max_relative_speed_km_s",
        "mean_relative_speed_km_s",
        "rms_relative_speed_km_s",
    }:
        speed = _finite_tail(_relative_speed_series(payload, object_id, ref_id))
        if speed.size == 0:
            return float("nan")
        if kind == "final_relative_speed_km_s":
            return float(speed[-1])
        if kind == "max_relative_speed_km_s":
            return float(np.max(speed))
        if kind == "mean_relative_speed_km_s":
            return float(np.mean(speed))
        return float(np.sqrt(np.mean(speed**2)))

    if kind in {"final_relative_position_error_m", "final_relative_velocity_error_m_s"}:
        if metric.target_relative_state_ric_m is None:
            return float("nan")
        ric = _relative_ric_rect_series(payload, object_id, ref_id)
        finite_rows = ric[np.all(np.isfinite(ric), axis=1)] if ric.size else np.empty((0, 6))
        if finite_rows.size == 0:
            return float("nan")
        target = np.asarray(metric.target_relative_state_ric_m, dtype=float)
        error = finite_rows[-1] - target / 1.0e3
        portion = error[:3] if kind == "final_relative_position_error_m" else error[3:]
        return float(np.linalg.norm(portion) * 1.0e3)

    if kind in {"max_position_error_m", "max_velocity_error_m_s", "max_compliant_dwell_s"}:
        field = {
            "max_position_error_m": "position_error_m",
            "max_velocity_error_m_s": "velocity_error_m_s",
            "max_compliant_dwell_s": "maintenance_compliant_s",
        }[kind]
        values = _numeric_diagnostic_series(payload, object_id, field)
        return float(np.max(values)) if values.size else float("nan")

    if kind in {"waypoint_sequence_completed", "waypoint_sequence_completed_count"}:
        if (
            not metric.waypoints_ric_m
            or metric.waypoint_tolerance_m is None
            or metric.waypoint_velocity_tolerance_m_s is None
        ):
            return False if kind == "waypoint_sequence_completed" else 0
        ric = _relative_ric_rect_series(payload, object_id, ref_id)
        finite_rows = ric[np.all(np.isfinite(ric), axis=1)] if ric.size else np.empty((0, 6))
        cursor = 0
        tolerance = float(metric.waypoint_tolerance_m)
        velocity_tolerance = float(metric.waypoint_velocity_tolerance_m_s)
        for row in finite_rows:
            if cursor >= len(metric.waypoints_ric_m):
                break
            target = np.asarray(metric.waypoints_ric_m[cursor], dtype=float) / 1.0e3
            position_error_m = float(np.linalg.norm(row[:3] - target[:3]) * 1.0e3)
            velocity_error_m_s = float(np.linalg.norm(row[3:] - target[3:]) * 1.0e3)
            if position_error_m <= tolerance and velocity_error_m_s <= velocity_tolerance:
                cursor += 1
        return cursor == len(metric.waypoints_ric_m) if kind == "waypoint_sequence_completed" else cursor

    if kind in {
        "final_relative_radial_error_km",
        "final_relative_intrack_error_km",
        "final_relative_crosstrack_error_km",
        "final_relative_radial_rate_km_s",
        "final_relative_intrack_rate_km_s",
        "final_relative_crosstrack_rate_km_s",
    }:
        ric = _relative_ric_rect_series(payload, object_id, ref_id)
        if ric.size == 0:
            return float("nan")
        finite_rows = ric[np.all(np.isfinite(ric), axis=1)]
        if finite_rows.size == 0:
            return float("nan")
        final = finite_rows[-1]
        idx_by_kind = {
            "final_relative_radial_error_km": 0,
            "final_relative_intrack_error_km": 1,
            "final_relative_crosstrack_error_km": 2,
            "final_relative_radial_rate_km_s": 3,
            "final_relative_intrack_rate_km_s": 4,
            "final_relative_crosstrack_rate_km_s": 5,
        }
        return float(abs(final[idx_by_kind[kind]]))

    if kind == "time_inside_keepout_s":
        keepout_radius_km = metric.keepout_radius_km
        if keepout_radius_km is None:
            return float("nan")
        dist = _relative_distance_series(payload, object_id, ref_id)
        if not np.any(np.isfinite(dist)):
            return float("nan")
        t_s = _time_s(payload)
        n = min(dist.size, t_s.size if t_s.size else dist.size)
        if n <= 0:
            return float("nan")
        if t_s.size:
            finite = np.isfinite(dist[:n]) & np.isfinite(t_s[:n])
            inside = finite & (dist[:n] < float(keepout_radius_km))
            return _sampled_dwell_time_s(inside, t_s[:n])
        return 0.0

    if kind == "keepout_exit_time_s":
        if metric.keepout_radius_km is None:
            return float("nan")
        dist = _relative_distance_series(payload, object_id, ref_id)
        t_s = _time_s(payload)
        n = min(dist.size, t_s.size)
        if n == 0 or not np.isfinite(dist[0]) or not np.isfinite(t_s[0]):
            return float("nan")
        if dist[0] >= float(metric.keepout_radius_km):
            return 0.0
        exits = np.where(np.isfinite(dist[:n]) & np.isfinite(t_s[:n]) & (dist[:n] >= float(metric.keepout_radius_km)))[
            0
        ]
        return float(t_s[int(exits[0])] - t_s[0]) if exits.size else float("inf")

    if kind == "time_to_rendezvous_s":
        radius_km = 0.01 if metric.keepout_radius_km is None else float(metric.keepout_radius_km)
        dist = _relative_distance_series(payload, object_id, ref_id)
        if not np.any(np.isfinite(dist)):
            return float("nan")
        t_s = _time_s(payload)
        n = min(dist.size, t_s.size if t_s.size else dist.size)
        if n <= 0:
            return float("nan")
        if t_s.size:
            valid = np.isfinite(dist[:n]) & np.isfinite(t_s[:n])
        else:
            valid = np.isfinite(dist[:n])
        hits = np.where(valid & (dist[:n] <= radius_km))[0]
        if hits.size == 0:
            return float("inf")
        idx = int(hits[0])
        return float(t_s[idx]) if t_s.size > idx else float(idx)

    if kind == "fuel_used_kg":
        truth = np.array(truth_map.get(object_id, []), dtype=float)
        if truth.ndim != 2 or truth.shape[1] < 14 or truth.shape[0] == 0:
            return float("nan")
        mass = truth[:, 13]
        finite = _finite_tail(mass)
        if finite.size == 0:
            return float("nan")
        return float(max(0.0, finite[0] - finite[-1]))

    if kind == "settle_time_s":
        return _attitude_settle_time_s(payload, metric)

    if kind == "settled_hold_duration_s":
        return _settled_hold_duration_s(payload, metric)

    if kind == "total_torque_norm_n_m_s":
        torque_map = dict(payload.get("applied_torque_by_object", {}) or {})
        torque = np.array(torque_map.get(object_id, []), dtype=float)
        if torque.ndim != 2 or torque.shape[1] < 3 or torque.shape[0] == 0:
            return float("nan")
        return _sampled_integral_norm_s(torque[:, :3], _time_s(payload))

    if kind in {
        "diagnostic_final",
        "diagnostic_min",
        "diagnostic_max",
        "diagnostic_mean",
        "diagnostic_true_fraction",
        "diagnostic_transition_count",
    }:
        field = str(metric.diagnostic_field or "").strip()
        if not field:
            return None
        values = _diagnostic_series(payload, object_id, field)
        if kind == "diagnostic_true_fraction":
            return sum(bool(value) for value in values) / len(values) if values else float("nan")
        if kind == "diagnostic_transition_count":
            return sum(values[index] != values[index - 1] for index in range(1, len(values)))
        numeric = _finite_values([_safe_float(value) for value in values])
        if numeric.size == 0:
            return float("nan")
        if kind == "diagnostic_final":
            return float(numeric[-1])
        if kind == "diagnostic_min":
            return float(np.min(numeric))
        if kind == "diagnostic_max":
            return float(np.max(numeric))
        return float(np.mean(numeric))

    return None
