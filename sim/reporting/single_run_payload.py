from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from sim.config import (
    SimulationScenarioConfig,
    default_pair_object_ids,
    default_reference_object_id,
    iter_object_sections,
)
from sim.dynamics.orbit.frames import frame_context_from_mapping
from sim.ground_stations import evaluate_ground_station_access, evaluate_ground_station_measurements


@dataclass(frozen=True)
class SingleRunPayloadContext:
    cfg: SimulationScenarioConfig
    object_ids: list[str]
    dt_s: float
    t_s: np.ndarray
    truth_hist: dict[str, np.ndarray]
    target_reference_orbit_truth: np.ndarray | None
    belief_hist: dict[str, np.ndarray]
    thrust_hist: dict[str, np.ndarray]
    torque_hist: dict[str, np.ndarray]
    desired_attitude_hist: dict[str, np.ndarray]
    knowledge_hist: dict[str, dict[str, np.ndarray]]
    knowledge_measurement_hist: dict[str, dict[str, np.ndarray]]
    bridge_hist: dict[str, list[dict[str, Any]]]
    controller_debug_hist: dict[str, list[dict[str, Any]]]
    command_decision_hist: dict[str, list[dict[str, Any]]]
    flight_software_evidence: dict[str, dict[str, Any]]
    rocket_throttle_cmd: np.ndarray
    rocket_metrics: dict[str, np.ndarray]
    reentry_metrics: dict[str, dict[str, np.ndarray]]
    thrust_stats: dict[str, dict[str, Any]]
    runtime_profile: dict[str, Any]
    object_initialization: dict[str, dict[str, Any]]
    object_propagation: dict[str, dict[str, Any]]
    attitude_guardrail_stats: dict[str, int]
    knowledge_detection_by_observer: dict[str, Any]
    knowledge_consistency_by_observer: dict[str, Any]
    terminated_early: bool
    termination_reason: str | None
    termination_time_s: float | None
    termination_object_id: str | None
    rocket_inserted: bool
    rocket_insertion_time_s: float | None
    planetary_environment: Any | None = None
    planetary_force_plans: Any | None = None


def _summarize_actuator_diagnostics(controller_debug_hist: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for oid, rows in dict(controller_debug_hist or {}).items():
        object_summary: dict[str, Any] = {
            "actuator_stack_samples": 0,
            "fault_stuck_off_samples": 0,
            "rcs_active_samples": 0,
            "magnetorquer_active_samples": 0,
            "cmg_active_samples": 0,
            "wheel_desaturation_samples": 0,
            "max_gimbal_angle_rad": 0.0,
            "max_commanded_rcs_thruster_force_n": 0.0,
            "max_electric_propulsion_thrust_n": 0.0,
            "rcs_allocation_saturated_samples": 0,
            "rcs_allocation_saturated_duration_s": 0.0,
            "max_rcs_force_residual_n": 0.0,
            "rms_rcs_force_residual_n": None,
            "max_rcs_torque_residual_nm": 0.0,
            "rms_rcs_torque_residual_nm": None,
            "min_rcs_thrust_margin_n": None,
            "max_attitude_error_deg": None,
            "propellant_consumed_kg": 0.0,
            "final_propellant_remaining_kg": None,
        }
        force_residuals: list[float] = []
        torque_residuals: list[float] = []
        attitude_errors: list[float] = []
        thrust_margins: list[float] = []
        final_propellant: float | None = None
        for row in list(rows or []):
            flags = dict(row.get("mode_flags", {}) or {})
            attitude_flags = dict(dict(row.get("command_attitude", {}) or {}).get("mode_flags", {}) or {})
            if bool(flags.get("actuator_stack_enabled", False)):
                object_summary["actuator_stack_samples"] += 1
            if bool(flags.get("actuator_fault_stuck_off", False)):
                object_summary["fault_stuck_off_samples"] += 1
            if flags.get("rcs_thruster_forces_n") is not None:
                forces = np.array(flags.get("rcs_thruster_forces_n", []), dtype=float).reshape(-1)
                finite = forces[np.isfinite(forces)]
                if finite.size and float(np.max(np.abs(finite))) > 1e-15:
                    object_summary["rcs_active_samples"] += 1
                    object_summary["max_commanded_rcs_thruster_force_n"] = max(
                        float(object_summary["max_commanded_rcs_thruster_force_n"]),
                        float(np.max(np.abs(finite))),
                    )
            if str(flags.get("magnetorquer_mode", "") or ""):
                object_summary["magnetorquer_active_samples"] += 1
            if flags.get("cmg_torque_body_nm") is not None:
                cmg = np.array(flags.get("cmg_torque_body_nm", []), dtype=float).reshape(-1)
                if cmg.size and float(np.linalg.norm(cmg)) > 1e-15:
                    object_summary["cmg_active_samples"] += 1
            if bool(flags.get("wheel_desaturation_active", False)):
                object_summary["wheel_desaturation_samples"] += 1
            if flags.get("gimbal_angle_rad") is not None:
                object_summary["max_gimbal_angle_rad"] = max(
                    float(object_summary["max_gimbal_angle_rad"]),
                    abs(float(flags.get("gimbal_angle_rad", 0.0))),
                )
            if flags.get("electric_propulsion_thrust_n") is not None:
                object_summary["max_electric_propulsion_thrust_n"] = max(
                    float(object_summary["max_electric_propulsion_thrust_n"]),
                    abs(float(flags.get("electric_propulsion_thrust_n", 0.0))),
                )
            force_residual = flags.get("rcs_force_residual_n", flags.get("rcs_force_error_n"))
            if force_residual is not None:
                vector = np.asarray(force_residual, dtype=float).reshape(-1)
                if vector.size and np.all(np.isfinite(vector)):
                    force_residuals.append(float(np.linalg.norm(vector)))
            torque_residual = flags.get("rcs_torque_residual_nm", flags.get("rcs_torque_error_nm"))
            if torque_residual is not None:
                vector = np.asarray(torque_residual, dtype=float).reshape(-1)
                if vector.size and np.all(np.isfinite(vector)):
                    torque_residuals.append(float(np.linalg.norm(vector)))
            if bool(flags.get("rcs_allocation_saturated", False)):
                object_summary["rcs_allocation_saturated_samples"] += 1
                object_summary["rcs_allocation_saturated_duration_s"] += float(row.get("dt_s", 0.0) or 0.0)
            margin = flags.get("rcs_min_thrust_margin_n")
            if margin is not None and np.isfinite(float(margin)):
                thrust_margins.append(float(margin))
            attitude_error = flags.get("attitude_error_deg", attitude_flags.get("attitude_error_deg"))
            if attitude_error is None:
                attitude_belief = row.get("attitude_belief") or row.get("belief")
                override = dict(
                    attitude_flags.get(
                        "attitude_state_override",
                        flags.get("attitude_state_override", {}),
                    )
                    or {}
                )
                desired_quat = override.get("q_next_bn")
                if attitude_belief is not None and desired_quat is not None:
                    belief_values = np.asarray(attitude_belief, dtype=float).reshape(-1)
                    desired_values = np.asarray(desired_quat, dtype=float).reshape(-1)
                    if belief_values.size >= 10 and desired_values.size == 4:
                        actual_quat = belief_values[6:10]
                        actual_norm = float(np.linalg.norm(actual_quat))
                        desired_norm = float(np.linalg.norm(desired_values))
                        if actual_norm > 0.0 and desired_norm > 0.0:
                            cosine = float(
                                np.clip(
                                    abs(np.dot(actual_quat / actual_norm, desired_values / desired_norm)),
                                    -1.0,
                                    1.0,
                                )
                            )
                            attitude_error = float(np.degrees(2.0 * np.arccos(cosine)))
            if attitude_error is not None and np.isfinite(float(attitude_error)):
                attitude_errors.append(abs(float(attitude_error)))
            delta_mass = flags.get("delta_mass_kg")
            if delta_mass is not None and np.isfinite(float(delta_mass)):
                object_summary["propellant_consumed_kg"] += max(float(delta_mass), 0.0)
            available = flags.get("available_propellant_kg")
            if available is not None and np.isfinite(float(available)):
                final_propellant = max(
                    float(available) - max(float(delta_mass or 0.0), 0.0),
                    0.0,
                )
        if force_residuals:
            object_summary["max_rcs_force_residual_n"] = max(force_residuals)
            object_summary["rms_rcs_force_residual_n"] = float(np.sqrt(np.mean(np.square(force_residuals))))
        if torque_residuals:
            object_summary["max_rcs_torque_residual_nm"] = max(torque_residuals)
            object_summary["rms_rcs_torque_residual_nm"] = float(np.sqrt(np.mean(np.square(torque_residuals))))
        if thrust_margins:
            object_summary["min_rcs_thrust_margin_n"] = min(thrust_margins)
        if attitude_errors:
            object_summary["max_attitude_error_deg"] = max(attitude_errors)
        object_summary["final_propellant_remaining_kg"] = final_propellant
        if any(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and np.isfinite(float(value))
            and float(value) > 0.0
            for value in object_summary.values()
        ):
            summary[str(oid)] = object_summary
    return summary


def _summarize_fsw_actuator_realizations(
    evidence_by_object: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Project typed v2 realization evidence into the existing report summary."""

    summary: dict[str, Any] = {}
    for object_id, evidence in dict(evidence_by_object or {}).items():
        rows = [dict(item or {}) for item in list(dict(evidence or {}).get("realizations", []) or [])]
        if not rows:
            continue
        force_norms = [
            float(np.linalg.norm(np.asarray(row.get("realized_force_n", (0.0, 0.0, 0.0)), dtype=float)))
            for row in rows
        ]
        continuous_force_norms = [
            norm
            for row, norm in zip(rows, force_norms, strict=True)
            if any(
                str(field.get("name", "")) == "throttle"
                for field in list(row.get("device_state", []) or [])
                if isinstance(field, dict)
            )
        ]
        summary[str(object_id)] = {
            "actuator_stack_samples": len(rows),
            "fault_stuck_off_samples": 0,
            "rcs_active_samples": 0,
            "magnetorquer_active_samples": 0,
            "cmg_active_samples": 0,
            "wheel_desaturation_samples": 0,
            "max_gimbal_angle_rad": 0.0,
            "max_commanded_rcs_thruster_force_n": 0.0,
            "max_electric_propulsion_thrust_n": max(continuous_force_norms, default=0.0),
            "saturated_samples": sum(bool(row.get("saturated", False)) for row in rows),
        }
    return summary


def _summarize_ground_station_measurements(measurements: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for station_id, station_payload_raw in sorted(dict(measurements or {}).items()):
        station_payload = dict(station_payload_raw or {})
        targets = dict(station_payload.get("targets", {}) or {})
        station_summary: dict[str, Any] = {
            "measurement_type": station_payload.get("measurement_type", ""),
            "target_count": len(targets),
            "measurement_count": 0,
            "targets": {},
        }
        for object_id, target_payload_raw in sorted(targets.items()):
            target_payload = dict(target_payload_raw or {})
            count = int(target_payload.get("measurement_count", 0) or 0)
            station_summary["measurement_count"] += count
            station_summary["targets"][str(object_id)] = {
                "measurement_count": count,
                "skipped": dict(target_payload.get("skipped", {}) or {}),
            }
        summary[str(station_id)] = station_summary
    return summary


def _finite_min(arr: np.ndarray) -> float | None:
    finite = arr[np.isfinite(arr)]
    return None if finite.size == 0 else float(np.min(finite))


def _finite_max(arr: np.ndarray) -> float | None:
    finite = arr[np.isfinite(arr)]
    return None if finite.size == 0 else float(np.max(finite))


def _last_finite(arr: np.ndarray) -> float | None:
    finite = arr[np.isfinite(arr)]
    return None if finite.size == 0 else float(finite[-1])


def _summarize_reentry_metrics(
    *,
    t_s: np.ndarray,
    reentry_metrics: dict[str, dict[str, np.ndarray]],
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for oid, metrics in dict(reentry_metrics or {}).items():
        active = np.array(metrics.get("active", []), dtype=float).reshape(-1)
        n = min(active.size, t_s.size)
        if n <= 0:
            continue
        entered = np.isfinite(active[:n]) & (active[:n] > 0.5)
        entry_time_s = None
        if bool(np.any(entered)):
            entry_time_s = float(t_s[int(np.flatnonzero(entered)[0])])
        starts = np.flatnonzero(entered & np.concatenate(([True], ~entered[:-1])))
        exits = np.flatnonzero(~entered & np.concatenate(([False], entered[:-1])))
        latest_exit_time_s = float(t_s[int(exits[-1])]) if exits.size else None

        object_summary = {
            "entered_reentry": bool(np.any(entered)),
            "currently_in_reentry": bool(entered[n - 1]) if n > 0 else False,
            "reentry_episode_count": int(starts.size),
            "entry_time_s": entry_time_s,
            "latest_exit_time_s": latest_exit_time_s,
            "min_altitude_km": _finite_min(np.array(metrics.get("altitude_km", []), dtype=float).reshape(-1)[:n]),
            "peak_density_kg_m3": _finite_max(np.array(metrics.get("density_kg_m3", []), dtype=float).reshape(-1)[:n]),
            "peak_relative_speed_m_s": _finite_max(
                np.array(metrics.get("relative_speed_m_s", []), dtype=float).reshape(-1)[:n]
            ),
            "peak_dynamic_pressure_pa": _finite_max(
                np.array(metrics.get("dynamic_pressure_pa", []), dtype=float).reshape(-1)[:n]
            ),
            "peak_drag_decel_m_s2": _finite_max(
                np.array(metrics.get("drag_decel_m_s2", []), dtype=float).reshape(-1)[:n]
            ),
            "peak_lift_accel_m_s2": _finite_max(
                np.array(metrics.get("lift_accel_m_s2", []), dtype=float).reshape(-1)[:n]
            ),
            "peak_lift_to_drag": _finite_max(np.array(metrics.get("lift_to_drag", []), dtype=float).reshape(-1)[:n]),
            "peak_g_load": _finite_max(np.array(metrics.get("g_load", []), dtype=float).reshape(-1)[:n]),
            "peak_heat_rate_w_m2": _finite_max(
                np.array(metrics.get("heat_rate_w_m2", []), dtype=float).reshape(-1)[:n]
            ),
            "final_heat_load_j_m2": _last_finite(
                np.array(metrics.get("heat_load_j_m2", []), dtype=float).reshape(-1)[:n]
            ),
        }
        summary[str(oid)] = object_summary
    return summary


def build_single_run_payload(context: SingleRunPayloadContext) -> dict[str, Any]:
    if context.cfg.simulator.central_body != "earth":
        from sim.reporting.planetary import build_payload
        return build_payload(context)
    from sim.installation.provenance import execution_provenance

    frame_context = frame_context_from_mapping(
        dict(getattr(context.cfg.simulator, "frames", {}) or {}),
        jd_utc_start=context.cfg.simulator.initial_jd_utc,
        source="scenario",
    )
    object_state_frames = {
        str(object_id): str(
            dict(context.object_propagation.get(object_id, {}) or {}).get("state_history_frame", "eci") or "eci"
        ).strip().lower()
        for object_id in context.object_ids
    }
    ground_station_access, ground_station_access_summary = evaluate_ground_station_access(
        ground_stations=list(context.cfg.ground_stations),
        t_s=context.t_s,
        truth_hist=context.truth_hist,
        jd_utc_start=context.cfg.simulator.initial_jd_utc,
        frame_context=frame_context,
        object_state_frames=object_state_frames,
    )
    ground_station_measurements = evaluate_ground_station_measurements(
        ground_stations=list(context.cfg.ground_stations),
        t_s=context.t_s,
        truth_hist=context.truth_hist,
        jd_utc_start=context.cfg.simulator.initial_jd_utc,
        frame_context=frame_context,
        object_state_frames=object_state_frames,
    )
    reference_object_id = default_reference_object_id(context.cfg, available_ids=context.object_ids)
    primary_pair = default_pair_object_ids(context.cfg, available_ids=context.object_ids)
    object_runtime_profiles = {
        str(object_id): str(getattr(section, "runtime_profile", "flight_software") or "flight_software")
        for object_id, section in iter_object_sections(context.cfg)
        if str(object_id) in context.object_ids
    }
    summary = {
        "scenario_name": context.cfg.scenario_name,
        "scenario_description": context.cfg.scenario_description,
        "objects": sorted(str(item) for item in context.object_ids),
        "samples": int(context.t_s.size),
        "dt_s": float(context.dt_s),
        "duration_s": float(context.t_s[-1]) if context.t_s.size else 0.0,
        "terminated_early": bool(context.terminated_early),
        "termination_reason": context.termination_reason,
        "termination_time_s": context.termination_time_s,
        "termination_object_id": context.termination_object_id,
        "rocket_insertion_achieved": bool(context.rocket_inserted),
        "rocket_insertion_time_s": context.rocket_insertion_time_s,
        "target_reference_orbit_enabled": bool(context.target_reference_orbit_truth is not None),
        "reference_object_id": reference_object_id,
        "primary_object_pair": list(primary_pair) if primary_pair is not None else [],
        "thrust_stats": context.thrust_stats,
        "runtime_profile": dict(context.runtime_profile or {}),
        "installation_provenance": execution_provenance(),
        "frame_provenance": frame_context.metadata(),
        "attitude_guardrail_stats": context.attitude_guardrail_stats,
        "knowledge_detection_by_observer": context.knowledge_detection_by_observer,
        "knowledge_consistency_by_observer": context.knowledge_consistency_by_observer,
        "ground_station_access_summary": ground_station_access_summary,
        "ground_station_measurement_summary": _summarize_ground_station_measurements(ground_station_measurements),
        "object_initialization": dict(context.object_initialization),
        "object_propagation": dict(context.object_propagation),
        "object_runtime_profiles": object_runtime_profiles,
        "actuator_diagnostics_summary": {
            **_summarize_actuator_diagnostics(context.controller_debug_hist),
            **_summarize_fsw_actuator_realizations(context.flight_software_evidence),
        },
        "plot_outputs": {},
        "animation_outputs": {},
    }
    reentry_summary = _summarize_reentry_metrics(t_s=context.t_s, reentry_metrics=context.reentry_metrics)
    if reentry_summary:
        summary["reentry_summary_by_object"] = reentry_summary
    rocket_summary: dict[str, Any] = {}
    if context.rocket_metrics:
        def _last_finite(name: str) -> float | None:
            arr = np.array(context.rocket_metrics.get(name, []), dtype=float).reshape(-1)
            finite = arr[np.isfinite(arr)]
            return None if finite.size == 0 else float(finite[-1])

        def _max_finite(name: str) -> float | None:
            arr = np.array(context.rocket_metrics.get(name, []), dtype=float).reshape(-1)
            finite = arr[np.isfinite(arr)]
            return None if finite.size == 0 else float(np.max(finite))

        def _max_abs_finite(name: str) -> float | None:
            arr = np.array(context.rocket_metrics.get(name, []), dtype=float).reshape(-1)
            finite = arr[np.isfinite(arr)]
            return None if finite.size == 0 else float(np.max(np.abs(finite)))

        rocket_summary = {
            "final_altitude_km": _last_finite("altitude_km"),
            "final_speed_km_s": _last_finite("speed_km_s"),
            "final_apoapsis_alt_km": _last_finite("apoapsis_alt_km"),
            "final_periapsis_alt_km": _last_finite("periapsis_alt_km"),
            "final_eccentricity": _last_finite("eccentricity"),
            "final_propellant_remaining_fraction": _last_finite("propellant_remaining_fraction"),
            "max_dynamic_pressure_pa": _max_finite("q_dyn_pa"),
            "max_mach": _max_finite("mach"),
            "max_abs_alpha_deg": _max_abs_finite("alpha_deg"),
            "max_tvc_gimbal_deg": _max_finite("tvc_gimbal_deg"),
            "max_aero_force_n": _max_finite("aero_force_n"),
            "max_aero_moment_nm": _max_finite("aero_moment_nm"),
        }
        summary["rocket_metrics_summary"] = rocket_summary
    trajectory_only = all(
        str(getattr(object_cfg, "runtime_profile", "") or "") == "trajectory_only"
        for object_cfg in context.cfg.objects.values()
    )
    if trajectory_only:
        orbital_delivery = {}
    else:
        from sim.analysis.orbital_delivery import build_orbital_delivery_summary

        orbital_delivery = build_orbital_delivery_summary(
            cfg=context.cfg,
            t_s=context.t_s,
            truth_hist=context.truth_hist,
        )
    if orbital_delivery:
        summary["orbital_delivery"] = orbital_delivery
    return {
        "summary": summary,
        "time_s": context.t_s.tolist(),
        "frame_provenance": frame_context.metadata(),
        "object_state_frames": object_state_frames,
        "truth_by_object": {k: v.tolist() for k, v in context.truth_hist.items()},
        "target_reference_orbit_truth": (
            [] if context.target_reference_orbit_truth is None else context.target_reference_orbit_truth.tolist()
        ),
        "belief_by_object": {k: v.tolist() for k, v in context.belief_hist.items()},
        "applied_thrust_by_object": {k: v.tolist() for k, v in context.thrust_hist.items()},
        "applied_torque_by_object": {k: v.tolist() for k, v in context.torque_hist.items()},
        "desired_attitude_by_object": {k: v.tolist() for k, v in context.desired_attitude_hist.items()},
        "knowledge_by_observer": {
            o: {t: a.tolist() for t, a in bt.items()} for o, bt in context.knowledge_hist.items()
        },
        "knowledge_measurements_by_observer": {
            o: {t: a.tolist() for t, a in bt.items()} for o, bt in context.knowledge_measurement_hist.items()
        },
        "knowledge_detection_by_observer": dict(context.knowledge_detection_by_observer),
        "knowledge_consistency_by_observer": dict(context.knowledge_consistency_by_observer),
        "ground_station_access": ground_station_access,
        "ground_station_access_summary": ground_station_access_summary,
        "ground_station_measurements": ground_station_measurements,
        "object_initialization": dict(context.object_initialization),
        "object_propagation": dict(context.object_propagation),
        "object_runtime_profiles": object_runtime_profiles,
        "bridge_events_by_object": context.bridge_hist,
        "controller_debug_by_object": context.controller_debug_hist,
        "command_decisions_by_object": context.command_decision_hist,
        "flight_software_evidence_by_object": context.flight_software_evidence,
        "rocket_throttle_cmd": context.rocket_throttle_cmd.tolist(),
        "rocket_metrics": {k: v.tolist() for k, v in context.rocket_metrics.items()},
        "reentry_metrics_by_object": {
            oid: {key: arr.tolist() for key, arr in metrics.items()}
            for oid, metrics in context.reentry_metrics.items()
        },
    }
