"""Validated OEL maneuver scenario and review-store continuity checks."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from sim.config import load_simulation_yaml, scenario_config_from_dict
from sim.review import ReviewWorkspace
from sim.single_run import _SingleRunEngine


def maneuver_scenario(spec: Any, result: dict[str, Any], output_dir: Path, *, kind: str) -> dict[str, Any]:
    """Build a one-second equivalent finite burn from the initial ECI state."""
    if kind not in {"apogee", "perigee"}:
        raise ValueError("kind must be apogee or perigee")
    burn_s = 1.0
    dt_s = 1.0
    event_time = float(result[f"first_{kind}_time_s"])
    duration_s = float(math.ceil(event_time + 8.0))
    if duration_s > 20_000.0:
        raise ValueError("Full maneuver verification is limited to 20,000 one-second samples")
    dv_eci_m_s = result["delta_v_eci_m_s"]
    assert dv_eci_m_s is not None
    planned_dv = float(np.linalg.norm(dv_eci_m_s))
    max_accel_m_s2 = max(1.0, planned_dv / burn_s * 1.01)
    return {
        "scenario_name": f"trajectory_target_{kind}_maneuver",
        "scenario_description": (
            f"One-second OEL scheduled finite-burn execution of the planned {kind} impulse, "
            "followed by a two-body coast. The finite burn differs from the ideal target model."
        ),
        "objects": {
            "vehicle": {
                "enabled": True,
                "kind": "satellite",
                "role": "vehicle",
                "specs": {"mass_kg": 100.0},
                "initial_state": {
                    "position_eci_km": list(spec.position_eci_km),
                    "velocity_eci_km_s": list(spec.velocity_eci_km_s),
                },
                "flight_software": {
                    "stack": "fsw.orbit_reference",
                    "hardware_profile": "hardware.ideal_wrench.v1",
                    "task_period_s": dt_s,
                    "params": {
                        "translation_mode": "scheduled_burn",
                        "max_acceleration_m_s2": max_accel_m_s2,
                        "max_force_n": 100.0 * max_accel_m_s2,
                        "scheduled_burns": [
                            {
                                "start_time_s": 0.0,
                                "duration_s": burn_s,
                                "frame": "eci",
                                "delta_v_m_s": list(dv_eci_m_s),
                            }
                        ],
                    },
                },
            },
        },
        "simulator": {
            "duration_s": duration_s,
            "dt_s": dt_s,
            "dynamics": {
                "orbit": {
                    "model": "two_body",
                    "orbit_substep_s": 0.5,
                    "j2": False,
                    "j3": False,
                    "j4": False,
                    "drag": False,
                    "srp": False,
                },
                "attitude": {"enabled": False},
            },
            "termination": {"earth_impact_enabled": False},
            "plugin_validation": {"strict": True},
        },
        "outputs": {
            "output_dir": str(output_dir / "maneuver_run"),
            "mode": "save",
            "stats": {"enabled": True, "print_summary": False, "save_json": True, "save_full_log": False},
            "plots": {"enabled": False, "figure_ids": []},
            "animations": {"enabled": False, "types": []},
            "review": {"enabled": True, "detail": "standard"},
        },
    }


def verify_maneuver_run(
    run_dir: Path,
    *,
    kind: str,
    target_radius_km: float,
    tolerance_km: float,
    planned_delta_v_eci_m_s: tuple[float, float, float],
    planned_event_time_s: float,
) -> dict[str, Any]:
    """Read burn and first apsis evidence from one OEL review store."""
    if kind not in {"apogee", "perigee"}:
        raise ValueError("kind must be apogee or perigee")
    workspace = ReviewWorkspace.open(run_dir)
    thrust_query = (
        "SELECT sample_index, time_s, accel_x_eci_km_s2, accel_y_eci_km_s2, "
        "accel_z_eci_km_s2, burn_active FROM thrust WHERE object_id = 'vehicle' "
        "AND (burn_active = 1 OR accel_norm_km_s2 > 0) ORDER BY sample_index"
    )
    state_query = (
        "SELECT sample_index, time_s, pos_x_eci_km, pos_y_eci_km, pos_z_eci_km, "
        "vel_x_eci_km_s, vel_y_eci_km_s, vel_z_eci_km_s "
        "FROM object_state WHERE object_id = 'vehicle' ORDER BY sample_index"
    )
    try:
        thrust_result = workspace.query(thrust_query, max_rows=20)
        state_result = workspace.query(state_query, max_rows=20_001, max_vm_steps=1_000_000)
        if thrust_result.truncated or state_result.truncated:
            raise RuntimeError("OEL maneuver review query was truncated")
        thrust_rows, state_rows = thrust_result.rows, state_result.rows
        if len(state_rows) < 4:
            raise RuntimeError("OEL maneuver review store has too few state rows")
        if not thrust_rows:
            raise RuntimeError("OEL maneuver review store has no applied burn")
        delivered = np.zeros(3)
        burn_samples: list[dict[str, Any]] = []
        for row in thrust_rows:
            index = int(row["sample_index"])
            if index < 1 or index >= len(state_rows):
                raise RuntimeError("Applied thrust row lacks matching state interval")
            dt_s = float(state_rows[index]["time_s"] - state_rows[index - 1]["time_s"])
            acceleration = np.array(
                [
                    row["accel_x_eci_km_s2"],
                    row["accel_y_eci_km_s2"],
                    row["accel_z_eci_km_s2"],
                ],
                dtype=float,
            )
            delivered += acceleration * dt_s * 1000.0
            burn_samples.append(
                {"sample_index": index, "time_s": row["time_s"], "dt_s": dt_s, "burn_active": bool(row["burn_active"])}
            )

        sign = 1.0 if kind == "apogee" else -1.0
        event_index = None
        radii: list[float] = []
        radial_signs: list[float] = []
        for row in state_rows:
            position = np.array([row["pos_x_eci_km"], row["pos_y_eci_km"], row["pos_z_eci_km"]], dtype=float)
            velocity = np.array([row["vel_x_eci_km_s"], row["vel_y_eci_km_s"], row["vel_z_eci_km_s"]], dtype=float)
            radii.append(float(np.linalg.norm(position)))
            radial_signs.append(sign * float(np.dot(position, velocity)))
        for index in range(2, len(state_rows)):
            if radial_signs[index - 1] > 0.0 and radial_signs[index] <= 0.0:
                event_index = index
                break
        if event_index is None:
            raise RuntimeError(f"OEL maneuver review store has no first {kind} crossing")
        left = event_index - 1
        right = event_index
        # A local parabola from three stored radii estimates the event between
        # two sampled radial-velocity signs. This is interpolation of review
        # evidence, not a new propagation or an exact OEL event row.
        fit_indices = (left, right, right + 1) if right + 1 < len(state_rows) else (left - 1, left, right)
        times = np.array([float(state_rows[i]["time_s"]) for i in fit_indices])
        local_time = times - times[0]
        a, b, c = np.polyfit(local_time, np.array([radii[i] for i in fit_indices]), 2)
        vertex_time = float(-b / (2.0 * a) + times[0])
        if not float(state_rows[left]["time_s"]) <= vertex_time <= float(state_rows[right]["time_s"]):
            raise RuntimeError(f"Interpolated {kind} event fell outside its radial-velocity bracket")
        event_radius = float(a * (vertex_time - times[0]) ** 2 + b * (vertex_time - times[0]) + c)
        residual_km = event_radius - target_radius_km
        planned = np.array(planned_delta_v_eci_m_s, dtype=float)
        burn_error_m_s = float(np.linalg.norm(delivered - planned))
        passed = (
            len(thrust_rows) == 1
            and burn_samples[0]["burn_active"]
            and burn_samples[0]["time_s"] == 1.0
            and burn_error_m_s <= 0.01
            and abs(residual_km) <= tolerance_km
            and abs(vertex_time - planned_event_time_s) <= 10.0
        )
        return {
            "passed": bool(passed),
            "review_evidence": workspace.evidence_identity(),
            "thrust_query": thrust_query,
            "state_query": state_query,
            "burn_samples": burn_samples,
            "planned_delta_v_eci_m_s": planned.tolist(),
            "delivered_delta_v_eci_m_s": delivered.tolist(),
            "delivered_delta_v_error_m_s": burn_error_m_s,
            "burn_model": "one-second scheduled equivalent finite burn with ideal wrench hardware",
            "first_event": {
                "kind": kind,
                "bracket_sample_indices": [left, right],
                "bracket_time_s": [state_rows[left]["time_s"], state_rows[right]["time_s"]],
                "interpolated_time_s": vertex_time,
                "interpolated_radius_km": event_radius,
                "radius_residual_km": residual_km,
                "method": "quadratic interpolation of three review-store radius samples within radial-velocity sign bracket",
            },
        }
    finally:
        workspace.close()


def exact_impulse_apsis_scenario(spec: Any, result: dict[str, Any], output_dir: Path, *, kind: str) -> dict[str, Any]:
    if kind not in {"apogee", "perigee"} or result.get("status") != "converged":
        raise ValueError("A converged named apsis target is required")
    event_time = float(result[f"first_{kind}_time_s"])
    scenario = maneuver_scenario(spec, result, output_dir.parent, kind=kind)
    scenario["scenario_name"] = f"trajectory_target_{kind}_exact_impulse"
    scenario["scenario_description"] = "Exact ECI impulse at departure, followed by a passive OEL coast to the named apsis."
    scenario["objects"]["vehicle"].pop("flight_software")
    scenario["objects"]["vehicle"]["runtime_profile"] = "trajectory_only"
    scenario["outputs"]["output_dir"] = str(output_dir)
    scenario["simulator"]["dt_s"] = 10.0
    scenario["simulator"]["duration_s"] = math.ceil((event_time + 60.0) / 10.0) * 10.0
    scenario["simulator"]["dynamics"]["orbit"]["orbit_substep_s"] = 5.0
    return scenario


def run_exact_impulse_apsis(
    spec: Any,
    result: dict[str, Any],
    output_dir: Path,
    *,
    kind: str,
    config_path: Path | None = None,
) -> dict[str, Any]:
    """Execute a planned instantaneous impulse in a passive OEL run."""
    event_time = float(result[f"first_{kind}_time_s"])
    scenario = exact_impulse_apsis_scenario(spec, result, output_dir, kind=kind)
    cfg = load_simulation_yaml(config_path) if config_path is not None else scenario_config_from_dict(scenario)
    engine = _SingleRunEngine(cfg)
    try:
        applied = engine.apply_impulse("vehicle", result["delta_v_eci_m_s"], maneuver_id="target_impulse")
        while float(engine.t_s[engine.current_index]) < event_time - 1.0e-9:
            now = float(engine.t_s[engine.current_index])
            engine.step(dt_s=min(10.0, event_time - now))
        event_index = engine.current_index
        event_state = engine.truth_hist["vehicle"][event_index, :6].copy()
        while not engine.done:
            engine.step()
        engine._shutdown_flight_software()
        engine.build_payload()
    finally:
        engine.object_step_executor.shutdown()
    workspace = ReviewWorkspace.open(output_dir)
    query = (
        "SELECT m.maneuver_id, m.time_s AS burn_time_s, m.dv_x_eci_m_s, m.dv_y_eci_m_s, "
        "m.dv_z_eci_m_s, s.time_s AS event_time_s, s.pos_x_eci_km, s.pos_y_eci_km, "
        "s.pos_z_eci_km, s.vel_x_eci_km_s, s.vel_y_eci_km_s, s.vel_z_eci_km_s "
        "FROM impulsive_maneuvers m JOIN object_state s ON s.object_id = m.object_id "
        f"AND s.sample_index = {int(event_index)} WHERE m.maneuver_id = 'target_impulse'"
    )
    try:
        rows = workspace.query(query, max_rows=2).rows
        if len(rows) != 1:
            raise RuntimeError("Exact impulse and target event are not in the same review store")
        radius = float(np.linalg.norm(event_state[:3]))
        radial_speed = float(np.dot(event_state[:3], event_state[3:]) / radius)
        residual = radius - float(getattr(spec, f"target_{kind}_radius_km"))
        return {
            "passed": abs(residual) <= float(spec.tolerance_km) and abs(radial_speed) <= 1.0e-4,
            "review_evidence": workspace.evidence_identity(),
            "query": query,
            "burn": applied,
            "event_sample_index": event_index,
            "event_time_s": event_time,
            "event_radius_km": radius,
            "radius_residual_km": residual,
            "radial_speed_km_s": radial_speed,
        }
    finally:
        workspace.close()
