"""Two-impulse targeting with the same configured ONP dynamics as the review run."""

from __future__ import annotations

import math
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from sim.config import scenario_config_from_dict, validate_scenario_plugins
from sim.dynamics.orbit.environment import EARTH_MU_KM3_S2
from sim.review import ReviewWorkspace
from sim.trajectory_targeting.onp_adapter import ONPTrial, prepare_onp_scenario
from sim.trajectory_targeting.two_burn import TwoBurnSpec, hohmann_reference
from sim.trajectory_targeting.vector import VectorTargetProblem, solve_vector_target


def onp_scenario(spec: TwoBurnSpec, template: dict[str, Any], output_dir: Path, *, duration_s: float) -> dict[str, Any]:
    """Preserve the user's OEL force configuration; replace only study state/output."""
    spec.validate()
    speed = math.sqrt(EARTH_MU_KM3_S2 / spec.initial_radius_km)
    source = prepare_onp_scenario(
        template, np.array([spec.initial_radius_km, 0.0, 0.0, 0.0, speed, 0.0]),
        output_dir, coast_step_s=spec.coast_step_s, duration_s=duration_s,
    )
    source["scenario_name"] = "trajectory_target_two_impulse_onp"
    return source


def target_two_burn_onp(
    spec: TwoBurnSpec, scenario: dict[str, Any], *, numeric_backend: str = "rust"
) -> dict[str, Any]:
    """Solve using fresh ONP runtimes so trial plugin state never crosses trials."""
    spec.validate()
    reference = hohmann_reference(spec)
    target_speed = math.sqrt(EARTH_MU_KM3_S2 / spec.target_radius_km)
    with tempfile.TemporaryDirectory(prefix="oel-onp-target-") as scratch:
        trial_duration = math.ceil(spec.max_coast_s / spec.coast_step_s) * spec.coast_step_s + spec.coast_step_s
        trial_scenario = onp_scenario(spec, scenario, Path(scratch), duration_s=trial_duration)
        cfg = scenario_config_from_dict(trial_scenario)
        plugin_errors = validate_scenario_plugins(cfg)
        if plugin_errors:
            raise ValueError("ONP template plugin validation failed: " + "; ".join(plugin_errors))

        def evaluate(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
            trial = ONPTrial(cfg)
            try:
                initial = trial.state()
                post1 = trial.impulse(np.array([0.0, float(x[0]), 0.0]))
                event_time, pre2 = trial.first_apsis(
                    "apogee", step_s=spec.coast_step_s, max_time_s=spec.max_coast_s)
                tangent = pre2[3:] / np.linalg.norm(pre2[3:])
                post2 = pre2.copy()
                post2[3:] += tangent * x[1] / 1000.0
                return initial, post1, pre2, event_time, post2
            finally:
                trial.close()

        def observe(x: np.ndarray) -> tuple[float, float]:
            _, _, pre2, _, post2 = evaluate(x)
            return float(np.linalg.norm(pre2[:3])), float(np.linalg.norm(post2[3:]))

        problem = VectorTargetProblem(
            variable_names=("first_prograde_delta_v_m_s", "second_prograde_delta_v_m_s"),
            goal_names=("first_apogee_radius_km", "post_second_burn_speed_km_s"),
            initial=(min(0.8 * reference["burn1_m_s"], spec.max_burn1_m_s),
                     min(1.1 * reference["burn2_m_s"], spec.max_burn2_m_s)),
            lower_bounds=(0.0, 0.0), upper_bounds=(spec.max_burn1_m_s, spec.max_burn2_m_s),
            targets=(spec.target_radius_km, target_speed),
            tolerances=(spec.radius_tolerance_km, spec.speed_tolerance_m_s / 1000.0),
            variable_scales=(max(reference["burn1_m_s"], 1.0), max(reference["burn2_m_s"], 1.0)),
            max_iterations=spec.max_iterations,
            numeric_backend=numeric_backend,
        )
        correction = solve_vector_target(problem, observe)
        result: dict[str, Any] = {
            "schema_version": "oel.trajectory_target.two_burn_result.v1",
            "status": correction.status, "reason": correction.reason,
            "spec": asdict(spec), "analytical_hohmann": reference,
            "solver": asdict(correction),
            "assumptions": {
                "model": "Configured OEL ONP Earth ECI forces, propagated through the trajectory_only dynamics path",
                "maneuvers": "two instantaneous prograde impulses, at departure and first radial apogee",
                "goal": "first apogee radius and osculating post-burn speed; not a persistent circular orbit under perturbations",
                "optimization": "bounded two-variable differential correction, not a global minimum-delta-v proof",
            },
        }
        if correction.status == "converged":
            initial, post1, pre2, event_time, post2 = evaluate(np.array(correction.variables))
            result.update({
                "burn1_delta_v_eci_m_s": ((post1[3:] - initial[3:]) * 1000).tolist(),
                "burn2_delta_v_eci_m_s": ((post2[3:] - pre2[3:]) * 1000).tolist(),
                "first_apogee_time_s": event_time,
                "initial_state_eci": initial.tolist(),
                "post_burn1_state_eci": post1.tolist(),
                "pre_burn2_state_eci": pre2.tolist(),
                "post_burn2_state_eci": post2.tolist(),
                "total_delta_v_m_s": float(sum(correction.variables)),
            })
        return result


def verify_onp_transfer(spec: TwoBurnSpec, result: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    """Check both impulse rows and the solved event in the same OEL review DB."""
    workspace = ReviewWorkspace.open(run_dir)
    query = ("SELECT maneuver_id, time_s, dv_x_eci_m_s, dv_y_eci_m_s, dv_z_eci_m_s, "
             "pre_pos_x_eci_km, pre_pos_y_eci_km, pre_pos_z_eci_km, "
             "post_pos_x_eci_km, post_pos_y_eci_km, post_pos_z_eci_km, "
             "pre_vel_x_eci_km_s, pre_vel_y_eci_km_s, pre_vel_z_eci_km_s, "
             "post_vel_x_eci_km_s, post_vel_y_eci_km_s, post_vel_z_eci_km_s "
             "FROM impulsive_maneuvers ORDER BY time_s")
    final_query = ("SELECT time_s, pos_x_eci_km, pos_y_eci_km, pos_z_eci_km, "
                   "vel_x_eci_km_s, vel_y_eci_km_s, vel_z_eci_km_s FROM object_state "
                   "WHERE object_id = 'vehicle' ORDER BY sample_index DESC LIMIT 1")
    try:
        rows = workspace.query(query, max_rows=3).rows
        final_rows = workspace.query(final_query, max_rows=1).rows
        if len(rows) != 2 or len(final_rows) != 1:
            raise RuntimeError("Expected two impulse rows and one final state in the same review store")
        checks = []
        for row in rows:
            pre_r = np.array([row[f"pre_pos_{axis}_eci_km"] for axis in "xyz"])
            post_r = np.array([row[f"post_pos_{axis}_eci_km"] for axis in "xyz"])
            pre_v = np.array([row[f"pre_vel_{axis}_eci_km_s"] for axis in "xyz"])
            post_v = np.array([row[f"post_vel_{axis}_eci_km_s"] for axis in "xyz"])
            dv = np.array([row[f"dv_{axis}_eci_m_s"] for axis in "xyz"])
            checks.append({"maneuver_id": row["maneuver_id"], "time_s": row["time_s"],
                           "position_jump_km": float(np.linalg.norm(post_r - pre_r)),
                           "delta_v_error_m_s": float(np.linalg.norm((post_v - pre_v) * 1000 - dv)),
                           "post_radius_km": float(np.linalg.norm(post_r)),
                           "post_speed_km_s": float(np.linalg.norm(post_v)),
                           "post_radial_speed_km_s": float(np.dot(post_r, post_v) / np.linalg.norm(post_r))})
        second = checks[1]
        speed_goal = math.sqrt(EARTH_MU_KM3_S2 / spec.target_radius_km)
        planned_pre2 = np.asarray(result["pre_burn2_state_eci"], dtype=float)
        actual_pre2 = np.array([rows[1][f"pre_pos_{axis}_eci_km"] for axis in "xyz"]
                               + [rows[1][f"pre_vel_{axis}_eci_km_s"] for axis in "xyz"])
        state_position_error = float(np.linalg.norm(actual_pre2[:3] - planned_pre2[:3]))
        state_velocity_error_m_s = float(np.linalg.norm(actual_pre2[3:] - planned_pre2[3:]) * 1000)
        final = final_rows[0]
        final_state = np.array([final[f"pos_{axis}_eci_km"] for axis in "xyz"]
                               + [final[f"vel_{axis}_eci_km_s"] for axis in "xyz"])
        passed = ([row["maneuver_id"] for row in checks] == ["departure", "circularization"]
                  and checks[0]["time_s"] == 0.0
                  and abs(second["time_s"] - result["first_apogee_time_s"]) <= 1e-6
                  and abs(second["post_radius_km"] - spec.target_radius_km) <= spec.radius_tolerance_km
                  and abs(second["post_speed_km_s"] - speed_goal) * 1000 <= spec.speed_tolerance_m_s
                  and abs(second["post_radial_speed_km_s"]) <= 1e-4
                  and state_position_error <= max(0.01, spec.radius_tolerance_km)
                  and state_velocity_error_m_s <= max(0.01, spec.speed_tolerance_m_s)
                  and np.isfinite(final_state).all()
                  and final["time_s"] >= second["time_s"] + 600.0 - 1e-6
                  and all(row["position_jump_km"] == 0.0 and row["delta_v_error_m_s"] <= 1e-8 for row in checks))
        return {"passed": bool(passed), "review_evidence": workspace.evidence_identity(),
                "burn_query": query, "final_query": final_query, "burns": checks,
                "radius_residual_km": second["post_radius_km"] - spec.target_radius_km,
                "speed_residual_m_s": (second["post_speed_km_s"] - speed_goal) * 1000,
                "trial_to_review_position_error_km": state_position_error,
                "trial_to_review_velocity_error_m_s": state_velocity_error_m_s,
                "final_time_s": final["time_s"]}
    finally:
        workspace.close()
