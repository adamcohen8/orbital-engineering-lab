"""Covariance-aware baseline, candidate and complete declared-inventory rescreens."""

from __future__ import annotations

import copy

import numpy as np

from sim.analysis.conjunction_geometry import encounter_frame, refine_time_of_closest_approach
from sim.analysis.conjunction_probability import collision_probability_2d, project_combined_covariance
from sim.pro_conjunction.contracts import EVIDENCE_SCHEMA, digest, validate_problem
from sim.pro_conjunction.probability import applicability, nonlinear_probability
from sim.pro_conjunction.propagation import propagate
from sim.pro_features import FEATURE_SCALE_ANALYTICS, require_pro_feature


def _pair(problem, primary_obj, secondary_obj, primary, secondary):
    ca = refine_time_of_closest_approach(primary.history, secondary.history, numeric_backend="python")
    t = ca["time_s"]
    # TCA may coincide with the incoming side of an impulse; select its covariance too.
    candidates = [primary.at(t, side=side) for side in ("right", "left")]
    x, p = min(candidates, key=lambda pair: np.linalg.norm(pair[0] - ca["primary_state_eci_km_km_s"]))
    y, q = secondary.at(t)
    combined = p + q
    radius = (primary_obj["hard_body_radius_m"] + secondary_obj["hard_body_radius_m"]) / 1000
    diagnostic = applicability(problem, ca, primary, secondary, combined, radius)
    projection, probability, frame = None, None, None
    try:
        frame = encounter_frame(ca["relative_position_eci_km"], ca["relative_velocity_eci_km_s"])
        projection = project_combined_covariance(p, q, frame["basis_rows_eci"])
        if diagnostic["status"] == "screen_passed":
            probability = collision_probability_2d(
                frame["plane_miss_km"], projection["encounter_plane_covariance_km2"], radius, numeric_backend="python"
            )
    except ValueError as exc:
        diagnostic["status"] = "withheld"
        diagnostic["reasons"].append("invalid_encounter_projection")
        diagnostic["message"] = str(exc)
    row = {
        "primary_id": primary_obj["object_id"],
        "secondary_id": secondary_obj["object_id"],
        "closest_approach": ca,
        "state_epoch_s": t,
        "encounter_frame": frame,
        "primary_covariance_at_tca": p.tolist(),
        "secondary_covariance_at_tca": q.tolist(),
        "primary_mean_at_tca": x.tolist(),
        "secondary_mean_at_tca": y.tolist(),
        "covariance_projection": projection,
        "applicability": diagnostic,
        "probability_2d": probability,
        "nonlinear_probability": None,
        "empirically_calibrated": False,
        "hard_body_radius_km": radius,
    }
    if "monte_carlo" in problem:
        row["nonlinear_probability"] = nonlinear_probability(problem, primary_obj, secondary_obj)
    return row


def evaluate(problem):
    """Internal normalized executor. Entry points own validation and entitlement."""
    objects = {o["object_id"]: o for o in problem["objects"]}
    primary_obj = objects[problem["primary_id"]]
    primary = propagate(problem, primary_obj, burn=problem.get("burn"))
    rows = []
    for object_id in problem["secondary_ids"]:
        obj = objects[object_id]
        rows.append(_pair(problem, primary_obj, obj, primary, propagate(problem, obj)))
    return {
        "encounters": rows,
        "primary_burn_jacobian": primary.burn_jacobian,
        "rescreened_object_ids": sorted(problem["secondary_ids"]),
    }


def assess(problem):
    require_pro_feature(FEATURE_SCALE_ANALYTICS)
    p = validate_problem(problem)
    baseline_problem = copy.deepcopy(p)
    baseline_problem.pop("burn", None)
    baseline = evaluate(baseline_problem)
    candidate = evaluate(p) if "burn" in p else None
    return {
        "schema_version": EVIDENCE_SCHEMA,
        "problem_sha256": digest(p),
        "problem": p,
        "status": "assessment_completed",
        "baseline": baseline,
        "candidate": candidate,
        "risk_disposition": "analyst_review_required",
        "empirically_calibrated": False,
        "limitations": [
            "Finite-window assessment of the entire explicitly declared inventory, not undeclared catalog objects.",
            "Shared ONP state transitions propagate uncertainty; they do not calibrate the input covariance.",
            "Independent-object and independent burn-execution errors are explicit assumptions.",
            "2D usage screens use analyst-declared thresholds and do not prove applicability.",
            "No maneuver authorization, global-optimality or operational safety claim.",
        ],
    }


def sensitivity(problem, scales=(0.5, 1.0, 2.0), execution_scales=(0.0, 1.0, 2.0)):
    require_pro_feature(FEATURE_SCALE_ANALYTICS)
    p = validate_problem(problem)
    if not scales or not execution_scales or len(scales) * len(execution_scales) > 25:
        raise ValueError("sensitivity requires 1..25 explicitly bounded combinations")
    rows = []
    from sim.pro_conjunction.contracts import number

    for scale in scales:
        scale = number(scale, low=1e-6, high=100)
        for execution_scale in execution_scales:
            execution_scale = number(execution_scale, low=0, high=100)
            trial = copy.deepcopy(p)
            for obj in trial["objects"]:
                obj["uncertainty"]["covariance"] = (np.array(obj["uncertainty"]["covariance"]) * scale).tolist()
            if "burn" in trial:
                trial["burn"]["execution_covariance_m2_s2"] = (
                    np.array(trial["burn"]["execution_covariance_m2_s2"]) * execution_scale
                ).tolist()
            rows.append(
                {
                    "covariance_multiplier": scale,
                    "execution_covariance_multiplier": execution_scale,
                    "assessment": assess(trial),
                }
            )
    return {
        "schema_version": "oel.pro_conjunction_sensitivity.v1",
        "problem_sha256": digest(p),
        "rows": rows,
        "interpretation": "assumption sensitivity, not empirical calibration",
    }
