"""Explicit 2D usage diagnostics and finite-window nonlinear ONP Monte Carlo."""

from __future__ import annotations

import copy

import numpy as np
from scipy.stats import beta

from sim.analysis.conjunction_geometry import refine_time_of_closest_approach
from sim.pro_conjunction.propagation import propagate


def applicability(problem, closest, primary, secondary, combined_covariance, radius_km):
    """Declared engineering screens; thresholds are not calibrated usage boundaries."""
    t = closest["time_s"]
    r = np.array(closest["relative_position_eci_km"])
    v = np.array(closest["relative_velocity_eci_km_s"])
    speed = np.linalg.norm(v)
    c = np.asarray(combined_covariance)
    eig = np.linalg.eigvalsh(c[:3, :3])
    reasons, metrics = [], {}
    if closest["at_search_boundary"]:
        reasons.append("tca_not_bracketed")
    if speed <= 1e-12 or eig[0] <= 0:
        return {
            "status": "withheld",
            "reasons": reasons + ["degenerate_velocity_or_position_covariance"],
            "metrics": metrics,
        }
    policy = problem["applicability"]
    z = v / speed
    sigma = np.sqrt(eig[0])
    half = (policy["sigma_extent"] * np.sqrt(z @ c[:3, :3] @ z) + radius_km) / speed
    metrics["encounter_half_width_s"] = float(half)
    ratio = float(np.sqrt(np.linalg.eigvalsh(c[3:, 3:])[-1]) / speed)
    metrics["velocity_sigma_fraction"] = ratio
    if ratio > policy["max_velocity_sigma_fraction"]:
        reasons.append("velocity_uncertainty_large")
    if t - half < 0 or t + half > problem["duration_s"]:
        reasons.append("uncertainty_encounter_not_bracketed")
    else:
        curvature, variation = 0.0, 0.0
        # Whiten position covariance so large directions cannot hide small ones.
        values, vectors = np.linalg.eigh(c[:3, :3])
        whitening = (vectors / np.sqrt(values)) @ vectors.T
        for offset in (-half, half):
            xp, pp = primary.at(t + offset)
            xs, ps = secondary.at(t + offset)
            departure = xp[:3] - xs[:3] - (r + v * offset)
            curvature = max(curvature, float(np.linalg.norm(departure) / sigma))
            change = whitening @ (pp[:3, :3] + ps[:3, :3] - c[:3, :3]) @ whitening.T
            variation = max(variation, float(np.linalg.norm(change, ord=2)))
        metrics.update(curvature_sigma=curvature, covariance_fraction=variation)
        if curvature > policy["max_curvature_sigma"]:
            reasons.append("nonlinear_encounter")
        if variation > policy["max_covariance_fraction"]:
            reasons.append("time_varying_covariance")
        burn = problem.get("burn")
        if burn is not None and t - half <= burn["time_s"] <= t + half:
            reasons.append("maneuver_inside_encounter")
    return {
        "status": "screen_passed" if not reasons else "withheld",
        "reasons": reasons,
        "metrics": metrics,
        "thresholds": policy,
        "claim": "analyst-declared engineering screen, not a proven or calibrated validity boundary",
    }


def binomial_interval(hits, trials, confidence):
    """Two-sided exact Clopper-Pearson interval, including zero/all-hit cases."""
    alpha = 1 - confidence
    return [
        0.0 if hits == 0 else float(beta.ppf(alpha / 2, hits, trials - hits + 1)),
        1.0 if hits == trials else float(beta.ppf(1 - alpha / 2, hits + 1, trials - hits)),
    ]


def _draw(rng, covariance):
    # Scale coordinates before factoring mixed position/velocity units.
    c = np.asarray(covariance)
    scales = np.sqrt(np.maximum(np.diag(c), 0))
    safe = np.where(scales > 0, scales, 1.0)
    corr = c / np.outer(safe, safe)
    values, vectors = np.linalg.eigh(0.5 * (corr + corr.T))
    return scales * ((vectors * np.sqrt(np.maximum(values, 0))) @ rng.standard_normal(len(c)))


def nonlinear_probability(problem, primary_obj, secondary_obj):
    """Probability of any hard-body overlap in the declared finite time window.

    Initial errors are Gaussian in the declared ECI Cartesian coordinates;
    execution errors are independent Gaussian in commanded impulse coordinates.
    Neither distribution is empirically calibrated by this calculation.
    """
    spec = problem["monte_carlo"]
    rng = np.random.default_rng(spec["seed"])
    radius = (primary_obj["hard_body_radius_m"] + secondary_obj["hard_body_radius_m"]) / 1000
    refined = {**problem, "step_s": problem["step_s"] / 2}
    hits, disagreements, boundary_minima, maximum_distance_change = 0, 0, 0, 0.0
    for _ in range(spec["trials"]):
        xp = np.array(primary_obj["state_eci_km_km_s"]) + _draw(rng, primary_obj["uncertainty"]["covariance"])
        xs = np.array(secondary_obj["state_eci_km_km_s"]) + _draw(rng, secondary_obj["uncertainty"]["covariance"])
        burn = copy.deepcopy(problem.get("burn"))
        if burn is not None:
            burn["delta_v_m_s"] = (
                np.array(burn["delta_v_m_s"]) + _draw(rng, burn["execution_covariance_m2_s2"])
            ).tolist()
        distances = []
        for settings in (problem, refined):
            p = propagate(settings, primary_obj, burn=burn, covariance=False, state=xp)
            s = propagate(settings, secondary_obj, covariance=False, state=xs)
            ca = refine_time_of_closest_approach(p.history, s.history, numeric_backend="python")
            distances.append(ca["miss_distance_km"])
        collision = distances[1] <= radius
        hits += collision
        disagreements += (distances[0] <= radius) != collision
        boundary_minima += ca["at_search_boundary"]
        maximum_distance_change = max(maximum_distance_change, abs(distances[1] - distances[0]))
    interval = binomial_interval(hits, spec["trials"], spec["confidence"])
    resolved = (
        0 < hits < spec["trials"] and interval[1] - interval[0] <= spec["max_interval_width"] and disagreements == 0
    )
    return {
        "method": "nonlinear_3d_cartesian_ONP_monte_carlo",
        "status": "resolved" if resolved else "unresolved",
        "pc": hits / spec["trials"] if resolved else None,
        "empirical_frequency": hits / spec["trials"],
        "confidence_interval": interval,
        "confidence": spec["confidence"],
        "trials": spec["trials"],
        "hits": int(hits),
        "seed": spec["seed"],
        "time_window_s": [0, problem["duration_s"]],
        "event": "at_least_one_spherical_hard_body_overlap_in_window",
        "step_convergence": {
            "coarse_step_s": problem["step_s"],
            "fine_step_s": refined["step_s"],
            "classification_disagreements": int(disagreements),
            "max_miss_distance_change_km": maximum_distance_change,
        },
        "boundary_minima": int(boundary_minima),
        "limitations": [
            "Conditional on declared independent Gaussian initial/execution errors and deterministic ONP forces.",
            "Binomial intervals measure sampling uncertainty, not covariance realism or model error.",
            "Step-halving is a numerical diagnostic, not a rigorous trajectory error bound.",
            "Zero hits produces an upper bound and an unresolved result, never a zero-risk claim.",
        ],
    }
