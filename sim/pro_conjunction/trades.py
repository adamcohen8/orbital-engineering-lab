"""Bounded local constrained avoidance; every accepted solution is freshly rescreened."""

from __future__ import annotations

import copy
import math

import numpy as np
from scipy.optimize import minimize

from sim.pro_conjunction.assessment import assess, evaluate
from sim.pro_conjunction.contracts import closed, covariance, digest, number, validate_problem
from sim.pro_features import FEATURE_SCALE_ANALYTICS, FEATURE_TRAJECTORY_OPTIMIZATION, require_pro_feature


def validate_trade(value, duration):
    t = closed(
        value,
        {
            "frame",
            "burn_time_bounds_s",
            "delta_v_bounds_m_s",
            "execution_covariance_m2_s2",
            "max_delta_v_m_s",
            "max_pc",
            "min_miss_distance_km",
            "acceptance_probability",
            "starts",
            "seed",
            "max_iterations",
            "max_evaluations",
        },
        name="avoidance trade",
    )
    if t["frame"] not in {"eci", "ric"} or t["acceptance_probability"] not in {"2d", "nonlinear_upper"}:
        raise ValueError("unsupported trade frame or acceptance probability")
    bounds = [t["burn_time_bounds_s"], *t["delta_v_bounds_m_s"]]
    if len(bounds) != 4:
        raise ValueError("three delta-v component bounds required")
    for i, bound in enumerate(bounds):
        if len(bound) != 2:
            raise ValueError("each bound is [lower, upper]")
        lo, hi = [number(v, low=0 if i == 0 else -1000, high=duration if i == 0 else 1000) for v in bound]
        if lo >= hi:
            raise ValueError("strictly increasing variable bounds required")
    for k, low, high, integer in (
        ("max_delta_v_m_s", 1e-6, 1000, False),
        ("max_pc", 1e-12, 1, False),
        ("min_miss_distance_km", 0, 1e5, False),
        ("starts", 1, 8, True),
        ("seed", 0, 2**32 - 1, True),
        ("max_iterations", 1, 100, True),
        ("max_evaluations", 1, 1000, True),
    ):
        t[k] = number(t[k], low=low, high=high, integer=integer)
    t["execution_covariance_m2_s2"] = covariance(t["execution_covariance_m2_s2"], 3)
    return copy.deepcopy(t)


def margins(result, trade):
    values = []
    for row in result["encounters"]:
        values.append(
            (row["closest_approach"]["miss_distance_km"] - trade["min_miss_distance_km"])
            / max(trade["min_miss_distance_km"], row["hard_body_radius_km"], 1e-6)
        )
        pc = row["probability_2d"]
        # An unavailable 2D result cannot satisfy a constraint.
        values.append(-1.0 if pc is None else 1 - pc["collision_probability"] / trade["max_pc"])
    return values


def optimize(problem, trade):
    require_pro_feature(FEATURE_SCALE_ANALYTICS)
    require_pro_feature(FEATURE_TRAJECTORY_OPTIMIZATION)
    p = validate_problem(problem)
    t = validate_trade(trade, p["duration_s"])
    if t["max_evaluations"] * len(p["objects"]) * math.ceil(p["duration_s"] / p["step_s"]) > 200000:
        raise ValueError("optimization exceeds 200000 covariance object-step search budget")
    if t["acceptance_probability"] == "nonlinear_upper" and "monte_carlo" not in p:
        raise ValueError("nonlinear acceptance requires an explicit Monte Carlo contract")
    bounds = np.array([t["burn_time_bounds_s"], *t["delta_v_bounds_m_s"]], dtype=float)
    lo, width = bounds[:, 0], bounds[:, 1] - bounds[:, 0]
    rng = np.random.default_rng(t["seed"])
    ledger, attempts, cache, candidates = [], [], {}, []

    def materialize(z):
        x = lo + np.asarray(z) * width
        trial = copy.deepcopy(p)
        trial.pop("monte_carlo", None)  # 2D search surrogate; independent final nonlinear acceptance below.
        trial["burn"] = {
            "time_s": float(x[0]),
            "frame": t["frame"],
            "delta_v_m_s": x[1:].tolist(),
            "execution_covariance_m2_s2": t["execution_covariance_m2_s2"],
        }
        return trial

    def evaluation(z):
        key = tuple(float(v) for v in z)
        if key in cache:
            return cache[key]
        if len(ledger) >= t["max_evaluations"]:
            raise RuntimeError("trajectory evaluation budget exhausted")
        trial = materialize(z)
        magnitude = float(np.linalg.norm(trial["burn"]["delta_v_m_s"]))
        row = {"decision": list(key), "burn": trial["burn"], "delta_v_m_s": magnitude}
        try:
            result = evaluate(trial)
            residuals = margins(result, t) + [1 - magnitude / t["max_delta_v_m_s"]]
            row.update(status="evaluated", margins=residuals)
        except (ValueError, RuntimeError) as exc:
            residuals = [-1.0] * (len(p["secondary_ids"]) * 2 + 1)
            row.update(status="failed", message=str(exc), margins=residuals)
        ledger.append(row)
        cache[key] = magnitude, np.asarray(residuals)
        return cache[key]

    for start in range(t["starts"]):
        z = np.full(4, 0.5) if start == 0 else rng.uniform(0, 1, 4)
        if start == 0:
            z[1:] = np.clip(-lo[1:] / width[1:], 0, 1)
        try:
            solution = minimize(
                lambda a: evaluation(a)[0] / t["max_delta_v_m_s"],
                z,
                method="SLSQP",
                bounds=[(0, 1)] * 4,
                constraints=[{"type": "ineq", "fun": lambda a: evaluation(a)[1]}],
                options={"maxiter": t["max_iterations"], "ftol": 1e-9, "eps": 1e-4},
            )
            magnitude, residuals = evaluation(solution.x)
            attempts.append(
                {
                    "start": start,
                    "solver_success": bool(solution.success),
                    "message": str(solution.message),
                    "decision": solution.x.tolist(),
                    "margins": residuals.tolist(),
                }
            )
            if np.all(residuals >= 0):
                candidates.append((magnitude, solution.x.copy()))
        except RuntimeError as exc:
            attempts.append({"start": start, "status": "budget_exhausted", "message": str(exc)})
            break
    # Also retain feasible evaluated points if a solver stopped for another reason.
    for row in ledger:
        if row["status"] == "evaluated" and min(row["margins"]) >= 0:
            candidates.append((row["delta_v_m_s"], np.array(row["decision"])))
    verification, selected = [], None
    seen = set()
    for magnitude, z in sorted(candidates, key=lambda row: (row[0], tuple(row[1]))):
        if tuple(z) in seen:
            continue
        seen.add(tuple(z))
        trial = materialize(z)
        if "monte_carlo" in p:
            trial["monte_carlo"] = p["monte_carlo"]
        packet = assess(trial)  # Fresh independent histories, covariance and complete declared rescreen.
        feasible = min(margins(packet["candidate"], t)) >= 0 and magnitude <= t["max_delta_v_m_s"]
        if t["acceptance_probability"] == "nonlinear_upper":
            feasible = feasible and all(
                row["nonlinear_probability"]["confidence_interval"][1] <= t["max_pc"]
                and row["nonlinear_probability"]["step_convergence"]["classification_disagreements"] == 0
                for row in packet["candidate"]["encounters"]
            )
        verification.append({"decision": z.tolist(), "feasible": bool(feasible), "assessment": packet})
        if feasible:
            selected = verification[-1]
            break
        if len(verification) >= 8:
            break
    return {
        "schema_version": "oel.pro_avoidance_trade.v1",
        "problem_sha256": digest(p),
        "problem": p,
        "trade": t,
        "trade_sha256": digest(t),
        "status": "feasible_local_candidate" if selected else "no_verified_candidate",
        "selected": selected,
        "verification": verification,
        "attempts": attempts,
        "evaluation_ledger": ledger,
        "risk_disposition": "analyst_review_required",
        "limitations": [
            "Local bounded SLSQP search; failure is not proof of infeasibility and success is not global optimality.",
            "2D probability is the search surrogate; nonlinear upper-bound acceptance is separately requested and verified.",
            "Every declared secondary is rescreened. Unlisted objects and post-horizon encounters are excluded.",
            "No empirically calibrated risk, operational safety or maneuver authorization claim.",
        ],
    }
