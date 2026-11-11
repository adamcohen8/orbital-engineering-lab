from __future__ import annotations

import csv
import json
import math
import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from sim.dynamics.orbit.environment import EARTH_MU_KM3_S2
from sim.dynamics.orbit.lambert import solve_lambert_universal_variable
from sim.dynamics.orbit.two_body import propagate_two_body_rk4
from sim.estimation.batch_least_squares import solve_batch_least_squares
from sim.estimation.parameters import EstimatedParameter, ParameterSet
from sim.estimation.weighting import prepare_covariance_whitener, whiten_residual_block
from sim.ingestion import ingest_state_vector
from sim.numeric_backend import normalize_numeric_backend
from sim.optical_tracking import OPTICAL_ANGLE_CONTRACT, los_to_ra_dec, validate_optical_angle_packet

IOD_RESULT_SCHEMA_VERSION = 1
EARTH_RADIUS_KM = 6378.137


@dataclass(frozen=True)
class IODHypothesis:
    hypothesis_id: str
    method: str
    epoch_time_s: float
    state_eci_km_s: tuple[float, float, float, float, float, float]
    source_observation_ids: tuple[str, ...]
    geometry: Mapping[str, Any]
    provenance: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def solve_gibbs_iod(
    positions_eci_km: Sequence[Sequence[float]],
    *,
    observation_ids: Sequence[str] = ("r1", "r2", "r3"),
    epoch_time_s: float = 0.0,
    mu_km3_s2: float = EARTH_MU_KM3_S2,
    max_coplanarity_deg: float = 3.0,
    min_separation_deg: float = 1.0,
) -> IODHypothesis:
    """Classical Gibbs position-triplet IOD at the middle observation."""

    r1, r2, r3 = _position_triplet(positions_eci_km)
    geometry = _position_geometry(r1, r2, r3)
    if float(geometry["coplanarity_deg"]) > float(max_coplanarity_deg):
        raise ValueError("Gibbs IOD requires a near-coplanar position triplet.")
    if float(geometry["minimum_separation_deg"]) < float(min_separation_deg):
        raise ValueError("Gibbs IOD geometry is too tightly spaced; use Herrick-Gibbs.")
    z12 = np.cross(r1, r2)
    z23 = np.cross(r2, r3)
    z31 = np.cross(r3, r1)
    n = np.linalg.norm(r1) * z23 + np.linalg.norm(r2) * z31 + np.linalg.norm(r3) * z12
    d = z12 + z23 + z31
    s = (
        r1 * (np.linalg.norm(r2) - np.linalg.norm(r3))
        + r2 * (np.linalg.norm(r3) - np.linalg.norm(r1))
        + r3 * (np.linalg.norm(r1) - np.linalg.norm(r2))
    )
    n_norm = float(np.linalg.norm(n))
    d_norm = float(np.linalg.norm(d))
    if n_norm <= 1.0e-12 or d_norm <= 1.0e-12 or float(np.dot(n, d)) <= 0.0:
        raise ValueError("Gibbs IOD geometry is singular or directionally inconsistent.")
    scale = float(np.sqrt(float(mu_km3_s2) / (n_norm * d_norm)))
    v2 = scale * (np.cross(d, r2) / np.linalg.norm(r2) + s)
    state = np.hstack((r2, v2))
    _require_finite_state(state)
    return IODHypothesis(
        hypothesis_id="gibbs:000",
        method="gibbs",
        epoch_time_s=float(epoch_time_s),
        state_eci_km_s=tuple(float(value) for value in state),
        source_observation_ids=tuple(str(item) for item in observation_ids),
        geometry=geometry,
        provenance={"mu_km3_s2": float(mu_km3_s2), "velocity_epoch": "middle_position"},
    )


def solve_herrick_gibbs_iod(
    positions_eci_km: Sequence[Sequence[float]],
    times_s: Sequence[float],
    *,
    observation_ids: Sequence[str] = ("r1", "r2", "r3"),
    mu_km3_s2: float = EARTH_MU_KM3_S2,
    max_coplanarity_deg: float = 3.0,
    max_separation_deg: float = 5.0,
) -> IODHypothesis:
    """Herrick-Gibbs short-arc position-triplet IOD at the middle epoch."""

    r1, r2, r3 = _position_triplet(positions_eci_km)
    times = np.asarray(times_s, dtype=float).reshape(3)
    if not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0.0):
        raise ValueError("Herrick-Gibbs times must be finite and strictly increasing.")
    geometry = _position_geometry(r1, r2, r3)
    if float(geometry["coplanarity_deg"]) > float(max_coplanarity_deg):
        raise ValueError("Herrick-Gibbs IOD requires a near-coplanar position triplet.")
    if float(geometry["maximum_separation_deg"]) > float(max_separation_deg):
        raise ValueError("Herrick-Gibbs is limited to tightly spaced position triplets.")
    dt21 = float(times[1] - times[0])
    dt32 = float(times[2] - times[1])
    dt31 = float(times[2] - times[0])
    mu = float(mu_km3_s2)
    term1 = -dt32 * (1.0 / (dt21 * dt31) + mu / (12.0 * np.linalg.norm(r1) ** 3)) * r1
    term2 = (dt32 - dt21) * (
        1.0 / (dt21 * dt32) + mu / (12.0 * np.linalg.norm(r2) ** 3)
    ) * r2
    term3 = dt21 * (1.0 / (dt32 * dt31) + mu / (12.0 * np.linalg.norm(r3) ** 3)) * r3
    state = np.hstack((r2, term1 + term2 + term3))
    _require_finite_state(state)
    geometry = {**geometry, "dt21_s": dt21, "dt32_s": dt32, "arc_s": dt31}
    return IODHypothesis(
        hypothesis_id="herrick_gibbs:000",
        method="herrick_gibbs",
        epoch_time_s=float(times[1]),
        state_eci_km_s=tuple(float(value) for value in state),
        source_observation_ids=tuple(str(item) for item in observation_ids),
        geometry=geometry,
        provenance={"mu_km3_s2": mu, "velocity_epoch": "middle_position"},
    )


def solve_lambert_position_iod(
    positions_eci_km: Sequence[Sequence[float]],
    times_s: Sequence[float],
    *,
    observation_ids: Sequence[str] = ("r1", "r2", "r3"),
    short_way: bool = True,
    mu_km3_s2: float = EARTH_MU_KM3_S2,
) -> IODHypothesis:
    """Use the checked-in Lambert solver to form a middle-epoch IOD hypothesis."""

    r1, _r2, r3 = _position_triplet(positions_eci_km)
    times = np.asarray(times_s, dtype=float).reshape(3)
    if not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0.0):
        raise ValueError("Lambert position IOD times must be finite and strictly increasing.")
    solution = solve_lambert_universal_variable(
        r1,
        r3,
        float(times[2] - times[0]),
        mu_km3_s2=float(mu_km3_s2),
        short_way=bool(short_way),
    )
    if not solution.converged:
        raise ValueError("Lambert position IOD did not converge.")
    state = _propagate_state(
        np.hstack((r1, solution.v1_km_s)),
        float(times[1] - times[0]),
        mu_km3_s2=float(mu_km3_s2),
    )
    return IODHypothesis(
        hypothesis_id=f"lambert_position:{'short' if short_way else 'long'}",
        method="lambert_position",
        epoch_time_s=float(times[1]),
        state_eci_km_s=tuple(float(value) for value in state),
        source_observation_ids=tuple(str(item) for item in observation_ids),
        geometry={**_position_geometry(r1, np.asarray(positions_eci_km)[1], r3), "short_way": bool(short_way)},
        provenance={"mu_km3_s2": float(mu_km3_s2), "lambert_residual_s": solution.residual_s},
    )


def solve_gauss_angles_only_iod(
    observations: Sequence[Mapping[str, Any]],
    *,
    mu_km3_s2: float = EARTH_MU_KM3_S2,
    min_radius_km: float = EARTH_RADIUS_KM + 100.0,
    max_radius_km: float = 100000.0,
    determinant_tolerance: float = 1.0e-10,
) -> list[IODHypothesis]:
    """Classical three-observation Gauss angles-only IOD with all physical roots retained."""

    if len(observations) != 3:
        raise ValueError("Gauss angles-only IOD requires exactly three observations.")
    rows = [dict(item) for item in observations]
    t = np.array([float(row["time_s"]) for row in rows], dtype=float)
    if not np.all(np.isfinite(t)) or np.any(np.diff(t) <= 0.0):
        raise ValueError("Gauss angles-only observation epochs must be finite and strictly increasing.")
    los = [np.asarray(row["los_eci"], dtype=float).reshape(3) for row in rows]
    observers = [np.asarray(row["observer_eci_km"], dtype=float).reshape(3) for row in rows]
    l1, l2, l3 = (item / np.linalg.norm(item) for item in los)
    rsite1, rsite2, rsite3 = observers
    tau1 = float(t[0] - t[1])
    tau3 = float(t[2] - t[1])
    tau = tau3 - tau1
    p1 = np.cross(l2, l3)
    p2 = np.cross(l1, l3)
    p3 = np.cross(l1, l2)
    d0 = float(np.dot(l1, p1))
    if abs(d0) <= float(determinant_tolerance):
        raise ValueError("Gauss angles-only geometry is singular or nearly coplanar in line-of-sight space.")
    d = np.array(
        [
            [np.dot(rsite1, p1), np.dot(rsite1, p2), np.dot(rsite1, p3)],
            [np.dot(rsite2, p1), np.dot(rsite2, p2), np.dot(rsite2, p3)],
            [np.dot(rsite3, p1), np.dot(rsite3, p2), np.dot(rsite3, p3)],
        ],
        dtype=float,
    )
    a_term = (-d[0, 1] * tau3 / tau + d[1, 1] + d[2, 1] * tau1 / tau) / d0
    b_term = (
        d[0, 1] * (tau3 * tau3 - tau * tau) * tau3 / tau
        + d[2, 1] * (tau * tau - tau1 * tau1) * tau1 / tau
    ) / (6.0 * d0)
    e_term = float(np.dot(rsite2, l2))
    mu = float(mu_km3_s2)
    polynomial = np.zeros(9, dtype=float)
    polynomial[0] = 1.0
    polynomial[2] = -(a_term * a_term + 2.0 * a_term * e_term + np.dot(rsite2, rsite2))
    polynomial[5] = -2.0 * mu * b_term * (a_term + e_term)
    polynomial[8] = -(mu * b_term) ** 2
    roots = np.roots(polynomial)
    real_roots = sorted(
        {
            round(float(root.real), 9)
            for root in roots
            if abs(float(root.imag)) <= 1.0e-7 * max(1.0, abs(float(root.real)))
            and float(min_radius_km) <= float(root.real) <= float(max_radius_km)
        }
    )
    hypotheses: list[IODHypothesis] = []
    for root_index, radius2 in enumerate(real_roots):
        f1 = 1.0 - 0.5 * mu * tau1 * tau1 / radius2**3
        f3 = 1.0 - 0.5 * mu * tau3 * tau3 / radius2**3
        g1 = tau1 - mu * tau1**3 / (6.0 * radius2**3)
        g3 = tau3 - mu * tau3**3 / (6.0 * radius2**3)
        denominator = f1 * g3 - f3 * g1
        if abs(denominator) <= 1.0e-12:
            continue
        radius_cubed = radius2**3
        rho1_denominator = 6.0 * radius_cubed + mu * (tau * tau - tau3 * tau3)
        rho3_denominator = 6.0 * radius_cubed + mu * (tau * tau - tau1 * tau1)
        rho1 = (
            (
                6.0 * (d[2, 0] * tau1 / tau3 + d[1, 0] * tau / tau3) * radius_cubed
                + mu * d[2, 0] * (tau * tau - tau1 * tau1) * tau1 / tau3
            )
            / rho1_denominator
            - d[0, 0]
        ) / d0
        rho2 = a_term + mu * b_term / radius_cubed
        rho3 = (
            (
                6.0 * (d[0, 2] * tau3 / tau1 - d[1, 2] * tau / tau1) * radius_cubed
                + mu * d[0, 2] * (tau * tau - tau3 * tau3) * tau3 / tau1
            )
            / rho3_denominator
            - d[2, 2]
        ) / d0
        if min(rho1, rho2, rho3) <= 0.0:
            continue
        r1 = rsite1 + rho1 * l1
        r2 = rsite2 + rho2 * l2
        r3 = rsite3 + rho3 * l3
        v2 = (-f3 * r1 + f1 * r3) / denominator
        state = np.hstack((r2, v2))
        if not np.all(np.isfinite(state)):
            continue
        hypotheses.append(
            IODHypothesis(
                hypothesis_id=f"gauss:{root_index:03d}",
                method="gauss_angles_only",
                epoch_time_s=float(t[1]),
                state_eci_km_s=tuple(float(value) for value in state),
                source_observation_ids=tuple(str(row["measurement_id"]) for row in rows),
                geometry={
                    "d0": d0,
                    "tau1_s": tau1,
                    "tau3_s": tau3,
                    "polynomial_root_radius_km": radius2,
                    "slant_ranges_km": [rho1, rho2, rho3],
                },
                provenance={"mu_km3_s2": mu, "series": "first_order_f_g"},
            )
        )
    return hypotheses


def generate_constrained_lambert_angle_hypotheses(
    observations: Sequence[Mapping[str, Any]],
    *,
    radius_grid_km: Sequence[float],
    mu_km3_s2: float = EARTH_MU_KM3_S2,
) -> list[IODHypothesis]:
    """Generate bounded range hypotheses and connect endpoint positions with Lambert."""

    if len(observations) != 3:
        raise ValueError("Constrained Lambert angle initialization requires exactly three observations.")
    rows = [dict(item) for item in observations]
    times = np.array([float(row["time_s"]) for row in rows], dtype=float)
    if not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0.0):
        raise ValueError("Constrained Lambert angle epochs must be finite and strictly increasing.")
    hypotheses: list[IODHypothesis] = []
    for radius in sorted({float(item) for item in radius_grid_km}):
        if not np.isfinite(radius) or radius <= EARTH_RADIUS_KM:
            continue
        positions: list[np.ndarray] = []
        ranges: list[float] = []
        for row in rows:
            observer = np.asarray(row["observer_eci_km"], dtype=float).reshape(3)
            los = np.asarray(row["los_eci"], dtype=float).reshape(3)
            rho = _positive_range_for_radius(observer, los, radius)
            if rho is None:
                positions = []
                break
            positions.append(observer + rho * los)
            ranges.append(rho)
        if len(positions) != 3:
            continue
        for short_way in (True, False):
            try:
                solution = solve_lambert_universal_variable(
                    positions[0],
                    positions[2],
                    float(times[2] - times[0]),
                    mu_km3_s2=float(mu_km3_s2),
                    short_way=short_way,
                )
                if not solution.converged:
                    continue
                state = _propagate_state(
                    np.hstack((positions[0], solution.v1_km_s)),
                    float(times[1] - times[0]),
                    mu_km3_s2=float(mu_km3_s2),
                )
            except ValueError:
                continue
            hypotheses.append(
                IODHypothesis(
                    hypothesis_id=f"lambert_angle:{radius:.3f}:{'short' if short_way else 'long'}",
                    method="constrained_lambert_angles",
                    epoch_time_s=float(times[1]),
                    state_eci_km_s=tuple(float(value) for value in state),
                    source_observation_ids=tuple(str(row["measurement_id"]) for row in rows),
                    geometry={
                        "assumed_geocentric_radius_km": radius,
                        "slant_ranges_km": ranges,
                        "short_way": short_way,
                    },
                    provenance={"mu_km3_s2": float(mu_km3_s2), "range_constraint": "radius_grid"},
                )
            )
    return hypotheses


def run_angles_only_iod(
    packet: Mapping[str, Any],
    *,
    output_dir: str | Path,
    fit_observation_count: int | None = None,
    radius_grid_km: Sequence[float] = (6678.0, 7000.0, 7378.0, 8378.0, 26560.0, 42164.0),
    min_arc_s: float = 30.0,
    min_los_span_deg: float = 0.25,
    ambiguity_score_delta: float = 0.25,
    max_hypotheses_to_refine: int = 8,
    max_refinement_nfev: int = 48,
    robust_loss: str = "huber",
    sigma_clip_threshold: float | None = 6.0,
    numeric_backend: str = "rust",
) -> dict[str, Any]:
    """Run ambiguity-aware Gauss/Lambert angles-only IOD and batch refinement."""

    numeric_backend = normalize_numeric_backend(numeric_backend, error_message="numeric_backend must be 'python' or 'rust'.")

    normalized = validate_optical_angle_packet(packet)
    rows = list(normalized["observations"])
    count = len(rows)
    fit_count = int(fit_observation_count or max(3, count - max(1, count // 3)))
    if fit_count < 3 or fit_count >= count:
        raise ValueError("Angles-only IOD requires at least three fit rows and one withheld row.")
    fit_rows = rows[:fit_count]
    holdout_rows = rows[fit_count:]
    arc_s = float(fit_rows[-1]["time_s"] - fit_rows[0]["time_s"])
    los_span = _los_span_deg(fit_rows)
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    if arc_s < float(min_arc_s) or los_span < float(min_los_span_deg):
        report = _insufficient_report(
            normalized,
            output=output,
            reason="short_arc_or_insufficient_angular_motion",
            geometry={"fit_arc_s": arc_s, "los_span_deg": los_span},
            numeric_backend=numeric_backend,
        )
        _write_iod_artifacts(output, report, [])
        return report

    triplets = _select_angle_triplets(fit_rows)
    hypotheses: list[IODHypothesis] = []
    method_failures: list[dict[str, Any]] = []
    for triplet_index, triplet in enumerate(triplets):
        try:
            gauss = solve_gauss_angles_only_iod(triplet)
            hypotheses.extend(_renumber(gauss, prefix=f"gauss_t{triplet_index}"))
        except ValueError as exc:
            method_failures.append({"method": "gauss_angles_only", "triplet": triplet_index, "reason": str(exc)})
        lambert = generate_constrained_lambert_angle_hypotheses(triplet, radius_grid_km=radius_grid_km)
        hypotheses.extend(_renumber(lambert, prefix=f"lambert_t{triplet_index}"))
    hypotheses = _deduplicate_hypotheses(hypotheses)
    if not hypotheses:
        report = _insufficient_report(
            normalized,
            output=output,
            reason="no_physical_gauss_or_lambert_hypotheses",
            geometry={"fit_arc_s": arc_s, "los_span_deg": los_span, "method_failures": method_failures},
            numeric_backend=numeric_backend,
        )
        _write_iod_artifacts(output, report, [])
        return report

    refinement_limit = int(max_hypotheses_to_refine)
    if refinement_limit <= 0:
        raise ValueError("max_hypotheses_to_refine must be positive.")
    ranked_seeds: list[tuple[float, IODHypothesis, dict[str, Any]]] = []
    for hypothesis in hypotheses:
        try:
            seed_metrics = _angle_metrics(hypothesis, fit_rows=fit_rows, holdout_rows=holdout_rows)
            seed_physical = _physical_state_diagnostics(
                np.asarray(hypothesis.state_eci_km_s),
                mu_km3_s2=EARTH_MU_KM3_S2,
            )
            seed_score = _hypothesis_score(seed_metrics, physical=seed_physical, solver_success=True)
            ranked_seeds.append((seed_score, hypothesis, seed_metrics))
        except (ValueError, FloatingPointError) as exc:
            method_failures.append(
                {"method": hypothesis.method, "hypothesis_id": hypothesis.hypothesis_id, "reason": str(exc)}
            )
    ranked_seeds.sort(key=lambda item: (item[0], item[1].hypothesis_id))
    evaluated = []
    for _seed_score, hypothesis, seed_metrics in ranked_seeds[:refinement_limit]:
        try:
            refined, refinement = _refine_angle_hypothesis(
                hypothesis,
                fit_rows,
                max_nfev=max_refinement_nfev,
                robust_loss=robust_loss,
                sigma_clip_threshold=sigma_clip_threshold,
                numeric_backend=numeric_backend,
            )
            metrics = _angle_metrics(refined, fit_rows=fit_rows, holdout_rows=holdout_rows)
            physical = _physical_state_diagnostics(
                np.asarray(refined.state_eci_km_s),
                mu_km3_s2=EARTH_MU_KM3_S2,
            )
            score = _hypothesis_score(metrics, physical=physical, solver_success=bool(refinement["success"]))
            evaluated.append(
                {
                    **refined.to_dict(),
                    "seed_hypothesis_id": hypothesis.hypothesis_id,
                    "seed_metrics": seed_metrics,
                    "fit_metrics": metrics["fit"],
                    "holdout_metrics": metrics["holdout"],
                    "physical_diagnostics": physical,
                    "refinement": refinement,
                    "score": score,
                    "eligible": bool(
                        physical["bound_earth_orbit"]
                        and physical["perigee_above_earth"]
                        and refinement["success"]
                    ),
                }
            )
        except (ValueError, FloatingPointError) as exc:
            method_failures.append(
                {"method": "batch_refinement", "hypothesis_id": hypothesis.hypothesis_id, "reason": str(exc)}
            )
    evaluated = _deduplicate_refined_hypotheses(evaluated)
    evaluated.sort(key=lambda item: (not bool(item["eligible"]), float(item["score"]), str(item["hypothesis_id"])))
    eligible = [item for item in evaluated if bool(item["eligible"])]
    if not eligible:
        status = "insufficiently_constrained"
        selected_id = None
    elif len(eligible) > 1 and float(eligible[1]["score"] - eligible[0]["score"]) <= float(ambiguity_score_delta):
        status = "ambiguous"
        selected_id = None
    else:
        status = "unique_solution"
        selected_id = str(eligible[0]["hypothesis_id"])
    selected = next((item for item in eligible if item["hypothesis_id"] == selected_id), None)
    report = {
        "schema_version": IOD_RESULT_SCHEMA_VERSION,
        "kind": "oel.initial_orbit_determination_result",
        "method_family": "angles_only",
        "status": status,
        "selected_hypothesis_id": selected_id,
        "selected_hypothesis": selected,
        "hypothesis_count": len(evaluated),
        "generated_hypothesis_count": len(hypotheses),
        "refinement_limit": refinement_limit,
        "eligible_hypothesis_count": len(eligible),
        "hypotheses": evaluated,
        "observation_contract": OPTICAL_ANGLE_CONTRACT,
        "observation_partition": {
            "fit_count": len(fit_rows),
            "holdout_count": len(holdout_rows),
            "fit_boundary": "first fit_observation_count rows",
        },
        "geometry": {"fit_arc_s": arc_s, "los_span_deg": los_span, "triplet_count": len(triplets)},
        "method_failures": method_failures,
        "ambiguity_policy": {
            "score_delta": float(ambiguity_score_delta),
            "unique_solution_requires": "one eligible hypothesis separated from the runner-up by more than score_delta",
        },
        "quality_gates": {
            "withheld_observations_evaluated": bool(holdout_rows),
            "false_precision_prevented": status != "unique_solution" or selected is not None,
            "physical_constraints_applied": True,
            "batch_refinement_executed": True,
            "evidence_status": "ready_with_caveats" if status == "unique_solution" else "review_required",
        },
        "claim_boundary": "L1-L2 synthetic angles-only IOD evidence; no association, custody, or operational optical calibration claim",
        "non_claims": [
            "Angles-only geometry does not directly observe range.",
            "A unique numerical minimum is not proof of correct object association.",
            "Covariance from this short-arc batch fit is not calibrated uncertainty.",
        ],
    }
    if numeric_backend == "rust":
        report["numeric_backend"] = numeric_backend
    _write_iod_artifacts(output, report, evaluated)
    if selected is not None:
        state = np.asarray(selected["state_eci_km_s"], dtype=float)
        epoch_jd = float(normalized["initial_jd_utc"]) + float(selected["epoch_time_s"]) / 86400.0
        packet_path = output / "iod_mission_input_packet.json"
        ingest_state_vector(
            object_id=str(normalized["object_id"]),
            position=state[:3],
            velocity=state[3:],
            epoch_jd_utc=epoch_jd,
            source_label="oel_phase5_angles_only_iod",
        ).write_json(packet_path)
        report["artifacts"]["mission_input_packet"] = str(packet_path)
        (output / "iod_report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return report


def evaluate_position_triplet_iod(
    positions_eci_km: Sequence[Sequence[float]],
    times_s: Sequence[float],
    *,
    observation_ids: Sequence[str] = ("r1", "r2", "r3"),
) -> dict[str, Any]:
    """Evaluate all applicable position-triplet methods without forcing a winner."""

    times = np.asarray(times_s, dtype=float).reshape(3)
    if not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0.0):
        raise ValueError("Position-triplet IOD times must be finite and strictly increasing.")
    methods: list[IODHypothesis] = []
    failures: list[dict[str, str]] = []
    calls = (
        ("gibbs", lambda: solve_gibbs_iod(positions_eci_km, observation_ids=observation_ids, epoch_time_s=float(times[1]))),
        ("herrick_gibbs", lambda: solve_herrick_gibbs_iod(positions_eci_km, times_s, observation_ids=observation_ids)),
        ("lambert_short", lambda: solve_lambert_position_iod(positions_eci_km, times_s, observation_ids=observation_ids, short_way=True)),
        ("lambert_long", lambda: solve_lambert_position_iod(positions_eci_km, times_s, observation_ids=observation_ids, short_way=False)),
    )
    for name, call in calls:
        try:
            methods.append(call())
        except ValueError as exc:
            failures.append({"method": name, "reason": str(exc)})
    return {
        "schema_version": IOD_RESULT_SCHEMA_VERSION,
        "kind": "oel.position_triplet_iod_candidates",
        "status": "candidates_generated" if methods else "insufficiently_constrained",
        "hypotheses": [item.to_dict() for item in methods],
        "failures": failures,
        "claim_boundary": "range-supported position-triplet IOD; separate from angles-only claims",
    }


def _refine_angle_hypothesis(
    hypothesis: IODHypothesis,
    fit_rows: Sequence[Mapping[str, Any]],
    *,
    max_nfev: int,
    robust_loss: str,
    sigma_clip_threshold: float | None,
    numeric_backend: str = "rust",
) -> tuple[IODHypothesis, dict[str, Any]]:
    seed = np.asarray(hypothesis.state_eci_km_s, dtype=float)
    parameters = ParameterSet(
        [
            EstimatedParameter("x_km", seed[0], scale=10.0, lower=-150000.0, upper=150000.0),
            EstimatedParameter("y_km", seed[1], scale=10.0, lower=-150000.0, upper=150000.0),
            EstimatedParameter("z_km", seed[2], scale=10.0, lower=-150000.0, upper=150000.0),
            EstimatedParameter("vx_km_s", seed[3], scale=0.01, lower=-20.0, upper=20.0),
            EstimatedParameter("vy_km_s", seed[4], scale=0.01, lower=-20.0, upper=20.0),
            EstimatedParameter("vz_km_s", seed[5], scale=0.01, lower=-20.0, upper=20.0),
        ]
    )

    def residual(state: np.ndarray) -> np.ndarray:
        try:
            return _angle_residual_vector(
                state,
                epoch_time_s=hypothesis.epoch_time_s,
                rows=fit_rows,
                numeric_backend=numeric_backend,
            )
        except (ValueError, FloatingPointError):
            return np.full(2 * len(fit_rows), 1.0e12, dtype=float)

    solve = solve_batch_least_squares(
        parameters,
        residual,
        max_nfev=int(max_nfev),
        robust_loss=robust_loss,
        robust_f_scale=2.0,
        sigma_clip_threshold=sigma_clip_threshold,
    )
    state = np.asarray(solve.x_native, dtype=float)
    refined = IODHypothesis(
        hypothesis_id=f"refined:{hypothesis.hypothesis_id}",
        method=f"{hypothesis.method}+batch_two_body_angles",
        epoch_time_s=hypothesis.epoch_time_s,
        state_eci_km_s=tuple(float(value) for value in state),
        source_observation_ids=hypothesis.source_observation_ids,
        geometry=hypothesis.geometry,
        provenance={**dict(hypothesis.provenance), "refined_from": hypothesis.hypothesis_id},
    )
    return refined, {
        "success": bool(solve.success),
        "message": solve.message,
        "nfev": int(solve.nfev),
        "initial_cost": float(solve.initial_cost),
        "final_cost": float(solve.cost),
        "rms_weighted_residual": float(solve.rms_residual),
        "diagnostics": solve.diagnostics,
        "decision_records": solve.decision_records,
    }


def _angle_residual_vector(
    state: np.ndarray,
    *,
    epoch_time_s: float,
    rows: Sequence[Mapping[str, Any]],
    numeric_backend: str = "rust",
) -> np.ndarray:
    relative_positions = []
    residual_rows: list[np.ndarray] = []
    covariance_blocks: list[np.ndarray] = []
    native_states = None
    if rows and normalize_numeric_backend(numeric_backend) == "rust":
        from sim.rust_estimation_backend import try_two_body_observation_states

        native_states = try_two_body_observation_states(
            state,
            [float(row["time_s"]) - float(epoch_time_s) for row in rows],
            mu_km3_s2=EARTH_MU_KM3_S2,
        )
    for index, row in enumerate(rows):
        predicted_state = (
            native_states[index]
            if native_states is not None
            else _propagate_state(
                state,
                float(row["time_s"]) - float(epoch_time_s),
                mu_km3_s2=EARTH_MU_KM3_S2,
            )
        )
        observer = np.asarray(row["observer_eci_km"], dtype=float)
        relative_position = predicted_state[:3] - observer
        relative_positions.append(relative_position)
        ra_pred, dec_pred = los_to_ra_dec(relative_position)
        residual = _ra_dec_coordinate_residual_deg(
            ra_pred,
            dec_pred,
            observed_ra_deg=float(row["ra_deg"]),
            observed_dec_deg=float(row["dec_deg"]),
        )
        residual_rows.append(residual)
        covariance_blocks.append(np.asarray(row["angular_covariance_deg2"], dtype=float))
    if not residual_rows:
        return np.zeros(0, dtype=float)
    if normalize_numeric_backend(numeric_backend) == "rust":
        # Keep the native optical geometry path available for callers that
        # need predicted angles as well.  The residual still uses the same
        # wrapped Python contract before whitening.
        from sim.rust_estimation_backend import measurement_whiten_variable_rows, optical_radec_predictions

        predictions = optical_radec_predictions(np.asarray(relative_positions, dtype=float))
        native_rows = [
            _ra_dec_coordinate_residual_deg(
                float(predictions[index, 0]),
                float(predictions[index, 1]),
                observed_ra_deg=float(row["ra_deg"]),
                observed_dec_deg=float(row["dec_deg"]),
            )
            for index, row in enumerate(rows)
        ]
        native_factors = [
            prepare_covariance_whitener(
                covariance,
                dimension=residual.size,
                field_name=f"optical observation {rows[index]['measurement_id']} covariance",
            )
            for index, (residual, covariance) in enumerate(zip(native_rows, covariance_blocks, strict=True))
        ]
        return measurement_whiten_variable_rows(native_rows, native_factors)
    return np.concatenate(
        [
            whiten_residual_block(
                residual,
                covariance,
                field_name=f"optical observation {rows[index]['measurement_id']} covariance",
            )
            for index, (residual, covariance) in enumerate(zip(residual_rows, covariance_blocks, strict=True))
        ]
    )


def _angle_metrics(
    hypothesis: IODHypothesis,
    *,
    fit_rows: Sequence[Mapping[str, Any]],
    holdout_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "fit": _angle_partition_metrics(hypothesis, fit_rows),
        "holdout": _angle_partition_metrics(hypothesis, holdout_rows),
    }


def _angle_partition_metrics(hypothesis: IODHypothesis, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"status": "not_evaluated", "count": 0}
    state = np.asarray(hypothesis.state_eci_km_s, dtype=float)
    errors_arcsec = []
    normalized = []
    for row in rows:
        predicted = _propagate_state(
            state,
            float(row["time_s"]) - float(hypothesis.epoch_time_s),
            mu_km3_s2=EARTH_MU_KM3_S2,
        )
        ra_pred, dec_pred = los_to_ra_dec(predicted[:3] - np.asarray(row["observer_eci_km"], dtype=float))
        coordinate_residual_deg = _ra_dec_coordinate_residual_deg(
            ra_pred,
            dec_pred,
            observed_ra_deg=float(row["ra_deg"]),
            observed_dec_deg=float(row["dec_deg"]),
        )
        # Great-circle small-angle reporting uses dRA*cos(dec), while
        # statistical whitening stays in the declared RA/Dec coordinate basis.
        # Applying cos(dec) to the residual but not its covariance (the former
        # behavior) over-weighted/under-weighted RA depending on declination.
        tangent_residual_deg = coordinate_residual_deg.copy()
        tangent_residual_deg[0] *= np.cos(np.deg2rad(float(row["dec_deg"])))
        errors_arcsec.append(float(np.linalg.norm(tangent_residual_deg) * 3600.0))
        normalized.extend(
            whiten_residual_block(
                coordinate_residual_deg,
                row["angular_covariance_deg2"],
                field_name=f"optical observation {row['measurement_id']} covariance",
            ).tolist()
        )
    errors = np.asarray(errors_arcsec, dtype=float)
    norm = np.asarray(normalized, dtype=float)
    return {
        "status": "evaluated",
        "count": len(rows),
        "angular_rms_arcsec": float(np.sqrt(np.mean(errors * errors))),
        "angular_max_arcsec": float(np.max(errors)),
        "normalized_rms": float(np.sqrt(np.mean(norm * norm))),
    }


def _hypothesis_score(metrics: Mapping[str, Any], *, physical: Mapping[str, Any], solver_success: bool) -> float:
    fit = float(dict(metrics["fit"]).get("normalized_rms", 1.0e6))
    holdout = float(dict(metrics["holdout"]).get("normalized_rms", 1.0e6))
    penalty = 0.0
    if not solver_success:
        penalty += 1.0e5
    if not bool(physical["bound_earth_orbit"]):
        penalty += 1.0e5
    if not bool(physical["perigee_above_earth"]):
        penalty += 1.0e5
    return fit + 2.0 * holdout + penalty


def _physical_state_diagnostics(state: np.ndarray, *, mu_km3_s2: float) -> dict[str, Any]:
    r = np.asarray(state[:3], dtype=float)
    v = np.asarray(state[3:6], dtype=float)
    rnorm = float(np.linalg.norm(r))
    vnorm = float(np.linalg.norm(v))
    energy = 0.5 * vnorm * vnorm - float(mu_km3_s2) / rnorm
    h = np.cross(r, v)
    evec = np.cross(v, h) / float(mu_km3_s2) - r / rnorm
    eccentricity = float(np.linalg.norm(evec))
    semi_major = -float(mu_km3_s2) / (2.0 * energy) if abs(energy) > 1.0e-14 else float("inf")
    perigee = semi_major * (1.0 - eccentricity) if semi_major > 0.0 else float("nan")
    return {
        "radius_km": rnorm,
        "speed_km_s": vnorm,
        "specific_energy_km2_s2": energy,
        "semi_major_axis_km": semi_major,
        "eccentricity": eccentricity,
        "perigee_radius_km": perigee,
        "bound_earth_orbit": bool(energy < 0.0 and eccentricity < 1.0),
        "perigee_above_earth": bool(np.isfinite(perigee) and perigee > EARTH_RADIUS_KM),
    }


def _position_triplet(values: Sequence[Sequence[float]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    array = np.asarray(values, dtype=float)
    if array.shape != (3, 3) or not np.all(np.isfinite(array)):
        raise ValueError("Position-triplet IOD requires a finite array with shape (3, 3).")
    if np.any(np.linalg.norm(array, axis=1) <= 0.0):
        raise ValueError("Position-triplet vectors must be nonzero.")
    return array[0], array[1], array[2]


def _position_geometry(r1: np.ndarray, r2: np.ndarray, r3: np.ndarray) -> dict[str, float]:
    separations = [_vector_angle_deg(r1, r2), _vector_angle_deg(r2, r3), _vector_angle_deg(r1, r3)]
    normal = np.cross(r2, r3)
    normal_norm = float(np.linalg.norm(normal))
    coplanarity = (
        90.0
        if normal_norm <= 1.0e-15
        else float(np.rad2deg(np.arcsin(np.clip(abs(np.dot(r1 / np.linalg.norm(r1), normal / normal_norm)), 0.0, 1.0))))
    )
    return {
        "coplanarity_deg": coplanarity,
        "separation_12_deg": separations[0],
        "separation_23_deg": separations[1],
        "separation_13_deg": separations[2],
        "minimum_separation_deg": min(separations[0], separations[1]),
        "maximum_separation_deg": max(separations),
    }


def _propagate_state(state: Sequence[float], dt_s: float, *, mu_km3_s2: float, max_step_s: float = 30.0) -> np.ndarray:
    x = np.asarray(state, dtype=float).reshape(6).copy()
    duration = float(dt_s)
    if duration == 0.0:
        return x
    steps = max(1, int(math.ceil(abs(duration) / float(max_step_s))))
    step = duration / steps
    zero = np.zeros(3, dtype=float)
    for _ in range(steps):
        x = propagate_two_body_rk4(x, step, float(mu_km3_s2), zero)
        if not np.all(np.isfinite(x)) or float(np.linalg.norm(x[:3])) <= EARTH_RADIUS_KM * 0.5:
            raise ValueError("IOD hypothesis propagation failed or entered a nonphysical radius.")
    return x


def _positive_range_for_radius(observer: np.ndarray, los: np.ndarray, radius_km: float) -> float | None:
    lhat = los / np.linalg.norm(los)
    projection = float(np.dot(observer, lhat))
    discriminant = projection * projection - (float(np.dot(observer, observer)) - radius_km * radius_km)
    if discriminant < 0.0:
        return None
    roots = [-projection + math.sqrt(discriminant), -projection - math.sqrt(discriminant)]
    positive = [value for value in roots if value > 0.0]
    return min(positive) if positive else None


def _vector_angle_deg(left: np.ndarray, right: np.ndarray) -> float:
    cosine = float(np.dot(left, right) / (np.linalg.norm(left) * np.linalg.norm(right)))
    return float(np.rad2deg(np.arccos(np.clip(cosine, -1.0, 1.0))))


def _los_span_deg(rows: Sequence[Mapping[str, Any]]) -> float:
    first = np.asarray(rows[0]["los_eci"], dtype=float)
    return max(_vector_angle_deg(first, np.asarray(row["los_eci"], dtype=float)) for row in rows[1:])


def _select_angle_triplets(rows: Sequence[Mapping[str, Any]]) -> list[list[dict[str, Any]]]:
    count = len(rows)
    indices = {(0, count // 2, count - 1)}
    if count >= 5:
        indices.add((0, 1, count - 1))
        indices.add((0, count - 2, count - 1))
    return [[dict(rows[index]) for index in triplet] for triplet in sorted(indices)]


def _renumber(hypotheses: Sequence[IODHypothesis], *, prefix: str) -> list[IODHypothesis]:
    return [
        IODHypothesis(
            hypothesis_id=f"{prefix}:{index:03d}",
            method=item.method,
            epoch_time_s=item.epoch_time_s,
            state_eci_km_s=item.state_eci_km_s,
            source_observation_ids=item.source_observation_ids,
            geometry=item.geometry,
            provenance=item.provenance,
        )
        for index, item in enumerate(hypotheses)
    ]


def _deduplicate_hypotheses(hypotheses: Sequence[IODHypothesis]) -> list[IODHypothesis]:
    kept: list[IODHypothesis] = []
    for item in hypotheses:
        state = np.asarray(item.state_eci_km_s, dtype=float)
        if any(np.linalg.norm(state - np.asarray(other.state_eci_km_s, dtype=float)) < 1.0e-4 for other in kept):
            continue
        kept.append(item)
    return kept


def _deduplicate_refined_hypotheses(hypotheses: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep one representative for each converged orbit basin."""

    kept: list[dict[str, Any]] = []
    ordered = sorted(hypotheses, key=lambda value: (float(value["score"]), str(value["hypothesis_id"])))
    for item in ordered:
        state = _propagate_state(
            item["state_eci_km_s"],
            -float(item["epoch_time_s"]),
            mu_km3_s2=EARTH_MU_KM3_S2,
        )
        duplicate = next(
            (
                other
                for other in kept
                if np.linalg.norm(
                    state[:3]
                    - _propagate_state(
                        other["state_eci_km_s"],
                        -float(other["epoch_time_s"]),
                        mu_km3_s2=EARTH_MU_KM3_S2,
                    )[:3]
                )
                < 1.0
                and np.linalg.norm(
                    state[3:]
                    - _propagate_state(
                        other["state_eci_km_s"],
                        -float(other["epoch_time_s"]),
                        mu_km3_s2=EARTH_MU_KM3_S2,
                    )[3:]
                )
                < 1.0e-3
            ),
            None,
        )
        if duplicate is None:
            item["converged_seed_ids"] = [item["seed_hypothesis_id"]]
            kept.append(item)
        else:
            duplicate.setdefault("converged_seed_ids", []).append(item["seed_hypothesis_id"])
    return kept


def _require_finite_state(state: np.ndarray) -> None:
    if state.shape != (6,) or not np.all(np.isfinite(state)):
        raise ValueError("IOD method produced a non-finite state.")


def _wrap_angle_deg(value: float) -> float:
    return (float(value) + 180.0) % 360.0 - 180.0


def _ra_dec_coordinate_residual_deg(
    predicted_ra_deg: float,
    predicted_dec_deg: float,
    *,
    observed_ra_deg: float,
    observed_dec_deg: float,
) -> np.ndarray:
    """Return a wrapped residual in the packet's declared RA/Dec basis."""

    return np.array(
        [
            _wrap_angle_deg(float(predicted_ra_deg) - float(observed_ra_deg)),
            float(predicted_dec_deg) - float(observed_dec_deg),
        ],
        dtype=float,
    )


def _insufficient_report(
    packet: Mapping[str, Any],
    *,
    output: Path,
    reason: str,
    geometry: Mapping[str, Any],
    numeric_backend: str = "rust",
) -> dict[str, Any]:
    report = {
        "schema_version": IOD_RESULT_SCHEMA_VERSION,
        "kind": "oel.initial_orbit_determination_result",
        "method_family": "angles_only",
        "status": "insufficiently_constrained",
        "selected_hypothesis_id": None,
        "selected_hypothesis": None,
        "hypothesis_count": 0,
        "eligible_hypothesis_count": 0,
        "hypotheses": [],
        "observation_contract": packet["measurement_contract"],
        "geometry": dict(geometry),
        "failure": {"classification": "insufficient_geometry", "reason": reason},
        "quality_gates": {
            "false_precision_prevented": True,
            "evidence_status": "review_required",
        },
        "claim_boundary": "No unique orbit is emitted when angles-only geometry is insufficient.",
        "artifacts": {"report_json": str(output / "iod_report.json")},
    }
    if normalize_numeric_backend(numeric_backend) == "rust":
        report["numeric_backend"] = "rust"
    return report


def _write_iod_artifacts(output: Path, report: dict[str, Any], hypotheses: Sequence[Mapping[str, Any]]) -> None:
    report_path = output / "iod_report.json"
    csv_path = output / "iod_hypotheses.csv"
    review_path = output / "review/run.sqlite"
    report.setdefault("artifacts", {})
    report["artifacts"].update(
        {"report_json": str(report_path), "hypotheses_csv": str(csv_path), "review_sqlite": str(review_path)}
    )
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    fields = [
        "hypothesis_id",
        "method",
        "eligible",
        "score",
        "fit_angular_rms_arcsec",
        "holdout_angular_rms_arcsec",
        "perigee_radius_km",
        "eccentricity",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for item in hypotheses:
            writer.writerow(
                {
                    "hypothesis_id": item.get("hypothesis_id"),
                    "method": item.get("method"),
                    "eligible": item.get("eligible"),
                    "score": item.get("score"),
                    "fit_angular_rms_arcsec": dict(item.get("fit_metrics", {})).get("angular_rms_arcsec"),
                    "holdout_angular_rms_arcsec": dict(item.get("holdout_metrics", {})).get("angular_rms_arcsec"),
                    "perigee_radius_km": dict(item.get("physical_diagnostics", {})).get("perigee_radius_km"),
                    "eccentricity": dict(item.get("physical_diagnostics", {})).get("eccentricity"),
                }
            )
    review_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(review_path) as connection:
        connection.executescript(
            """
            DROP TABLE IF EXISTS iod_hypotheses;
            CREATE TABLE iod_hypotheses (
                hypothesis_id TEXT, method TEXT, eligible INTEGER, score REAL,
                fit_angular_rms_arcsec REAL, holdout_angular_rms_arcsec REAL,
                perigee_radius_km REAL, eccentricity REAL, state_json TEXT
            );
            """
        )
        for item in hypotheses:
            connection.execute(
                "INSERT INTO iod_hypotheses VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    str(item.get("hypothesis_id")),
                    str(item.get("method")),
                    int(bool(item.get("eligible"))),
                    float(item.get("score", math.nan)),
                    float(dict(item.get("fit_metrics", {})).get("angular_rms_arcsec", math.nan)),
                    float(dict(item.get("holdout_metrics", {})).get("angular_rms_arcsec", math.nan)),
                    float(dict(item.get("physical_diagnostics", {})).get("perigee_radius_km", math.nan)),
                    float(dict(item.get("physical_diagnostics", {})).get("eccentricity", math.nan)),
                    json.dumps(item.get("state_eci_km_s")),
                ),
            )
