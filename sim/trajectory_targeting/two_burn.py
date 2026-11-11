"""Bounded two-impulse circular-orbit transfer using OEL event propagation."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from sim.config import load_simulation_yaml, scenario_config_from_dict
from sim.dynamics.orbit.environment import EARTH_MU_KM3_S2, EARTH_RADIUS_KM
from sim.dynamics.orbit.lambert import (
    solve_lambert_universal_variable,
    solve_lambert_universal_variable_batch,
)
from sim.dynamics.orbit.two_body import propagate_two_body_rk4
from sim.numeric_backend import normalize_numeric_backend
from sim.review import ReviewWorkspace
from sim.single_run import _SingleRunEngine
from sim.trajectory_targeting.apogee import _first_apogee
from sim.trajectory_targeting.vector import VectorTargetProblem, solve_vector_target


@dataclass(frozen=True)
class TwoBurnSpec:
    initial_radius_km: float
    target_radius_km: float
    max_burn1_m_s: float = 4000.0
    max_burn2_m_s: float = 4000.0
    radius_tolerance_km: float = 0.01
    speed_tolerance_m_s: float = 0.01
    coast_step_s: float = 30.0
    max_coast_s: float = 20_000.0
    max_iterations: int = 20

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> TwoBurnSpec:
        allowed = set(cls.__dataclass_fields__) | {"schema_version"}
        unknown = sorted(set(data) - allowed)
        if unknown:
            raise ValueError(f"Unknown two-burn fields: {', '.join(unknown)}")
        if data.get("schema_version") != "oel.trajectory_target.two_burn.v1":
            raise ValueError("Unsupported two-burn schema_version")
        values: dict[str, float | int] = {}
        for key in cls.__dataclass_fields__:
            if key not in data:
                continue
            raw = data[key]
            if isinstance(raw, bool):
                raise ValueError(f"{key} must be numeric")
            try:
                numeric = float(raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{key} must be numeric") from exc
            if key == "max_iterations":
                if not numeric.is_integer():
                    raise ValueError("max_iterations must be an integer")
                values[key] = int(numeric)
            else:
                values[key] = numeric
        return cls(**values)

    def validate(self) -> None:
        values = [float(getattr(self, name)) for name in self.__dataclass_fields__]
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Two-burn spec values must be finite")
        if not (
            EARTH_RADIUS_KM < self.initial_radius_km < self.target_radius_km <= 100_000.0
            and 0 < self.max_burn1_m_s <= 100_000.0
            and 0 < self.max_burn2_m_s <= 100_000.0
            and 0 < self.radius_tolerance_km <= 10.0
            and 0 < self.speed_tolerance_m_s <= 10.0
            and 0.1 <= self.coast_step_s <= 300.0
            and self.coast_step_s < self.max_coast_s <= 86_400.0
            and self.max_coast_s / self.coast_step_s <= 5000
            and isinstance(self.max_iterations, int) and 1 <= self.max_iterations <= 100
        ):
            raise ValueError("Two-burn spec is outside the bounded coplanar transfer envelope")


def hohmann_reference(spec: TwoBurnSpec) -> dict[str, float]:
    """Independent vis-viva seed and analytical check for circular coplanar orbits."""
    spec.validate()
    mu, r1, r2 = EARTH_MU_KM3_S2, spec.initial_radius_km, spec.target_radius_km
    a = 0.5 * (r1 + r2)
    vc1, vc2 = math.sqrt(mu / r1), math.sqrt(mu / r2)
    vt1 = math.sqrt(mu * (2 / r1 - 1 / a))
    vt2 = math.sqrt(mu * (2 / r2 - 1 / a))
    return {
        "burn1_m_s": 1000 * (vt1 - vc1),
        "burn2_m_s": 1000 * (vc2 - vt2),
        "transfer_time_s": math.pi * math.sqrt(a**3 / mu),
        "total_delta_v_m_s": 1000 * (vt1 - vc1 + vc2 - vt2),
    }


def target_two_burn(spec: TwoBurnSpec, *, numeric_backend: str = "rust") -> dict[str, Any]:
    spec.validate()
    backend = normalize_numeric_backend(numeric_backend, error_message="numeric_backend must be python or rust")
    analytical = hohmann_reference(spec)
    initial = np.array([spec.initial_radius_km, 0, 0, 0, math.sqrt(EARTH_MU_KM3_S2 / spec.initial_radius_km), 0])
    target_speed = math.sqrt(EARTH_MU_KM3_S2 / spec.target_radius_km)

    def evaluate(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        post1 = initial.copy()
        post1[4] += x[0] / 1000.0
        native_event = None
        if backend == "rust":
            from sim.rust_targeting_backend import try_first_apsis

            native_event = try_first_apsis(
                post1, kind="apogee", step_s=spec.coast_step_s,
                max_coast_s=spec.max_coast_s, mu_km3_s2=EARTH_MU_KM3_S2,
            )
        event_time, pre2 = native_event if native_event is not None else _first_apogee(
            post1, step_s=spec.coast_step_s, max_coast_s=spec.max_coast_s,
        )
        post2 = pre2.copy()
        tangent = pre2[3:] / np.linalg.norm(pre2[3:])
        post2[3:] += tangent * x[1] / 1000.0
        return post1, pre2, event_time, post2

    def observe(x: np.ndarray) -> tuple[float, float]:
        _, pre2, _, post2 = evaluate(x)
        return float(np.linalg.norm(pre2[:3])), float(np.linalg.norm(post2[3:]))

    problem = VectorTargetProblem(
        variable_names=("first_prograde_delta_v_m_s", "second_prograde_delta_v_m_s"),
        goal_names=("first_apogee_radius_km", "post_second_burn_speed_km_s"),
        initial=(min(0.8 * analytical["burn1_m_s"], spec.max_burn1_m_s),
                 min(1.1 * analytical["burn2_m_s"], spec.max_burn2_m_s)),
        lower_bounds=(0.0, 0.0),
        upper_bounds=(spec.max_burn1_m_s, spec.max_burn2_m_s),
        targets=(spec.target_radius_km, target_speed),
        tolerances=(spec.radius_tolerance_km, spec.speed_tolerance_m_s / 1000.0),
        variable_scales=(max(analytical["burn1_m_s"], 1.0), max(analytical["burn2_m_s"], 1.0)),
        max_iterations=spec.max_iterations,
        numeric_backend=backend,
    )
    correction = solve_vector_target(problem, observe)
    result: dict[str, Any] = {
        "schema_version": "oel.trajectory_target.two_burn_result.v1",
        "status": correction.status,
        "reason": correction.reason,
        "spec": asdict(spec),
        "analytical_hohmann": analytical,
        "solver": asdict(correction),
        "assumptions": {
            "model": "OEL two-body RK4 event propagation; Earth ECI; coplanar circular endpoints",
            "maneuvers": "two instantaneous prograde impulses, at departure and first apogee",
            "bounds": "nonnegative prograde burns at or below individual configured maxima",
            "optimization": "two-variable differential correction, not a global minimum-delta-v proof",
        },
    }
    if correction.status == "converged":
        x = np.array(correction.variables)
        post1, pre2, event_time, post2 = evaluate(x)
        result["burn1_delta_v_eci_m_s"] = ((post1[3:] - initial[3:]) * 1000.0).tolist()
        result["burn2_delta_v_eci_m_s"] = ((post2[3:] - pre2[3:]) * 1000.0).tolist()
        result["first_apogee_time_s"] = event_time
        result["initial_state_eci"] = initial.tolist()
        result["post_burn1_state_eci"] = post1.tolist()
        result["pre_burn2_state_eci"] = pre2.tolist()
        result["post_burn2_state_eci"] = post2.tolist()
        result["total_delta_v_m_s"] = float(sum(x))
    return result


def lambert_time_trade(
    spec: TwoBurnSpec,
    *,
    fractions: tuple[float, ...] = (0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2),
    numeric_backend: str = "rust",
) -> dict[str, Any]:
    """Sample a bounded two-impulse rendezvous trade over arrival time.

    The target starts at the Hohmann phasing angle and travels on the target
    circular orbit. This is a sampled local trade, not a continuous optimum.
    """
    reference = hohmann_reference(spec)
    backend = normalize_numeric_backend(numeric_backend, error_message="numeric_backend must be python or rust")
    r1, r2, mu = spec.initial_radius_km, spec.target_radius_km, EARTH_MU_KM3_S2
    initial_phase = math.pi - math.sqrt(mu / r2**3) * reference["transfer_time_s"]
    departure = np.array([r1, 0.0, 0.0])
    departure_v = np.array([0.0, math.sqrt(mu / r1), 0.0])
    geometry: dict[int, tuple[float, float, np.ndarray, np.ndarray, bool]] = {}
    batch_groups: dict[bool, list[tuple[int, np.ndarray, float]]] = {True: [], False: []}
    for index, fraction in enumerate(fractions):
        if not math.isfinite(fraction) or not 0.5 <= fraction <= 1.5:
            raise ValueError("Flight-time fractions must be finite and between 0.5 and 1.5")
        tof = reference["transfer_time_s"] * fraction
        phase = initial_phase + math.sqrt(mu / r2**3) * tof
        arrival = r2 * np.array([math.cos(phase), math.sin(phase), 0.0])
        arrival_v = math.sqrt(mu / r2) * np.array([-math.sin(phase), math.cos(phase), 0.0])
        singular = abs(math.sin(phase)) < 1e-10
        geometry[index] = (tof, phase, arrival, arrival_v, singular)
        if backend == "rust" and not singular:
            batch_groups[phase <= math.pi].append((index, arrival, tof))
    native_solutions: dict[int, Any] = {}
    native_errors: dict[int, str] = {}
    if backend == "rust":
        for short_way, group in batch_groups.items():
            if not group:
                continue
            indices = [row[0] for row in group]
            arrivals = np.vstack([row[1] for row in group])
            times = np.asarray([row[2] for row in group], dtype=float)
            solutions, errors = solve_lambert_universal_variable_batch(
                np.repeat(departure.reshape(1, 3), len(group), axis=0),
                arrivals,
                times,
                mu_km3_s2=mu,
                short_way=short_way,
                revolutions=0,
                backend="rust",
            )
            native_solutions.update({index: solution for index, solution in zip(indices, solutions, strict=True)})
            native_errors.update({indices[index]: message for index, message in errors})
    candidates: list[dict[str, Any]] = []
    for fraction in fractions:
        index = len(candidates)
        tof, phase, arrival, arrival_v, singular = geometry[index]
        if singular:
            # The Hohmann antipode is singular in this Lambert formulation;
            # analytical vis-viva provides the independently checkable point.
            candidates.append({"fraction": fraction, "time_of_flight_s": tof,
                               "method": "analytical_hohmann", "total_delta_v_m_s": reference["total_delta_v_m_s"],
                               "verified": True})
            continue
        try:
            if backend == "rust":
                solution = native_solutions.get(index)
                if solution is None:
                    raise ValueError(native_errors.get(index, "Lambert solver did not converge"))
            else:
                solution = solve_lambert_universal_variable(
                    departure, arrival, tof, short_way=phase <= math.pi, revolutions=0, backend="python"
                )
            if not solution.converged:
                raise ValueError("Lambert solver did not meet its time tolerance")
            burn1 = solution.v1_km_s - departure_v
            burn2 = arrival_v - solution.v2_km_s
            state = np.concatenate((departure, departure_v + burn1))
            n = max(1, math.ceil(tof / spec.coast_step_s))
            native_state = None
            if backend == "rust":
                from sim.rust_targeting_backend import try_two_body_coast

                native_state = try_two_body_coast(state, step_s=tof / n, steps=n, mu_km3_s2=mu)
            if native_state is not None:
                state = native_state
            else:
                for _ in range(n):
                    state = propagate_two_body_rk4(state, tof / n, mu, np.zeros(3))
            position_error = float(np.linalg.norm(state[:3] - arrival))
            velocity_error = float(np.linalg.norm(state[3:] + burn2 - arrival_v) * 1000.0)
            total = float((np.linalg.norm(burn1) + np.linalg.norm(burn2)) * 1000.0)
            candidates.append({"fraction": fraction, "time_of_flight_s": tof, "method": "oel_lambert",
                               "burn1_delta_v_eci_m_s": (burn1 * 1000).tolist(),
                               "burn2_delta_v_eci_m_s": (burn2 * 1000).tolist(),
                               "total_delta_v_m_s": total, "position_residual_km": position_error,
                               "velocity_residual_m_s": velocity_error,
                               "verified": bool(position_error <= 0.05 and velocity_error <= 0.05)})
        except ValueError as exc:
            candidates.append({"fraction": fraction, "time_of_flight_s": tof, "method": "oel_lambert",
                               "verified": False, "failure_reason": str(exc)})
    accepted = [row for row in candidates if row["verified"] and np.linalg.norm(row.get("burn1_delta_v_eci_m_s", [reference["burn1_m_s"], 0, 0])) <= spec.max_burn1_m_s and np.linalg.norm(row.get("burn2_delta_v_eci_m_s", [reference["burn2_m_s"], 0, 0])) <= spec.max_burn2_m_s]
    selected = min(accepted, key=lambda row: row["total_delta_v_m_s"]) if accepted else None
    return {"schema_version": "oel.trajectory_target.lambert_time_trade.v1", "candidates": candidates,
            "selected_fraction": None if selected is None else selected["fraction"],
            "selected_delta_v_m_s": None if selected is None else selected["total_delta_v_m_s"],
            "claim": "minimum only among verified, bounded sampled candidates"}


def impulsive_scenario(spec: TwoBurnSpec, result: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    if result["status"] != "converged":
        raise ValueError("Cannot run a non-converged target")
    duration = math.ceil((float(result["first_apogee_time_s"]) + 600.0) / spec.coast_step_s) * spec.coast_step_s
    return {
        "scenario_name": "trajectory_target_two_impulse_transfer",
        "scenario_description": "Two exact ECI impulse operations and a passive two-body coast.",
        "objects": {"vehicle": {"enabled": True, "kind": "satellite", "role": "vehicle",
                                "runtime_profile": "trajectory_only", "specs": {"mass_kg": 100.0},
                                "initial_state": {"position_eci_km": result["initial_state_eci"][:3],
                                                  "velocity_eci_km_s": result["initial_state_eci"][3:]}}},
        "simulator": {"duration_s": duration, "dt_s": spec.coast_step_s,
                      "dynamics": {"orbit": {"model": "two_body", "orbit_substep_s": min(spec.coast_step_s / 2, 15.0),
                                             "j2": False, "j3": False, "j4": False, "drag": False, "srp": False},
                                   "attitude": {"enabled": False}},
                      "termination": {"earth_impact_enabled": False}, "plugin_validation": {"strict": True}},
        "outputs": {"output_dir": str(output_dir), "mode": "save",
                    "stats": {"enabled": True, "print_summary": False, "save_json": True, "save_full_log": False},
                    "plots": {"enabled": False, "figure_ids": []},
                    "animations": {"enabled": False, "types": []},
                    "review": {"enabled": True, "detail": "standard"}},
    }


def run_impulsive_transfer(
    spec: TwoBurnSpec, result: dict[str, Any], output_dir: Path, *, config_path: Path | None = None
) -> dict[str, Any]:
    """Execute both exact impulses at OEL sample boundaries and save one review store."""
    cfg = (load_simulation_yaml(config_path) if config_path is not None
           else scenario_config_from_dict(impulsive_scenario(spec, result, output_dir)))
    engine = _SingleRunEngine(cfg)
    try:
        burn1 = engine.apply_impulse("vehicle", result["burn1_delta_v_eci_m_s"], maneuver_id="departure")
        event_time = float(result["first_apogee_time_s"])
        while float(engine.t_s[engine.current_index]) < event_time - 1e-9:
            now = float(engine.t_s[engine.current_index])
            engine.step(dt_s=min(spec.coast_step_s, event_time - now))
        burn2 = engine.apply_impulse("vehicle", result["burn2_delta_v_eci_m_s"], maneuver_id="circularization")
        while not engine.done:
            engine.step()
        engine._shutdown_flight_software()
        payload = engine.build_payload()
        return {"burns": [burn1, burn2], "review_store": str(output_dir / "review" / "run.sqlite"),
                "summary": payload["summary"]}
    finally:
        engine.object_step_executor.shutdown()


def verify_impulsive_transfer(spec: TwoBurnSpec, run_dir: Path) -> dict[str, Any]:
    workspace = ReviewWorkspace.open(run_dir)
    burn_query = (
        "SELECT maneuver_id, sample_index, time_s, dv_x_eci_m_s, dv_y_eci_m_s, dv_z_eci_m_s, "
        "pre_pos_x_eci_km, pre_pos_y_eci_km, pre_pos_z_eci_km, "
        "post_pos_x_eci_km, post_pos_y_eci_km, post_pos_z_eci_km, "
        "pre_vel_x_eci_km_s, pre_vel_y_eci_km_s, pre_vel_z_eci_km_s, "
        "post_vel_x_eci_km_s, post_vel_y_eci_km_s, post_vel_z_eci_km_s "
        "FROM impulsive_maneuvers ORDER BY time_s"
    )
    final_query = (
        "SELECT sample_index, time_s, pos_x_eci_km, pos_y_eci_km, pos_z_eci_km, "
        "vel_x_eci_km_s, vel_y_eci_km_s, vel_z_eci_km_s FROM object_state "
        "WHERE object_id = 'vehicle' ORDER BY sample_index DESC LIMIT 1"
    )
    try:
        burns = workspace.query(burn_query, max_rows=3).rows
        final_rows = workspace.query(final_query, max_rows=1).rows
        if len(burns) != 2 or len(final_rows) != 1:
            raise RuntimeError("Expected two impulses and one final state in the same review store")
        checks = []
        for burn in burns:
            pre_r = np.array([burn[f"pre_pos_{axis}_eci_km"] for axis in "xyz"])
            post_r = np.array([burn[f"post_pos_{axis}_eci_km"] for axis in "xyz"])
            pre_v = np.array([burn[f"pre_vel_{axis}_eci_km_s"] for axis in "xyz"])
            post_v = np.array([burn[f"post_vel_{axis}_eci_km_s"] for axis in "xyz"])
            dv = np.array([burn[f"dv_{axis}_eci_m_s"] for axis in "xyz"])
            checks.append({"maneuver_id": burn["maneuver_id"], "time_s": burn["time_s"],
                           "position_jump_km": float(np.linalg.norm(post_r - pre_r)),
                           "delta_v_error_m_s": float(np.linalg.norm((post_v - pre_v) * 1000 - dv)),
                           "post_radius_km": float(np.linalg.norm(post_r)),
                           "post_radial_speed_km_s": float(np.dot(post_r, post_v) / np.linalg.norm(post_r))})
        final = final_rows[0]
        r = np.array([final[f"pos_{axis}_eci_km"] for axis in "xyz"])
        v = np.array([final[f"vel_{axis}_eci_km_s"] for axis in "xyz"])
        radius_residual = float(np.linalg.norm(r) - spec.target_radius_km)
        circular_speed = math.sqrt(EARTH_MU_KM3_S2 / spec.target_radius_km)
        speed_residual_m_s = float((np.linalg.norm(v) - circular_speed) * 1000)
        passed = (
            [check["maneuver_id"] for check in checks] == ["departure", "circularization"]
            and checks[0]["time_s"] == 0.0
            and abs(checks[1]["post_radius_km"] - spec.target_radius_km) <= spec.radius_tolerance_km
            and abs(checks[1]["post_radial_speed_km_s"]) <= 1.0e-4
            and all(check["position_jump_km"] == 0 and check["delta_v_error_m_s"] <= 1.0e-8 for check in checks)
            and abs(radius_residual) <= max(0.05, spec.radius_tolerance_km)
            and abs(speed_residual_m_s) <= max(0.05, spec.speed_tolerance_m_s)
        )
        return {"passed": bool(passed), "review_evidence": workspace.evidence_identity(),
                "burn_query": burn_query, "final_query": final_query, "burns": checks,
                "final_radius_residual_km": radius_residual,
                "final_speed_residual_m_s": speed_residual_m_s}
    finally:
        workspace.close()
