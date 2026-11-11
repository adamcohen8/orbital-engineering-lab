"""CLI for bounded one-impulse apsis targets and OEL run verification."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

from sim.review import ReviewWorkspace
from sim.trajectory_targeting import ApogeeTargetSpec, PerigeeTargetSpec, target_apogee, target_perigee
from sim.trajectory_targeting.integration import (
    exact_impulse_apsis_scenario,
    maneuver_scenario,
    run_exact_impulse_apsis,
    verify_maneuver_run,
)


def _coast_scenario(
    result: dict, spec: ApogeeTargetSpec | PerigeeTargetSpec, output_dir: Path, *, kind: str = "apogee"
) -> dict:
    state = result["post_burn_state_eci"]
    assert state is not None and result[f"first_{kind}_time_s"] is not None
    duration_s = float(result[f"first_{kind}_time_s"])
    sample_count = max(1, math.ceil(duration_s / min(30.0, spec.coast_step_s)))
    dt_s = duration_s / sample_count
    return {
        "scenario_name": f"trajectory_target_{kind}_coast",
        "scenario_description": (
            "Post-impulse coast verification for an experimental apogee target. "
            "The instantaneous burn is represented by the initial state, not executed in this scenario."
        ),
        "objects": {
            "vehicle": {
                "enabled": True,
                "kind": "satellite",
                "role": "vehicle",
                "specs": {"mass_kg": 100.0},
                "initial_state": {
                    "position_eci_km": state[:3],
                    "velocity_eci_km_s": state[3:],
                },
                "flight_software": {
                    "stack": "fsw.passive",
                    "hardware_profile": "hardware.passive.v1",
                    "params": {},
                },
            },
        },
        "simulator": {
            "duration_s": duration_s,
            "dt_s": dt_s,
            "dynamics": {
                "orbit": {
                    "model": "two_body",
                    "orbit_substep_s": dt_s / 2.0,
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
            "output_dir": str(output_dir / "coast_run"),
            "mode": "save",
            "stats": {"enabled": True, "print_summary": False, "save_json": True, "save_full_log": False},
            "plots": {"enabled": False, "figure_ids": []},
            "animations": {"enabled": False, "types": []},
            "review": {"enabled": True, "detail": "standard"},
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path, help="Safe YAML target specification")
    parser.add_argument("--output-dir", required=True, type=Path, help="New evidence directory")
    parser.add_argument("--run-coast", action="store_true", help="Run the validated OEL post-burn coast scenario")
    parser.add_argument("--run-impulsive", action="store_true", help="Execute the exact impulse and event in one OEL review store")
    parser.add_argument(
        "--run-maneuver", action="store_true", help="Run a full OEL scheduled-burn scenario and verify its review store"
    )
    args = parser.parse_args(argv)

    raw = args.spec.read_bytes()
    data = yaml.safe_load(raw)
    if not isinstance(data, dict):
        parser.error("Target specification must be a YAML mapping")
    schema = data.get("schema_version")
    if schema == "oel.trajectory_target.apogee.v1":
        kind = "apogee"
        spec = ApogeeTargetSpec.from_mapping(data)
        result = target_apogee(spec)
    elif schema == "oel.trajectory_target.perigee.v1":
        kind = "perigee"
        spec = PerigeeTargetSpec.from_mapping(data)
        result = target_perigee(spec)
    else:
        parser.error("Unsupported trajectory target schema_version")
    outdir = args.output_dir.expanduser().resolve()
    if outdir.exists() and any(outdir.iterdir()):
        parser.error(f"Evidence directory is not empty: {outdir}")
    outdir.mkdir(parents=True, exist_ok=True)

    packet = {
        "schema_version": f"oel.trajectory_target.{kind}_result.v1",
        "input_sha256": hashlib.sha256(raw).hexdigest(),
        "source_sha256": hashlib.sha256(Path(__file__).with_name(f"{kind}.py").read_bytes()).hexdigest(),
        "scalar_solver_sha256": hashlib.sha256(Path(__file__).with_name("scalar.py").read_bytes()).hexdigest(),
        "spec": {"schema_version": data["schema_version"], **{key: value for key, value in vars(spec).items()}},
        "result": result.to_dict(),
        "reference_comparison": "GMAT comparison pending independent execution",
    }
    result_path = outdir / "targeting_result.json"
    result_path.write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (outdir / "iterations.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("iteration", "delta_v_m_s", f"{kind}_radius_km", "residual_km"))
        writer.writeheader()
        writer.writerows(result.iterations)

    if result.status == "converged":
        if args.run_impulsive:
            exact_config = outdir / "exact_impulse.yaml"
            exact_config.write_text(
                yaml.safe_dump(exact_impulse_apsis_scenario(spec, result.to_dict(), outdir / "exact_impulse_run", kind=kind), sort_keys=False),
                encoding="utf-8",
            )
            runner = Path(__file__).resolve().parents[2] / "run_simulation.py"
            validation = subprocess.run(
                [sys.executable, str(runner), "--config", str(exact_config), "--validate-only"],
                cwd=runner.parent, capture_output=True, text=True, check=False,
            )
            (outdir / "exact_impulse_validation.txt").write_text(
                validation.stdout + validation.stderr, encoding="utf-8"
            )
            if validation.returncode:
                raise RuntimeError(f"Generated OEL exact impulse scenario failed validation: {outdir / 'exact_impulse_validation.txt'}")
            packet["oel_exact_impulse_verification"] = run_exact_impulse_apsis(
                spec, result.to_dict(), outdir / "exact_impulse_run", kind=kind, config_path=exact_config
            )
            result_path.write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            if not packet["oel_exact_impulse_verification"]["passed"]:
                raise RuntimeError(f"OEL exact-impulse verification failed: {result_path}")
        coast_config = outdir / "post_burn_coast.yaml"
        coast_config.write_text(
            yaml.safe_dump(_coast_scenario(result.to_dict(), spec, outdir, kind=kind), sort_keys=False),
            encoding="utf-8",
        )
        runner = Path(__file__).resolve().parents[2] / "run_simulation.py"
        validation = subprocess.run(
            [sys.executable, str(runner), "--config", str(coast_config), "--validate-only"],
            cwd=runner.parent,
            capture_output=True,
            text=True,
            check=False,
        )
        (outdir / "coast_validation.txt").write_text(
            validation.stdout + validation.stderr,
            encoding="utf-8",
        )
        if validation.returncode:
            raise RuntimeError(f"Generated OEL coast scenario failed validation: {outdir / 'coast_validation.txt'}")
        if args.run_coast:
            run = subprocess.run(
                [sys.executable, str(runner), "--config", str(coast_config)],
                cwd=runner.parent,
                capture_output=True,
                text=True,
                check=False,
            )
            (outdir / "coast_run_console.txt").write_text(run.stdout + run.stderr, encoding="utf-8")
            if run.returncode:
                raise RuntimeError(f"OEL coast run failed: {outdir / 'coast_run_console.txt'}")
            workspace = ReviewWorkspace.open(outdir / "coast_run")
            try:
                query = (
                    "SELECT sample_index, time_s, pos_x_eci_km, pos_y_eci_km, pos_z_eci_km, "
                    "vel_x_eci_km_s, vel_y_eci_km_s, vel_z_eci_km_s "
                    "FROM object_state WHERE object_id = 'vehicle' ORDER BY sample_index DESC LIMIT 1"
                )
                rows = workspace.query(query, max_rows=1).rows
                if len(rows) != 1:
                    raise RuntimeError("OEL coast review store has no final vehicle state")
                row = rows[0]
                position = np.array(
                    [row[name] for name in ("pos_x_eci_km", "pos_y_eci_km", "pos_z_eci_km")], dtype=float
                )
                velocity = np.array(
                    [row[name] for name in ("vel_x_eci_km_s", "vel_y_eci_km_s", "vel_z_eci_km_s")], dtype=float
                )
                target_radius = getattr(spec, f"target_{kind}_radius_km")
                residual_km = float(np.linalg.norm(position) - target_radius)
                radial_speed_km_s = float(np.dot(position, velocity) / np.linalg.norm(position))
                packet["oel_coast_verification"] = {
                    "review_evidence": workspace.evidence_identity(),
                    "query": query,
                    "final_sample_index": row["sample_index"],
                    "final_time_s": row["time_s"],
                    "radius_residual_km": residual_km,
                    "radial_speed_km_s": radial_speed_km_s,
                    "passed": abs(residual_km) <= spec.tolerance_km and abs(radial_speed_km_s) <= 1.0e-4,
                }
            finally:
                workspace.close()
            result_path.write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            if not packet["oel_coast_verification"]["passed"]:
                raise RuntimeError(f"OEL review-store coast verification failed: {result_path}")

        if args.run_maneuver:
            maneuver_config = outdir / "full_maneuver.yaml"
            maneuver_config.write_text(
                yaml.safe_dump(maneuver_scenario(spec, result.to_dict(), outdir, kind=kind), sort_keys=False),
                encoding="utf-8",
            )
            validation = subprocess.run(
                [sys.executable, str(runner), "--config", str(maneuver_config), "--validate-only"],
                cwd=runner.parent,
                capture_output=True,
                text=True,
                check=False,
            )
            (outdir / "maneuver_validation.txt").write_text(validation.stdout + validation.stderr, encoding="utf-8")
            if validation.returncode:
                raise RuntimeError(
                    f"Generated OEL maneuver scenario failed validation: {outdir / 'maneuver_validation.txt'}"
                )
            run = subprocess.run(
                [sys.executable, str(runner), "--config", str(maneuver_config)],
                cwd=runner.parent,
                capture_output=True,
                text=True,
                check=False,
            )
            (outdir / "maneuver_run_console.txt").write_text(run.stdout + run.stderr, encoding="utf-8")
            if run.returncode:
                raise RuntimeError(f"OEL maneuver run failed: {outdir / 'maneuver_run_console.txt'}")
            packet["oel_maneuver_verification"] = verify_maneuver_run(
                outdir / "maneuver_run",
                kind=kind,
                target_radius_km=getattr(spec, f"target_{kind}_radius_km"),
                tolerance_km=spec.tolerance_km,
                planned_delta_v_eci_m_s=result.delta_v_eci_m_s,
                planned_event_time_s=getattr(result, f"first_{kind}_time_s"),
            )
            result_path.write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            if not packet["oel_maneuver_verification"]["passed"]:
                raise RuntimeError(f"OEL review-store maneuver verification failed: {result_path}")

    print(
        json.dumps(
            {"status": result.status, "result": str(result_path), "delta_v_m_s": result.delta_v_m_s}, sort_keys=True
        )
    )
    return 0 if result.status == "converged" else 2


if __name__ == "__main__":
    raise SystemExit(main())
