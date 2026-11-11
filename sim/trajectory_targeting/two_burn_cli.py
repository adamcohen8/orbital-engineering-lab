"""Run a bounded two-impulse target, sampled time trade, and OEL review verification."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

from sim.config import load_simulation_yaml
from sim.trajectory_targeting.onp_target import onp_scenario, target_two_burn_onp, verify_onp_transfer
from sim.trajectory_targeting.two_burn import (
    TwoBurnSpec,
    impulsive_scenario,
    lambert_time_trade,
    run_impulsive_transfer,
    target_two_burn,
    verify_impulsive_transfer,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-impulsive", action="store_true")
    parser.add_argument("--numeric-backend", choices=("python", "rust"), default="rust")
    parser.add_argument("--onp-template", type=Path,
                        help="Trusted OEL scenario supplying ONP forces, epoch, environment, and vehicle specs")
    args = parser.parse_args(argv)
    raw = args.spec.read_bytes()
    mapping = yaml.safe_load(raw)
    if not isinstance(mapping, dict):
        parser.error("Two-burn spec must be a YAML mapping")
    spec = TwoBurnSpec.from_mapping(mapping)
    runner = Path(__file__).resolve().parents[2] / "run_simulation.py"
    template = None
    if args.onp_template is not None:
        inspection = subprocess.run(
            [sys.executable, str(runner), "--config", str(args.onp_template), "--safe-validate"],
            cwd=runner.parent, capture_output=True, text=True, check=False,
        )
        if inspection.returncode:
            parser.error(f"ONP template failed safe validation: {inspection.stdout + inspection.stderr}")
        template = load_simulation_yaml(args.onp_template).to_dict()
        with tempfile.TemporaryDirectory(prefix="oel-onp-preflight-") as scratch:
            trial_duration = math.ceil(spec.max_coast_s / spec.coast_step_s) * spec.coast_step_s + spec.coast_step_s
            trial_path = Path(scratch) / "onp_trial.yaml"
            trial_path.write_text(yaml.safe_dump(
                onp_scenario(spec, template, Path(scratch) / "run", duration_s=trial_duration),
                sort_keys=False), encoding="utf-8")
            validation = subprocess.run(
                [sys.executable, str(runner), "--config", str(trial_path), "--validate-only"],
                cwd=runner.parent, capture_output=True, text=True, check=False,
            )
            if validation.returncode:
                parser.error(f"Generated ONP trial failed validation: {validation.stdout + validation.stderr}")
    result = (
        target_two_burn(spec, numeric_backend=args.numeric_backend)
        if template is None
        else target_two_burn_onp(spec, template, numeric_backend=args.numeric_backend)
    )
    outdir = args.output_dir.expanduser().resolve()
    if outdir.exists() and any(outdir.iterdir()):
        parser.error(f"Evidence directory is not empty: {outdir}")
    outdir.mkdir(parents=True, exist_ok=True)
    packet = {
        "input_sha256": hashlib.sha256(raw).hexdigest(),
        "targeter_sha256": hashlib.sha256(Path(__file__).with_name("two_burn.py").read_bytes()).hexdigest(),
        "vector_solver_sha256": hashlib.sha256(Path(__file__).with_name("vector.py").read_bytes()).hexdigest(),
        "result": result,
        "time_trade": lambert_time_trade(spec, numeric_backend=args.numeric_backend) if template is None else {
            "status": "not_evaluated", "reason": "Lambert sampled trade uses Earth point-mass dynamics only"},
    }
    if template is not None:
        packet["onp_template_sha256"] = hashlib.sha256(args.onp_template.read_bytes()).hexdigest()
        packet["onp_template_path"] = str(args.onp_template.expanduser().resolve())
        packet["onp_targeter_sha256"] = hashlib.sha256(Path(__file__).with_name("onp_target.py").read_bytes()).hexdigest()
    result_path = outdir / "two_burn_result.json"
    if result["status"] == "converged":
        scenario_path = outdir / "impulsive_transfer.yaml"
        if template is None:
            generated = impulsive_scenario(spec, result, outdir / "run")
        else:
            duration = math.ceil((float(result["first_apogee_time_s"]) + 600.0) / spec.coast_step_s) * spec.coast_step_s
            generated = onp_scenario(spec, template, outdir / "run", duration_s=duration)
        scenario_path.write_text(yaml.safe_dump(generated, sort_keys=False), encoding="utf-8")
        packet["generated_scenario_sha256"] = hashlib.sha256(scenario_path.read_bytes()).hexdigest()
        validation = subprocess.run(
            [sys.executable, str(runner), "--config", str(scenario_path), "--validate-only"],
            cwd=runner.parent, capture_output=True, text=True, check=False,
        )
        (outdir / "validation.txt").write_text(validation.stdout + validation.stderr, encoding="utf-8")
        if validation.returncode:
            raise RuntimeError(f"Generated transfer scenario failed validation: {outdir / 'validation.txt'}")
        if args.run_impulsive:
            packet["oel_execution"] = run_impulsive_transfer(spec, result, outdir / "run", config_path=scenario_path)
            packet["review_verification"] = (
                verify_impulsive_transfer(spec, outdir / "run") if template is None
                else verify_onp_transfer(spec, result, outdir / "run")
            )
            if not packet["review_verification"]["passed"]:
                result_path.write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                raise RuntimeError(f"OEL impulsive transfer review failed: {result_path}")
    result_path.write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "result": str(result_path)}, sort_keys=True))
    return 0 if result["status"] == "converged" else 2


if __name__ == "__main__":
    raise SystemExit(main())
