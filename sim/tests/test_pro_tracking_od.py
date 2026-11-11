from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from sim.dynamics.orbit.accelerations import OrbitContext
from sim.dynamics.orbit.environment import EARTH_MU_KM3_S2
from sim.dynamics.orbit.frames import frame_context_from_mapping
from sim.dynamics.orbit.propagator import OrbitPropagator
from sim.estimation.ground_station_od import predict_ground_station_measurement
from sim.optical_tracking import los_to_ra_dec
from sim.pro_tracking_od import (
    PROBLEM_SCHEMA,
    ProTrackingOdError,
    ProTrackingOdProblem,
    replay_evidence,
    run_pro_tracking_od,
    validate_problem,
)
from sim.pro_tracking_od.__main__ import main


def _truth_states() -> list[np.ndarray]:
    propagator = OrbitPropagator(integrator="rk4")
    context = OrbitContext(mu_km3_s2=EARTH_MU_KM3_S2, mass_kg=100.0)
    states = [np.asarray([7000.0, 0.0, 0.0, 0.0, 7.546049108166282, 0.1])]
    for index in range(1, 6):
        state = states[-1].copy()
        for substep in range(12):
            state = propagator.propagate(
                x_eci=state,
                dt_s=10.0,
                t_s=(index - 1) * 120.0 + substep * 10.0,
                command_accel_eci_km_s2=np.zeros(3),
                env={},
                ctx=context,
            )
        states.append(np.asarray(state))
    return states


def _base(measurement_model: str, measurements: list[dict]) -> dict:
    return {
        "schema_version": PROBLEM_SCHEMA,
        "name": f"synthetic-{measurement_model}",
        "object_id": "SAT-1",
        "measurement_model": measurement_model,
        "measurements": measurements,
        "initial_jd_utc": 2461182.5,
        "initial_state_eci_km_km_s": [7000.05, -0.04, 0.02, 0.00001, 7.54604, 0.10001],
        "initial_covariance_eci": np.diag([0.1**2] * 3 + [1.0e-4**2] * 3).tolist(),
        "fit_duration_s": 360.0,
        "holdout_duration_s": 240.0,
        "propagation": {"step_s": 10.0, "j2": False},
        "filter": {"process_noise_model": "white_acceleration", "spectral_density": [1e-14] * 3},
    }


def _ground_problem() -> dict:
    station = {"id": "GS-1", "lat_deg": 35.0, "lon_deg": -106.0, "alt_km": 1.6}
    context = frame_context_from_mapping({}, jd_utc_start=2461182.5, source="test")
    rows = []
    for index, state in enumerate(_truth_states()):
        time_s = index * 120.0
        prediction = predict_ground_station_measurement(
            target_state_eci=state,
            station=station,
            t_s=time_s,
            jd_utc_start=2461182.5,
            frame_context=context,
        )
        rows.append(
            {
                "measurement_id": f"ground:{index}",
                "station_id": "GS-1",
                "station": station,
                "time_jd_utc": 2461182.5 + time_s / 86400.0,
                "components": ["range_km", "range_rate_km_s"],
                "vector": [prediction["range_km"], prediction["range_rate_km_s"]],
                "sigma": [0.01, 1e-5],
            }
        )
    return _base("ground_topocentric", rows)


def _optical_problem() -> dict:
    observer = np.asarray([6378.137, 0.0, 0.0])
    rows = []
    for index, state in enumerate(_truth_states()):
        time_s = index * 120.0
        ra_deg, dec_deg = los_to_ra_dec(state[:3] - observer)
        rows.append(
            {
                "measurement_id": f"optical:{index}",
                "time_s": time_s,
                "ra_deg": ra_deg,
                "dec_deg": dec_deg,
                "observer_id": "OBS-1",
                "observer_eci_km": observer.tolist(),
                "sigma_arcsec": 5.0,
            }
        )
    return _base("optical_ra_dec", rows)


@pytest.mark.parametrize("factory", [_ground_problem, _optical_problem])
def test_pro_tracking_od_validates_runs_and_replays(factory, tmp_path: Path) -> None:
    problem = factory()
    validation = validate_problem(problem)
    assert validation["fit_count"] == 4
    assert validation["holdout_count"] == 2
    output = tmp_path / "evidence"
    result = run_pro_tracking_od(problem, output)
    assert result["status"] == "complete"
    assert result["holdout_statistics"]["finite"] is True
    assert result["filter_summary"]["covariance_checks"]["all_symmetric_psd"] is True
    assert replay_evidence(output)["status"] == "verified"


def test_pro_tracking_boundary_tolerance_does_not_pull_post_boundary_into_fit() -> None:
    problem = _optical_problem()
    problem["measurements"][3]["time_s"] = 360.0005
    validation = validate_problem(problem)
    assert validation["fit_count"] == 3
    assert validation["holdout_count"] == 3


def test_pro_tracking_od_contract_and_evidence_fail_closed(tmp_path: Path) -> None:
    problem = _ground_problem()
    problem["typo"] = True
    with pytest.raises(ProTrackingOdError, match="Unknown problem"):
        ProTrackingOdProblem.from_mapping(problem)
    problem = _ground_problem()
    problem["filter"]["spectral_density"] = "not-a-vector"
    with pytest.raises(ProTrackingOdError, match="process noise is invalid"):
        ProTrackingOdProblem.from_mapping(problem)
    problem = _ground_problem()
    problem["fit_duration_s"] = 600.0
    with pytest.raises(ProTrackingOdError, match="holdout"):
        validate_problem(problem)
    output = tmp_path / "evidence"
    run_pro_tracking_od(_ground_problem(), output)
    summary = output / "pro_tracking_od_summary.json"
    payload = json.loads(summary.read_text(encoding="utf-8"))
    payload["status"] = "tampered"
    summary.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ProTrackingOdError, match="receipt mismatch"):
        replay_evidence(output)


def test_pro_tracking_od_rejects_non_numeric_state_and_covariance() -> None:
    problem = _ground_problem()
    problem["initial_state_eci_km_km_s"] = ["bad"] * 6
    with pytest.raises(ProTrackingOdError, match="initial_state_eci_km_km_s"):
        ProTrackingOdProblem.from_mapping(problem)

    problem = _ground_problem()
    problem["initial_covariance_eci"] = [["bad"] * 6 for _ in range(6)]
    with pytest.raises(ProTrackingOdError, match="initial_covariance_eci"):
        ProTrackingOdProblem.from_mapping(problem)


def test_replay_rejects_self_receipted_derived_artifact_and_manifest_claims(tmp_path: Path) -> None:
    forged = tmp_path / "forged-artifact"
    run_pro_tracking_od(_ground_problem(), forged)
    prediction = forged / "holdout_predictions.csv"
    prediction.write_bytes(prediction.read_bytes() + b"FORGED\n")
    manifest_path = forged / "pro_tracking_od_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    receipt = next(item for item in manifest["files"] if item["path"] == prediction.name)
    content = prediction.read_bytes()
    receipt.update(bytes=len(content), sha256=hashlib.sha256(content).hexdigest())
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ProTrackingOdError, match="every retained artifact"):
        replay_evidence(forged)

    contradicted = tmp_path / "contradicted-manifest"
    run_pro_tracking_od(_ground_problem(), contradicted)
    manifest_path = contradicted / "pro_tracking_od_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["status"] = "failed"
    manifest["problem_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ProTrackingOdError, match="manifest claims"):
        replay_evidence(contradicted)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda problem: problem.update(propagation={"step_s": 10.0, "j2": "false"}), "must be a boolean"),
        (lambda problem: problem.update(measurements=[*problem["measurements"][:-1], "bad-row"]), "JSON object"),
        (
            lambda problem: problem["measurements"][0].update(time_jd_utc=problem["initial_jd_utc"] + 1.0 / 86400.0),
            "first ground measurement epoch",
        ),
        (lambda problem: problem["measurements"][-1].update(time_jd_utc=problem["initial_jd_utc"] + 720.0 / 86400.0), "declared fit and holdout"),
    ],
)
def test_pro_tracking_od_rejects_ambiguous_epoch_and_contract_inputs(mutate, message: str) -> None:
    problem = _ground_problem()
    mutate(problem)
    with pytest.raises(ProTrackingOdError, match=message):
        validate_problem(problem)


def test_optical_initial_state_epoch_must_match_first_observation() -> None:
    problem = _optical_problem()
    for row in problem["measurements"]:
        row["time_s"] += 1.0
    with pytest.raises(ProTrackingOdError, match="first measurement"):
        validate_problem(problem)


def test_optical_change_point_is_screened_without_cartesian_reinitialization(tmp_path: Path) -> None:
    problem = _optical_problem()
    for index in (1, 2):
        problem["measurements"][index]["ra_deg"] = (
            float(problem["measurements"][index]["ra_deg"]) + 120.0
        ) % 360.0

    result = run_pro_tracking_od(problem, tmp_path / "change-point")

    assert result["status"] == "complete"
    assert result["filter_summary"]["rejected_count"] >= 1
    assert result["claim_posture"]["maneuver_estimation"].startswith("innovation change-point screening only")
    assert replay_evidence(tmp_path / "change-point")["status"] == "verified"


def test_cli_validate_run_and_replay(tmp_path: Path, capsys) -> None:
    problem_path = tmp_path / "problem.json"
    problem_path.write_text(json.dumps(_optical_problem()), encoding="utf-8")
    output = tmp_path / "evidence"
    assert main(["validate", str(problem_path)]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "valid"
    assert main(["run", str(problem_path), "--output-dir", str(output)]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "complete"
    assert main(["replay", str(output)]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "verified"
