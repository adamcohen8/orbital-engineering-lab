from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import yaml

from sim.dynamics.orbit.accelerations import OrbitContext
from sim.dynamics.orbit.environment import EARTH_MU_KM3_S2
from sim.dynamics.orbit.propagator import OrbitPropagator
from sim.estimation.augmented_dynamics import AugmentedONPDynamics, AugmentedParameter, discrete_noise
from sim.estimation.tracking_measurements import TrackingMeasurementModel
from sim.pro_tracking_od.continuous import (
    SCHEMA,
    normalize_rows,
    replay_continuous_tracking_od,
    run_continuous_tracking_od,
)

ROOT = Path(__file__).resolve().parents[2]


def dynamics(parameters=(), **kwargs):
    return AugmentedONPDynamics(
        propagator=OrbitPropagator(numeric_backend="python"),
        context=OrbitContext(EARTH_MU_KM3_S2, 100.0),
        environment={},
        parameters=parameters,
        step_s=20.0,
        **kwargs,
    )


def problem():
    scenario = yaml.safe_load((ROOT / "configs/automation_smoke.yaml").read_text())
    scenario["objects"] = {"target": scenario["objects"]["target"]}
    scenario["simulator"]["dynamics"]["attitude"]["enabled"] = False
    scenario["simulator"]["dynamics"]["orbit"]["numeric_backend"] = "python"
    scenario["simulator"]["initial_jd_utc"] = 2459580.5
    scenario["simulator"]["duration_s"] = 360.0
    radius = 7000.0
    speed = np.sqrt(EARTH_MU_KM3_S2 / radius)
    rate = speed / radius
    rows = []
    for i, time in enumerate(np.arange(0.0, 361.0, 60.0)):
        c, s = np.cos(rate * time), np.sin(rate * time)
        vector = [radius * c, radius * s, 0.0, -speed * s, speed * c, 0.0]
        rows.append(
            {
                "kind": "pv",
                "measurement_id": f"pv-{i}",
                "time_s": float(time),
                "vector": vector,
                "covariance": np.diag([0.005**2] * 3 + [5e-6**2] * 3).tolist(),
            }
        )
    return {
        "schema": SCHEMA,
        "object_id": "target",
        "scenario": scenario,
        "initial_state": [radius + 0.01, -0.01, 0.01, 1e-5, speed - 1e-5, 1e-5],
        "initial_covariance": np.diag([0.02**2] * 3 + [2e-5**2] * 3).tolist(),
        "parameters": [],
        "events": [],
        "step_s": 20.0,
        "acceleration_spectral_density": [1e-16] * 3,
        "gate_probability": 0.999,
        "measurements": rows,
        "closed_through_s": 360.0,
        "fit_through_s": 240.0,
        "independent_measurements": True,
        "epoch_covariances": {},
        "sources": [],
        "inertial_frame": "EME2000",
        "epoch_jd_utc": 2459580.5,
    }


def test_continuous_noise_has_correct_force_cross_terms():
    a = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]])
    q = discrete_noise(a, np.diag([0.0, 0.0, 2.0]), 3.0)
    expected = 2 * np.array(
        [[3.0**5 / 20, 3.0**4 / 8, 3.0**3 / 6], [3.0**4 / 8, 3.0**3 / 3, 3.0**2 / 2], [3.0**3 / 6, 3.0**2 / 2, 3.0]]
    )
    np.testing.assert_allclose(q, expected, atol=1e-12)


@pytest.mark.parametrize("row", [None, {"measurement_id": "bad", "time_s": "not-a-number"}])
def test_normalize_rows_rejects_malformed_rows_with_value_error(row):
    with pytest.raises(ValueError, match="tracking row|tracking epochs"):
        normalize_rows([row])


def test_gauss_markov_decay_noise_and_schmidt_mask():
    p = AugmentedParameter(
        "station_bias",
        "range_bias",
        "km",
        1e-4,
        model="gauss_markov",
        spectral_density=2e-8,
        mean=0.01,
        correlation_time_s=100.0,
        station_id="A",
        estimate=False,
    )
    model = dynamics([p])
    state = np.array([7000.0, 0.0, 0.0, 0.0, 7.5, 0.0, 0.03])
    end, phi, q = model.predict(state, 0.0, 60.0)
    assert end[-1] == pytest.approx(0.01 + 0.02 * np.exp(-0.6))
    assert phi[-1, -1] == pytest.approx(np.exp(-0.6), rel=1e-10)
    assert q[-1, -1] == pytest.approx(2e-8 * 50 * (1 - np.exp(-1.2)), rel=1e-9)
    assert not model.update_mask(60.0)[-1]


def test_append_restart_matches_uninterrupted_and_holdout_unassimilated(tmp_path):
    raw = problem()
    full = run_continuous_tracking_od(raw, tmp_path / "full")
    partial = copy.deepcopy(raw)
    partial["measurements"] = raw["measurements"][:3]
    partial["closed_through_s"] = 120.0
    partial["fit_through_s"] = 240.0  # The fixed fit boundary may lie beyond an early ingestion watermark.
    run_continuous_tracking_od(partial, tmp_path / "append")
    appended = run_continuous_tracking_od(raw, tmp_path / "append", append=True)
    assert full["records"] == appended["records"]
    assert full["holdout"] == appended["holdout"]
    assert all(not r["assimilated"] for r in full["holdout"])
    assert len(full["holdout"]) == 2
    assert replay_continuous_tracking_od(tmp_path / "append", tmp_path / "replay")["status"] == "replay_verified"


def test_epoch_update_order_is_canonical_and_correlations_are_used(tmp_path):
    raw = problem()
    extra = copy.deepcopy(raw["measurements"][1])
    extra["measurement_id"] = "other"
    raw["measurements"].append(extra)
    # Correlated repeated data: no accidental independent double counting.
    r = np.asarray(extra["covariance"])
    raw["epoch_covariances"]["60.0"] = np.block([[r, 0.5 * r], [0.5 * r, r]]).tolist()
    a = run_continuous_tracking_od(raw, tmp_path / "a")
    raw["measurements"].reverse()
    b = run_continuous_tracking_od(raw, tmp_path / "b")
    assert a["records"] == b["records"]
    assert a["records"][1]["measurement_dimension"] == 12


@pytest.mark.parametrize("mutation", ["duplicate", "closed", "gate", "checkpoint"])
def test_append_rejects_ambiguous_or_changed_history(tmp_path, mutation):
    raw = problem()
    run_continuous_tracking_od(raw, tmp_path / "run")
    if mutation == "duplicate":
        raw["measurements"][0]["vector"][0] += 1.0
    elif mutation == "closed":
        extra = copy.deepcopy(raw["measurements"][0])
        extra["measurement_id"] = "late"
        raw["measurements"].append(extra)
    elif mutation == "gate":
        raw["gate_probability"] = 0.99
    else:
        (tmp_path / "run/revision-000000/filter_checkpoint.json").write_text("{}")
    with pytest.raises(ValueError):
        run_continuous_tracking_od(raw, tmp_path / "run", append=True)


def slr_row():
    return {
        "kind": "slr",
        "measurement_id": "slr-1",
        "station_id": "A",
        "time_s": 0.0,
        "time_jd_utc": 2459580.5,
        "epoch_event": 0,
        "observed_time_of_flight_s": 0.01,
        "sigma_range_km": 0.001,
        "station_metadata": {"ecef_km": [6378.137, 0.0, 0.0]},
        "correction_flags": {"tropospheric_refraction_applied": True},
        "corrections": {
            k: {"status": "applied", "source_sha256": "a" * 64}
            for k in ("troposphere", "center_of_mass", "target_signature", "station_eccentricity", "instrument")
        },
    }


def test_slr_corrections_are_applied_once_and_share_batch_prediction():
    model = TrackingMeasurementModel(dynamics())
    state = np.array([7000.0, 0.0, 0.0, 0.0, 7.5, 0.0])
    row = slr_row()
    base = model.vectors(row, state)[1][0]
    row["corrections"]["center_of_mass"] = {
        "status": "model",
        "source_sha256": "b" * 64,
        "one_way_equivalent_km": 0.002,
    }
    updated = model(row, state)
    assert updated["predicted"][0] - base == pytest.approx(0.002, abs=1e-11)
    np.testing.assert_allclose(
        model.batch_residual([row], lambda _t: state),
        np.linalg.solve(np.linalg.cholesky(updated["covariance"]), updated["innovation"]),
    )
    row["corrections"]["troposphere"]["status"] = "model"
    with pytest.raises(ValueError, match="flag"):
        model(row, state)


def test_slr_relative_light_time_has_no_julian_date_quantization():
    from sim.slr_tracking import predict_slr_two_way_time_of_flight_s

    # A stationary target makes the exact two-way length independently available.
    station = np.array([6378.137, 0.0, 0.0])
    target = np.array([0.0, 0.0, 10000.0, 0.0, 0.0, 0.0])
    p = predict_slr_two_way_time_of_flight_s(
        epoch_jd_utc=2459580.5,
        epoch_event=0,
        station_ecef_km=station,
        state_at_jd_utc=lambda _jd: target,
        state_at_offset_s=lambda _t: target,
    )
    expected = 2 * np.hypot(6378.137, 10000.0) / 299792.458
    assert p["predicted_time_of_flight_s"] == pytest.approx(expected, abs=1e-13)
    assert p["bounce_offset_s"] < 0.0


def test_native_augmented_update_matches_python(tmp_path):
    raw = problem()
    python = run_continuous_tracking_od(raw, tmp_path / "python")
    raw["scenario"]["simulator"]["dynamics"]["orbit"]["numeric_backend"] = "rust"
    rust = run_continuous_tracking_od(raw, tmp_path / "rust")
    np.testing.assert_allclose(
        [r["filtered_state"] for r in python["records"]],
        [r["filtered_state"] for r in rust["records"]],
        atol=2e-8,
        rtol=1e-10,
    )


def test_continuing_assimilation_and_future_parameter_changes(tmp_path):
    raw = problem()
    raw.update(analysis_mode="continuous", fit_through_s=360.0)
    raw["parameters"] = [
        {"name": "bias_A", "kind": "range_bias", "unit": "km", "step": 0.0001, "station_id": "A", "estimate": True}
    ]
    raw["initial_state"].append(0.0)
    p = np.zeros((7, 7))
    p[:6, :6] = raw["initial_covariance"]
    p[6, 6] = 0.01**2
    p[0, 6] = p[6, 0] = 0.00001
    raw["initial_covariance"] = p.tolist()
    full_raw = copy.deepcopy(raw)
    full_raw["events"] = [{"time_s": 180.0, "parameters": {"bias_A": {"estimate": False}}}]
    full = run_continuous_tracking_od(full_raw, tmp_path / "full")
    partial = copy.deepcopy(raw)
    partial.update(closed_through_s=120.0, fit_through_s=120.0)
    partial["measurements"] = raw["measurements"][:3]
    run_continuous_tracking_od(partial, tmp_path / "append")
    appended = run_continuous_tracking_od(full_raw, tmp_path / "append", append=True)
    assert full["records"] == appended["records"]
    assert len(appended["records"]) == 7 and not appended["holdout"]
    assert appended["records"][3]["smoothing_boundary"]
    assert np.asarray(appended["records"][-1]["filtered_covariance"])[0, 6] != 0
    bad = copy.deepcopy(full_raw)
    bad["events"][0]["time_s"] = 60.0
    with pytest.raises(ValueError, match="digest"):
        run_continuous_tracking_od(bad, tmp_path / "append", append=True)


def test_shared_batch_fits_same_prediction_owner(tmp_path):
    from sim.pro_tracking_od.batch import fit_tracking_batch

    raw = problem()
    raw["acceleration_spectral_density"] = [0.0] * 3
    result = fit_tracking_batch(raw, max_nfev=15)
    assert result["success"]
    assert np.linalg.norm(result["residual"]) < 0.001
    assert result["data_covariance"] is not None


def test_interrupted_revision_resumes_without_changing_result(tmp_path, monkeypatch):
    import sim.estimation.sequential_od as owner

    raw = problem()
    expected = run_continuous_tracking_od(raw, tmp_path / "full")
    original = owner._write_checkpoint

    def interrupt(*args, **kwargs):
        original(*args, **kwargs)
        raise InterruptedError("simulated interruption after checkpoint")

    monkeypatch.setattr(owner, "_write_checkpoint", interrupt)
    with pytest.raises(InterruptedError):
        run_continuous_tracking_od(raw, tmp_path / "interrupted")
    assert not (tmp_path / "interrupted/current.json").exists()
    monkeypatch.setattr(owner, "_write_checkpoint", original)
    recovered = run_continuous_tracking_od(raw, tmp_path / "interrupted")
    assert recovered == expected


def test_drag_srp_scales_have_engine_sensitivities_and_joint_noise():
    from sim.dynamics.orbit.propagator import drag_plugin, srp_plugin

    params = [
        AugmentedParameter("drag", "cd_scale", "1", 1e-3, mean=1.0, model="random_walk", spectral_density=1e-8),
        AugmentedParameter("srp", "srp_scale", "1", 1e-3, mean=1.0),
    ]
    model = AugmentedONPDynamics(
        propagator=OrbitPropagator(plugins=[drag_plugin, srp_plugin], numeric_backend="rust"),
        context=OrbitContext(EARTH_MU_KM3_S2, 100.0, area_m2=4.0),
        environment={"density_kg_m3": 1e-12, "sun_dir_eci": [1.0, 0.0, 0.0], "srp_shadow_model": "none"},
        parameters=params,
        step_s=10.0,
    )
    state = np.array([7000.0, 0.0, 0.0, 0.0, 7.5, 0.0, 1.4, 0.75])
    ctx, env = model.force_inputs(state)
    assert ctx.cd == pytest.approx(2.2 * 1.4) and ctx.cr == pytest.approx(1.2 * 0.75)
    assert env["drag_coefficient"] == ctx.cd
    _end, phi, q = model.predict(state, 0.0, 60.0)
    assert np.linalg.norm(phi[:6, 6]) > 0
    assert np.linalg.norm(phi[:6, 7]) > 0
    assert np.linalg.norm(q[:6, 6]) > 0
    assert q[6, 6] == pytest.approx(60e-8)


def test_scenario_precision_owner_is_reused(tmp_path):
    from sim.pro_tracking_od.continuous import validate_problem

    raw = problem()
    raw["scenario"]["simulator"]["dynamics"]["orbit"]["schwarzschild"] = {"enabled": True}
    _rows, _groups, model, _measurement, _duplicates = validate_problem(raw)
    assert any(p.__module__ == "sim.pro_perturbations.schwarzschild" for p in model.propagator.plugins)
    assert "sim/pro_perturbations/schwarzschild.py" in model.identity()["model"]["sources"]["files"]


def test_zero_noise_adaptive_interval_and_grid_have_numerical_parity():
    model = AugmentedONPDynamics(
        propagator=OrbitPropagator(
            integrator="rkf78", numeric_backend="rust", adaptive_rtol=1e-12, adaptive_atol=1e-13
        ),
        context=OrbitContext(EARTH_MU_KM3_S2, 100.0),
        environment={},
        step_s=20.0,
    )
    state = np.array([7000.0, 0.0, 0.0, 0.0, 7.5, 0.1])
    predicted, phi, noise = model.predict(state, 0.0, 600.0)
    gridded = model.propagate_mean(state, 0.0, 600.0)
    np.testing.assert_allclose(predicted, gridded, atol=1e-7, rtol=0)
    step = 1e-3
    plus, minus = state.copy(), state.copy()
    plus[0] += step
    minus[0] -= step
    expected = (model.propagate_mean(plus, 0.0, 600.0) - model.propagate_mean(minus, 0.0, 600.0)) / (2 * step)
    np.testing.assert_allclose(phi[:, 0], expected, atol=2e-7, rtol=2e-6)
    assert np.count_nonzero(noise) == 0


@pytest.mark.parametrize("case", ["indefinite", "unknown", "missing_independence"])
def test_simultaneous_contract_rejects_invalid_statistics(tmp_path, case):
    raw = problem()
    if case == "unknown":
        raw["measurements"][0]["kind"] = "raw_doppler"
    elif case == "missing_independence":
        raw["independent_measurements"] = False
    else:
        r = np.asarray(raw["measurements"][0]["covariance"])
        r[0, 1] = r[1, 0] = r[0, 0] * 2
        raw["epoch_covariances"]["0.0"] = r.tolist()
    with pytest.raises(ValueError):
        run_continuous_tracking_od(raw, tmp_path / "run")


def test_augmented_covariance_preserves_small_directions_beside_broad_bias_priors():
    model = dynamics([AugmentedParameter("unseen", "range_bias", "km", 1e-4, station_id="A", estimate=False)])
    # A deterministic long-gap map: broad bias variance must not perturb the
    # tightly correlated position/velocity block. The old fixed-scale eig
    # reconstruction perturbed this block enough to destabilize RTS inversion.
    covariance = np.diag([1e-7] * 3 + [1e-15] * 3 + [0.25**2])
    phi = np.eye(7)
    phi[:3, 3:6] = np.eye(3) * 22000
    expected = phi @ covariance @ phi.T
    for _ in range(80):
        covariance = model.stabilize(expected)
        np.testing.assert_array_equal(covariance, expected)
    with pytest.raises(ValueError, match="semidefiniteness"):
        bad = expected.copy()
        bad[0, 3] = bad[3, 0] = np.sqrt(bad[0, 0] * bad[3, 3]) * 1.01
        model.stabilize(bad)


def test_mixed_unit_deterministic_rts_recovers_terminal_information():
    from sim.estimation.sequential_od import _rts_smooth

    model = dynamics([AugmentedParameter("unseen", "range_bias", "km", 1e-4, station_id="A", estimate=False)])
    phi = np.eye(7)
    phi[:3, 3:6] = np.eye(3) * 22000
    prior = np.diag([1e-7] * 3 + [1e-15] * 3 + [0.25**2])
    predicted = phi @ prior @ phi.T
    terminal = predicted * 0.5
    records = [
        {"filtered_state": np.zeros(7), "filtered_covariance": prior},
        {
            "filtered_state": phi @ np.ones(7) * 1e-6,
            "filtered_covariance": terminal,
            "predicted_state": np.zeros(7),
            "predicted_covariance": predicted,
            "transition_from_previous": phi,
        },
    ]
    state, covariance = _rts_smooth(records, 0.0, stabilize=model.stabilize)
    np.testing.assert_allclose(state[0], np.ones(7) * 1e-6, atol=1e-18)
    np.testing.assert_allclose(covariance[0], prior * 0.5, rtol=1e-12, atol=1e-22)
