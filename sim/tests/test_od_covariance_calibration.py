from copy import deepcopy

import numpy as np
import pytest

from sim.estimation.batch_least_squares import solve_batch_least_squares
from sim.estimation.covariance import covariance_from_jacobian
from sim.estimation.covariance_calibration import (
    apply_scale_profile,
    assess_covariance,
    fit_noise_profile,
    fit_scale_profile,
    innovation_diagnostics,
    noise_covariance,
)
from sim.estimation.orbit_od_covariance import consider_parameter_covariance, propagate_joint_covariance
from sim.estimation.parameters import EstimatedParameter, ParameterSet


def test_svd_preserves_weak_full_rank_direction_and_rejects_nullspace():
    p = covariance_from_jacobian(np.diag([1.0, 1e-8]), np.zeros(2), scale_policy="absolute")
    assert p[1, 1] == pytest.approx(1e16)
    with pytest.raises(ValueError, match="rank-deficient"):
        covariance_from_jacobian(np.ones((5, 2)), np.zeros(5))


def test_prior_regularization_never_masquerades_as_data_information():
    parameters = ParameterSet([EstimatedParameter("a", 0), EstimatedParameter("b", 0)])
    result = solve_batch_least_squares(
        parameters,
        lambda x: np.repeat(x.sum() - 2, 6),
        prior_mean_native=np.zeros(2),
        prior_covariance_native=np.eye(2),
        covariance_scale_policy="absolute",
    )
    assert result.covariance_native is not None
    assert result.data_covariance_native is None
    assert result.covariance_evidence["prior_information_included"]
    assert not result.covariance_evidence["empirically_calibrated"]


def test_degenerate_fit_has_no_finite_covariance():
    result = solve_batch_least_squares(
        ParameterSet([EstimatedParameter("a", 0), EstimatedParameter("b", 0)]), lambda x: np.repeat(x.sum() - 2, 6)
    )
    assert result.covariance_native is None
    assert result.covariance_evidence["status"] == "unavailable_rank_deficient"


def test_noise_multiplier_does_not_inflate_the_prior():
    # One data-sensitive direction with very large residual variance; the
    # independent prior-only direction retains variance 4, rather than 4*s².
    parameters = ParameterSet([EstimatedParameter("a", 0), EstimatedParameter("b", 0)])
    result = solve_batch_least_squares(
        parameters,
        lambda x: np.array([x[0] - 10, x[0] + 10, x[0]]),
        prior_parameter_names=["b"],
        prior_mean_native=np.array([0]),
        prior_covariance_native=np.array([[4.0]]),
    )
    assert result.covariance_native[1, 1] == pytest.approx(4.0)


@pytest.mark.parametrize("kind", ["bounds", "robust", "rejection"])
def test_non_gaussian_processing_is_explicitly_labeled(kind):
    p = ParameterSet([EstimatedParameter("a", 0, lower=-1, upper=1)])
    kwargs = (
        {"robust_loss": "huber"} if kind == "robust" else {"sigma_clip_threshold": 3} if kind == "rejection" else {}
    )
    target = np.r_[np.zeros(20), 100.0] if kind != "bounds" else np.repeat(10.0, 20)
    result = solve_batch_least_squares(p, lambda x: x - target, **kwargs)
    assert result.covariance_evidence["status"] != "formal_local_gaussian"


def test_joint_forecast_keeps_cross_terms_consider_response_and_process_noise():
    a = np.array([[2.0, 1.0], [1.0, 3.0], [4.0, -1.0]])
    b = np.array([[1.0], [2.0], [1.0]])
    nuisance = np.array([[0.3]])
    extra, cross = consider_parameter_covariance(a, b, nuisance)
    p = np.eye(2) + extra
    g = np.arange(12.0).reshape(6, 2) / 10
    h = np.arange(6.0).reshape(6, 1) / 10
    q = np.eye(6)[None] * 0.02
    result = propagate_joint_covariance(
        lambda x, c: (g @ x + h @ c)[None],
        parameter_values=[0, 0],
        parameter_covariance=p,
        parameter_steps=[0.01, 0.01],
        parameter_names=["x", "drag"],
        times_s=[10],
        consider_values=[0],
        consider_covariance=nuisance,
        consider_steps=[0.01],
        consider_names=["density"],
        parameter_consider_cross_covariance=cross,
        process_noise_covariances=q,
    )
    expected = g @ p @ g.T + h @ nuisance @ h.T + g @ cross @ h.T + h @ cross.T @ g.T + q[0]
    np.testing.assert_allclose(result["records"][0]["covariance_eci_km_km_s"], expected, atol=1e-13)
    assert not result["empirically_calibrated"]


def test_invalid_joint_consider_covariance_is_rejected():
    with pytest.raises(ValueError, match="positive semidefinite"):
        propagate_joint_covariance(
            lambda x, c: np.zeros((1, 6)),
            parameter_values=[0],
            parameter_covariance=np.eye(1),
            parameter_steps=[1],
            parameter_names=["x"],
            times_s=[1],
            consider_values=[0],
            consider_covariance=np.eye(1),
            consider_steps=[1],
            consider_names=["density"],
            parameter_consider_cross_covariance=np.array([[2.0]]),
        )


def _ensemble(seed=719, n=1000):
    return np.random.default_rng(seed).normal(size=(n, 3)), np.repeat(np.eye(3)[None], n, axis=0)


def test_coverage_accepts_nominal_and_detects_under_and_overconfidence_and_bias():
    e, p = _ensemble()
    ids = [f"qualification:{i}" for i in range(len(e))]
    assert assess_covariance(e, p, trial_ids=ids)["passed"]
    assert not assess_covariance(e, p / 4, trial_ids=ids)["passed"]
    assert not assess_covariance(e, p * 100, trial_ids=ids)["passed"]
    assert not assess_covariance(e + 1, p, trial_ids=ids)["checks"]["unbiased"]


def test_independent_reference_uncertainty_and_cross_covariance_are_accounted():
    e, p = _ensemble()
    ids = list(range(len(e)))
    report = assess_covariance(
        e * np.sqrt(1.5), p, reference_covariances=p, estimate_reference_cross_covariances=p * 0.25, trial_ids=ids
    )
    assert report["passed"]
    with pytest.raises(ValueError, match="positive semidefinite"):
        assess_covariance(e, p, reference_covariances=p, estimate_reference_cross_covariances=p * 2, trial_ids=ids)


def test_profiles_are_frozen_scoped_and_require_disjoint_units():
    e, p = _ensemble()
    profile = fit_scale_profile(e * 2, p, development_ids=range(len(e)), envelope={"regime": "GEO"})
    assert 3.8 < profile["scale"] < 4.2
    with pytest.raises(ValueError, match="development"):
        apply_scale_profile(profile, p, trial_ids=range(len(e)), envelope={"regime": "GEO"})
    with pytest.raises(ValueError, match="envelope"):
        apply_scale_profile(profile, p, trial_ids=range(1000, 2000), envelope={"regime": "LEO"})
    changed = deepcopy(profile)
    changed["scale"] = 8
    with pytest.raises(ValueError, match="checksum"):
        apply_scale_profile(changed, p, trial_ids=range(1000, 2000), envelope={"regime": "GEO"})
    with pytest.raises(ValueError, match="overlap"):
        assess_covariance(e, p, trial_ids=range(len(e)), development_ids=[0])


def test_correlated_noise_calibration_whitens_fresh_arcs():
    rng = np.random.default_rng(712)
    arcs = rng.normal(size=(100, 80, 1))
    for t in range(1, 80):
        arcs[:, t] = 0.7 * arcs[:, t - 1] + np.sqrt(1 - 0.7**2) * arcs[:, t]
    profile = fit_noise_profile(arcs[:40], development_ids=range(40), envelope={"station": "A", "cadence_s": 10})
    assert profile["rho"] == pytest.approx(0.7, abs=0.06)
    full = noise_covariance(profile, range(80), envelope=profile["envelope"], arc_id="fresh")
    white = np.stack([np.linalg.solve(np.linalg.cholesky(full), a.ravel()) for a in arcs[40:]])
    assert abs(np.corrcoef(white[:, :-1].ravel(), white[:, 1:].ravel())[0, 1]) < 0.08
    with pytest.raises(ValueError, match="disjoint"):
        noise_covariance(profile, range(80), envelope=profile["envelope"], arc_id=0)


def test_innovation_checks_detect_correlations_even_with_correct_terminal_nis():
    rng = np.random.default_rng(20)
    values = rng.normal(size=(300, 40, 1))
    for t in range(1, 40):
        values[:, t] = 0.8 * values[:, t - 1] + 0.6 * values[:, t]
    report = innovation_diagnostics(values, np.ones((300, 40, 1, 1)), trial_ids=range(300))
    assert not report["checks"]["innovation_whiteness"]


@pytest.mark.parametrize("use_process_noise", [False, True])
def test_production_api_forecasts_joint_covariance_and_temporal_measurement_noise(tmp_path, use_process_noise):
    from sim.dynamics.orbit.accelerations import OrbitContext
    from sim.dynamics.orbit.propagator import OrbitPropagator
    from sim.estimation.orbit_determination import solve_dynamics_orbit_determination
    from sim.observations import ingest_observations

    times = np.arange(0, 601, 60.0)
    state = np.array([7000.0, 0, 0, 0, 7.546, 0])
    prop = OrbitPropagator(integrator="rk4", numeric_backend="rust")
    states = [state]
    for t in times[1:]:
        state = prop.propagate(state, 60, t - 60, np.zeros(3), {}, OrbitContext(mu_km3_s2=398600.4418, mass_kg=100.0))
        states.append(state)
    packet = ingest_observations(
        object_id="s",
        observations=[
            {
                "time_s": t,
                "position_eci_km": x[:3].tolist(),
                "velocity_eci_km_s": x[3:].tolist(),
                "position_sigma_km": 0.005,
                "velocity_sigma_km_s": 0.000005,
            }
            for t, x in zip(times, states, strict=True)
        ],
    )
    covariance = np.kron(
        0.2 ** np.abs(np.arange(7)[:, None] - np.arange(7)[None]), np.diag([0.005**2] * 3 + [0.000005**2] * 3)
    )
    result = solve_dynamics_orbit_determination(
        packet,
        output_dir=tmp_path / "od",
        fit_duration_s=360,
        holdout_duration_s=240,
        dt_s=60,
        orbit_force_model={"orbit_substep_s": 1.0, "integrator": "rk4"},
        forecast_uncertainty=True,
        covariance_scale_policy="absolute",
        fit_observation_covariance=covariance,
        forecast_process_noise={"spectral_density": [1e-15 if use_process_noise else 0] * 3},
        forecast_consider={"names": ["position_bias_x_km"], "covariance": [[0.0001]], "steps": [0.001]},
    )
    forecast = result["forecast_uncertainty"]
    assert forecast["parameter_cross_correlations_included"]
    assert forecast["consider_names"] == ["position_bias_x_km"]
    if use_process_noise:
        assert forecast["process_noise_policy"]["transport"] == "ONP_finite_difference_state_transition"
        assert forecast["process_noise_policy"]["start_time_s"] == 360
    assert len(forecast["records"]) == len(times)
    assert forecast["formal_covariance_policy"]["scale_policy"] == "absolute"
    assert result["estimation_policy"]["fit_temporal_covariance_supplied"]
    assert np.linalg.eigvalsh(forecast["records"][-1]["covariance_eci_km_km_s"])[0] > 0


def test_passive_calendar_preserves_irregular_boundaries_and_refuses_external_inputs(tmp_path):
    from sim.api import SimulationSession
    from sim.dynamics.orbit.environment import EARTH_MU_KM3_S2
    from sim.estimation.orbit_od_solver import _onp_capture_native_step_schedule
    from sim.ingestion import build_basic_propagation_scenario, ingest_state_vector

    packet = ingest_state_vector(
        object_id="target", role="target", position=[7000.0, 0.0, 0.0],
        velocity=[0.0, (EARTH_MU_KM3_S2 / 7000.0) ** 0.5, 0.0],
    )
    artifact = build_basic_propagation_scenario(
        packet, scenario_name="passive_calendar", output_dir=tmp_path / "artifact",
        duration_s=4.9, dt_s=0.7, dynamics_model="two_body", j2=False,
    )
    s = SimulationSession.from_config(artifact, history_mode="dynamic")
    s.reset()
    runtime = s._engine.agents["target"].flight_software_runtime
    before = runtime.stack.snapshot()
    boundaries = runtime.passive_prediction_boundaries_ns(start_time_ns=0, end_time_ns=4_005_000_000)
    assert boundaries[-1] == 4_005_000_000
    assert runtime.stack.snapshot() == before
    runtime.add_input_publisher(lambda *a, **k: ())
    assert runtime.passive_prediction_boundaries_ns(start_time_ns=0, end_time_ns=4_005_000_000) is None
    metadata = {}
    _onp_capture_native_step_schedule(
        artifact,
        object_id="target",
        epochs_s=[0, 0.23, 1.01, 4.005],
        fallback_step_s=0.1,
        metadata=metadata,
        allow_passive_calendar=True,
    )
    assert metadata["calendar_only"]


def test_process_noise_selection_penalizes_both_under_and_overdispersion():
    from sim.estimation.covariance_calibration import fit_process_noise_profile, process_noise_from_profile

    errors = np.random.default_rng(812).normal(size=(500, 3))

    def evaluate(policy):
        return errors, np.repeat(np.eye(3)[None] * policy.spectral_density[0], len(errors), axis=0)

    profile = fit_process_noise_profile(
        evaluate,
        [{"spectral_density": (x, x, x)} for x in (0.01, 1, 100)],
        development_ids=range(500),
        envelope={"regime": "test"},
    )
    assert profile["process_noise"]["spectral_density"] == [1, 1, 1]
    assert process_noise_from_profile(profile, trial_ids=["new"], envelope={"regime": "test"}).spectral_density == (
        1,
        1,
        1,
    )
