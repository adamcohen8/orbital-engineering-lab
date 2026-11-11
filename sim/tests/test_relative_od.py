from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from sim.dynamics.orbit.elements import coe_to_rv_eci
from sim.dynamics.orbit.environment import EARTH_MU_KM3_S2
from sim.estimation.relative_od import (
    _relative_transition_cache,
    compare_relative_od_models,
    make_relative_od_arc,
    sample_relative_od_observations,
    solve_batch_relative_od,
)
from sim.estimation.relative_od_integrated import generate_nonlinear_relative_truth, run_integrated_relative_od


def _circular_mean_motion(radius_km: float = 7000.0) -> float:
    return float(np.sqrt(EARTH_MU_KM3_S2 / radius_km**3))


def _eccentric_chief() -> tuple[np.ndarray, float]:
    a_km = 9000.0
    r_vec, v_vec = coe_to_rv_eci(
        a_km=a_km,
        ecc=0.25,
        inc_deg=28.0,
        raan_deg=15.0,
        argp_deg=35.0,
        true_anomaly_deg=20.0,
    )
    return np.hstack((r_vec, v_vec)), float(np.sqrt(EARTH_MU_KM3_S2 / a_km**3))


def test_sequential_th_and_ya_transition_cache_has_explicit_rounding_gate() -> None:
    chief, _mean_motion = _eccentric_chief()
    times = np.arange(0.0, 1440.1, 120.0)
    kwargs = {
        "mean_motion_rad_s": None,
        "chief_initial_state_eci_km_s": chief,
        "mu_km3_s2": EARTH_MU_KM3_S2,
        "j2": 1.08262668e-3,
        "earth_radius_km": 6378.137,
        "maximum_ss_eccentricity": 0.01,
        "integration_substep_s": 5.0,
    }
    for model, tolerance in (("th", 3.0e-12), ("ya", 2.0e-8)):
        direct = _relative_transition_cache(model, times, **kwargs)
        sequential = _relative_transition_cache(
            model,
            times,
            transition_evaluation_method="sequential_composition",
            **kwargs,
        )
        assert max(np.max(np.abs(direct[t] - sequential[t])) for t in times) <= tolerance


def test_sequential_transition_cache_falls_back_for_irregular_duplicate_and_negative_epochs() -> None:
    chief, _mean_motion = _eccentric_chief()
    kwargs = {
        "mean_motion_rad_s": None,
        "chief_initial_state_eci_km_s": chief,
        "mu_km3_s2": EARTH_MU_KM3_S2,
        "j2": 1.08262668e-3,
        "earth_radius_km": 6378.137,
        "maximum_ss_eccentricity": 0.01,
        "integration_substep_s": 5.0,
    }
    for times, reason in (
        (np.linspace(0.0, 1440.0, 51), "outside_validated_sequential_envelope"),
        (np.array([0.0, 120.0, 120.0, 241.0]), "outside_validated_sequential_envelope"),
        (np.array([-5.0, 0.0, 120.0]), "negative_epoch_requires_direct_evaluation"),
    ):
        provenance: dict[str, object] = {}
        direct = _relative_transition_cache("th", times, **kwargs)
        observed = _relative_transition_cache(
            "th",
            times,
            transition_evaluation_method="sequential_composition",
            provenance=provenance,
            **kwargs,
        )
        assert provenance["requested_method"] == "sequential_composition"
        assert provenance["observed_method"] == "direct"
        assert provenance["fallback_reason"] == reason
        assert all(np.array_equal(direct[t], observed[t]) for t in np.unique(times))

    duplicate_times = np.array([0.0, 120.0, 120.0, 240.0])
    provenance = {}
    direct = _relative_transition_cache("th", duplicate_times, **kwargs)
    observed = _relative_transition_cache(
        "th",
        duplicate_times,
        transition_evaluation_method="sequential_composition",
        provenance=provenance,
        **kwargs,
    )
    assert provenance["observed_method"] == "sequential_composition"
    assert provenance["fallback_used"] is False
    assert set(observed) == set(np.unique(duplicate_times))
    assert max(np.max(np.abs(direct[t] - observed[t])) for t in observed) <= 3.0e-12


def test_sequential_reviewer_attack_cases_fall_back_exactly_to_direct() -> None:
    chief, _mean_motion = _eccentric_chief()
    base = {
        "mean_motion_rad_s": None,
        "chief_initial_state_eci_km_s": chief,
        "mu_km3_s2": EARTH_MU_KM3_S2,
        "j2": 1.08262668e-3,
        "earth_radius_km": 6378.137,
        "maximum_ss_eccentricity": 0.01,
    }
    attacks = (
        (np.arange(0.0, 1440.1, 120.0), 10.0),
        (np.arange(0.0, 7200.1, 120.0), 5.0),
        (np.array([0.0, 120.0, 300.0, 420.0]), 5.0),
    )
    for model in ("th", "ya"):
        for times, substep in attacks:
            kwargs = {**base, "integration_substep_s": substep}
            direct = _relative_transition_cache(model, times, **kwargs)
            provenance: dict[str, object] = {}
            observed = _relative_transition_cache(
                model,
                times,
                transition_evaluation_method="sequential_composition",
                provenance=provenance,
                **kwargs,
            )
            assert provenance["observed_method"] == "direct"
            assert provenance["fallback_reason"] == "outside_validated_sequential_envelope"
            assert all(np.array_equal(direct[t], observed[t]) for t in np.unique(times))


def test_validated_sequential_chief_envelope_has_conservative_observed_stm_bound() -> None:
    chief_states = []
    for a_km, ecc, anomaly_deg, inc_deg in (
        (7000.0, 0.0, 270.0, 63.0),
        (7500.0, 0.1, 270.0, 63.0),
        (9000.0, 0.25, 300.0, 63.0),
        (12000.0, 0.2, 300.0, 63.0),
        (42000.0, 0.2, 180.0, 63.0),
    ):
        position, velocity = coe_to_rv_eci(
            a_km=a_km,
            ecc=ecc,
            inc_deg=inc_deg,
            raan_deg=15.0,
            argp_deg=35.0,
            true_anomaly_deg=anomaly_deg,
        )
        chief_states.append(np.hstack((position, velocity)))
    times = np.arange(0.0, 1440.1, 120.0)
    for chief in chief_states:
        kwargs = {
            "mean_motion_rad_s": None,
            "chief_initial_state_eci_km_s": chief,
            "mu_km3_s2": EARTH_MU_KM3_S2,
            "j2": 1.08262668e-3,
            "earth_radius_km": 6378.137,
            "maximum_ss_eccentricity": 0.01,
            "integration_substep_s": 5.0,
        }
        for model in ("th", "ya"):
            direct = _relative_transition_cache(model, times, **kwargs)
            provenance: dict[str, object] = {}
            sequential = _relative_transition_cache(
                model,
                times,
                transition_evaluation_method="sequential_composition",
                provenance=provenance,
                **kwargs,
            )
            assert provenance["observed_method"] == "sequential_composition"
            assert max(np.max(np.abs(direct[t] - sequential[t])) for t in times) <= 1.0e-7

    high_eccentricity_position, high_eccentricity_velocity = coe_to_rv_eci(
        a_km=12000.0,
        ecc=0.75,
        inc_deg=63.0,
        raan_deg=15.0,
        argp_deg=35.0,
        true_anomaly_deg=0.0,
    )
    provenance = {}
    _relative_transition_cache(
        "ya",
        times,
        mean_motion_rad_s=None,
        chief_initial_state_eci_km_s=np.hstack((high_eccentricity_position, high_eccentricity_velocity)),
        mu_km3_s2=EARTH_MU_KM3_S2,
        j2=1.08262668e-3,
        earth_radius_km=6378.137,
        maximum_ss_eccentricity=0.01,
        integration_substep_s=5.0,
        transition_evaluation_method="sequential_composition",
        provenance=provenance,
    )
    assert provenance["observed_method"] == "direct"
    assert provenance["fallback_reason"] == "outside_validated_sequential_chief_envelope"


def test_hcw_batch_relative_od_recovers_initial_state_from_relative_state_arc(tmp_path: Path) -> None:
    n = _circular_mean_motion()
    truth = np.array([0.08, -0.12, 0.035, 0.00002, -0.00001, 0.000004], dtype=float)
    guess = truth + np.array([0.02, -0.015, 0.01, 3e-6, -2e-6, 1e-6], dtype=float)
    times = np.arange(0.0, 660.0, 60.0)
    arc = sample_relative_od_observations(
        dynamics_model="hcw",
        initial_relative_state_ric=truth,
        times_s=times,
        measurement_model="relative_state",
        noise_diag=np.array([1e-8, 1e-8, 1e-8, 1e-12, 1e-12, 1e-12]),
        mean_motion_rad_s=n,
    )

    result = solve_batch_relative_od(
        arc,
        dynamics_model="hcw",
        initial_guess_ric=guess,
        output_dir=tmp_path / "hcw",
        fit_duration_s=360.0,
        holdout_duration_s=240.0,
        mean_motion_rad_s=n,
        truth_initial_relative_state_ric=truth,
    )

    assert result["method"] == "batch_relative_orbit_determination"
    assert result["quality_gates"]["fit_improved_prefit_rms"] is True
    assert result["fit_metrics"]["weighted_rms"] < 1.0e-6
    assert result["holdout_metrics"]["weighted_rms"] < 1.0e-6
    assert np.allclose(result["estimated_initial_state_ric"], truth, atol=1e-10)
    assert (tmp_path / "hcw" / "relative_od_fit_report.json").exists()
    assert (tmp_path / "hcw" / "relative_od_fit_residuals.csv").exists()


def test_sampled_relative_od_rebases_absolute_request_times() -> None:
    truth = np.array([0.08, -0.12, 0.035, 0.00002, -0.00001, 0.000004], dtype=float)
    arc = sample_relative_od_observations(
        dynamics_model="hcw",
        initial_relative_state_ric=truth,
        times_s=[100.0, 160.0],
        measurement_model="relative_state",
        noise_diag=np.ones(6) * 1.0e-12,
        mean_motion_rad_s=_circular_mean_motion(),
    )

    assert [observation.time_s for observation in arc.observations] == [0.0, 60.0]
    np.testing.assert_array_equal(arc.observations[0].vector, truth)


def test_relative_od_holdout_duration_limits_holdout_samples() -> None:
    n = _circular_mean_motion()
    truth = np.array([0.08, -0.12, 0.035, 0.00002, -0.00001, 0.000004], dtype=float)
    times = np.arange(0.0, 1200.0, 120.0)
    arc = sample_relative_od_observations(
        dynamics_model="hcw",
        initial_relative_state_ric=truth,
        times_s=times,
        measurement_model="relative_state",
        noise_diag=np.array([1e-8, 1e-8, 1e-8, 1e-12, 1e-12, 1e-12]),
        mean_motion_rad_s=n,
    )

    result = solve_batch_relative_od(
        arc,
        dynamics_model="hcw",
        initial_guess_ric=truth,
        fit_duration_s=360.0,
        holdout_duration_s=240.0,
        mean_motion_rad_s=n,
    )

    assert result["holdout_observation_count"] == 2
    assert result["holdout_metrics"]["sample_count"] == 12


def test_th_and_ya_batch_relative_od_recover_eccentric_angles_range_rate_arc(tmp_path: Path) -> None:
    chief, _n = _eccentric_chief()
    truth = np.array([0.08, -0.18, 0.06, 0.000006, -0.000004, 0.000003], dtype=float)
    guess = truth + np.array([0.015, -0.012, 0.008, -1.5e-6, 1.0e-6, 8.0e-7], dtype=float)
    times = np.arange(0.0, 1020.0, 120.0)
    noise_diag = np.array([1.0e-8, 1.0e-8, 1.0e-6, 1.0e-10], dtype=float)
    arc = sample_relative_od_observations(
        dynamics_model="ya",
        initial_relative_state_ric=truth,
        times_s=times,
        measurement_model="relative_angles_range_rate",
        noise_diag=noise_diag,
        chief_initial_state_eci_km_s=chief,
    )

    th = solve_batch_relative_od(
        arc,
        dynamics_model="th",
        initial_guess_ric=guess,
        output_dir=tmp_path / "th",
        fit_duration_s=480.0,
        holdout_duration_s=480.0,
        chief_initial_state_eci_km_s=chief,
        truth_initial_relative_state_ric=truth,
        integration_substep_s=5.0,
        max_nfev=24,
        transition_evaluation_method="sequential_composition",
    )
    ya = solve_batch_relative_od(
        arc,
        dynamics_model="ya",
        initial_guess_ric=guess,
        output_dir=tmp_path / "ya",
        fit_duration_s=480.0,
        holdout_duration_s=480.0,
        chief_initial_state_eci_km_s=chief,
        truth_initial_relative_state_ric=truth,
        integration_substep_s=5.0,
        max_nfev=24,
    )

    assert th["fit_metrics"]["weighted_rms"] < 1.0e-5
    assert th["holdout_metrics"]["weighted_rms"] < 1.0e-5
    assert ya["fit_metrics"]["weighted_rms"] < 1.0e-5
    assert ya["holdout_metrics"]["weighted_rms"] < 1.0e-5
    assert np.allclose(th["estimated_initial_state_ric"], ya["estimated_initial_state_ric"], atol=2e-8)
    assert (tmp_path / "th" / "relative_od_fit_report.md").exists()
    assert (tmp_path / "ya" / "relative_od_holdout_residuals.csv").exists()
    assert th["epoch_evaluation"]["method"] == "relative_dynamics_sequential_transition_composition"
    assert th["epoch_evaluation"]["transition_provenance"]["observed_method"] == "sequential_composition"
    with (tmp_path / "th" / "relative_od_fit_residuals.csv").open(encoding="utf-8") as handle:
        assert {row["epoch_evaluation_method"] for row in csv.DictReader(handle)} == {
            "relative_dynamics_sequential_transition_composition"
        }


def test_relative_od_comparison_matrix_shows_hcw_holdout_degradation_on_eccentric_arc(tmp_path: Path) -> None:
    chief, mean_motion = _eccentric_chief()
    truth = np.array([0.08, -0.18, 0.06, 0.000006, -0.000004, 0.000003], dtype=float)
    guess = truth + np.array([0.02, -0.015, 0.01, 2.0e-6, -1.0e-6, 1.0e-6], dtype=float)
    times = np.arange(0.0, 1500.0, 120.0)
    arc = sample_relative_od_observations(
        dynamics_model="ya",
        initial_relative_state_ric=truth,
        times_s=times,
        measurement_model="relative_angles_range_rate",
        noise_diag=np.array([1.0e-8, 1.0e-8, 1.0e-6, 1.0e-10], dtype=float),
        chief_initial_state_eci_km_s=chief,
    )

    comparison = compare_relative_od_models(
        arc,
        models=("hcw", "th", "ya"),
        initial_guess_ric=guess,
        output_dir=tmp_path / "comparison",
        fit_duration_s=600.0,
        holdout_duration_s=720.0,
        mean_motion_rad_s=mean_motion,
        chief_initial_state_eci_km_s=chief,
        integration_substep_s=5.0,
        max_nfev=24,
        eci_baseline={
            "fit_weighted_rms": 0.0,
            "holdout_weighted_rms": 0.0,
            "note": "placeholder external ECI batch OD baseline row",
        },
    )
    rows = {row["model"]: row for row in comparison["rows"]}

    assert set(rows) == {"hcw", "th", "ya", "eci_baseline"}
    assert rows["hcw"]["holdout_weighted_rms"] > rows["th"]["holdout_weighted_rms"] * 10.0
    assert rows["th"]["holdout_weighted_rms"] < 1.0e-4
    assert rows["ya"]["holdout_weighted_rms"] < 1.0e-5
    report = json.loads((tmp_path / "comparison" / "relative_od_comparison_report.json").read_text(encoding="utf-8"))
    assert report["method"] == "relative_od_model_comparison"
    assert (tmp_path / "comparison" / "relative_od_comparison_report.md").exists()


def test_relative_od_residual_csv_uses_full_correlated_covariance(tmp_path: Path) -> None:
    n = _circular_mean_motion()
    truth = np.array([0.08, -0.12, 0.035, 2.0e-5, -1.0e-5, 4.0e-6])
    times = np.arange(0.0, 600.0 + 1.0, 120.0)
    sampled = sample_relative_od_observations(
        dynamics_model="hcw",
        initial_relative_state_ric=truth,
        times_s=times,
        measurement_model="relative_state",
        noise_diag=np.array([1.0e-8] * 3 + [1.0e-12] * 3),
        mean_motion_rad_s=n,
    )
    measurements = [observation.vector.copy() for observation in sampled.observations]
    measurements[-1][:2] += np.array([2.0e-4, -1.0e-4])
    covariance = np.diag([1.0e-8] * 3 + [1.0e-12] * 3)
    covariance[0, 1] = covariance[1, 0] = 0.75e-8
    arc = make_relative_od_arc(
        times_s=times,
        measurements=measurements,
        measurement_model="relative_state",
        noise_diag=np.diag(covariance),
        covariance_blocks=covariance,
    )
    output = tmp_path / "correlated"
    solve_batch_relative_od(
        arc,
        dynamics_model="hcw",
        initial_guess_ric=truth,
        output_dir=output,
        fit_duration_s=360.0,
        holdout_duration_s=240.0,
        mean_motion_rad_s=n,
    )

    rows = list(csv.DictReader((output / "relative_od_holdout_residuals.csv").open()))
    terminal_rows = rows[-6:]
    residual = np.array([float(row["residual"]) for row in terminal_rows])
    reported = np.array([float(row["weighted_residual"]) for row in terminal_rows])
    expected = np.linalg.solve(np.linalg.cholesky(covariance), residual)
    np.testing.assert_allclose(reported, expected, rtol=1.0e-12, atol=1.0e-12)
    assert float(terminal_rows[0]["sigma"]) == np.sqrt(covariance[0, 0])


def integrated_relative_case(*, maneuver: bool = False):
    times = np.arange(0.0, 3600.0 + 1.0, 300.0)
    chief = np.array([7000.0, 0.0, 0.0, 0.0, 7.546, 0.1])
    relative = np.array([1.0, 0.2, 0.1, 0.0, 1.0e-4, 0.0])
    truth = generate_nonlinear_relative_truth(
        times_s=times,
        chief_initial_state_eci_km_s=chief,
        initial_relative_state_ric=relative,
        maneuver_index=7 if maneuver else None,
        maneuver_delta_v_ric_m_s=[0.0, 0.25, 0.0] if maneuver else None,
    )
    sigma = np.array([1.0e-3] * 3 + [1.0e-6] * 3)
    rng = np.random.default_rng(82)
    measurements = truth.relative_states_ric + rng.normal(size=(times.size, 6)) * sigma
    sensor_covariance = np.diag(sigma**2)
    chief_covariance = np.diag([1.0e-4**2] * 3 + [1.0e-7**2] * 3)
    guess = relative + np.array([0.01, -0.01, 0.005, 1.0e-5, -1.0e-5, 0.0])
    return times, chief, relative, truth, measurements, sensor_covariance, chief_covariance, guess


def test_phase8_all_models_report_nonlinear_withheld_disagreement_and_chief_uncertainty() -> None:
    times, chief, relative, truth, measurements, sensor_covariance, chief_covariance, guess = integrated_relative_case()
    report = run_integrated_relative_od(
        times_s=times,
        measurements_ric=measurements,
        sensor_covariance=sensor_covariance,
        chief_initial_state_eci_km_s=chief,
        chief_initial_covariance=chief_covariance,
        initial_guess_ric=guess,
        truth_initial_relative_state_ric=relative,
        fit_duration_s=2400.0,
        nonlinear_truth=truth,
    )
    rows = report["nonlinear_baseline"]["rows"]
    assert {row["model"] for row in rows} == {"hcw", "ss_j2", "th", "ya"}
    assert all(row["holdout"]["available"] for row in rows)
    assert report["selected_model_by_withheld_nonlinear_position_rms"] in {"hcw", "ss_j2", "th", "ya"}
    assert report["selection_policy"].startswith("rank by withheld nonlinear-baseline")
    assert report["recommended_model_inside_validated_envelope"] is None
    assert report["chief_uncertainty"]["chief_not_treated_as_perfect"] is True
    assert report["chief_uncertainty"]["initial_effective_covariance_trace"] > report["chief_uncertainty"][
        "initial_sensor_covariance_trace"
    ]
    assert all(
        row["covariance_symmetric_psd"] for row in report["sequential_model_comparison"].values()
    )
