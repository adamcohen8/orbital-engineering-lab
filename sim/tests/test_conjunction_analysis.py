from __future__ import annotations

import copy
import math

import numpy as np
import pytest
from scipy.integrate import dblquad

from sim.analysis.conjunction_geometry import (
    StateHistory,
    encounter_frame,
    interpolate_history,
    refine_time_of_closest_approach,
)
from sim.analysis.conjunction_probability import (
    ConjunctionProbabilityError,
    collision_probability_2d,
    project_combined_covariance,
    small_object_collision_probability,
)
from sim.analysis.conjunction_workflow import (
    ConjunctionAssessmentError,
    ConjunctionObject,
    assess_conjunction,
    assess_histories,
    propagate_history,
)
from sim.analysis.trajectory_targeting import PropagationSettings


def test_tca_refinement_recovers_linear_analytic_solution() -> None:
    times = [0.0, 10.0]
    primary = StateHistory.from_arrays(times, [[-5.0, 2.0, 0.0, 1.0, 0.0, 0.0], [5.0, 2.0, 0.0, 1.0, 0.0, 0.0]])
    secondary = StateHistory.from_arrays(times, [[0.0] * 6, [0.0] * 6])
    result = refine_time_of_closest_approach(primary, secondary)
    assert result["time_s"] == pytest.approx(5.0, abs=1.0e-6)
    assert result["miss_distance_km"] == pytest.approx(2.0, abs=1.0e-12)
    assert result["relative_speed_km_s"] == pytest.approx(1.0)


def test_tca_refinement_agrees_with_independent_dense_sampling() -> None:
    times = [0.0, 4.0, 10.0]
    primary = StateHistory.from_arrays(
        times,
        [[-4.0, 1.5, 0.2, 1.0, -0.1, 0.0], [0.1, 1.2, 0.2, 1.05, -0.05, 0.0], [6.2, 1.1, 0.2, 0.95, 0.01, 0.0]],
    )
    secondary = StateHistory.from_arrays(times, [[0.0] * 6, [0.0] * 6, [0.0] * 6])
    refined = refine_time_of_closest_approach(primary, secondary)
    from sim.analysis.conjunction_geometry import interpolate_history

    dense_times = np.linspace(0.0, 10.0, 200_001)
    dense_distances = np.array([np.linalg.norm(interpolate_history(primary, value)[:3]) for value in dense_times])
    dense_index = int(np.argmin(dense_distances))
    assert refined["time_s"] == pytest.approx(float(dense_times[dense_index]), abs=5.1e-5)
    assert refined["miss_distance_km"] == pytest.approx(float(dense_distances[dense_index]), abs=1.0e-8)


def test_tca_refinement_evaluates_repeated_minima_in_one_interval() -> None:
    primary = StateHistory.from_arrays(
        [0.0, 10.0],
        [[-0.08, 1.0, 0.0, 0.066, 0.0, 0.0], [0.08, 1.0, 0.0, 0.066, 0.0, 0.0]],
    )
    secondary = StateHistory.from_arrays([0.0, 10.0], [[0.0] * 6, [0.0] * 6])
    refined = refine_time_of_closest_approach(primary, secondary)
    assert refined["time_s"] == pytest.approx(2.0, abs=1.0e-9)
    assert refined["miss_distance_km"] == pytest.approx(1.0, abs=1.0e-14)
    assert refined["resources"]["winning_interval_stationary_roots"] == 5


def test_impulsive_burn_does_not_change_preburn_interpolation() -> None:
    initial_state = [7000.0, 0.0, 0.0, 0.0, 7.546, 0.0]
    settings = PropagationSettings.from_mapping({"step_s": 10.0, "integrator": "rk4"})
    baseline = propagate_history(initial_state, 30.0, settings)
    maneuvered = propagate_history(
        initial_state,
        30.0,
        settings,
        burn_time_s=10.0,
        burn_frame="eci",
        delta_v_m_s=[100.0, 0.0, 0.0],
    )

    assert interpolate_history(maneuvered, 5.0) == pytest.approx(interpolate_history(baseline, 5.0), abs=1.0e-14)
    assert interpolate_history(maneuvered, 10.0, side="left") == pytest.approx(
        interpolate_history(baseline, 10.0), abs=1.0e-14
    )
    assert (
        interpolate_history(maneuvered, 10.0, side="right")[3]
        - interpolate_history(maneuvered, 10.0, side="left")[3]
    ) == pytest.approx(0.1, abs=1.0e-14)


def test_tca_boundary_result_uses_the_selected_preimpulse_state() -> None:
    times = [0.0, 10.0, 20.0]
    primary = StateHistory.from_arrays(
        times,
        [
            [20.0, 0.0, 0.0, -1.0, 0.0, 0.0],
            [10.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [20.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        ],
        incoming_velocities_eci_km_s=[[-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
    )
    secondary = StateHistory.from_arrays(times, [[0.0] * 6] * 3)

    result = refine_time_of_closest_approach(primary, secondary, numeric_backend="python")

    assert result["time_s"] == pytest.approx(10.0)
    assert result["relative_position_eci_km"] == pytest.approx([10.0, 0.0, 0.0])
    assert result["relative_velocity_eci_km_s"] == pytest.approx([-1.0, 0.0, 0.0])
    assert result["relative_position_dot_velocity_km2_s"] == pytest.approx(-10.0)
    assert result["relative_range_rate_km_s"] == pytest.approx(-1.0)
    assert result["miss_distance_km"] == pytest.approx(np.linalg.norm(result["relative_position_eci_km"]))


def test_ric_burn_rejects_zero_angular_momentum() -> None:
    settings = PropagationSettings.from_mapping({"step_s": 10.0})
    with pytest.raises(ConjunctionAssessmentError, match="nonzero angular momentum"):
        propagate_history(
            [7000.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            20.0,
            settings,
            burn_time_s=10.0,
            burn_frame="ric",
            delta_v_m_s=[1.0, 0.0, 0.0],
        )


def test_encounter_frame_is_orthonormal_and_places_miss_on_x() -> None:
    result = encounter_frame([1.0, 2.0, 3.0], [0.0, 10.0, 0.0])
    basis = np.asarray(result["basis_rows_eci"])
    assert basis @ basis.T == pytest.approx(np.eye(3), abs=1.0e-14)
    assert result["plane_miss_km"][1] == pytest.approx(0.0, abs=1.0e-14)


def test_2d_pc_matches_isotropic_zero_mean_closed_form() -> None:
    sigma_km = 0.1
    radius_km = 0.02
    result = collision_probability_2d([0.0, 0.0], np.eye(2) * sigma_km**2, radius_km)
    expected = 1.0 - math.exp(-(radius_km**2) / (2.0 * sigma_km**2))
    assert result["collision_probability"] == pytest.approx(expected, rel=1.0e-11, abs=1.0e-14)
    assert result["quadrature"]["absolute_convergence_estimate"] <= result["quadrature"]["acceptance_tolerance"]


def test_2d_pc_matches_independent_adaptive_integration() -> None:
    mean = np.array([0.025, -0.012])
    covariance = np.array([[0.0016, 0.00024], [0.00024, 0.0009]])
    radius = 0.015
    inverse = np.linalg.inv(covariance)
    normalization = 1.0 / (2.0 * math.pi * math.sqrt(float(np.linalg.det(covariance))))

    def density(y: float, x: float) -> float:
        offset = np.array([x, y]) - mean
        return normalization * math.exp(-0.5 * float(offset.T @ inverse @ offset))

    reference, reference_error = dblquad(
        density,
        -radius,
        radius,
        lambda x: -math.sqrt(max(radius**2 - x**2, 0.0)),
        lambda x: math.sqrt(max(radius**2 - x**2, 0.0)),
        epsabs=1.0e-12,
        epsrel=1.0e-11,
    )
    result = collision_probability_2d(mean, covariance, radius)
    assert reference_error < 1.0e-10
    assert result["collision_probability"] == pytest.approx(reference, rel=2.0e-10, abs=1.0e-12)


def test_2d_pc_resolves_concentrated_probability_mass_inside_hard_body() -> None:
    result = collision_probability_2d([0.005, 0.0], np.eye(2) * 1.0e-10, 0.01)
    assert result["collision_probability"] == pytest.approx(1.0, abs=1.0e-12)
    assert result["quadrature"]["absolute_convergence_estimate"] <= result["quadrature"]["acceptance_tolerance"]


def test_2d_pc_fails_closed_when_adaptive_error_is_not_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "sim.analysis.conjunction_probability.quad",
        lambda *args, **kwargs: (0.5, 1.0, {}),
    )
    with pytest.raises(ConjunctionProbabilityError, match="failed closed"):
        collision_probability_2d([0.0, 0.0], np.eye(2) * 0.01, 0.02)


@pytest.mark.parametrize("radius", [0.0, -0.01, float("nan"), float("inf")])
def test_legacy_small_object_probability_rejects_invalid_hard_body_radius(radius: float) -> None:
    with pytest.raises(ConjunctionProbabilityError, match="Hard-body radius must be positive and finite"):
        small_object_collision_probability([0.0, 0.0], np.eye(2), radius)


def test_covariance_projection_rejects_non_psd_inputs() -> None:
    bad = np.eye(6)
    bad[0, 0] = -1.0
    with pytest.raises(ConjunctionProbabilityError, match="positive semidefinite"):
        project_combined_covariance(bad, np.eye(6), np.eye(3))


def test_boundary_closest_approach_is_incomplete_and_withholds_pc() -> None:
    covariance = (np.eye(6) * 1.0e-4).tolist()
    primary = ConjunctionObject.from_mapping(
        {
            "object_id": "PRIMARY",
            "initial_state_eci_km_km_s": [7100.0, 0.0, 0.0, -1.0, 0.0, 0.0],
            "covariance_eci_km_at_tca": covariance,
            "hard_body_radius_m": 5.0,
        }
    )
    secondary = ConjunctionObject.from_mapping(
        {
            "object_id": "SECONDARY",
            "initial_state_eci_km_km_s": [7000.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "covariance_eci_km_at_tca": covariance,
            "hard_body_radius_m": 5.0,
        }
    )
    primary_history = StateHistory.from_arrays(
        [0.0, 10.0],
        [[7100.0, 0.0, 0.0, -1.0, 0.0, 0.0], [7090.0, 0.0, 0.0, -1.0, 0.0, 0.0]],
    )
    secondary_history = StateHistory.from_arrays(
        [0.0, 10.0],
        [[7000.0, 0.0, 0.0, 0.0, 0.0, 0.0], [7000.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
    )

    result = assess_histories(primary, secondary, primary_history, secondary_history)
    assert result["status"] == "incomplete_search_window"
    assert result["probability"] is None
    assert result["encounter_frame"] is None
    assert result["closest_approach"]["miss_distance_km"] == pytest.approx(90.0)
    assert result["closest_approach"]["relative_position_dot_velocity_km2_s"] == pytest.approx(-90.0)
    assert result["closest_approach"]["relative_range_rate_km_s"] == pytest.approx(-1.0)

    workflow = assess_conjunction(
        {
            "schema_version": "oel.conjunction_assessment_problem.v1",
            "name": "boundary_window",
            "duration_s": 20.0,
            "propagation": {"step_s": 10.0, "integrator": "rk4"},
            "primary": primary.to_dict(),
            "secondary": secondary.to_dict(),
            "avoidance_candidates": [],
        }
    )
    assert workflow["status"] == "incomplete_search_window"
    assert workflow["baseline"]["probability"] is None


def test_avoidance_candidate_is_targeted_repropagated_and_secondarily_rescreened() -> None:
    covariance = (np.eye(6) * 1.0e-4).tolist()
    common = {"covariance_eci_km_at_tca": covariance, "hard_body_radius_m": 5.0}
    problem = {
        "schema_version": "oel.conjunction_assessment_problem.v1",
        "name": "synthetic_crossing",
        "duration_s": 240.0,
        "propagation": {"step_s": 10.0, "integrator": "rk4"},
        "primary": {**common, "object_id": "PRIMARY", "initial_state_eci_km_km_s": [7000.0, 0.0, 0.0, 0.0, 7.546, 0.0]},
        "secondary": {
            **common,
            "object_id": "SECONDARY",
            "initial_state_eci_km_km_s": [7000.02, -1.0, 0.0, 0.0, 7.556, 0.0],
        },
        "screening_objects": [
            {**common, "object_id": "SCREEN-2", "initial_state_eci_km_km_s": [7000.5, -1.0, 0.0, 0.0, 7.556, 0.001]}
        ],
        "avoidance_candidates": [
            {
                "name": "radial_terminal_offset",
                "burn_time_s": 0.0,
                "frame": "ric",
                "burn_component": "r",
                "terminal_quantity": "position_x_km",
                "target_offset": 0.05,
                "tolerance": 1.0e-5,
                "initial_delta_v_m_s": 0.0,
                "perturbation_m_s": 1.0e-3,
                "max_abs_delta_v_m_s": 10.0,
            }
        ],
    }
    evidence = assess_conjunction(problem)
    candidate = evidence["avoidance_candidates"][0]
    assert evidence["baseline"]["closest_approach"]["miss_distance_km"] == pytest.approx(0.0199211, rel=1.0e-4)
    assert candidate["targeter"]["converged"] is True
    assert candidate["assessment_completed"] is True
    assert candidate["risk_disposition"] == "not_evaluated_no_acceptance_criteria"
    assert "accepted" not in candidate
    assert (
        candidate["assessment"]["closest_approach"]["miss_distance_km"]
        > evidence["baseline"]["closest_approach"]["miss_distance_km"]
    )
    assert len(candidate["secondary_rescreen"]) == 1
    assert candidate["authoritative_history_continuity"]["max_abs_state_difference"] < 1.0e-6

    rank_deficient = copy.deepcopy(problem)
    rank_deficient["avoidance_candidates"][0]["burn_component"] = "c"
    failed_candidate = assess_conjunction(rank_deficient)["avoidance_candidates"][0]
    assert failed_candidate["assessment_completed"] is False
    assert failed_candidate["disposition"] == "targeter_not_converged"
    assert failed_candidate["targeter"]["status"] == "rank_deficient"

    late = copy.deepcopy(problem)
    late["avoidance_candidates"][0]["burn_time_s"] = 150.0
    late_candidate = assess_conjunction(late)["avoidance_candidates"][0]
    assert late_candidate["assessment_completed"] is False
    assert late_candidate["disposition"] == "invalid_candidate"
    assert "must precede baseline TCA" in late_candidate["message"]

    adverse = copy.deepcopy(problem)
    adverse["screening_objects"] = []
    adverse["avoidance_candidates"][0]["target_offset"] = 0.02
    adverse_evidence = assess_conjunction(adverse)
    adverse_candidate = adverse_evidence["avoidance_candidates"][0]
    assert adverse_candidate["assessment_completed"] is True
    assert adverse_candidate["risk_disposition"] == "not_evaluated_no_acceptance_criteria"
    assert "accepted" not in adverse_candidate
    assert (
        adverse_candidate["assessment"]["closest_approach"]["miss_distance_km"]
        < adverse_evidence["baseline"]["closest_approach"]["miss_distance_km"]
    )
