from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from sim.dynamics.orbit.environment import EARTH_MU_KM3_S2
from sim.trajectory_design import (
    TRAJECTORY_TARGETING_EVIDENCE_SCHEMA,
    TrajectoryTargetingError,
    TrajectoryTargetingProblem,
    execute_trajectory,
    finite_difference_jacobian,
    main,
    solve_trajectory_target,
    write_trajectory_targeting_evidence,
)


def circular_state(radius_km: float = 7000.0) -> list[float]:
    return [radius_km, 0.0, 0.0, 0.0, math.sqrt(EARTH_MU_KM3_S2 / radius_km), 0.0]


def problem_from_dict(data: dict) -> TrajectoryTargetingProblem:
    return TrajectoryTargetingProblem.from_mapping(
        {
            "schema_version": "oel.trajectory_targeting_problem.v1",
            "name": "test_problem",
            "initial_state_eci_km_km_s": circular_state(),
            "propagation": {"step_s": 10.0},
            "segments": [{"type": "coast", "name": "coast", "duration_s": 10.0}],
            "constraints": [{"name": "elapsed", "quantity": "elapsed_time_s", "target": 10.0, "tolerance": 1.0e-9}],
            **data,
        }
    )


@pytest.mark.parametrize(
    "field",
    ["mu_km3_s2", "central_body_radius_km", "mass_kg", "event_time_tolerance_s", "event_value_tolerance"],
)
def test_propagation_settings_reject_nonfinite_values(field: str) -> None:
    with pytest.raises(TrajectoryTargetingError, match="positive and finite"):
        problem_from_dict({"propagation": {field: float("nan")}})


def test_targeting_rejects_unknown_fields_at_each_mapping_boundary() -> None:
    cases = [
        {"unexpected": 1.0},
        {"propagation": {"unexpected": 1.0}},
        {"solver": {"unexpected": 1.0}},
        {"segments": [{"type": "coast", "name": "coast", "duration_s": 10.0, "unexpected": 1.0}]},
        {
            "variables": [
                {
                    "name": "duration",
                    "segment": "coast",
                    "field": "duration_s",
                    "initial": 10.0,
                    "perturbation": 0.1,
                    "unexpected": 1.0,
                }
            ]
        },
        {
            "constraints": [
                {
                    "name": "elapsed",
                    "quantity": "elapsed_time_s",
                    "target": 10.0,
                    "tolerance": 1.0e-9,
                    "unexpected": 1.0,
                }
            ]
        },
        {
            "segments": [
                {
                    "type": "coast",
                    "name": "coast",
                    "stop": {
                        "quantity": "elapsed_time_s",
                        "target": 10.0,
                        "max_duration_s": 20.0,
                        "unexpected": 1.0,
                    },
                }
            ]
        },
    ]
    for case in cases:
        with pytest.raises(TrajectoryTargetingError, match="unknown fields"):
            problem_from_dict(case)


def test_targeting_rejects_booleans_before_numeric_conversion() -> None:
    cases = [
        {"initial_state_eci_km_km_s": [True, 0.0, 0.0, 0.0, 7.5, 0.0]},
        {"propagation": {"step_s": True}},
        {"segments": [{"type": "coast", "name": "coast", "duration_s": True}]},
        {
            "segments": [
                {"type": "impulsive_burn", "name": "burn", "delta_v_m_s": [True, 0.0, 0.0]}
            ]
        },
        {
            "variables": [
                {
                    "name": "duration",
                    "segment": "coast",
                    "field": "duration_s",
                    "initial": True,
                    "perturbation": 0.1,
                }
            ]
        },
        {
            "constraints": [
                {"name": "elapsed", "quantity": "elapsed_time_s", "target": True, "tolerance": 1.0e-9}
            ]
        },
        {"solver": {"max_iterations": True}},
        {
            "segments": [
                {
                    "type": "coast",
                    "name": "coast",
                    "stop": {
                        "quantity": "elapsed_time_s",
                        "target": 10.0,
                        "max_duration_s": True,
                    },
                }
            ]
        },
    ]
    for case in cases:
        with pytest.raises(TrajectoryTargetingError, match="boolean"):
            problem_from_dict(case)


@pytest.mark.parametrize("integrator", ["rkf78", "dopri5", "adaptive"])
def test_rust_direct_execution_rejects_non_rk4_before_native_probe(
    monkeypatch: pytest.MonkeyPatch, integrator: str
) -> None:
    import sim.rust_targeting_backend as native_targeting

    def fail_if_native_is_requested() -> None:
        pytest.fail("unsupported Rust targeting execution probed the native extension")

    monkeypatch.setattr(native_targeting, "_extension", fail_if_native_is_requested)
    problem = problem_from_dict({"propagation": {"step_s": 10.0, "integrator": integrator}})

    with pytest.raises(
        TrajectoryTargetingError,
        match=r"Rust direct trajectory execution supports only.*numeric_backend='python'",
    ):
        execute_trajectory(problem, numeric_backend="rust")


def test_python_backend_retains_explicit_adaptive_execution(monkeypatch: pytest.MonkeyPatch) -> None:
    import sim.rust_targeting_backend as native_targeting

    def fail_if_native_is_requested() -> None:
        pytest.fail("Python-selected adaptive targeting probed the native extension")

    monkeypatch.setattr(native_targeting, "_extension", fail_if_native_is_requested)
    problem = problem_from_dict({"propagation": {"step_s": 10.0, "integrator": "adaptive"}})

    execution = execute_trajectory(problem, numeric_backend="python")

    assert execution["status"] == "completed"
    assert execution["numeric_backend"] == "python"
    assert execution["propagator"]["integrator"] == "adaptive"


def test_execute_multisegment_coast_burn_coast_uses_onp_and_records_resources() -> None:
    problem = problem_from_dict(
        {
            "segments": [
                {"type": "coast", "name": "before", "duration_s": 120.0},
                {
                    "type": "impulsive_burn",
                    "name": "burn",
                    "frame": "ric",
                    "delta_v_m_s": [0.0, 10.0, 0.0],
                },
                {"type": "coast", "name": "after", "duration_s": 180.0},
            ],
            "constraints": [{"name": "elapsed", "quantity": "elapsed_time_s", "target": 300.0, "tolerance": 1.0e-9}],
        }
    )

    execution = execute_trajectory(problem)

    assert execution["status"] == "completed"
    assert execution["elapsed_time_s"] == pytest.approx(300.0)
    assert [segment["type"] for segment in execution["segments"]] == ["coast", "impulsive_burn", "coast"]
    assert execution["resources"] == {
        "segment_count": 3,
        "coast_segment_count": 2,
        "burn_count": 1,
        "propagation_steps": 30,
        "total_delta_v_m_s": pytest.approx(10.0),
        "coast_time_s": pytest.approx(300.0),
    }
    assert execution["propagator"]["propagator_family"] == "ONP"
    assert execution["propagator"]["native_frame"] == "eci"


def test_radial_velocity_event_stops_at_apoapsis() -> None:
    initial = circular_state()
    initial[4] += 0.25
    problem = problem_from_dict(
        {
            "initial_state_eci_km_km_s": initial,
            "segments": [
                {
                    "type": "coast",
                    "name": "to_apoapsis",
                    "stop": {
                        "quantity": "radial_velocity_km_s",
                        "target": 0.0,
                        "direction": "decreasing",
                        "minimum_elapsed_s": 60.0,
                        "max_duration_s": 4000.0,
                    },
                }
            ],
            "constraints": [
                {
                    "name": "at_apoapsis",
                    "quantity": "radial_velocity_km_s",
                    "target": 0.0,
                    "tolerance": 1.0e-7,
                }
            ],
        }
    )

    execution = execute_trajectory(problem)
    event = execution["segments"][0]["stop_event"]

    assert event["status"] == "found"
    assert event["direction"] == "decreasing"
    assert event["elapsed_in_segment_s"] > 1000.0
    assert abs(event["residual"]) < 1.0e-7


def test_single_shooting_recovers_analytic_hohmann_departure_burn() -> None:
    initial_radius_km = 7000.0
    final_radius_km = 9000.0
    circular_speed = math.sqrt(EARTH_MU_KM3_S2 / initial_radius_km)
    transfer_speed = math.sqrt(
        EARTH_MU_KM3_S2 * (2.0 / initial_radius_km - 2.0 / (initial_radius_km + final_radius_km))
    )
    expected_departure_m_s = 1000.0 * (transfer_speed - circular_speed)
    problem = problem_from_dict(
        {
            "name": "hohmann_departure_target",
            "segments": [
                {
                    "type": "impulsive_burn",
                    "name": "departure",
                    "frame": "ric",
                    "delta_v_m_s": [0.0, 0.0, 0.0],
                },
                {
                    "type": "coast",
                    "name": "to_apoapsis",
                    "stop": {
                        "quantity": "radial_velocity_km_s",
                        "target": 0.0,
                        "direction": "decreasing",
                        "minimum_elapsed_s": 60.0,
                        "max_duration_s": 5000.0,
                    },
                },
            ],
            "variables": [
                {
                    "name": "departure_intrack_m_s",
                    "segment": "departure",
                    "field": "delta_v_i_m_s",
                    "initial": 300.0,
                    "perturbation": 0.1,
                }
            ],
            "constraints": [
                {
                    "name": "apoapsis_radius",
                    "quantity": "radius_km",
                    "target": final_radius_km,
                    "tolerance": 0.01,
                }
            ],
        }
    )

    evidence = solve_trajectory_target(problem)

    assert evidence["status"] == "converged"
    assert evidence["converged"] is True
    assert evidence["decision_values"][0] == pytest.approx(expected_departure_m_s, abs=0.01)
    assert evidence["authoritative_repropagation"]["status"] == "verified"
    assert evidence["authoritative_repropagation"]["constraint_evaluation"]["all_satisfied"] is True
    assert evidence["authoritative_repropagation"]["final_position_difference_norm_km"] == 0.0
    assert evidence["resources"]["jacobian_evaluations"] >= 1
    assert evidence["convergence_history"][0]["jacobian_rank"] == 1


def test_plane_change_targets_inclination_with_cross_track_impulse() -> None:
    target_inclination_deg = 5.0
    speed_km_s = circular_state()[4]
    analytic_burn_m_s = 1000.0 * speed_km_s * math.tan(math.radians(target_inclination_deg))
    problem = problem_from_dict(
        {
            "segments": [
                {
                    "type": "impulsive_burn",
                    "name": "plane_change",
                    "frame": "ric",
                    "delta_v_m_s": [0.0, 0.0, 0.0],
                }
            ],
            "variables": [
                {
                    "name": "cross_track_m_s",
                    "segment": "plane_change",
                    "field": "delta_v_c_m_s",
                    "initial": 500.0,
                    "perturbation": 0.1,
                }
            ],
            "constraints": [
                {
                    "name": "inclination",
                    "quantity": "inclination_deg",
                    "target": target_inclination_deg,
                    "tolerance": 1.0e-6,
                }
            ],
        }
    )

    evidence = solve_trajectory_target(problem)

    assert evidence["converged"] is True
    assert evidence["decision_values"][0] == pytest.approx(analytic_burn_m_s, abs=1.0e-3)


def test_timing_target_changes_fixed_coast_duration() -> None:
    problem = problem_from_dict(
        {
            "segments": [{"type": "coast", "name": "timed_coast", "duration_s": 80.0}],
            "variables": [
                {
                    "name": "duration",
                    "segment": "timed_coast",
                    "field": "duration_s",
                    "initial": 80.0,
                    "perturbation": 0.01,
                }
            ],
            "constraints": [
                {"name": "arrival_time", "quantity": "elapsed_time_s", "target": 100.0, "tolerance": 1.0e-8}
            ],
        }
    )

    evidence = solve_trajectory_target(problem)

    assert evidence["converged"] is True
    assert evidence["decision_values"] == pytest.approx([100.0], abs=1.0e-9)


def test_cartesian_velocity_target_and_jacobian_have_expected_units() -> None:
    initial_speed = circular_state()[4]
    problem = problem_from_dict(
        {
            "segments": [
                {
                    "type": "impulsive_burn",
                    "name": "burn",
                    "frame": "eci",
                    "delta_v_m_s": [0.0, 0.0, 0.0],
                }
            ],
            "variables": [
                {
                    "name": "dv_y",
                    "segment": "burn",
                    "field": "delta_v_y_m_s",
                    "initial": 0.0,
                    "perturbation": 0.01,
                }
            ],
            "constraints": [
                {
                    "name": "vy",
                    "quantity": "velocity_y_km_s",
                    "target": initial_speed + 0.01,
                    "tolerance": 1.0e-6,
                }
            ],
        }
    )

    jacobian, accounting = finite_difference_jacobian(problem, [0.0])
    evidence = solve_trajectory_target(problem)

    assert jacobian.shape == (1, 1)
    assert jacobian[0, 0] == pytest.approx(1000.0, rel=1.0e-9)
    assert accounting == {
        "trajectory_evaluations": 2,
        "propagation_steps": 0,
        "effective_perturbations": [0.01],
    }
    assert evidence["converged"] is True
    assert evidence["decision_values"] == pytest.approx([10.0], abs=1.0e-9)


def test_two_body_coast_conserves_energy_and_angular_momentum_and_is_reversible() -> None:
    period_s = 2.0 * math.pi * math.sqrt(7000.0**3 / EARTH_MU_KM3_S2)
    forward = problem_from_dict(
        {
            "propagation": {"step_s": 1.0},
            "segments": [{"type": "coast", "name": "orbit", "duration_s": period_s}],
            "constraints": [{"name": "period", "quantity": "elapsed_time_s", "target": period_s, "tolerance": 1.0e-9}],
        }
    )
    execution = execute_trajectory(forward)
    initial = np.asarray(forward.initial_state_eci_km_km_s)
    final = np.asarray(execution["final_state_eci_km_km_s"])

    def invariants(state: np.ndarray) -> tuple[float, float]:
        energy = 0.5 * float(np.dot(state[3:], state[3:])) - EARTH_MU_KM3_S2 / float(np.linalg.norm(state[:3]))
        angular_momentum = float(np.linalg.norm(np.cross(state[:3], state[3:])))
        return energy, angular_momentum

    initial_energy, initial_h = invariants(initial)
    final_energy, final_h = invariants(final)
    assert final_energy == pytest.approx(initial_energy, rel=2.0e-12)
    assert final_h == pytest.approx(initial_h, rel=2.0e-12)
    assert np.linalg.norm(final[:3] - initial[:3]) < 1.0e-7
    assert np.linalg.norm(final[3:] - initial[3:]) < 1.0e-10


def test_rank_deficient_problem_fails_with_structured_evidence() -> None:
    problem = problem_from_dict(
        {
            "segments": [
                {
                    "type": "impulsive_burn",
                    "name": "burn",
                    "frame": "eci",
                    "delta_v_m_s": [0.0, 0.0, 0.0],
                }
            ],
            "variables": [
                {"name": "x", "segment": "burn", "field": "delta_v_x_m_s", "initial": 0.0, "perturbation": 0.1},
                {"name": "y", "segment": "burn", "field": "delta_v_y_m_s", "initial": 0.0, "perturbation": 0.1},
            ],
            "constraints": [
                {"name": "radius_a", "quantity": "radius_km", "target": 7100.0, "tolerance": 1.0},
                {"name": "radius_b", "quantity": "radius_km", "target": 7200.0, "tolerance": 1.0},
            ],
        }
    )

    evidence = solve_trajectory_target(problem)

    assert evidence["status"] == "rank_deficient"
    assert evidence["converged"] is False
    assert evidence["convergence_history"][0]["jacobian_rank"] == 0


def test_missed_event_and_nonconvergence_are_not_reported_as_success() -> None:
    missed = problem_from_dict(
        {
            "segments": [
                {
                    "type": "impulsive_burn",
                    "name": "burn",
                    "frame": "ric",
                    "delta_v_m_s": [0.0, 0.0, 0.0],
                },
                {
                    "type": "coast",
                    "name": "never_reached",
                    "stop": {
                        "quantity": "radius_km",
                        "target": 10000.0,
                        "direction": "increasing",
                        "minimum_elapsed_s": 0.0,
                        "max_duration_s": 100.0,
                    },
                },
            ],
            "variables": [
                {
                    "name": "departure",
                    "segment": "burn",
                    "field": "delta_v_i_m_s",
                    "initial": 0.0,
                    "perturbation": 0.1,
                }
            ],
            "constraints": [{"name": "radius", "quantity": "radius_km", "target": 10000.0, "tolerance": 1.0}],
        }
    )
    limited = problem_from_dict(
        {
            "segments": [
                {
                    "type": "impulsive_burn",
                    "name": "burn",
                    "frame": "eci",
                    "delta_v_m_s": [0.0, 0.0, 0.0],
                },
                {"type": "coast", "name": "coast", "duration_s": 1000.0},
            ],
            "variables": [
                {"name": "x", "segment": "burn", "field": "delta_v_x_m_s", "initial": 0.0, "perturbation": 0.1}
            ],
            "constraints": [{"name": "x", "quantity": "position_x_km", "target": 8000.0, "tolerance": 1.0e-12}],
            "solver": {"max_iterations": 1},
        }
    )

    missed_evidence = solve_trajectory_target(missed)
    limited_evidence = solve_trajectory_target(limited)

    assert missed_evidence["status"] == "missed_event"
    assert missed_evidence["best_execution"]["event"]["status"] == "missed"
    assert missed_evidence["resources"] == {
        "trajectory_evaluations": 1,
        "jacobian_evaluations": 0,
        "propagation_steps": 10,
    }
    assert limited_evidence["status"] == "non_convergent"
    assert limited_evidence["converged"] is False


def test_event_search_never_refines_before_minimum_elapsed_boundary() -> None:
    problem = problem_from_dict(
        {
            "segments": [
                {
                    "type": "coast",
                    "name": "guarded_event",
                    "stop": {
                        "quantity": "elapsed_time_s",
                        "target": 2.0,
                        "direction": "increasing",
                        "minimum_elapsed_s": 5.0,
                        "max_duration_s": 20.0,
                    },
                }
            ],
            "constraints": [
                {"name": "time", "quantity": "elapsed_time_s", "target": 2.0, "tolerance": 10.0}
            ],
        }
    )

    evidence = solve_trajectory_target(problem)

    assert evidence["status"] == "missed_event"
    assert evidence["best_execution"]["event"]["final_time_s"] == pytest.approx(20.0)
    assert evidence["resources"] == {
        "trajectory_evaluations": 1,
        "jacobian_evaluations": 0,
        "propagation_steps": 3,
    }


def test_event_refinement_iteration_limit_fails_structurally() -> None:
    problem = problem_from_dict(
        {
            "propagation": {
                "step_s": 10.0,
                "event_time_tolerance_s": 1.0e-9,
                "event_value_tolerance": 1.0e-12,
                "event_max_iterations": 1,
            },
            "segments": [
                {
                    "type": "coast",
                    "name": "under_refined",
                    "stop": {
                        "quantity": "elapsed_time_s",
                        "target": 1.0,
                        "direction": "increasing",
                        "minimum_elapsed_s": 0.0,
                        "max_duration_s": 20.0,
                    },
                }
            ],
            "constraints": [
                {"name": "time", "quantity": "elapsed_time_s", "target": 1.0, "tolerance": 10.0}
            ],
        }
    )

    evidence = solve_trajectory_target(problem)

    assert evidence["status"] == "event_refinement_failed"
    assert evidence["converged"] is False
    receipt = evidence["best_execution"]["event"]
    assert receipt["status"] == "refinement_failed"
    assert receipt["bracket_end_s"] - receipt["bracket_start_s"] > 1.0e-9
    assert abs(receipt["residual"]) > 1.0e-12
    assert evidence["resources"]["trajectory_evaluations"] == 1
    assert evidence["resources"]["propagation_steps"] == receipt["propagation_steps"]


def test_true_anomaly_event_crosses_zero_on_continuous_angular_branch() -> None:
    from sim.dynamics.orbit.elements import coe_to_rv_eci

    position, velocity = coe_to_rv_eci(
        a_km=7000.0,
        ecc=0.1,
        inc_deg=20.0,
        raan_deg=30.0,
        argp_deg=40.0,
        true_anomaly_deg=359.0,
    )
    problem = problem_from_dict(
        {
            "initial_state_eci_km_km_s": [*position, *velocity],
            "segments": [
                {
                    "type": "coast",
                    "name": "wrap_event",
                    "stop": {
                        "quantity": "true_anomaly_deg",
                        "target": 0.0,
                        "direction": "increasing",
                        "minimum_elapsed_s": 0.0,
                        "max_duration_s": 100.0,
                    },
                }
            ],
            "constraints": [
                {
                    "name": "anomaly",
                    "quantity": "true_anomaly_deg",
                    "target": 0.0,
                    "tolerance": 1.0e-6,
                }
            ],
        }
    )

    execution = execute_trajectory(problem)
    event = execution["segments"][0]["stop_event"]

    assert event["status"] == "found"
    assert 0.0 < event["elapsed_in_segment_s"] < 100.0
    assert abs(event["residual"]) <= 1.0e-6
    assert event["propagation_steps"] == execution["resources"]["propagation_steps"]


def test_duration_jacobian_reduces_perturbation_to_stay_inside_positive_domain() -> None:
    problem = problem_from_dict(
        {
            "segments": [{"type": "coast", "name": "timed", "duration_s": 1.0}],
            "variables": [
                {
                    "name": "duration",
                    "segment": "timed",
                    "field": "duration_s",
                    "initial": 1.0,
                    "perturbation": 2.0,
                }
            ],
            "constraints": [
                {"name": "time", "quantity": "elapsed_time_s", "target": 2.0, "tolerance": 1.0e-6}
            ],
        }
    )

    jacobian, accounting = finite_difference_jacobian(problem, [1.0])
    evidence = solve_trajectory_target(problem)

    assert accounting["effective_perturbations"] == pytest.approx([0.5])
    assert jacobian[0, 0] == pytest.approx(1.0e6)
    assert evidence["converged"] is True
    assert evidence["decision_values"] == pytest.approx([2.0])
    assert evidence["convergence_history"][0]["effective_perturbations"] == pytest.approx([0.5])


def test_fixed_infeasible_problem_and_underdetermined_problem_fail_closed() -> None:
    fixed = problem_from_dict(
        {"constraints": [{"name": "wrong_time", "quantity": "elapsed_time_s", "target": 20.0, "tolerance": 0.1}]}
    )
    evidence = solve_trajectory_target(fixed)
    assert evidence["status"] == "infeasible"
    assert evidence["converged"] is False

    with pytest.raises(TrajectoryTargetingError, match="at least as many constraints"):
        problem_from_dict(
            {
                "segments": [
                    {
                        "type": "impulsive_burn",
                        "name": "burn",
                        "frame": "eci",
                        "delta_v_m_s": [0.0, 0.0, 0.0],
                    }
                ],
                "variables": [
                    {"name": "x", "segment": "burn", "field": "delta_v_x_m_s", "initial": 0.0, "perturbation": 1.0},
                    {"name": "y", "segment": "burn", "field": "delta_v_y_m_s", "initial": 0.0, "perturbation": 1.0},
                ],
                "constraints": [{"name": "vx", "quantity": "velocity_x_km_s", "target": 0.0, "tolerance": 1.0}],
            }
        )


def test_evidence_writer_emits_canonical_json(tmp_path: Path) -> None:
    problem = problem_from_dict({})
    evidence = solve_trajectory_target(problem)

    path = write_trajectory_targeting_evidence(tmp_path / "targeting.json", evidence)
    loaded = json.loads(path.read_text(encoding="utf-8"))

    assert loaded["schema_version"] == TRAJECTORY_TARGETING_EVIDENCE_SCHEMA
    assert loaded["converged"] is True
    assert loaded["authoritative_repropagation"]["status"] == "verified"


def test_cli_writes_solved_evidence(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    problem = problem_from_dict({}).to_dict()
    source = tmp_path / "problem.json"
    output = tmp_path / "evidence.json"
    source.write_text(json.dumps(problem), encoding="utf-8")

    return_code = main(["solve", str(source), "--output", str(output)])
    stdout = json.loads(capsys.readouterr().out)
    retained = json.loads(output.read_text(encoding="utf-8"))

    assert return_code == 0
    assert stdout["status"] == "converged"
    assert retained == stdout
