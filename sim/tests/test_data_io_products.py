from __future__ import annotations

import json
from pathlib import Path

import pytest

from sim.ccsds import compare_cdm
from sim.conjunction import assess_conjunction
from sim.data_io.api import encode_data
from sim.data_io.common import json_bytes
from sim.data_io.contracts import DataIOError
from sim.data_io.navigation import parse_navigation
from sim.data_io.products import cdm_from_assessment

ROOT = Path(__file__).resolve().parents[2]


def test_completed_assessment_generates_cdm_without_another_propagation(tmp_path, monkeypatch):
    problem_path = ROOT / "examples/conjunction/synthetic_crossing.json"
    problem = json.loads(problem_path.read_text())
    evidence = assess_conjunction(problem, numeric_backend="python", probability_numeric_backend="python")
    path = tmp_path / "assessment.json"
    path.write_bytes(json_bytes(evidence))

    def forbidden(*args, **kwargs):
        raise AssertionError("An IO post-processor must not rerun propagation.")

    monkeypatch.setattr("sim.analysis.conjunction_workflow.propagate_history", forbidden)
    message = cdm_from_assessment(path, problem_path, initial_epoch="2024-01-01T00:00:00")
    for format in ("cdm-kvn", "cdm-xml"):
        raw, _ = encode_data(message, format=format)
        recovered = parse_navigation(raw.decode(), format=format)
        assert compare_cdm(message, recovered)["equivalent"]
    assert message.relative.miss_distance_m == pytest.approx(
        evidence["baseline"]["closest_approach"]["miss_distance_km"] * 1000
    )
    assert message.relative.collision_probability == evidence["baseline"]["probability"]["collision_probability"]
    evidence["problem_sha256"] = "0" * 64
    path.write_bytes(json_bytes(evidence))
    with pytest.raises(DataIOError, match="bound"):
        cdm_from_assessment(path, problem_path, initial_epoch="2024-01-01T00:00:00")


def test_opm_maneuvers_normalize_to_a_plan_and_roundtrip_without_execution(tmp_path):
    from sim.data_io.api import convert, read_data
    from sim.data_io.products import maneuver_plan_from_opm
    from sim.data_io.tabular import trajectory_document

    opm = next((ROOT / "sim/interchange/examples").glob("*.opm"))
    message = read_data(opm, format="opm-kvn")
    plan = maneuver_plan_from_opm(message)
    assert plan["execution_authorized"] is False
    for native, normalized in zip(message.maneuvers, plan["maneuvers"]):
        assert normalized["delta_v_m_s"][0] == float(native["MAN_DV_1"]) * 1000
    source_state = read_data(opm, format="opm-kvn")
    from sim.data_io.api import as_trajectory

    state = as_trajectory(source_state)
    path = tmp_path / "state.json"
    path.write_bytes(json_bytes(trajectory_document(state)))
    plan_path = tmp_path / "plan.json"
    plan_path.write_bytes(json_bytes(plan))
    target = tmp_path / "planned.opm"
    result = convert(
        plan_path,
        target,
        input_format="maneuver-plan-json",
        output_format="opm-kvn",
        options={"reference_trajectory": str(path), "reference_format": "trajectory-json", "sample_index": 0},
    )
    recovered = read_data(target, format="opm-kvn")
    assert len(recovered.maneuvers) == len(message.maneuvers)
    assert result["simulation_executed"] is False


def test_historical_review_impulses_become_a_new_opm(tmp_path):
    import sqlite3

    from sim.data_io.products import opm_from_review_maneuvers
    from sim.tests.data_io_helpers import make_review

    run = make_review(tmp_path / "run")
    with sqlite3.connect(run / "review/run.sqlite") as conn:
        conn.execute(
            "INSERT INTO impulsive_maneuvers (maneuver_id, object_id, sample_index, time_s, frame, dv_x_eci_m_s, dv_y_eci_m_s, dv_z_eci_m_s, pre_pos_x_eci_km,pre_pos_y_eci_km,pre_pos_z_eci_km,pre_vel_x_eci_km_s,pre_vel_y_eci_km_s,pre_vel_z_eci_km_s,post_pos_x_eci_km,post_pos_y_eci_km,post_pos_z_eci_km,post_vel_x_eci_km_s,post_vel_y_eci_km_s,post_vel_z_eci_km_s) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                "impulse-1",
                "SAT",
                1,
                30.0,
                "eci",
                1.0,
                2.0,
                3.0,
                7000.0,
                0.0,
                0.0,
                0.0,
                7.5,
                0.0,
                7000.0,
                0.0,
                0.0,
                0.001,
                7.502,
                0.003,
            ),
        )
    message, receipt = opm_from_review_maneuvers(run, object_id="SAT")
    assert float(message.maneuvers[0]["MAN_DV_1"]) == 0.001
    assert receipt["maneuver_ids"] == ["impulse-1"]
