from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pytest
import yaml

from sim.api import SimulationConfig, SimulationSession
from sim.flight_software import DockingMechanismCommand, DockingPortMeasurement
from sim.flight_software.schemas import from_primitive, to_primitive


def run_example(output, configure=None):
    root = Path(__file__).resolve().parents[2]
    raw = yaml.safe_load((root / "examples/configs/public_docking_flight_software.yaml").read_text())
    raw["outputs"]["output_dir"] = str(output)
    if configure is not None:
        configure(raw)
    result = SimulationSession.from_config(SimulationConfig.from_dict(raw)).run()
    evidence = json.loads((output / "docking.json").read_text())
    return result, evidence


def test_two_stacks_dock_through_typed_sensors_and_hardware(tmp_path):
    result, dock = run_example(tmp_path)
    assert dock["summary"]["phase"] == "mated"
    assert [row["event"] for row in dock["events"]] == ["first_contact", "soft_capture", "hard_mate"]
    first, capture, mate = dock["events"]
    assert capture["capture_armed"] and mate["hard_mate_requested"]
    assert dock["samples"][0]["lateral_m"] > .05
    assert first["lateral_m"] < .002
    assert dock["samples"][0]["angle_rad"] > .09
    assert first["angle_rad"] < .001
    assert mate["time_s"] < 19.
    joined = [row["offset_receiver_m"] for row in dock["samples"] if row["phase"] == "mated"]
    np.testing.assert_allclose(joined, np.broadcast_to(joined[0], (len(joined), 3)), atol=1e-8, rtol=0)
    assert result.summary["thrust_stats"]["chaser"]["total_dv_m_s"] > 0.
    with sqlite3.connect(tmp_path / "review/run.sqlite") as conn:
        assert conn.execute("""
            SELECT DISTINCT object_id FROM fsw_input_events
            WHERE json_extract(detail_json, '$.payload.measurement_type') = 'docking_port.v1'
            ORDER BY object_id
        """).fetchall() == [("chaser",), ("target",)]
        assert conn.execute("""
            SELECT DISTINCT object_id FROM actuator_realization
            WHERE actuator_id = 'wrench' AND (
                abs(realized_torque_x_n_m) + abs(realized_torque_y_n_m) + abs(realized_torque_z_n_m) > 0
            ) ORDER BY object_id
        """).fetchall() == [("chaser",), ("target",)]
        assert conn.execute("""
            SELECT DISTINCT object_id FROM actuator_device_state
            WHERE actuator_id = 'docking' AND field_name = 'request_hard_mate' AND value_integer = 1
            ORDER BY object_id
        """).fetchall() == [("chaser",), ("target",)]
        assert conn.execute("""
            SELECT DISTINCT object_id FROM fsw_diagnostic_fields
            WHERE topic = 'oel.example.docking.v1' AND field_name = 'mode' AND value_text = 'docked'
            ORDER BY object_id
        """).fetchall() == [("chaser",), ("target",)]
        assert conn.execute("""
            SELECT DISTINCT lower(disposition) FROM actuator_command_receipts
        """).fetchall() in [[("accepted",)], [("commanddisposition.accepted",)]]


@pytest.mark.parametrize("object_id", ["chaser", "target"])
def test_hard_mate_requires_both_stack_requests(tmp_path, object_id):
    def withhold_request(raw):
        raw["objects"][object_id]["flight_software"]["params"]["request_hard_mate"] = False

    _, dock = run_example(tmp_path, withhold_request)
    assert dock["summary"]["phase"] == "captured"
    assert [row["event"] for row in dock["events"]] == ["first_contact", "soft_capture"]


def test_receiver_without_arm_prevents_closing(tmp_path):
    def withhold_arm(raw):
        raw["objects"]["target"]["flight_software"]["params"]["arm_capture"] = False
        raw["simulator"]["duration_s"] = 6.

    _, dock = run_example(tmp_path, withhold_arm)
    assert not dock["events"]
    assert dock["samples"][-1]["gap_m"] > .3


def test_docking_packets_round_trip():
    port = DockingPortMeasurement(
        (.1, .2, .3), (0., 0., -.04), (0., 0., 0.), (0., 0., 0.),
        ((1., 0., 0.), (0., 1., 0.), (0., 0., 1.)), "approach", True,
    )
    command = DockingMechanismCommand(request_hard_mate=True)
    for packet in (port, command):
        assert from_primitive(type(packet), to_primitive(packet)) == packet
