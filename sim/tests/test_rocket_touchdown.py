import json
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from sim.config import scenario_config_from_dict
from sim.dynamics.orbit.frames import frame_context_from_environment, transform_state
from sim.review import ReviewWorkspace
from sim.rocket import HoldAttitudeGuidance, RocketTouchdownLimits
from sim.single_run import _SingleRunEngine
from sim.utils.geodesy import ecef_to_geodetic_deg_km


def _scenario(output_dir, limits):
    return {
        "scenario_name": "touchdown",
        "objects": {
            "article": {
                "kind": "rocket",
                "enabled": True,
                "initial_state": {
                    "launch_lat_deg": 35.0,
                    "launch_lon_deg": -116.0,
                    "launch_alt_km": 0.025,
                },
                "base_guidance": {
                    "module": "sim.rocket.guidance",
                    "class_name": "HoldAttitudeGuidance",
                    "params": {"throttle": 0.0},
                },
            },
        },
        "simulator": {
            "initial_jd_utc": 2460000.5,
            "duration_s": 2.0,
            "dt_s": 1.0,
            "dynamics": {
                "orbit": {"drag": False, "j2": False, "numeric_backend": "python"},
                "rocket": {"numeric_backend": "python", "aero_model_enabled": False},
            },
            "termination": {"by_object": {"article": {"touchdown": limits}}},
        },
        "outputs": {
            "output_dir": str(output_dir),
            "plots": {"enabled": False},
            "stats": {"save_json": True, "save_full_log": False, "print_summary": False},
            "review": {"enabled": True, "detail": "standard"},
        },
    }


def test_scenario_lands_at_elevated_contact_height_and_writes_review(tmp_path):
    output = tmp_path / "landing"
    engine = _SingleRunEngine(scenario_config_from_dict(_scenario(output, {
        "surface_altitude_m": 20.0,
        "contact_height_m": 2.0,
        "max_normal_energy_j": 1.0e12,
        "max_lateral_speed_m_s": 0.05,
        "max_tilt_deg": 1.1,
        "max_angular_rate_deg_s": 1.0,
    })))
    payload = engine.run()
    summary = payload["summary"]
    event = summary["rocket_touchdown"]
    assert summary["termination_reason"] == "rocket_landing"
    assert summary["termination_object_id"] == "article"
    assert 0.7 < event["time_s"] < 0.9
    assert event["time_s"] == summary["termination_time_s"] == summary["duration_s"]
    snapshot = engine.snapshot()
    assert snapshot["time_s"] == event["time_s"]
    np.testing.assert_array_equal(snapshot["truth"]["article"][:3], event["position_eci_km"])
    assert event["bracket_end_s"] - event["bracket_start_s"] <= 1.0e-7
    assert all(check["passed"] for check in event["checks"].values())
    assert event["normal_closing_speed_m_s"] == pytest.approx(7.7, abs=0.15)
    context = frame_context_from_environment(engine.rocket.rocket_sim.sim_cfg.atmosphere_env)
    position_ecef, _ = transform_state(
        event["position_eci_km"], event["velocity_eci_km_s"], "eci", "ecef",
        context=context, t_s=event["time_s"],
    )
    assert ecef_to_geodetic_deg_km(position_ecef)[2] == pytest.approx(0.022, abs=1.0e-8)

    saved = json.loads((output / "master_run_summary.json").read_text())
    assert saved["rocket_touchdown"] == event
    workspace = ReviewWorkspace.open(output)
    row, = workspace.query(
        "SELECT time_s, sample_index, object_id, severity FROM events WHERE event_type = 'rocket_touchdown'"
    ).rows
    assert row == {"time_s": event["time_s"], "sample_index": 1, "object_id": "article", "severity": "info"}
    state, = workspace.query(
        "SELECT time_s, pos_x_eci_km, vel_x_eci_km_s, mass_kg FROM object_state WHERE sample_index = 1"
    ).rows
    assert state == {
        "time_s": event["time_s"],
        "pos_x_eci_km": event["position_eci_km"][0],
        "vel_x_eci_km_s": event["velocity_eci_km_s"][0],
        "mass_kg": event["mass_kg"],
    }


@pytest.mark.parametrize("backend", ["python", "rust"])
def test_airborne_run_registers_each_exceeded_limit_and_uses_contact_mass(tmp_path, backend):
    if backend == "rust":
        pytest.importorskip("oel_rust_orbit")
    scenario = _scenario(tmp_path / "crash", {
        "max_normal_energy_j": 100.0,
        "max_lateral_speed_m_s": 0.5,
        "max_tilt_deg": 0.1,
        "max_angular_rate_deg_s": 0.5,
    })
    scenario["simulator"]["dynamics"]["rocket"]["numeric_backend"] = backend
    engine = _SingleRunEngine(scenario_config_from_dict(scenario))
    simulator = engine.rocket.rocket_sim
    simulator.guidance = HoldAttitudeGuidance(throttle=0.1)
    start = simulator.initial_state()
    up = start.position_eci_km / np.linalg.norm(start.position_eci_km)
    east = np.cross([0.0, 0.0, 1.0], up)
    east /= np.linalg.norm(east)
    start.velocity_eci_km_s += 0.002 * east - 0.005 * up
    start.angular_rate_body_rad_s[:] = [0.0, 0.0, np.deg2rad(2.0)]
    result = simulator.run(state0=start)
    event = result.touchdown
    assert result.termination_reason == "rocket_crash"
    assert event is not None
    assert result.time_s[-2] < event["time_s"] == result.time_s[-1]
    assert event["mass_kg"] == result.mass_kg[-1] < start.mass_kg
    command = simulator.guidance.command(start, simulator.sim_cfg, simulator.vehicle_cfg)
    assert simulator.step(start, command, dt_s=2.0).mass_kg < event["mass_kg"]
    assert event["normal_energy_j"] == pytest.approx(
        0.5 * result.mass_kg[-1] * event["normal_closing_speed_m_s"]**2
    )
    assert set(event["failed_limits"]) == {
        "normal_energy_j", "lateral_speed_m_s", "tilt_deg", "angular_rate_deg_s",
    }

    # Equality is accepted for the required energy limit.
    simulator.sim_cfg = replace(
        simulator.sim_cfg, touchdown=RocketTouchdownLimits(event["normal_energy_j"]),
    )
    assert simulator.run(state0=start).termination_reason == "rocket_landing"


@pytest.mark.parametrize("limits, message", [
    ({}, "max_normal_energy_j is required"),
    ({"max_normal_energy_j": -1.0}, "must be non-negative"),
    ({"max_normal_energy_j": float("nan")}, "must be a finite number"),
    ({"max_normal_energy_j": 0.0, "max_tilt_deg": 181.0}, "must be at most 180"),
    ({"max_normal_energy_j": 0.0, "max_sideways": 1.0}, "max_sideways"),
])
def test_touchdown_input_validation(tmp_path, limits, message):
    with pytest.raises(ValueError, match=message):
        scenario_config_from_dict(_scenario(tmp_path, limits))


def test_touchdown_picks_first_contact_across_rockets(tmp_path):
    scenario = _scenario(tmp_path / "pair", {
        "surface_altitude_m": 21.0,
        "max_normal_energy_j": 1.0e20,
    })
    earlier = deepcopy(scenario["objects"]["article"])
    earlier["initial_state"]["launch_alt_km"] = 0.023
    scenario["objects"]["second"] = earlier
    scenario["simulator"]["termination"]["by_object"]["second"] = {
        "touchdown": {"surface_altitude_m": 22.0, "max_normal_energy_j": 0.0},
    }
    engine = _SingleRunEngine(scenario_config_from_dict(scenario))
    summary = engine.run()["summary"]
    assert summary["termination_reason"] == "rocket_crash"
    assert summary["termination_object_id"] == "second"
    assert summary["rocket_touchdown"]["failed_limits"] == ["normal_energy_j"]
