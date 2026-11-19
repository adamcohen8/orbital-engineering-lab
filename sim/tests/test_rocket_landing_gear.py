from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from sim.config import scenario_config_from_dict
from sim.review import ReviewWorkspace
from sim.rocket import GuidanceCommand, LandingGearCommand
from sim.single_run import _SingleRunEngine
from sim.tests.test_rocket_touchdown import _scenario


def gear_scenario(output: Path, *, locked: bool = True, stroke: float = 0.7, surface_altitude: float = 22.0):
    scenario = _scenario(output, {"max_normal_energy_j": 1e9, "surface_altitude_m": surface_altitude})
    scenario["scenario_name"] = "rocket_landing_gear"
    scenario["simulator"]["duration_s"] = 7.0
    scenario["simulator"]["dt_s"] = 0.1
    article = scenario["objects"]["article"]
    article["specs"] = {"landing_gear": {
        "body_contact_points_m": [[-0.5, y, z] for y in (-1.0, 1.0) for z in (-1.0, 1.0)],
        "legs": [
            {
                "id": f"leg-{index}",
                "mount_body_m": [0.0, 0.0, 0.0],
                "stowed_offset_m": [-0.1, y, z],
                "deployed_offset_m": [-2.5, y, z],
                "travel_time_s": 0.4,
                "stiffness_n_m": 8e6,
                "damping_n_s_m": 1e6,
                "max_stroke_m": stroke,
                "friction_coefficient": 0.35,
                "initially_deployed": locked,
                "retractable": True,
            }
            for index, (y, z) in enumerate(((-2, -2), (-2, 2), (2, -2), (2, 2)))
        ],
    }}
    article["specs"]["mass_properties"] = {
        "inertia_kg_m2": [[8e6, 0, 0], [0, 8e6, 0], [0, 0, 8e6]],
        "center_of_mass_body_m": [0, 0, 0],
        "inertia_reference_point": "center_of_mass",
    }
    return scenario


def test_deploy_and_stow_commands_report_achieved_state(tmp_path):
    scenario = gear_scenario(tmp_path / "commands", locked=False, surface_altitude=-50.0)
    engine = _SingleRunEngine(scenario_config_from_dict(scenario))
    model = engine.rocket.rocket_sim
    start = model.initial_state()
    deploy = GuidanceCommand(0.0, landing_gear_commands=(LandingGearCommand("leg-0", "deploy"),))
    mid = model.step(start, deploy, 0.15)
    assert start.landing_legs["leg-0"].status == "stowed"
    assert mid.landing_legs["leg-0"].status == "deploying"
    locked = model.step(mid, deploy, 0.3)
    assert locked.landing_legs["leg-0"].status == "down_locked"
    assert locked.landing_legs["leg-0"].last_lock_time_s == pytest.approx(0.4)
    assert locked.landing_legs["leg-1"].status == "stowed"
    stow = GuidanceCommand(0.0, landing_gear_commands=(LandingGearCommand("leg-0", "stow"),))
    assert model.step(locked, stow, 0.45).landing_legs["leg-0"].status == "stowed"


def test_mission_execution_can_deploy_using_gear_feedback(tmp_path):
    scenario = gear_scenario(tmp_path / "mission", locked=False, surface_altitude=-50.0)
    engine = _SingleRunEngine(scenario_config_from_dict(scenario))

    class GearExecution:
        def update(self, *, rocket_state, **kwargs):
            leg = rocket_state.landing_legs["leg-0"]
            if leg.status == "stowed":
                return {"landing_gear_commands": (LandingGearCommand("leg-0", "deploy"),)}
            return {}

    engine.rocket.mission_execution = GearExecution()
    for _ in range(5):
        engine.step()
    assert engine.rocket.rocket_state.landing_legs["leg-0"].status == "down_locked"


def test_center_foot_does_not_block_settling_inside_support_polygon(tmp_path):
    scenario = gear_scenario(tmp_path / "center_foot")
    legs = scenario["objects"]["article"]["specs"]["landing_gear"]["legs"]
    legs.append({**legs[0], "id": "center", "deployed_offset_m": [-2.5, 0.0, 0.0]})
    simulator = _SingleRunEngine(scenario_config_from_dict(scenario)).rocket.rocket_sim
    model = simulator.gear_model
    state = simulator.initial_state()
    points, _, eci_to_ecef = model.sample(state)
    loaded = [
        replace(point, normal_load_n=1.0, velocity_ecef_m_s=np.zeros(3))
        for point in points if point.leg is not None
    ]
    center = eci_to_ecef @ state.position_eci_km
    loaded[-1] = replace(loaded[-1], position_ecef_km=center)

    assert model.supported(state, loaded[:-1])
    assert model.supported(state, loaded)
    assert not model.supported(state, loaded[:2] + loaded[-1:])


@pytest.mark.parametrize(("mode", "outcome", "reason"), [
    ("nominal", "landing", "settled"),
    ("late", "crash", "gear_not_locked"),
    ("stowed", "crash", "body_strike"),
    ("short_stroke", "crash", "leg_stroke_exceeded"),
])
def test_gear_contact_and_vehicle_outcome(tmp_path, mode, outcome, reason):
    output = tmp_path / mode
    scenario = gear_scenario(output, locked=mode != "stowed" and mode != "late",
                             stroke=0.01 if mode == "short_stroke" else 0.7)
    if mode == "late":
        gear = scenario["objects"]["article"]["specs"]["landing_gear"]
        gear["commands"] = [{"time_s": 0.25, "leg_id": leg["id"], "action": "deploy"} for leg in gear["legs"]]
    engine = _SingleRunEngine(scenario_config_from_dict(scenario))
    summary = engine.run()["summary"]
    evidence = summary["rocket_landing_gear_by_object"]["article"]
    first = evidence["touchdown"]
    terminal = evidence["terminal_event"]
    assert summary["termination_reason"] == f"rocket_{outcome}"
    assert terminal["reason"] == reason
    assert terminal["time_s"] == summary["termination_time_s"] == summary["duration_s"]
    assert first["time_s"] <= terminal["time_s"]
    assert first["contact_point_id"] != "body:0" or mode == "stowed"
    if mode == "nominal":
        assert first["outcome"] == "contact"
        assert terminal["time_s"] - first["time_s"] > 0.5
        assert all(leg["peak_normal_load_n"] > 0 for leg in evidence["legs"].values())
        assert all(0 < leg["peak_compression_m"] < 0.7 for leg in evidence["legs"].values())
        assert first["contact_point_id"].startswith("leg-")
        assert first["contact_point_status"] == "down_locked"
    if mode == "late":
        assert first["contact_point_status"] == "deploying"
        assert first["contact_normal_closing_speed_m_s"] > first["normal_closing_speed_m_s"] + 1.0
    review = ReviewWorkspace.open(output)
    rows = review.query(
        "SELECT event_type, time_s FROM events "
        "WHERE source='rocket_dynamics' ORDER BY time_s, event_type DESC"
    ).rows
    assert {"rocket_touchdown", f"rocket_{outcome}"} <= {row["event_type"] for row in rows}
    assert engine.snapshot()["time_s"] == terminal["time_s"]


def test_off_center_contact_applies_torque_and_friction(tmp_path):
    scenario = gear_scenario(tmp_path / "wrench")
    def simulate(friction):
        for leg in scenario["objects"]["article"]["specs"]["landing_gear"]["legs"]:
            leg["friction_coefficient"] = friction
        model = _SingleRunEngine(scenario_config_from_dict(scenario)).rocket.rocket_sim
        model.sim_cfg = replace(model.sim_cfg, max_time_s=0.52, dt_s=0.05)
        start = model.initial_state()
        from sim.dynamics.orbit.frames import transform_state

        pos, vel = transform_state(start.position_eci_km, start.velocity_eci_km_s,
                                   "eci", "ecef", t_s=0, context=model.gear_model.context)
        east = np.cross([0, 0, 1], pos)
        east /= np.linalg.norm(east)
        _, start.velocity_eci_km_s = transform_state(pos, vel + 0.0004 * east, "ecef", "eci",
                                                      t_s=0, context=model.gear_model.context)
        result = model.run(start)
        _, final = transform_state(result.position_eci_km[-1], result.velocity_eci_km_s[-1],
                                   "eci", "ecef", t_s=result.time_s[-1], context=model.gear_model.context)
        return result, float(final @ east)

    result, friction_speed = simulate(0.35)
    _, free_speed = simulate(0.0)
    assert result.touchdown["outcome"] == "contact"
    assert np.max(np.linalg.norm(result.angular_rate_body_rad_s, axis=1)) > 1e-3
    assert abs(friction_speed) < 0.9 * abs(free_speed)
