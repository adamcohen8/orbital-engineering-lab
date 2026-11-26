"""Primary-aware attitude, native orbit forcing and synchronized review evidence."""

import json
import math
import sqlite3

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from sim.api import SimulationSession
from sim.config.scenario_yaml import scenario_config_from_dict
from sim.dynamics.planetary.definitions import BODIES
from sim.dynamics.planetary.preparation import prepare_environment
from sim.tests.test_planetary_onp import scenario


def rotating(body="saturn", torques=None, duration=10.0):
    radius = 2.5 * BODIES[body].reference_radius_km
    raw = scenario(
        body,
        duration,
        state={
            "position_inertial_km": [radius, 0, 0],
            "velocity_inertial_km_s": [0, math.sqrt(BODIES[body].mu_km3_s2 / radius), 0],
            "angular_rate_body_rad_s": [0, 0, 0.02],
        },
    )
    raw["objects"]["probe"]["specs"]["mass_properties"] = {
        "inertia_kg_m2": np.diag([2.0, 3.0, 4.0]).tolist(),
        "inertia_reference_point": "center_of_mass",
    }
    raw["simulator"]["dynamics"]["attitude"] = {"enabled": True, "disturbance_torques": torques or {}}
    return raw


def run(raw, target):
    raw["outputs"]["output_dir"] = str(target)
    return SimulationSession(scenario_config_from_dict(raw)).run()


@pytest.mark.parametrize("body", sorted(BODIES))
def test_free_spin_and_review_each_primary(body, tmp_path):
    raw = rotating(body)
    result = run(raw, tmp_path / body)
    with sqlite3.connect(result.summary["review_sqlite_path"]) as db:
        rows = db.execute("""SELECT a.time_s, q0, q3, wz_body_rad_s, s.origin_center_id
            FROM planetary_attitude a JOIN planetary_state s USING(object_id,sample_index,time_s)
            ORDER BY a.sample_index""").fetchall()
    assert len(rows) == 3
    t, q0, q3, w, origin = rows[-1]
    np.testing.assert_allclose([q0, q3, w], [math.cos(0.01 * t), math.sin(0.01 * t), 0.02], atol=2e-8, rtol=0)
    assert origin == BODIES[body].center_id


@pytest.mark.parametrize("scale", [1.0, 1.0e25])
def test_saturn_libration_agrees_with_independent_planar_equation(scale, tmp_path):
    raw = rotating(torques={"gravity_gradient": True})
    obj = raw["objects"]["probe"]
    radius = obj["initial_state"]["position_inertial_km"][0]
    n = math.sqrt(BODIES["saturn"].mu_km3_s2 / radius**3)
    inertia = np.diag([2.0, 3.0, 4.0]) * scale
    obj["specs"]["mass_properties"]["inertia_kg_m2"] = inertia.tolist()
    phi0 = 0.02
    obj["initial_state"].update(
        angular_rate_body_rad_s=[0, 0, n], attitude_quat_bn=[math.cos(phi0 / 2), 0, 0, math.sin(phi0 / 2)]
    )
    frequency = math.sqrt(3 / 4) * n
    quarter = math.pi / (2 * frequency)
    dt = quarter / 30
    raw["simulator"].update(dt_s=dt, duration_s=30 * dt)

    def reference(t, y):
        return [y[1], -0.5 * frequency**2 * math.sin(2 * y[0])]

    expected = solve_ivp(reference, (0, quarter), [phi0, 0], method="DOP853", rtol=1e-12, atol=1e-14).y[:, -1]
    errors = []
    for m in (1, 2):
        raw["simulator"]["dynamics"]["attitude"]["attitude_substep_s"] = dt / m
        result = run(raw, tmp_path / str(m))
        row = result.state_history("probe")[-1]
        phi = 2 * math.atan2(row[9], row[6]) - n * quarter
        err = np.linalg.norm([phi - expected[0], (row[12] - n - expected[1]) / n])
        errors.append(err)
        assert err < 1.0e-6
    assert errors[1] < errors[0] / 8


def test_pressure_moments_use_selected_illumination_and_atmosphere(tmp_path):
    from sim.dynamics.planetary.frames import prepare_rotation

    frame = prepare_rotation("mars", {"model": "iau2000"}, 2457065.5)
    ephemeris = tmp_path / "mars-sun.json"
    ephemeris.write_text(
        json.dumps(
            {
                "observer_center_id": 499,
                "inertial_axes": "J2000",
                "aberration": "NONE",
                "position_units": "km",
                "velocity_units": "km/s",
                "time_scale": "TDB",
                "source": "Synthetic Mars-relative solar illumination fixture",
                "tdb_seconds_past_j2000": [frame.elapsed_tdb_s(-1), frame.elapsed_tdb_s(20)],
                "states_by_target_center": {"10": [[2e8, 1e7, 0, 0, 0, 0]] * 2},
            }
        )
    )
    raw = rotating("mars", {"drag": True, "srp": True})
    raw["simulator"]["planetary"] = {
        "atmosphere": {
            "model": "exponential",
            "reference_density_kg_m3": 1e-9,
            "reference_altitude_km": 5000.0,
            "scale_height_km": 50.0,
            "altitude_domain_km": [0.0, 10000.0],
            "altitude_datum": {"kind": "radial_sphere", "radius_km": 3396.19},
        },
        "drag": {"enabled": True},
        "srp": {"enabled": True},
        "ephemeris": {"data_path": str(ephemeris)},
    }
    obj = raw["objects"]["probe"]
    obj["specs"].update(drag_cp_offset_body_m=[1.0, 0.0, 0.0], srp_cp_offset_body_m=[0.0, 0.0, 1.0])
    result = run(raw, tmp_path / "pressure")
    with sqlite3.connect(result.summary["review_sqlite_path"]) as db:
        fj, tj = db.execute("""SELECT forces_json, disturbance_torques_body_nm_json FROM planetary_state
                       JOIN planetary_attitude USING(object_id,sample_index) WHERE sample_index=0""").fetchone()
    forces, torques = json.loads(fj), json.loads(tj)
    for name in ("drag", "srp"):
        expected = np.cross(
            obj["specs"][name + "_cp_offset_body_m"], 1000 * obj["specs"]["mass_kg"] * np.array(forces[name])
        )
        assert np.linalg.norm(expected) > 0
        np.testing.assert_allclose(torques[name], expected, rtol=1e-13)


def test_rotating_primary_field_and_validation(tmp_path):
    raw = rotating("saturn", {"magnetic": True})
    obj = raw["objects"]["probe"]
    obj["specs"]["magnetic_dipole_body_a_m2"] = [1.0, 0.0, 0.0]
    with pytest.raises(ValueError, match="body-specific"):
        scenario_config_from_dict(raw)
    field = {
        "model": "centered_dipole",
        "body_id": "saturn",
        "frame_id": "IAU2000_SATURN",
        "dipole_t_m3": [2e18, 0, 1e18],
    }
    raw["simulator"]["planetary"]["magnetic_field"] = field
    cfg = scenario_config_from_dict(raw)
    env = prepare_environment(cfg)
    radius = obj["initial_state"]["position_inertial_km"][0] * 1000
    pole = env.gravity_frame.evaluate(0)[0].T @ field["dipole_t_m3"]
    expected_field = np.array([2 * pole[0], -pole[1], -pole[2]]) / radius**3
    result = run(raw, tmp_path / "magnetic")
    with sqlite3.connect(result.summary["review_sqlite_path"]) as db:
        row = db.execute(
            "SELECT disturbance_torques_body_nm_json FROM planetary_attitude WHERE sample_index=0"
        ).fetchone()
    np.testing.assert_allclose(json.loads(row[0])["magnetic"], np.cross([1, 0, 0], expected_field), rtol=1e-13)
    field["body_id"] = "earth"
    with pytest.raises(ValueError, match="selected body_id"):
        scenario_config_from_dict(raw)
    field.update(body_id="saturn", frame_id="IAU2000_MARS")
    with pytest.raises(ValueError, match="selected gravity frame"):
        scenario_config_from_dict(raw)


@pytest.mark.parametrize("gravity", ["point_mass", "gmm2b_static"])
def test_gravity_gradient_uses_effective_mars_gm(gravity, tmp_path):
    raw = rotating("mars", {"gravity_gradient": True})
    raw["simulator"]["planetary"]["gravity"] = {"model": gravity}
    angle = math.pi / 4
    raw["objects"]["probe"]["initial_state"]["attitude_quat_bn"] = [math.cos(angle / 2), 0, 0, math.sin(angle / 2)]
    cfg = scenario_config_from_dict(raw)
    gm = prepare_environment(cfg).gravity.mu_km3_s2
    radius = raw["objects"]["probe"]["initial_state"]["position_inertial_km"][0]
    expected = -1.5 * gm / radius**3  # I_y - I_x = 1 kg m²
    result = run(raw, tmp_path / gravity)
    with sqlite3.connect(result.summary["review_sqlite_path"]) as db:
        entry = db.execute(
            "SELECT disturbance_torques_body_nm_json FROM planetary_attitude WHERE sample_index=0"
        ).fetchone()
    np.testing.assert_allclose(json.loads(entry[0])["gravity_gradient"], [0, 0, expected], atol=0, rtol=2e-14)


@pytest.mark.parametrize("torque", ["drag", "srp"])
def test_pressure_torque_needs_selected_force_and_lever_arm(torque):
    raw = rotating("sun" if torque == "srp" else "mars", {torque: True})
    with pytest.raises(ValueError, match=rf"planetary.{torque}.enabled"):
        scenario_config_from_dict(raw)
    raw["simulator"]["planetary"][torque] = {"enabled": True}
    if torque == "drag":
        raw["simulator"]["planetary"]["atmosphere"] = {"model": "mars_reference_table"}
    with pytest.raises(ValueError, match=f"{torque}_cp_offset_body_m"):
        scenario_config_from_dict(raw)
