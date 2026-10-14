"""Independent numerical and contract tests for native planetary ONP.

Reference forces use SciPy associated Legendre functions and finite differences,
not the production solid-harmonic recurrence. Physics tolerances are fixed here.
"""

import json
import math
import re
import sqlite3

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.special import lpmv

from sim.config.scenario_yaml import scenario_config_from_dict
from sim.dynamics.planetary.preparation import prepare_environment
from sim.dynamics.planetary.propagation import PlanetaryForcePlan, PlanetaryPropagator
from sim.dynamics.planetary.runtime import initial_truth


def scenario(body="mars", duration=120.0, planetary=None, state=None):
    return {
        "scenario_name": "planetary_test",
        "objects": {
            "probe": {
                "kind": "satellite",
                "specs": {
                    "mass_kg": 100.0,
                    "cd": 2.2,
                    "drag_area_m2": 1.0,
                    "cr": 1.3,
                    "srp_area_m2": 2.0,
                },
                "initial_state": state
                or {
                    "position_inertial_km": [
                        10000.0 if body not in {"mars", "moon"} else 4000.0,
                        100.0,
                        200.0,
                    ],
                    "velocity_inertial_km_s": [-0.1, 3.0, 0.2],
                },
            }
        },
        "simulator": {
            "central_body": body,
            "initial_jd_utc": 2457065.5,
            "duration_s": duration,
            "dt_s": 5.0,
            "planetary": planetary or {},
            "dynamics": {"orbit": {"integrator": "rkf78"}},
        },
        "outputs": {"stats": {"print_summary": False}},
    }


def prepared(raw):
    cfg = scenario_config_from_dict(raw)
    env = prepare_environment(cfg)
    plan = PlanetaryForcePlan(
        env,
        **{
            k: cfg.objects["probe"].specs[k]
            for k in ("mass_kg", "cd", "drag_area_m2", "cr", "srp_area_m2")
        },
    )
    return cfg, env, plan


def test_body_constants_match_original_naif_sources():
    from sim.dynamics.planetary.definitions import BODIES
    from sim.dynamics.planetary.resources import DATA_ROOT

    gm = (DATA_ROOT / "gm_de440.tpc").read_text()
    pck = (DATA_ROOT / "pck00008.tpc").read_text()
    for body in BODIES.values():
        value = re.search(rf"BODY{body.center_id}_GM\s*=\s*\(\s*([^\s)]+)", gm)
        assert float(value[1].replace("D", "E")) == body.mu_km3_s2
        if body.body_id != "sun":
            radii = re.search(rf"BODY{body.center_id}_RADII\s*=\s*\(\s*([^\s)]+)", pck)
            assert float(radii[1]) == body.reference_radius_km


def potential(gravity, r):
    distance = np.linalg.norm(r)
    latitude = r[2] / distance
    longitude = math.atan2(r[1], r[0])
    c = gravity.compiled
    value = 0.0
    for n in range(2, c.n_max + 1):
        for m in range(min(n, c.m_max) + 1):
            normalization = math.sqrt(
                (1.0 if m == 0 else 2.0)
                * (2 * n + 1)
                * math.factorial(n - m)
                / math.factorial(n + m)
            )
            p = (-1.0) ** m * lpmv(m, n, latitude) * normalization
            value += (
                (gravity.radius_km / distance) ** n
                * p
                * (
                    c.c_nm[n, m] * math.cos(m * longitude)
                    + c.s_nm[n, m] * math.sin(m * longitude)
                )
            )
    return gravity.mu_km3_s2 / distance * value


def reference_harmonic(gravity, r):
    # Five-point Cartesian potential gradient, in km/s2.
    h = 0.05
    result = []
    for axis in range(3):
        delta = np.eye(3)[axis] * h
        result.append(
            (
                -potential(gravity, r + 2 * delta)
                + 8 * potential(gravity, r + delta)
                - 8 * potential(gravity, r - delta)
                + potential(gravity, r - 2 * delta)
            )
            / (12 * h)
        )
    return np.array(result)


@pytest.mark.parametrize("degree", [2, 8, 20])
@pytest.mark.parametrize(
    "r",
    [
        [4000.0, 100.0, 200.0],
        [-4000.0, 1e-8, 500.0],
        [1.0, 2.0, 4000.0],
        [0.0, 0.0, 4000.0],
    ],
)
def test_gmm2b_independent_force(degree, r):
    _, env, plan = prepared(
        scenario(planetary={"gravity": {"model": "gmm2b_static", "max_degree": degree}})
    )
    rotation, _ = env.gravity_frame.evaluate(37.0)
    r = np.array(r)
    x = np.r_[rotation.T @ r, [0.0, 0.0, 0.0]]
    got = plan.stage(37.0, x)[1]["forces_km_s2"]["harmonics"]
    expected = rotation.T @ reference_harmonic(env.gravity, r)
    np.testing.assert_allclose(got, expected, atol=3e-11, rtol=0.0)


@pytest.mark.parametrize("body", ["mars", "moon", "venus", "sun"])
@pytest.mark.parametrize("method", ["rk4", "rkf78", "dopri5"])
def test_circular_kepler(body, method):
    cfg, env, plan = prepared(scenario(body, duration=600.0))
    radius = 1.5e8 if body == "sun" else 10000.0
    speed = math.sqrt(env.gravity.mu_km3_s2 / radius)
    x = np.array([radius, 0.0, 0.0, 0.0, speed, 0.0])
    propagator = PlanetaryPropagator(
        plan, {"integrator": method, "adaptive_atol": 1e-12, "adaptive_rtol": 1e-12}
    )
    for t in np.arange(0.0, 600.0, 5.0):
        x = propagator.propagate(x, 5.0, t)
    angle = speed / radius * 600.0
    expected = np.array(
        [
            radius * math.cos(angle),
            radius * math.sin(angle),
            0.0,
            -speed * math.sin(angle),
            speed * math.cos(angle),
            0.0,
        ]
    )
    np.testing.assert_allclose(x[:3], expected[:3], atol=2e-6, rtol=0.0)
    np.testing.assert_allclose(x[3:], expected[3:], atol=2e-9, rtol=0.0)


def test_atmosphere_original_samples_and_boundaries():
    _, env, _ = prepared(
        scenario(
            planetary={"atmosphere": {"model": "mars_reference_table"}},
            state={
                "position_inertial_km": [3546.19, 0.0, 0.0],
                "velocity_inertial_km_s": [0.0, 3.0, 0.0],
            },
        )
    )
    a = env.atmosphere
    for h, rho in zip(a.heights, a.densities):
        sample = a.sample([3396.19 + h, 0.0, 0.0])
        assert sample.density_kg_m3 == pytest.approx(rho, rel=3e-13)
    mid = (a.heights[30] + a.heights[31]) / 2
    sample = a.sample([3396.19 + mid, 0.0, 0.0])
    assert sample.density_kg_m3 == pytest.approx(
        math.sqrt(a.densities[30] * a.densities[31]), rel=3e-13
    )
    with pytest.raises(ValueError, match="outside"):
        a.sample([3396.19 + 170.0, 0.0, 0.0])
    from dataclasses import replace

    vacuum = replace(a, policy="vacuum_above").sample([3396.19 + 170.0, 0.0, 0.0])
    assert vacuum.density_kg_m3 == 0.0 and not vacuum.within_domain
    with pytest.raises(ValueError, match="outside"):
        replace(a, policy="vacuum_above").sample([3396.19 + 130.0, 0.0, 0.0])


def test_drag_units_rotation_and_wind():
    selections = {
        "atmosphere": {
            "model": "exponential",
            "altitude_datum": {"kind": "radial_sphere", "radius_km": 3396.19},
            "altitude_domain_km": [0.0, 2000.0],
            "reference_density_kg_m3": 1e-8,
            "reference_altitude_km": 600.0,
            "scale_height_km": 20.0,
            "wind_body_m_s": [30.0, -20.0, 10.0],
        },
        "drag": {"enabled": True},
    }
    _, env, plan = prepared(scenario(planetary=selections))
    rotation, derivative = env.atmosphere_frame.evaluate(0.0)
    rb = np.array([3996.19, 0.0, 0.0])
    vb = np.array([0.0, 3.0, 0.0])
    r = rotation.T @ rb
    v = rotation.T @ (vb - derivative @ r)
    total, diag = plan.stage(0.0, np.r_[r, v])
    relative = rotation.T @ (vb - np.array([30.0, -20.0, 10.0]) / 1000.0)
    expected = -0.5 * 1e-8 * 2.2 / 100.0 * np.linalg.norm(relative) * 1000.0 * relative
    got = np.array(diag["forces_km_s2"]["drag"])
    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-18)
    assert np.dot(got, relative) < 0.0
    zero_v = rotation.T @ (np.array([30.0, -20.0, 10.0]) / 1000.0 - derivative @ r)
    assert (
        np.linalg.norm(plan.stage(0.0, np.r_[r, zero_v])[1]["forces_km_s2"]["drag"])
        < 1e-28
    )


def test_rotation_derivative_and_state_roundtrip():
    for body in ("mars", "moon", "neptune", "venus", "sun"):
        _, env, _ = prepared(scenario(body))
        frame = env.atmosphere_frame
        r, d = frame.evaluate(50.0)
        plus = frame.evaluate(50.5)[0]
        minus = frame.evaluate(49.5)[0]
        np.testing.assert_allclose(d, plus - minus, atol=2e-11, rtol=0.0)
        x = np.array([4000.0, 100.0, 500.0])
        v = np.array([-0.2, 3.0, 1.0])
        np.testing.assert_allclose(
            r.T @ (r @ v + d @ x) - r.T @ d @ x, v, atol=2e-15, rtol=0.0
        )


def test_sun_srp_no_self_shadow():
    _, env, plan = prepared(scenario("sun", planetary={"srp": {"enabled": True}}))
    distance = 149597870.7
    force = plan.stage(0.0, [distance, 0.0, 0.0, 0.0, 30.0, 0.0])[1]["forces_km_s2"][
        "srp"
    ]
    assert force[0] == pytest.approx(
        1361.0 / 299792458.0 * 1.3 * 2.0 / 100.0 / 1000.0, rel=2e-15
    )
    assert force[1:] == [0.0, 0.0]
    with pytest.raises(ValueError, match="eclipse"):
        prepared(
            scenario(
                "sun",
                planetary={"srp": {"enabled": True, "shadow_model": "cylindrical"}},
            )
        )


def test_combined_reference_trajectory_and_repeatability():
    # Deliberately frozen, controlled broad-domain atmosphere isolates integration.
    selections = {
        "gravity": {"model": "gmm2b_static", "max_degree": 8},
        "atmosphere": {
            "model": "exponential",
            "altitude_datum": {"kind": "radial_sphere", "radius_km": 3396.19},
            "altitude_domain_km": [0.0, 2000.0],
            "reference_density_kg_m3": 1e-8,
            "reference_altitude_km": 600.0,
            "scale_height_km": 100.0,
        },
        "drag": {"enabled": True},
    }
    cfg, env, plan = prepared(scenario(duration=120.0, planetary=selections))
    truth = initial_truth(cfg.objects["probe"], env)
    x0 = np.r_[truth.position_eci_km, truth.velocity_eci_km_s]

    def derivative(t, x):
        r = x[:3]
        v = x[3:]
        radius = np.linalg.norm(r)
        R, _ = env.gravity_frame.evaluate(t)
        A, D = env.atmosphere_frame.evaluate(t)
        rel = A.T @ (A @ v + D @ r)
        rho = 1e-8 * math.exp(-(radius - 3396.19 - 600.0) / 100.0)
        a = (
            -env.gravity.mu_km3_s2 * r / radius**3
            + R.T @ reference_harmonic(env.gravity, R @ r)
            - 0.5 * rho * 2.2 / 100.0 * np.linalg.norm(rel) * 1000.0 * rel
        )
        return np.r_[v, a]

    ref = solve_ivp(
        derivative, (0.0, 120.0), x0, method="DOP853", rtol=2e-12, atol=2e-13
    ).y[:, -1]
    results = []
    for method, dt in (("rkf78", 120.0), ("rk4", 2.0), ("rk4", 1.0)):
        p = PlanetaryPropagator(
            plan, {"integrator": method, "adaptive_rtol": 1e-12, "adaptive_atol": 1e-12}
        )
        x = x0.copy()
        for t in np.arange(0.0, 120.0, dt):
            x = p.propagate(x, dt, t)
        np.testing.assert_allclose(x[:3], ref[:3], atol=3e-6, rtol=0.0)
        np.testing.assert_allclose(x[3:], ref[3:], atol=3e-9, rtol=0.0)
        results.append(x)
    assert np.linalg.norm(results[2] - ref) < np.linalg.norm(results[1] - ref)
    np.testing.assert_array_equal(
        plan.native.propagate(x0.tolist(), 0.0, 120.0, "rkf78", 1e-12, 1e-12)[0],
        results[0],
    )


def test_config_roundtrip_and_consumer_boundaries(tmp_path):
    from sim.api import SimulationSession

    cfg, env, _ = prepared(scenario(duration=10.0))
    assert scenario_config_from_dict(cfg.to_dict()).to_dict() == cfg.to_dict()
    session = SimulationSession(cfg)
    first = session.reset()
    assert first.state_metadata["origin_center_id"] == 499
    session.step()
    result = session.run()
    assert result.snapshot(0).state_metadata["origin_center_id"] == 499
    assert "x_inertial_km" in result.to_records(object_id="probe")[0]
    from pathlib import Path

    assert "mars-centered J2000" in Path(result.output_dir, "index.md").read_text()
    from sim.review import ReviewWorkspace

    with sqlite3.connect(result.summary["review_sqlite_path"]) as db:
        assert db.execute("select count(*) from object_state").fetchone()[0] == 0
        assert (
            db.execute(
                "select distinct origin_center_id from planetary_state"
            ).fetchone()[0]
            == 499
        )
    from sim.interchange.adapters.review_store import _require_review_contract

    with pytest.raises(ValueError, match="[Pp]lanetary"):
        _require_review_contract(
            ReviewWorkspace.open(result.summary["review_sqlite_path"])
        )


@pytest.mark.parametrize(
    "change,match",
    [
        ({"central_body": "bogus"}, "Unsupported"),
        ({"planetary": {"gravity": {"model": "gmm3_static"}}}, "GMM-3"),
        ({"planetary": {"drag": {"enabled": True}}}, "atmosphere"),
        ({"dynamics": {"orbit": {"numeric_backend": "numpy"}}}, "rust"),
        ({"dynamics": {"orbit": {"j2": True}}}, "unsupported"),
        ({"frames": {"model": "simple_gmst"}}, "Non-Earth"),
    ],
)
def test_unsupported_config_fails(change, match):
    raw = scenario()
    raw["simulator"].update(change)
    with pytest.raises(ValueError, match=match):
        scenario_config_from_dict(raw)


def test_resource_tamper_is_detected(tmp_path):
    from sim.dynamics.planetary.resources import DATA_ROOT

    path = tmp_path / "frame.json"
    path.write_bytes((DATA_ROOT / "mars_iau2000_rotation.json").read_bytes())
    _, env, _ = prepared(
        scenario(
            planetary={
                "atmosphere_frame": {"model": "resource", "data_path": str(path)}
            }
        )
    )
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="SHA256"):
        env.verify_resources()


def test_generic_lunar_gravity_and_ephemeris(tmp_path):
    # Synthetic coefficients explicitly test the data contract, not lunar fidelity.
    from sim.dynamics.planetary.frames import prepare_rotation

    coeff = tmp_path / "moon.sha"
    label = tmp_path / "moon.lbl"
    coeff.write_text(
        "1737.4,4902.8,0,2,2,1,0,0\n2,0,-0.0002,0,0,0\n2,1,0.00001,0.00002,0,0\n2,2,0.00003,-0.00001,0,0\n"
    )
    label.write_text('TARGET_NAME = "MOON"\n')
    frame = prepare_rotation("moon", {"model": "iau2000"}, 2457065.5)
    times = [frame.elapsed_tdb_s(0.0) - 10.0, frame.elapsed_tdb_s(120.0) + 10.0]
    table = tmp_path / "ephem.json"
    table.write_text(
        json.dumps(
            {
                "observer_center_id": 301,
                "inertial_axes": "J2000",
                "aberration": "NONE",
                "position_units": "km",
                "velocity_units": "km/s",
                "time_scale": "TDB",
                "source": "Synthetic static geometric fixture; not an astronomical ephemeris",
                "tdb_seconds_past_j2000": times,
                "states_by_target_center": {
                    "10": [[1.5e8, 2e7, 0.0, 0.0, 0.0, 0.0]] * 2
                },
            }
        )
    )
    raw = scenario(
        "moon",
        planetary={
            "gravity": {
                "model": "pds_sha_static",
                "coeff_path": str(coeff),
                "label_path": str(label),
                "frame_id": "IAU2000_MOON",
                "max_degree": 2,
                "max_order": 2,
            },
            "third_bodies": [{"target_center_id": 10, "mu_km3_s2": 132712440041.27942}],
            "srp": {"enabled": True},
            "ephemeris": {"model": "table", "data_path": str(table)},
        },
    )
    _, env, plan = prepared(raw)
    r = np.array([4000.0, 100.0, 200.0])
    x = np.r_[r, [0.0, 3.0, 0.0]]
    forces = plan.stage(30.0, x)[1]["forces_km_s2"]
    sun = np.array([1.5e8, 2e7, 0.0])
    delta = sun - r
    expected = 132712440041.27942 * (
        delta / np.linalg.norm(delta) ** 3 - sun / np.linalg.norm(sun) ** 3
    )
    np.testing.assert_allclose(
        forces["third_body_10"], expected, atol=2e-20, rtol=1e-12
    )
    R, _ = env.gravity_frame.evaluate(30.0)
    np.testing.assert_allclose(
        forces["harmonics"],
        R.T @ reference_harmonic(env.gravity, R @ r),
        atol=3e-11,
        rtol=0.0,
    )
    document = json.loads(table.read_text())
    document["states_by_target_center"]["010"] = document["states_by_target_center"][
        "10"
    ]
    table.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="canonical integers"):
        prepared(raw)
    del document["states_by_target_center"]["010"]
    table.write_text(json.dumps(document))
    raw["simulator"]["planetary"]["ephemeris"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA256"):
        prepared(raw)


@pytest.mark.parametrize("ecc", [0.5, 1.5])
def test_elliptic_and_hyperbolic_independent_integration(ecc):
    raw = scenario(
        duration=600.0,
        state={
            "reference_axes": "J2000",
            "coes": {
                "a_km": 10000.0 if ecc < 1.0 else -10000.0,
                "ecc": ecc,
                "inc_deg": 20.0,
                "raan_deg": 30.0,
                "argp_deg": 40.0,
                "true_anomaly_deg": 0.0,
            },
        },
    )
    if ecc > 1.0:
        raw["objects"]["probe"]["initial_state"] = {
            "position_inertial_km": [5000.0, 0.0, 0.0],
            "velocity_inertial_km_s": [
                0.0,
                math.sqrt(42828.37362069909 * (2.0 / 5000.0 + 1.0 / 10000.0)),
                0.0,
            ],
        }
    cfg, env, plan = prepared(raw)
    truth = initial_truth(cfg.objects["probe"], env)
    x = np.r_[truth.position_eci_km, truth.velocity_eci_km_s]

    def f(t, x):
        return np.r_[x[3:], -env.gravity.mu_km3_s2 * x[:3] / np.linalg.norm(x[:3]) ** 3]

    ref = solve_ivp(f, (0.0, 600.0), x, method="DOP853", rtol=2e-13, atol=2e-14).y[
        :, -1
    ]
    got = np.array(
        plan.native.propagate(x.tolist(), 0.0, 600.0, "rkf78", 1e-12, 1e-12)[0]
    )
    np.testing.assert_allclose(got[:3], ref[:3], atol=2e-6, rtol=0.0)
    np.testing.assert_allclose(got[3:], ref[3:], atol=2e-9, rtol=0.0)


def test_planetary_impact_and_body_identity(tmp_path):
    from sim.api import SimulationSession

    raw = scenario(
        duration=30.0,
        state={
            "position_inertial_km": [3400.0, 0.0, 0.0],
            "velocity_inertial_km_s": [-1.0, 0.0, 0.0],
        },
    )
    raw["outputs"]["output_dir"] = str(tmp_path / "impact")
    result = SimulationSession(raw).run()
    assert result.summary["termination_reason"] == "central_body_impact"
    assert result.summary["central_body"] == "mars"
    assert result.summary["termination_time_s"] == 5.0


def test_earth_default_explicit_parity():
    from pathlib import Path

    from sim.config.scenario_yaml import load_simulation_yaml

    cfg = load_simulation_yaml(
        Path(__file__).resolve().parents[2] / "configs/automation_smoke.yaml"
    )
    raw = cfg.to_dict()
    raw["simulator"]["central_body"] = "earth"
    assert scenario_config_from_dict(raw).to_dict() == cfg.to_dict()


@pytest.mark.parametrize("field", ["dry_mass_kg", "fuel_mass_kg"])
@pytest.mark.parametrize("value", [0.0, 200.0])
def test_planetary_rejects_unsupported_mass_fields(field, value):
    raw = scenario()
    raw["objects"]["probe"]["specs"][field] = value
    with pytest.raises(ValueError, match="fixed total mass_kg"):
        scenario_config_from_dict(raw)


def test_planetary_preserves_fixed_initial_attitude(tmp_path):
    from sim.api import SimulationSession

    raw = scenario(duration=10.0)
    initial = raw["objects"]["probe"]["initial_state"]
    initial.update(attitude_quat_bn=[0.0, 1.0, 0.0, 0.0], angular_rate_body_rad_s=[1.0, 2.0, 3.0])
    raw["outputs"]["output_dir"] = str(tmp_path / "fixed_attitude")
    result = SimulationSession(raw).run()
    history = result.state_history("probe")
    np.testing.assert_array_equal(history[:, 6:10], np.tile(initial["attitude_quat_bn"], (len(history), 1)))
    np.testing.assert_array_equal(history[:, 10:13], np.tile(initial["angular_rate_body_rad_s"], (len(history), 1)))


def test_planetary_anomaly_alias_precedence_matches_earth():
    from sim.runtime.state_initialization import _rv_from_initial_state

    raw = scenario(state={"coes": {"a_km": 4000.0, "ta_deg": 90.0, "true_anomaly_deg": 0.0}, "reference_axes": "J2000"})
    cfg, env, _ = prepared(raw)
    expected_position, _ = _rv_from_initial_state({"coes": raw["objects"]["probe"]["initial_state"]["coes"]})
    np.testing.assert_array_equal(initial_truth(cfg.objects["probe"], env).position_eci_km, expected_position)


@pytest.mark.parametrize("model,extra", [
    ("none", {"data_path": "unused.json"}),
    ("none", {"sha256": "0" * 64}),
    ("mars_reference_table", {"data_path": "unused.json"}),
    ("mars_reference_table", {"reference_density_kg_m3": 1.0}),
    ("table", {"data_path": "unused.json", "wind_body_m_s": [1.0, 0.0, 0.0]}),
    ("exponential", {"sha256": "0" * 64}),
])
def test_atmosphere_rejects_options_for_other_models(model, extra):
    with pytest.raises(ValueError, match="does not accept"):
        scenario_config_from_dict(scenario(planetary={"atmosphere": {"model": model, **extra}}))


def test_builtin_atmosphere_verifies_optional_pin():
    from sim.dynamics.planetary.resources import DATA_ROOT, ResourceReceipt

    selection = {"model": "mars_reference_table", "sha256": "0" * 64}
    state = {"position_inertial_km": [3546.19, 0.0, 0.0], "velocity_inertial_km_s": [0.0, 3.0, 0.0]}
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        prepared(scenario(planetary={"atmosphere": selection}, state=state))
    selection["sha256"] = ResourceReceipt.read(DATA_ROOT / "mars_reference_atmosphere.json")[0].sha256
    _, env, _ = prepared(scenario(planetary={"atmosphere": selection}, state=state))
    assert env.atmosphere.receipt.sha256 == selection["sha256"]


@pytest.mark.parametrize("method", ["rk4", "rkf78", "dopri5", "adaptive"])
@pytest.mark.parametrize("failing_first", [True, False])
def test_harmonic_domain_exit_keeps_last_valid_sample_and_rolls_back_all_objects(tmp_path, method, failing_first):
    from copy import deepcopy

    from sim.api import SimulationSession

    raw = scenario(duration=30.0, planetary={"gravity": {"model": "gmm2b_static"}}, state={
        "position_inertial_km": [3405.0, 0.0, 0.0], "velocity_inertial_km_s": [-1.0, 0.0, 0.0],
    })
    raw["simulator"]["dynamics"]["orbit"]["integrator"] = method
    safe = deepcopy(raw["objects"]["probe"])
    safe["initial_state"] = {"position_inertial_km": [4000.0, 0.0, 0.0], "velocity_inertial_km_s": [0.0, 3.0, 0.0]}
    raw["objects"] = {"probe": raw["objects"]["probe"], "safe": safe} if failing_first else {"safe": safe, "probe": raw["objects"]["probe"]}
    raw["outputs"]["output_dir"] = str(tmp_path / "domain_exit")
    session = SimulationSession(raw)
    result = session.run()
    assert result.summary["termination_reason"] == "primary_gravity_domain_exit"
    assert result.summary["termination_object_id"] == "probe"
    assert result.summary["termination_time_s"] == 5.0
    np.testing.assert_array_equal(result.time_s, [0.0, 5.0])
    event = result.summary["gravity_domain_exit"]
    assert event["attempted_end_time_s"] == 10.0
    assert not event["physical_impact_recorded"]
    for oid in ("probe", "safe"):
        history = result.state_history(oid)
        assert np.all(np.linalg.norm(history[:, :3], axis=1) > 3397.0)
        truth = session._engine.agents[oid].truth
        np.testing.assert_array_equal(np.r_[truth.position_eci_km, truth.velocity_eci_km_s,
                                            truth.attitude_quat_bn, truth.angular_rate_body_rad_s,
                                            truth.mass_kg], history[-1])
    with sqlite3.connect(tmp_path / "domain_exit/review/run.sqlite") as conn:
        assert conn.execute("SELECT max(time_s) FROM planetary_state").fetchone()[0] == 5.0
        assert conn.execute("SELECT count(*) FROM planetary_state").fetchone()[0] == 4


def test_atmosphere_errors_are_not_reclassified_as_gravity_domain_exit(tmp_path):
    from sim.api import SimulationSession

    raw = scenario(planetary={"gravity": {"model": "gmm2b_static"}, "atmosphere": {"model": "mars_reference_table"}}, state={
        "position_inertial_km": [3546.19, 0.0, 0.0], "velocity_inertial_km_s": [50.0, 0.0, 0.0],
    })
    raw["outputs"]["output_dir"] = str(tmp_path / "atmosphere_error")
    with pytest.raises(ValueError, match="atmosphere altitude"):
        SimulationSession(raw).run()


def test_mars_solar_geometry_and_cylindrical_shadow(tmp_path):
    from sim.dynamics.planetary.frames import prepare_rotation

    frame = prepare_rotation("mars", {"model": "iau2000"}, 2457065.5)
    table = tmp_path / "mars_sun.json"
    table.write_text(
        json.dumps(
            {
                "observer_center_id": 499,
                "inertial_axes": "J2000",
                "aberration": "NONE",
                "position_units": "km",
                "velocity_units": "km/s",
                "time_scale": "TDB",
                "source": "Synthetic fixed solar geometry for force/shadow checks only",
                "tdb_seconds_past_j2000": [
                    frame.elapsed_tdb_s(0.0) - 1,
                    frame.elapsed_tdb_s(120.0) + 1,
                ],
                "states_by_target_center": {"10": [[2e8, 0, 0, 0, 0, 0]] * 2},
            }
        )
    )
    _, _, plan = prepared(
        scenario(
            planetary={
                "third_bodies": [
                    {"target_center_id": 10, "mu_km3_s2": 132712440041.27942}
                ],
                "srp": {"enabled": True, "shadow_model": "cylindrical"},
                "ephemeris": {"data_path": str(table)},
            }
        )
    )
    for r in (np.array([4000.0, 0, 0]), np.array([-4000.0, 0, 0])):
        forces = plan.stage(37.0, np.r_[r, [0, 3, 0]])[1]["forces_km_s2"]
        delta = np.array([2e8, 0, 0]) - r
        expected = 132712440041.27942 * (
            delta / np.linalg.norm(delta) ** 3 - np.array([1 / (2e8) ** 2, 0, 0])
        )
        np.testing.assert_allclose(
            forces["third_body_10"], expected, atol=2e-20, rtol=1e-12
        )
        if r[0] < 0:
            assert forces["srp"] == [0, 0, 0]
        else:
            pressure = 1361 / 299792458 * (149597870.7 / np.linalg.norm(delta)) ** 2
            np.testing.assert_allclose(
                forces["srp"],
                [-pressure * 1.3 * 2 / 100 / 1000, 0, 0],
                rtol=2e-15,
                atol=0,
            )


def test_sun_external_planet_uses_differential_geometry(tmp_path):
    from sim.dynamics.planetary.frames import prepare_rotation

    frame = prepare_rotation("sun", {"model": "iau2000"}, 2457065.5)
    p = tmp_path / "venus.json"
    p.write_text(
        json.dumps(
            {
                "observer_center_id": 10,
                "inertial_axes": "J2000",
                "aberration": "NONE",
                "position_units": "km",
                "velocity_units": "km/s",
                "time_scale": "TDB",
                "source": "Synthetic Venus center fixture, no astronomy accuracy claim",
                "tdb_seconds_past_j2000": [
                    frame.elapsed_tdb_s(0.0) - 1.0,
                    frame.elapsed_tdb_s(120.0) + 1.0,
                ],
                "states_by_target_center": {
                    "299": [[1e8, 5e7, 0.0, 0.0, 0.0, 0.0]] * 2
                },
            }
        )
    )
    _, env, plan = prepared(
        scenario(
            "sun",
            planetary={
                "third_bodies": [{"target_center_id": 299, "mu_km3_s2": 324858.592}],
                "ephemeris": {"data_path": str(p)},
                "srp": {"enabled": True},
            },
        )
    )
    r = np.array([1.5e8, 0.0, 0.0])
    target = np.array([1e8, 5e7, 0.0])
    delta = target - r
    forces = plan.stage(0.0, np.r_[r, [0.0, 30.0, 0.0]])[1]["forces_km_s2"]
    expected = 324858.592 * (
        delta / np.linalg.norm(delta) ** 3 - target / np.linalg.norm(target) ** 3
    )
    np.testing.assert_allclose(
        forces["third_body_299"], expected, atol=1e-23, rtol=1e-13
    )
    assert forces["srp"][0] > 0.0
