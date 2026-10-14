"""Independent field, trajectory, routing and resource-contract evidence."""

import copy
import json
import math
import pickle
import sqlite3
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from sim.config.scenario_yaml import scenario_config_from_dict
from sim.dynamics.orbit.accelerations import OrbitContext, accel_j2
from sim.dynamics.orbit.environment import EARTH_MU_KM3_S2
from sim.dynamics.planetary.additional_gravity import _clock, prepare_additional_gravity
from sim.dynamics.planetary.preparation import prepare_environment
from sim.dynamics.planetary.propagation import PlanetaryForcePlan
from sim.runtime.satellite_factory import _build_orbit_propagator
from sim.tests.test_planetary_onp import reference_harmonic, scenario


def resource(tmp_path, body="moon", tesseral=True, rotation=0.0, radius=None):
    radius = radius or (1737.4 if body == "moon" else 6378.137)
    mu = 4902.8 if body == "moon" else EARTH_MU_KM3_S2
    coeff, label, frame = [tmp_path / f"{body}.{ext}" for ext in ("sha", "lbl", "json")]
    coeff.write_text(
        f"{radius},{mu},0,2,2,1,0,0\n2,0,-0.0002,0,0,0\n2,1,{1.0e-5 if tesseral else 0},{2.0e-5 if tesseral else 0},0,0\n2,2,{3.0e-5 if tesseral else 0},{-1.0e-5 if tesseral else 0},0,0\n"
    )
    label.write_text(f'TARGET_NAME = "{body.upper()}"\n')
    frame.write_text(
        json.dumps(
            {
                "body_id": body,
                "frame_id": f"TEST_{body.upper()}",
                "inertial_axes": "J2000",
                "source": "Synthetic test frame, not mission data",
                "kind": "euler_series",
                "axes": [2],
                "angles": [{"offset_deg": rotation, "rate_deg_day": 0.0}],
                "valid_tdb_seconds": [-1.0e12, 1.0e12],
            }
        )
    )
    return {
        "body_id": body,
        "gravity": {
            "model": "pds_sha_static",
            "coeff_path": str(coeff),
            "label_path": str(label),
            "frame_id": f"TEST_{body.upper()}",
            "max_degree": 2,
            "max_order": 2,
        },
        "gravity_frame": {"model": "resource", "data_path": str(frame)},
    }


def raw_scenario(tmp_path, *, origin="earth", tesseral=True, method="rkf78", rotation=0.0):
    raw = scenario("moon" if origin == "moon" else "mars", duration=60.0)
    raw["simulator"]["central_body"] = origin
    raw["simulator"]["initial_jd_utc"] = 2459669.5
    raw["simulator"]["dynamics"]["orbit"].update(
        numeric_backend="rust", integrator=method, adaptive_atol=1.0e-12, adaptive_rtol=1.0e-12
    )
    raw["simulator"]["dt_s"] = 0.5
    if origin == "earth":
        raw["simulator"].pop("planetary")
        raw["simulator"]["dynamics"]["attitude"] = {"enabled": False}
        raw["objects"]["probe"]["runtime_profile"] = "trajectory_only"
        raw["objects"]["probe"]["initial_state"] = {
            "position_eci_km": [381900.0, 100.0, 300.0],
            "velocity_eci_km_s": [0.0, 1.2, 0.0],
        }
        raw["outputs"].update(plots={"enabled": False}, animations={"enabled": False}, review={"enabled": True})
    source = resource(tmp_path, body="moon" if origin == "earth" else "earth", tesseral=tesseral, rotation=rotation)
    clock = _clock(raw["simulator"]["initial_jd_utc"])
    ephem = tmp_path / "ephemeris.json"
    ephem.write_text(
        json.dumps(
            {
                "observer_center_id": 399 if origin == "earth" else 301,
                "inertial_axes": "J2000",
                "aberration": "NONE",
                "position_units": "km",
                "velocity_units": "km/s",
                "time_scale": "TDB",
                "source": "Synthetic stationary fixture; not an astronomical ephemeris",
                "tdb_seconds_past_j2000": [clock.elapsed_tdb_s(0) - 100.0, clock.elapsed_tdb_s(60) + 100.0],
                "states_by_target_center": {
                    "301" if origin == "earth" else "399": [[384400.0, 0.0, 0.0, 0.0, 0.0, 0.0]] * 2
                },
            }
        )
    )
    raw["simulator"]["additional_gravity"] = {
        "ephemeris": {"model": "table", "data_path": str(ephem)},
        "sources": [source],
    }
    return raw


@pytest.mark.parametrize("origin", ["earth", "moon"])
@pytest.mark.parametrize("rotation", [0.0, 73.0])
def test_full_rotated_field_and_origin_correction(tmp_path, origin, rotation):
    raw = raw_scenario(tmp_path, origin=origin, rotation=rotation)
    cfg = scenario_config_from_dict(raw)
    additional = prepare_additional_gravity(cfg)
    from sim.dynamics.planetary.additional_gravity import external_body
    from sim.dynamics.planetary.frames import prepare_rotation
    from sim.dynamics.planetary.gravity import prepare_gravity

    s = cfg.simulator.additional_gravity["sources"][0]
    gravity = prepare_gravity(external_body(s["body_id"]), s["gravity"])
    R, _ = prepare_rotation(s["body_id"], s["gravity_frame"], cfg.simulator.initial_jd_utc).evaluate(0.0)
    x = np.array([374400.0, 2100.0, -3300.0, 0.0, 0.0, 0.0])
    body = np.array([384400.0, 0.0, 0.0])

    def field(r):
        return -gravity.mu_km3_s2 * r / np.linalg.norm(r) ** 3 + R.T @ reference_harmonic(gravity, R @ r)

    got = np.asarray(list(additional.components(0.0, x).values())[0])
    expected = field(x[:3] - body) - field(-body)
    np.testing.assert_allclose(got, expected, atol=2.0e-16, rtol=1.0e-9)
    assert additional.components(0.0, np.zeros(6)) == {
        f"additional_gravity_{additional.sources[0]['target_center_id']}": [0.0, 0.0, 0.0]
    }


@pytest.mark.parametrize("method", ["rk4", "rkf78", "dopri5", "adaptive"])
def test_earth_trajectory_matches_independent_zonal_reference(tmp_path, method):
    cfg = scenario_config_from_dict(raw_scenario(tmp_path, tesseral=False, method=method))
    prop = _build_orbit_propagator(cfg)
    x = np.array([381900.0, 100.0, 300.0, 0.0, 1.2, 0.0])
    initial = x.copy()
    body = np.array([384400.0, 0.0, 0.0])

    def field(r):
        return -4902.8 * r / np.linalg.norm(r) ** 3 + accel_j2(r, 4902.8, 0.0002 * math.sqrt(5), 1737.4)

    def derivative(t, state):
        r = state[:3]
        return np.r_[state[3:], -EARTH_MU_KM3_S2 * r / np.linalg.norm(r) ** 3 + field(r - body) - field(-body)]

    reference = solve_ivp(derivative, [0.0, 60.0], initial, method="DOP853", rtol=2.0e-13, atol=1.0e-13).y[:, -1]
    # A zero-length segment must preserve state and not poison adaptive reuse.
    np.testing.assert_array_equal(
        prop.propagate(x, 0.0, 0.0, np.zeros(3), {}, OrbitContext(EARTH_MU_KM3_S2, 100.0)), x
    )
    for i in range(120):
        x = prop.propagate(x, 0.5, i * 0.5, np.zeros(3), {}, OrbitContext(EARTH_MU_KM3_S2, 100.0))
    np.testing.assert_allclose(x, reference, atol=1.0e-8, rtol=0.0)
    assert prop.last_numeric_path == "rust_additional_gravity"
    assert prop.try_propagate_sampled_steps(None) is None


def test_planetary_composition_and_primary_harmonics(tmp_path):
    raw = raw_scenario(tmp_path, origin="moon")
    primary = resource(tmp_path, body="moon")
    raw["simulator"]["planetary"] = {"gravity": primary["gravity"], "gravity_frame": primary["gravity_frame"]}
    cfg = scenario_config_from_dict(raw)
    e = prepare_environment(cfg)
    plan = PlanetaryForcePlan(e, mass_kg=100.0)
    x = np.array([4000.0, 100.0, 200.0, 0.0, 1.0, 0.0])
    total, d = plan.stage(0.0, x)
    assert set(d["forces_km_s2"]) == {"point_mass", "harmonics", "drag", "srp", "additional_gravity_399"}
    np.testing.assert_allclose(total, np.sum(list(d["forces_km_s2"].values()), axis=0), atol=1.0e-18)
    for method in ("rk4", "rkf78", "dopri5", "adaptive"):
        assert np.all(np.isfinite(plan.native.propagate(x.tolist(), 0.0, 1.0, method, 1.0e-12, 1.0e-12)[0]))
    assert plan.native.propagate(x.tolist(), 0.0, 20.0, "adaptive", 1.0e-12, 1.0e-12) == plan.native.propagate(
        x.tolist(), 0.0, 20.0, "dopri5", 1.0e-12, 1.0e-12
    )


def test_earth_external_icgem_and_native_eop(tmp_path):
    raw = raw_scenario(tmp_path, origin="moon")
    source = raw["simulator"]["additional_gravity"]["sources"][0]
    gfc = tmp_path / "earth.gfc"
    gfc.write_text(
        "earth_gravity_constant 398600441500000\nradius 6378137\nnorm fully_normalized\nend_of_head\ngfc 2 0 -0.0002 0\ngfc 2 1 0.00001 0.00002\ngfc 2 2 0.00003 -0.00001\n"
    )
    source["gravity"] = {"model": "earth_static", "coeff_path": str(gfc), "frame_id": "EARTH_TEST", "max_degree": 2}
    eop = tmp_path / "synthetic_eop.txt"
    eop.write_text(
        "NUM_OBSERVED_POINTS 2\n2022 03 31 59669 0.1 0.2 -0.1 0.0 0.0 0.0 0.0 0.0 37\n2022 04 01 59670 0.1 0.2 -0.1 0.0 0.0 0.0 0.0 0.0 37\n"
    )
    source["gravity_frame"] = {
        "model": "earth",
        "frame_id": "EARTH_TEST",
        "eop_path": str(eop),
    }
    cfg = scenario_config_from_dict(raw)
    a = prepare_additional_gravity(cfg)
    assert a.sources[0]["gravity_frame"]["model"] == "iau76_80_eop"
    from sim.dynamics.orbit.frames import eci_to_ecef_rotation_hpop_like
    from sim.dynamics.planetary.additional_gravity import _earth_gravity, external_body

    gravity = _earth_gravity(external_body("earth"), cfg.simulator.additional_gravity["sources"][0]["gravity"])
    R = eci_to_ecef_rotation_hpop_like(30.0, jd_utc_start=cfg.simulator.initial_jd_utc, eop_path=str(eop))
    x = np.array([4000.0, 100.0, 200.0, 0.0, 0.0, 0.0])
    b = np.array([384400.0, 0.0, 0.0])

    def field(r):
        return -gravity.mu_km3_s2 * r / np.linalg.norm(r) ** 3 + R.T @ reference_harmonic(gravity, R @ r)

    np.testing.assert_allclose(
        a.native.stage(30.0, x.tolist())[0], field(x[:3] - b) - field(-b), atol=2.0e-18, rtol=1.0e-10
    )
    gfc.write_text(gfc.read_text() + "gfct 2 0 0 0\n")
    with pytest.raises(ValueError, match="static"):
        prepare_additional_gravity(cfg)


@pytest.mark.parametrize("extra,match", [
    ("GFC 2 0 -0.002 0\n", "duplicate"),
    ("GfC 2 1 0.002 0.003\n", "duplicate"),
    ("GFCT 2 0 -0.002 0\n", "static"),
    ("TrNd 2 0 0 0\n", "static"),
    ("NORM unnormalized\n", "duplicate"),
    ("RADIUS 6378137\n", "duplicate"),
    ("EARTH_GRAVITY_CONSTANT 398600441500000\n", "duplicate"),
])
def test_external_icgem_case_and_duplicate_guards(tmp_path, extra, match):
    from sim.dynamics.planetary.additional_gravity import _earth_gravity, external_body

    coeff = tmp_path / "earth.gfc"
    coeff.write_text("earth_gravity_constant 398600441500000\nradius 6378137\nnorm fully_normalized\nend_of_head\ngfc 2 0 -0.0002 0\ngfc 2 1 0.00001 0.00002\ngfc 2 2 0.00003 -0.00001\n" + extra)
    with pytest.raises(ValueError, match=match):
        _earth_gravity(external_body("earth"), {"coeff_path": str(coeff), "frame_id": "TEST", "max_degree": 2, "max_order": 2})


def test_external_icgem_compiles_only_the_validated_records(tmp_path):
    from sim.dynamics.planetary.additional_gravity import _earth_gravity, external_body

    coeff = tmp_path / "earth.gfc"
    coeff.write_text("EARTH_GRAVITY_CONSTANT 398600441500000\nRadius 6378137\nNorm FULLY_NORMALIZED\nend_of_head\nGFC 2 0 -0.0002 0\nGfC 2 1 0.00001 0.00002\ngfc 2 2 0.00003 -0.00001\n")
    gravity = _earth_gravity(external_body("earth"), {"coeff_path": str(coeff), "frame_id": "TEST", "max_degree": 2, "max_order": 2})
    assert gravity.mu_km3_s2 == 398600.4415
    assert gravity.radius_km == 6378.137
    assert gravity.compiled.all_normalized
    assert gravity.compiled.c_nm[2, 0] == -0.0002
    assert gravity.compiled.s_nm[2, 1] == 0.00002


@pytest.mark.parametrize("method", ["rk4", "rkf78", "dopri5", "adaptive"])
def test_planetary_primary_domain_exit_with_external_field(tmp_path, method):
    from sim.api import SimulationSession

    raw = raw_scenario(tmp_path, origin="moon", method=method)
    primary = resource(tmp_path, body="moon")
    raw["simulator"]["planetary"] = {"gravity": primary["gravity"], "gravity_frame": primary["gravity_frame"]}
    raw["simulator"]["dt_s"] = 5.0
    raw["objects"]["probe"]["initial_state"] = {
        "position_inertial_km": [1745.0, 0.0, 0.0], "velocity_inertial_km_s": [-1.0, 0.0, 0.0],
    }
    raw["outputs"]["output_dir"] = str(tmp_path / "extended_domain_exit")
    result = SimulationSession(raw).run()
    assert result.summary["termination_reason"] == "primary_gravity_domain_exit"
    np.testing.assert_array_equal(result.time_s, [0.0, 5.0])
    assert result.summary["gravity_domain_exit"]["gravity_reference_radius_km"] == 1737.4
    assert result.payload["planetary_environment"]["additional_gravity"]["sources"][0]["body_id"] == "earth"



@pytest.mark.parametrize(
    "change,match",
    [
        ("duplicate", "unique"),
        ("self", "external"),
        ("pointmass", "requires harmonics"),
        ("frame", "frame identity"),
        ("double_moon", "twice"),
        ("python", "Rust"),
        ("degree", "degree"),
        ("ephemeris_origin", "physical origin"),
        ("coverage", "cover"),
        ("unknown", "unsupported field"),
    ],
)
def test_invalid_selections_fail_closed(tmp_path, change, match):
    raw = raw_scenario(tmp_path)
    a = raw["simulator"]["additional_gravity"]
    s = a["sources"][0]
    if change == "duplicate":
        a["sources"].append(copy.deepcopy(s))
    elif change == "self":
        s["body_id"] = "earth"
    elif change == "pointmass":
        s["gravity"] = {"model": "point_mass"}
    elif change == "frame":
        s["gravity"]["frame_id"] = "WRONG"
    elif change == "double_moon":
        raw["simulator"]["dynamics"]["orbit"]["third_body_moon"] = True
    elif change == "python":
        raw["simulator"]["dynamics"]["orbit"]["numeric_backend"] = "python"
    elif change == "degree":
        s["gravity"]["max_degree"] = 21
    elif change == "unknown":
        a["invented"] = True
    else:
        p = Path(a["ephemeris"]["data_path"])
        data = json.loads(p.read_text())
        if change == "ephemeris_origin":
            data["observer_center_id"] = 301
        else:
            data["tdb_seconds_past_j2000"] = [0.0, 1.0]
        p.write_text(json.dumps(data))
    with pytest.raises(ValueError, match=match):
        scenario_config_from_dict(raw)


def test_resource_mutation_pickle_and_domains(tmp_path):
    cfg = scenario_config_from_dict(raw_scenario(tmp_path))
    prop = _build_orbit_propagator(cfg)
    serialized = pickle.dumps(prop)
    restored = pickle.loads(serialized)
    x = np.array([374400.0, 2100.0, -3300.0, 0.0, 0.0, 0.0])
    np.testing.assert_array_equal(
        prop.additional_gravity.native.stage(0.0, x.tolist()), restored.additional_gravity.native.stage(0.0, x.tolist())
    )
    with pytest.raises(ValueError, match="reference sphere"):
        prop.additional_gravity.native.stage(0.0, [384400.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="domain"):
        prop.additional_gravity.native.stage(61.0, x.tolist())
    p = Path(cfg.simulator.additional_gravity["sources"][0]["gravity"]["coeff_path"])
    p.write_text(p.read_text() + "\n")
    with pytest.raises(ValueError, match="SHA256"):
        prop.propagate(x, 1.0, 0.0, np.zeros(3), {}, OrbitContext(EARTH_MU_KM3_S2, 100.0))
    with pytest.raises(ValueError, match="changed since serialization"):
        pickle.loads(serialized)


def test_ordinary_earth_retains_exact_type_and_serialization(tmp_path):
    from sim.dynamics.orbit.propagator import OrbitPropagator

    raw = raw_scenario(tmp_path)
    raw["simulator"].pop("additional_gravity")
    cfg = scenario_config_from_dict(raw)
    assert "additional_gravity" not in cfg.to_dict()["simulator"]
    assert type(_build_orbit_propagator(cfg)) is OrbitPropagator


def test_primary_earth_harmonics_plus_lunar_harmonics(tmp_path):
    from sim.dynamics.orbit.spherical_harmonics import configure_spherical_harmonics_env

    raw = raw_scenario(tmp_path, tesseral=False)
    gfc = tmp_path / "primary.gfc"
    gfc.write_text(
        "earth_gravity_constant 398600441500000\nradius 6378137\nnorm fully_normalized\ngfc 2 0 -0.0004 0\ngfc 2 1 0 0\ngfc 2 2 0 0\n"
    )
    orbit = raw["simulator"]["dynamics"]["orbit"]
    orbit["spherical_harmonics"] = {
        "enabled": True,
        "source": "icgem",
        "coeff_path": str(gfc),
        "degree": 2,
        "order": 2,
        "frame_model": "simple",
    }
    cfg = scenario_config_from_dict(raw)
    prop = _build_orbit_propagator(cfg)
    env = configure_spherical_harmonics_env({"jd_utc_start": cfg.simulator.initial_jd_utc}, orbit)
    x = np.array([381900.0, 100.0, 300.0, 0.0, 1.2, 0.0])
    extra = np.asarray(list(prop.additional_gravity.components(0.0, x).values())[0])
    expected = (
        -EARTH_MU_KM3_S2 * x[:3] / np.linalg.norm(x[:3]) ** 3
        + accel_j2(x[:3], EARTH_MU_KM3_S2, 0.0004 * math.sqrt(5), 6378.137)
        + extra
    )
    got = prop.acceleration_at(
        t_s=0.0, x_eci=x, command_accel_eci_km_s2=np.zeros(3), env=env, ctx=OrbitContext(EARTH_MU_KM3_S2, 100.0)
    )
    np.testing.assert_allclose(got, expected, atol=1.0e-18, rtol=1.0e-13)


def test_multiple_sources_and_monopole_limit(tmp_path):
    raw = raw_scenario(tmp_path)
    extra = raw["simulator"]["additional_gravity"]
    extra["sources"].append({"body_id": "mars", "gravity": {"model": "gmm2b_static", "max_degree": 2}})
    ephem = Path(extra["ephemeris"]["data_path"])
    data = json.loads(ephem.read_text())
    data["states_by_target_center"]["499"] = [[2.0e8, 1.0e8, 0.0, 0.0, 0.0, 0.0]] * 2
    ephem.write_text(json.dumps(data))
    cfg = scenario_config_from_dict(raw)
    additional = prepare_additional_gravity(cfg)
    assert additional.sources[1]["gravity_frame"]["frame_id"] == "IAU1991_MARS"
    parts = additional.components(0.0, [381900.0, 100.0, 300.0, 0.0, 0.0, 0.0])
    assert set(parts) == {"additional_gravity_301", "additional_gravity_499"}
    assert all(np.linalg.norm(part) > 0.0 for part in parts.values())
    import oel_rust_orbit as native

    rotation = ([2], [0.0, 0.0, 0.0], [], [-1.0e12, 1.0e12], False)
    zero = native.AdditionalGravityPlan(
        399,
        0.0,
        60.0,
        ([-100.0, 200.0], [301], [[384400.0, 0.0, 0.0, 0.0, 0.0, 0.0] * 2]),
        [(301, 4902.8, 1737.4, (2, 2), ([0.0] * 9, [0.0] * 9), rotation)],
        [None],
    )
    r = np.array([7000.0, 200.0, 300.0])
    b = np.array([384400.0, 0.0, 0.0])
    expected = 4902.8 * ((b - r) / np.linalg.norm(b - r) ** 3 - b / np.linalg.norm(b) ** 3)
    np.testing.assert_allclose(zero.stage(0.0, [*r, 0.0, 0.0, 0.0])[0], expected, atol=1.0e-22, rtol=0.0)


@pytest.mark.parametrize(
    "section,key",
    [("gravity", "coeff_path"), ("gravity", "label_path"), ("gravity_frame", "data_path"), ("ephemeris", "data_path")],
)
def test_additional_paths_stay_inside_authorized_roots(tmp_path, section, key):
    from sim.security.config_paths import ConfigPathPolicy, ConfigPathSecurityError

    config_dir = tmp_path / "config"
    config_dir.mkdir()
    raw = raw_scenario(config_dir)
    selection = raw["simulator"]["additional_gravity"]
    if section == "ephemeris":
        selection[section][key] = str(tmp_path / "external_resource")
    else:
        selection["sources"][0][section][key] = str(tmp_path / "external_resource")
    policy = ConfigPathPolicy.default(config_path=config_dir / "scenario.yaml", workspace_root=config_dir)
    with pytest.raises(ConfigPathSecurityError, match="additional_gravity"):
        scenario_config_from_dict(raw, source_path=config_dir / "scenario.yaml", path_policy=policy)


def test_earth_run_review_evidence(tmp_path):
    from sim.api import SimulationSession

    raw = raw_scenario(tmp_path)
    raw["simulator"]["duration_s"] = 1.0
    raw["outputs"]["output_dir"] = str(tmp_path / "output")
    cfg = scenario_config_from_dict(raw)
    SimulationSession(cfg).run()
    with sqlite3.connect(tmp_path / "output/review/run.sqlite") as db:
        assert db.execute("SELECT count(*) FROM additional_gravity_acceleration").fetchone()[0] == 3
        provenance = json.loads(db.execute("SELECT provenance_json FROM additional_gravity_environment").fetchone()[0])
        assert provenance["sources"][0]["target_center_id"] == 301
        assert db.execute("SELECT count(*) FROM object_state").fetchone()[0] == 3
