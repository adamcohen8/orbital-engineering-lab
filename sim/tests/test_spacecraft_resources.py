import json
import math
import sqlite3

import numpy as np
import pytest
import yaml

from sim.config import scenario_config_from_dict
from sim.core.models import Command, StateTruth
from sim.dynamics.model import OrbitalAttitudeDynamics
from sim.dynamics.orbit.environment import EARTH_RADIUS_KM
from sim.dynamics.orbit.epoch import AU_KM
from sim.dynamics.spacecraft_geometry import RectangularPrismGeometry
from sim.spacecraft_resources import SpacecraftResources, validate_resource_specs
from sim.spacecraft_resources.model import SIGMA, earth_radiation


def specs():
    return {
        "thermal": {
            "enabled": True,
            "initial_temperature_k": 290.0,
            "heat_capacity_j_k": 10000.0,
            "solar_absorptivity": 0.3,
            "infrared_emissivity": 0.8,
            "projected_area_m2": 1.0,
            "radiating_area_m2": 4.0,
            "internal_heat_w": 0.0,
        },
        "power": {
            "enabled": True,
            "panels": [{"area_m2": 2.0, "normal_body": [1, 0, 0], "efficiency": 0.3}],
            "baseline_load_w": 100.0,
            "battery_capacity_wh": 100.0,
            "initial_soc": 0.5,
            "max_charge_w": 200.0,
            "max_discharge_w": 200.0,
        },
    }


def truth(x=7000.0, q=None, t=0.0):
    return StateTruth(
        np.array([x, 0.0, 0.0]),
        np.array([0.0, 7.5, 0.0]),
        np.array(q if q is not None else [1.0, 0.0, 0.0, 0.0]),
        np.zeros(3),
        100.0,
        t,
    )


ENV = {"sun_pos_eci_km": [AU_KM, 0.0, 0.0], "srp_shadow_model": "conical", "srp": False}


def advance(model, state=None, h=1.0):
    start = truth() if state is None else state
    end = start.copy()
    end.t_s += h
    end.resource_state = model.advance(start, end, ENV, h)
    return end


def test_day_night_and_energy_ledgers():
    model = SpacecraftResources.from_specs(specs())
    day = advance(model).resource_state
    night = advance(model, truth(-7000.0)).resource_state
    assert day["sunlit_fraction"] == 1.0
    assert day["solar_generation_w"] > 700.0
    assert day["battery_soc"] > 0.5
    assert night["sunlit_fraction"] == 0.0
    assert night["solar_generation_w"] == 0.0
    assert night["solar_heat_w"] == 0.0
    assert night["albedo_heat_w"] == 0.0
    assert night["earth_ir_heat_w"] > 0.0
    assert night["battery_soc"] < 0.5
    for row in (day, night):
        assert abs(row["thermal_balance_residual_w"]) < 1e-8
        assert abs(row["power_balance_residual_w"]) < 1e-10
        expected = (row["battery_charge_w"] * 0.95 - row["battery_discharge_w"] / 0.95) / 3600.0
        assert row["battery_energy_wh"] - 50.0 == pytest.approx(expected)


@pytest.mark.parametrize("q", [[0.0, 0.0, 0.0, 1.0], [math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)]])
def test_panels_backside_and_edge_on(q):
    row = advance(SpacecraftResources.from_specs(specs()), truth(q=q)).resource_state
    assert row["solar_generation_w"] == pytest.approx(0.0, abs=1e-10)


def test_full_empty_battery_and_unmet_load():
    cfg = specs()
    cfg["power"]["initial_soc"] = 1.0
    full = advance(SpacecraftResources.from_specs(cfg), h=1000.0).resource_state
    assert full["battery_soc"] == 1.0
    assert full["curtailed_power_w"] > 0.0
    assert full["battery_charge_w"] == 0.0
    cfg["power"]["initial_soc"] = 0.001
    empty = advance(SpacecraftResources.from_specs(cfg), truth(-7000.0), h=1000.0).resource_state
    assert empty["battery_soc"] == pytest.approx(0.0, abs=1e-15)
    assert empty["unmet_load_w"] > 0.0
    assert empty["load_served_w"] < empty["load_demand_w"]
    assert abs(empty["power_balance_residual_w"]) < 1e-10


def test_thermal_equilibrium_and_constant_heating():
    cfg = specs()
    cfg.pop("power")
    cfg["thermal"].update(solar_absorptivity=0.0, earth_ir_w_m2=0.0, internal_heat_w=0.8 * SIGMA * 4.0 * 290.0**4)
    row = advance(SpacecraftResources.from_specs(cfg), h=100.0).resource_state
    assert row["temperature_k"] == pytest.approx(290.0, abs=1e-10)
    cfg["thermal"].update(infrared_emissivity=0.0, internal_heat_w=100.0)
    row = advance(SpacecraftResources.from_specs(cfg), h=100.0).resource_state
    assert row["temperature_k"] == pytest.approx(291.0, abs=1e-10)


def test_earth_solid_angle_and_albedo_geometry():
    rays, weights, illumination = earth_radiation(np.array([7000.0, 0.0, 0.0]), np.array([AU_KM, 0.0, 0.0]), 16)
    expected = 2 * math.pi * (1 - math.sqrt(1 - (EARTH_RADIUS_KM / 7000.0) ** 2))
    assert sum(weights) == pytest.approx(expected, rel=1e-13)
    assert np.max(abs(np.linalg.norm(rays, axis=1) - 1)) < 1e-12
    assert np.all(illumination > 0.0)
    _, _, dark = earth_radiation(np.array([-7000.0, 0.0, 0.0]), np.array([AU_KM, 0.0, 0.0]), 16)
    assert np.all(dark == 0.0)


@pytest.mark.parametrize(
    "position, sun_position, order",
    [
        ([7000.0, 0.0, float("nan")], [AU_KM, 0.0, 0.0], 16),
        ([7000.0, 0.0, 0.0], [AU_KM, float("inf"), 0.0], 16),
        ([7000.0, 0.0, 0.0], [AU_KM, 0.0, 0.0], 3),
        ([7000.0, 0.0, 0.0], [AU_KM, 0.0, 0.0], 65),
    ],
)
def test_earth_radiation_rejects_nonfinite_vectors_and_invalid_order(position, sun_position, order):
    with pytest.raises(ValueError):
        earth_radiation(position, sun_position, order, numeric_backend="python")


def test_geometry_area_and_rotation():
    cfg = specs()
    cfg.pop("power")
    cfg["thermal"].pop("projected_area_m2")
    cfg["thermal"]["area_mode"] = "geometry"
    model = SpacecraftResources.from_specs(cfg, RectangularPrismGeometry(1.0, 2.0, 3.0))
    assert model._area(np.eye(3)) == pytest.approx([6.0, 3.0, 2.0])
    with pytest.raises(ValueError, match="requires spacecraft geometry"):
        SpacecraftResources.from_specs(cfg)


def test_independent_optional_models():
    assert SpacecraftResources.from_specs({}) is None
    for name, output, absent in (
        ("thermal", "temperature_k", "battery_soc"),
        ("power", "battery_soc", "temperature_k"),
    ):
        row = advance(SpacecraftResources.from_specs({name: specs()[name]})).resource_state
        assert output in row and absent not in row


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("thermal", "heat_capacity_j_k", 0),
        ("thermal", "solar_absorptivity", 1.1),
        ("thermal", "initial_temperature_k", float("nan")),
        ("thermal", "emissivity", 0.8),
        ("thermal", "earth_quadrature_order", 1),
        ("power", "initial_soc", -1),
        ("power", "battery_capacity_wh", 0),
        ("power", "charge_efficiency", 0),
        ("power", "baseline_load_w", True),
        ("power", "max_discharge_w", -1),
    ],
)
def test_invalid_configuration(section, key, value):
    cfg = specs()
    cfg[section][key] = value
    with pytest.raises(ValueError):
        validate_resource_specs(cfg)


def test_mechanical_substeps_do_not_multiply_mass_consumption():
    model = SpacecraftResources.from_specs({"power": specs()["power"]})
    dynamics = OrbitalAttitudeDynamics(398600.4418, np.eye(3), propagate_attitude=False, resource_model=model)
    command = Command(mode_flags={"delta_mass_kg": 1.0, "min_mass_kg": 50.0})
    state = truth()
    following = dynamics.step(state, command, ENV, 10.0)
    assert following.mass_kg == pytest.approx(99.0)
    assert following.t_s == 10.0
    assert state.resource_state is None
    assert command.mode_flags["delta_mass_kg"] == 1.0


def scenario(tmp_path, duration=3.0):
    document = yaml.safe_load(open("configs/spacecraft_resources_demo.yaml"))
    document["simulator"].update(duration_s=duration, dt_s=1.0)
    document["simulator"]["dynamics"]["orbit"]["orbit_substep_s"] = 1.0
    document["outputs"]["output_dir"] = str(tmp_path / "run")
    return document


def test_yaml_api_artifact_and_review_roundtrip(tmp_path):
    from sim.api import SimulationConfig, SimulationSession

    session = SimulationSession(SimulationConfig.from_dict(scenario(tmp_path)))
    initial = session.reset()
    assert initial.spacecraft_resources["spacecraft"]["temperature_k"] == 290.0
    live = session.step()
    assert live.spacecraft_resources["spacecraft"]["time_s"] == 1.0
    result = session.run()
    rows = result.payload["spacecraft_resources"]["spacecraft"]
    assert len(rows) == 4
    assert rows[-1]["time_s"] == 3.0
    root = tmp_path / "run"
    assert json.loads((root / "spacecraft_resources.json").read_text())["semantics"] == "simulation_truth"
    with sqlite3.connect(root / "review/run.sqlite") as conn:
        assert conn.execute("SELECT count(*) FROM spacecraft_resources").fetchone()[0] == 4
        assert (
            conn.execute("SELECT max(abs(thermal_balance_residual_w)) FROM spacecraft_resources").fetchone()[0] < 1e-7
        )


def test_unsupported_propagation_fails_validation(tmp_path):
    doc = scenario(tmp_path)
    doc["simulator"]["dynamics"]["orbit"]["model"] = "cr3bp"
    with pytest.raises(ValueError, match="resources require"):
        scenario_config_from_dict(doc)


def test_disabled_mechanical_parity():
    a = OrbitalAttitudeDynamics(398600.4418, np.eye(3), propagate_attitude=False)
    b = OrbitalAttitudeDynamics(
        398600.4418,
        np.eye(3),
        propagate_attitude=False,
        resource_model=SpacecraftResources.from_specs({"thermal": {"enabled": False}}),
    )
    first = a.step(truth(), Command.zero(), ENV, 1.0)
    second = b.step(truth(), Command.zero(), ENV, 1.0)
    assert np.array_equal(first.position_eci_km, second.position_eci_km)
    assert first.resource_state is second.resource_state is None


def test_radiative_cooling_converges_to_analytic_solution():
    cfg = specs()
    cfg.pop("power")
    cfg["thermal"].update(solar_absorptivity=0.0, earth_ir_w_m2=0.0, internal_heat_w=0.0, heat_capacity_j_k=1000.0)
    model = SpacecraftResources.from_specs(cfg)
    duration = 60.0
    exact = (290.0**-3 + 3 * 0.8 * SIGMA * 4.0 / 1000.0 * duration) ** (-1.0 / 3.0)
    errors = []
    for h in (2.0, 1.0, 0.5):
        state = truth()
        for _ in range(int(duration / h)):
            state = advance(model, state, h)
        errors.append(abs(state.resource_state["temperature_k"] - exact))
        row = state.resource_state
        assert 1000.0 * (row["temperature_k"] - 290.0) == pytest.approx(-row["radiated_heat_energy_j"], abs=1e-7)
    assert errors[2] < errors[1] < errors[0]
    assert errors[1] / errors[0] < 0.6


def test_accumulated_bus_and_storage_energy():
    model = SpacecraftResources.from_specs(specs())
    state = truth()
    for x in (7000.0, -7000.0, -7000.0, 7000.0):
        state.position_eci_km[0] = x
        state = advance(model, state, 100.0)
    row = state.resource_state
    assert row["solar_generation_energy_j"] + row["battery_discharge_energy_j"] == pytest.approx(
        row["load_served_energy_j"] + row["battery_charge_energy_j"] + row["curtailed_energy_j"], abs=1e-8
    )
    assert (row["battery_energy_wh"] - 50.0) * 3600.0 == pytest.approx(
        row["battery_charge_energy_j"] - row["battery_discharge_energy_j"] - row["battery_loss_energy_j"], abs=1e-8
    )


def test_output_cadence_preserves_resource_state(tmp_path):
    from sim.api import SimulationConfig, SimulationSession

    results = []
    for cadence in (1.0, 5.0):
        doc = scenario(tmp_path / str(cadence), duration=10.0)
        doc["simulator"]["dt_s"] = cadence
        doc["simulator"]["dynamics"]["orbit"]["orbit_substep_s"] = cadence
        result = SimulationSession(SimulationConfig.from_dict(doc)).run()
        results.append(result.payload["spacecraft_resources"]["spacecraft"][-1])
    for key in ("temperature_k", "battery_energy_wh", "solar_generation_energy_j", "radiated_heat_energy_j"):
        assert results[0][key] == pytest.approx(results[1][key], rel=1e-12)


def test_dynamic_history_retention_keeps_resource_samples_bounded(tmp_path):
    from sim.single_run import _SingleRunEngine

    cfg = scenario_config_from_dict(scenario(tmp_path, duration=10.0))
    engine = _SingleRunEngine(cfg, history_mode="dynamic", max_history_samples=4)
    for _ in range(8):
        engine.step()
    assert len(engine.resource_hist["spacecraft"]) <= 4
    snapshot = engine.snapshot()
    assert snapshot["spacecraft_resources"]["spacecraft"]["time_s"] == snapshot["time_s"]


def test_geometry_configuration_requires_profile(tmp_path):
    doc = scenario(tmp_path)
    thermal = doc["objects"]["spacecraft"]["specs"]["thermal"]
    thermal.pop("projected_area_m2")
    thermal["area_mode"] = "geometry"
    with pytest.raises(ValueError, match="requires a geometry profile"):
        scenario_config_from_dict(doc)


def test_resource_continuation_cannot_silently_reset():
    from sim.interchange.materialization import _continued_object_specs

    with pytest.raises(ValueError, match="resource-state handoff"):
        _continued_object_specs({"object_specs": specs(), "resource_state": {"mass_kg": 100.0}})


def test_resource_state_copy_is_independent():
    state = advance(SpacecraftResources.from_specs(specs()))
    copied = state.copy()
    copied.resource_state["temperature_k"] = 1.0
    assert state.resource_state["temperature_k"] != 1.0
