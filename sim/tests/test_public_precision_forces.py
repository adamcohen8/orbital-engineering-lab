"""Public precision force construction and synthetic-data smoke contracts."""

import builtins
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from sim.config import scenario_config_from_dict
from sim.dynamics.orbit.accelerations import OrbitContext
from sim.dynamics.orbit.frames import FrameContext
from sim.pro_perturbations.earth_radiation import EarthRadiationPressure
from sim.pro_perturbations.ocean_tides import OceanTides
from sim.pro_perturbations.schwarzschild import SchwarzschildAcceleration
from sim.pro_perturbations.solid_earth_tides import SolidEarthTides
from sim.runtime.satellite_factory import _build_orbit_propagator

ROOT = Path(__file__).resolve().parents[2]


def test_public_forces_do_not_request_entitlement(monkeypatch):
    def deny(*args, **kwargs):
        pytest.fail("public force requested a Pro entitlement")

    # The private runtime loads licensing; the generated public tree excludes it.
    # Exercise construction and evaluation in both, instrumenting it when present.
    licensing = sys.modules.get("sim.licensing.features")
    if licensing is not None:
        monkeypatch.setattr(licensing, "require_pro_feature", deny)
    frames = FrameContext(
        model="iau76_80_eop",
        jd_utc_start=2459669.5,
        dut1_s=-0.0992395,
        dat_s=37.0,
        xp_arcsec=0.043546,
        yp_arcsec=0.424847,
    )
    forces = [
        EarthRadiationPressure(frames),
        SchwarzschildAcceleration(),
        SolidEarthTides(frames, "tide_free"),
        OceanTides(frames, str(ROOT / "sim/dynamics/orbit/data/ocean_tide_synthetic.txt"), degree=2, order=2),
    ]
    state = np.array([7000.0, 0.0, 0.0, 0.0, 7.5, 0.0])
    env = {"ephemeris_mode": "analytic_enhanced", "jd_utc_start": 2459669.5}
    ctx = OrbitContext(398600.4418, 300.0, area_m2=4.0, cr=1.2)
    for force in forces:
        acceleration = force(0.0, state, env, ctx)
        assert acceleration.shape == (3,)
        assert np.all(np.isfinite(acceleration))
        assert np.linalg.norm(acceleration) > 0


def test_public_demo_builds_all_four_forces_and_preserves_disabled_path():
    raw = yaml.safe_load((ROOT / "configs/precision_forces_demo.yaml").read_text())
    raw["simulator"]["dynamics"]["orbit"]["ocean_tides"]["coeff_path"] = str(
        ROOT / "sim/dynamics/orbit/data/ocean_tide_synthetic.txt"
    )
    cfg = scenario_config_from_dict(raw)
    prop = _build_orbit_propagator(cfg)
    assert {type(f).__name__ for f in prop.plugins} == {
        "EarthRadiationPressure",
        "SchwarzschildAcceleration",
        "SolidEarthTides",
        "OceanTides",
    }
    orbit = raw["simulator"]["dynamics"]["orbit"]
    for key in ("earth_radiation", "schwarzschild", "solid_earth_tides", "ocean_tides"):
        orbit[key]["enabled"] = False
    disabled = _build_orbit_propagator(scenario_config_from_dict(raw))
    assert disabled.plugins == []


def _load_precision_adapter():
    path = ROOT / "sim/dynamics/orbit/rust_precision_force_plan.py"
    spec = importlib.util.spec_from_file_location("_precision_adapter_boundary_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_precision_adapter_preserves_public_forces_when_private_ecom_is_absent(monkeypatch):
    original_import = builtins.__import__

    def without_private_ecom(name, *args, **kwargs):
        if name == "sim.pro_perturbations.ecom":
            raise ModuleNotFoundError("private ECOM is absent", name=name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_private_ecom)
    adapter = _load_precision_adapter()
    force = SchwarzschildAcceleration()
    assert adapter.precision_code(force) == 10
    assert adapter.specifications([force], OrbitContext(398600.4418, 300.0)) == [(10, [], [])]


@pytest.mark.parametrize("failure", [
    ModuleNotFoundError("missing internal ECOM dependency", name="missing_ecom_dependency"),
    ImportError("broken internal ECOM import"),
])
def test_precision_adapter_does_not_hide_private_dependency_failures(monkeypatch, failure):
    original_import = builtins.__import__

    def broken_private_ecom(name, *args, **kwargs):
        if name == "sim.pro_perturbations.ecom":
            raise failure
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", broken_private_ecom)
    with pytest.raises(type(failure), match=str(failure)) as caught:
        _load_precision_adapter()
    assert caught.value is failure
