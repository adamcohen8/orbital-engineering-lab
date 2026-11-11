from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_scenario_parsing_does_not_import_ground_runtime() -> None:
    code = """
import sys
from sim.config import scenario_config_from_dict
from sim.ground_segment.config import normalize_config

assert normalize_config(None) == {}
scenario_config_from_dict({
    "schema_version": "oel.scenario.v1",
    "scenario_name": "cold_config",
    "objects": {"sat": {
        "kind": "satellite",
        "runtime_profile": "trajectory_only",
        "specs": {"mass_kg": 100.0},
        "initial_state": {
            "position_eci_km": [7000.0, 0.0, 0.0],
            "velocity_eci_km_s": [0.0, 7.5, 0.0],
        },
    }},
    "simulator": {
        "duration_s": 1.0,
        "dt_s": 1.0,
        "dynamics": {"attitude": {"enabled": False}},
    },
})
blocked = [
    "sim.ground_segment.runtime", "sim.ground_segment.estimation",
    "sim.estimation.ground_station_od", "scipy", "sim.review",
]
assert not any(name in sys.modules for name in blocked), [
    name for name in blocked if name in sys.modules
]
"""
    subprocess.run(
        [sys.executable, "-B", "-c", code], text=True, capture_output=True, check=True, timeout=30,
    )


def test_ground_segment_lazy_exports_preserve_identity() -> None:
    code = """
import sim.ground_segment as ground
assert set(ground.__all__) <= set(dir(ground))
try:
    ground.not_a_ground_export
except AttributeError:
    pass
else:
    raise AssertionError("unknown export accepted")
from sim.ground_segment import GroundSegment, SimulatedGroundSegment
from sim.ground_segment.runtime import GroundSegment as RuntimeGround
from sim.ground_segment.runtime import SimulatedGroundSegment as RuntimeSimulated
assert GroundSegment is RuntimeGround is ground.GroundSegment
assert SimulatedGroundSegment is RuntimeSimulated is ground.SimulatedGroundSegment
assert GroundSegment.__module__ == "sim.ground_segment.runtime"
"""
    subprocess.run(
        [sys.executable, "-B", "-c", code], text=True, capture_output=True, check=True, timeout=30,
    )


def test_import_sim_keeps_heavy_feature_modules_lazy() -> None:
    code = """
import json
import sys
import sim

blocked = [
    "sim.control",
    "sim.dynamics",
    "sim.estimation",
    "sim.master_simulator",
    "sim.mission",
    "sim.optimization",
    "sim.sensors",
    "sim.utils.plotting",
    "sim.utils.plotting_capabilities",
    "matplotlib",
    "pygame",
]
print(json.dumps({name: name in sys.modules for name in blocked}, sort_keys=True))
"""
    proc = subprocess.run([sys.executable, "-c", code], text=True, capture_output=True, check=True)
    imported = json.loads(proc.stdout)

    assert imported == {name: False for name in imported}


def test_lazy_top_level_exports_preserve_from_sim_import_style() -> None:
    code = """
from sim import HCWLQRController, SimulationConfig, StateTruth

assert SimulationConfig.__name__ == "SimulationConfig"
assert HCWLQRController.__name__ == "HCWLQRController"
assert StateTruth.__name__ == "StateTruth"
"""
    subprocess.run([sys.executable, "-c", code], text=True, capture_output=True, check=True)


def test_single_run_import_keeps_unrelated_od_and_controller_families_lazy() -> None:
    code = """
import json
import sys
import sim.single_run

blocked = [
    "sim.control.orbit.advanced",
    "sim.control.orbit.hcw_mpc",
    "sim.control.attitude.cmg_steering",
    "sim.estimation.ogp_od",
    "sim.estimation.sgp4_od",
    "sim.estimation.slr_od",
    "sim.dynamics.orbit.sgp4",
    "sim.dynamics.orbit.sdp4",
    "sim.acceleration.kernels.orbit",
    "sim.acceleration.kernels.attitude",
    "sim.acceleration.kernels.estimation",
    "numba",
    "scipy",
    "scipy.stats",
]
print(json.dumps({name: name in sys.modules for name in blocked}, sort_keys=True))
"""
    proc = subprocess.run([sys.executable, "-c", code], text=True, capture_output=True, check=True)
    imported = json.loads(proc.stdout)

    assert imported == {name: False for name in imported}


def test_public_workflow_imports_resolve_within_active_checkout() -> None:
    code = r"""
import importlib
import json
import sys
from pathlib import Path

root = Path.cwd().resolve()
for name in (
    "sim.ccsds",
    "sim.collection",
    "sim.conjunction",
    "sim.frame_time",
    "sim.mission_scheduling",
    "sim.orbit_lifetime",
    "sim.spacecraft_power",
    "sim.study",
    "sim.tracking_od",
    "sim.trajectory_design",
):
    importlib.import_module(name)

leaks = {}
for name, module in sorted(sys.modules.items()):
    if name != "sim" and not name.startswith("sim."):
        continue
    value = getattr(module, "__file__", None)
    if not value:
        continue
    path = Path(value).resolve()
    if path != root and root not in path.parents:
        leaks[name] = str(path)
print(json.dumps(leaks, sort_keys=True))
"""
    proc = subprocess.run(
        [sys.executable, "-c", code],
        text=True,
        capture_output=True,
        check=True,
        cwd=Path.cwd(),
    )

    assert json.loads(proc.stdout) == {}
