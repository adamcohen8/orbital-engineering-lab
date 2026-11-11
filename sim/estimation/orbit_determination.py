# ruff: noqa: F401,I001
"""Compatibility façade for dynamics orbit determination."""

from . import orbit_od_common as _common
from . import orbit_od_parameters as _parameters
from . import orbit_od_artifacts as _artifacts
from . import orbit_od_quality as _quality
from . import orbit_od_maneuvers as _maneuvers
from . import orbit_od_solver as _solver

ORBIT_OD_CAPABILITY_FAMILIES = {
    "parameters": "sim.estimation.orbit_od_parameters",
    "solving": "sim.estimation.orbit_od_solver",
    "quality": "sim.estimation.orbit_od_quality",
    "maneuvers": "sim.estimation.orbit_od_maneuvers",
    "artifacts": "sim.estimation.orbit_od_artifacts",
}

for _module in (_common, _parameters, _artifacts, _quality, _maneuvers, _solver):
    globals().update({name: value for name, value in vars(_module).items() if not name.startswith("__")})

del _module
