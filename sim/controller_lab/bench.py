# ruff: noqa: F401,I001
"""Compatibility façade for the controller benchmark laboratory."""

from . import bench_common as _common
from . import config as _config
from . import variants as _variants
from . import cases as _cases
from . import criteria as _criteria
from . import diagnostics as _diagnostics
from . import objectives as _objectives
from . import optimization as _optimization
from . import engineering as _engineering
from . import runner as _runner

CONTROLLER_BENCH_CAPABILITY_FAMILIES = {
    "config": "sim.controller_lab.config",
    "variants": "sim.controller_lab.variants",
    "cases": "sim.controller_lab.cases",
    "criteria": "sim.controller_lab.criteria",
    "diagnostics": "sim.controller_lab.diagnostics",
    "objectives": "sim.controller_lab.objectives",
    "optimization": "sim.controller_lab.optimization",
    "engineering": "sim.controller_lab.engineering",
    "runner": "sim.controller_lab.runner",
}

for _module in (_common, _config, _variants, _cases, _criteria, _diagnostics, _objectives, _optimization, _engineering, _runner):
    globals().update({name: value for name, value in vars(_module).items() if not name.startswith("__")})

del _module
