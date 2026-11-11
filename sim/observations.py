# ruff: noqa: F401,F821,I001
"""Compatibility façade for observation packets, fitting, and TLE sampling."""

from . import observation_contracts as _contracts
from . import observation_normalization as _normalization
from . import observation_ingestion as _ingestion
from . import observation_fitting as _fitting
from . import observation_tle as _tle
from . import observation_cli as _cli

OBSERVATION_CAPABILITY_FAMILIES = {
    "contracts": "sim.observation_contracts",
    "normalization": "sim.observation_normalization",
    "ingestion": "sim.observation_ingestion",
    "fitting": "sim.observation_fitting",
    "tle_sampling": "sim.observation_tle",
    "cli": "sim.observation_cli",
}

for _module in (_contracts, _normalization, _ingestion, _fitting, _tle, _cli):
    globals().update({name: value for name, value in vars(_module).items() if not name.startswith("__")})

ObservationPacket.__module__ = __name__
del _module

if __name__ == "__main__":
    raise SystemExit(main())
