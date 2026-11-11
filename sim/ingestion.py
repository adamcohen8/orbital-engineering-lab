# ruff: noqa: F401,F821,I001
"""Compatibility façade for mission-input ingestion and scenario building."""

from . import ingestion_contracts as _contracts
from . import ingestion_normalization as _normalization
from . import ingestion_sources as _sources
from . import scenario_builders as _builders
from . import ingestion_cli as _cli

INGESTION_CAPABILITY_FAMILIES = {
    "contracts": "sim.ingestion_contracts",
    "normalization": "sim.ingestion_normalization",
    "sources": "sim.ingestion_sources",
    "scenario_builders": "sim.scenario_builders",
    "cli": "sim.ingestion_cli",
}

for _module in (_contracts, _normalization, _sources, _builders, _cli):
    globals().update({name: value for name, value in vars(_module).items() if not name.startswith("__")})

MissionInputPacket.__module__ = __name__
del _module

if __name__ == "__main__":
    raise SystemExit(main())
