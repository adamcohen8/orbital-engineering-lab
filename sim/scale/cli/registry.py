from __future__ import annotations

from collections.abc import Callable
from typing import Any

from sim.scale.cli.handlers.campaigns import COMMAND_HANDLERS as CAMPAIGN_HANDLERS
from sim.scale.cli.handlers.catalog import COMMAND_HANDLERS as CATALOG_HANDLERS
from sim.scale.cli.handlers.observations import COMMAND_HANDLERS as OBSERVATION_HANDLERS
from sim.scale.cli.handlers.operations import COMMAND_HANDLERS as OPERATION_HANDLERS

COMMAND_HANDLERS: dict[str, Callable[[Any], int]] = {}
for family in (CATALOG_HANDLERS, OBSERVATION_HANDLERS, OPERATION_HANDLERS, CAMPAIGN_HANDLERS):
    duplicate = COMMAND_HANDLERS.keys() & family.keys()
    if duplicate:
        raise RuntimeError(f"Duplicate Scale CLI command registration: {sorted(duplicate)}")
    COMMAND_HANDLERS.update(family)
