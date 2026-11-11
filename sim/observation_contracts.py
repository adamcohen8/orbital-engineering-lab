# ruff: noqa: E402,F401,F403,F405,I001
from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from sim.ingestion import MissionInputPacket, ingest_state_vector

OBSERVATION_PACKET_VERSION = 1
OBSERVATION_SCHEMA_VERSION = 1
_JD_TO_S = 86400.0
_PARTITIONS = {"fit", "holdout", "excluded", "unassigned"}

@dataclass(frozen=True)
class ObservationPacket:
    """Normalized observation packet for preliminary orbit-determination tools."""

    data: Mapping[str, Any]

    @property
    def observations(self) -> list[dict[str, Any]]:
        return [dict(item) for item in list(self.data.get("observations", []) or [])]

    @property
    def warnings(self) -> list[str]:
        return [str(item) for item in list(self.data.get("warnings", []) or [])]

    def to_dict(self) -> dict[str, Any]:
        return deepcopy(dict(self.data))

    def write_json(self, path: str | Path) -> Path:
        target = Path(path).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
        )
        return target

__all__ = [name for name in globals() if not name.startswith("__")]
