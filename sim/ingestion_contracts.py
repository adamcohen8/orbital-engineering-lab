# ruff: noqa: E402,F401,F403,F405,I001
from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sim.dynamics.orbit.tle import parse_tle_lines
from sim.scenarios import ScenarioArtifact

PACKET_VERSION = 1

@dataclass(frozen=True)
class MissionInputPacket:
    """Normalized mission input packet for agent-mediated scenario generation."""

    data: Mapping[str, Any]

    @property
    def objects(self) -> dict[str, Any]:
        value = self.data.get("objects", {})
        return dict(value if isinstance(value, Mapping) else {})

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


def packet_from_dict(data: Mapping[str, Any]) -> MissionInputPacket:
    packet = dict(data)
    if int(packet.get("packet_version", 0) or 0) != PACKET_VERSION:
        raise ValueError(f"unsupported mission input packet version: {packet.get('packet_version')!r}")
    if str(packet.get("kind", "") or "") != "oel.mission_input_packet":
        raise ValueError("mission input packet kind must be 'oel.mission_input_packet'.")
    objects = packet.get("objects", {})
    if not isinstance(objects, Mapping):
        raise ValueError("mission input packet objects must be a mapping.")
    return MissionInputPacket(packet)


def load_packet(path: str | Path) -> MissionInputPacket:
    return packet_from_dict(json.loads(Path(path).expanduser().read_text(encoding="utf-8")))

__all__ = [name for name in globals() if not name.startswith("__")]
