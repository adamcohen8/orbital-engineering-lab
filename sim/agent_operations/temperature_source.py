"""Trusted observation/device seams and explicitly synthetic trace playback.

Playback never computes temperature or forecasts cooling. Fixture temperatures
are independent of commands; unsuccessful responses can be tested just as easily.
"""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from .temperature_contracts import TemperatureReading, ThermalProcedurePolicy


class TemperatureSource(Protocol):
    def read(self, mission_time_s: float) -> TemperatureReading | None: ...


class PayloadDevice(Protocol):
    def observed_mode(self) -> str: ...
    def set_mode(self, mode: str, request_id: int) -> dict: ...


class ReceivedTemperatureSource:
    """Ingest an adapter's typed observations without relabeling their timestamps."""

    def __init__(self):
        self.latest = None

    def ingest(self, reading: TemperatureReading):
        if not isinstance(reading, TemperatureReading):
            raise TypeError("adapter must provide a typed temperature reading")
        self.latest = reading

    def read(self, mission_time_s):
        return self.latest


class PlaybackTemperatureSource(ReceivedTemperatureSource):
    def __init__(self, path: Path, policy: ThermalProcedurePolicy):
        super().__init__()
        packet = json.loads(path.read_text())
        if set(packet) != {"schema", "samples"} or packet["schema"] != "oel.synthetic_temperature_playback.v1":
            raise ValueError("not a synthetic temperature playback fixture")
        self.samples, self.policy = packet["samples"], policy
        if not self.samples or len(self.samples) > 1000:
            raise ValueError("fixture requires 1 to 1000 samples")
        previous = -1.0
        for index, row in enumerate(self.samples):
            if set(row) != {"time_s", "temperature_c", "quality"}:
                raise ValueError("invalid playback sample fields")
            TemperatureReading(
                policy.component,
                policy.sensor,
                policy.source,
                "validation",
                row["time_s"],
                index + 1,
                "2026-10-02T00:00:00Z",
                row["temperature_c"],
                row["quality"],
            )
            if row["time_s"] <= previous:
                raise ValueError("playback times must strictly increase")
            previous = row["time_s"]
        if self.samples[0]["time_s"] != 0:
            raise ValueError("playback starts at zero")
        self.clock = "synthetic-mission/" + str(uuid4())
        self.index = -1

    def read(self, mission_time_s):
        index = max(i for i, row in enumerate(self.samples) if row["time_s"] <= mission_time_s)
        if index != self.index:
            row = self.samples[index]
            self.ingest(
                TemperatureReading(
                    self.policy.component,
                    self.policy.sensor,
                    self.policy.source,
                    self.clock,
                    row["time_s"],
                    index + 1,
                    datetime.now(timezone.utc).isoformat(),
                    row["temperature_c"],
                    row["quality"],
                )
            )
            self.index = index
        reading = super().read(mission_time_s)
        if mission_time_s - reading.source_time_s > self.policy.maximum_gap_s:
            return replace(reading, quality="invalid")  # Missing source samples in the accelerated fixture clock.
        return reading


class SyntheticPayloadDevice:
    """Discrete fixture device, not OEL flight software or a thermal/power model."""

    def __init__(self):
        self.mode, self.command_count = "operating", 0

    def observed_mode(self):
        return self.mode

    def set_mode(self, mode, request_id):
        if mode not in ("operating", "standby"):
            raise ValueError("unsupported payload mode")
        self.mode = mode
        self.command_count += 1
        return {"status": "accepted", "request_id": request_id, "adapter": "synthetic-payload-device"}
