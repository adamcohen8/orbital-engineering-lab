"""Temperature observations and owner-configured procedures, independent of physics."""

from __future__ import annotations

import math
import struct
from dataclasses import asdict, dataclass
from datetime import datetime

from .contracts import digest

PROCEDURE_WIRE = struct.Struct(">4I")  # request, approval, component, procedure
THERMAL_NUMERIC = ("component_temperature_c", "temperature_source_time_s", "temperature_sequence")
THERMAL_TEXT = (
    "temperature_component",
    "temperature_sensor",
    "temperature_source",
    "temperature_clock",
    "temperature_measured_at",
    "temperature_quality",
    "payload_mode",
    "temperature_package_digest",
    "temperature_packet_id",
)
THERMAL_PARAMETERS = THERMAL_NUMERIC + THERMAL_TEXT


def utc_seconds(value: str) -> float:
    stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if stamp.tzinfo is None:
        raise ValueError("measurement timestamp requires a UTC offset")
    result = stamp.timestamp()
    if not math.isfinite(result):
        raise ValueError("invalid measurement timestamp")
    return result


@dataclass(frozen=True)
class TemperatureReading:
    component: str
    sensor: str
    source: str
    clock: str
    source_time_s: float
    sequence: int
    measured_at: str
    temperature_c: float
    quality: str = "valid"

    def __post_init__(self):
        if any(type(x) is not str or not x.strip() for x in (self.component, self.sensor, self.source, self.clock)):
            raise ValueError("temperature identity fields must be nonempty strings")
        if type(self.sequence) is not int or self.sequence <= 0:
            raise ValueError("temperature sequence must be positive integer")
        if any(type(x) not in (int, float) or not math.isfinite(x) for x in (self.source_time_s, self.temperature_c)):
            raise ValueError("temperature and source time must be finite")
        if self.source_time_s < 0 or self.temperature_c < -273.15 or self.quality not in ("valid", "invalid"):
            raise ValueError("invalid temperature reading")
        utc_seconds(self.measured_at)


@dataclass(frozen=True)
class ThermalProcedurePolicy:
    component: str = "payload"
    component_code: int = 1
    sensor: str = "payload-thermistor-1"
    source: str = "synthetic-playback-v1"
    approval_reference: str = "local synthetic exercise procedure book; not flight qualified"
    suspend_at_c: float = 75.0
    resume_below_c: float = 65.0
    minimum_samples: int = 3
    minimum_span_s: float = 10.0
    recovery_span_s: float = 20.0
    maximum_gap_s: float = 15.0
    verification_timeout_s: float = 90.0
    max_telemetry_age_s: float = 90.0
    ticket_ttl_s: float = 45.0

    def __post_init__(self):
        if type(self.component_code) is not int or not 0 < self.component_code < 2**32:
            raise ValueError("component code must be positive uint32")
        if type(self.minimum_samples) is not int or self.minimum_samples < 3:
            raise ValueError("trend requires at least three samples")
        for key, value in asdict(self).items():
            if key in ("component_code", "minimum_samples"):
                continue
            if key in ("component", "sensor", "source", "approval_reference"):
                if type(value) is not str or not value.strip():
                    raise ValueError(f"{key} must be nonempty")
            elif type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"{key} must be finite")
        if not -273.15 <= self.resume_below_c < self.suspend_at_c:
            raise ValueError("resume threshold must be below suspend threshold")
        if (
            min(
                self.minimum_span_s,
                self.recovery_span_s,
                self.maximum_gap_s,
                self.verification_timeout_s,
                self.max_telemetry_age_s,
                self.ticket_ttl_s,
            )
            <= 0
        ):
            raise ValueError("procedure durations must be positive")
        if self.verification_timeout_s < self.recovery_span_s:
            raise ValueError("verification timeout is shorter than recovery window")

    @property
    def identity(self):
        return digest(asdict(self))


@dataclass(frozen=True)
class ProcedureAction:
    request_id: int
    component_code: int
    procedure: str

    def __post_init__(self):
        if any(type(x) is not int or not 0 < x < 2**32 for x in (self.request_id, self.component_code)):
            raise ValueError("request and component codes must be positive uint32")
        if self.procedure not in ("suspend_payload", "resume_payload"):
            raise ValueError("procedure is outside the approved command vocabulary")

    @property
    def operation_code(self):
        return 1 if self.procedure == "suspend_payload" else 2

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        if set(value) != {"request_id", "component_code", "procedure"}:
            raise ValueError("unexpected or missing procedure fields")
        return cls(**value)

    def encode(self, ticket):
        return PROCEDURE_WIRE.pack(self.request_id, ticket, self.component_code, self.operation_code)

    @classmethod
    def decode(cls, data):
        request, ticket, component, operation = PROCEDURE_WIRE.unpack(data)
        if operation not in (1, 2):
            raise ValueError("unknown procedure operation")
        return cls(request, component, "suspend_payload" if operation == 1 else "resume_payload"), ticket
