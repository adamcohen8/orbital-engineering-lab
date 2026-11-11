"""Versioned operations contracts. No simulator or provider dependencies."""

from __future__ import annotations

import json
import math
import struct
from dataclasses import asdict, dataclass
from hashlib import sha256

SCHEMA = "oel.agent_operations.v1"
WIRE = struct.Struct(">II4d")  # request, ticket, ECI target xyz (m), hold (s)
NAV_FIELDS = (
    tuple(f"position_{a}_m" for a in "xyz")
    + tuple(f"velocity_{a}_m_s" for a in "xyz")
    + tuple(f"q_{a}" for a in "wxyz")
    + tuple(f"omega_{a}_rad_s" for a in "xyz")
)
NUMERIC_FIELDS = (
    ("sim_time_s", "battery_soc", "attitude_error_rad", "target_update_count", "wheel_momentum_fraction")
    + NAV_FIELDS
    + tuple(f"wheel_{a}_momentum" for a in "xyz")
)
STRING_FIELDS = (
    "spacecraft_id",
    "package_digest",
    "mode",
    "command_status",
    "last_request_id",
    "sample_id",
    "epoch_utc",
)
PARAMETERS = NUMERIC_FIELDS + STRING_FIELDS


def digest(value: object) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class PointingAction:
    request_id: int
    target_eci_m: tuple[float, float, float]
    hold_s: float

    def __post_init__(self):
        if type(self.request_id) is not int or not 0 < self.request_id < 2**32:
            raise ValueError("request_id must be a positive uint32")
        if len(self.target_eci_m) != 3 or not all(
            type(v) in (int, float) and math.isfinite(v) for v in (*self.target_eci_m, self.hold_s)
        ):
            raise ValueError("target and hold must be finite numbers")
        if self.hold_s <= 0:
            raise ValueError("hold_s must be positive")
        object.__setattr__(self, "target_eci_m", tuple(float(v) for v in self.target_eci_m))
        object.__setattr__(self, "hold_s", float(self.hold_s))

    @classmethod
    def from_dict(cls, value: dict):
        if set(value) != {"request_id", "target_eci_m", "hold_s"}:
            raise ValueError("unexpected or missing action fields")
        return cls(value["request_id"], tuple(value["target_eci_m"]), value["hold_s"])

    def to_dict(self):
        return asdict(self)

    def encode(self, ticket: int) -> bytes:
        return WIRE.pack(self.request_id, ticket, *self.target_eci_m, self.hold_s)

    @classmethod
    def decode(cls, data: bytes):
        request, ticket, x, y, z, hold = WIRE.unpack(data)
        return cls(request, (x, y, z), hold), ticket


@dataclass(frozen=True)
class Policy:
    battery_reserve: float = 0.30
    battery_margin: float = 0.02
    wheel_fraction_limit: float = 0.80
    max_rate_rad_s: float = 0.12
    max_hold_s: float = 120.0
    max_telemetry_age_s: float = 90.0
    ticket_ttl_s: float = 45.0
    recovery_s: float = 20.0

    def __post_init__(self):
        if not all(math.isfinite(v) and v > 0 for v in asdict(self).values()):
            raise ValueError("policy limits must be positive and finite")
        if self.battery_reserve + self.battery_margin >= 1 or self.wheel_fraction_limit > 1:
            raise ValueError("invalid resource limits")

    @property
    def identity(self):
        return digest(asdict(self))
