"""Strict, versioned requests for bounded, non-executing data acquisition."""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

VERSION = "oel.data-acquisition.v1"
RECEIPT = "oel.acquired-data.v1"
INSPECTOR_VERSION = 1
MAX_BYTES = 512 * 1024 * 1024
PRODUCTS = {
    "spacetrack": ("gp", "gp_history", "satcat"),
    "gravity": ("egm96", "egm2008"),
    "celestrak": ("eop", "space_weather"),
}


class AcquisitionError(ValueError):
    def __init__(self, code: str, message: str, *, retry_after_s: float | None = None):
        super().__init__(message)
        self.code = code
        self.retry_after_s = retry_after_s

    def result(self):
        return {"schema": VERSION, "ok": False, "error": self.code,
                "message": str(self), "retry_after_s": self.retry_after_s}


def canonical(value) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def stamp(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def epoch(value: str) -> datetime:
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            raise ValueError("timezone missing")
        return dt.astimezone(timezone.utc)
    except (ValueError, AttributeError, TypeError) as exc:
        raise AcquisitionError("INVALID_REQUEST", "Use an absolute ISO-8601 timestamp with timezone.") from exc


def alias(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}", value):
        raise AcquisitionError("INVALID_REQUEST", "Connection names require 1..64 letters, numbers, '_' or '-'.")
    return value


def strict_json(raw: bytes | str):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate key")
            result[key] = value
        return result

    def invalid(_):
        raise ValueError("nonfinite JSON")

    try:
        return json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise AcquisitionError("INVALID_DATA", "Expected strict UTF-8 JSON without duplicate keys or nonfinite numbers.") from exc


@dataclass(frozen=True)
class Request:
    provider: str
    product: str
    connection: str | None = None
    object_ids: tuple[int, ...] = ()
    start: str | None = None
    stop: str | None = None
    published_before: str | None = None
    max_bytes: int = 64 * 1024 * 1024
    max_records: int = 10_000
    max_requests: int = 8
    schema: str = VERSION

    @classmethod
    def from_dict(cls, value: dict):
        if not isinstance(value, dict) or set(value) - set(cls.__dataclass_fields__):
            raise AcquisitionError("INVALID_REQUEST", "Request must be an object with documented fields only.")
        try:
            result = cls(**value)
        except TypeError as exc:
            raise AcquisitionError("INVALID_REQUEST", "Request requires provider and product.") from exc
        return result.validated()

    def validated(self):
        if not isinstance(self.provider, str) or not isinstance(self.product, str) or self.schema != VERSION or self.provider not in PRODUCTS or self.product not in PRODUCTS[self.provider]:
            raise AcquisitionError("INVALID_REQUEST", "Unknown request schema, provider, or product.")
        for value, cap, label in ((self.max_bytes, MAX_BYTES, "max_bytes"), (self.max_records, 50_000, "max_records"), (self.max_requests, 16, "max_requests")):
            if type(value) is not int or not 1 <= value <= cap:
                raise AcquisitionError("INVALID_REQUEST", f"{label} must be an integer in 1..{cap}.")
        if not isinstance(self.object_ids, (list, tuple)) or any(type(i) is not int or not 1 <= i <= 999_999_999 for i in self.object_ids):
            raise AcquisitionError("INVALID_REQUEST", "object_ids must be positive catalog integers of at most nine digits.")
        ids = tuple(sorted(set(self.object_ids)))
        if len(self.object_ids) > 32:
            raise AcquisitionError("BOUNDS_EXCEEDED", "Supply at most 32 catalog identifiers.")
        if self.provider == "spacetrack":
            alias(self.connection)
            if not 1 <= len(ids) <= 32:
                raise AcquisitionError("BOUNDS_EXCEEDED", "Select 1..32 objects; use the provider bulk-data route for larger requests.")
        elif self.connection is not None or ids:
            raise AcquisitionError("INVALID_REQUEST", "This public provider does not accept a connection or object_ids.")
        if bool(self.start) != bool(self.stop):
            raise AcquisitionError("INVALID_REQUEST", "Specify both start and stop or neither.")
        start = stamp(epoch(self.start)) if self.start is not None else None
        stop = stamp(epoch(self.stop)) if self.stop is not None else None
        if start and epoch(start) >= epoch(stop):
            raise AcquisitionError("INVALID_REQUEST", "Intervals are [start, stop) with start before stop.")
        if self.product == "gp_history":
            if start is None or (epoch(stop) - epoch(start)).total_seconds() > 180 * 86400:
                raise AcquisitionError("BOUNDS_EXCEEDED", "GP history requires an interval of at most 180 days; use the provider archive for larger histories.")
        elif self.provider == "spacetrack" and start:
            raise AcquisitionError("INVALID_REQUEST", "Only gp_history accepts an epoch interval on Space-Track.")
        elif self.provider == "gravity" and start:
            raise AcquisitionError("INVALID_REQUEST", "Static gravity requests select a model, not a time interval.")
        cutoff = stamp(epoch(self.published_before)) if self.published_before is not None else None
        if cutoff and self.product != "gp_history":
            raise AcquisitionError("INVALID_REQUEST", "published_before is supported only for GP history.")
        value = asdict(self)
        value.update(object_ids=ids, start=start, stop=stop, published_before=cutoff)
        return Request(**value)

    def to_dict(self):
        result = asdict(self)
        result["object_ids"] = list(self.object_ids)
        return result

    def key(self, scope: str):
        # Resource bounds do not change dataset identity. They remain enforced on cache hits.
        value = self.to_dict()
        for field in ("max_bytes", "max_records", "max_requests", "connection"):
            value.pop(field)
        return digest(canonical({"scope": scope, "request": value}))
