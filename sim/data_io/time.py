"""Interchange epoch helpers; reuse the canonical OEL time owner."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone

from sim.frame_time import Epoch, TimeScale, format_epoch, parse_epoch

from .contracts import DataIOError


def parse_time(value: str, scale: str = "UTC") -> Epoch:
    scale = scale.upper()
    if scale == "GPS":
        # GPS = TAI - 19 SI seconds, with no leap seconds in GPS.
        result = parse_epoch(value, TimeScale.TAI)
        return replace(result, tai_seconds=result.tai_seconds + 19.0)
    try:
        return parse_epoch(value, scale)
    except (ValueError, TypeError) as exc:
        raise DataIOError(f"Unsupported or invalid {scale} epoch: {value}.") from exc


def at(epoch: Epoch, seconds: float) -> Epoch:
    return replace(epoch, tai_seconds=epoch.tai_seconds + float(seconds))


def format_time(epoch: Epoch, scale: str = "UTC") -> str:
    if scale.upper() == "GPS":
        return format_epoch(at(epoch, -19.0), TimeScale.TAI)
    return format_epoch(epoch, scale)


def creation_time() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")
