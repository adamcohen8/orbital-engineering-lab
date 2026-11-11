"""Reporting helpers for simulation campaigns."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = ["SUPPORTED_THRESHOLDS", "build_maneuver_readiness_packet"]


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(name)
    value = getattr(import_module("sim.reporting.maneuver_readiness"), name)
    globals()[name] = value
    return value
