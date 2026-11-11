"""Reusable ground knowledge, independent of games and dashboards."""

from importlib import import_module
from typing import Any

__all__ = ["GroundSegment", "SimulatedGroundSegment"]


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(".runtime", __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
