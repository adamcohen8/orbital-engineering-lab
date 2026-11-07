from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class EstimatedParameter:
    """A bounded estimated scalar with an optimizer scaling."""

    name: str
    value: float
    scale: float = 1.0
    lower: float = -np.inf
    upper: float = np.inf
    unit: str = ""
    description: str = ""

    def __post_init__(self) -> None:
        if not str(self.name).strip():
            raise ValueError("EstimatedParameter name must be non-empty.")
        try:
            value = float(self.value)
            scale = float(self.scale)
            lower = float(self.lower)
            upper = float(self.upper)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"Parameter {self.name} value, scale, and bounds must be numeric.") from exc
        if not np.isfinite(value):
            raise ValueError(f"Parameter {self.name} initial value must be finite.")
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError(f"Parameter {self.name} scale must be finite and positive.")
        # Bounds may be unbounded, but NaN would poison the optimizer's
        # scaled bounds while making the ordering comparison silently false.
        if np.isnan(lower) or np.isnan(upper):
            raise ValueError(f"Parameter {self.name} bounds must not be NaN.")
        if lower > upper:
            raise ValueError(f"Parameter {self.name} lower bound exceeds upper bound.")
        if value < lower or value > upper:
            raise ValueError(f"Parameter {self.name} initial value must lie within its bounds.")


class ParameterSet:
    """Ordered parameter vector with native/scaled conversions."""

    def __init__(self, parameters: Iterable[EstimatedParameter]):
        self.parameters = tuple(parameters)
        if not self.parameters:
            raise ValueError("ParameterSet requires at least one parameter.")
        names = [p.name for p in self.parameters]
        if len(names) != len(set(names)):
            raise ValueError("Parameter names must be unique.")

    @property
    def names(self) -> list[str]:
        return [p.name for p in self.parameters]

    @property
    def units(self) -> list[str]:
        return [p.unit for p in self.parameters]

    @property
    def descriptions(self) -> list[str]:
        return [p.description for p in self.parameters]

    def initial_native(self) -> np.ndarray:
        return np.array([p.value for p in self.parameters], dtype=float)

    def scales(self) -> np.ndarray:
        return np.array([p.scale for p in self.parameters], dtype=float)

    def lower_native(self) -> np.ndarray:
        return np.array([p.lower for p in self.parameters], dtype=float)

    def upper_native(self) -> np.ndarray:
        return np.array([p.upper for p in self.parameters], dtype=float)

    def to_scaled(self, native: np.ndarray) -> np.ndarray:
        return np.asarray(native, dtype=float).reshape(-1) / self.scales()

    def to_native(self, scaled: np.ndarray) -> np.ndarray:
        return np.asarray(scaled, dtype=float).reshape(-1) * self.scales()

    def lower_scaled(self) -> np.ndarray:
        return self.to_scaled(self.lower_native())

    def upper_scaled(self) -> np.ndarray:
        return self.to_scaled(self.upper_native())

    def mapping(self, native: np.ndarray) -> dict[str, float]:
        values = np.asarray(native, dtype=float).reshape(-1)
        if values.size != len(self.parameters):
            raise ValueError("Parameter vector length does not match ParameterSet.")
        return {name: float(value) for name, value in zip(self.names, values)}

    def metadata(self, native: np.ndarray | None = None) -> list[dict[str, float | str]]:
        values = self.initial_native() if native is None else np.asarray(native, dtype=float).reshape(-1)
        return [
            {
                "name": p.name,
                "value": float(values[idx]),
                "scale": float(p.scale),
                "lower": float(p.lower),
                "upper": float(p.upper),
                "unit": p.unit,
                "description": p.description,
            }
            for idx, p in enumerate(self.parameters)
        ]
