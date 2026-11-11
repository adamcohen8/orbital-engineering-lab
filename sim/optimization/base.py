from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol

import numpy as np


@dataclass(frozen=True)
class ParameterBound:
    name: str
    lower: float
    upper: float

    def __post_init__(self) -> None:
        if not np.isfinite(float(self.lower)) or not np.isfinite(float(self.upper)):
            raise ValueError(f"Invalid bounds for {self.name}: bounds must be finite.")
        if self.upper <= self.lower:
            raise ValueError(f"Invalid bounds for {self.name}: upper must be greater than lower.")


@dataclass
class OptimizationResult:
    best_x: np.ndarray
    best_cost: float
    history_best_cost: list[float] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)


ObjectiveFn = Callable[[np.ndarray], float]


class VectorOptimizer(Protocol):
    def optimize(self, objective: ObjectiveFn, bounds: list[ParameterBound], seed: int = 0) -> OptimizationResult: ...
