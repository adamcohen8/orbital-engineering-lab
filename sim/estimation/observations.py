from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ObservationBatch:
    """Time-tagged vector observations used by batch OD workflows."""

    time_s: np.ndarray
    values: np.ndarray
    sigma: np.ndarray
    kind: str = "position_eci"
    frame: str = "eci"

    def __post_init__(self) -> None:
        t = np.asarray(self.time_s, dtype=float).reshape(-1)
        values = np.asarray(self.values, dtype=float)
        sigma = np.asarray(self.sigma, dtype=float)
        if t.size < 2:
            raise ValueError("ObservationBatch requires at least two samples.")
        if values.ndim != 2 or values.shape[0] != t.size:
            raise ValueError("ObservationBatch values must have shape (N, M).")
        if sigma.ndim == 0:
            sigma = np.full(values.shape, float(sigma), dtype=float)
        elif sigma.ndim == 1:
            if sigma.size != values.shape[1]:
                raise ValueError("ObservationBatch 1D sigma must have one value per observed component.")
            sigma = np.broadcast_to(sigma.reshape(1, -1), values.shape).copy()
        elif sigma.shape != values.shape:
            raise ValueError("ObservationBatch sigma must be scalar, component vector, or shape (N, M).")
        if np.any(~np.isfinite(t)) or np.any(np.diff(t) <= 0.0):
            raise ValueError("ObservationBatch time_s must be finite and strictly increasing.")
        if np.any(~np.isfinite(values)):
            raise ValueError("ObservationBatch values must be finite.")
        if np.any(~np.isfinite(sigma)) or np.any(sigma <= 0.0):
            raise ValueError("ObservationBatch sigma values must be finite and positive.")
        object.__setattr__(self, "time_s", t)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "sigma", sigma)

    @property
    def weights(self) -> np.ndarray:
        return 1.0 / self.sigma
