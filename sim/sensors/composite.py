from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sim.core.interfaces import SensorModel
from sim.core.models import Measurement, StateTruth


@dataclass
class CompositeSensorModel(SensorModel):
    sensors: list[SensorModel]
    time_tolerance_s: float = 1e-9

    def __post_init__(self) -> None:
        if isinstance(self.time_tolerance_s, (bool, np.bool_)):
            raise ValueError("time_tolerance_s must be finite and nonnegative.")
        tolerance = float(self.time_tolerance_s)
        if not np.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("time_tolerance_s must be finite and nonnegative.")
        self.time_tolerance_s = tolerance

    def measure(self, truth: StateTruth, env: dict, t_s: float) -> Measurement | None:
        parts = []
        sample_times = []
        for sensor in self.sensors:
            m = sensor.measure(truth, env, t_s)
            if m is None:
                return None
            parts.append(m.vector)
            sample_times.append(float(m.t_s))
        if not parts:
            return None
        sample_time = sample_times[0]
        if any(abs(sample_t - sample_time) > self.time_tolerance_s for sample_t in sample_times[1:]):
            return None
        return Measurement(vector=np.concatenate(parts), t_s=sample_time)
