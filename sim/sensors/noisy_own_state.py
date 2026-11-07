from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sim.core.interfaces import SensorModel
from sim.core.models import Measurement, StateTruth
from sim.sensors.access import _cadence_due


@dataclass
class NoisyOwnStateSensor(SensorModel):
    pos_sigma_km: float
    vel_sigma_km_s: float
    rng: np.random.Generator
    update_cadence_s: float = 0.0
    _last_update_t_s: float = -np.inf

    def __post_init__(self) -> None:
        for field_name in ("pos_sigma_km", "vel_sigma_km_s"):
            value = getattr(self, field_name)
            if isinstance(value, (bool, np.bool_)):
                raise ValueError(f"{field_name} must be finite and nonnegative.")
            normalized = float(value)
            if not np.isfinite(normalized) or normalized < 0.0:
                raise ValueError(f"{field_name} must be finite and nonnegative.")
            setattr(self, field_name, normalized)
        if isinstance(self.update_cadence_s, (bool, np.bool_)):
            raise ValueError("update_cadence_s must be finite and nonnegative.")
        cadence = float(self.update_cadence_s)
        if not np.isfinite(cadence) or cadence < 0.0:
            raise ValueError("update_cadence_s must be finite and nonnegative.")
        self.update_cadence_s = cadence

    def measure(self, truth: StateTruth, env: dict, t_s: float) -> Measurement | None:
        if not _cadence_due(t_s, self._last_update_t_s, self.update_cadence_s):
            return None
        self._last_update_t_s = t_s
        pos_noise = self.rng.normal(0.0, self.pos_sigma_km, size=3)
        vel_noise = self.rng.normal(0.0, self.vel_sigma_km_s, size=3)
        z = np.hstack((truth.position_eci_km + pos_noise, truth.velocity_eci_km_s + vel_noise))
        return Measurement(vector=z, t_s=t_s)
