from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from sim.core.interfaces import SensorModel
from sim.core.models import Measurement, StateTruth
from sim.sensors.access import _cadence_due
from sim.utils.quaternion import normalize_quaternion


@dataclass
class JointStateSensor(SensorModel):
    pos_sigma_km: float = 1e-3
    vel_sigma_km_s: float = 1e-5
    quat_sigma: float = 1e-3
    omega_sigma_rad_s: float = 1e-4
    update_cadence_s: float = 1.0
    dropout_prob: float = 0.0
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(0))
    _last_update_t_s: float = -np.inf

    def __post_init__(self) -> None:
        for field_name in ("pos_sigma_km", "vel_sigma_km_s", "quat_sigma", "omega_sigma_rad_s"):
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
        if isinstance(self.dropout_prob, (bool, np.bool_)):
            raise ValueError("dropout_prob must be finite within [0, 1].")
        dropout = float(self.dropout_prob)
        if not np.isfinite(dropout) or not 0.0 <= dropout <= 1.0:
            raise ValueError("dropout_prob must be finite within [0, 1].")
        self.dropout_prob = dropout

    def measure(self, truth: StateTruth, env: dict, t_s: float) -> Measurement | None:
        if not _cadence_due(t_s, self._last_update_t_s, self.update_cadence_s):
            return None
        self._last_update_t_s = t_s
        if self.rng.random() < self.dropout_prob:
            return None

        pos = truth.position_eci_km + self.rng.normal(0.0, self.pos_sigma_km, size=3)
        vel = truth.velocity_eci_km_s + self.rng.normal(0.0, self.vel_sigma_km_s, size=3)

        q = truth.attitude_quat_bn.copy()
        q[1:] += self.rng.normal(0.0, self.quat_sigma, size=3)
        q = normalize_quaternion(q)

        w = truth.angular_rate_body_rad_s + self.rng.normal(0.0, self.omega_sigma_rad_s, size=3)

        z = np.hstack((pos, vel, q, w))
        return Measurement(vector=z, t_s=t_s)
