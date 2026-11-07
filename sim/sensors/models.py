from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from sim.core.interfaces import SensorModel
from sim.core.models import Measurement, StateTruth
from sim.sensors.access import AccessModel


def _noise_vector(value: object, field_name: str, *, allow_empty: bool) -> np.ndarray:
    try:
        vector = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{field_name} must be a one-dimensional finite numeric vector.") from exc
    if vector.ndim != 1 or (not allow_empty and vector.size == 0) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{field_name} must be a one-dimensional finite numeric vector.")
    return np.array(vector, dtype=float, copy=True)


@dataclass
class SensorNoiseConfig:
    sigma: np.ndarray
    bias: np.ndarray = field(default_factory=lambda: np.zeros(0))
    dropout_prob: float = 0.0
    latency_s: float = 0.0

    def __post_init__(self) -> None:
        sigma = _noise_vector(self.sigma, "sigma", allow_empty=False)
        bias = _noise_vector(self.bias, "bias", allow_empty=True)
        if np.any(sigma < 0.0):
            raise ValueError("sigma must be nonnegative.")
        if isinstance(self.dropout_prob, (bool, np.bool_)):
            raise ValueError("dropout_prob must be finite within [0, 1].")
        dropout = float(self.dropout_prob)
        if not np.isfinite(dropout) or not 0.0 <= dropout <= 1.0:
            raise ValueError("dropout_prob must be finite within [0, 1].")
        if isinstance(self.latency_s, (bool, np.bool_)):
            raise ValueError("latency_s must be finite and nonnegative.")
        latency = float(self.latency_s)
        if not np.isfinite(latency) or latency < 0.0:
            raise ValueError("latency_s must be finite and nonnegative.")
        self.sigma = sigma
        self.bias = bias
        self.dropout_prob = dropout
        self.latency_s = latency


@dataclass
class OwnStateSensor(SensorModel):
    noise: SensorNoiseConfig
    rng: np.random.Generator
    access_model: AccessModel | None = None
    _latency_queue: list[tuple[float, float, np.ndarray]] = field(default_factory=list)

    def measure(self, truth: StateTruth, env: dict, t_s: float) -> Measurement | None:
        if self.access_model is not None and not self.access_model.can_update(
            truth.position_eci_km, truth.position_eci_km, t_s
        ):
            return None
        if self.rng.random() < self.noise.dropout_prob:
            return None

        x = np.hstack((truth.position_eci_km, truth.velocity_eci_km_s))
        bias = self._bias(len(x))
        sigma = self._sigma(len(x))
        z = x + bias + self.rng.normal(0.0, sigma, size=len(x))

        if self.noise.latency_s <= 0.0:
            return Measurement(vector=z, t_s=t_s)

        acquisition_t = float(t_s)
        release_t = acquisition_t + self.noise.latency_s
        self._latency_queue.append((release_t, acquisition_t, z))
        for i, (ready_t, sample_t, vec) in enumerate(self._latency_queue):
            if ready_t <= t_s:
                self._latency_queue.pop(i)
                return Measurement(vector=vec, t_s=sample_t)
        return None

    def _sigma(self, n: int) -> np.ndarray:
        if self.noise.sigma.size == n:
            return self.noise.sigma
        return np.full(n, float(self.noise.sigma[0]))

    def _bias(self, n: int) -> np.ndarray:
        if self.noise.bias.size == n:
            return self.noise.bias
        if self.noise.bias.size == 0:
            return np.zeros(n)
        return np.full(n, float(self.noise.bias[0]))


@dataclass
class RelativeSensor(SensorModel):
    target_id: str
    mode: str
    noise: SensorNoiseConfig
    rng: np.random.Generator
    access_model: AccessModel | None = None

    def measure(self, truth: StateTruth, env: dict, t_s: float) -> Measurement | None:
        world = env.get("world_truth", {})
        target = world.get(self.target_id)
        if target is None:
            return None

        if self.access_model is not None and not self.access_model.can_update(
            truth.position_eci_km, target.position_eci_km, t_s
        ):
            return None
        if self.rng.random() < self.noise.dropout_prob:
            return None

        rel_r = target.position_eci_km - truth.position_eci_km
        rel_v = target.velocity_eci_km_s - truth.velocity_eci_km_s
        rng_km = np.linalg.norm(rel_r)
        if rng_km == 0.0:
            los = np.zeros(3)
        else:
            los = rel_r / rng_km

        if self.mode == "angle_only":
            az = np.arctan2(los[1], los[0])
            el = np.arcsin(np.clip(los[2], -1.0, 1.0))
            z = np.array([az, el])
        elif self.mode == "range":
            z = np.array([rng_km])
        elif self.mode == "range_rate":
            rr = 0.0 if rng_km == 0.0 else np.dot(rel_v, los)
            z = np.array([rng_km, rr])
        else:
            raise ValueError(f"unsupported relative sensor mode: {self.mode}")

        sigma = self._sigma(len(z))
        bias = self._bias(len(z))
        z = z + bias + self.rng.normal(0.0, sigma, size=len(z))
        return Measurement(vector=z, t_s=t_s)

    def _sigma(self, n: int) -> np.ndarray:
        if self.noise.sigma.size == n:
            return self.noise.sigma
        return np.full(n, float(self.noise.sigma[0]))

    def _bias(self, n: int) -> np.ndarray:
        if self.noise.bias.size == n:
            return self.noise.bias
        if self.noise.bias.size == 0:
            return np.zeros(n)
        return np.full(n, float(self.noise.bias[0]))
