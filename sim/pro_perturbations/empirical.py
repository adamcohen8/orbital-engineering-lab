"""Explicit constant and harmonic accelerations in the instantaneous radial/transverse/normal frame.

An empirical force represents unmodeled dynamics; it does not identify their
physical cause. Coefficients are prescribed or estimated on declared fit data,
then kept fixed during prediction. Units are km/s²; transverse is N cross R.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def empirical_ric_acceleration(state, coefficients_km_s2):
    state = np.asarray(state, dtype=float)
    a = np.asarray(coefficients_km_s2, dtype=float)
    if state.shape != (6,) or a.shape != (3,) or not np.all(np.isfinite(state)) or not np.all(np.isfinite(a)):
        raise ValueError("Empirical RIC acceleration requires finite state and three coefficients.")
    r = state[:3]
    h = np.cross(r, state[3:])
    rn, hn = np.linalg.norm(r), np.linalg.norm(h)
    if rn <= 0.0 or hn <= 0.0:
        raise ValueError("Empirical RIC frame requires nonzero radius and angular momentum.")
    radial, normal = r / rn, h / hn
    transverse = np.cross(normal, radial)
    return a[0] * radial + a[1] * transverse + a[2] * normal


@dataclass(frozen=True)
class EmpiricalRICAcceleration:
    coefficients_km_s2: tuple[float, float, float] = (0.0, 0.0, 0.0)
    cosine_km_s2: tuple[float, float, float] = (0.0, 0.0, 0.0)
    sine_km_s2: tuple[float, float, float] = (0.0, 0.0, 0.0)
    angular_frequency_rad_s: float = 0.0
    reference_time_s: float = 0.0

    def __post_init__(self):
        for name in ("coefficients_km_s2", "cosine_km_s2", "sine_km_s2"):
            a = np.asarray(getattr(self, name), dtype=float)
            if a.shape != (3,) or not np.all(np.isfinite(a)):
                raise ValueError("Empirical RIC acceleration requires three finite coefficients.")
            object.__setattr__(self, name, tuple(map(float, a)))
        if (
            not math.isfinite(self.angular_frequency_rad_s)
            or self.angular_frequency_rad_s < 0.0
            or not math.isfinite(self.reference_time_s)
        ):
            raise ValueError(
                "Empirical RIC harmonic frequency and reference time must be finite, with nonnegative frequency."
            )
        if self.angular_frequency_rad_s == 0.0 and any(self.cosine_km_s2 + self.sine_km_s2):
            raise ValueError("Empirical harmonic coefficients require positive frequency.")

    def __call__(self, t_s, state, env, ctx):
        phase = self.angular_frequency_rad_s * (float(t_s) - self.reference_time_s)
        a = np.array(self.coefficients_km_s2)
        if self.angular_frequency_rad_s:
            a += np.array(self.cosine_km_s2) * math.cos(phase) + np.array(self.sine_km_s2) * math.sin(phase)
        return empirical_ric_acceleration(state, a)
