"""Primary-centered first post-Newtonian Schwarzschild acceleration (km/s²).

Additive correction in inertial Cartesian coordinates; Newtonian attraction is
supplied by ONP. No spin, third-body relativity, or clock corrections.
"""

import math
from dataclasses import dataclass, field

import numpy as np

SPEED_OF_LIGHT_KM_S = 299792.458


def schwarzschild_acceleration(state, mu_km3_s2, *, use_acceleration=False):
    """Return the weak-field, slow-motion Schwarzschild correction only."""
    x = np.asarray(state, dtype=float)
    mu = float(mu_km3_s2)
    if x.shape != (6,) or not np.all(np.isfinite(x)):
        raise ValueError("Schwarzschild requires a finite Cartesian 6-state in km and km/s.")
    if not math.isfinite(mu) or mu <= 0:
        raise ValueError("Schwarzschild requires finite positive gravitational mu.")
    r, v = x[:3], x[3:]
    radius = float(np.linalg.norm(r))
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError("Schwarzschild requires a finite positive geocentric radius.")
    if use_acceleration:
        from sim.pro_perturbations._kernels import schwarzschild_kernel

        return schwarzschild_kernel(x, mu, radius)
    return mu / (SPEED_OF_LIGHT_KM_S**2 * radius**3) * ((4 * mu / radius - v @ v) * r + 4 * (r @ v) * v)


@dataclass(frozen=True)
class SchwarzschildAcceleration:
    """Opt-in, mass-independent OEL orbit acceleration plugin."""

    acceleration_mode: str = "off"
    _accelerated: bool = field(init=False, repr=False, compare=False, default=False)
    _rust_acceleration: object | None = field(init=False, repr=False, compare=False, default=None)

    def __post_init__(self):
        from sim.acceleration.settings import acceleration_settings_from_mode

        object.__setattr__(self, "_accelerated", acceleration_settings_from_mode(self.acceleration_mode).enabled)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_rust_acceleration"] = None
        return state

    def __setstate__(self, state):
        for name, value in state.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "_rust_acceleration", None)

    def __call__(self, t_s, x_eci, env, ctx):
        if env.get("_rust_numeric_backend") == "rust":
            prepared = self._rust_acceleration
            if prepared is None:
                from sim.rust_environment_backend import prepare_schwarzschild_acceleration

                prepared = prepare_schwarzschild_acceleration() or False
                object.__setattr__(self, "_rust_acceleration", prepared)
            if prepared:
                return prepared(x_eci, mu_km3_s2=ctx.mu_km3_s2)
            from sim.rust_environment_backend import try_schwarzschild_acceleration

            native = try_schwarzschild_acceleration(x_eci, mu_km3_s2=ctx.mu_km3_s2)
            if native is not None:
                return native
        return schwarzschild_acceleration(x_eci, ctx.mu_km3_s2, use_acceleration=self._accelerated)
