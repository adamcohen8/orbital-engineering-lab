"""Seven-coefficient ECOM2 radiation acceleration in the instantaneous DYB frame.

D points toward the Sun, Y is normalized D cross geocentric radial, and B=D
cross Y. Phase is the satellite argument of latitude relative to the Sun's
projection into its instantaneous orbital plane. Coefficients are km/s² at
one AU and multiply inverse-square distance and conical Earth illumination.
This empirical model represents radiation residuals, not optical properties.
See Bury et al. (2020), doi:10.1007/s00190-020-01342-2, section 1.2.
"""

from dataclasses import dataclass

import numpy as np

from sim.dynamics.orbit.eclipse import srp_shadow_factor
from sim.dynamics.orbit.epoch import AU_KM, resolve_sun_moon_positions


def ecom2_acceleration(state, sun_eci_km, coefficients_km_s2):
    x, sun, c = map(lambda a: np.asarray(a, dtype=float), (state, sun_eci_km, coefficients_km_s2))
    if x.shape != (6,) or sun.shape != (3,) or c.shape != (7,) or not all(np.all(np.isfinite(a)) for a in (x, sun, c)):
        raise ValueError("ECOM2 requires finite state, Sun and seven coefficients")
    r, v = x[:3], x[3:]
    delta = sun - r
    distance, radius = np.linalg.norm(delta), np.linalg.norm(r)
    h = np.cross(r, v)
    hn = np.linalg.norm(h)
    if distance <= 0 or radius <= 0 or hn <= 0:
        raise ValueError("ECOM2 requires nondegenerate Sun/orbit geometry")
    d, n, radial = delta / distance, h / hn, r / radius
    y = np.cross(d, radial)
    yn = np.linalg.norm(y)
    projected = d - (d @ n) * n
    pn = np.linalg.norm(projected)
    if min(yn, pn) < 1e-12:
        raise ValueError("ECOM2 Sun/orbit frame is singular")
    y /= yn
    b = np.cross(d, y)
    projected /= pn
    cs, sn = radial @ projected, radial @ np.cross(n, projected)
    acc = (c[0] + c[3] * (2 * cs * cs - 1) + c[4] * 2 * sn * cs) * d + c[1] * y + (c[2] + c[5] * cs + c[6] * sn) * b
    shadow = srp_shadow_factor(r_sc_eci_km=r, t_s=0.0, env={"sun_pos_eci_km": sun, "srp_shadow_model": "conical"})
    return acc * (AU_KM / distance) ** 2 * shadow


@dataclass(frozen=True)
class ECOM2Radiation:
    coefficients_km_s2: tuple[float, ...] = (0.0,) * 7

    def __post_init__(self):
        c = np.asarray(self.coefficients_km_s2, dtype=float)
        if c.shape != (7,) or not np.all(np.isfinite(c)):
            raise ValueError("ECOM2 requires seven finite coefficients")
        object.__setattr__(self, "coefficients_km_s2", tuple(map(float, c)))

    def __call__(self, t_s, state, env, ctx):
        sun, _ = resolve_sun_moon_positions(env, t_s)
        return ecom2_acceleration(state, sun, self.coefficients_km_s2)
