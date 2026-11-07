from __future__ import annotations

import numpy as np

from sim.utils.vector_math import dot3


def _two_body_accel(r: np.ndarray, mu_km3_s2: float, accel_cmd_eci_km_s2: np.ndarray) -> np.ndarray:
    r2 = dot3(r, r)
    if r2 == 0.0:
        return np.array(accel_cmd_eci_km_s2, dtype=float)
    r_norm = float(np.sqrt(r2))
    return (-mu_km3_s2 / (r_norm * r2)) * r + accel_cmd_eci_km_s2


def two_body_derivative(x: np.ndarray, mu_km3_s2: float, accel_cmd_eci_km_s2: np.ndarray) -> np.ndarray:
    dx = np.empty(6, dtype=float)
    dx[:3] = x[3:]
    dx[3:] = _two_body_accel(x[:3], mu_km3_s2, accel_cmd_eci_km_s2)
    return dx


def propagate_two_body_rk4(
    x_eci: np.ndarray,
    dt_s: float,
    mu_km3_s2: float,
    accel_cmd_eci_km_s2: np.ndarray,
) -> np.ndarray:
    x = np.asarray(x_eci, dtype=float)
    accel_cmd = np.asarray(accel_cmd_eci_km_s2, dtype=float)

    k1 = two_body_derivative(x, mu_km3_s2, accel_cmd)
    k2 = two_body_derivative(x + 0.5 * dt_s * k1, mu_km3_s2, accel_cmd)
    k3 = two_body_derivative(x + 0.5 * dt_s * k2, mu_km3_s2, accel_cmd)
    k4 = two_body_derivative(x + dt_s * k3, mu_km3_s2, accel_cmd)
    return x + (dt_s / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
