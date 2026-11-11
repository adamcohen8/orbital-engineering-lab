from __future__ import annotations

import numpy as np

from sim.utils.frames import eci_relative_to_ric_rect


def state_difference_residuals(
    sim_states: np.ndarray,
    ref_states: np.ndarray,
    *,
    component_sigma: np.ndarray | float = 1.0,
    columns: slice = slice(0, 3),
) -> np.ndarray:
    """Return flattened weighted state-difference residuals."""

    sim = np.asarray(sim_states, dtype=float)
    ref = np.asarray(ref_states, dtype=float)
    if sim.shape != ref.shape or sim.ndim != 2:
        raise ValueError("sim_states and ref_states must be 2D arrays with matching shape.")
    diff = sim[:, columns] - ref[:, columns]
    sigma = np.asarray(component_sigma, dtype=float)
    if not np.all(np.isfinite(sigma)) or np.any(sigma <= 0.0):
        raise ValueError("component_sigma must contain finite positive values.")
    return (diff / sigma).reshape(-1)


def ric_position_residuals_m(
    sim_states_eci_km_km_s: np.ndarray,
    ref_states_eci_km_km_s: np.ndarray,
    *,
    sigma_m: np.ndarray | float = 1.0,
) -> np.ndarray:
    """Truth-centered RIC position residuals in meters, flattened and weighted."""

    sim = np.asarray(sim_states_eci_km_km_s, dtype=float)
    ref = np.asarray(ref_states_eci_km_km_s, dtype=float)
    if sim.shape != ref.shape or sim.ndim != 2 or sim.shape[1] < 6:
        raise ValueError("sim/ref states must have shape (N, >=6).")
    ric = np.vstack([eci_relative_to_ric_rect(sim[k, :6], ref[k, :6])[:3] for k in range(ref.shape[0])])
    sigma = np.asarray(sigma_m, dtype=float)
    if not np.all(np.isfinite(sigma)) or np.any(sigma <= 0.0):
        raise ValueError("sigma_m must contain finite positive values.")
    return (ric * 1.0e3 / sigma).reshape(-1)
