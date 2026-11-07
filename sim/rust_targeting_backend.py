"""Optional native two-body coast kernels for bounded trajectory targeting."""

from __future__ import annotations

from importlib import import_module
from operator import index as as_index

import numpy as np


def _extension():
    try:
        return import_module("oel_rust_orbit")
    except ImportError as exc:
        raise RuntimeError("Rust targeting backend requires oel_rust_orbit") from exc


def _state_bytes(state: np.ndarray) -> bytes:
    values = np.asarray(state, dtype="<f8").reshape(6)
    if not np.all(np.isfinite(values)):
        raise ValueError("targeting state must be finite")
    return values.tobytes()


def try_first_apsis(
    state: np.ndarray, *, kind: str, step_s: float, max_coast_s: float, mu_km3_s2: float,
) -> tuple[float, np.ndarray] | None:
    if kind not in {"apogee", "perigee"}:
        raise ValueError("kind must be apogee or perigee")
    native = getattr(_extension(), "targeting_first_apsis_bytes", None)
    if native is None:
        return None
    try:
        time_s, raw = native(
            _state_bytes(state), 0 if kind == "apogee" else 1,
            float(step_s), float(max_coast_s), float(mu_km3_s2),
        )
    except ValueError as exc:
        if str(exc) == "No first apsis event occurred within max_coast_s":
            raise ValueError(f"No first {kind} event occurred within max_coast_s") from exc
        raise
    return float(time_s), np.frombuffer(raw, dtype="<f8").copy().reshape(6)


def try_two_body_coast(
    state: np.ndarray, *, step_s: float, steps: int, mu_km3_s2: float,
) -> np.ndarray | None:
    if isinstance(steps, bool):
        raise ValueError("steps must be a nonnegative integer")
    try:
        step_count = as_index(steps)
    except TypeError as exc:
        raise ValueError("steps must be a nonnegative integer") from exc
    if step_count < 0:
        raise ValueError("steps must be a nonnegative integer")
    native = getattr(_extension(), "targeting_two_body_coast_bytes", None)
    if native is None:
        return None
    raw = native(_state_bytes(state), float(step_s), int(step_count), float(mu_km3_s2))
    return np.frombuffer(raw, dtype="<f8").copy().reshape(6)


def fixed_duration_rollouts(
    state: np.ndarray, trial_segments: list[list[dict]], *, step_s: float, mu_km3_s2: float,
    force_model=(),
) -> np.ndarray:
    """Evaluate fixed-duration RK4 trial trajectories in candidate order."""

    native = getattr(_extension(), "targeting_rollouts_bytes", None)
    if native is None:
        raise RuntimeError("Rust targeting backend requires targeting_rollouts_bytes")
    actions = []
    for segments in trial_segments:
        trial = []
        for segment in segments:
            if segment["type"] == "coast" and "duration_s" in segment:
                trial.append([0.0, float(segment["duration_s"]), 0.0, 0.0, 0.0])
            elif segment["type"] == "impulsive_burn":
                trial.append([1.0 if segment["frame"] == "eci" else 2.0,
                              *[float(value) for value in segment["delta_v_m_s"]], 0.0])
            else:
                raise ValueError("Rust targeting rollouts require fixed-duration coasts and impulsive burns")
        actions.append(trial)
    if force_model:
        native = getattr(_extension(), "targeting_zonal_rollouts_bytes", None)
        if native is None:
            raise RuntimeError("Rust zonal targeting requires targeting_zonal_rollouts_bytes")
        raw = native(_state_bytes(state), actions, float(step_s), float(mu_km3_s2),
                     [{"j2": 2, "j3": 3, "j4": 4}[name] for name in force_model])
    else:
        raw = native(_state_bytes(state), actions, float(step_s), float(mu_km3_s2))
    return np.frombuffer(raw, dtype="<f8").copy().reshape(len(actions), 8)


def zonal_coast_history(state, widths, *, mu_km3_s2, force_model=()):
    """Numerical samples only; Python retains all crossing/refinement policy."""
    native = getattr(_extension(), "orbit_zonal_history_bytes", None)
    if native is None:
        raise RuntimeError("Rust event targeting requires orbit_zonal_history_bytes")
    steps = np.asarray(widths, dtype="<f8").reshape(-1)
    raw = native(_state_bytes(state), steps.tobytes(), float(mu_km3_s2),
                 [{"j2": 2, "j3": 3, "j4": 4}[name] for name in force_model])
    return np.frombuffer(raw, dtype="<f8").copy().reshape(steps.size + 1, 6)
