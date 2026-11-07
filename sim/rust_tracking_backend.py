"""Optional Rust kernels for onboard relative tracking measurements.

The Python knowledge layer remains responsible for access gates, sensor pose,
noise/bias generation, RNG ordering, and estimator state.  This adapter only
passes prepared ECI target/observer states to the native fixed-size geometry
and forward finite-difference kernels.
"""

from __future__ import annotations

from math import isfinite
from typing import Iterable

import numpy as np

from sim.rust_orbit_backend import _extension

MODEL_CODES: dict[str, int] = {
    "relative_range": 0,
    "relative_range_rate": 1,
    "relative_angles": 2,
    "relative_angles_range": 3,
    "relative_angles_range_rate": 4,
}
MODEL_DIMENSIONS: dict[str, int] = {
    "relative_range": 1,
    "relative_range_rate": 2,
    "relative_angles": 2,
    "relative_angles_range": 3,
    "relative_angles_range_rate": 4,
}
DEFAULT_FINITE_DIFFERENCE_STEPS = np.array(
    [1.0e-3, 1.0e-3, 1.0e-3, 1.0e-6, 1.0e-6, 1.0e-6], dtype=np.float64
)
_BATCH_RECORD_SIZE = 1 + 4 + 24


def _fixed_state(value: np.ndarray | Iterable[float], name: str) -> list[float]:
    state = np.asarray(value, dtype=np.float64).reshape(-1)
    if state.shape != (6,):
        raise ValueError(f"{name} must have shape (6,)")
    values = state.tolist()
    if not all(map(isfinite, values)):
        raise ValueError(f"{name} must contain finite values")
    return values


def _state_batch(value: np.ndarray | Iterable[float], name: str) -> np.ndarray:
    states = np.asarray(value, dtype=np.float64)
    if states.ndim == 1:
        if states.size % 6 != 0:
            raise ValueError(f"{name} must contain rows of six values")
        states = states.reshape(-1, 6)
    if states.ndim != 2 or states.shape[1] != 6:
        raise ValueError(f"{name} must have shape (samples, 6)")
    if not np.isfinite(states).all():
        raise ValueError(f"{name} must contain finite values")
    return np.ascontiguousarray(states)


def _model_code(model: str) -> int:
    key = str(model).strip().lower().replace("-", "_")
    if key not in MODEL_CODES:
        valid = ", ".join(sorted(MODEL_CODES))
        raise ValueError(f"Unsupported Rust relative tracking model {model!r}; valid models: {valid}")
    return MODEL_CODES[key]


def _steps(value: np.ndarray | Iterable[float] | None) -> list[float]:
    steps = DEFAULT_FINITE_DIFFERENCE_STEPS if value is None else np.asarray(value, dtype=np.float64).reshape(-1)
    if steps.shape != (6,):
        raise ValueError("finite-difference steps must have shape (6,)")
    values = steps.tolist()
    if not all(isfinite(step) and step > 0.0 for step in values):
        raise ValueError("finite-difference steps must be finite and positive")
    return values


def _batch_steps(value: np.ndarray | Iterable[float] | None, rows: int) -> list[float]:
    """Normalize the native batch step ABI (shared or one row per sample)."""

    if value is None:
        return DEFAULT_FINITE_DIFFERENCE_STEPS.tolist()
    steps = np.asarray(value, dtype=np.float64).reshape(-1)
    if steps.size not in (6, rows * 6):
        raise ValueError(
            "finite-difference steps must contain 6 values or 6 values per tracking row"
        )
    values = steps.tolist()
    if not all(isfinite(step) and step > 0.0 for step in values):
        raise ValueError("finite-difference steps must be finite and positive")
    return values


def _native(name: str):
    function = getattr(_extension(), name, None)
    if function is None:
        raise RuntimeError(
            "Rust tracking backend requires an oel_rust_orbit wheel with tracking kernels "
            f"({name})"
        )
    return function


def relative_measurement(
    target_state: np.ndarray,
    observer_state: np.ndarray,
    model: str,
) -> np.ndarray:
    """Evaluate one relative range/angle measurement in ECI geometry."""

    key = str(model).strip().lower().replace("-", "_")
    result = _native("tracking_measurement")(
        _fixed_state(target_state, "target_state"),
        _fixed_state(observer_state, "observer_state"),
        int(_model_code(key)),
    )
    output = np.asarray(result, dtype=np.float64).reshape(-1)
    expected = MODEL_DIMENSIONS[key]
    if output.shape != (expected,):
        raise RuntimeError("Rust tracking kernel returned an invalid measurement shape")
    return output.copy()


def relative_measurement_and_jacobian(
    target_state: np.ndarray,
    observer_state: np.ndarray,
    model: str,
    *,
    steps: np.ndarray | Iterable[float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate one measurement and all six forward FD predictions in Rust."""

    key = str(model).strip().lower().replace("-", "_")
    code = _model_code(key)
    dimension = MODEL_DIMENSIONS[key]
    native = _extension()
    byte_function = getattr(native, "tracking_measurement_and_jacobian_bytes", None)
    measurement, jacobian = (byte_function or _native("tracking_measurement_and_jacobian"))(
        _fixed_state(target_state, "target_state"),
        _fixed_state(observer_state, "observer_state"),
        int(code),
        _steps(steps),
    )
    if byte_function is not None:
        measurement_array = np.frombuffer(measurement, dtype="<f8")
        jacobian_array = np.frombuffer(jacobian, dtype="<f8")
    else:
        measurement_array = np.asarray(measurement, dtype=np.float64).reshape(-1)
        jacobian_array = np.asarray(jacobian, dtype=np.float64).reshape(-1)
    if measurement_array.shape != (dimension,):
        raise RuntimeError("Rust tracking kernel returned an invalid measurement shape")
    if jacobian_array.size != dimension * 6:
        raise RuntimeError("Rust tracking kernel returned an invalid Jacobian shape")
    return measurement_array.copy(), jacobian_array.reshape(dimension, 6).copy()


def relative_measurement_batch(
    target_states: np.ndarray,
    observer_states: np.ndarray,
    model: str,
) -> np.ndarray:
    """Evaluate a same-model batch, returning ``(samples, dimension)``."""

    targets = _state_batch(target_states, "target_states")
    observers = _state_batch(observer_states, "observer_states")
    if targets.shape != observers.shape:
        raise ValueError("target_states and observer_states must have matching shapes")
    key = str(model).strip().lower().replace("-", "_")
    dimension = MODEL_DIMENSIONS.get(key)
    if dimension is None:
        _model_code(key)
    byte_function = getattr(_extension(), "tracking_measurement_batch_bytes", None)
    if byte_function is not None:
        result = byte_function(np.asarray(targets, dtype="<f8").tobytes(), np.asarray(observers, dtype="<f8").tobytes(), int(MODEL_CODES[key]))
        output = np.frombuffer(result, dtype="<f8")
    else:
        result = _native("tracking_measurement_batch")(
            targets.reshape(-1).tolist(), observers.reshape(-1).tolist(), int(MODEL_CODES[key])
        )
        output = np.asarray(result, dtype=np.float64).reshape(-1)
    expected = targets.shape[0] * int(dimension)
    if output.size != expected:
        raise RuntimeError("Rust tracking batch returned an invalid measurement shape")
    return output.reshape(targets.shape[0], int(dimension)).copy()


def relative_measurement_and_jacobian_batch(
    target_states: np.ndarray,
    observer_states: np.ndarray,
    models: Iterable[str],
    *,
    steps: np.ndarray | Iterable[float] | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Evaluate mixed-model rows and parse Rust's fixed-width records."""

    targets = _state_batch(target_states, "target_states")
    observers = _state_batch(observer_states, "observer_states")
    if targets.shape != observers.shape:
        raise ValueError("target_states and observer_states must have matching shapes")
    model_list = [str(model).strip().lower().replace("-", "_") for model in models]
    if len(model_list) != targets.shape[0]:
        raise ValueError("models must contain one model per tracking row")
    codes = [_model_code(model) for model in model_list]
    step_array = _batch_steps(steps, targets.shape[0])
    byte_function = getattr(_extension(), "tracking_measurement_and_jacobian_batch_bytes", None)
    if byte_function is not None:
        result = byte_function(np.asarray(targets, dtype="<f8").tobytes(), np.asarray(observers, dtype="<f8").tobytes(), codes, step_array)
        values = np.frombuffer(result, dtype="<f8")
    else:
        result = _native("tracking_measurement_and_jacobian_batch")(
            targets.reshape(-1).tolist(), observers.reshape(-1).tolist(), codes, step_array,
        )
        values = np.asarray(result, dtype=np.float64).reshape(-1)
    expected = targets.shape[0] * _BATCH_RECORD_SIZE
    if values.size != expected:
        raise RuntimeError("Rust tracking batch returned an invalid record layout")
    output: list[tuple[np.ndarray, np.ndarray]] = []
    for row, model in enumerate(model_list):
        record = values[row * _BATCH_RECORD_SIZE : (row + 1) * _BATCH_RECORD_SIZE]
        dimension = MODEL_DIMENSIONS[model]
        if int(record[0]) != dimension:
            raise RuntimeError("Rust tracking batch returned an invalid model dimension")
        output.append(
            (
                record[1 : 1 + dimension].copy(),
                record[5 : 5 + dimension * 6].reshape(dimension, 6).copy(),
            )
        )
    return output


__all__ = [
    "DEFAULT_FINITE_DIFFERENCE_STEPS",
    "MODEL_CODES",
    "MODEL_DIMENSIONS",
    "relative_measurement",
    "relative_measurement_and_jacobian",
    "relative_measurement_batch",
    "relative_measurement_and_jacobian_batch",
]
