"""Small validation helpers shared by attitude-control implementations."""

from __future__ import annotations

import numpy as np


def require_finite_scalar(value: float, name: str, *, allow_positive_inf: bool = False) -> float:
    """Return a scalar after rejecting NaN and unsupported infinity."""

    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric scalar.") from exc
    if np.isnan(result) or (
        not np.isfinite(result) and not (allow_positive_inf and np.isposinf(result))
    ):
        if allow_positive_inf:
            raise ValueError(f"{name} must be finite or positive infinity.")
        raise ValueError(f"{name} must be finite.")
    return result


def require_finite_vector(
    value: object,
    name: str,
    *,
    size: int | None = None,
    allow_positive_inf: bool = False,
) -> np.ndarray:
    """Return a flattened numeric vector after finite-value validation."""

    try:
        result = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric vector.") from exc
    if size is not None and result.size != size:
        raise ValueError(f"{name} must be length-{size}.")
    if allow_positive_inf:
        valid = bool(np.all(np.isfinite(result) | np.isposinf(result)))
    else:
        valid = bool(np.all(np.isfinite(result)))
    if not valid:
        if allow_positive_inf:
            raise ValueError(f"{name} must contain finite values or positive infinity.")
        raise ValueError(f"{name} must contain finite values.")
    return result


def require_finite_matrix(
    value: object,
    name: str,
    *,
    shape: tuple[int, ...] | None = None,
) -> np.ndarray:
    """Return a numeric matrix after shape and finite-value validation."""

    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric array.") from exc
    if shape is not None and result.shape != shape:
        raise ValueError(f"{name} must have shape {shape}.")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain finite values.")
    return result
