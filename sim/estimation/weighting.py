from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from scipy.linalg import solve_triangular


def validate_covariance_block(
    covariance: Sequence[Sequence[float]] | np.ndarray,
    *,
    dimension: int,
    field_name: str = "covariance",
    require_positive_definite: bool = True,
) -> np.ndarray:
    cov = np.asarray(covariance, dtype=float)
    expected = (int(dimension), int(dimension))
    if cov.shape != expected:
        raise ValueError(f"{field_name} must have shape {expected}.")
    if not np.all(np.isfinite(cov)):
        raise ValueError(f"{field_name} must be finite.")
    if not np.allclose(cov, cov.T, rtol=1.0e-10, atol=1.0e-14):
        raise ValueError(f"{field_name} must be symmetric.")
    eigenvalues = np.linalg.eigvalsh(0.5 * (cov + cov.T))
    minimum = float(np.min(eigenvalues))
    if require_positive_definite and minimum <= 0.0:
        raise ValueError(f"{field_name} must be positive definite for whitening.")
    if not require_positive_definite and minimum < -1.0e-14:
        raise ValueError(f"{field_name} must be positive semidefinite.")
    return 0.5 * (cov + cov.T)


def whiten_residual_block(
    residual: Sequence[float] | np.ndarray,
    covariance: Sequence[Sequence[float]] | np.ndarray,
    *,
    field_name: str = "covariance",
) -> np.ndarray:
    vector = np.asarray(residual, dtype=float).reshape(-1)
    factor = prepare_covariance_whitener(
        covariance,
        dimension=vector.size,
        field_name=field_name,
    )
    return whiten_residual_with_factor(vector, factor, field_name=field_name)


def prepare_covariance_whitener(
    covariance: Sequence[Sequence[float]] | np.ndarray,
    *,
    dimension: int,
    field_name: str = "covariance",
) -> np.ndarray:
    """Validate and factor an immutable covariance block once.

    Optimizer residual functions may call :func:`whiten_residual_with_factor`
    repeatedly with the returned factor.  Validation remains at the ingestion
    boundary instead of being silently bypassed inside the hot loop.
    """

    cov = validate_covariance_block(
        covariance,
        dimension=dimension,
        field_name=field_name,
        require_positive_definite=True,
    )
    try:
        return np.linalg.cholesky(cov)
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"{field_name} could not be factored for whitening.") from exc


def whiten_residual_with_factor(
    residual: Sequence[float] | np.ndarray,
    factor: Sequence[Sequence[float]] | np.ndarray,
    *,
    field_name: str = "covariance",
) -> np.ndarray:
    """Whiten one residual with a previously validated Cholesky factor."""

    vector = np.asarray(residual, dtype=float).reshape(-1)
    if not np.all(np.isfinite(vector)):
        raise ValueError("residual block must be finite.")
    matrix = np.asarray(factor, dtype=float)
    if matrix.shape != (vector.size, vector.size):
        raise ValueError(f"{field_name} factor must have shape {(vector.size, vector.size)}.")
    return solve_triangular(matrix, vector, lower=True, check_finite=True)


def covariance_from_sigmas(sigmas: Sequence[float] | np.ndarray) -> np.ndarray:
    values = np.asarray(sigmas, dtype=float).reshape(-1)
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("sigma values must be finite and positive.")
    return np.diag(values * values)


def observation_covariance(
    observation: Mapping[str, Any],
    *,
    sigmas: Sequence[float] | np.ndarray,
    dimension: int,
) -> tuple[np.ndarray, str]:
    uncertainty = dict(observation.get("uncertainty", {}) or {})
    matrix = uncertainty.get("matrix")
    if uncertainty.get("representation") == "covariance" and matrix is not None:
        matrix_array = np.asarray(matrix, dtype=float)
        # Dynamics OD explicitly falls back to a position-only fit when a
        # packet mixes position and PV rows.  A PV row's 6x6 covariance still
        # carries the valid marginal position covariance in its leading block;
        # use that block for the 3D residual instead of rejecting the packet
        # after ingestion has accepted the documented fallback.
        if int(dimension) == 3 and matrix_array.shape == (6, 6):
            matrix = matrix_array[:3, :3]
        cov = validate_covariance_block(
            matrix,
            dimension=dimension,
            field_name="observation uncertainty covariance",
            require_positive_definite=True,
        )
        return cov, str(uncertainty.get("source", "provided_covariance"))
    return covariance_from_sigmas(sigmas), "diagonal_sigmas"


def robust_weights(
    residual: Sequence[float] | np.ndarray,
    *,
    loss: str,
    f_scale: float,
) -> np.ndarray:
    values = np.asarray(residual, dtype=float).reshape(-1)
    scale = float(f_scale)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("robust f_scale must be finite and positive.")
    key = str(loss or "linear").strip().lower()
    supported = {"linear", "soft_l1", "huber", "cauchy", "arctan"}
    if key not in supported:
        raise ValueError(f"robust loss must be one of {sorted(supported)}.")
    z = np.abs(values) / scale
    if key == "linear":
        return np.ones_like(values)
    if key == "soft_l1":
        return 1.0 / np.sqrt(1.0 + z * z)
    if key == "huber":
        return np.where(z <= 1.0, 1.0, 1.0 / np.maximum(z, 1.0e-15))
    if key == "cauchy":
        return 1.0 / (1.0 + z * z)
    return 1.0 / (1.0 + z**4)
