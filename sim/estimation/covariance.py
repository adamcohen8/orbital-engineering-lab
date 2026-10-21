from __future__ import annotations

import numpy as np


def covariance_from_jacobian(
    jacobian: np.ndarray, residual: np.ndarray, *, dof: int | None = None, scale_policy: str = "conservative"
) -> np.ndarray:
    """Full-rank local covariance from a whitened Jacobian, without squaring its condition.

    ``absolute`` trusts supplied measurement variances; ``residual`` estimates
    one common noise multiplier; ``conservative`` preserves the historic floor
    of one. A singular likelihood has no finite covariance: never interpret a
    pseudoinverse's zero null-space variance as certainty.
    """

    j = np.asarray(jacobian, dtype=float)
    r = np.asarray(residual, dtype=float).reshape(-1)
    if j.ndim != 2:
        raise ValueError("jacobian must be a 2D array.")
    if j.shape[0] != r.size:
        raise ValueError("jacobian row count must match residual length.")
    if not np.all(np.isfinite(j)) or not np.all(np.isfinite(r)) or j.shape[1] == 0:
        raise ValueError("jacobian and residual must be finite with a nonzero parameter dimension.")
    if scale_policy not in {"absolute", "residual", "conservative"}:
        raise ValueError("scale_policy must be absolute, residual, or conservative.")
    n_params = j.shape[1]
    degrees = int(dof if dof is not None else max(r.size - n_params, 1))
    variance = float(np.dot(r, r) / max(degrees, 1))
    variance = 1.0 if scale_policy == "absolute" else max(variance, 1.0) if scale_policy == "conservative" else variance
    _, singular, vt = np.linalg.svd(j, full_matrices=False)
    threshold = max(j.shape) * np.finfo(float).eps * singular[0] if singular.size else 0.0
    if singular.size < n_params or np.count_nonzero(singular > threshold) != n_params:
        raise ValueError("rank-deficient Jacobian has no finite full-parameter covariance.")
    inverse = vt.T / singular
    covariance = (inverse @ inverse.T) * variance
    return 0.5 * (covariance + covariance.T)


def validate_covariance(value: np.ndarray, dimension: int, *, name: str = "covariance") -> np.ndarray:
    """Validate a finite symmetric PSD matrix without silently repairing inputs."""
    covariance = np.asarray(value, dtype=float)
    if covariance.shape != (dimension, dimension) or not np.all(np.isfinite(covariance)):
        raise ValueError(f"{name} must be a finite ({dimension}, {dimension}) matrix.")
    # Normalize by each coordinate's own standard deviation. A tolerance
    # based on the largest Cartesian entry can hide invalid velocity or force
    # variances when position and velocity have very different native scales.
    diagonal = np.diag(covariance)
    if np.any(diagonal < 0):
        raise ValueError(f"{name} must have nonnegative diagonal variances.")
    positive = diagonal > 0
    if np.any(covariance[~positive, :] != 0) or np.any(covariance[:, ~positive] != 0):
        raise ValueError(f"{name} has a nonzero covariance with a zero-variance coordinate.")
    if np.any(positive):
        scales = np.sqrt(diagonal[positive])
        normalized = covariance[np.ix_(positive, positive)] / np.outer(scales, scales)
        if np.max(np.abs(normalized - normalized.T)) > 1e-10 or np.linalg.eigvalsh(0.5 * (normalized + normalized.T))[
            0
        ] < -1e-10 * len(scales):
            raise ValueError(f"{name} must be symmetric positive semidefinite in normalized coordinates.")
    return covariance


def correlation_from_covariance(covariance: np.ndarray) -> np.ndarray:
    cov = np.asarray(covariance, dtype=float)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError("covariance must be a square matrix.")
    diag = np.sqrt(np.maximum(np.diag(cov), 0.0))
    denom = np.outer(diag, diag)
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.divide(cov, denom, out=np.zeros_like(cov), where=denom > 0.0)
    np.fill_diagonal(corr, 1.0)
    return corr
