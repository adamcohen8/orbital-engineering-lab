"""Shared vectorized observer/target geometry for OEL analysis products."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SurfaceTargetGeometry:
    range_km: np.ndarray
    cosine_off_axis: np.ndarray
    horizon_clearance_km: np.ndarray
    visible: np.ndarray
    inside_pattern: np.ndarray
    inside_range: np.ndarray
    available: np.ndarray


def evaluate_surface_targets_ecef(
    *,
    observer_ecef_km: np.ndarray,
    target_ecef_km: np.ndarray,
    target_outward_normal_ecef: np.ndarray,
    boresight_ecef: np.ndarray,
    half_angle_rad: float,
    max_range_km: float | None = None,
    angular_tolerance_rad: float = 1.0e-12,
    range_tolerance_km: float = 1.0e-9,
    numeric_backend: str = "rust",
) -> SurfaceTargetGeometry:
    """Evaluate clear-line-of-sight hard-cone geometry to convex Earth targets.

    Targets must lie on the WGS84 ellipsoid and normals must be unit outward
    normals. Positive normal clearance is equivalent to an unobstructed open
    segment for a convex ellipsoid; exact tangency is blocked.
    """

    backend = str(numeric_backend).strip().lower()
    if backend not in {"python", "rust"}:
        raise ValueError(f"Unknown coverage numeric backend {numeric_backend!r}.")
    observer = np.asarray(observer_ecef_km, dtype=float).reshape(3)
    targets = np.asarray(target_ecef_km, dtype=float)
    normals = np.asarray(target_outward_normal_ecef, dtype=float)
    boresight = np.asarray(boresight_ecef, dtype=float).reshape(3)
    if targets.ndim != 2 or targets.shape[1] != 3 or not np.all(np.isfinite(targets)):
        raise ValueError("target_ecef_km must be finite with shape (targets, 3).")
    if normals.shape != targets.shape or not np.all(np.isfinite(normals)):
        raise ValueError("target_outward_normal_ecef must be finite and match target_ecef_km.")
    if not np.all(np.isfinite(observer)) or not np.all(np.isfinite(boresight)):
        raise ValueError("Observer and boresight vectors must be finite.")
    boresight_norm = float(np.linalg.norm(boresight))
    if abs(boresight_norm - 1.0) > 1.0e-10:
        raise ValueError("boresight_ecef must be a unit vector within 1e-10.")
    normal_norms = np.linalg.norm(normals, axis=1)
    if np.any(np.abs(normal_norms - 1.0) > 1.0e-10):
        raise ValueError("Target outward normals must be unit vectors within 1e-10.")
    half_angle = float(half_angle_rad)
    if not np.isfinite(half_angle) or not 0.0 < half_angle < 0.5 * np.pi:
        raise ValueError("half_angle_rad must be finite and strictly within (0, pi/2).")
    maximum_range = None if max_range_km is None else float(max_range_km)
    if maximum_range is not None and (not np.isfinite(maximum_range) or maximum_range <= 0.0):
        raise ValueError("max_range_km must be positive and finite when provided.")
    angular_tolerance = float(angular_tolerance_rad)
    range_tolerance = float(range_tolerance_km)
    if not np.isfinite(angular_tolerance) or angular_tolerance < 0.0:
        raise ValueError("angular_tolerance_rad must be finite and nonnegative.")
    if not np.isfinite(range_tolerance) or range_tolerance < 0.0:
        raise ValueError("range_tolerance_km must be finite and nonnegative.")
    if half_angle + angular_tolerance >= np.pi:
        raise ValueError("half_angle_rad plus angular_tolerance_rad must remain below pi.")

    if backend == "rust":
        from sim.rust_coverage_backend import surface_targets_ecef

        result = surface_targets_ecef(
            observer,
            targets,
            normals,
            boresight,
            half_angle,
            maximum_range,
            angular_tolerance_rad=angular_tolerance,
            range_tolerance_km=range_tolerance,
        )
        return SurfaceTargetGeometry(
            range_km=result.range_km,
            cosine_off_axis=result.cosine_off_axis,
            horizon_clearance_km=result.horizon_clearance_km,
            visible=result.visible,
            inside_pattern=result.inside_pattern,
            inside_range=result.inside_range,
            available=result.available,
        )

    delta = targets - observer
    ranges = np.linalg.norm(delta, axis=1)
    if np.any(~np.isfinite(ranges)) or np.any(ranges <= 0.0):
        raise ValueError("Observer/target range must be positive and finite.")
    horizon_clearance = np.einsum("ij,ij->i", normals, observer - targets)
    cosine_off_axis = (delta @ boresight) / ranges
    visible = horizon_clearance > range_tolerance
    inside_pattern = cosine_off_axis >= float(
        np.cos(half_angle + angular_tolerance)
    )
    if maximum_range is None:
        inside_range = np.ones(ranges.shape, dtype=bool)
    else:
        inside_range = ranges <= maximum_range + range_tolerance
    return SurfaceTargetGeometry(
        range_km=ranges,
        cosine_off_axis=cosine_off_axis,
        horizon_clearance_km=horizon_clearance,
        visible=visible,
        inside_pattern=inside_pattern,
        inside_range=inside_range,
        available=visible & inside_pattern & inside_range,
    )


__all__ = ["SurfaceTargetGeometry", "evaluate_surface_targets_ecef"]
