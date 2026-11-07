"""Shared vectorized observer/target geometry for OEL analysis products."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sim.utils.geodesy import WGS84_A_KM, WGS84_B_KM

# The surface level is dimensionless.  The normal check uses a dot-product
# alignment tolerance because both the supplied and reference normals are
# unit vectors.  These tolerances admit round-off from geodetic construction
# while rejecting physically off-surface or inward-facing targets.
_WGS84_SURFACE_LEVEL_TOLERANCE = 1.0e-12
_WGS84_NORMAL_ALIGNMENT_TOLERANCE = 1.0e-10


@dataclass(frozen=True)
class SurfaceTargetGeometry:
    range_km: np.ndarray
    cosine_off_axis: np.ndarray
    horizon_clearance_km: np.ndarray
    visible: np.ndarray
    inside_pattern: np.ndarray
    inside_range: np.ndarray
    available: np.ndarray


def _validate_wgs84_surface_targets(
    targets: np.ndarray,
    normals: np.ndarray,
) -> None:
    """Validate WGS84 surface membership and outward-normal orientation.

    ``targets`` must satisfy the WGS84 ellipsoid equation within a
    dimensionless ``1e-12`` level tolerance.  ``normals`` must be aligned with
    the normalized ellipsoid gradient within a ``1e-10`` dot-product
    tolerance.  This shared boundary runs before either the Python geometry
    kernel or the optional native kernel.
    """

    axes = np.array([WGS84_A_KM, WGS84_A_KM, WGS84_B_KM], dtype=float)
    level = np.einsum("ij,ij->i", targets / axes, targets / axes)
    invalid_surface = np.abs(level - 1.0) > _WGS84_SURFACE_LEVEL_TOLERANCE
    if np.any(invalid_surface):
        index = int(np.flatnonzero(invalid_surface)[0])
        raise ValueError(
            "target_ecef_km must lie on the WGS84 ellipsoid within "
            f"{_WGS84_SURFACE_LEVEL_TOLERANCE:.1e} normalized level "
            f"(sample {index} has level {level[index]:.17g})."
        )

    gradient = targets / (axes * axes)
    gradient_norm = np.linalg.norm(gradient, axis=1)
    if np.any(~np.isfinite(gradient_norm)) or np.any(gradient_norm <= 0.0):
        raise ValueError("WGS84 target ellipsoid gradients must have positive finite norm.")
    expected_normals = gradient / gradient_norm[:, None]
    alignment = np.einsum("ij,ij->i", normals, expected_normals)
    invalid_normal = alignment < 1.0 - _WGS84_NORMAL_ALIGNMENT_TOLERANCE
    if np.any(invalid_normal):
        index = int(np.flatnonzero(invalid_normal)[0])
        raise ValueError(
            "target_outward_normal_ecef must be the unit outward WGS84 normal "
            f"within {_WGS84_NORMAL_ALIGNMENT_TOLERANCE:.1e} alignment "
            f"(sample {index} has alignment {alignment[index]:.17g})."
        )


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
    _validate_wgs84_surface_targets(targets, normals)
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
