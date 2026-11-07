"""Transparent covariance projection and educational two-dimensional Pc."""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr


class ConjunctionProbabilityError(ValueError):
    """Raised for invalid covariance or probability inputs."""


def validate_covariance(matrix: Sequence[Sequence[float]], *, dimension: int) -> np.ndarray:
    covariance = np.asarray(matrix, dtype=float)
    if covariance.shape != (dimension, dimension) or not np.all(np.isfinite(covariance)):
        raise ConjunctionProbabilityError(f"Covariance must be a finite {dimension}x{dimension} matrix.")
    scale = float(np.max(np.abs(covariance)))
    if scale == 0.0:
        return covariance
    scaled = covariance / scale
    if float(np.max(np.abs(scaled - scaled.T))) > 1.0e-12:
        raise ConjunctionProbabilityError("Covariance must be symmetric.")
    scaled = 0.5 * (scaled + scaled.T)
    minimum = float(np.min(np.linalg.eigvalsh(scaled)))
    if minimum < -1.0e-12:
        raise ConjunctionProbabilityError(f"Covariance must be positive semidefinite; minimum eigenvalue={minimum}.")
    return scaled * scale


def ric_basis(state_eci_km_km_s: Sequence[float]) -> np.ndarray:
    state = np.asarray(state_eci_km_km_s, dtype=float)
    if state.shape != (6,) or not np.all(np.isfinite(state)):
        raise ConjunctionProbabilityError("RIC basis requires one finite six-element Cartesian state.")
    radial = state[:3] / float(np.linalg.norm(state[:3]))
    cross = np.cross(state[:3], state[3:])
    cross_norm = float(np.linalg.norm(cross))
    if cross_norm <= 1.0e-12:
        raise ConjunctionProbabilityError("RIC basis requires nonzero orbital angular momentum.")
    cross_track = cross / cross_norm
    in_track = np.cross(cross_track, radial)
    return np.column_stack((radial, in_track, cross_track))


def covariance_rtn_si_to_eci_km(matrix: Sequence[Sequence[float]], state_eci_km_km_s: Sequence[float]) -> np.ndarray:
    """Rotate a CDM RTN covariance from SI units into ECI km/km-s units."""

    covariance_si = validate_covariance(matrix, dimension=6)
    rotation = ric_basis(state_eci_km_km_s)
    transform = np.zeros((6, 6), dtype=float)
    transform[:3, :3] = rotation
    transform[3:, 3:] = rotation
    covariance_km = covariance_si * 1.0e-6
    return 0.5 * (transform @ covariance_km @ transform.T + (transform @ covariance_km @ transform.T).T)


def project_combined_covariance(
    primary_covariance_eci_km: Sequence[Sequence[float]],
    secondary_covariance_eci_km: Sequence[Sequence[float]],
    encounter_basis_rows_eci: Sequence[Sequence[float]],
) -> dict[str, Any]:
    primary = validate_covariance(primary_covariance_eci_km, dimension=6)
    secondary = validate_covariance(secondary_covariance_eci_km, dimension=6)
    basis = np.asarray(encounter_basis_rows_eci, dtype=float)
    if basis.shape != (3, 3) or float(np.max(np.abs(basis @ basis.T - np.eye(3)))) > 1.0e-9:
        raise ConjunctionProbabilityError("Encounter basis must be an orthonormal 3x3 matrix.")
    projection = basis[:2, :]
    combined_position = primary[:3, :3] + secondary[:3, :3]
    plane = 0.5 * (projection @ combined_position @ projection.T + (projection @ combined_position @ projection.T).T)
    validate_covariance(plane, dimension=2)
    eigenvalues = np.linalg.eigvalsh(plane)
    if float(np.min(eigenvalues)) <= 0.0:
        raise ConjunctionProbabilityError("Encounter-plane covariance must be positive definite for 2D Pc.")
    return {
        "combined_position_covariance_eci_km2": combined_position.tolist(),
        "encounter_plane_covariance_km2": plane.tolist(),
        "sigma_minor_km": math.sqrt(float(np.min(eigenvalues))),
        "sigma_major_km": math.sqrt(float(np.max(eigenvalues))),
    }


def collision_probability_2d(
    mean_plane_km: Sequence[float],
    covariance_plane_km2: Sequence[Sequence[float]],
    hard_body_radius_km: float,
    *,
    radial_order: int = 48,
    angular_order: int = 96,
    numeric_backend: str = "rust",
) -> dict[str, Any]:
    """Integrate a bivariate Gaussian over a circular hard-body region.

    The integral is reduced to one standardized marginal coordinate and solved
    with adaptive quadrature.  Standardization prevents a concentrated Gaussian
    from falling between a fixed set of physical-space nodes.
    """

    backend = str(numeric_backend).strip().lower()
    if backend not in {"python", "rust"}:
        raise ConjunctionProbabilityError("numeric_backend must be python or rust.")
    mean = np.asarray(mean_plane_km, dtype=float)
    covariance = validate_covariance(covariance_plane_km2, dimension=2)
    radius = float(hard_body_radius_km)
    if mean.shape != (2,) or not np.all(np.isfinite(mean)):
        raise ConjunctionProbabilityError("Encounter-plane mean must be a finite two-vector.")
    if not math.isfinite(radius) or radius <= 0.0:
        raise ConjunctionProbabilityError("Hard-body radius must be positive and finite.")
    if radial_order < 8 or angular_order < 16:
        raise ConjunctionProbabilityError("Quadrature orders must be at least 8 radial and 16 angular.")
    determinant_sign, _ = np.linalg.slogdet(covariance)
    if determinant_sign <= 0.0:
        raise ConjunctionProbabilityError("Encounter-plane covariance must be positive definite.")
    variance_x = float(covariance[0, 0])
    variance_y = float(covariance[1, 1])
    covariance_xy = float(covariance[0, 1])
    sigma_x = math.sqrt(variance_x)
    conditional_variance_y = variance_y - covariance_xy * covariance_xy / variance_x
    if not math.isfinite(conditional_variance_y) or conditional_variance_y <= 0.0:
        raise ConjunctionProbabilityError("Conditional encounter-plane variance must be positive and finite.")
    conditional_sigma_y = math.sqrt(conditional_variance_y)
    conditional_slope = covariance_xy / sigma_x
    standard_tail_limit = 12.0
    with np.errstate(over="ignore", invalid="ignore"):
        lower_z = max(float((-radius - mean[0]) / sigma_x), -standard_tail_limit)
        upper_z = min(float((radius - mean[0]) / sigma_x), standard_tail_limit)

    def cdf_interval(lower: float, upper: float) -> float:
        if lower >= upper:
            return 0.0
        if lower > 0.0:
            return float(ndtr(-lower) - ndtr(-upper))
        return float(ndtr(upper) - ndtr(lower))

    def integrand(z_value: float) -> float:
        x_value = float(mean[0]) + sigma_x * z_value
        half_height_squared = radius * radius - x_value * x_value
        if half_height_squared <= 0.0:
            return 0.0
        half_height = math.sqrt(half_height_squared)
        conditional_mean_y = float(mean[1]) + conditional_slope * z_value
        lower_y = (-half_height - conditional_mean_y) / conditional_sigma_y
        upper_y = (half_height - conditional_mean_y) / conditional_sigma_y
        standard_density = math.exp(-0.5 * z_value * z_value) / math.sqrt(2.0 * math.pi)
        return standard_density * cdf_interval(lower_y, upper_y)

    callback = integrand
    callback_args = ()
    if backend == "rust":
        from sim.rust_conjunction_probability_backend import integrand as native_integrand

        callback = native_integrand()
        callback_args = (float(mean[0]), sigma_x, radius, float(mean[1]), conditional_slope, conditional_sigma_y)
    integration_limit = max(100, int(radial_order) + int(angular_order))

    def integrate(*, absolute_tolerance: float, relative_tolerance: float, limit: int) -> tuple[float, float]:
        if not lower_z < upper_z:
            return 0.0, float(2.0 * ndtr(-standard_tail_limit))
        result = quad(
            callback,
            lower_z,
            upper_z,
            epsabs=absolute_tolerance,
            epsrel=relative_tolerance,
            limit=limit,
            full_output=1,
            **({"args": callback_args} if backend == "rust" else {}),
        )
        if len(result) != 3:
            message = str(result[3]) if len(result) > 3 else "adaptive integration did not converge"
            raise ConjunctionProbabilityError(f"2D Pc integration failed closed: {message}")
        value, estimated_error, _ = result
        if not math.isfinite(value) or not math.isfinite(estimated_error):
            raise ConjunctionProbabilityError("2D Pc integration produced non-finite value or error evidence.")
        return float(value), float(estimated_error + 2.0 * ndtr(-standard_tail_limit))

    probability, fine_error = integrate(
        absolute_tolerance=1.0e-13,
        relative_tolerance=1.0e-10,
        limit=integration_limit,
    )
    coarse, coarse_error = integrate(
        absolute_tolerance=1.0e-10,
        relative_tolerance=1.0e-8,
        limit=max(50, integration_limit // 2),
    )
    convergence = max(abs(probability - coarse), fine_error)
    acceptance_tolerance = max(5.0e-12, 5.0e-9 * abs(probability))
    if convergence > acceptance_tolerance:
        raise ConjunctionProbabilityError(
            "2D Pc integration failed closed because the adaptive convergence estimate "
            f"{convergence} exceeds {acceptance_tolerance}."
        )
    if probability < -acceptance_tolerance or probability > 1.0 + acceptance_tolerance:
        raise ConjunctionProbabilityError("2D Pc integration produced a probability outside [0, 1].")
    probability = float(min(max(probability, 0.0), 1.0))
    return {
        "collision_probability": probability,
        **({"numeric_backend": backend} if backend == "rust" else {}),
        "method": "foster_2d_gaussian_disk_conditional_adaptive",
        "hard_body_radius_km": radius,
        "hard_body_radius_m": radius * 1000.0,
        "quadrature": {
            "algorithm": "standardized_conditional_gaussian_scipy_quad",
            "legacy_work_controls": {
                "radial_order": radial_order,
                "angular_order": angular_order,
            },
            "adaptive_subinterval_limit": integration_limit,
            "coarse_probability": coarse,
            "fine_error_estimate": fine_error,
            "coarse_error_estimate": coarse_error,
            "absolute_convergence_estimate": convergence,
            "acceptance_tolerance": acceptance_tolerance,
            "standard_normal_tail_limit": standard_tail_limit,
        },
        "assumptions": [
            "linear relative motion through the encounter",
            "independent primary and secondary Gaussian state errors",
            "position uncertainty projected onto the plane normal to relative velocity",
            "circular combined hard-body region",
        ],
        "qualification": "educational screening evidence; not operational maneuver authorization",
    }


def small_object_collision_probability(
    mean_plane_km: Sequence[float], covariance_plane_km2: Sequence[Sequence[float]], hard_body_radius_km: float
) -> float:
    """Legacy density-at-origin approximation retained for scenario compatibility."""

    mean = np.asarray(mean_plane_km, dtype=float)
    if mean.shape != (2,) or not np.all(np.isfinite(mean)):
        raise ConjunctionProbabilityError("Encounter-plane mean must be a finite two-vector.")
    covariance = validate_covariance(covariance_plane_km2, dimension=2)
    radius = float(hard_body_radius_km)
    if not math.isfinite(radius) or radius <= 0.0:
        raise ConjunctionProbabilityError("Hard-body radius must be positive and finite.")
    determinant = float(np.linalg.det(covariance))
    if determinant <= 0.0:
        raise ConjunctionProbabilityError("Encounter-plane covariance must be positive definite.")
    exponent = -0.5 * float(mean.T @ np.linalg.inv(covariance) @ mean)
    if exponent < -745.0:
        return 0.0
    density = math.exp(exponent) / (2.0 * math.pi * math.sqrt(determinant))
    return float(min(max(math.pi * radius**2 * density, 0.0), 1.0))


__all__ = [
    "ConjunctionProbabilityError",
    "collision_probability_2d",
    "covariance_rtn_si_to_eci_km",
    "project_combined_covariance",
    "ric_basis",
    "small_object_collision_probability",
    "validate_covariance",
]
