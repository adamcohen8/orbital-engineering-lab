"""Recheck a geometry qualification against its source mesh at acceptance."""

from __future__ import annotations

import math

import numpy as np

from sim.dynamics.spacecraft_geometry import GeometryAreaProfile, fibonacci_sphere_directions, load_stl_triangles

from . import geometry as geometry_builder
from .contracts import contained, file_digest, physical_output_path, positive, rotation, vector

_ERROR_KEYS = (
    "raster_error_normalized",
    "interpolation_error_normalized",
    "raster_moment_error_normalized",
    "interpolation_moment_error_normalized",
)


def _matches(actual, expected):
    return math.isfinite(float(actual)) and math.isclose(float(actual), float(expected), rel_tol=1e-9, abs_tol=1e-12)


def verify_geometry_qualification(model, root, report, sources):
    """Require reproducible held-out evidence, not a self-declared pass flag."""
    geometry = model["geometry"]
    identity = report.get("identity")
    if report.get("schema") != "oel.twin_geometry_qualification.v1" or not isinstance(identity, dict):
        raise ValueError("Geometry qualification has an invalid schema or identity.")
    if report.get("profile_sha256") != geometry["sha256"]:
        raise ValueError("Geometry qualification is for a different profile.")
    if identity.get("algorithm") != "silhouette_raster_union_v1" or identity.get("implementation_sha256") != file_digest(
        geometry_builder.__file__
    ):
        raise ValueError("Geometry qualification implementation changed; regenerate its evidence.")
    if identity.get("configuration") != geometry["configuration"]:
        raise ValueError("Geometry qualification configuration changed.")
    sample_count, resolution = identity.get("sample_count"), identity.get("resolution")
    if (
        isinstance(sample_count, bool) or not isinstance(sample_count, int) or not 12 <= sample_count <= 2048
        or isinstance(resolution, bool) or not isinstance(resolution, int) or not 16 <= resolution <= 512
    ):
        raise ValueError("Geometry qualification has invalid sampling settings.")
    scale = positive(identity.get("scale_to_m"))
    tolerance = positive(identity.get("tolerance"))
    if tolerance > 0.25 or not _matches(report.get("tolerance"), tolerance):
        raise ValueError("Geometry qualification tolerance is invalid.")
    transform = rotation(identity.get("rotation_to_body"))
    translation = vector(identity.get("translation_body_m"))
    mesh_source = next((source for source in sources.values() if source["sha256"] == identity.get("mesh_sha256")), None)
    if mesh_source is None:
        raise ValueError("Geometry qualification mesh is not a verified twin source.")
    mesh_path = physical_output_path(contained(root, mesh_source["path"]))
    if mesh_path.stat().st_size > 32 * 1024 * 1024:
        raise ValueError("Geometry qualification mesh exceeds 32 MiB.")
    profile_path = physical_output_path(contained(root, geometry["profile_path"]))
    profile = GeometryAreaProfile.load(profile_path)
    if profile.metadata != identity or not np.allclose(
        profile.directions_body, fibonacci_sphere_directions(sample_count), atol=1e-12, rtol=0
    ):
        raise ValueError("Geometry profile does not match qualification settings.")
    triangles, _ = load_stl_triangles(mesh_path)
    if len(triangles) > 100000 or not np.all(np.isfinite(triangles)):
        raise ValueError("Geometry qualification mesh is invalid.")
    triangles = (triangles * scale) @ transform.T + translation
    normalizer = max(float(profile.projected_area_m2.max()), 1e-12)
    mesh_extent = max(float(np.linalg.norm(np.ptp(triangles.reshape(-1, 3), axis=0))), 1e-12)
    directions = fibonacci_sphere_directions(31, include_body_axes=False)
    rows = report.get("heldout")
    if not isinstance(rows, list) or len(rows) != len(directions):
        raise ValueError("Geometry qualification lacks the held-out reference rows.")
    maxima = dict.fromkeys(_ERROR_KEYS, 0.0)
    for direction, row in zip(directions, rows, strict=True):
        if not isinstance(row, dict) or set(row) != {"direction_body", "reference_area_m2", *_ERROR_KEYS}:
            raise ValueError("Geometry qualification held-out row is invalid.")
        if not np.allclose(vector(row["direction_body"]), direction, atol=1e-12, rtol=0):
            raise ValueError("Geometry qualification held-out directions changed.")
        coarse, coarse_cp = geometry_builder.silhouette(
            triangles, direction, resolution, numeric_backend=identity.get("numeric_backend", "python")
        )
        fine, fine_cp = geometry_builder.silhouette(
            triangles, direction, resolution * 2, numeric_backend=identity.get("numeric_backend", "python")
        )
        estimate = profile.lookup(direction)
        expected = {
            "reference_area_m2": fine,
            "raster_error_normalized": abs(coarse - fine) / normalizer,
            "interpolation_error_normalized": abs(estimate.projected_area_m2 - fine) / normalizer,
            "raster_moment_error_normalized": float(
                np.linalg.norm(np.cross(coarse * coarse_cp - fine * fine_cp, direction))
            ) / (normalizer * mesh_extent),
            "interpolation_moment_error_normalized": float(
                np.linalg.norm(np.cross(
                    estimate.projected_area_m2 * estimate.center_of_pressure_body_m - fine * fine_cp, direction
                ))
            ) / (normalizer * mesh_extent),
        }
        if any(not _matches(row[key], value) for key, value in expected.items()):
            raise ValueError("Geometry qualification held-out evidence does not reproduce from the mesh.")
        for key in _ERROR_KEYS:
            maxima[key] = max(maxima[key], expected[key])
    reported_maxima = report.get("max_errors")
    if not isinstance(reported_maxima, dict) or set(reported_maxima) != set(_ERROR_KEYS) or any(
        not _matches(reported_maxima[key], value) for key, value in maxima.items()
    ):
        raise ValueError("Geometry qualification maxima do not match held-out evidence.")
    if report.get("status") != "passed" or max(maxima.values()) > tolerance:
        raise ValueError("Geometry qualification does not pass its declared tolerance.")
