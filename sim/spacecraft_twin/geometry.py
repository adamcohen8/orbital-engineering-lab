"""Offline occlusion-aware projected silhouettes, bounded raster convergence and resumable profiles.

Union of projected triangles counts each occupied pixel once. Centroids are in
an incident-normal plane through the body origin; displacement parallel to the
incident force does not affect the uniform-pressure torque. No material BRDF,
articulation, or aerodynamic coefficient is inferred from the mesh.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from sim.dynamics.spacecraft_geometry import GeometryAreaProfile, fibonacci_sphere_directions, load_stl_triangles
from sim.security.output_writes import current_output_writer, mkdir_output, open_output, staged_output_file

from .contracts import (
    digest,
    file_digest,
    load,
    physical_output_path,
    positive,
    rotation,
    vector,
    write_json,
)


def silhouette(triangles, direction, resolution=128, *, numeric_backend="rust"):
    triangles = np.asarray(triangles, dtype=float)
    if triangles.ndim != 3 or triangles.shape[1:] != (3, 3) or not np.all(np.isfinite(triangles)):
        raise ValueError("Expected finite Nx3x3 mesh triangles.")
    if len(triangles) > 100000:
        raise ValueError("Mesh exceeds 100000 triangles.")
    if not 16 <= resolution <= 1024:
        raise ValueError("Raster resolution must be 16..1024.")
    if numeric_backend not in {"python", "rust"}:
        raise ValueError("spacecraft geometry numeric_backend must be python or rust")
    if numeric_backend == "rust":
        from sim.rust_spacecraft_backend import silhouette_batch as rust_silhouette_batch

        areas, cps = rust_silhouette_batch(triangles, np.asarray([direction], dtype=float), resolution)
        return float(areas[0]), np.asarray(cps[0], dtype=float)
    u = vector(direction)
    if np.linalg.norm(u) <= 0:
        raise ValueError("Direction cannot be zero.")
    u = u / np.linalg.norm(u)
    helper = np.eye(3)[np.argmin(np.abs(u))]
    x = np.cross(u, helper)
    x /= np.linalg.norm(x)
    y = np.cross(u, x)
    points = np.stack((triangles @ x, triangles @ y), axis=-1)
    lo, hi = points.min(axis=(0, 1)), points.max(axis=(0, 1))
    width = hi - lo
    if np.min(width) <= 1e-14:
        return 0.0, np.zeros(3)
    step = width / resolution
    occupied = np.zeros((resolution, resolution), dtype=bool)
    for tri in points:
        a, b, c = tri
        denominator = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
        if abs(denominator) <= 1e-20:
            continue
        start = np.maximum(0, np.floor((tri.min(axis=0) - lo) / step).astype(int))
        end = np.minimum(resolution, np.ceil((tri.max(axis=0) - lo) / step).astype(int))
        if np.any(end <= start):
            continue
        xx = lo[0] + (np.arange(start[0], end[0]) + 0.5) * step[0]
        yy = lo[1] + (np.arange(start[1], end[1]) + 0.5) * step[1]
        xx, yy = np.meshgrid(xx, yy, indexing="ij")
        aa = ((b[1] - c[1]) * (xx - c[0]) + (c[0] - b[0]) * (yy - c[1])) / denominator
        bb = ((c[1] - a[1]) * (xx - c[0]) + (a[0] - c[0]) * (yy - c[1])) / denominator
        occupied[start[0] : end[0], start[1] : end[1]] |= (aa >= -1e-12) & (bb >= -1e-12) & (aa + bb <= 1 + 1e-12)
    indices = np.argwhere(occupied)
    if not len(indices):
        return 0.0, np.zeros(3)
    centroid = lo + (indices.mean(axis=0) + 0.5) * step
    return float(len(indices) * np.prod(step)), centroid[0] * x + centroid[1] * y


def _build_profile(
    mesh_path,
    output_dir,
    *,
    sample_count=128,
    resolution=128,
    scale_to_m=1.0,
    rotation_to_body=None,
    translation_body_m=None,
    configuration="fixed",
    tolerance=0.05,
    numeric_backend="rust",
    resume=False,
    cancellation_check=lambda: None,
    progress=None,
):
    if not 12 <= sample_count <= 2048 or not 16 <= resolution <= 512:
        raise ValueError("Samples must be 12..2048 and resolution 16..512 (reference uses 2x resolution).")
    if numeric_backend not in {"python", "rust"}:
        raise ValueError("spacecraft geometry numeric_backend must be python or rust")
    scale = positive(scale_to_m)
    tolerance = positive(tolerance)
    if tolerance > 0.25:
        raise ValueError("Qualification tolerance must be <= 0.25.")
    r = rotation(np.eye(3) if rotation_to_body is None else rotation_to_body)
    t = vector([0, 0, 0] if translation_body_m is None else translation_body_m)
    mesh_path, out = Path(mesh_path), Path(output_dir)
    identity = {
        "mesh_sha256": file_digest(mesh_path),
        "sample_count": sample_count,
        "resolution": resolution,
        "scale_to_m": scale,
        "rotation_to_body": r.tolist(),
        "translation_body_m": t.tolist(),
        "configuration": configuration,
        "tolerance": tolerance,
        "algorithm": "silhouette_raster_union_v1",
        "implementation_sha256": file_digest(__file__),
    }
    # Keep the historical Python identity shape stable.  Native selection is
    # recorded because it affects the numerical implementation; a missing
    # field in an older checkpoint/profile therefore means the Python path.
    if numeric_backend == "rust":
        identity["numeric_backend"] = "rust"
    fingerprint = digest(identity)
    checkpoint = out / "checkpoint.json"
    if resume:
        physical_out = physical_output_path(out)
        if any(p.is_symlink() for p in physical_out.rglob("*")):
            raise ValueError("Geometry resume refuses symlink artifacts.")
        state = load(checkpoint)
        if state.get("samples_digest") != digest(state.get("samples")):
            raise ValueError("Geometry checkpoint sample identity changed.")
        if state["identity"] != fingerprint:
            raise ValueError("Geometry resume rejected: input/settings identity changed.")
    else:
        mkdir_output(out)
        state = {"identity": fingerprint, "samples": [], "status": "running"}
        state["samples_digest"] = digest(state["samples"])
        write_json(checkpoint, state)
    if mesh_path.stat().st_size > 32 * 1024 * 1024:
        raise ValueError("Mesh exceeds 32 MiB.")
    triangles, _ = load_stl_triangles(mesh_path)
    if len(triangles) > 100000 or not np.all(np.isfinite(triangles)):
        raise ValueError("Mesh must be finite and contain at most 100000 triangles.")
    if not np.any(
        np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1) > 0
    ):
        raise ValueError("Mesh contains no nondegenerate surface triangles.")
    triangles = (triangles * scale) @ r.T + t
    directions = fibonacci_sphere_directions(sample_count)
    try:
        start_index = len(state["samples"])
        if numeric_backend == "rust" and start_index < len(directions):
            from sim.rust_spacecraft_backend import silhouette_batch as rust_silhouette_batch

            pending_areas, pending_cps = rust_silhouette_batch(
                triangles,
                directions[start_index:],
                resolution,
            )
            for offset, (area, cp) in enumerate(zip(pending_areas, pending_cps)):
                i = start_index + offset
                cancellation_check()
                state["samples"].append({"area": float(area), "cp": np.asarray(cp).tolist()})
                state["samples_digest"] = digest(state["samples"])
                write_json(checkpoint, state)
                if progress:
                    progress(i + 1, len(directions), "Precomputing projected silhouette")
        else:
            for i in range(start_index, len(directions)):
                cancellation_check()
                area, cp = silhouette(triangles, directions[i], resolution, numeric_backend=numeric_backend)
                state["samples"].append({"area": area, "cp": cp.tolist()})
                state["samples_digest"] = digest(state["samples"])
                write_json(checkpoint, state)
                if progress:
                    progress(i + 1, len(directions), "Precomputing projected silhouette")
        profile = GeometryAreaProfile(
            directions,
            np.array([s["area"] for s in state["samples"]]),
            np.array([s["cp"] for s in state["samples"]]),
            metadata=identity,
        )
        profile_path = out / "profile.json"
        writer = current_output_writer()
        if writer is not None and physical_output_path(profile_path).exists():
            payload = (json.dumps(profile.to_dict(), indent=2, sort_keys=True) + "\n").encode("utf-8")
            writer.update_existing_bytes(profile_path.relative_to(Path(writer.logical_root)), payload)
        else:
            with staged_output_file(profile_path) as staged_profile_path:
                profile.save(staged_profile_path)
        # Independent directions avoid testing only table knots. Absolute errors
        # are normalized by maximum sampled area to remain meaningful edge-on.
        heldout = fibonacci_sphere_directions(31, include_body_axes=False)
        normalizer = max(float(profile.projected_area_m2.max()), 1e-12)
        mesh_extent = max(float(np.linalg.norm(np.ptp(triangles.reshape(-1, 3), axis=0))), 1e-12)
        rows = []
        for direction in heldout:
            cancellation_check()
            coarse, coarse_cp = silhouette(triangles, direction, resolution, numeric_backend=numeric_backend)
            fine, fine_cp = silhouette(triangles, direction, resolution * 2, numeric_backend=numeric_backend)
            estimate = profile.lookup(direction)
            rows.append(
                {
                    "direction_body": direction.tolist(),
                    "reference_area_m2": fine,
                    "raster_error_normalized": abs(coarse - fine) / normalizer,
                    "interpolation_error_normalized": abs(estimate.projected_area_m2 - fine) / normalizer,
                    "raster_moment_error_normalized": float(
                        np.linalg.norm(np.cross(coarse * coarse_cp - fine * fine_cp, direction))
                    )
                    / (normalizer * mesh_extent),
                    "interpolation_moment_error_normalized": float(
                        np.linalg.norm(
                            np.cross(
                                estimate.projected_area_m2 * estimate.center_of_pressure_body_m - fine * fine_cp,
                                direction,
                            )
                        )
                    )
                    / (normalizer * mesh_extent),
                }
            )
        maxima = {key: max(row[key] for row in rows) for key in rows[0] if key.endswith("_normalized")}
        report = {
            "schema": "oel.twin_geometry_qualification.v1",
            "status": "passed" if max(maxima.values()) <= tolerance else "failed",
            "profile_sha256": file_digest(profile_path),
            "identity": identity,
            "tolerance": tolerance,
            "max_errors": maxima,
            "heldout": rows,
            "limits": [
                "Finite held-out convergence evidence, not a global error bound.",
                "Uniform pressure silhouette; no material model or moving articulation.",
                "STL scale, orientation and component interpretation require source-backed review.",
            ],
        }
        write_json(out / "qualification.json", report)
        state["status"] = "completed"
        state["samples_digest"] = digest(state["samples"])
        write_json(checkpoint, state)
        return report
    except BaseException as exc:
        state["status"] = "interrupted"
        state["error"] = type(exc).__name__
        state["samples_digest"] = digest(state["samples"])
        write_json(checkpoint, state)
        raise


def build_profile(mesh_path, output_dir, **kwargs):
    """Serialize writers within the job directory; leave resumable data after interruption."""
    out = Path(output_dir)
    if current_output_writer() is None and out.exists() and not kwargs.get("resume", False):
        raise ValueError("Geometry output already exists; use resume for identical inputs.")
    mkdir_output(out)
    writer = current_output_writer()
    if writer is not None and kwargs.get("resume", False):
        for name in ("checkpoint.json", "profile.json", "qualification.json"):
            if physical_output_path(out / name).is_file():
                writer.adopt_existing(name)
    lock = out / ".geometry-writer.lock"
    try:
        with open_output(lock, "xb") as stream:
            stream.write(str(os.getpid()).encode())
    except FileExistsError as exc:
        raise ValueError(
            "Geometry job already has a writer; inspect the process before recovering a stale lock."
        ) from exc
    try:
        return _build_profile(mesh_path, out, **kwargs)
    finally:
        try:
            physical_output_path(lock).unlink()
        except FileNotFoundError:
            pass
