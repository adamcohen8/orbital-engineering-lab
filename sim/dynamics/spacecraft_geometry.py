from __future__ import annotations

import json
import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class RectangularPrismGeometry:
    lx_m: float
    ly_m: float
    lz_m: float

    def __post_init__(self) -> None:
        if self.lx_m <= 0.0 or self.ly_m <= 0.0 or self.lz_m <= 0.0:
            raise ValueError("Rectangular prism dimensions must be positive.")

    def face_centers_body_m(self) -> np.ndarray:
        return np.array(
            [
                [0.5 * self.lx_m, 0.0, 0.0],
                [-0.5 * self.lx_m, 0.0, 0.0],
                [0.0, 0.5 * self.ly_m, 0.0],
                [0.0, -0.5 * self.ly_m, 0.0],
                [0.0, 0.0, 0.5 * self.lz_m],
                [0.0, 0.0, -0.5 * self.lz_m],
            ],
            dtype=float,
        )

    def face_normals_body(self) -> np.ndarray:
        return np.array(
            [
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, -1.0],
            ],
            dtype=float,
        )

    def face_areas_m2(self) -> np.ndarray:
        a_x = self.ly_m * self.lz_m
        a_y = self.lx_m * self.lz_m
        a_z = self.lx_m * self.ly_m
        return np.array([a_x, a_x, a_y, a_y, a_z, a_z], dtype=float)

    def projected_area_m2(self, incident_dir_body: np.ndarray) -> float:
        u = np.array(incident_dir_body, dtype=float)
        n = float(np.linalg.norm(u))
        if n <= 0.0:
            return 0.0
        u = u / n
        normals = self.face_normals_body()
        areas = self.face_areas_m2()
        illum = np.maximum(0.0, -(normals @ u))
        return float(np.sum(areas * illum))

    def face_forces_body_n(self, incident_dir_body: np.ndarray, pressure_n_m2: float) -> np.ndarray:
        u = np.array(incident_dir_body, dtype=float)
        n = float(np.linalg.norm(u))
        if n <= 0.0 or pressure_n_m2 <= 0.0:
            return np.zeros((6, 3))
        u = u / n
        normals = self.face_normals_body()
        areas = self.face_areas_m2()
        illum = np.maximum(0.0, -(normals @ u))
        mags = pressure_n_m2 * areas * illum
        # Lumped absorber-model force follows the incoming momentum flux direction.
        return mags[:, None] * u[None, :]

    def face_torque_sum_body_nm(
        self,
        incident_dir_body: np.ndarray,
        pressure_n_m2: float,
        *,
        moment_origin_body_m: np.ndarray | None = None,
    ) -> np.ndarray:
        r_faces = self.face_centers_body_m()
        if moment_origin_body_m is not None:
            r_faces = r_faces - np.array(moment_origin_body_m, dtype=float).reshape(3)
        f_faces = self.face_forces_body_n(incident_dir_body, pressure_n_m2)
        tau_faces = np.cross(r_faces, f_faces)
        return np.sum(tau_faces, axis=0)


@dataclass(frozen=True)
class GeometryProfileLookup:
    projected_area_m2: float
    center_of_pressure_body_m: np.ndarray


@dataclass(frozen=True)
class GeometryAreaProfile:
    """Direction-dependent projected area profile derived from body-frame mesh facets.

    The profile is an offline approximation: each lookup samples precomputed
    facet projected area and projected-area-weighted centroid for the requested
    incoming flux direction. It intentionally does not ray-trace self-shadowing,
    articulation, or material differences.
    """

    directions_body: np.ndarray
    projected_area_m2: np.ndarray
    center_of_pressure_body_m: np.ndarray
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        directions = np.asarray(self.directions_body, dtype=float)
        areas = np.asarray(self.projected_area_m2, dtype=float)
        cps = np.asarray(self.center_of_pressure_body_m, dtype=float)
        if directions.ndim != 2 or directions.shape[1] != 3 or directions.shape[0] == 0:
            raise ValueError("Geometry area profile directions must be an Nx3 array.")
        if areas.shape != (directions.shape[0],):
            raise ValueError("Geometry area profile projected_area_m2 must have one value per direction.")
        if cps.shape != (directions.shape[0], 3):
            raise ValueError("Geometry area profile center_of_pressure_body_m must be an Nx3 array.")
        norms = np.linalg.norm(directions, axis=1)
        if np.any(norms <= 0.0) or not np.all(np.isfinite(norms)):
            raise ValueError("Geometry area profile directions must be finite nonzero vectors.")
        if np.any(~np.isfinite(areas)) or np.any(areas < 0.0):
            raise ValueError("Geometry area profile projected areas must be finite and nonnegative.")
        if np.any(~np.isfinite(cps)):
            raise ValueError("Geometry area profile centers of pressure must be finite.")
        object.__setattr__(self, "directions_body", directions / norms[:, None])
        object.__setattr__(self, "projected_area_m2", areas)
        object.__setattr__(self, "center_of_pressure_body_m", cps)
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    @classmethod
    def from_stl(
        cls,
        path: str | Path,
        *,
        sample_count: int = 642,
        include_body_axes: bool = True,
        metadata: dict[str, Any] | None = None,
        numeric_backend: str = "rust",
    ) -> GeometryAreaProfile:
        vertices, normals = load_stl_triangles(path)
        directions = fibonacci_sphere_directions(sample_count, include_body_axes=include_body_axes)
        profile_metadata = {
            "schema": "oel.geometry_area_profile.v1",
            "source_format": "stl",
            "source_path": str(Path(path).expanduser()),
            "triangle_count": int(vertices.shape[0]),
            "sample_count": int(directions.shape[0]),
            "method": "facet_projected_area_no_self_shadowing",
            "notes": [
                "Incoming directions are expressed in the spacecraft body frame.",
                "Projected areas use illuminated mesh facets only; no ray-traced self-shadowing is applied.",
                "Center of pressure is the projected-area-weighted centroid of illuminated facets.",
            ],
        }
        profile_metadata.update(dict(metadata or {}))
        return cls.from_triangles(
            vertices,
            normals,
            directions_body=directions,
            metadata=profile_metadata,
            numeric_backend=numeric_backend,
        )

    @classmethod
    def from_triangles(
        cls,
        vertices_body_m: np.ndarray,
        normals_body: np.ndarray | None = None,
        *,
        directions_body: np.ndarray,
        metadata: dict[str, Any] | None = None,
        numeric_backend: str = "rust",
    ) -> GeometryAreaProfile:
        triangles = np.asarray(vertices_body_m, dtype=float)
        if triangles.ndim != 3 or triangles.shape[1:] != (3, 3) or triangles.shape[0] == 0:
            raise ValueError("Mesh triangles must be an Nx3x3 array of body-frame vertices in meters.")
        edge_a = triangles[:, 1, :] - triangles[:, 0, :]
        edge_b = triangles[:, 2, :] - triangles[:, 0, :]
        area_vectors = 0.5 * np.cross(edge_a, edge_b)
        areas = np.linalg.norm(area_vectors, axis=1)
        keep = areas > 0.0
        if not np.any(keep):
            raise ValueError("Mesh does not contain any nondegenerate triangles.")
        triangles = triangles[keep]
        areas = areas[keep]
        area_vectors = area_vectors[keep]

        if normals_body is None:
            normals = area_vectors / areas[:, None]
        else:
            normals_raw = np.asarray(normals_body, dtype=float)
            if normals_raw.shape != (keep.shape[0], 3):
                raise ValueError("Mesh normals must be an Nx3 array matching the input triangles.")
            normals_raw = normals_raw[keep]
            normal_norms = np.linalg.norm(normals_raw, axis=1)
            normals = np.divide(
                normals_raw,
                normal_norms[:, None],
                out=np.zeros_like(normals_raw),
                where=normal_norms[:, None] > 0.0,
            )
            fallback = normal_norms <= 0.0
            if np.any(fallback):
                normals[fallback] = area_vectors[fallback] / areas[fallback, None]

        directions = np.asarray(directions_body, dtype=float)
        if directions.ndim != 2 or directions.shape[1] != 3:
            raise ValueError("Profile directions must be an Nx3 array of nonzero body-frame vectors.")
        direction_norms = np.linalg.norm(directions, axis=1)
        if np.any(~np.isfinite(direction_norms)) or np.any(direction_norms <= 0.0):
            raise ValueError("Profile directions must be an Nx3 array of nonzero body-frame vectors.")
        directions = directions / direction_norms[:, None]

        if numeric_backend not in {"python", "rust"}:
            raise ValueError("spacecraft geometry numeric_backend must be python or rust")
        if numeric_backend == "rust":
            from sim.rust_spacecraft_backend import facet_projected_geometry

            profile_areas, profile_cps = facet_projected_geometry(
                triangles,
                directions,
                normals_body=normals,
            )
            return cls(
                directions_body=directions,
                projected_area_m2=profile_areas,
                center_of_pressure_body_m=profile_cps,
                metadata=metadata,
            )

        centroids = np.mean(triangles, axis=1)
        profile_areas: list[float] = []
        profile_cps: list[np.ndarray] = []
        for direction in directions:
            weights = areas * np.maximum(0.0, -(normals @ direction))
            projected_area = float(np.sum(weights))
            profile_areas.append(projected_area)
            if projected_area > 0.0:
                profile_cps.append(np.sum(centroids * weights[:, None], axis=0) / projected_area)
            else:
                profile_cps.append(np.zeros(3))

        return cls(
            directions_body=directions,
            projected_area_m2=np.asarray(profile_areas, dtype=float),
            center_of_pressure_body_m=np.asarray(profile_cps, dtype=float),
            metadata=metadata,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GeometryAreaProfile:
        return cls(
            directions_body=np.asarray(data.get("directions_body"), dtype=float),
            projected_area_m2=np.asarray(data.get("projected_area_m2"), dtype=float),
            center_of_pressure_body_m=np.asarray(data.get("center_of_pressure_body_m"), dtype=float),
            metadata=dict(data.get("metadata", {}) or {}),
        )

    @classmethod
    def load(cls, path: str | Path) -> GeometryAreaProfile:
        with Path(path).expanduser().open("r", encoding="utf-8") as f:
            raw = json.load(f)
        if not isinstance(raw, dict):
            raise ValueError("Geometry area profile JSON root must be an object.")
        return cls.from_dict(raw)

    def to_dict(self) -> dict[str, Any]:
        return {
            "metadata": dict(self.metadata or {}),
            "directions_body": self.directions_body.tolist(),
            "projected_area_m2": self.projected_area_m2.tolist(),
            "center_of_pressure_body_m": self.center_of_pressure_body_m.tolist(),
        }

    def save(self, path: str | Path) -> Path:
        out = Path(path).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, sort_keys=True)
            f.write("\n")
        return out

    def lookup(
        self,
        incident_dir_body: np.ndarray,
        *,
        nearest_neighbors: int = 4,
        distance_power: float = 2.0,
    ) -> GeometryProfileLookup:
        u = np.asarray(incident_dir_body, dtype=float).reshape(3)
        n = float(np.linalg.norm(u))
        if n <= 0.0:
            return GeometryProfileLookup(projected_area_m2=0.0, center_of_pressure_body_m=np.zeros(3))
        u = u / n
        dots = self.directions_body @ u
        best = int(np.argmax(dots))
        if dots[best] >= 1.0 - 1e-12:
            return GeometryProfileLookup(
                projected_area_m2=float(self.projected_area_m2[best]),
                center_of_pressure_body_m=np.array(self.center_of_pressure_body_m[best], dtype=float),
            )
        k = int(max(1, min(nearest_neighbors, self.directions_body.shape[0])))
        idx = np.argpartition(-dots, k - 1)[:k]
        chord = np.linalg.norm(self.directions_body[idx] - u[None, :], axis=1)
        weights = 1.0 / np.maximum(chord, 1e-12) ** float(max(distance_power, 1e-9))
        weights = weights / np.sum(weights)
        weighted_areas = self.projected_area_m2[idx] * weights
        area = float(np.sum(weighted_areas))
        if area > 0.0:
            cp = np.sum(self.center_of_pressure_body_m[idx] * weighted_areas[:, None], axis=0) / area
        else:
            cp = np.sum(self.center_of_pressure_body_m[idx] * weights[:, None], axis=0)
        return GeometryProfileLookup(projected_area_m2=area, center_of_pressure_body_m=cp)

    def projected_area_for_direction_m2(self, incident_dir_body: np.ndarray) -> float:
        return float(self.lookup(incident_dir_body).projected_area_m2)

    def pressure_torque_sum_body_nm(
        self,
        incident_dir_body: np.ndarray,
        pressure_n_m2: float,
        *,
        moment_origin_body_m: np.ndarray | None = None,
    ) -> np.ndarray:
        if pressure_n_m2 <= 0.0:
            return np.zeros(3)
        lookup = self.lookup(incident_dir_body)
        if lookup.projected_area_m2 <= 0.0:
            return np.zeros(3)
        u = np.asarray(incident_dir_body, dtype=float).reshape(3)
        n = float(np.linalg.norm(u))
        if n <= 0.0:
            return np.zeros(3)
        force_body_n = float(pressure_n_m2) * lookup.projected_area_m2 * (u / n)
        arm_body_m = np.array(lookup.center_of_pressure_body_m, dtype=float)
        if moment_origin_body_m is not None:
            arm_body_m = arm_body_m - np.array(moment_origin_body_m, dtype=float).reshape(3)
        return np.cross(arm_body_m, force_body_n)


def fibonacci_sphere_directions(sample_count: int, *, include_body_axes: bool = True) -> np.ndarray:
    count = int(sample_count)
    if count <= 0:
        raise ValueError("sample_count must be positive.")
    directions: list[np.ndarray] = []
    if include_body_axes:
        directions.extend(
            [
                np.array([1.0, 0.0, 0.0]),
                np.array([-1.0, 0.0, 0.0]),
                np.array([0.0, 1.0, 0.0]),
                np.array([0.0, -1.0, 0.0]),
                np.array([0.0, 0.0, 1.0]),
                np.array([0.0, 0.0, -1.0]),
            ]
        )
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    for i in range(count):
        z = 1.0 - (2.0 * (i + 0.5) / count)
        radius = float(np.sqrt(max(0.0, 1.0 - z * z)))
        theta = golden_angle * i
        directions.append(np.array([np.cos(theta) * radius, np.sin(theta) * radius, z], dtype=float))
    return _dedupe_unit_directions(np.asarray(directions, dtype=float))


def load_stl_triangles(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    raw = Path(path).expanduser().read_bytes()
    if len(raw) >= 84:
        triangle_count = struct.unpack("<I", raw[80:84])[0]
        expected_size = 84 + 50 * int(triangle_count)
        if expected_size == len(raw):
            return _load_binary_stl(raw, triangle_count)
    return _load_ascii_stl(raw.decode("utf-8", errors="ignore"))


def _load_binary_stl(raw: bytes, triangle_count: int) -> tuple[np.ndarray, np.ndarray]:
    vertices = np.zeros((int(triangle_count), 3, 3), dtype=float)
    normals = np.zeros((int(triangle_count), 3), dtype=float)
    offset = 84
    for i in range(int(triangle_count)):
        values = struct.unpack("<12f", raw[offset : offset + 48])
        normals[i] = np.asarray(values[:3], dtype=float)
        vertices[i] = np.asarray(values[3:], dtype=float).reshape(3, 3)
        offset += 50
    return vertices, normals


def _load_ascii_stl(text: str) -> tuple[np.ndarray, np.ndarray]:
    facet_pattern = re.compile(
        r"facet\s+normal\s+([^\n]+?)\s+outer\s+loop\s+"
        r"vertex\s+([^\n]+?)\s+vertex\s+([^\n]+?)\s+vertex\s+([^\n]+?)\s+endloop\s+endfacet",
        flags=re.IGNORECASE | re.DOTALL,
    )
    vertices: list[np.ndarray] = []
    normals: list[np.ndarray] = []
    for match in facet_pattern.finditer(text):
        normals.append(_parse_stl_vector(match.group(1)))
        vertices.append(
            np.vstack(
                [
                    _parse_stl_vector(match.group(2)),
                    _parse_stl_vector(match.group(3)),
                    _parse_stl_vector(match.group(4)),
                ]
            )
        )
    if not vertices:
        raise ValueError("STL file does not contain any triangles.")
    return np.asarray(vertices, dtype=float), np.asarray(normals, dtype=float)


def _parse_stl_vector(text: str) -> np.ndarray:
    parts = [float(part) for part in text.strip().split()[:3]]
    if len(parts) != 3:
        raise ValueError(f"Invalid STL vector: {text!r}")
    return np.asarray(parts, dtype=float)


def _dedupe_unit_directions(directions: np.ndarray) -> np.ndarray:
    unit: list[np.ndarray] = []
    for direction in directions:
        n = float(np.linalg.norm(direction))
        if n <= 0.0:
            continue
        candidate = direction / n
        if not any(float(np.dot(candidate, existing)) > 1.0 - 1e-12 for existing in unit):
            unit.append(candidate)
    return np.asarray(unit, dtype=float)
