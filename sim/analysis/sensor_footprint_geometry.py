"""Rich hard-FOV, surface-service, and WGS84 boundary geometry."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sim.utils.geodesy import WGS84_A_KM, WGS84_B_KM

SUPPORTED_HARD_FOV_KINDS = frozenset(
    {
        "axisymmetric_hard_cone",
        "rectangular_hard_fov",
        "pushbroom_hard_fov",
    }
)
PRIMARY_REASON_NAMES = (
    "available",
    "earth_blocked",
    "outside_pattern",
    "outside_range",
    "off_nadir_exceeded",
    "incidence_exceeded",
    "illumination_rejected",
)


@dataclass(frozen=True)
class HardFOVPattern:
    """A hard-edged sensor pattern using +Z boresight and +X/+Y axes."""

    kind: str
    x_half_angle_rad: float
    y_half_angle_rad: float

    def __post_init__(self) -> None:
        kind = str(self.kind or "").strip().lower()
        if kind not in SUPPORTED_HARD_FOV_KINDS:
            choices = ", ".join(sorted(SUPPORTED_HARD_FOV_KINDS))
            raise ValueError(f"FOV kind must be one of: {choices}.")
        x_angle = float(self.x_half_angle_rad)
        y_angle = float(self.y_half_angle_rad)
        for value, label in ((x_angle, "x_half_angle_rad"), (y_angle, "y_half_angle_rad")):
            if not np.isfinite(value) or not 0.0 < value < 0.5 * np.pi:
                raise ValueError(f"{label} must be finite and strictly within (0, pi/2).")
        if kind == "axisymmetric_hard_cone" and x_angle != y_angle:
            raise ValueError("Axisymmetric cone x and y half-angles must be identical.")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "x_half_angle_rad", x_angle)
        object.__setattr__(self, "y_half_angle_rad", y_angle)

    @classmethod
    def axisymmetric_cone(cls, half_angle_rad: float) -> HardFOVPattern:
        return cls("axisymmetric_hard_cone", half_angle_rad, half_angle_rad)

    @classmethod
    def rectangular(
        cls,
        horizontal_half_angle_rad: float,
        vertical_half_angle_rad: float,
    ) -> HardFOVPattern:
        return cls(
            "rectangular_hard_fov",
            horizontal_half_angle_rad,
            vertical_half_angle_rad,
        )

    @classmethod
    def pushbroom(
        cls,
        cross_track_half_angle_rad: float,
        along_track_half_angle_rad: float,
    ) -> HardFOVPattern:
        return cls(
            "pushbroom_hard_fov",
            cross_track_half_angle_rad,
            along_track_half_angle_rad,
        )


@dataclass(frozen=True)
class SurfaceServiceConstraints:
    maximum_target_off_nadir_rad: float | None = None
    maximum_incidence_rad: float | None = None
    minimum_sun_elevation_rad: float | None = None
    maximum_sun_elevation_rad: float | None = None

    def __post_init__(self) -> None:
        for field_name in ("maximum_target_off_nadir_rad", "maximum_incidence_rad"):
            value = getattr(self, field_name)
            if value is None:
                continue
            normalized = float(value)
            if not np.isfinite(normalized) or not 0.0 <= normalized <= 0.5 * np.pi:
                raise ValueError(f"{field_name} must be finite and within [0, pi/2].")
            object.__setattr__(self, field_name, normalized)
        for field_name in ("minimum_sun_elevation_rad", "maximum_sun_elevation_rad"):
            value = getattr(self, field_name)
            if value is None:
                continue
            normalized = float(value)
            if not np.isfinite(normalized) or not -0.5 * np.pi <= normalized <= 0.5 * np.pi:
                raise ValueError(f"{field_name} must be finite and within [-pi/2, pi/2].")
            object.__setattr__(self, field_name, normalized)
        if (
            self.minimum_sun_elevation_rad is not None
            and self.maximum_sun_elevation_rad is not None
            and self.minimum_sun_elevation_rad > self.maximum_sun_elevation_rad
        ):
            raise ValueError("minimum_sun_elevation_rad must not exceed maximum_sun_elevation_rad.")

    @property
    def illumination_enabled(self) -> bool:
        return self.minimum_sun_elevation_rad is not None or self.maximum_sun_elevation_rad is not None


@dataclass(frozen=True)
class RichSurfaceTargetGeometry:
    range_km: np.ndarray
    sensor_horizontal_angle_rad: np.ndarray
    sensor_vertical_angle_rad: np.ndarray
    sensor_off_axis_angle_rad: np.ndarray
    target_off_nadir_angle_rad: np.ndarray
    incidence_angle_rad: np.ndarray
    sun_elevation_rad: np.ndarray
    visible: np.ndarray
    inside_pattern: np.ndarray
    inside_range: np.ndarray
    inside_off_nadir: np.ndarray
    inside_incidence: np.ndarray
    inside_illumination: np.ndarray
    available: np.ndarray
    primary_reason_code: np.ndarray


@dataclass(frozen=True)
class WGS84RayIntersections:
    hit: np.ndarray
    distance_km: np.ndarray
    point_ecef_km: np.ndarray


def _unit_rows(values: np.ndarray, field_name: str) -> tuple[np.ndarray, np.ndarray]:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or array.shape[1] != 3 or not np.all(np.isfinite(array)):
        raise ValueError(f"{field_name} must be finite with shape (items, 3).")
    norms = np.linalg.norm(array, axis=1)
    if np.any(norms <= 0.0) or not np.all(np.isfinite(norms)):
        raise ValueError(f"{field_name} vectors must have positive finite norm.")
    return array / norms[:, None], norms


def evaluate_rich_surface_targets_ecef(
    *,
    observer_ecef_km: np.ndarray,
    target_ecef_km: np.ndarray,
    target_outward_normal_ecef: np.ndarray,
    dcm_sensor_from_ecef: np.ndarray,
    pattern: HardFOVPattern,
    constraints: SurfaceServiceConstraints | None = None,
    max_range_km: float | None = None,
    sun_ecef_km: np.ndarray | None = None,
    angular_tolerance_rad: float = 1.0e-12,
    range_tolerance_km: float = 1.0e-9,
) -> RichSurfaceTargetGeometry:
    """Evaluate rich sampled surface-target gates with ordered dispositions."""

    constraints = constraints or SurfaceServiceConstraints()
    observer = np.asarray(observer_ecef_km, dtype=float).reshape(3)
    targets = np.asarray(target_ecef_km, dtype=float)
    normals = np.asarray(target_outward_normal_ecef, dtype=float)
    rotation = np.asarray(dcm_sensor_from_ecef, dtype=float)
    if not np.all(np.isfinite(observer)):
        raise ValueError("observer_ecef_km must contain three finite values.")
    if targets.ndim != 2 or targets.shape[1] != 3 or not np.all(np.isfinite(targets)):
        raise ValueError("target_ecef_km must be finite with shape (targets, 3).")
    if normals.shape != targets.shape or not np.all(np.isfinite(normals)):
        raise ValueError("target_outward_normal_ecef must be finite and match target_ecef_km.")
    if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
        raise ValueError("dcm_sensor_from_ecef must be a finite 3x3 matrix.")
    if not np.allclose(rotation @ rotation.T, np.eye(3), rtol=0.0, atol=1.0e-10):
        raise ValueError("dcm_sensor_from_ecef must be orthonormal within 1e-10.")
    if np.linalg.det(rotation) <= 0.0:
        raise ValueError("dcm_sensor_from_ecef must be a proper rotation.")
    normal_norms = np.linalg.norm(normals, axis=1)
    if np.any(np.abs(normal_norms - 1.0) > 1.0e-10):
        raise ValueError("Target outward normals must be unit vectors within 1e-10.")
    if max_range_km is not None and (
        not np.isfinite(float(max_range_km)) or float(max_range_km) <= 0.0
    ):
        raise ValueError("max_range_km must be positive and finite when provided.")
    tolerance = float(angular_tolerance_rad)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("angular_tolerance_rad must be finite and non-negative.")

    direction_ecef, ranges = _unit_rows(targets - observer, "observer-to-target vectors")
    direction_sensor = direction_ecef @ rotation.T
    horizontal = np.arctan2(direction_sensor[:, 0], direction_sensor[:, 2])
    vertical = np.arctan2(direction_sensor[:, 1], direction_sensor[:, 2])
    off_axis = np.arccos(np.clip(direction_sensor[:, 2], -1.0, 1.0))
    if pattern.kind == "axisymmetric_hard_cone":
        inside_pattern = off_axis <= pattern.x_half_angle_rad + tolerance
    else:
        inside_pattern = (
            (direction_sensor[:, 2] > 0.0)
            & (np.abs(horizontal) <= pattern.x_half_angle_rad + tolerance)
            & (np.abs(vertical) <= pattern.y_half_angle_rad + tolerance)
        )

    horizon_clearance = np.einsum("ij,ij->i", normals, observer - targets)
    visible = horizon_clearance > float(range_tolerance_km)
    if max_range_km is None:
        inside_range = np.ones(ranges.shape, dtype=bool)
    else:
        inside_range = ranges <= float(max_range_km) + float(range_tolerance_km)

    observer_normal = np.array(
        [
            observer[0] / (WGS84_A_KM**2),
            observer[1] / (WGS84_A_KM**2),
            observer[2] / (WGS84_B_KM**2),
        ],
        dtype=float,
    )
    observer_normal /= np.linalg.norm(observer_normal)
    nadir_ecef = -observer_normal
    target_off_nadir = np.arccos(np.clip(direction_ecef @ nadir_ecef, -1.0, 1.0))
    incidence = np.arccos(
        np.clip(np.einsum("ij,ij->i", normals, -direction_ecef), -1.0, 1.0)
    )
    if constraints.maximum_target_off_nadir_rad is None:
        inside_off_nadir = np.ones(ranges.shape, dtype=bool)
    else:
        inside_off_nadir = target_off_nadir <= constraints.maximum_target_off_nadir_rad + tolerance
    if constraints.maximum_incidence_rad is None:
        inside_incidence = np.ones(ranges.shape, dtype=bool)
    else:
        inside_incidence = incidence <= constraints.maximum_incidence_rad + tolerance

    sun_elevation = np.full(ranges.shape, np.nan, dtype=float)
    inside_illumination = np.ones(ranges.shape, dtype=bool)
    if constraints.illumination_enabled:
        if sun_ecef_km is None:
            raise ValueError("Explicit sun_ecef_km is required by illumination constraints.")
        sun = np.asarray(sun_ecef_km, dtype=float).reshape(3)
        if not np.all(np.isfinite(sun)):
            raise ValueError("sun_ecef_km must contain three finite values.")
        sun_direction, _ = _unit_rows(sun[None, :] - targets, "target-to-Sun vectors")
        sun_elevation = np.arcsin(
            np.clip(np.einsum("ij,ij->i", normals, sun_direction), -1.0, 1.0)
        )
        if constraints.minimum_sun_elevation_rad is not None:
            inside_illumination &= (
                sun_elevation >= constraints.minimum_sun_elevation_rad - tolerance
            )
        if constraints.maximum_sun_elevation_rad is not None:
            inside_illumination &= (
                sun_elevation <= constraints.maximum_sun_elevation_rad + tolerance
            )

    gates = (
        visible,
        inside_pattern,
        inside_range,
        inside_off_nadir,
        inside_incidence,
        inside_illumination,
    )
    primary_reason = np.zeros(ranges.shape, dtype=np.uint8)
    unresolved = np.ones(ranges.shape, dtype=bool)
    for reason_code, gate in enumerate(gates, start=1):
        failed = unresolved & (~gate)
        primary_reason[failed] = reason_code
        unresolved &= gate
    return RichSurfaceTargetGeometry(
        range_km=ranges,
        sensor_horizontal_angle_rad=horizontal,
        sensor_vertical_angle_rad=vertical,
        sensor_off_axis_angle_rad=off_axis,
        target_off_nadir_angle_rad=target_off_nadir,
        incidence_angle_rad=incidence,
        sun_elevation_rad=sun_elevation,
        visible=visible,
        inside_pattern=inside_pattern,
        inside_range=inside_range,
        inside_off_nadir=inside_off_nadir,
        inside_incidence=inside_incidence,
        inside_illumination=inside_illumination,
        available=unresolved,
        primary_reason_code=primary_reason,
    )


def fov_boundary_rays_sensor(
    pattern: HardFOVPattern,
    *,
    samples_per_edge: int,
) -> np.ndarray:
    """Return ordered unit sensor-frame rays around a hard FOV boundary."""

    if isinstance(samples_per_edge, (bool, np.bool_)) or int(samples_per_edge) != samples_per_edge:
        raise ValueError("samples_per_edge must be an integer.")
    count = int(samples_per_edge)
    if not 2 <= count <= 4096:
        raise ValueError("samples_per_edge must be within [2, 4096].")
    if pattern.kind == "axisymmetric_hard_cone":
        azimuth = np.linspace(0.0, 2.0 * np.pi, 4 * count, endpoint=False)
        rays = np.column_stack(
            (
                np.sin(pattern.x_half_angle_rad) * np.cos(azimuth),
                np.sin(pattern.x_half_angle_rad) * np.sin(azimuth),
                np.full(azimuth.shape, np.cos(pattern.x_half_angle_rad)),
            )
        )
        return rays

    x_limit = np.tan(pattern.x_half_angle_rad)
    y_limit = np.tan(pattern.y_half_angle_rad)
    top_x = np.linspace(-x_limit, x_limit, count, endpoint=False)
    right_y = np.linspace(y_limit, -y_limit, count, endpoint=False)
    bottom_x = np.linspace(x_limit, -x_limit, count, endpoint=False)
    left_y = np.linspace(-y_limit, y_limit, count, endpoint=False)
    rays = np.vstack(
        (
            np.column_stack((top_x, np.full(count, y_limit), np.ones(count))),
            np.column_stack((np.full(count, x_limit), right_y, np.ones(count))),
            np.column_stack((bottom_x, np.full(count, -y_limit), np.ones(count))),
            np.column_stack((np.full(count, -x_limit), left_y, np.ones(count))),
        )
    )
    return rays / np.linalg.norm(rays, axis=1)[:, None]


def intersect_rays_wgs84(
    observer_ecef_km: np.ndarray,
    direction_ecef: np.ndarray,
    *,
    discriminant_tolerance: float = 1.0e-12,
    distance_tolerance_km: float = 1.0e-9,
    numeric_backend: str = "rust",
) -> WGS84RayIntersections:
    """Intersect rays with WGS84 and retain each nearest positive solution."""

    backend = str(numeric_backend).strip().lower()
    if backend not in {"python", "rust"}:
        raise ValueError(f"Unknown coverage numeric backend {numeric_backend!r}.")
    observer = np.asarray(observer_ecef_km, dtype=float).reshape(3)
    directions, _ = _unit_rows(direction_ecef, "direction_ecef")
    discriminant_tolerance = float(discriminant_tolerance)
    distance_tolerance_km = float(distance_tolerance_km)
    if not np.isfinite(discriminant_tolerance) or discriminant_tolerance < 0.0:
        raise ValueError("discriminant_tolerance must be finite and non-negative.")
    if not np.isfinite(distance_tolerance_km) or distance_tolerance_km < 0.0:
        raise ValueError("distance_tolerance_km must be finite and non-negative.")
    if not np.all(np.isfinite(observer)):
        raise ValueError("observer_ecef_km must contain three finite values.")
    axes = np.array([WGS84_A_KM, WGS84_A_KM, WGS84_B_KM], dtype=float)
    observer_scaled = observer / axes
    if float(np.dot(observer_scaled, observer_scaled)) <= 1.0:
        raise ValueError("Ray observer must be outside the WGS84 ellipsoid.")
    if backend == "rust":
        from sim.rust_coverage_backend import intersect_rays_wgs84 as rust_intersect_rays_wgs84

        return rust_intersect_rays_wgs84(
            observer,
            directions,
            discriminant_tolerance=discriminant_tolerance,
            distance_tolerance_km=distance_tolerance_km,
        )

    direction_scaled = directions / axes[None, :]
    quadratic = np.einsum("ij,ij->i", direction_scaled, direction_scaled)
    linear = direction_scaled @ observer_scaled
    constant = float(np.dot(observer_scaled, observer_scaled) - 1.0)
    discriminant = linear * linear - quadratic * constant
    candidate = discriminant >= -discriminant_tolerance
    root = np.sqrt(np.clip(discriminant, 0.0, None))
    first = (-linear - root) / quadratic
    second = (-linear + root) / quadratic
    first_valid = first > distance_tolerance_km
    second_valid = second > distance_tolerance_km
    distance = np.where(first_valid, first, np.where(second_valid, second, np.nan))
    hit = candidate & (first_valid | second_valid)
    distance = np.where(hit, distance, np.nan)
    points = observer[None, :] + distance[:, None] * directions
    points[~hit] = np.nan
    return WGS84RayIntersections(
        hit=hit,
        distance_km=distance,
        point_ecef_km=points,
    )


__all__ = [
    "PRIMARY_REASON_NAMES",
    "SUPPORTED_HARD_FOV_KINDS",
    "HardFOVPattern",
    "RichSurfaceTargetGeometry",
    "SurfaceServiceConstraints",
    "WGS84RayIntersections",
    "evaluate_rich_surface_targets_ecef",
    "fov_boundary_rays_sensor",
    "intersect_rays_wgs84",
]
