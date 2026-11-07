from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from sim.dynamics.orbit.frames import FrameContext, eci_to_ecef_rotation_context
from sim.numeric_backend import normalize_numeric_backend
from sim.utils.geodesy import geodetic_to_ecef_km


def _cadence_due(t_s: float, last_update_t_s: float, cadence_s: float) -> bool:
    """Allow roundoff at a due boundary without admitting extra samples."""
    if not math.isfinite(t_s):
        raise ValueError("sensor time_s must be finite.")
    elapsed = t_s - last_update_t_s
    if elapsed >= cadence_s:
        return True
    if elapsed <= 0.0 or not math.isfinite(elapsed):
        return False
    tolerance = min(
        cadence_s * 1.0e-9,
        4.0 * max(math.ulp(t_s), math.ulp(last_update_t_s), math.ulp(cadence_s)),
    )
    return cadence_s - elapsed <= tolerance


@dataclass(frozen=True)
class GroundSite:
    lat_rad: float
    lon_rad: float
    min_elevation_rad: float = 0.0
    alt_km: float = 0.0

    def __post_init__(self) -> None:
        normalized: dict[str, float] = {}
        for field_name in ("lat_rad", "lon_rad", "alt_km", "min_elevation_rad"):
            value = getattr(self, field_name)
            if isinstance(value, (bool, np.bool_)):
                raise ValueError("ground-site coordinates and elevation mask must be finite numeric values")
            try:
                parsed = float(value)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError("ground-site coordinates and elevation mask must be finite numeric values") from exc
            if not np.isfinite(parsed):
                raise ValueError("ground-site coordinates and elevation mask must be finite numeric values")
            normalized[field_name] = parsed
        for field_name, value in normalized.items():
            object.__setattr__(self, field_name, value)
        if not -0.5 * np.pi <= normalized["lat_rad"] <= 0.5 * np.pi:
            raise ValueError("ground-site latitude must be within [-pi/2, pi/2]")
        if not 0.0 <= normalized["min_elevation_rad"] <= 0.5 * np.pi:
            raise ValueError("ground-site minimum elevation must be within [0, pi/2]")


@dataclass(frozen=True)
class AccessConfig:
    update_cadence_s: float = 1.0
    max_range_km: float | None = None
    fov_half_angle_rad: float | None = None
    solid_angle_sr: float | None = None
    require_ground_visibility: bool = False
    ground_site: GroundSite | None = None
    frame_context: FrameContext | None = None
    numeric_backend: str = "rust"

    def __post_init__(self) -> None:
        if type(self.require_ground_visibility) is not bool:
            raise ValueError("require_ground_visibility must be a boolean.")
        if isinstance(self.update_cadence_s, (bool, np.bool_)):
            raise ValueError("update_cadence_s must be positive and finite.")
        try:
            update_cadence_s = float(self.update_cadence_s)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("update_cadence_s must be positive and finite.") from exc
        if not np.isfinite(update_cadence_s) or update_cadence_s <= 0.0:
            raise ValueError("update_cadence_s must be positive and finite.")
        object.__setattr__(self, "update_cadence_s", update_cadence_s)
        if self.max_range_km is not None:
            if isinstance(self.max_range_km, (bool, np.bool_)):
                raise ValueError("max_range_km must be positive and finite when provided.")
            try:
                max_range_km = float(self.max_range_km)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError("max_range_km must be positive and finite when provided.") from exc
            if not np.isfinite(max_range_km) or max_range_km <= 0.0:
                raise ValueError("max_range_km must be positive and finite when provided.")
            object.__setattr__(self, "max_range_km", max_range_km)
        if self.fov_half_angle_rad is not None and self.solid_angle_sr is not None:
            raise ValueError(
                "fov_half_angle_rad and solid_angle_sr are mutually exclusive; configure exactly one."
            )
        if self.fov_half_angle_rad is not None:
            if isinstance(self.fov_half_angle_rad, (bool, np.bool_)):
                raise ValueError("fov_half_angle_rad must be finite and within [0, pi].")
            try:
                fov_half_angle_rad = float(self.fov_half_angle_rad)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError("fov_half_angle_rad must be finite and within [0, pi].") from exc
            if not np.isfinite(fov_half_angle_rad) or not 0.0 <= fov_half_angle_rad <= np.pi:
                raise ValueError("fov_half_angle_rad must be finite and within [0, pi].")
            object.__setattr__(self, "fov_half_angle_rad", fov_half_angle_rad)
        if self.solid_angle_sr is not None:
            if isinstance(self.solid_angle_sr, (bool, np.bool_)):
                raise ValueError("solid_angle_sr must be finite and within [0, 4*pi].")
            try:
                solid_angle_sr = float(self.solid_angle_sr)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError("solid_angle_sr must be finite and within [0, 4*pi].") from exc
            if not np.isfinite(solid_angle_sr) or not 0.0 <= solid_angle_sr <= 4.0 * np.pi:
                raise ValueError("solid_angle_sr must be finite and within [0, 4*pi].")
            object.__setattr__(self, "solid_angle_sr", solid_angle_sr)
        if self.require_ground_visibility and self.ground_site is None:
            raise ValueError("ground visibility requires a configured ground_site.")
        if self.require_ground_visibility and (
            self.frame_context is None or self.frame_context.jd_utc_start is None
        ):
            raise ValueError(
                "ground visibility requires a FrameContext with an absolute jd_utc_start."
            )
        backend = normalize_numeric_backend(self.numeric_backend, error_message="numeric_backend must be python or rust.")
        object.__setattr__(self, "numeric_backend", backend)


class AccessModel:
    def __init__(self, cfg: AccessConfig):
        self.cfg = cfg
        self._last_update_t_s = -np.inf

    def evaluate(
        self,
        observer_eci_km: np.ndarray,
        target_eci_km: np.ndarray,
        t_s: float,
        *,
        boresight_eci: np.ndarray | None = None,
    ) -> tuple[bool, str]:
        if not _cadence_due(t_s, self._last_update_t_s, self.cfg.update_cadence_s):
            return False, "cadence"

        observer = np.asarray(observer_eci_km, dtype=float)
        target = np.asarray(target_eci_km, dtype=float)
        boresight = None if boresight_eci is None else np.asarray(boresight_eci, dtype=float)
        for name, vector in (("observer", observer), ("target", target), ("boresight", boresight)):
            if vector is not None and (vector.shape != (3,) or not np.all(np.isfinite(vector))):
                raise ValueError(f"access {name} must contain three finite coordinates.")
        ground_zenith: np.ndarray | None = None
        if self.cfg.require_ground_visibility:
            if self.cfg.ground_site is None:
                return False, "ground_site_missing"
            assert self.cfg.frame_context is not None
            observer, ground_zenith = _ground_site_geometry_eci(
                self.cfg.ground_site,
                t_s,
                self.cfg.frame_context,
            )
        los = target - observer
        rng = float(np.linalg.norm(los))
        if not np.isfinite(rng):
            raise ValueError("access observer and target separation must be finite.")
        if self.cfg.max_range_km is not None and rng > self.cfg.max_range_km:
            return False, "range"

        fov_half_angle_rad = self.cfg.fov_half_angle_rad
        if fov_half_angle_rad is None:
            fov_half_angle_rad = _solid_angle_to_half_angle_rad(self.cfg.solid_angle_sr)
        if fov_half_angle_rad is not None and rng > 0.0:
            if boresight_eci is None:
                boresight = observer / max(np.linalg.norm(observer), 1e-12)
            else:
                boresight = np.array(boresight_eci, dtype=float).reshape(3)
                bn = float(np.linalg.norm(boresight))
                if bn <= 0.0:
                    return False, "boresight"
                boresight = boresight / bn
            cosang = np.clip(np.dot(boresight, los / rng), -1.0, 1.0)
            if np.arccos(cosang) > fov_half_angle_rad:
                return False, "solid_angle"

        if self.cfg.require_ground_visibility:
            assert self.cfg.ground_site is not None
            assert ground_zenith is not None
            zenith = ground_zenith
            elevation = float(np.arcsin(np.clip(np.dot(zenith, los / max(rng, 1.0e-12)), -1.0, 1.0)))
            if elevation < self.cfg.ground_site.min_elevation_rad:
                return False, "ground_elevation"

        return True, "ok"

    def can_update(
        self,
        observer_eci_km: np.ndarray,
        target_eci_km: np.ndarray,
        t_s: float,
        *,
        boresight_eci: np.ndarray | None = None,
    ) -> bool:
        allowed, _ = self.evaluate(observer_eci_km, target_eci_km, t_s, boresight_eci=boresight_eci)
        if not allowed:
            return False
        self._last_update_t_s = t_s
        return True

    def evaluate_batch(
        self,
        observer_eci_km: np.ndarray,
        target_eci_km: np.ndarray,
        times_s: np.ndarray,
        *,
        boresight_eci: np.ndarray | None = None,
    ) -> tuple[np.ndarray, tuple[str, ...]]:
        """Evaluate a sampled access history while preserving cadence state.

        Rust is used only for the stateless range/cone gates.  Ground-site
        visibility still uses the frame-aware scalar path because it depends on
        the caller's EOP context and therefore must remain one authoritative
        implementation.
        """
        times = np.asarray(times_s, dtype=float)
        observer = np.asarray(observer_eci_km, dtype=float)
        target = np.asarray(target_eci_km, dtype=float)
        if times.ndim != 1 or not np.all(np.isfinite(times)) or np.any(np.diff(times) < 0.0):
            raise ValueError("access times must be a finite nondecreasing vector")
        shape = (times.size, 3)
        if observer.shape != shape or target.shape != shape:
            raise ValueError("observer and target must have shape (samples, 3)")
        boresight = None if boresight_eci is None else np.asarray(boresight_eci, dtype=float)
        if boresight is not None and boresight.shape == (3,):
            boresight = np.broadcast_to(boresight, shape)
        if boresight is not None and boresight.shape != shape:
            raise ValueError("boresight must have shape (3,) or (samples, 3)")
        if not np.all(np.isfinite(observer)) or not np.all(np.isfinite(target)) or (
            boresight is not None and not np.all(np.isfinite(boresight))
        ):
            raise ValueError("access coordinates must be finite")
        mask = np.zeros(times.size, dtype=bool)
        reasons = []
        geometry = None
        if self.cfg.numeric_backend == "rust" and not self.cfg.require_ground_visibility:
            from sim.rust_coverage_backend import access_batch

            half_angle = self.cfg.fov_half_angle_rad
            if half_angle is None:
                half_angle = _solid_angle_to_half_angle_rad(self.cfg.solid_angle_sr)
            _, available, geometry_reasons = access_batch(
                observer, target, boresight_eci=boresight,
                max_range_km=self.cfg.max_range_km, fov_half_angle_rad=half_angle,
            )
            geometry = (available, geometry_reasons)
        for index, epoch in enumerate(times):
            if not _cadence_due(float(epoch), self._last_update_t_s, self.cfg.update_cadence_s):
                allowed, reason = False, "cadence"
            elif geometry is not None:
                allowed, reason = bool(geometry[0][index]), geometry[1][index]
            else:
                allowed, reason = self.evaluate(
                    observer[index], target[index], float(epoch),
                    boresight_eci=None if boresight is None else boresight[index],
                )
            mask[index] = allowed
            reasons.append(reason)
            if allowed:
                self._last_update_t_s = float(epoch)
        return mask, tuple(reasons)


def _ground_site_geometry_eci(
    site: GroundSite,
    t_s: float,
    frame_context: FrameContext,
) -> tuple[np.ndarray, np.ndarray]:
    lat = float(site.lat_rad)
    lon = float(site.lon_rad)
    position_ecef = geodetic_to_ecef_km(
        np.rad2deg(lat),
        np.rad2deg(lon),
        float(site.alt_km),
    )
    up_ecef = np.array(
        [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)],
        dtype=float,
    )
    ecef_from_eci = eci_to_ecef_rotation_context(float(t_s), frame_context)
    return ecef_from_eci.T @ position_ecef, ecef_from_eci.T @ up_ecef


def _solid_angle_to_half_angle_rad(solid_angle_sr: float | None) -> float | None:
    if solid_angle_sr is None:
        return None
    omega = float(solid_angle_sr)
    if not np.isfinite(omega) or omega <= 0.0:
        return 0.0
    if omega >= (4.0 * np.pi - 1e-12):
        return None
    cos_half = float(np.clip(1.0 - omega / (2.0 * np.pi), -1.0, 1.0))
    return float(np.arccos(cos_half))
