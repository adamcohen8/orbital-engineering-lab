from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from sim.dynamics.orbit.frames import FrameContext, frame_context_from_mapping, transform_position
from sim.estimation.weighting import validate_covariance_block
from sim.utils.geodesy import geodetic_to_ecef_km

OPTICAL_ANGLE_PACKET_SCHEMA_VERSION = 1
OPTICAL_ANGLE_CONTRACT = "oel_optical_ra_dec_v1"
_DAY_S = 86400.0


def adapt_optical_ra_dec_observations(
    observations: Sequence[Mapping[str, Any]],
    *,
    object_id: str,
    initial_jd_utc: float | None = None,
    frame_context: FrameContext | None = None,
    default_sigma_arcsec: float = 5.0,
) -> dict[str, Any]:
    """Normalize inertial optical RA/Dec rows with explicit observer geometry."""

    if len(observations) < 3:
        raise ValueError("Optical RA/Dec adaptation requires at least three observations.")
    default_sigma_deg = float(default_sigma_arcsec) / 3600.0
    if not np.isfinite(default_sigma_deg) or default_sigma_deg <= 0.0:
        raise ValueError("default_sigma_arcsec must be finite and positive.")
    raw_rows = [dict(item) for item in observations]
    jd_values = [float(row["time_jd_utc"]) for row in raw_rows if row.get("time_jd_utc") is not None]
    if initial_jd_utc is None:
        if not jd_values:
            raise ValueError("initial_jd_utc is required when optical rows do not carry time_jd_utc.")
        epoch_jd = min(jd_values)
    else:
        epoch_jd = float(initial_jd_utc)
    if not np.isfinite(epoch_jd):
        raise ValueError("initial_jd_utc must be finite.")
    context = frame_context or frame_context_from_mapping(
        {},
        jd_utc_start=epoch_jd,
        source="optical_ra_dec_adapter",
    )
    _require_frame_context_epoch(context, epoch_jd)
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_rows):
        _require_optical_frame_and_time(raw)
        measurement_id = str(raw.get("measurement_id") or f"optical:{index:06d}").strip()
        if not measurement_id or measurement_id in seen:
            raise ValueError("Optical measurement_id values must be non-empty and unique.")
        seen.add(measurement_id)
        time_s, jd_utc = _normalize_epoch(raw, epoch_jd=epoch_jd)
        ra_deg = float(raw["ra_deg"]) % 360.0
        dec_deg = float(raw["dec_deg"])
        if not np.isfinite(ra_deg) or not np.isfinite(dec_deg) or not -90.0 <= dec_deg <= 90.0:
            raise ValueError("Optical RA/Dec values must be finite and declination must be within [-90, 90].")
        observer, observer_meta = _observer_eci(raw, time_s=time_s, context=context)
        covariance = _angular_covariance(raw, default_sigma_deg=default_sigma_deg)
        partition = str(raw.get("partition", "unassigned") or "unassigned").strip().lower()
        if partition not in {"fit", "holdout", "excluded", "unassigned"}:
            raise ValueError(f"Unsupported optical partition {partition!r}.")
        rows.append(
            {
                "measurement_id": measurement_id,
                "measurement_type": "optical_ra_dec",
                "measurement_contract": OPTICAL_ANGLE_CONTRACT,
                "object_id": str(object_id),
                "observer_id": str(raw.get("observer_id") or observer_meta.get("station_id") or "observer"),
                "time_s": time_s,
                "time_jd_utc": jd_utc,
                "time_system": "UTC",
                "frame": "ECI",
                "ra_deg": ra_deg,
                "dec_deg": dec_deg,
                "los_eci": ra_dec_to_los(ra_deg, dec_deg).tolist(),
                "observer_eci_km": observer.tolist(),
                "observer_metadata": observer_meta,
                "angular_covariance_deg2": covariance.tolist(),
                "partition": partition,
                "bias_metadata": deepcopy(dict(raw.get("bias_metadata", {}) or {})),
                "quality_flags": deepcopy(dict(raw.get("quality_flags", {}) or {})),
            }
        )
    rows.sort(key=lambda item: (float(item["time_s"]), str(item["measurement_id"])))
    times = np.array([float(row["time_s"]) for row in rows], dtype=float)
    if np.any(np.diff(times) <= 0.0):
        raise ValueError("Optical RA/Dec epochs must be strictly increasing.")
    return {
        "schema_version": OPTICAL_ANGLE_PACKET_SCHEMA_VERSION,
        "kind": "oel.optical_angle_packet",
        "measurement_contract": OPTICAL_ANGLE_CONTRACT,
        "object_id": str(object_id),
        "initial_jd_utc": epoch_jd,
        "time_system": "UTC",
        "frame": "ECI",
        "measurement_count": len(rows),
        "observations": rows,
        "observer_ids": sorted({str(row["observer_id"]) for row in rows}),
        "arc_length_s": float(times[-1] - times[0]),
        "frame_context": context.metadata(),
        "claim_boundary": "inertial optical direction observations with governed observer geometry; no range information",
    }


def ra_dec_to_los(ra_deg: float, dec_deg: float) -> np.ndarray:
    ra = np.deg2rad(float(ra_deg))
    dec = np.deg2rad(float(dec_deg))
    return np.array(
        [np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)],
        dtype=float,
    )


def los_to_ra_dec(los_eci: Sequence[float]) -> tuple[float, float]:
    los = np.asarray(los_eci, dtype=float).reshape(3)
    norm = float(np.linalg.norm(los))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("Line-of-sight vector must be finite and nonzero.")
    unit = los / norm
    return (
        float(np.rad2deg(np.arctan2(unit[1], unit[0])) % 360.0),
        float(np.rad2deg(np.arcsin(np.clip(unit[2], -1.0, 1.0)))),
    )


def load_optical_angle_packet(path: str | Path) -> dict[str, Any]:
    import json

    data = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    return validate_optical_angle_packet(data)


def validate_optical_angle_packet(packet: Mapping[str, Any]) -> dict[str, Any]:
    data = deepcopy(dict(packet))
    _require_optical_frame_and_time(data)
    if int(data.get("schema_version", 0)) != OPTICAL_ANGLE_PACKET_SCHEMA_VERSION:
        raise ValueError(f"Optical packet schema_version must be {OPTICAL_ANGLE_PACKET_SCHEMA_VERSION}.")
    if str(data.get("kind")) != "oel.optical_angle_packet":
        raise ValueError("Optical packet kind must be 'oel.optical_angle_packet'.")
    if str(data.get("measurement_contract")) != OPTICAL_ANGLE_CONTRACT:
        raise ValueError(f"Optical packet requires measurement_contract={OPTICAL_ANGLE_CONTRACT!r}.")
    rows = list(data.get("observations", []) or [])
    retained_context = data.get("frame_context")
    if retained_context is not None:
        if not isinstance(retained_context, Mapping):
            raise ValueError("Optical frame_context must be a mapping.")
        context = frame_context_from_mapping(dict(retained_context))
        _require_frame_context_epoch(context, float(data["initial_jd_utc"]))
        if any(row.get("observer_eci_km") is None for row in rows):
            raise ValueError("Normalized optical packets with frame_context require explicit observer_eci_km.")
    normalized = adapt_optical_ra_dec_observations(
        rows,
        object_id=str(data.get("object_id") or "target"),
        initial_jd_utc=float(data["initial_jd_utc"]),
    )
    if retained_context is not None:
        # Coordinates are already normalized. Retain their declared provenance,
        # without re-reading a producer's EOP file or recomputing station states.
        normalized["frame_context"] = deepcopy(dict(retained_context))
    return normalized


def _require_optical_frame_and_time(data: Mapping[str, Any]) -> None:
    for field, expected in (("frame", "ECI"), ("time_system", "UTC")):
        if field in data and str(data[field]).strip().upper() != expected:
            raise ValueError(f"Optical {field} must be {expected}; implicit conversion is not supported.")


def _require_frame_context_epoch(context: FrameContext, epoch_jd: float) -> None:
    if (
        context.jd_utc_start is None
        or not np.isfinite(context.jd_utc_start)
        or not np.isfinite(epoch_jd)
        or abs((context.jd_utc_start - epoch_jd) * _DAY_S) > 1.0e-3
    ):
        raise ValueError("Optical frame context epoch must match initial_jd_utc.")


def _normalize_epoch(raw: Mapping[str, Any], *, epoch_jd: float) -> tuple[float, float]:
    has_time = raw.get("time_s") is not None
    has_jd = raw.get("time_jd_utc") is not None
    if not has_time and not has_jd:
        raise ValueError("Each optical observation requires time_s or time_jd_utc.")
    time_s = float(raw["time_s"]) if has_time else (float(raw["time_jd_utc"]) - epoch_jd) * _DAY_S
    jd_utc = float(raw["time_jd_utc"]) if has_jd else epoch_jd + time_s / _DAY_S
    if not np.isfinite(time_s) or not np.isfinite(jd_utc):
        raise ValueError("Optical observation epochs must be finite.")
    if has_time and has_jd and abs((jd_utc - epoch_jd) * _DAY_S - time_s) > 1.0e-3:
        raise ValueError("Optical time_s and time_jd_utc are inconsistent.")
    return time_s, jd_utc


def _observer_eci(
    raw: Mapping[str, Any],
    *,
    time_s: float,
    context: FrameContext,
) -> tuple[np.ndarray, dict[str, Any]]:
    if raw.get("observer_eci_km") is not None:
        observer = np.asarray(raw["observer_eci_km"], dtype=float).reshape(3)
        metadata = deepcopy(dict(raw.get("observer_metadata", {}) or {}))
        metadata.setdefault("source", "explicit_observer_eci_km")
        metadata.setdefault("frame", "ECI")
    else:
        station = deepcopy(dict(raw.get("station_metadata", {}) or {}))
        required = {"lat_deg", "lon_deg"}
        if not required.issubset(station):
            raise ValueError("Optical rows require observer_eci_km or station_metadata lat_deg/lon_deg.")
        station_ecef = geodetic_to_ecef_km(
            float(station["lat_deg"]),
            float(station["lon_deg"]),
            float(station.get("alt_km", 0.0) or 0.0),
        )
        observer = transform_position(station_ecef, "ecef", "eci", t_s=float(time_s), context=context)
        metadata = {
            **station,
            "source": "wgs84_geodetic_station",
            "station_id": str(raw.get("observer_id") or station.get("station_id") or "station"),
            "source_frame": "WGS84_geodetic",
            "output_frame": "ECI",
        }
    if not np.all(np.isfinite(observer)):
        raise ValueError("Observer ECI coordinates must be finite.")
    return observer, metadata


def _angular_covariance(raw: Mapping[str, Any], *, default_sigma_deg: float) -> np.ndarray:
    if raw.get("angular_covariance_deg2") is not None:
        return validate_covariance_block(
            raw["angular_covariance_deg2"],
            dimension=2,
            field_name="optical angular_covariance_deg2",
        )
    ra_sigma = float(raw.get("ra_sigma_arcsec", default_sigma_deg * 3600.0)) / 3600.0
    dec_sigma = float(raw.get("dec_sigma_arcsec", default_sigma_deg * 3600.0)) / 3600.0
    if not np.isfinite(ra_sigma) or not np.isfinite(dec_sigma) or ra_sigma <= 0.0 or dec_sigma <= 0.0:
        raise ValueError("Optical RA/Dec sigmas must be finite and positive.")
    return np.diag([ra_sigma * ra_sigma, dec_sigma * dec_sigma])
