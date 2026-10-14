from __future__ import annotations

import hashlib
from dataclasses import asdict
from typing import Any

import numpy as np

from sim.config import GroundStationSection
from sim.dynamics.orbit.frames import (
    FrameContext,
    eci_to_ecef_rotation_context,
    eci_to_ecef_rotation_derivative_context,
    frame_context_from_mapping,
    transform_position,
    transform_state,
)
from sim.observations import ObservationPacket, ingest_observations
from sim.utils.geodesy import (
    WGS84_A_KM,
    WGS84_B_KM,
    ecef_to_enu_rotation,
    enu_to_ecef_rotation,
    geodetic_to_ecef_km,
)


def _json_float(value: float) -> float | None:
    x = float(value)
    return x if np.isfinite(x) else None


def _first_last_time(t_s: np.ndarray, mask: np.ndarray) -> tuple[float | None, float | None]:
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return None, None
    return float(t_s[int(idx[0])]), float(t_s[int(idx[-1])])


def _line_of_sight_from_ground(station_ecef_km: np.ndarray, target_ecef_km: np.ndarray) -> bool:
    """Return whether the segment stays outside the WGS-84 reference ellipsoid.

    Inputs are ECEF coordinates, where the WGS-84 ellipsoid axes are aligned.
    Keeping this test in ECEF also preserves correctness when a high-fidelity
    ECI/ECEF transform includes polar-motion tilts.
    """

    station = np.array(station_ecef_km, dtype=float).reshape(3)
    target = np.array(target_ecef_km, dtype=float).reshape(3)
    scale = np.array([WGS84_A_KM, WGS84_A_KM, WGS84_B_KM], dtype=float)
    axes_squared = scale * scale
    station_level = float(np.dot(station * station, 1.0 / axes_squared))
    target_level = float(np.dot(target * target, 1.0 / axes_squared))
    # Geodetic surface construction can round a boundary level by a few ulps;
    # retain a small dimensionless tolerance while still rejecting physical
    # penetration well beyond floating-point noise.
    ellipsoid_level_tolerance = 1.0e-12
    if (
        station_level < 1.0 - ellipsoid_level_tolerance
        or target_level < 1.0 - ellipsoid_level_tolerance
    ):
        return False
    station_scaled = station / scale
    segment = (target - station) / scale
    denom = float(np.dot(segment, segment))
    if denom <= 0.0:
        return bool(station_level >= 1.0 - ellipsoid_level_tolerance)
    tau = float(np.clip(-np.dot(station_scaled, segment) / denom, 0.0, 1.0))
    closest = station_scaled + tau * segment
    return bool(np.dot(closest, closest) >= 1.0 - ellipsoid_level_tolerance)


def evaluate_ground_station_access(
    *,
    ground_stations: list[GroundStationSection],
    t_s: np.ndarray,
    truth_hist: dict[str, np.ndarray],
    jd_utc_start: float | None = None,
    frame_context: FrameContext | None = None,
    object_state_frames: dict[str, str] | None = None,
) -> tuple[dict[str, dict[str, dict[str, Any]]], dict[str, dict[str, dict[str, Any]]]]:
    """
    Evaluate passive ground-station access to each simulated object.

    Access is true when the target has geometric line of sight from the station,
    elevation is at least the station minimum, and range is no greater than the
    configured maximum range when one is supplied.
    """
    stations = [station for station in ground_stations if station.enabled]
    tt = np.array(t_s, dtype=float).reshape(-1)
    if not stations or tt.size == 0:
        return {}, {}
    frame_ctx = frame_context or frame_context_from_mapping({}, jd_utc_start=jd_utc_start, source="ground_station")

    histories: dict[str, dict[str, dict[str, Any]]] = {}
    summaries: dict[str, dict[str, dict[str, Any]]] = {}
    dt = np.diff(tt)
    total_duration_s = float(tt[-1] - tt[0]) if tt.size > 1 else 0.0
    rotations_by_index: dict[int, np.ndarray] = {}
    target_ecef_by_object_index: dict[tuple[str, int], np.ndarray] = {}

    for station in stations:
        station_ecef = geodetic_to_ecef_km(station.lat_deg, station.lon_deg, station.alt_km)
        enu_rot = ecef_to_enu_rotation(station.lat_deg, station.lon_deg)
        min_elev = float(station.min_elevation_deg)
        max_range = station.max_range_km
        station_hist: dict[str, dict[str, Any]] = {}
        station_summary: dict[str, dict[str, Any]] = {}
        station_eci_by_index: dict[int, np.ndarray] = {}

        for object_id, hist in sorted(truth_hist.items()):
            if str(dict(object_state_frames or {}).get(str(object_id), "eci") or "eci").lower() != "eci":
                continue
            arr = np.array(hist, dtype=float)
            n = min(tt.size, arr.shape[0])
            access = np.zeros(tt.size, dtype=bool)
            los_ok = np.zeros(tt.size, dtype=bool)
            range_km = np.full(tt.size, np.nan)
            elevation_deg = np.full(tt.size, np.nan)
            reason: list[str] = ["inactive"] * tt.size

            for k in range(n):
                state = arr[k, :]
                if state.size < 3 or not np.all(np.isfinite(state[:3])):
                    continue
                target_eci = np.array(state[:3], dtype=float)
                rotation = rotations_by_index.get(k)
                if rotation is None:
                    rotation = eci_to_ecef_rotation_context(float(tt[k]), frame_ctx)
                    rotations_by_index[k] = rotation
                station_eci = station_eci_by_index.get(k)
                if station_eci is None:
                    station_eci = rotation.T @ np.array(station_ecef, dtype=float)
                    station_eci_by_index[k] = station_eci
                target_key = (str(object_id), k)
                target_ecef = target_ecef_by_object_index.get(target_key)
                if target_ecef is None:
                    target_ecef = rotation @ np.array(target_eci, dtype=float)
                    target_ecef_by_object_index[target_key] = target_ecef
                rho_ecef = target_ecef - station_ecef
                rng = float(np.linalg.norm(rho_ecef))
                range_km[k] = rng
                if rng <= 0.0:
                    elevation_deg[k] = 90.0
                else:
                    enu = enu_rot @ rho_ecef
                    elevation_deg[k] = float(np.rad2deg(np.arcsin(np.clip(enu[2] / rng, -1.0, 1.0))))

                los_ok[k] = bool(_line_of_sight_from_ground(station_ecef, target_ecef))
                if not los_ok[k]:
                    reason[k] = "line_of_sight"
                    continue
                if elevation_deg[k] < min_elev:
                    reason[k] = "elevation"
                    continue
                if max_range is not None and rng > float(max_range):
                    reason[k] = "range"
                    continue
                access[k] = True
                reason[k] = "ok"

            first_t, last_t = _first_last_time(tt, access)
            access_duration_s = 0.0
            if tt.size > 1:
                access_duration_s = float(np.sum(dt * access[:-1].astype(float)))
            finite_range = range_km[np.isfinite(range_km)]
            finite_elev = elevation_deg[np.isfinite(elevation_deg)]
            station_hist[str(object_id)] = {
                "access": access.tolist(),
                "line_of_sight": los_ok.tolist(),
                "range_km": [_json_float(x) for x in range_km],
                "elevation_deg": [_json_float(x) for x in elevation_deg],
                "reason": reason,
            }
            station_summary[str(object_id)] = {
                "samples": int(tt.size),
                "access_samples": int(np.count_nonzero(access)),
                "no_access_samples": int(tt.size - np.count_nonzero(access)),
                "access_fraction": float(np.mean(access)) if access.size else 0.0,
                "access_duration_s": access_duration_s,
                "total_duration_s": total_duration_s,
                "minimum_elevation_deg": min_elev,
                "maximum_range_km": None if max_range is None else float(max_range),
                "reason_sample_count": {
                    name: int(reason.count(name)) for name in sorted(set(reason))
                },
                "first_access_time_s": first_t,
                "last_access_time_s": last_t,
                "min_range_km": float(np.min(finite_range)) if finite_range.size else None,
                "max_elevation_deg": float(np.max(finite_elev)) if finite_elev.size else None,
            }

        histories[station.id] = {
            "station": asdict(station),
            "targets": station_hist,
        }
        summaries[station.id] = station_summary

    return histories, summaries


def evaluate_ground_station_measurements(
    *,
    ground_stations: list[GroundStationSection],
    t_s: np.ndarray,
    truth_hist: dict[str, np.ndarray],
    jd_utc_start: float | None = None,
    frame_context: FrameContext | None = None,
    object_state_frames: dict[str, str] | None = None,
    stream_state: dict | None = None,
) -> dict[str, dict[str, Any]]:
    """Emit opt-in ground-station sensor measurements for visible targets.

    Measurements are generated only for stations with `measurements.enabled`
    configured. The first supported sensor model emits azimuth, elevation,
    slant range, and optionally range rate. The geometry is deterministic; noise
    is optional and seeded per station/target.
    """

    stations = [station for station in ground_stations if station.enabled and _measurements_enabled(station)]
    tt = np.array(t_s, dtype=float).reshape(-1)
    if not stations or tt.size == 0:
        return {}
    frame_ctx = frame_context or frame_context_from_mapping({}, jd_utc_start=jd_utc_start, source="ground_station")

    output: dict[str, dict[str, Any]] = {}
    rotations_by_index: dict[int, np.ndarray] = {}
    rotation_derivatives_by_index: dict[int, np.ndarray] = {}
    target_ecef_by_object_index: dict[tuple[str, int], np.ndarray] = {}
    for station in stations:
        station_ecef = geodetic_to_ecef_km(station.lat_deg, station.lon_deg, station.alt_km)
        ecef_to_enu = ecef_to_enu_rotation(station.lat_deg, station.lon_deg)
        cfg = _measurement_config(station)
        station_payload: dict[str, Any] = {
            "station": asdict(station),
            "measurement_type": cfg["measurement_type"],
            "noise": {
                "range_sigma_km": cfg["range_sigma_km"],
                "range_rate_sigma_km_s": cfg["range_rate_sigma_km_s"],
                "angle_sigma_deg": cfg["angle_sigma_deg"],
            },
            "targets": {},
        }
        station_state_by_index: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for object_id, hist in sorted(truth_hist.items()):
            if str(dict(object_state_frames or {}).get(str(object_id), "eci") or "eci").lower() != "eci":
                continue
            arr = np.array(hist, dtype=float)
            n = min(tt.size, arr.shape[0])
            rows: list[dict[str, Any]] = []
            skipped = {"cadence": 0, "line_of_sight": 0, "elevation": 0, "range": 0, "invalid_state": 0}
            last_emit_t = -np.inf
            rng = np.random.default_rng(_measurement_seed(cfg["seed"], station.id, str(object_id)))
            stream_key = (station.id, str(object_id))
            if stream_state is not None and stream_key in stream_state:
                last_emit_t, rng = stream_state[stream_key]
            for k in range(n):
                state = arr[k, :]
                if state.size < 6 or not np.all(np.isfinite(state[:6])):
                    skipped["invalid_state"] += 1
                    continue
                t = float(tt[k])
                if t - last_emit_t < cfg["update_cadence_s"] - 1.0e-12:
                    skipped["cadence"] += 1
                    continue
                rotation = rotations_by_index.get(k)
                if rotation is None:
                    rotation = eci_to_ecef_rotation_context(t, frame_ctx)
                    rotations_by_index[k] = rotation
                rotation_derivative = rotation_derivatives_by_index.get(k)
                if rotation_derivative is None:
                    rotation_derivative = eci_to_ecef_rotation_derivative_context(t, frame_ctx)
                    rotation_derivatives_by_index[k] = rotation_derivative
                station_state = station_state_by_index.get(k)
                if station_state is None:
                    station_position_eci = rotation.T @ np.array(station_ecef, dtype=float)
                    station_velocity_eci = rotation.T @ (
                        np.zeros(3) - rotation_derivative @ station_position_eci
                    )
                    station_state = (station_position_eci, station_velocity_eci)
                    station_state_by_index[k] = station_state
                target_key = (str(object_id), k)
                target_ecef = target_ecef_by_object_index.get(target_key)
                if target_ecef is None:
                    target_ecef = rotation @ np.array(state[:3], dtype=float)
                    target_ecef_by_object_index[target_key] = target_ecef
                geometry = _ground_measurement_geometry(
                    station=station,
                    station_ecef_km=station_ecef,
                    ecef_to_enu=ecef_to_enu,
                    target_state_eci=np.array(state[:6], dtype=float),
                    t_s=t,
                    jd_utc_start=jd_utc_start,
                    frame_context=frame_ctx,
                    ecef_from_eci=rotation,
                    station_state_eci=station_state,
                    target_ecef_km=target_ecef,
                )
                reason = _access_reason(
                    station=station,
                    line_of_sight=bool(geometry["line_of_sight"]),
                    elevation_deg=float(geometry["elevation_deg"]),
                    range_km=float(geometry["range_km"]),
                )
                if reason != "ok":
                    skipped[reason] = skipped.get(reason, 0) + 1
                    continue
                measured_range = float(geometry["range_km"]) + _normal(rng, cfg["range_sigma_km"])
                measured_range_rate = float(geometry["range_rate_km_s"]) + _normal(rng, cfg["range_rate_sigma_km_s"])
                measured_az = (float(geometry["azimuth_deg"]) + _normal(rng, cfg["angle_sigma_deg"])) % 360.0
                measured_el = float(geometry["elevation_deg"]) + _normal(rng, cfg["angle_sigma_deg"])
                vector = [measured_az, measured_el, measured_range]
                components = ["azimuth_deg", "elevation_deg", "range_km"]
                sigmas = [cfg["angle_sigma_deg"], cfg["angle_sigma_deg"], cfg["range_sigma_km"]]
                if cfg["measurement_type"] == "az_el_range_rate":
                    vector.append(measured_range_rate)
                    components.append("range_rate_km_s")
                    sigmas.append(cfg["range_rate_sigma_km_s"])
                rows.append(
                    {
                        "time_s": t,
                        "jd_utc": None if jd_utc_start is None else float(jd_utc_start) + t / 86400.0,
                        "station_id": station.id,
                        "object_id": str(object_id),
                        "measurement_type": cfg["measurement_type"],
                        "components": components,
                        "vector": vector,
                        "sigma": sigmas,
                        "azimuth_deg": measured_az,
                        "elevation_deg": measured_el,
                        "range_km": measured_range,
                        "range_rate_km_s": measured_range_rate,
                        "truth_azimuth_deg": float(geometry["azimuth_deg"]),
                        "truth_elevation_deg": float(geometry["elevation_deg"]),
                        "truth_range_km": float(geometry["range_km"]),
                        "truth_range_rate_km_s": float(geometry["range_rate_km_s"]),
                    }
                )
                last_emit_t = t
            if stream_state is not None:
                stream_state[stream_key] = (last_emit_t, rng)
            station_payload["targets"][str(object_id)] = {
                "measurements": rows,
                "measurement_count": len(rows),
                "skipped": skipped,
            }
        output[station.id] = station_payload
    return output


def ground_station_measurements_to_observation_packet(
    *,
    ground_station_measurements: dict[str, Any],
    station_id: str,
    object_id: str,
    source_label: str = "ground_station_measurements",
    jd_utc_start: float | None = None,
    frame_context: FrameContext | None = None,
) -> ObservationPacket:
    """Convert az/el/range ground measurements into ECI position observations."""

    station_payload = dict(dict(ground_station_measurements or {}).get(station_id, {}) or {})
    if not station_payload:
        raise ValueError(f"ground station measurement payload does not contain station {station_id!r}.")
    station = dict(station_payload.get("station", {}) or {})
    target_payload = dict(dict(station_payload.get("targets", {}) or {}).get(object_id, {}) or {})
    rows = list(target_payload.get("measurements", []) or [])
    if len(rows) < 2:
        raise ValueError("at least two ground-station measurements are required for an observation packet.")
    station_ecef = geodetic_to_ecef_km(
        float(station["lat_deg"]),
        float(station["lon_deg"]),
        float(station.get("alt_km", 0.0) or 0.0),
    )
    enu_to_ecef = enu_to_ecef_rotation(float(station["lat_deg"]), float(station["lon_deg"]))
    frame_ctx = frame_context or frame_context_from_mapping(
        {},
        jd_utc_start=jd_utc_start,
        source="ground_station_observation_packet",
    )
    obs_rows: list[dict[str, Any]] = []
    for row in rows:
        t = float(row["time_s"])
        jd = row.get("jd_utc")
        if jd is None and jd_utc_start is not None:
            jd = float(jd_utc_start) + t / 86400.0
        row_frame_ctx = frame_ctx
        if frame_context is None and jd_utc_start is None and jd is not None:
            # Measurement rows carry an absolute sample epoch.  Recover the
            # run start before applying the row's relative time so the ECI
            # conversion does not silently discard the absolute epoch or add
            # the row time twice.
            row_start_jd = float(jd) - t / 86400.0
            row_frame_ctx = frame_context_from_mapping(
                {},
                jd_utc_start=row_start_jd,
                source="ground_station_observation_packet",
            )
        az = np.deg2rad(float(row["azimuth_deg"]))
        el = np.deg2rad(float(row["elevation_deg"]))
        rng = float(row["range_km"])
        enu = rng * np.array([np.cos(el) * np.sin(az), np.cos(el) * np.cos(az), np.sin(el)], dtype=float)
        target_ecef = station_ecef + enu_to_ecef @ enu
        row_frame_ctx = frame_ctx
        if frame_context is None and jd_utc_start is None and jd is not None:
            # Measurement rows carry an absolute sample epoch. Recover the
            # run start before applying the row's relative time so the ECI
            # conversion does not silently discard the absolute epoch or add
            # the row time twice.
            row_start_jd = float(jd) - t / 86400.0
            row_frame_ctx = frame_context_from_mapping(
                {},
                jd_utc_start=row_start_jd,
                source="ground_station_observation_packet",
            )
        target_eci = transform_position(target_ecef, "ecef", "eci", t_s=t, context=row_frame_ctx)
        sigma = _position_sigma_from_measurement(row)
        obs: dict[str, Any] = {
            "time_s": t,
            "position_eci_km": [float(x) for x in target_eci],
            "position_sigma_km": sigma,
        }
        if jd is not None:
            obs["jd_utc"] = float(jd)
        obs_rows.append(obs)
    return ingest_observations(
        object_id=object_id,
        observations=obs_rows,
        source_label=source_label,
        frame="eci",
        source_metadata={
            "type": "ground_station_az_el_range",
            "station_id": station_id,
            "measurement_type": station_payload.get("measurement_type"),
            "non_claims": [
                "Converted ECI positions are derived from measured azimuth, elevation, and range.",
                "Range-rate and angular-rate information are not converted into velocity observations in this bridge.",
            ],
        },
    )


def _ground_measurement_geometry(
    *,
    station: GroundStationSection,
    station_ecef_km: np.ndarray,
    ecef_to_enu: np.ndarray,
    target_state_eci: np.ndarray,
    t_s: float,
    jd_utc_start: float | None,
    frame_context: FrameContext | None = None,
    ecef_from_eci: np.ndarray | None = None,
    station_state_eci: tuple[np.ndarray, np.ndarray] | None = None,
    target_ecef_km: np.ndarray | None = None,
) -> dict[str, float | bool]:
    frame_ctx = frame_context or frame_context_from_mapping({}, jd_utc_start=jd_utc_start, source="ground_station")
    target_eci = np.array(target_state_eci[:3], dtype=float)
    target_vel_eci = np.array(target_state_eci[3:6], dtype=float)
    if station_state_eci is None:
        station_eci, station_vel_eci = transform_state(
            station_ecef_km,
            np.zeros(3),
            "ecef",
            "eci",
            t_s=t_s,
            context=frame_ctx,
        )
    else:
        station_eci, station_vel_eci = station_state_eci
    target_ecef = (
        np.asarray(target_ecef_km, dtype=float)
        if target_ecef_km is not None
        else (
            transform_position(target_eci, "eci", "ecef", t_s=t_s, context=frame_ctx)
            if ecef_from_eci is None
            else ecef_from_eci @ np.array(target_eci, dtype=float)
        )
    )
    rho_ecef = target_ecef - station_ecef_km
    rng = float(np.linalg.norm(rho_ecef))
    enu = ecef_to_enu @ rho_ecef
    if rng <= 0.0:
        azimuth_deg = 0.0
        elevation_deg = 90.0
        range_rate = 0.0
    else:
        azimuth_deg = float(np.rad2deg(np.arctan2(enu[0], enu[1])) % 360.0)
        elevation_deg = float(np.rad2deg(np.arcsin(np.clip(enu[2] / rng, -1.0, 1.0))))
        los_eci = (target_eci - station_eci) / max(float(np.linalg.norm(target_eci - station_eci)), 1.0e-12)
        range_rate = float(np.dot(target_vel_eci - station_vel_eci, los_eci))
    return {
        "range_km": rng,
        "range_rate_km_s": range_rate,
        "azimuth_deg": azimuth_deg,
        "elevation_deg": elevation_deg,
        "line_of_sight": _line_of_sight_from_ground(station_ecef_km, target_ecef),
        "min_elevation_deg": float(station.min_elevation_deg),
    }


def _measurement_config(station: GroundStationSection) -> dict[str, Any]:
    raw = dict(station.measurements or {})
    noise = dict(raw.get("noise", {}) or {})
    return {
        "enabled": bool(raw.get("enabled", False)),
        "measurement_type": str(raw.get("measurement_type", "az_el_range_rate") or "az_el_range_rate").lower(),
        "update_cadence_s": float(raw.get("update_cadence_s", 1.0) or 1.0),
        "seed": int(raw.get("seed", 0) or 0),
        "range_sigma_km": float(noise.get("range_sigma_km", raw.get("range_sigma_km", 0.0)) or 0.0),
        "range_rate_sigma_km_s": float(
            noise.get("range_rate_sigma_km_s", raw.get("range_rate_sigma_km_s", 0.0)) or 0.0
        ),
        "angle_sigma_deg": float(noise.get("angle_sigma_deg", raw.get("angle_sigma_deg", 0.0)) or 0.0),
    }


def _measurements_enabled(station: GroundStationSection) -> bool:
    return bool(dict(station.measurements or {}).get("enabled", False))


def _measurement_seed(base_seed: int, station_id: str, object_id: str) -> int:
    payload = f"{int(base_seed)}:{station_id}:{object_id}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False) % (2**32)


def _normal(rng: np.random.Generator, sigma: float) -> float:
    s = float(sigma)
    if s <= 0.0:
        return 0.0
    return float(rng.normal(0.0, s))


def _access_reason(
    *,
    station: GroundStationSection,
    line_of_sight: bool,
    elevation_deg: float,
    range_km: float,
) -> str:
    if not line_of_sight:
        return "line_of_sight"
    if float(elevation_deg) < float(station.min_elevation_deg):
        return "elevation"
    if station.max_range_km is not None and float(range_km) > float(station.max_range_km):
        return "range"
    return "ok"


def _position_sigma_from_measurement(row: dict[str, Any]) -> float:
    sigma = list(row.get("sigma", []) or [])
    range_sigma = float(sigma[2]) if len(sigma) >= 3 else 1.0
    angle_sigma_rad = np.deg2rad(max(float(sigma[0]) if sigma else 0.0, float(sigma[1]) if len(sigma) > 1 else 0.0))
    angular_sigma_km = abs(float(row.get("range_km", 0.0) or 0.0)) * float(angle_sigma_rad)
    return max(range_sigma, angular_sigma_km, 1.0e-9)
