from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

from sim.dynamics.orbit.frames import FrameContext, frame_context_from_mapping, transform_position
from sim.dynamics.orbit.ogp import ogp_propagate_teme
from sim.dynamics.orbit.sgp4 import transform_teme_to_output_frame
from sim.dynamics.orbit.tle import TLEElements, with_tle_checksum
from sim.utils.geodesy import ecef_to_geodetic_deg_km

DAY_S = 86400.0


def truth_arc_tracking_network_from_ogp(
    truth: TLEElements,
    *,
    duration_s: float,
    station_spacing_s: float,
    station_longitude_offset_deg: float,
    range_sigma_km: float,
    range_rate_sigma_km_s: float,
    angle_sigma_deg: float,
    frame_transform: str,
    seed: int = 7,
    frame_context: FrameContext | None = None,
) -> list[dict[str, Any]]:
    """Create deterministic synthetic ground sites along an OGP truth arc."""

    duration = _finite_float(duration_s, name="duration_s", minimum=0.0)
    spacing = _finite_float(station_spacing_s, name="station_spacing_s", positive=True)
    longitude_offset = _finite_float(
        station_longitude_offset_deg,
        name="station_longitude_offset_deg",
    )
    range_sigma = _finite_float(range_sigma_km, name="range_sigma_km", minimum=0.0)
    range_rate_sigma = _finite_float(range_rate_sigma_km_s, name="range_rate_sigma_km_s", minimum=0.0)
    angle_sigma = _finite_float(angle_sigma_deg, name="angle_sigma_deg", minimum=0.0)
    stations = []
    frame_ctx = frame_context or frame_context_from_mapping(
        {},
        jd_utc_start=float(truth.epoch_jd_utc),
        source="ground_station_synthetic",
    )
    for idx, t_s in enumerate(np.arange(0.0, duration + 0.5 * spacing, spacing)):
        jd = float(truth.epoch_jd_utc) + float(t_s) / DAY_S
        state = ogp_propagate_teme(truth, float(t_s) / 60.0)
        if state.error:
            raise ValueError(state.error)
        pos_eci, _ = transform_teme_to_output_frame(
            state.position_teme_km,
            state.velocity_teme_km_s,
            jd_utc=jd,
            output_frame="eci",
            frame_transform=str(frame_transform or "teme_as_eci"),
        )
        lat_deg, lon_deg, _ = ecef_to_geodetic_deg_km(
            transform_position(np.array(pos_eci, dtype=float), "eci", "ecef", t_s=float(t_s), context=frame_ctx)
        )
        stations.append(
            ground_station_measurement_site(
                station_id=f"truth_arc_site_{idx:02d}",
                lat_deg=lat_deg,
                lon_deg=lon_deg,
                station_longitude_offset_deg=longitude_offset,
                update_cadence_s=spacing,
                range_sigma_km=range_sigma,
                range_rate_sigma_km_s=range_rate_sigma,
                angle_sigma_deg=angle_sigma,
                seed=seed,
            )
        )
    return stations


def truth_arc_tracking_network_from_eci_history(
    truth_history: Sequence[Sequence[float]],
    *,
    duration_s: float,
    step_s: float,
    station_spacing_s: float,
    station_longitude_offset_deg: float,
    range_sigma_km: float,
    range_rate_sigma_km_s: float,
    angle_sigma_deg: float,
    epoch_jd_utc: float,
    seed: int = 11,
    frame_context: FrameContext | None = None,
) -> list[dict[str, Any]]:
    """Create deterministic synthetic ground sites along a Cartesian truth arc."""

    duration = _finite_float(duration_s, name="duration_s", minimum=0.0)
    spacing = _finite_float(station_spacing_s, name="station_spacing_s", positive=True)
    step = _finite_float(step_s, name="step_s", positive=True)
    epoch = _finite_float(epoch_jd_utc, name="epoch_jd_utc")
    longitude_offset = _finite_float(
        station_longitude_offset_deg,
        name="station_longitude_offset_deg",
    )
    range_sigma = _finite_float(range_sigma_km, name="range_sigma_km", minimum=0.0)
    range_rate_sigma = _finite_float(range_rate_sigma_km_s, name="range_rate_sigma_km_s", minimum=0.0)
    angle_sigma = _finite_float(angle_sigma_deg, name="angle_sigma_deg", minimum=0.0)

    truth = np.array(truth_history, dtype=float)
    if truth.ndim != 2 or truth.shape[1] < 6:
        raise ValueError("truth_history must be a two-dimensional array with at least six columns.")
    if not np.all(np.isfinite(truth)):
        raise ValueError("truth_history must contain only finite values.")

    stations = []
    frame_ctx = frame_context or frame_context_from_mapping(
        {},
        jd_utc_start=epoch,
        source="ground_station_synthetic",
    )
    for idx, t_s in enumerate(np.arange(0.0, duration + 1.0e-12, spacing)):
        sample_index = min(max(int(round(float(t_s) / step)), 0), truth.shape[0] - 1)
        sample_time_s = float(sample_index) * step
        state = truth[sample_index, :6]
        lat_deg, lon_deg, _ = ecef_to_geodetic_deg_km(
            transform_position(state[:3], "eci", "ecef", t_s=sample_time_s, context=frame_ctx)
        )
        stations.append(
            ground_station_measurement_site(
                station_id=f"truth_arc_site_{idx:02d}",
                lat_deg=lat_deg,
                lon_deg=lon_deg,
                station_longitude_offset_deg=longitude_offset,
                update_cadence_s=spacing,
                range_sigma_km=range_sigma,
                range_rate_sigma_km_s=range_rate_sigma,
                angle_sigma_deg=angle_sigma,
                seed=seed,
            )
        )
    return stations


def ground_station_measurement_site(
    *,
    station_id: str,
    lat_deg: float,
    lon_deg: float,
    station_longitude_offset_deg: float,
    update_cadence_s: float,
    range_sigma_km: float,
    range_rate_sigma_km_s: float,
    angle_sigma_deg: float,
    seed: int,
) -> dict[str, Any]:
    """Build the scenario YAML block for a synthetic ground-station sensor."""

    latitude = _finite_float(lat_deg, name="lat_deg")
    longitude = _finite_float(lon_deg, name="lon_deg")
    longitude_offset = _finite_float(
        station_longitude_offset_deg,
        name="station_longitude_offset_deg",
    )
    cadence = _finite_float(update_cadence_s, name="update_cadence_s", positive=True)
    range_sigma = _finite_float(range_sigma_km, name="range_sigma_km", minimum=0.0)
    range_rate_sigma = _finite_float(range_rate_sigma_km_s, name="range_rate_sigma_km_s", minimum=0.0)
    angle_sigma = _finite_float(angle_sigma_deg, name="angle_sigma_deg", minimum=0.0)
    if not -90.0 <= latitude <= 90.0:
        raise ValueError("lat_deg must be between -90 and 90 degrees.")
    return {
        "id": str(station_id),
        "lat_deg": latitude,
        "lon_deg": float(((longitude + longitude_offset + 180.0) % 360.0) - 180.0),
        "alt_km": 0.0,
        "min_elevation_deg": 0.0,
        "max_range_km": 100000.0,
        "measurements": {
            "enabled": True,
            "measurement_type": "az_el_range_rate",
            "update_cadence_s": cadence,
            "seed": int(seed),
            "noise": {
                "range_sigma_km": range_sigma,
                "range_rate_sigma_km_s": range_rate_sigma,
                "angle_sigma_deg": angle_sigma,
            },
        },
    }


def _finite_float(
    value: Any,
    *,
    name: str,
    minimum: float | None = None,
    positive: bool = False,
) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number.") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be a finite number.")
    if positive and parsed <= 0.0:
        raise ValueError(f"{name} must be positive.")
    if minimum is not None and parsed < float(minimum):
        raise ValueError(f"{name} must be >= {float(minimum):g}.")
    return parsed


def ground_station_measurement_count(payload: dict[str, Any]) -> int:
    """Count target measurements in a ground-station measurement payload."""

    measurements = dict(payload.get("ground_station_measurements", payload) or {})
    count = 0
    for station in measurements.values():
        targets = dict(dict(station or {}).get("targets", {}) or {})
        for target in targets.values():
            target_row = dict(target or {})
            count += int(
                target_row.get(
                    "usable_measurement_count",
                    target_row.get("measurement_count", 0),
                )
                or 0
            )
    return count


def offset_cartesian_state(
    truth_state: Sequence[float],
    position_offset_m: Sequence[float],
    velocity_offset_mm_s: Sequence[float],
) -> np.ndarray:
    """Apply meter and millimeter-per-second offsets to an ECI Cartesian state."""

    truth = np.array(truth_state, dtype=float).reshape(6)
    pos_offset = np.array(position_offset_m, dtype=float).reshape(3) / 1000.0
    vel_offset = np.array(velocity_offset_mm_s, dtype=float).reshape(3) / 1.0e6
    return truth + np.hstack((pos_offset, vel_offset))


def state_comparison_payload(
    *,
    truth_state: Sequence[float],
    prior_state: Sequence[float],
    fitted_state: Sequence[float],
) -> dict[str, Any]:
    """Compare truth, prior, and fitted Cartesian states in user-facing units."""

    truth = np.array(truth_state, dtype=float).reshape(6)
    prior = np.array(prior_state, dtype=float).reshape(6)
    fitted = np.array(fitted_state, dtype=float)
    if fitted.size != 6:
        fitted = np.full(6, np.nan, dtype=float)
    else:
        fitted = fitted.reshape(6)
    prior_err = prior - truth
    fitted_err = fitted - truth
    return {
        "prior_position_error_m": float(np.linalg.norm(prior_err[:3]) * 1000.0),
        "prior_velocity_error_mm_s": float(np.linalg.norm(prior_err[3:]) * 1.0e6),
        "fitted_position_error_m": float(np.linalg.norm(fitted_err[:3]) * 1000.0),
        "fitted_velocity_error_mm_s": float(np.linalg.norm(fitted_err[3:]) * 1.0e6),
        "truth_state_eci_km_s": truth.tolist(),
        "prior_state_eci_km_s": prior.tolist(),
        "fitted_state_eci_km_s": fitted.tolist(),
    }


def tle_element_comparison_payload(
    *,
    prior: TLEElements,
    truth: TLEElements,
    estimated: dict[str, Any],
) -> dict[str, Any]:
    """Compare prior, truth, and estimated SGP4 mean elements."""

    fields = [
        "inclination_deg",
        "raan_deg",
        "eccentricity",
        "argp_deg",
        "mean_anomaly_deg",
        "mean_motion_rev_per_day",
    ]
    rows: dict[str, dict[str, float]] = {}
    for field in fields:
        prior_value = float(getattr(prior, field))
        truth_value = float(getattr(truth, field))
        estimated_value = float(estimated.get(field, np.nan))
        error = angle_delta_deg(estimated_value, truth_value) if field.endswith("_deg") else estimated_value - truth_value
        prior_error = angle_delta_deg(prior_value, truth_value) if field.endswith("_deg") else prior_value - truth_value
        rows[field] = {
            "prior": prior_value,
            "truth": truth_value,
            "estimated": estimated_value,
            "estimated_minus_truth": error,
            "prior_minus_truth": prior_error,
        }
    return {"elements": rows}


def tle_line2_with_mean_fields(
    line2: str,
    *,
    mean_anomaly_deg: float,
    mean_motion_rev_per_day: float,
) -> str:
    """Edit TLE line 2 mean fields and recompute its checksum."""

    anomaly_field = f"{float(mean_anomaly_deg) % 360.0:8.4f}"
    motion_field = f"{float(mean_motion_rev_per_day):11.8f}"
    updated = f"{line2[:43]}{anomaly_field}{line2[51:52]}{motion_field}{line2[63:]}"
    return with_tle_checksum(updated)


def angle_delta_deg(a_deg: float, b_deg: float) -> float:
    return float(((float(a_deg) - float(b_deg) + 180.0) % 360.0) - 180.0)


truth_arc_tracking_network_from_sgp4 = truth_arc_tracking_network_from_ogp
