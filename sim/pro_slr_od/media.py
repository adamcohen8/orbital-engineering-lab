"""Deterministic optical troposphere for the private SLR research workflow.

Mendes--Pavlis zenith delay and FCULa mapping (IERS 2010, chapter 9),
with CIPM-2007 water-vapor pressure. Coefficients/reference case cross-checked
against Orekit 13.0 MendesPavlisModel/CIPM2007 (CS GROUP, Apache-2.0).
This is a path-delay model, not a station/reflector calibration model.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, Callable

import numpy as np

from sim.dynamics.orbit.frames import eci_to_ecef_rotation_hpop_like
from sim.utils.geodesy import ecef_to_enu_rotation, ecef_to_geodetic_deg_km

_MAPPING = np.array([[12100.8e-7, 1729.5e-9, 319.1e-7, -1847.8e-11],
                     [30496.5e-7, 234.4e-8, -103.5e-6, -185.6e-10],
                     [6877.7e-5, 197.2e-7, -345.8e-5, 106.0e-9]])


def optical_troposphere_delay_m(*, elevation_rad: float, latitude_rad: float,
                               height_m: float, pressure_mbar: float, temperature_k: float,
                               relative_humidity_percent: float, wavelength_nm: float) -> dict[str, float]:
    """Return one-way optical path delay; no low-elevation clamping/defaults."""
    values = (elevation_rad, latitude_rad, height_m, pressure_mbar, temperature_k,
              relative_humidity_percent, wavelength_nm)
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("Optical troposphere inputs must be finite.")
    if not math.radians(3.0) <= elevation_rad <= math.pi / 2.0 + 1.0e-12:
        raise ValueError("Optical troposphere requires geometric elevation in [3, 90] degrees.")
    if not (-math.pi / 2 <= latitude_rad <= math.pi / 2 and -500 <= height_m <= 6000
            and 200 <= pressure_mbar <= 1100 and 200 <= temperature_k <= 330
            and 0 <= relative_humidity_percent <= 100 and 355 <= wavelength_nm <= 1064):
        raise ValueError("Optical troposphere input is outside the supported physical envelope.")
    sigma2 = (1000.0 / wavelength_nm)**2
    hydro_dispersion = 0.01 * 0.99995995 * (
        19990.975 * (238.0185 + sigma2) / (238.0185 - sigma2)**2
        + 579.55174 * (57.362 + sigma2) / (57.362 - sigma2)**2)
    wet_dispersion = 0.003101 * (295.235 + 3 * 2.6422 * sigma2
        - 5 * 0.032380 * sigma2**2 + 7 * 0.004028 * sigma2**3)
    celsius = temperature_k - 273.15
    vapor_pa = (relative_humidity_percent / 100.0) * (
        1.00062 + 3.14e-6 * pressure_mbar + 5.6e-7 * celsius**2) * math.exp(
        temperature_k * (temperature_k * 1.2378847e-5 - 1.9121316e-2)
        + 33.93711047 - 6343.1645 / temperature_k)
    site = 1.0 - 0.00266 * math.cos(2 * latitude_rad) - 2.8e-7 * height_m
    hydro = 0.002416579 * pressure_mbar * hydro_dispersion / site
    wet = 1.0e-6 * (5.316 * wet_dispersion - 3.759 * hydro_dispersion) * vapor_pa / site
    a, b, c = _MAPPING @ np.array([1.0, celsius, math.cos(latitude_rad), height_m])
    sine = math.sin(elevation_rad)
    mapping = (1 + a / (1 + b / (1 + c))) / (sine + a / (sine + b / (sine + c)))
    return {"zenith_hydrostatic_m": float(hydro), "zenith_wet_m": float(wet),
            "mapping_factor": float(mapping), "one_way_delay_m": float((hydro + wet) * mapping)}


def attach_slr_media_inputs(rows: Sequence[Mapping[str, Any]], packet: Mapping[str, Any],
                            *, maximum_weather_age_s: float = 3600.0) -> list[dict[str, Any]]:
    """Bind each normal point to its own session/configuration and measured weather.

    Linear interpolation inside the session weather span, nearest boundary
    outside it, with an explicit one-hour age limit. No standard atmosphere
    or cross-session weather substitution. Midnight wrapping uses the day
    nearest the normal point, including weather before session start.
    """
    if not math.isfinite(maximum_weather_age_s) or maximum_weather_age_s <= 0.0:
        raise ValueError("Weather age limit must be finite and positive.")
    result = []
    for source in rows:
        row = dict(source)
        if bool(row["correction_flags"].get("tropospheric_refraction_applied", False)):
            row["slr_media"] = {"already_applied": True}
            result.append(row)
            continue
        _, session_index, point_index = str(row["measurement_id"]).split(":")
        session = packet["sessions"][int(session_index)]
        point = session["normal_points"][int(point_index)]
        if session["station"]["code"].upper() != row["station_id"]:
            raise ValueError("SLR media session/station identity differs from the observation.")
        if not np.isclose(point["time_jd_utc"], row["time_jd_utc"], rtol=0.0, atol=1.0e-10):
            raise ValueError("SLR media epoch differs from the observation.")
        wavelengths = {float(record["raw"].split()[2]) for record in session["configuration_records"]
                       if record["record_type"] == "C0"
                       and record["raw"].split()[3] == point["system_configuration_id"]}
        if len(wavelengths) != 1 or not session["meteorology"]:
            raise ValueError("Uncorrected SLR observation requires unique configuration wavelength and measured meteorology.")
        jd = float(row["time_jd_utc"])
        midnight = math.floor(jd - 0.5) + 0.5
        weather = []
        for measurement in session["meteorology"]:
            met_jd = midnight + float(measurement["seconds_of_day"]) / 86400.0
            met_jd += round(jd - met_jd)
            weather.append(((met_jd - jd) * 86400.0, measurement))
        weather.sort(key=lambda item: item[0])
        offsets = np.array([item[0] for item in weather])
        if np.min(np.abs(offsets)) > maximum_weather_age_s:
            raise ValueError("SLR observation has no sufficiently fresh session meteorology.")
        if np.any(np.diff(offsets) <= 0.0):
            raise ValueError("SLR session meteorology epochs must be distinct.")
        conditions = {key: float(np.interp(0.0, offsets, [item[1][key] for item in weather]))
                      for key in ("pressure_mbar", "temperature_k", "relative_humidity_percent")}
        row["slr_media"] = {**conditions, "wavelength_nm": wavelengths.pop(),
                            "nearest_weather_age_s": float(np.min(np.abs(offsets))),
                            "weather_policy": "session_linear_interpolation_boundary_hold_maximum_age_3600s"}
        result.append(row)
    return result


def two_way_optical_media_correction(row: Mapping[str, Any], geometry: Mapping[str, Any],
                                    state_at_jd: Callable[[float], np.ndarray], *,
                                    frame_model: str, eop_path: str | None,
                                    state_at_offset_s: Callable[[float], np.ndarray] | None = None) -> dict[str, float]:
    """Average outbound/inbound delay as one-way-equivalent range in km.

    Uses the OEL two-way geometric bounce/transmit/receive epochs and frame
    transformations. Media delay is evaluated on the converged geometric
    legs; a coupled atmospheric light-time iteration is not performed.
    """
    if bool(row["correction_flags"].get("tropospheric_refraction_applied", False)):
        return {"one_way_equivalent_km": 0.0, "troposphere_m": 0.0}
    media = dict(row["slr_media"])
    station = np.array(row["station_metadata"]["ecef_km"], dtype=float)
    lat, lon, height = ecef_to_geodetic_deg_km(station)
    enu = ecef_to_enu_rotation(lat, lon)
    relative_time = state_at_offset_s is not None
    satellite = np.asarray(state_at_offset_s(float(geometry["bounce_offset_s"])) if relative_time
        else state_at_jd(float(geometry["bounce_jd_utc"])), dtype=float)[:3]
    delays, elevations = [], []
    for name in ("transmit_jd_utc", "receive_jd_utc"):
        jd = float(geometry[name])
        midnight = math.floor(jd - 0.5) + 0.5
        offset_key = name.replace("_jd_utc", "_offset_s")
        rotation = eci_to_ecef_rotation_hpop_like(
            float(geometry[offset_key]) + float(geometry.get("epoch_offset_s", 0.)) if relative_time else (jd - midnight) * 86400.0,
            jd_utc_start=float(row["time_jd_utc"]) if relative_time else midnight,
            eop_path=eop_path if frame_model == "hpop_like" else None)
        relative = enu @ (rotation @ satellite - station)
        elevation = math.atan2(relative[2], np.hypot(relative[0], relative[1]))
        delay = optical_troposphere_delay_m(elevation_rad=elevation, latitude_rad=math.radians(lat),
            height_m=1000.0 * height, **{key: media[key] for key in (
                "pressure_mbar", "temperature_k", "relative_humidity_percent", "wavelength_nm")})
        delays.append(delay["one_way_delay_m"])
        elevations.append(elevation)
    value = 0.5 * sum(delays)
    return {"one_way_equivalent_km": value / 1000.0, "troposphere_m": value,
            "transmit_elevation_rad": float(elevations[0]), "receive_elevation_rad": float(elevations[1])}
