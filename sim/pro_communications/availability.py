"""Statistical attenuation and availability products kept separate from epoch links."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np

from sim.pro_communications.atmosphere import (
    p618_14_rain_attenuation,
    p618_14_scintillation_fade_depth,
    p618_14_total_attenuation,
    p676_13_slant_attenuation,
    p840_9_cloud_attenuation,
)
from sim.pro_communications.contracts import AtmosphericProfile

STATISTICAL_AVAILABILITY_CONTRACT_VERSION = "oel.pro-statistical-availability.v0.2"


def _validated_points(points: int) -> int:
    if isinstance(points, (bool, np.bool_)) or not isinstance(points, (int, np.integer)):
        raise ValueError("points must be an integer within [31, 2001].")
    normalized = int(points)
    if not 31 <= normalized <= 2001:
        raise ValueError("points must be an integer within [31, 2001].")
    return normalized


@dataclass(frozen=True)
class StatisticalRainAvailabilityResult:
    profile: AtmosphericProfile
    carrier_frequency_hz: float
    elevation_rad: float
    clear_sky_margin_db: float
    exceedance_percentage: np.ndarray
    rain_attenuation_db: np.ndarray
    estimated_rain_outage_percentage: float | None
    availability_percentage: float | None
    disposition: str
    summary: dict[str, object]
    semantic_sha256: str


@dataclass(frozen=True)
class StatisticalAggregateAvailabilityResult:
    profile: AtmosphericProfile
    carrier_frequency_hz: float
    elevation_rad: float
    clear_path_margin_db: float
    exceedance_percentage: np.ndarray
    gaseous_attenuation_db: np.ndarray
    cloud_attenuation_db: np.ndarray
    rain_attenuation_db: np.ndarray
    scintillation_fade_depth_db: np.ndarray
    total_attenuation_db: np.ndarray
    estimated_outage_percentage: float | None
    availability_percentage: float | None
    disposition: str
    summary: dict[str, object]
    semantic_sha256: str


def _invert_curve(
    percentages: np.ndarray,
    attenuation: np.ndarray,
    margin: float,
) -> tuple[float | None, float | None, str]:
    if margin < float(attenuation[-1]):
        return None, None, "outside_envelope_outage_exceeds_5_percent"
    if margin >= float(attenuation[0]):
        return None, None, "outside_envelope_outage_below_0.001_percent"
    outage = float(np.exp(np.interp(margin, attenuation[::-1], np.log(percentages[::-1]))))
    return outage, 100.0 - outage, "interpolated_within_p618_promoted_percentage_envelope"


def evaluate_statistical_rain_availability(
    profile: AtmosphericProfile,
    *,
    carrier_frequency_hz: float,
    elevation_rad: float,
    clear_sky_margin_db: float,
    points: int = 301,
) -> StatisticalRainAvailabilityResult:
    """Invert the declared P.618 attenuation curve without calling it epoch weather."""

    if profile.rain_model != "itu-r-p.618-14-declared-r0.01":
        raise ValueError("Statistical rain availability requires the P.618-14 declared-R0.01 profile.")
    frequency_hz = float(carrier_frequency_hz)
    elevation = float(elevation_rad)
    margin = float(clear_sky_margin_db)
    if not np.isfinite(frequency_hz) or frequency_hz <= 0.0:
        raise ValueError("carrier_frequency_hz must be positive and finite.")
    if not np.isfinite(elevation) or not np.deg2rad(5.0) <= elevation <= np.pi / 2.0:
        raise ValueError("elevation_rad must lie within the promoted [5, 90] degree envelope.")
    if not np.isfinite(margin):
        raise ValueError("clear_sky_margin_db must be finite.")
    point_count = _validated_points(points)
    percentages = np.geomspace(0.001, 5.0, point_count)
    attenuation = p618_14_rain_attenuation(
        frequency_hz / 1.0e9,
        elevation_deg=np.rad2deg(elevation),
        station_latitude_deg=profile.station_latitude_deg,
        station_height_km=profile.station_height_km,
        rain_height_km=profile.rain_height_km,
        rain_rate_r001_mm_h=profile.rainfall_rate_mm_h,
        polarization_tilt_deg=profile.polarization_tilt_deg,
        exceedance_percentage=percentages,
    )
    p837_bound = profile.provenance is not None and profile.provenance.source_id == "itu-r-p837-8-annex1-official-maps"
    # Attenuation decreases with increasing exceedance percentage.  The root
    # is interpolated in log(p), the same scale used to freeze the curve.
    outage, availability, disposition = _invert_curve(percentages, attenuation, margin)
    payload = {
        "contract_version": STATISTICAL_AVAILABILITY_CONTRACT_VERSION,
        "profile_id": profile.profile_id,
        "carrier_frequency_hz": frequency_hz,
        "elevation_rad": elevation,
        "clear_sky_margin_db": margin,
        "estimated_rain_outage_percentage": outage,
        "availability_percentage": availability,
        "disposition": disposition,
        "model_edition": "ITU-R P.618-14",
        "rain_input_semantics": (
            "P.837-8 official-map-derived R0.01 climatology" if p837_bound else "directly declared R0.01"
        ),
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())
    digest.update(np.ascontiguousarray(percentages, dtype="<f8").tobytes())
    digest.update(np.ascontiguousarray(attenuation, dtype="<f8").tobytes())
    semantic_sha256 = digest.hexdigest()
    summary = {
        **payload,
        "semantic_sha256": semantic_sha256,
        "claims": [
            "Rain attenuation exceeded for declared percentages under ITU-R P.618-14 and supplied R0.01.",
            "Interpolated rain-only statistical availability when the closure threshold lies within the promoted curve.",
        ],
        "non_claims": [
            "Not deterministic epoch weather or a weather replay.",
            "Not aggregate propagation availability or service assurance.",
        ],
    }
    return StatisticalRainAvailabilityResult(
        profile=profile,
        carrier_frequency_hz=frequency_hz,
        elevation_rad=elevation,
        clear_sky_margin_db=margin,
        exceedance_percentage=percentages,
        rain_attenuation_db=attenuation,
        estimated_rain_outage_percentage=outage,
        availability_percentage=availability,
        disposition=disposition,
        summary=summary,
        semantic_sha256=semantic_sha256,
    )


def evaluate_statistical_aggregate_availability(
    profile: AtmosphericProfile,
    *,
    carrier_frequency_hz: float,
    elevation_rad: float,
    clear_path_margin_db: float,
    median_wet_refractivity_n_units: float,
    antenna_diameter_m: float,
    antenna_efficiency: float,
    points: int = 301,
) -> StatisticalAggregateAvailabilityResult:
    """Evaluate P.618-14 simultaneous gas, cloud, rain, and scintillation attenuation."""

    if profile.rain_model != "itu-r-p.618-14-declared-r0.01":
        raise ValueError("Aggregate availability requires the P.618-14 declared-R0.01 rain profile.")
    frequency_hz = float(carrier_frequency_hz)
    elevation = float(elevation_rad)
    margin = float(clear_path_margin_db)
    if not np.isfinite(frequency_hz) or frequency_hz <= 0.0:
        raise ValueError("carrier_frequency_hz must be positive and finite.")
    if not np.isfinite(elevation) or not np.deg2rad(5.0) < elevation <= np.pi / 2.0:
        raise ValueError("elevation_rad must lie within the promoted (5, 90] degree envelope.")
    if not np.isfinite(margin):
        raise ValueError("clear_path_margin_db must be finite.")
    point_count = _validated_points(points)
    percentages = np.geomspace(0.001, 5.0, point_count)
    frequency_ghz = frequency_hz / 1.0e9
    elevation_deg = float(np.rad2deg(elevation))
    rain = p618_14_rain_attenuation(
        frequency_ghz,
        elevation_deg=elevation_deg,
        station_latitude_deg=profile.station_latitude_deg,
        station_height_km=profile.station_height_km,
        rain_height_km=profile.rain_height_km,
        rain_rate_r001_mm_h=profile.rainfall_rate_mm_h,
        polarization_tilt_deg=profile.polarization_tilt_deg,
        exceedance_percentage=percentages,
    )
    scintillation = p618_14_scintillation_fade_depth(
        frequency_ghz,
        elevation_deg=elevation_deg,
        exceedance_percentage=percentages,
        median_wet_refractivity_n_units=median_wet_refractivity_n_units,
        antenna_diameter_m=antenna_diameter_m,
        antenna_efficiency=antenna_efficiency,
    )
    gas_scalar = (
        0.0
        if profile.gaseous_model == "none"
        else float(
            p676_13_slant_attenuation(
                frequency_ghz,
                elevation_deg=elevation_deg,
                total_pressure_hpa=profile.total_pressure_hpa,
                temperature_k=profile.temperature_k,
                water_vapour_density_g_m3=profile.water_vapour_density_g_m3,
            ).total_db
        )
    )
    cloud_scalar = (
        0.0
        if profile.cloud_model == "none"
        else float(
            p840_9_cloud_attenuation(
                frequency_ghz,
                elevation_deg=elevation_deg,
                integrated_cloud_liquid_water_kg_m2=profile.integrated_cloud_liquid_water_kg_m2,
            )
        )
    )
    gas = np.full(percentages.shape, gas_scalar)
    cloud = np.full(percentages.shape, cloud_scalar)
    total = p618_14_total_attenuation(
        gaseous_attenuation_db=gas,
        cloud_attenuation_db=cloud,
        rain_attenuation_db=rain,
        scintillation_fade_depth_db=scintillation,
    )
    outage, availability, disposition = _invert_curve(percentages, total, margin)
    payload = {
        "contract_version": STATISTICAL_AVAILABILITY_CONTRACT_VERSION,
        "profile_id": profile.profile_id,
        "carrier_frequency_hz": frequency_hz,
        "elevation_rad": elevation,
        "clear_path_margin_db": margin,
        "median_wet_refractivity_n_units": float(median_wet_refractivity_n_units),
        "antenna_diameter_m": float(antenna_diameter_m),
        "antenna_efficiency": float(antenna_efficiency),
        "estimated_outage_percentage": outage,
        "availability_percentage": availability,
        "disposition": disposition,
        "model_edition": "ITU-R P.618-14",
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())
    for values in (percentages, gas, cloud, rain, scintillation, total):
        digest.update(np.ascontiguousarray(values, dtype="<f8").tobytes())
    semantic_sha256 = digest.hexdigest()
    summary = {
        **payload,
        "semantic_sha256": semantic_sha256,
        "claims": [
            "P.618-14 simultaneous-source statistical attenuation under declared gas, cloud, rain, wet-refractivity, and antenna inputs.",
            "Interpolated aggregate statistical availability when the closure threshold lies within the promoted curve.",
        ],
        "non_claims": [
            "Not deterministic epoch weather, a forecast, or a weather replay.",
            "Not measured site availability, packet delivery, or service assurance.",
        ],
    }
    return StatisticalAggregateAvailabilityResult(
        profile=profile,
        carrier_frequency_hz=frequency_hz,
        elevation_rad=elevation,
        clear_path_margin_db=margin,
        exceedance_percentage=percentages,
        gaseous_attenuation_db=gas,
        cloud_attenuation_db=cloud,
        rain_attenuation_db=rain,
        scintillation_fade_depth_db=scintillation,
        total_attenuation_db=total,
        estimated_outage_percentage=outage,
        availability_percentage=availability,
        disposition=disposition,
        summary=summary,
        semantic_sha256=semantic_sha256,
    )


__all__ = [
    "STATISTICAL_AVAILABILITY_CONTRACT_VERSION",
    "StatisticalRainAvailabilityResult",
    "StatisticalAggregateAvailabilityResult",
    "evaluate_statistical_aggregate_availability",
    "evaluate_statistical_rain_availability",
]
