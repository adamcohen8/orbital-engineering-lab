"""ITU-R P.837-8 Annex 1 rainfall-rate climatology over official external maps."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat
from scipy.special import erfc

from sim.pro_communications.contracts import AtmosphericProfile, SourceProvenance

P837_MODEL_EDITION = "ITU-R P.837-8"
P837_REFERENCE_URL = "https://www.itu.int/rec/R-REC-P.837-8-202509-I/en"
P837_ANNEX_SOFTWARE_URL = "https://www.itu.int/en/ITU-R/study-groups/rsg3/rwp3m/Software%20Products/P_837_8_Annex_1.zip"
P837_MT_T_FILENAME = "_ITU-R_P0837-8_MT_T_maps.mat"
P837_R001_LOW_FILENAME = "_ITU-R_P0837-8_R001_map_low_resolution.mat"
P837_R001_HIGH_FILENAME = "_ITU-R_P0837-8_R001_map.mat"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class P837MapSet:
    root: Path
    monthly_maps: Path
    r001_map: Path
    resolution_deg: float
    monthly_maps_sha256: str
    r001_map_sha256: str


@dataclass(frozen=True)
class P837RainfallRateResult:
    latitude_deg: float
    longitude_deg: float
    exceedance_percentage: tuple[float, ...]
    annual_rainfall_rate_mm_h: tuple[float, ...]
    annual_rain_probability_percentage: float
    monthly_rain_amount_mm: tuple[float, ...]
    monthly_temperature_c: tuple[float, ...]
    model_edition: str
    map_resolution_deg: float
    map_hashes: dict[str, str]
    semantic_sha256: str

    def rate_at(self, exceedance_percentage: float) -> float:
        target = float(exceedance_percentage)
        for probability, rate in zip(
            self.exceedance_percentage,
            self.annual_rainfall_rate_mm_h,
            strict=True,
        ):
            if probability == target:
                return rate
        raise KeyError(f"P.837 result does not contain p={target!r}%.")


def resolve_p837_map_set(root: str | Path, *, high_resolution: bool = False) -> P837MapSet:
    """Resolve and hash an extracted official P.837-8 Annex 1 package."""

    source = Path(root).expanduser().resolve()
    monthly = source / P837_MT_T_FILENAME
    r001 = source / (P837_R001_HIGH_FILENAME if high_resolution else P837_R001_LOW_FILENAME)
    missing = [path.name for path in (monthly, r001) if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Official P.837-8 map files are missing: {missing}.")
    return P837MapSet(
        root=source,
        monthly_maps=monthly,
        r001_map=r001,
        resolution_deg=0.125 if high_resolution else 0.25,
        monthly_maps_sha256=_sha256(monthly),
        r001_map_sha256=_sha256(r001),
    )


@lru_cache(maxsize=4)
def _load_maps(monthly_path: str, r001_path: str) -> tuple[dict[str, Any], dict[str, Any]]:
    return loadmat(monthly_path), loadmat(r001_path)


def _bilinear(longitude: np.ndarray, latitude: np.ndarray, values: np.ndarray, lon: float, lat: float) -> float:
    x = np.asarray(longitude[0, :], dtype=float)
    y = np.asarray(latitude[:, 0], dtype=float)
    z = np.asarray(values, dtype=float)
    if y[0] > y[-1]:
        y = y[::-1]
        z = z[::-1, :]
    if not (x[0] <= lon <= x[-1] and y[0] <= lat <= y[-1]):
        raise ValueError("Requested coordinate lies outside the supplied P.837 map grid.")
    ix = min(max(int(np.searchsorted(x, lon, side="right") - 1), 0), x.size - 2)
    iy = min(max(int(np.searchsorted(y, lat, side="right") - 1), 0), y.size - 2)
    x0, x1 = x[ix], x[ix + 1]
    y0, y1 = y[iy], y[iy + 1]
    tx = (lon - x0) / (x1 - x0)
    ty = (lat - y0) / (y1 - y0)
    return float(
        (1.0 - tx) * (1.0 - ty) * z[iy, ix]
        + tx * (1.0 - ty) * z[iy, ix + 1]
        + (1.0 - tx) * ty * z[iy + 1, ix]
        + tx * ty * z[iy + 1, ix + 1]
    )


def evaluate_itu_p837_8_rainfall_rate(
    maps: P837MapSet,
    *,
    exceedance_percentage: tuple[float, ...],
    latitude_deg: float,
    longitude_deg: float,
) -> P837RainfallRateResult:
    """Evaluate annual P.837-8 Annex 1 rain rates using official map files."""

    if not isinstance(maps, P837MapSet):
        raise ValueError("maps must be a resolved P837MapSet.")
    latitude = float(latitude_deg)
    longitude = float(longitude_deg)
    if not np.isfinite(latitude) or not -90.0 <= latitude <= 90.0:
        raise ValueError("latitude_deg must be finite within [-90, 90].")
    if not np.isfinite(longitude):
        raise ValueError("longitude_deg must be finite.")
    longitude = ((longitude + 180.0) % 360.0) - 180.0
    probabilities = tuple(float(value) for value in exceedance_percentage)
    if not probabilities or any(not np.isfinite(value) or value <= 0.0 or value > 100.0 for value in probabilities):
        raise ValueError("exceedance_percentage values must be finite within (0, 100].")

    monthly_maps, r001_maps = _load_maps(str(maps.monthly_maps), str(maps.r001_map))
    rain_amount = np.asarray(
        [
            _bilinear(
                monthly_maps["lon_025"],
                monthly_maps["lat_025"],
                monthly_maps["Mt_025"][:, :, month],
                longitude,
                latitude,
            )
            for month in range(12)
        ]
    )
    temperature_c = np.asarray(
        [
            _bilinear(
                monthly_maps["lon_075"],
                monthly_maps["lat_075"],
                monthly_maps["T_075"][:, :, month],
                longitude,
                latitude,
            )
            - 273.15
            for month in range(12)
        ]
    )
    days = np.asarray([31.0, 28.25, 31.0, 30.0, 31.0, 30.0, 31.0, 31.0, 30.0, 31.0, 30.0, 31.0])
    mean_rain_rate = 0.5874 * np.exp(0.0883 * temperature_c)
    mean_rain_rate = np.where(temperature_c < 0.0, 0.5874, mean_rain_rate)
    monthly_rain_probability = 100.0 * rain_amount / (24.0 * days * mean_rain_rate)
    capped = monthly_rain_probability > 70.0
    monthly_rain_probability = np.where(capped, 70.0, monthly_rain_probability)
    mean_rain_rate = np.where(capped, 100.0 * rain_amount / (70.0 * 24.0 * days), mean_rain_rate)
    annual_rain_probability = float(np.dot(monthly_rain_probability, days) / np.sum(days))
    initial_rate = _bilinear(
        r001_maps["longitude"],
        r001_maps["latitude"],
        r001_maps["R001"],
        longitude,
        latitude,
    )
    if initial_rate == 0.0:
        initial_rate = 10.0
    sigma = 1.26
    rates = []
    for probability in probabilities:
        if probability > annual_rain_probability:
            rates.append(0.0)
            continue
        low, high, reference_rate = 0.0, 500.0, initial_rate
        for _iteration in range(128):
            conditional = 0.5 * erfc(
                (np.log(reference_rate) + sigma**2 / 2.0 - np.log(mean_rain_rate)) / (sigma * np.sqrt(2.0))
            )
            final_probability = float(np.dot(monthly_rain_probability * conditional, days) / np.sum(days))
            relative_error_percent = 100.0 * (final_probability - probability) / probability
            if abs(relative_error_percent) <= 1.0e-3:
                break
            if final_probability < probability:
                high = reference_rate
            else:
                low = reference_rate
            reference_rate = 0.5 * (high + low)
        else:
            raise RuntimeError("P.837-8 bisection did not satisfy the reference tolerance.")
        rates.append(float(reference_rate))

    identity = {
        "model_edition": P837_MODEL_EDITION,
        "latitude_deg": latitude,
        "longitude_deg": longitude,
        "exceedance_percentage": probabilities,
        "annual_rainfall_rate_mm_h": rates,
        "annual_rain_probability_percentage": annual_rain_probability,
        "monthly_rain_amount_mm": rain_amount.tolist(),
        "monthly_temperature_c": temperature_c.tolist(),
        "map_resolution_deg": maps.resolution_deg,
        "map_hashes": {
            maps.monthly_maps.name: maps.monthly_maps_sha256,
            maps.r001_map.name: maps.r001_map_sha256,
        },
    }
    semantic_sha256 = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    return P837RainfallRateResult(
        latitude_deg=latitude,
        longitude_deg=longitude,
        exceedance_percentage=probabilities,
        annual_rainfall_rate_mm_h=tuple(rates),
        annual_rain_probability_percentage=annual_rain_probability,
        monthly_rain_amount_mm=tuple(float(value) for value in rain_amount),
        monthly_temperature_c=tuple(float(value) for value in temperature_c),
        model_edition=P837_MODEL_EDITION,
        map_resolution_deg=maps.resolution_deg,
        map_hashes=identity["map_hashes"],
        semantic_sha256=semantic_sha256,
    )


def apply_p837_rainfall_rate(
    atmosphere: AtmosphericProfile,
    result: P837RainfallRateResult,
    *,
    exceedance_percentage: float = 0.01,
) -> AtmosphericProfile:
    """Return a source-documented atmospheric profile using one P.837-8 rate."""

    if not isinstance(atmosphere, AtmosphericProfile) or not isinstance(result, P837RainfallRateResult):
        raise ValueError("atmosphere and result must be typed Pro communications products.")
    return replace(
        atmosphere,
        rainfall_rate_mm_h=result.rate_at(exceedance_percentage),
        station_latitude_deg=result.latitude_deg,
        rain_exceedance_percentage=float(exceedance_percentage),
        provenance=SourceProvenance(
            source_id="itu-r-p837-8-annex1-official-maps",
            source_version=P837_MODEL_EDITION,
            reference=P837_REFERENCE_URL,
            applicability="Climatological rainfall-rate prediction where reliable local long-term data is absent.",
            sha256=result.semantic_sha256,
        ),
    )


__all__ = [
    "P837_ANNEX_SOFTWARE_URL",
    "P837_MODEL_EDITION",
    "P837_REFERENCE_URL",
    "P837MapSet",
    "P837RainfallRateResult",
    "apply_p837_rainfall_rate",
    "evaluate_itu_p837_8_rainfall_rate",
    "resolve_p837_map_set",
]
