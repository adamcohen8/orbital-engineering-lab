"""Native, version-pinned atmospheric loss methods for communications studies."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from sim.pro_communications.contracts import AtmosphericProfile

P676_EDITION = "ITU-R P.676-13"
P618_EDITION = "ITU-R P.618-14"
P838_EDITION = "ITU-R P.838-3"
P840_EDITION = "ITU-R P.840-9"

# f0, a1, a2, a3, a4, a5, a6; Recommendation ITU-R P.676-13, Table 1.
_OXYGEN_LINES = np.asarray(
    [
        (50.474214, 0.975, 9.651, 6.690, 0.0, 2.566, 6.850),
        (50.987745, 2.529, 8.653, 7.170, 0.0, 2.246, 6.800),
        (51.503360, 6.193, 7.709, 7.640, 0.0, 1.947, 6.729),
        (52.021429, 14.320, 6.819, 8.110, 0.0, 1.667, 6.640),
        (52.542418, 31.240, 5.983, 8.580, 0.0, 1.388, 6.526),
        (53.066934, 64.290, 5.201, 9.060, 0.0, 1.349, 6.206),
        (53.595775, 124.600, 4.474, 9.550, 0.0, 2.227, 5.085),
        (54.130025, 227.300, 3.800, 9.960, 0.0, 3.170, 3.750),
        (54.671180, 389.700, 3.182, 10.370, 0.0, 3.558, 2.654),
        (55.221384, 627.100, 2.618, 10.890, 0.0, 2.560, 2.952),
        (55.783815, 945.300, 2.109, 11.340, 0.0, -1.172, 6.135),
        (56.264774, 543.400, 0.014, 17.030, 0.0, 3.525, -0.978),
        (56.363399, 1331.800, 1.654, 11.890, 0.0, -2.378, 6.547),
        (56.968211, 1746.600, 1.255, 12.230, 0.0, -3.545, 6.451),
        (57.612486, 2120.100, 0.910, 12.620, 0.0, -5.416, 6.056),
        (58.323877, 2363.700, 0.621, 12.950, 0.0, -1.932, 0.436),
        (58.446588, 1442.100, 0.083, 14.910, 0.0, 6.768, -1.273),
        (59.164204, 2379.900, 0.387, 13.530, 0.0, -6.561, 2.309),
        (59.590983, 2090.700, 0.207, 14.080, 0.0, 6.957, -0.776),
        (60.306056, 2103.400, 0.207, 14.150, 0.0, -6.395, 0.699),
        (60.434778, 2438.000, 0.386, 13.390, 0.0, 6.342, -2.825),
        (61.150562, 2479.500, 0.621, 12.920, 0.0, 1.014, -0.584),
        (61.800158, 2275.900, 0.910, 12.630, 0.0, 5.014, -6.619),
        (62.411220, 1915.400, 1.255, 12.170, 0.0, 3.029, -6.759),
        (62.486253, 1503.000, 0.083, 15.130, 0.0, -4.499, 0.844),
        (62.997984, 1490.200, 1.654, 11.740, 0.0, 1.856, -6.675),
        (63.568526, 1078.000, 2.108, 11.340, 0.0, 0.658, -6.139),
        (64.127775, 728.700, 2.617, 10.880, 0.0, -3.036, -2.895),
        (64.678910, 461.300, 3.181, 10.380, 0.0, -3.968, -2.590),
        (65.224078, 274.000, 3.800, 9.960, 0.0, -3.528, -3.680),
        (65.764779, 153.000, 4.473, 9.550, 0.0, -2.548, -5.002),
        (66.302096, 80.400, 5.200, 9.060, 0.0, -1.660, -6.091),
        (66.836834, 39.800, 5.982, 8.580, 0.0, -1.680, -6.393),
        (67.369601, 18.560, 6.818, 8.110, 0.0, -1.956, -6.475),
        (67.900868, 8.172, 7.708, 7.640, 0.0, -2.216, -6.545),
        (68.431006, 3.397, 8.652, 7.170, 0.0, -2.492, -6.600),
        (68.960312, 1.334, 9.650, 6.690, 0.0, -2.773, -6.650),
        (118.750334, 940.300, 0.010, 16.640, 0.0, -0.439, 0.079),
        (368.498246, 67.400, 0.048, 16.400, 0.0, 0.000, 0.000),
        (424.763020, 637.700, 0.044, 16.400, 0.0, 0.000, 0.000),
        (487.249273, 237.400, 0.049, 16.000, 0.0, 0.000, 0.000),
        (715.392902, 98.100, 0.145, 16.000, 0.0, 0.000, 0.000),
        (773.839490, 572.300, 0.141, 16.200, 0.0, 0.000, 0.000),
        (834.145546, 183.100, 0.145, 14.700, 0.0, 0.000, 0.000),
    ],
    dtype=float,
)

# f0, b1, b2, b3, b4, b5, b6; Recommendation ITU-R P.676-13, Table 2.
_WATER_LINES = np.asarray(
    [
        (22.235080, 0.1079, 2.144, 26.38, 0.76, 5.087, 1.00),
        (67.803960, 0.0011, 8.732, 28.58, 0.69, 4.930, 0.82),
        (119.995940, 0.0007, 8.353, 29.48, 0.70, 4.780, 0.79),
        (183.310087, 2.2730, 0.668, 29.06, 0.77, 5.022, 0.85),
        (321.225630, 0.0470, 6.179, 24.04, 0.67, 4.398, 0.54),
        (325.152888, 1.5140, 1.541, 28.23, 0.64, 4.893, 0.74),
        (336.227764, 0.0010, 9.825, 26.93, 0.69, 4.740, 0.61),
        (380.197353, 11.670, 1.048, 28.11, 0.54, 5.063, 0.89),
        (390.134508, 0.0045, 7.347, 21.52, 0.63, 4.810, 0.55),
        (437.346667, 0.0632, 5.048, 18.45, 0.60, 4.230, 0.48),
        (439.150807, 0.9098, 3.595, 20.07, 0.63, 4.483, 0.52),
        (443.018343, 0.1920, 5.048, 15.55, 0.60, 5.083, 0.50),
        (448.001085, 10.410, 1.405, 25.64, 0.66, 5.028, 0.67),
        (470.888999, 0.3254, 3.597, 21.34, 0.66, 4.506, 0.65),
        (474.689092, 1.2600, 2.379, 23.20, 0.65, 4.804, 0.64),
        (488.490108, 0.2529, 2.852, 25.86, 0.69, 5.201, 0.72),
        (503.568532, 0.0372, 6.731, 16.12, 0.61, 3.980, 0.43),
        (504.482692, 0.0124, 6.731, 16.12, 0.61, 4.010, 0.45),
        (547.676440, 0.9785, 0.158, 26.00, 0.70, 4.500, 1.00),
        (552.020960, 0.1840, 0.158, 26.00, 0.70, 4.500, 1.00),
        (556.935985, 497.00, 0.159, 30.86, 0.69, 4.552, 1.00),
        (620.700807, 5.0150, 2.391, 24.38, 0.71, 4.856, 0.68),
        (645.766085, 0.0067, 8.633, 18.00, 0.60, 4.000, 0.50),
        (658.005280, 0.2732, 7.816, 32.10, 0.69, 4.140, 1.00),
        (752.033113, 243.40, 0.396, 30.86, 0.68, 4.352, 0.84),
        (841.051732, 0.0134, 8.177, 15.90, 0.33, 5.760, 0.45),
        (859.965698, 0.1325, 8.055, 30.60, 0.68, 4.090, 0.84),
        (899.303175, 0.0547, 7.914, 29.85, 0.68, 4.530, 0.90),
        (902.611085, 0.0386, 8.429, 28.65, 0.70, 5.100, 0.95),
        (906.205957, 0.1836, 5.110, 24.08, 0.70, 4.700, 0.53),
        (916.171582, 8.4000, 1.441, 26.73, 0.70, 5.150, 0.78),
        (923.112692, 0.0079, 10.293, 29.00, 0.70, 5.000, 0.80),
        (970.315022, 9.0090, 1.919, 25.50, 0.64, 4.940, 0.67),
        (987.926764, 134.60, 0.257, 29.85, 0.68, 4.550, 0.90),
        (1780.000000, 17506.0, 0.952, 196.3, 2.00, 24.15, 5.00),
    ],
    dtype=float,
)


@dataclass(frozen=True)
class GaseousSpecificAttenuation:
    oxygen_db_per_km: np.ndarray
    water_vapour_db_per_km: np.ndarray
    total_db_per_km: np.ndarray


@dataclass(frozen=True)
class GaseousSlantAttenuation:
    oxygen_db: np.ndarray
    water_vapour_db: np.ndarray
    total_db: np.ndarray
    oxygen_effective_height_km: np.ndarray
    water_vapour_effective_height_km: np.ndarray


@dataclass(frozen=True)
class AtmosphericLossLedger:
    gaseous_db: np.ndarray
    rain_db: np.ndarray
    cloud_db: np.ndarray
    total_db: np.ndarray
    statistical_components: tuple[str, ...]
    model_editions: tuple[str, ...]


def _positive_finite(values: float | np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if np.any(~np.isfinite(array)) or np.any(array <= 0.0):
        raise ValueError(f"{name} must contain positive finite values.")
    return array


def _elevation(values: float | np.ndarray, *, minimum_deg: float = 5.0) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if np.any(~np.isfinite(array)) or np.any(array < minimum_deg) or np.any(array > 90.0):
        raise ValueError(f"elevation_deg must be within [{minimum_deg}, 90].")
    return array


def p676_13_specific_attenuation(
    frequency_ghz: float | np.ndarray,
    *,
    total_pressure_hpa: float | np.ndarray,
    temperature_k: float | np.ndarray,
    water_vapour_density_g_m3: float | np.ndarray,
    numeric_backend: str = "rust",
) -> GaseousSpecificAttenuation:
    """Evaluate P.676-13 Annex 1 equations (1)-(9), vectorized by case."""

    frequency, pressure, temperature, density = np.broadcast_arrays(
        _positive_finite(frequency_ghz, "frequency_ghz"),
        _positive_finite(total_pressure_hpa, "total_pressure_hpa"),
        _positive_finite(temperature_k, "temperature_k"),
        np.asarray(water_vapour_density_g_m3, dtype=float),
    )
    if np.any(~np.isfinite(density)) or np.any(density < 0.0):
        raise ValueError("water_vapour_density_g_m3 must contain nonnegative finite values.")
    if np.any(frequency > 1000.0):
        raise ValueError("P.676-13 line-by-line specific attenuation supports at most 1000 GHz.")
    theta = 300.0 / temperature
    vapour_pressure = density * temperature / 216.7
    dry_pressure = pressure - vapour_pressure
    if np.any(dry_pressure <= 0.0):
        raise ValueError("Water vapour partial pressure must remain below total pressure.")

    backend = str(numeric_backend).strip().lower()
    if backend not in {"python", "rust"}:
        raise ValueError(f"Unknown atmospheric numeric backend {numeric_backend!r}.")
    if backend == "rust":
        from sim.pro_communications.rust_atmosphere import specific_batch

        oxygen, water = specific_batch(frequency, pressure, temperature, density, _OXYGEN_LINES, _WATER_LINES)
        return GaseousSpecificAttenuation(oxygen_db_per_km=oxygen,
            water_vapour_db_per_km=water, total_db_per_km=oxygen + water)

    shape = frequency.shape
    f = frequency.reshape(-1, 1)
    p = dry_pressure.reshape(-1, 1)
    e = vapour_pressure.reshape(-1, 1)
    th = theta.reshape(-1, 1)

    f0, a1, a2, a3, a4, a5, a6 = _OXYGEN_LINES.T
    strength_o = a1 * 1.0e-7 * p * th**3.0 * np.exp(a2 * (1.0 - th))
    width_o = a3 * 1.0e-4 * (p * th ** (0.8 - a4) + 1.1 * e * th)
    width_o = np.sqrt(width_o**2 + 2.25e-6)
    delta_o = (a5 + a6 * th) * 1.0e-4 * (p + e) * th**0.8
    line_shape_o = (f / f0) * (
        (width_o - delta_o * (f0 - f)) / ((f0 - f) ** 2 + width_o**2)
        + (width_o - delta_o * (f0 + f)) / ((f0 + f) ** 2 + width_o**2)
    )
    debye_width = 5.6e-4 * (p + e) * th**0.8
    dry_continuum = (
        f
        * p
        * th**2
        * (6.14e-5 / (debye_width * (1.0 + (f / debye_width) ** 2)) + 1.4e-12 * p * th**1.5 / (1.0 + 1.9e-5 * f**1.5))
    )
    refractivity_o = np.sum(strength_o * line_shape_o, axis=1) + dry_continuum[:, 0]

    f0, b1, b2, b3, b4, b5, b6 = _WATER_LINES.T
    strength_w = b1 * 1.0e-1 * e * th**3.5 * np.exp(b2 * (1.0 - th))
    base_width_w = b3 * 1.0e-4 * (p * th**b4 + b5 * e * th**b6)
    width_w = 0.535 * base_width_w + np.sqrt(0.217 * base_width_w**2 + 2.1316e-12 * f0**2 / th)
    line_shape_w = (f / f0) * (width_w / ((f0 - f) ** 2 + width_w**2) + width_w / ((f0 + f) ** 2 + width_w**2))
    refractivity_w = np.sum(strength_w * line_shape_w, axis=1)

    oxygen = (0.1820 * frequency.reshape(-1) * refractivity_o).reshape(shape)
    water = (0.1820 * frequency.reshape(-1) * refractivity_w).reshape(shape)
    return GaseousSpecificAttenuation(
        oxygen_db_per_km=oxygen,
        water_vapour_db_per_km=water,
        total_db_per_km=oxygen + water,
    )


@lru_cache(maxsize=1)
def _p676_oxygen_effective_height_coefficients(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(
            "Set OEL_P676_ANNEX2_PART1 to an authorized local ITU-R P.676-13 Annex 2 Part 1 file."
        )
    values = np.loadtxt(path, dtype=float)
    if values.shape != (700, 5) or not np.all(np.isfinite(values)):
        raise ValueError("P.676-13 Annex 2 Part 1 must contain exactly 700 five-column rows.")
    order = np.argsort(values[:, 0], kind="stable")
    values = values[order]
    if np.any(np.diff(values[:, 0]) <= 0.0):
        raise ValueError("P.676-13 Annex 2 Part 1 frequencies must be unique and increasing.")
    return values


def p676_13_slant_attenuation(
    frequency_ghz: float | np.ndarray,
    *,
    elevation_deg: float | np.ndarray,
    total_pressure_hpa: float | np.ndarray,
    temperature_k: float | np.ndarray,
    water_vapour_density_g_m3: float | np.ndarray,
    numeric_backend: str = "rust",
) -> GaseousSlantAttenuation:
    """Evaluate P.676-13 Annex 2 instantaneous method 1, equations (29)-(37)."""

    frequency, elevation, pressure, temperature, density = np.broadcast_arrays(
        _positive_finite(frequency_ghz, "frequency_ghz"),
        _elevation(elevation_deg),
        _positive_finite(total_pressure_hpa, "total_pressure_hpa"),
        _positive_finite(temperature_k, "temperature_k"),
        np.asarray(water_vapour_density_g_m3, dtype=float),
    )
    if np.any(frequency < 1.0) or np.any(frequency > 350.0):
        raise ValueError("P.676-13 Annex 2 supports frequencies within [1, 350] GHz.")
    if np.any(~np.isfinite(density)) or np.any(density < 0.0):
        raise ValueError("water_vapour_density_g_m3 must contain nonnegative finite values.")
    specific = p676_13_specific_attenuation(
        frequency,
        total_pressure_hpa=pressure,
        temperature_k=temperature,
        water_vapour_density_g_m3=density,
        numeric_backend=numeric_backend,
    )
    bundled = Path(__file__).with_name("data") / "p676-13-annex2-part1.txt"
    coefficient_path = Path(os.environ.get("OEL_P676_ANNEX2_PART1", str(bundled))).expanduser()
    coefficients = _p676_oxygen_effective_height_coefficients(coefficient_path.resolve())
    flattened_frequency = frequency.reshape(-1)
    interpolated = np.column_stack(
        [np.interp(flattened_frequency, coefficients[:, 0], coefficients[:, column]) for column in range(1, 5)]
    ).reshape(frequency.shape + (4,))
    if str(numeric_backend).strip().lower() == "rust":
        from sim.pro_communications.rust_atmosphere import slant_batch

        oxygen, water, oxygen_height, water_height = slant_batch(
            frequency, elevation, pressure, temperature, density, specific, interpolated,
            _OXYGEN_LINES, _WATER_LINES,
        )
        return GaseousSlantAttenuation(
            oxygen_db=oxygen, water_vapour_db=water, total_db=oxygen + water,
            oxygen_effective_height_km=oxygen_height, water_vapour_effective_height_km=water_height,
        )
    oxygen_height = (
        interpolated[..., 0]
        + interpolated[..., 1] * temperature
        + interpolated[..., 2] * pressure
        + interpolated[..., 3] * density
    )
    water_height = 5.6585e-5 * frequency + 1.8348
    for line_frequency, numerator, denominator in (
        (22.235080, 2.6846, 2.7649),
        (183.310087, 5.8905, 4.9219),
        (325.152888, 2.9810, 3.0748),
    ):
        water_height += numerator / ((frequency - line_frequency) ** 2 + denominator)
    sine = np.sin(np.deg2rad(elevation))
    oxygen = specific.oxygen_db_per_km * oxygen_height / sine
    water = specific.water_vapour_db_per_km * water_height / sine
    return GaseousSlantAttenuation(
        oxygen_db=oxygen,
        water_vapour_db=water,
        total_db=oxygen + water,
        oxygen_effective_height_km=oxygen_height,
        water_vapour_effective_height_km=water_height,
    )


_KH = (
    np.asarray(
        [
            (-5.33980, -0.10008, 1.13098),
            (-0.35351, 1.26970, 0.45400),
            (-0.23789, 0.86036, 0.15354),
            (-0.94158, 0.64552, 0.16817),
        ]
    ),
    -0.18961,
    0.71147,
)
_KV = (
    np.asarray(
        [
            (-3.80595, 0.56934, 0.81061),
            (-3.44965, -0.22911, 0.51059),
            (-0.39902, 0.73042, 0.11899),
            (0.50167, 1.07319, 0.27195),
        ]
    ),
    -0.16398,
    0.63297,
)
_AH = (
    np.asarray(
        [
            (-0.14318, 1.82442, -0.55187),
            (0.29591, 0.77564, 0.19822),
            (0.32177, 0.63773, 0.13164),
            (-5.37610, -0.96230, 1.47828),
            (16.1721, -3.29980, 3.43990),
        ]
    ),
    0.67849,
    -1.95537,
)
_AV = (
    np.asarray(
        [
            (-0.07771, 2.33840, -0.76284),
            (0.56727, 0.95545, 0.54039),
            (-0.20238, 1.14520, 0.26809),
            (-48.2991, 0.791669, 0.116226),
            (48.5833, 0.791459, 0.116479),
        ]
    ),
    -0.053739,
    0.83433,
)


def _p838_coefficient(
    frequency_ghz: np.ndarray, constants: tuple[np.ndarray, float, float], *, log: bool
) -> np.ndarray:
    rows, slope, intercept = constants
    log_frequency = np.log10(frequency_ghz)[..., None]
    summed = np.sum(rows[:, 0] * np.exp(-(((log_frequency - rows[:, 1]) / rows[:, 2]) ** 2)), axis=-1)
    result = summed + slope * np.log10(frequency_ghz) + intercept
    return 10.0**result if log else result


def p838_3_rain_coefficients(
    frequency_ghz: float | np.ndarray,
    *,
    elevation_deg: float | np.ndarray,
    polarization_tilt_deg: float | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return geometry-adjusted k and alpha from P.838-3 equations (2)-(5)."""

    frequency, elevation, tilt = np.broadcast_arrays(
        _positive_finite(frequency_ghz, "frequency_ghz"),
        _elevation(elevation_deg, minimum_deg=0.0),
        np.asarray(polarization_tilt_deg, dtype=float),
    )
    if np.any(frequency < 1.0) or np.any(frequency > 1000.0):
        raise ValueError("P.838-3 supports frequencies within [1, 1000] GHz.")
    if np.any(~np.isfinite(tilt)):
        raise ValueError("polarization_tilt_deg must contain finite values.")
    kh = _p838_coefficient(frequency, _KH, log=True)
    kv = _p838_coefficient(frequency, _KV, log=True)
    ah = _p838_coefficient(frequency, _AH, log=False)
    av = _p838_coefficient(frequency, _AV, log=False)
    geometry = np.cos(np.deg2rad(elevation)) ** 2 * np.cos(2.0 * np.deg2rad(tilt))
    k = 0.5 * (kh + kv + (kh - kv) * geometry)
    alpha = 0.5 * (kh * ah + kv * av + (kh * ah - kv * av) * geometry) / k
    return k, alpha


def p838_3_specific_rain_attenuation(
    frequency_ghz: float | np.ndarray,
    *,
    rain_rate_mm_h: float | np.ndarray,
    elevation_deg: float | np.ndarray,
    polarization_tilt_deg: float | np.ndarray,
    numeric_backend: str = "rust",
) -> np.ndarray:
    backend = str(numeric_backend).strip().lower()
    if backend not in {"python", "rust"}:
        raise ValueError(f"Unknown atmospheric numeric backend {numeric_backend!r}.")
    if backend == "rust":
        from sim.rust_coverage_backend import p838_3_specific_rain_attenuation

        return p838_3_specific_rain_attenuation(
            np.asarray(frequency_ghz, dtype=float),
            rain_rate_mm_h=np.asarray(rain_rate_mm_h, dtype=float),
            elevation_deg=np.asarray(elevation_deg, dtype=float),
            polarization_tilt_deg=np.asarray(polarization_tilt_deg, dtype=float),
        )
    rate = np.asarray(rain_rate_mm_h, dtype=float)
    if np.any(~np.isfinite(rate)) or np.any(rate < 0.0):
        raise ValueError("rain_rate_mm_h must contain nonnegative finite values.")
    k, alpha = p838_3_rain_coefficients(
        frequency_ghz,
        elevation_deg=elevation_deg,
        polarization_tilt_deg=polarization_tilt_deg,
    )
    rate = np.broadcast_to(rate, k.shape)
    return k * rate**alpha


def p618_14_rain_attenuation(
    frequency_ghz: float | np.ndarray,
    *,
    elevation_deg: float | np.ndarray,
    station_latitude_deg: float | np.ndarray,
    station_height_km: float | np.ndarray,
    rain_height_km: float | np.ndarray,
    rain_rate_r001_mm_h: float | np.ndarray,
    polarization_tilt_deg: float | np.ndarray = 0.0,
    exceedance_percentage: float | np.ndarray = 0.01,
    numeric_backend: str = "rust",
) -> np.ndarray:
    """Evaluate P.618-14 section 2.2.1.1 for declared R0.01 inputs."""

    frequency, elevation, latitude, station_height, rain_height, rain_rate, tilt, percentage = np.broadcast_arrays(
        _positive_finite(frequency_ghz, "frequency_ghz"),
        _elevation(elevation_deg),
        np.asarray(station_latitude_deg, dtype=float),
        np.asarray(station_height_km, dtype=float),
        np.asarray(rain_height_km, dtype=float),
        np.asarray(rain_rate_r001_mm_h, dtype=float),
        np.asarray(polarization_tilt_deg, dtype=float),
        np.asarray(exceedance_percentage, dtype=float),
    )
    if np.any(frequency > 55.0):
        raise ValueError("P.618-14 rain attenuation method supports frequencies at most 55 GHz.")
    for values, name in (
        (latitude, "station_latitude_deg"),
        (station_height, "station_height_km"),
        (rain_height, "rain_height_km"),
        (tilt, "polarization_tilt_deg"),
    ):
        if np.any(~np.isfinite(values)):
            raise ValueError(f"{name} must contain finite values.")
    if np.any(np.abs(latitude) > 90.0):
        raise ValueError("station_latitude_deg must be within [-90, 90].")
    if np.any(~np.isfinite(rain_rate)) or np.any(rain_rate < 0.0):
        raise ValueError("rain_rate_r001_mm_h must contain nonnegative finite values.")
    if np.any(~np.isfinite(percentage)) or np.any((percentage < 0.001) | (percentage > 5.0)):
        raise ValueError("exceedance_percentage must be within [0.001, 5].")
    result = np.zeros(frequency.shape, dtype=float)
    active = (rain_height > station_height) & (rain_rate > 0.0)
    if not np.any(active):
        return result
    theta = np.deg2rad(elevation)
    delta_height = rain_height - station_height
    earth_radius = 8500.0
    high = elevation >= 5.0
    slant = np.where(
        high,
        delta_height / np.sin(theta),
        2.0 * delta_height / (np.sqrt(np.sin(theta) ** 2 + 2.0 * delta_height / earth_radius) + np.sin(theta)),
    )
    horizontal = slant * np.cos(theta)
    gamma = p838_3_specific_rain_attenuation(
        frequency,
        rain_rate_mm_h=rain_rate,
        elevation_deg=elevation,
        polarization_tilt_deg=tilt,
        numeric_backend=numeric_backend,
    )
    horizontal_reduction = 1.0 / (
        1.0 + 0.78 * np.sqrt(horizontal * gamma / frequency) - 0.38 * (1.0 - np.exp(-2.0 * horizontal))
    )
    zeta = np.rad2deg(np.arctan2(delta_height, horizontal * horizontal_reduction))
    rain_path = np.where(
        zeta > elevation,
        horizontal * horizontal_reduction / np.cos(theta),
        delta_height / np.sin(theta),
    )
    chi = np.maximum(36.0 - np.abs(latitude), 0.0)
    vertical_adjustment = 1.0 / (
        1.0
        + np.sqrt(np.sin(theta))
        * (31.0 * (1.0 - np.exp(-elevation / (1.0 + chi))) * np.sqrt(rain_path * gamma) / frequency**2 - 0.45)
    )
    attenuation_001 = gamma * rain_path * vertical_adjustment
    beta = np.zeros(frequency.shape, dtype=float)
    low_percentage = percentage < 1.0
    low_latitude = np.abs(latitude) < 36.0
    beta = np.where(low_percentage & low_latitude & (elevation >= 25.0), -0.005 * (np.abs(latitude) - 36.0), beta)
    beta = np.where(
        low_percentage & low_latitude & (elevation < 25.0),
        -0.005 * (np.abs(latitude) - 36.0) + 1.8 - 4.25 * np.sin(theta),
        beta,
    )
    exponent = -(
        0.655
        + 0.033 * np.log(percentage)
        - 0.045 * np.log(np.maximum(attenuation_001, 1.0e-300))
        - beta * (1.0 - percentage) * np.sin(theta)
    )
    scaled = attenuation_001 * (percentage / 0.01) ** exponent
    result[active] = scaled[active]
    return result


def p840_9_cloud_liquid_mass_absorption_coefficient(frequency_ghz: float | np.ndarray) -> np.ndarray:
    frequency = _positive_finite(frequency_ghz, "frequency_ghz")
    if np.any(frequency < 1.0) or np.any(frequency > 200.0):
        raise ValueError("P.840-9 cloud attenuation supports frequencies within [1, 200] GHz.")
    temperature = 273.75
    theta = 300.0 / temperature - 1.0
    epsilon_0 = 77.66 + 103.3 * theta
    epsilon_1 = 0.0671 * epsilon_0
    epsilon_2 = 3.52
    f_p = 20.20 - 146.0 * theta + 316.0 * theta**2
    f_s = 39.8 * f_p
    epsilon_imag = frequency * (epsilon_0 - epsilon_1) / (f_p * (1.0 + (frequency / f_p) ** 2)) + frequency * (
        epsilon_1 - epsilon_2
    ) / (f_s * (1.0 + (frequency / f_s) ** 2))
    epsilon_real = (
        (epsilon_0 - epsilon_1) / (1.0 + (frequency / f_p) ** 2)
        + (epsilon_1 - epsilon_2) / (1.0 + (frequency / f_s) ** 2)
        + epsilon_2
    )
    eta = (2.0 + epsilon_real) / epsilon_imag
    base = 0.819 * frequency / (epsilon_imag * (1.0 + eta**2))
    correction = (
        0.1522 * np.exp(-((frequency + 23.9589) ** 2) / 3.2991e3)
        + 11.51 * np.exp(-((frequency - 219.2096) ** 2) / 2.7595e6)
        - 10.4912
    )
    return base * correction


def p840_9_cloud_attenuation(
    frequency_ghz: float | np.ndarray,
    *,
    elevation_deg: float | np.ndarray,
    integrated_cloud_liquid_water_kg_m2: float | np.ndarray,
    numeric_backend: str = "rust",
) -> np.ndarray:
    backend = str(numeric_backend).strip().lower()
    if backend not in {"python", "rust"}:
        raise ValueError(f"Unknown atmospheric numeric backend {numeric_backend!r}.")
    if backend == "rust":
        from sim.rust_coverage_backend import p840_9_cloud_attenuation

        return p840_9_cloud_attenuation(
            np.asarray(frequency_ghz, dtype=float),
            elevation_deg=np.asarray(elevation_deg, dtype=float),
            integrated_cloud_liquid_water_kg_m2=np.asarray(
                integrated_cloud_liquid_water_kg_m2,
                dtype=float,
            ),
        )
    coefficient, elevation, liquid = np.broadcast_arrays(
        p840_9_cloud_liquid_mass_absorption_coefficient(frequency_ghz),
        _elevation(elevation_deg),
        np.asarray(integrated_cloud_liquid_water_kg_m2, dtype=float),
    )
    if np.any(~np.isfinite(liquid)) or np.any(liquid < 0.0):
        raise ValueError("integrated_cloud_liquid_water_kg_m2 must contain nonnegative finite values.")
    return coefficient * liquid / np.sin(np.deg2rad(elevation))


def p618_14_scintillation_fade_depth(
    frequency_ghz: float,
    *,
    elevation_deg: float | np.ndarray,
    exceedance_percentage: float | np.ndarray,
    median_wet_refractivity_n_units: float,
    antenna_diameter_m: float,
    antenna_efficiency: float,
) -> np.ndarray:
    """P.618-14 section 2.4.1 long-term amplitude scintillation above 5 degrees."""

    frequency = float(frequency_ghz)
    wet = float(median_wet_refractivity_n_units)
    diameter = float(antenna_diameter_m)
    efficiency = float(antenna_efficiency)
    if not np.isfinite(frequency) or not 4.0 <= frequency <= 55.0:
        raise ValueError("frequency_ghz must lie within the promoted [4, 55] GHz envelope.")
    if not np.isfinite(wet) or wet < 0.0:
        raise ValueError("median_wet_refractivity_n_units must be nonnegative and finite.")
    if not np.isfinite(diameter) or diameter <= 0.0:
        raise ValueError("antenna_diameter_m must be positive and finite.")
    if not np.isfinite(efficiency) or not 0.0 < efficiency <= 1.0:
        raise ValueError("antenna_efficiency must lie within (0, 1].")
    elevation, percentage = np.broadcast_arrays(
        _elevation(elevation_deg),
        np.asarray(exceedance_percentage, dtype=float),
    )
    if np.any(elevation <= 5.0):
        raise ValueError("The promoted P.618-14 scintillation path requires elevation above 5 degrees.")
    if np.any(~np.isfinite(percentage)) or np.any((percentage < 0.001) | (percentage > 50.0)):
        raise ValueError("exceedance_percentage must lie within [0.001, 50].")
    sine = np.sin(np.deg2rad(elevation))
    reference_sigma = 3.6e-3 + 1.0e-4 * wet
    effective_path_m = 2000.0 / (np.sqrt(sine**2 + 2.35e-4) + sine)
    x = 1.22 * diameter**2 * efficiency * frequency / effective_path_m
    antenna_averaging = np.sqrt(
        3.86 * (x**2 + 1.0) ** (11.0 / 12.0) * np.sin((11.0 / 6.0) * np.arctan(1.0 / x)) - 7.08 * x ** (5.0 / 6.0)
    )
    signal_sigma = reference_sigma * frequency ** (7.0 / 12.0) * antenna_averaging / sine**1.2
    log_probability = np.log10(percentage)
    time_percentage_factor = -0.061 * log_probability**3 + 0.072 * log_probability**2 - 1.71 * log_probability + 3.0
    return time_percentage_factor * signal_sigma


def p618_14_total_attenuation(
    *,
    gaseous_attenuation_db: float | np.ndarray,
    cloud_attenuation_db: float | np.ndarray,
    rain_attenuation_db: float | np.ndarray,
    scintillation_fade_depth_db: float | np.ndarray,
    numeric_backend: str = "rust",
) -> np.ndarray:
    """P.618-14 section 2.5 aggregate attenuation for simultaneous sources."""

    backend = str(numeric_backend).strip().lower()
    if backend not in {"python", "rust"}:
        raise ValueError(f"Unknown atmospheric numeric backend {numeric_backend!r}.")
    if backend == "rust":
        from sim.rust_coverage_backend import p618_14_total_attenuation

        return p618_14_total_attenuation(
            gaseous_attenuation_db=np.asarray(gaseous_attenuation_db, dtype=float),
            cloud_attenuation_db=np.asarray(cloud_attenuation_db, dtype=float),
            rain_attenuation_db=np.asarray(rain_attenuation_db, dtype=float),
            scintillation_fade_depth_db=np.asarray(scintillation_fade_depth_db, dtype=float),
        )

    gas, cloud, rain, scintillation = np.broadcast_arrays(
        np.asarray(gaseous_attenuation_db, dtype=float),
        np.asarray(cloud_attenuation_db, dtype=float),
        np.asarray(rain_attenuation_db, dtype=float),
        np.asarray(scintillation_fade_depth_db, dtype=float),
    )
    if any(np.any(~np.isfinite(value)) or np.any(value < 0.0) for value in (gas, cloud, rain, scintillation)):
        raise ValueError("Aggregate attenuation components must be nonnegative and finite.")
    return gas + np.sqrt((rain + cloud) ** 2 + scintillation**2)


def evaluate_atmospheric_losses(
    profile: AtmosphericProfile,
    *,
    carrier_frequency_hz: float,
    elevation_rad: float | np.ndarray,
    numeric_backend: str = "rust",
) -> AtmosphericLossLedger:
    if not isinstance(profile, AtmosphericProfile):
        raise ValueError("profile must be an AtmosphericProfile.")
    backend = str(numeric_backend).strip().lower()
    if backend not in {"python", "rust"}:
        raise ValueError(f"Unknown atmospheric numeric backend {numeric_backend!r}.")
    frequency_ghz = float(carrier_frequency_hz) / 1.0e9
    if not np.isfinite(frequency_ghz) or frequency_ghz <= 0.0:
        raise ValueError("carrier_frequency_hz must be positive and finite.")
    elevation_deg = np.rad2deg(np.asarray(elevation_rad, dtype=float))
    if np.any(~np.isfinite(elevation_deg)):
        raise ValueError("elevation_rad must contain finite values.")
    shape = elevation_deg.shape
    gaseous = np.zeros(shape, dtype=float)
    rain = np.zeros(shape, dtype=float)
    cloud = np.zeros(shape, dtype=float)
    editions: list[str] = []
    statistical: list[str] = []
    if profile.gaseous_model != "none":
        gaseous = p676_13_slant_attenuation(
            frequency_ghz,
            elevation_deg=elevation_deg,
            total_pressure_hpa=float(profile.total_pressure_hpa),
            temperature_k=float(profile.temperature_k),
            water_vapour_density_g_m3=float(profile.water_vapour_density_g_m3),
            numeric_backend=backend,
        ).total_db
        editions.append(P676_EDITION)
    if profile.rain_model != "none":
        rain = p618_14_rain_attenuation(
            frequency_ghz,
            elevation_deg=elevation_deg,
            station_latitude_deg=float(profile.station_latitude_deg),
            station_height_km=profile.station_height_km,
            rain_height_km=float(profile.rain_height_km),
            rain_rate_r001_mm_h=float(profile.rainfall_rate_mm_h),
            polarization_tilt_deg=profile.polarization_tilt_deg,
            exceedance_percentage=profile.rain_exceedance_percentage,
            numeric_backend=backend,
        )
        editions.extend((P618_EDITION, P838_EDITION))
        statistical.append("rain_attenuation_exceeded_percentage")
    if profile.cloud_model != "none":
        cloud = p840_9_cloud_attenuation(
            frequency_ghz,
            elevation_deg=elevation_deg,
            integrated_cloud_liquid_water_kg_m2=float(profile.integrated_cloud_liquid_water_kg_m2),
            numeric_backend=backend,
        )
        editions.append(P840_EDITION)
    return AtmosphericLossLedger(
        gaseous_db=gaseous,
        rain_db=rain,
        cloud_db=cloud,
        total_db=gaseous + rain + cloud,
        statistical_components=tuple(statistical),
        model_editions=tuple(editions),
    )


__all__ = [
    "AtmosphericLossLedger",
    "GaseousSlantAttenuation",
    "GaseousSpecificAttenuation",
    "evaluate_atmospheric_losses",
    "p618_14_rain_attenuation",
    "p618_14_scintillation_fade_depth",
    "p618_14_total_attenuation",
    "p676_13_slant_attenuation",
    "p676_13_specific_attenuation",
    "p838_3_rain_coefficients",
    "p838_3_specific_rain_attenuation",
    "p840_9_cloud_attenuation",
    "p840_9_cloud_liquid_mass_absorption_coefficient",
]
