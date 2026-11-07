from __future__ import annotations

import calendar
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np

from sim.dynamics.orbit.environment import EARTH_MU_KM3_S2
from sim.dynamics.orbit.epoch import datetime_to_julian_date

TLE_INITIALIZATION_MODEL_OGP = "ogp"
TLE_INITIALIZATION_MODEL_KEPLERIAN = "keplerian_mean_elements"

_OGP_INITIALIZATION_ALIASES = {
    "ogp",
    "general",
    "general_perturbations",
    "gp",
    "sgp4",
    "sdp4",
    "ogp_sgp4",
    "ogp_sdp4",
}
_KEPLERIAN_INITIALIZATION_ALIASES = {
    "keplerian",
    "keplerian_mean_elements",
    "mean_elements",
    "two_body",
    "legacy",
}


@dataclass(frozen=True)
class TLEElements:
    line1: str
    line2: str
    norad_number: str
    classification: str
    international_designator: str
    epoch_text: str
    epoch_jd_utc: float
    mean_motion_derivative_rev_per_day2: float
    mean_motion_second_derivative_rev_per_day3: float
    bstar: float
    ephemeris_type: str
    element_number: int
    inclination_deg: float
    raan_deg: float
    eccentricity: float
    argp_deg: float
    mean_anomaly_deg: float
    mean_motion_rev_per_day: float
    revolution_number: int


_OGP_MEAN_ELEMENT_REQUIRED_FIELDS = (
    "epoch_jd_utc",
    "inclination_deg",
    "raan_deg",
    "eccentricity",
    "argp_deg",
    "mean_anomaly_deg",
    "mean_motion_rev_per_day",
)


def ogp_mean_elements_from_mapping(value: dict[str, Any]) -> TLEElements:
    """Build an OGP element set without manufacturing TLE text.

    This is the native interchange path for fitted mean-element products.  The
    empty ``line1``/``line2`` values are intentional: callers must not imply
    that fitted floating-point elements are an encoded or checksum-valid TLE.
    """

    raw = dict(value or {})
    missing = [name for name in _OGP_MEAN_ELEMENT_REQUIRED_FIELDS if name not in raw]
    if missing:
        raise ValueError(f"OGP mean-element state is missing required field(s): {', '.join(missing)}.")

    def finite(name: str, default: float = 0.0) -> float:
        number = float(raw.get(name, default))
        if not math.isfinite(number):
            raise ValueError(f"OGP mean-element field {name} must be finite.")
        return number

    epoch = finite("epoch_jd_utc")
    eccentricity = finite("eccentricity")
    mean_motion = finite("mean_motion_rev_per_day")
    if epoch <= 0.0:
        raise ValueError("OGP mean-element epoch_jd_utc must be positive.")
    if not 0.0 <= eccentricity < 1.0:
        raise ValueError("OGP mean-element eccentricity must be in [0, 1).")
    if mean_motion <= 0.0:
        raise ValueError("OGP mean-element mean_motion_rev_per_day must be positive.")
    elements = TLEElements(
        line1="",
        line2="",
        norad_number=str(raw.get("norad_number", "") or ""),
        classification=str(raw.get("classification", "") or ""),
        international_designator=str(raw.get("international_designator", "") or ""),
        epoch_text=str(raw.get("epoch_text", "") or ""),
        epoch_jd_utc=epoch,
        mean_motion_derivative_rev_per_day2=finite("mean_motion_derivative_rev_per_day2"),
        mean_motion_second_derivative_rev_per_day3=finite("mean_motion_second_derivative_rev_per_day3"),
        bstar=finite("bstar"),
        ephemeris_type=str(raw.get("ephemeris_type", "") or ""),
        element_number=int(raw.get("element_number", 0) or 0),
        inclination_deg=finite("inclination_deg"),
        raan_deg=finite("raan_deg"),
        eccentricity=eccentricity,
        argp_deg=finite("argp_deg"),
        mean_anomaly_deg=finite("mean_anomaly_deg"),
        mean_motion_rev_per_day=mean_motion,
        revolution_number=int(raw.get("revolution_number", 0) or 0),
    )
    if not 0.0 <= elements.inclination_deg <= 180.0:
        raise ValueError("OGP mean-element inclination_deg must be in [0, 180].")
    return elements


def ogp_mean_elements_to_mapping(elements: TLEElements) -> dict[str, Any]:
    """Return the native mean-element fields, deliberately excluding TLE lines."""

    return {
        name: getattr(elements, name)
        for name in TLEElements.__dataclass_fields__
        if name not in {"line1", "line2"}
    }


def tle_epoch_to_julian_date(epoch_text: str) -> float:
    text = str(epoch_text or "").strip()
    if len(text) < 5:
        raise ValueError("TLE epoch must use YYDDD.DDDDDDDD format.")
    year_two = int(text[:2])
    year = 2000 + year_two if year_two < 57 else 1900 + year_two
    day_of_year = float(text[2:])
    max_day_of_year = 366 if calendar.isleap(year) else 365
    if not math.isfinite(day_of_year) or not (1.0 <= day_of_year < max_day_of_year + 1.0):
        raise ValueError(
            f"TLE epoch day-of-year must be in [1, {max_day_of_year + 1}) for year {year}."
        )
    day_index = int(math.floor(day_of_year))
    frac_day = day_of_year - day_index
    dt = datetime(year, 1, 1, tzinfo=timezone.utc) + timedelta(days=day_index - 1, seconds=frac_day * 86400.0)
    return datetime_to_julian_date(dt)


def tle_checksum_digit(line: str) -> int:
    """Return the standard TLE checksum digit for columns 1 through 68."""

    total = 0
    for ch in line[:68]:
        if ch.isdigit():
            total += int(ch)
        elif ch == "-":
            total += 1
    return total % 10


def with_tle_checksum(line: str) -> str:
    """Return a 69-column TLE line with a checksum matching its first 68 columns."""

    text = str(line).rstrip("\r\n")
    if len(text) < 68:
        raise ValueError("TLE line must contain at least 68 columns before checksum generation.")
    body = text[:68]
    return f"{body}{tle_checksum_digit(body)}"


def _checksum_ok(line: str) -> bool:
    if len(line) < 69 or not line[68].isdigit():
        return False
    return tle_checksum_digit(line) == int(line[68])


def _parse_tle_float(text: str, *, default: float = 0.0) -> float:
    raw = str(text or "").strip()
    if not raw:
        return float(default)
    return float(raw)


def _parse_tle_int(text: str, *, default: int = 0) -> int:
    raw = str(text or "").strip()
    if not raw:
        return int(default)
    return int(raw)


def _parse_tle_compact_exponential(text: str) -> float:
    raw = str(text or "").strip()
    if not raw:
        return 0.0
    raw = raw.replace(" ", "")
    if "e" in raw.lower():
        return float(raw)
    if len(raw) < 3:
        return float(raw)
    mantissa_text = raw[:-2]
    exponent_text = raw[-2:]
    if mantissa_text in {"", "+", "-"}:
        return 0.0
    sign = -1.0 if mantissa_text.startswith("-") else 1.0
    mantissa_digits = mantissa_text.lstrip("+-")
    if "." in mantissa_digits:
        mantissa = float(mantissa_text)
    else:
        mantissa = sign * float(f"0.{mantissa_digits}")
    return float(mantissa * (10.0 ** int(exponent_text)))


def parse_tle_lines(line1: str, line2: str, *, require_checksum: bool = False) -> TLEElements:
    l1 = str(line1 or "").rstrip("\r\n")
    l2 = str(line2 or "").rstrip("\r\n")
    if not l1.startswith("1 ") or not l2.startswith("2 "):
        raise ValueError("TLE line1 must start with '1 ' and line2 must start with '2 '.")
    if len(l1) != 69 or len(l2) != 69:
        raise ValueError("TLE lines must contain exactly 69 standard columns.")
    if require_checksum and (not _checksum_ok(l1) or not _checksum_ok(l2)):
        raise ValueError("TLE checksum validation failed.")
    if l1[2:7] != l2[2:7]:
        raise ValueError("TLE line1 and line2 NORAD catalog numbers must match.")

    ecc_text = l2[26:33].strip()
    if not ecc_text or not ecc_text.isdigit():
        raise ValueError("TLE eccentricity field is invalid.")

    elements = TLEElements(
        line1=l1,
        line2=l2,
        norad_number=l1[2:7].strip(),
        classification=l1[7:8].strip(),
        international_designator=l1[9:17].strip(),
        epoch_text=l1[18:32].strip(),
        epoch_jd_utc=tle_epoch_to_julian_date(l1[18:32]),
        mean_motion_derivative_rev_per_day2=_parse_tle_float(l1[33:43]),
        mean_motion_second_derivative_rev_per_day3=_parse_tle_compact_exponential(l1[44:52]),
        bstar=_parse_tle_compact_exponential(l1[53:61]),
        ephemeris_type=l1[62:63].strip(),
        element_number=_parse_tle_int(l1[64:68]),
        inclination_deg=float(l2[8:16]),
        raan_deg=float(l2[17:25]),
        eccentricity=float(f"0.{ecc_text}"),
        argp_deg=float(l2[34:42]),
        mean_anomaly_deg=float(l2[43:51]),
        mean_motion_rev_per_day=float(l2[52:63]),
        revolution_number=_parse_tle_int(l2[63:68]),
    )
    numeric = (
        elements.epoch_jd_utc,
        elements.mean_motion_derivative_rev_per_day2,
        elements.mean_motion_second_derivative_rev_per_day3,
        elements.bstar,
        elements.inclination_deg,
        elements.raan_deg,
        elements.eccentricity,
        elements.argp_deg,
        elements.mean_anomaly_deg,
        elements.mean_motion_rev_per_day,
    )
    if not all(math.isfinite(float(value)) for value in numeric):
        raise ValueError("TLE numeric fields must be finite.")
    if not 0.0 <= elements.inclination_deg <= 180.0:
        raise ValueError("TLE inclination must be in [0, 180] degrees.")
    if not 0.0 <= elements.eccentricity < 1.0:
        raise ValueError("TLE eccentricity must be in [0, 1).")
    if elements.mean_motion_rev_per_day <= 0.0:
        raise ValueError("TLE mean motion must be positive.")
    return elements


def _solve_eccentric_anomaly(mean_anomaly_rad: float, eccentricity: float) -> float:
    mean_anomaly_rad = float(np.mod(mean_anomaly_rad, 2.0 * np.pi))
    e = float(eccentricity)
    ecc_anomaly = mean_anomaly_rad if e < 0.8 else np.pi
    for _ in range(30):
        f = ecc_anomaly - e * np.sin(ecc_anomaly) - mean_anomaly_rad
        fp = 1.0 - e * np.cos(ecc_anomaly)
        step = f / fp
        ecc_anomaly -= step
        if abs(step) < 1e-13:
            break
    return float(ecc_anomaly)


def _coe_to_rv_eci(
    *,
    a_km: float,
    ecc: float,
    inc_deg: float,
    raan_deg: float,
    argp_deg: float,
    true_anomaly_deg: float,
    mu_km3_s2: float = EARTH_MU_KM3_S2,
) -> tuple[np.ndarray, np.ndarray]:
    a = float(a_km)
    e = float(ecc)
    inc = np.deg2rad(float(inc_deg))
    raan = np.deg2rad(float(raan_deg))
    argp = np.deg2rad(float(argp_deg))
    nu = np.deg2rad(float(true_anomaly_deg))
    p = a * (1.0 - e * e)
    if a <= 0.0 or p <= 0.0:
        raise ValueError("TLE-derived orbit is invalid.")

    cnu, snu = np.cos(nu), np.sin(nu)
    r_pf = np.array([p * cnu / (1.0 + e * cnu), p * snu / (1.0 + e * cnu), 0.0], dtype=float)
    v_pf = np.sqrt(mu_km3_s2 / p) * np.array([-snu, e + cnu, 0.0], dtype=float)

    cO, sO = np.cos(raan), np.sin(raan)
    ci, si = np.cos(inc), np.sin(inc)
    cw, sw = np.cos(argp), np.sin(argp)
    q_pf_to_eci = np.array(
        [
            [cO * cw - sO * sw * ci, -cO * sw - sO * cw * ci, sO * si],
            [sO * cw + cO * sw * ci, -sO * sw + cO * cw * ci, -cO * si],
            [sw * si, cw * si, ci],
        ],
        dtype=float,
    )
    return q_pf_to_eci @ r_pf, q_pf_to_eci @ v_pf


def tle_to_rv_eci(
    elements: TLEElements,
    *,
    target_jd_utc: float | None = None,
    mu_km3_s2: float = EARTH_MU_KM3_S2,
) -> tuple[np.ndarray, np.ndarray]:
    mean_motion_rad_s = float(elements.mean_motion_rev_per_day) * 2.0 * np.pi / 86400.0
    if mean_motion_rad_s <= 0.0:
        raise ValueError("TLE mean motion must be positive.")
    a_km = (float(mu_km3_s2) / (mean_motion_rad_s**2)) ** (1.0 / 3.0)
    dt_s = 0.0 if target_jd_utc is None else (float(target_jd_utc) - float(elements.epoch_jd_utc)) * 86400.0
    mean_anomaly_rad = np.deg2rad(float(elements.mean_anomaly_deg)) + mean_motion_rad_s * dt_s
    ecc_anomaly = _solve_eccentric_anomaly(mean_anomaly_rad, float(elements.eccentricity))
    e = float(elements.eccentricity)
    true_anomaly_rad = 2.0 * np.arctan2(
        np.sqrt(1.0 + e) * np.sin(ecc_anomaly / 2.0),
        np.sqrt(1.0 - e) * np.cos(ecc_anomaly / 2.0),
    )
    return _coe_to_rv_eci(
        a_km=a_km,
        ecc=e,
        inc_deg=float(elements.inclination_deg),
        raan_deg=float(elements.raan_deg),
        argp_deg=float(elements.argp_deg),
        true_anomaly_deg=float(np.rad2deg(true_anomaly_rad)),
        mu_km3_s2=mu_km3_s2,
    )


def _normalise_initialization_model(raw: Any) -> str:
    key = str(raw or TLE_INITIALIZATION_MODEL_OGP).strip().lower().replace("-", "_")
    if key in _OGP_INITIALIZATION_ALIASES:
        return TLE_INITIALIZATION_MODEL_OGP
    if key in _KEPLERIAN_INITIALIZATION_ALIASES:
        return TLE_INITIALIZATION_MODEL_KEPLERIAN
    aliases = sorted(_OGP_INITIALIZATION_ALIASES | _KEPLERIAN_INITIALIZATION_ALIASES)
    raise ValueError(f"Unsupported TLE initialization_model {raw!r}; expected one of: {', '.join(aliases)}.")


def _tle_lines_from_block(tle_block: dict[str, Any]) -> tuple[str, str]:
    block = dict(tle_block or {})
    lines = block.get("lines")
    if isinstance(lines, (list, tuple)) and len(lines) >= 2:
        return str(lines[0]), str(lines[1])
    return str(block.get("line1", "") or ""), str(block.get("line2", "") or "")


def tle_to_rv_eci_ogp(
    elements: TLEElements,
    *,
    target_jd_utc: float | None = None,
    frame_transform: str = "teme_to_eci_iau80",
    numeric_backend: str = "rust",
) -> tuple[np.ndarray, np.ndarray]:
    transform = _normalize_ogp_initialization_frame_transform(frame_transform)
    jd_utc = float(elements.epoch_jd_utc if target_jd_utc is None else target_jd_utc)
    if not math.isfinite(jd_utc):
        raise ValueError("TLE OGP target_jd_utc must be finite.")
    tsince_min = (jd_utc - float(elements.epoch_jd_utc)) * 1440.0
    from sim.dynamics.orbit.ogp import ogp_propagate_teme
    from sim.dynamics.orbit.sgp4 import transform_teme_to_output_frame

    state = ogp_propagate_teme(elements, tsince_min, backend=numeric_backend)
    if state.error:
        raise ValueError(state.error)
    return transform_teme_to_output_frame(
        state.position_teme_km,
        state.velocity_teme_km_s,
        jd_utc=jd_utc,
        output_frame="eci",
        frame_transform=transform,
    )


def tle_block_to_rv_eci(
    tle_block: dict[str, Any], *, target_jd_utc: float | None = None, numeric_backend: str = "rust"
) -> tuple[np.ndarray, np.ndarray]:
    block = dict(tle_block or {})
    line1, line2 = _tle_lines_from_block(block)
    elements = parse_tle_lines(line1, line2, require_checksum=bool(block.get("require_checksum", True)))
    propagate = bool(block.get("propagate_to_initial_epoch", True))
    effective_jd = target_jd_utc if propagate else None
    model = _normalise_initialization_model(block.get("initialization_model"))
    if model == TLE_INITIALIZATION_MODEL_KEPLERIAN:
        return tle_to_rv_eci(elements, target_jd_utc=effective_jd)
    return tle_to_rv_eci_ogp(
        elements,
        target_jd_utc=effective_jd,
        frame_transform=str(
            block.get("initialization_frame_transform", "teme_to_eci_iau80") or "teme_to_eci_iau80"
        ),
        numeric_backend=numeric_backend,
    )


def tle_block_initialization_metadata(
    tle_block: dict[str, Any],
    *,
    target_jd_utc: float | None = None,
    duration_s: float = 0.0,
) -> dict[str, Any]:
    block = dict(tle_block or {})
    line1, line2 = _tle_lines_from_block(block)
    elements = parse_tle_lines(line1, line2, require_checksum=bool(block.get("require_checksum", True)))
    propagate = bool(block.get("propagate_to_initial_epoch", True))
    effective_jd = float(target_jd_utc if propagate and target_jd_utc is not None else elements.epoch_jd_utc)
    if not math.isfinite(effective_jd):
        raise ValueError("TLE initialization target_jd_utc must be finite.")
    model = _normalise_initialization_model(block.get("initialization_model"))
    if model == TLE_INITIALIZATION_MODEL_OGP:
        from sim.dynamics.orbit.ogp import ogp_propagator_name_for_elements

        native_frame = "teme"
        frame_transform = _normalize_ogp_initialization_frame_transform(
            block.get("initialization_frame_transform", "teme_to_eci_iau80")
        )
        propagator_family = "OGP"
        propagator_name = ogp_propagator_name_for_elements(elements)
    else:
        native_frame = "eci"
        frame_transform = "keplerian_mean_elements"
        propagator_family = "two_body"
        propagator_name = "Keplerian mean-elements conversion"
    age_days = float(effective_jd - float(elements.epoch_jd_utc))
    return {
        "source": "tle",
        "initialization_model": model,
        "initialization_propagator_family": propagator_family,
        "initialization_propagator_name": propagator_name,
        "handoff_propagation_method": "special",
        "native_frame": native_frame,
        "output_frame": "eci",
        "frame_transform": frame_transform,
        "tle_epoch_jd_utc": float(elements.epoch_jd_utc),
        "initial_jd_utc": float(effective_jd),
        "tle_age_initialization_days": age_days,
        "propagate_to_initial_epoch": bool(propagate),
        "simulation_duration_s": float(duration_s),
        "note": "TLE recovered the initial state only; subsequent trajectory uses configured ONP dynamics.",
    }


def _normalize_ogp_initialization_frame_transform(value: Any) -> str:
    transform = str(value or "teme_to_eci_iau80").strip().lower()
    if transform not in {"teme_as_eci", "teme_to_eci_iau80"}:
        raise ValueError(
            "TLE OGP initialization supports initialization_frame_transform values "
            "'teme_as_eci' and 'teme_to_eci_iau80'."
        )
    return transform
