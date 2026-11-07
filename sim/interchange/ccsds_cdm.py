"""Bounded CCSDS CDM 1.0 KVN interchange for public analysis workflows."""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from sim.data_io.common import read_bytes as read_bounded_bytes
from sim.data_io.contracts import DataIOError
from sim.frame_time import FrameTimeError, TimeScale, parse_epoch

CCSDS_CDM_PROFILE = "oel.ccsds-cdm-kvn.v0.1"
MAX_CDM_BYTES = 8 * 1024 * 1024
MAX_CDM_LINES = 100_000
_UNIT_VALUE = re.compile(r"^(?P<value>.*?)\s*\[(?P<unit>[^\]]+)\]\s*$")
_NUMBER = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?$")
_HEADER_KEYS = ("CCSDS_CDM_VERS", "CREATION_DATE", "ORIGINATOR", "MESSAGE_FOR", "MESSAGE_ID")
_RELATIVE_KEYS = (
    "TCA",
    "MISS_DISTANCE",
    "RELATIVE_SPEED",
    "RELATIVE_POSITION_R",
    "RELATIVE_POSITION_T",
    "RELATIVE_POSITION_N",
    "RELATIVE_VELOCITY_R",
    "RELATIVE_VELOCITY_T",
    "RELATIVE_VELOCITY_N",
    "COLLISION_PROBABILITY",
    "COLLISION_PROBABILITY_METHOD",
    "SCREEN_START",
    "SCREEN_STOP",
    "SCREEN_VOLUME_FRAME",
    "SCREEN_VOLUME_SHAPE",
    "SCREEN_VOLUME_X",
    "SCREEN_VOLUME_Y",
    "SCREEN_VOLUME_Z",
)
_OBJECT_METADATA_KEYS = (
    "OBJECT",
    "OBJECT_DESIGNATOR",
    "CATALOG_NAME",
    "OBJECT_NAME",
    "INTERNATIONAL_DESIGNATOR",
    "OBJECT_TYPE",
    "OPERATOR_CONTACT_POSITION",
    "OPERATOR_ORGANIZATION",
    "OPERATOR_PHONE",
    "OPERATOR_EMAIL",
    "EPHEMERIS_NAME",
    "COVARIANCE_METHOD",
    "MANEUVERABLE",
    "ORBIT_CENTER",
    "REF_FRAME",
    "GRAVITY_MODEL",
    "ATMOSPHERIC_MODEL",
    "N_BODY_PERTURBATIONS",
    "SOLAR_RAD_PRESSURE",
    "EARTH_TIDES",
    "INTRACK_THRUST",
)
_OD_KEYS = (
    "TIME_LASTOB_START",
    "TIME_LASTOB_END",
    "RECOMMENDED_OD_SPAN",
    "ACTUAL_OD_SPAN",
    "OBS_AVAILABLE",
    "OBS_USED",
    "TRACKS_AVAILABLE",
    "TRACKS_USED",
    "RESIDUALS_ACCEPTED",
    "WEIGHTED_RMS",
)
_ADDITIONAL_KEYS = (
    "AREA_PC",
    "AREA_DRG",
    "AREA_SRP",
    "MASS",
    "CD_AREA_OVER_MASS",
    "CR_AREA_OVER_MASS",
    "THRUST_ACCELERATION",
    "SEDR",
)
_STATE_KEYS = ("X", "Y", "Z", "X_DOT", "Y_DOT", "Z_DOT")
_COVARIANCE_KEYS = (
    "CR_R",
    "CT_R",
    "CT_T",
    "CN_R",
    "CN_T",
    "CN_N",
    "CRDOT_R",
    "CRDOT_T",
    "CRDOT_N",
    "CRDOT_RDOT",
    "CTDOT_R",
    "CTDOT_T",
    "CTDOT_N",
    "CTDOT_RDOT",
    "CTDOT_TDOT",
    "CNDOT_R",
    "CNDOT_T",
    "CNDOT_N",
    "CNDOT_RDOT",
    "CNDOT_TDOT",
    "CNDOT_NDOT",
)
_COVARIANCE_POSITIONS = (
    (0, 0),
    (1, 0),
    (1, 1),
    (2, 0),
    (2, 1),
    (2, 2),
    (3, 0),
    (3, 1),
    (3, 2),
    (3, 3),
    (4, 0),
    (4, 1),
    (4, 2),
    (4, 3),
    (4, 4),
    (5, 0),
    (5, 1),
    (5, 2),
    (5, 3),
    (5, 4),
    (5, 5),
)
_UNSUPPORTED_EXTENDED_COVARIANCE_PREFIXES = ("CDRG_", "CSRP_", "CTHR_")


class CcsdsCdmError(ValueError):
    """Raised when a CDM is invalid or outside OEL's bounded public profile."""


@dataclass(frozen=True)
class CdmHeader:
    version: str
    creation_date: str
    originator: str
    message_for: str | None = None
    message_id: str | None = None


@dataclass(frozen=True)
class CdmRelativeMetadata:
    tca: str
    miss_distance_m: float
    relative_speed_m_s: float
    relative_position_rtn_m: tuple[float, float, float]
    relative_velocity_rtn_m_s: tuple[float, float, float]
    collision_probability: float | None = None
    collision_probability_method: str | None = None
    optional: Mapping[str, str] | None = None


@dataclass(frozen=True)
class CdmObject:
    object: str
    metadata: Mapping[str, str]
    state_eci_km_km_s: tuple[float, float, float, float, float, float]
    covariance_rtn_si: tuple[tuple[float, ...], ...]
    od_parameters: Mapping[str, str]
    additional_parameters: Mapping[str, str]
    user_defined: Mapping[str, str]


@dataclass(frozen=True)
class CdmMessage:
    header: CdmHeader
    relative: CdmRelativeMetadata
    objects: tuple[CdmObject, CdmObject]
    user_defined: Mapping[str, str] | None = None
    comments: tuple[str, ...] = ()
    source_sha256: str | None = None


def _assignment(line: str, line_number: int) -> tuple[str, str, str | None]:
    if "=" not in line:
        raise CcsdsCdmError(f"line {line_number}: expected KEY = VALUE assignment.")
    key, raw_value = (part.strip() for part in line.split("=", 1))
    if not key or not raw_value or not re.fullmatch(r"[A-Z][A-Z0-9_]*", key):
        raise CcsdsCdmError(f"line {line_number}: malformed CDM assignment.")
    match = _UNIT_VALUE.match(raw_value)
    value = match.group("value").strip() if match else raw_value
    unit = match.group("unit").strip() if match else None
    if not value:
        raise CcsdsCdmError(f"line {line_number}: {key} requires a value.")
    return key, value, unit


def _normalized_unit(unit: str) -> str:
    return unit.lower().replace(" ", "").replace("^", "**")


def _validate_unit(key: str, unit: str | None, line_number: int) -> None:
    if unit is None:
        return
    expected: set[str] | None = None
    if key in {"MISS_DISTANCE", "RELATIVE_POSITION_R", "RELATIVE_POSITION_T", "RELATIVE_POSITION_N"}:
        expected = {"m"}
    elif key in {"RELATIVE_SPEED", "RELATIVE_VELOCITY_R", "RELATIVE_VELOCITY_T", "RELATIVE_VELOCITY_N"}:
        expected = {"m/s", "m*s**-1"}
    elif key in _STATE_KEYS[:3]:
        expected = {"km"}
    elif key in _STATE_KEYS[3:]:
        expected = {"km/s", "km*s**-1"}
    elif key in _COVARIANCE_KEYS:
        row, column = _COVARIANCE_POSITIONS[_COVARIANCE_KEYS.index(key)]
        expected = (
            {"m**2"}
            if row < 3 and column < 3
            else {"m**2/s**2", "m**2*s**-2"}
            if row >= 3 and column >= 3
            else {"m**2/s", "m**2*s**-1"}
        )
    if expected is not None and _normalized_unit(unit) not in expected:
        raise CcsdsCdmError(f"line {line_number}: {key} unit [{unit}] is invalid; expected one of {sorted(expected)}.")


def _finite(values: Mapping[str, str], key: str) -> float:
    value = values.get(key)
    if value is None:
        raise CcsdsCdmError(f"CDM is missing required keyword {key}.")
    if not _NUMBER.fullmatch(value):
        raise CcsdsCdmError(f"CDM keyword {key} must be numeric.")
    result = float(value)
    if not math.isfinite(result):
        raise CcsdsCdmError(f"CDM keyword {key} must be finite.")
    return result


def _date(value: str, key: str) -> None:
    try:
        parse_epoch(value, TimeScale.UTC)
    except (FrameTimeError, ValueError) as exc:
        raise CcsdsCdmError(f"{key} must be a valid UTC CCSDS epoch: {exc}") from exc


def _covariance(values: Mapping[str, str]) -> tuple[tuple[float, ...], ...]:
    missing = [key for key in _COVARIANCE_KEYS if key not in values]
    if missing:
        raise CcsdsCdmError(f"CDM object covariance is missing: {missing}.")
    matrix = np.zeros((6, 6), dtype=float)
    for key, (row, column) in zip(_COVARIANCE_KEYS, _COVARIANCE_POSITIONS, strict=True):
        matrix[row, column] = matrix[column, row] = _finite(values, key)
    return tuple(tuple(float(value) for value in row) for row in matrix)


def _object(values: Mapping[str, str], expected: str) -> CdmObject:
    required_metadata = (
        "OBJECT",
        "OBJECT_DESIGNATOR",
        "CATALOG_NAME",
        "OBJECT_NAME",
        "INTERNATIONAL_DESIGNATOR",
        "EPHEMERIS_NAME",
        "COVARIANCE_METHOD",
        "MANEUVERABLE",
        "REF_FRAME",
    )
    missing = [key for key in required_metadata if not values.get(key, "").strip()]
    if missing:
        raise CcsdsCdmError(f"{expected} metadata is missing: {missing}.")
    if values["OBJECT"].upper() != expected:
        raise CcsdsCdmError(f"CDM object segments must be ordered OBJECT1 then OBJECT2; found {values['OBJECT']!r}.")
    state = tuple(_finite(values, key) for key in _STATE_KEYS)
    return CdmObject(
        object=expected,
        metadata={key: values[key] for key in _OBJECT_METADATA_KEYS if key in values and key != "OBJECT"},
        state_eci_km_km_s=state,
        covariance_rtn_si=_covariance(values),
        od_parameters={key: values[key] for key in _OD_KEYS if key in values},
        additional_parameters={key: values[key] for key in _ADDITIONAL_KEYS if key in values},
        user_defined={key: value for key, value in values.items() if key.startswith("USER_DEFINED_")},
    )


def parse_cdm_kvn(text: str, *, source_sha256: str | None = None) -> CdmMessage:
    if not isinstance(text, str) or "\x00" in text:
        raise CcsdsCdmError("CDM input must be NUL-free Unicode text.")
    if len(text.encode("utf-8")) > MAX_CDM_BYTES:
        raise CcsdsCdmError(f"CDM input exceeds the {MAX_CDM_BYTES}-byte limit.")
    lines = text.splitlines()
    if len(lines) > MAX_CDM_LINES:
        raise CcsdsCdmError(f"CDM input exceeds the {MAX_CDM_LINES} line limit.")
    preamble: dict[str, str] = {}
    object_values: list[dict[str, str]] = []
    active = preamble
    comments: list[str] = []
    first_key: str | None = None
    for line_number, raw in enumerate(lines, start=1):
        line = raw.strip()
        if not line:
            continue
        if line.startswith("COMMENT"):
            comment = line[len("COMMENT") :].strip()
            if not comment:
                raise CcsdsCdmError(f"line {line_number}: COMMENT requires text.")
            comments.append(comment)
            continue
        key, value, unit = _assignment(line, line_number)
        _validate_unit(key, unit, line_number)
        if first_key is None:
            first_key = key
            if key != "CCSDS_CDM_VERS":
                raise CcsdsCdmError("The first non-comment CDM keyword must be CCSDS_CDM_VERS.")
        if any(key.startswith(prefix) for prefix in _UNSUPPORTED_EXTENDED_COVARIANCE_PREFIXES):
            raise CcsdsCdmError(f"line {line_number}: {key} is outside the public 6x6 RTN covariance profile.")
        if key == "OBJECT":
            active = {}
            object_values.append(active)
        if key in active:
            raise CcsdsCdmError(f"line {line_number}: duplicate keyword {key} in one CDM segment.")
        active[key] = value
    if first_key != "CCSDS_CDM_VERS":
        raise CcsdsCdmError("The first non-comment CDM keyword must be CCSDS_CDM_VERS.")
    if len(object_values) != 2:
        raise CcsdsCdmError("The bounded CDM profile requires exactly OBJECT1 and OBJECT2 segments.")
    allowed_preamble = set(_HEADER_KEYS) | set(_RELATIVE_KEYS)
    unknown_preamble = sorted(
        key for key in preamble if key not in allowed_preamble and not key.startswith("USER_DEFINED_")
    )
    allowed_object = (
        set(_OBJECT_METADATA_KEYS) | set(_OD_KEYS) | set(_ADDITIONAL_KEYS) | set(_STATE_KEYS) | set(_COVARIANCE_KEYS)
    )
    unknown_objects = sorted(
        {
            key
            for values in object_values
            for key in values
            if key not in allowed_object and not key.startswith("USER_DEFINED_")
        }
    )
    if unknown_preamble or unknown_objects:
        raise CcsdsCdmError(f"Unsupported CDM keywords: {unknown_preamble + unknown_objects}.")
    for key in ("CCSDS_CDM_VERS", "CREATION_DATE", "ORIGINATOR"):
        if not preamble.get(key, "").strip():
            raise CcsdsCdmError(f"CDM is missing required header keyword {key}.")
    required_relative = _RELATIVE_KEYS[:9]
    missing_relative = [key for key in required_relative if not preamble.get(key, "").strip()]
    if missing_relative:
        raise CcsdsCdmError(f"CDM relative metadata is missing: {missing_relative}.")
    message = CdmMessage(
        header=CdmHeader(
            version=preamble["CCSDS_CDM_VERS"],
            creation_date=preamble["CREATION_DATE"],
            originator=preamble["ORIGINATOR"],
            message_for=preamble.get("MESSAGE_FOR"),
            message_id=preamble.get("MESSAGE_ID"),
        ),
        relative=CdmRelativeMetadata(
            tca=preamble["TCA"],
            miss_distance_m=_finite(preamble, "MISS_DISTANCE"),
            relative_speed_m_s=_finite(preamble, "RELATIVE_SPEED"),
            relative_position_rtn_m=tuple(_finite(preamble, key) for key in _RELATIVE_KEYS[2 + 1 : 6]),
            relative_velocity_rtn_m_s=tuple(_finite(preamble, key) for key in _RELATIVE_KEYS[6:9]),
            collision_probability=_finite(preamble, "COLLISION_PROBABILITY")
            if "COLLISION_PROBABILITY" in preamble
            else None,
            collision_probability_method=preamble.get("COLLISION_PROBABILITY_METHOD"),
            optional={key: preamble[key] for key in _RELATIVE_KEYS[11:] if key in preamble},
        ),
        objects=(_object(object_values[0], "OBJECT1"), _object(object_values[1], "OBJECT2")),
        user_defined={key: value for key, value in preamble.items() if key.startswith("USER_DEFINED_")},
        comments=tuple(comments),
        source_sha256=source_sha256,
    )
    validate_cdm(message)
    return message


def read_cdm_kvn(path: str | Path, *, max_bytes: int = MAX_CDM_BYTES) -> CdmMessage:
    source = Path(path).expanduser().resolve()
    limit = int(max_bytes)
    try:
        raw = read_bounded_bytes(source, limit=limit)
    except DataIOError as exc:
        raise CcsdsCdmError(f"CDM input exceeds the {limit} byte limit.") from exc
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise CcsdsCdmError("CDM KVN must be valid UTF-8.") from exc
    return parse_cdm_kvn(text, source_sha256=hashlib.sha256(raw).hexdigest())


def validate_cdm(message: CdmMessage) -> None:
    if message.header.version != "1.0":
        raise CcsdsCdmError(f"{CCSDS_CDM_PROFILE} supports CCSDS CDM version 1.0 only.")
    _date(message.header.creation_date, "CREATION_DATE")
    _date(message.relative.tca, "TCA")
    scalar_values = (
        ("MISS_DISTANCE", message.relative.miss_distance_m),
        ("RELATIVE_SPEED", message.relative.relative_speed_m_s),
    )
    for label, value in scalar_values:
        try:
            finite_value = float(value)
        except (TypeError, ValueError) as exc:
            raise CcsdsCdmError(f"{label} must be finite numeric data.") from exc
        if not math.isfinite(finite_value):
            raise CcsdsCdmError(f"{label} must be finite numeric data.")
    try:
        relative_position = np.asarray(message.relative.relative_position_rtn_m, dtype=float)
        relative_velocity = np.asarray(message.relative.relative_velocity_rtn_m_s, dtype=float)
    except (TypeError, ValueError) as exc:
        raise CcsdsCdmError("CDM relative position and velocity must contain finite length-3 vectors.") from exc
    if (
        relative_position.shape != (3,)
        or relative_velocity.shape != (3,)
        or not np.all(np.isfinite(relative_position))
        or not np.all(np.isfinite(relative_velocity))
    ):
        raise CcsdsCdmError("CDM relative position and velocity must contain finite length-3 vectors.")
    if float(message.relative.miss_distance_m) < 0.0 or float(message.relative.relative_speed_m_s) < 0.0:
        raise CcsdsCdmError("MISS_DISTANCE and RELATIVE_SPEED must be nonnegative.")
    if message.relative.collision_probability is not None:
        try:
            probability = float(message.relative.collision_probability)
        except (TypeError, ValueError) as exc:
            raise CcsdsCdmError("COLLISION_PROBABILITY must be finite numeric data.") from exc
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise CcsdsCdmError("COLLISION_PROBABILITY must be finite and lie in [0, 1].")
    for item in message.objects:
        try:
            state = np.asarray(item.state_eci_km_km_s, dtype=float)
            covariance = np.asarray(item.covariance_rtn_si, dtype=float)
        except (TypeError, ValueError) as exc:
            raise CcsdsCdmError(f"{item.object} state and covariance must be numeric arrays.") from exc
        if state.shape != (6,) or not np.all(np.isfinite(state)):
            raise CcsdsCdmError(f"{item.object} state must contain six finite values.")
        if covariance.shape != (6, 6) or not np.all(np.isfinite(covariance)):
            raise CcsdsCdmError(f"{item.object} covariance must be a finite 6x6 matrix.")
        scale = max(1.0, float(np.max(np.abs(covariance))))
        if float(np.max(np.abs(covariance - covariance.T))) > 1.0e-12 * scale:
            raise CcsdsCdmError(f"{item.object} covariance must be symmetric.")
        minimum = float(np.min(np.linalg.eigvalsh(covariance)))
        if minimum < -1.0e-12 * scale:
            raise CcsdsCdmError(f"{item.object} covariance is not positive semidefinite; minimum eigenvalue={minimum}.")


def _format_number(value: float) -> str:
    return format(float(value), ".17g")


def serialize_cdm_kvn(message: CdmMessage) -> str:
    validate_cdm(message)
    lines = [
        f"CCSDS_CDM_VERS = {message.header.version}",
        f"CREATION_DATE = {message.header.creation_date}",
        f"ORIGINATOR = {message.header.originator}",
    ]
    lines.extend(f"COMMENT {comment}" for comment in message.comments)
    for key, value in (("MESSAGE_FOR", message.header.message_for), ("MESSAGE_ID", message.header.message_id)):
        if value is not None:
            lines.append(f"{key} = {value}")
    for key in sorted(message.user_defined or {}):
        lines.append(f"{key} = {(message.user_defined or {})[key]}")
    relative = message.relative
    lines.extend(
        (
            f"TCA = {relative.tca}",
            f"MISS_DISTANCE = {_format_number(relative.miss_distance_m)} [m]",
            f"RELATIVE_SPEED = {_format_number(relative.relative_speed_m_s)} [m/s]",
        )
    )
    for key, value in zip(_RELATIVE_KEYS[3:6], relative.relative_position_rtn_m, strict=True):
        lines.append(f"{key} = {_format_number(value)} [m]")
    for key, value in zip(_RELATIVE_KEYS[6:9], relative.relative_velocity_rtn_m_s, strict=True):
        lines.append(f"{key} = {_format_number(value)} [m/s]")
    if relative.collision_probability is not None:
        lines.append(f"COLLISION_PROBABILITY = {_format_number(relative.collision_probability)}")
    if relative.collision_probability_method is not None:
        lines.append(f"COLLISION_PROBABILITY_METHOD = {relative.collision_probability_method}")
    for key in _RELATIVE_KEYS[11:]:
        if relative.optional and key in relative.optional:
            lines.append(f"{key} = {relative.optional[key]}")
    for item in message.objects:
        lines.append("")
        lines.append(f"OBJECT = {item.object}")
        for key in _OBJECT_METADATA_KEYS[1:]:
            if key in item.metadata:
                lines.append(f"{key} = {item.metadata[key]}")
        for collection, keys in ((item.od_parameters, _OD_KEYS), (item.additional_parameters, _ADDITIONAL_KEYS)):
            for key in keys:
                if key in collection:
                    lines.append(f"{key} = {collection[key]}")
        for key, value, unit in zip(
            _STATE_KEYS, item.state_eci_km_km_s, ("km", "km", "km", "km/s", "km/s", "km/s"), strict=True
        ):
            lines.append(f"{key} = {_format_number(value)} [{unit}]")
        covariance = np.asarray(item.covariance_rtn_si, dtype=float)
        for key, (row, column) in zip(_COVARIANCE_KEYS, _COVARIANCE_POSITIONS, strict=True):
            unit = "m**2" if row < 3 and column < 3 else "m**2/s**2" if row >= 3 and column >= 3 else "m**2/s"
            lines.append(f"{key} = {_format_number(covariance[row, column])} [{unit}]")
        for key in sorted(item.user_defined):
            lines.append(f"{key} = {item.user_defined[key]}")
    return "\n".join(lines) + "\n"


def write_cdm_kvn(message: CdmMessage, path: str | Path) -> Path:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(serialize_cdm_kvn(message), encoding="utf-8")
    return target


def inspect_cdm(message_or_path: CdmMessage | str | Path) -> dict[str, Any]:
    message = read_cdm_kvn(message_or_path) if not isinstance(message_or_path, CdmMessage) else message_or_path
    reported_position_norm = float(np.linalg.norm(message.relative.relative_position_rtn_m))
    ready_issues: list[str] = []
    for item in message.objects:
        if item.metadata.get("REF_FRAME", "").upper() not in {"EME2000", "GCRF"}:
            ready_issues.append(f"{item.object} REF_FRAME is not EME2000 or GCRF.")
    frames = {item.metadata.get("REF_FRAME", "").upper() for item in message.objects}
    if len(frames) != 1:
        ready_issues.append("OBJECT1 and OBJECT2 REF_FRAME values differ; explicit frame conversion is required.")
    return {
        "profile": CCSDS_CDM_PROFILE,
        "valid": True,
        "analysis_ready": not ready_issues,
        "analysis_ready_issues": ready_issues,
        "version": message.header.version,
        "tca": message.relative.tca,
        "objects": [
            {
                "object": item.object,
                "object_name": item.metadata.get("OBJECT_NAME"),
                "object_designator": item.metadata.get("OBJECT_DESIGNATOR"),
                "ref_frame": item.metadata.get("REF_FRAME"),
            }
            for item in message.objects
        ],
        "reported": {
            "miss_distance_m": message.relative.miss_distance_m,
            "relative_speed_m_s": message.relative.relative_speed_m_s,
            "collision_probability": message.relative.collision_probability,
        },
        "semantic_checks": {
            "relative_position_norm_m": reported_position_norm,
            "miss_distance_minus_position_norm_m": message.relative.miss_distance_m - reported_position_norm,
        },
        "source_sha256": message.source_sha256,
        "limitations": [
            "The public profile accepts one KVN message with exactly two objects and 6x6 RTN covariances.",
            "XML, NDM containers, extended drag/SRP/thrust covariance terms, and operational disposition are not supported.",
            "Inspection validates structure and semantics; it does not endorse the source or reported probability.",
        ],
    }


def compare_cdm(left: CdmMessage | str | Path, right: CdmMessage | str | Path) -> dict[str, Any]:
    left_message = read_cdm_kvn(left) if not isinstance(left, CdmMessage) else left
    right_message = read_cdm_kvn(right) if not isinstance(right, CdmMessage) else right
    left_text = serialize_cdm_kvn(left_message)
    right_text = serialize_cdm_kvn(right_message)
    return {
        "equivalent": left_text == right_text,
        "left_canonical_sha256": hashlib.sha256(left_text.encode("utf-8")).hexdigest(),
        "right_canonical_sha256": hashlib.sha256(right_text.encode("utf-8")).hexdigest(),
    }


__all__ = [
    "CCSDS_CDM_PROFILE",
    "CcsdsCdmError",
    "CdmHeader",
    "CdmMessage",
    "CdmObject",
    "CdmRelativeMetadata",
    "compare_cdm",
    "inspect_cdm",
    "parse_cdm_kvn",
    "read_cdm_kvn",
    "serialize_cdm_kvn",
    "validate_cdm",
    "write_cdm_kvn",
]
