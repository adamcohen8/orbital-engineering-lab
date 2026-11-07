"""Bounded CCSDS OPM/OMM 3.0 KVN interoperability."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from sim.data_io.common import read_bytes as read_bounded_bytes
from sim.data_io.contracts import DataIOError
from sim.frame_time import FrameTimeError, TimeScale, parse_epoch
from sim.interchange.public_mission_input import MissionInputPacket, ingest_ephemeris_samples

CCSDS_ODM_PROFILE = "oel.ccsds-opm-omm-kvn.v0.1"
MAX_ODM_BYTES = 8 * 1024 * 1024
MAX_ODM_LINES = 100_000
_NUMBER = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?$")
_UNIT_VALUE = re.compile(r"^(?P<value>.*?)\s*\[(?P<unit>[^\]]+)\]\s*$")
_HEADER = ("CREATION_DATE", "ORIGINATOR")
_METADATA = ("OBJECT_NAME", "OBJECT_ID", "CENTER_NAME", "REF_FRAME", "TIME_SYSTEM")
_STATE = ("EPOCH", "X", "Y", "Z", "X_DOT", "Y_DOT", "Z_DOT")
_KEPLERIAN = (
    "SEMI_MAJOR_AXIS", "ECCENTRICITY", "INCLINATION", "RA_OF_ASC_NODE",
    "ARG_OF_PERICENTER", "TRUE_ANOMALY", "GM",
)
_MEAN = (
    "EPOCH", "MEAN_MOTION", "ECCENTRICITY", "INCLINATION", "RA_OF_ASC_NODE",
    "ARG_OF_PERICENTER", "MEAN_ANOMALY", "GM",
)
_TLE = (
    "EPHEMERIS_TYPE", "CLASSIFICATION_TYPE", "NORAD_CAT_ID", "ELEMENT_SET_NO",
    "REV_AT_EPOCH", "BSTAR", "MEAN_MOTION_DOT", "MEAN_MOTION_DDOT",
)
_PHYSICAL = ("MASS", "SOLAR_RAD_AREA", "SOLAR_RAD_COEFF", "DRAG_AREA", "DRAG_COEFF")
_MANEUVER = (
    "MAN_EPOCH_IGNITION", "MAN_DURATION", "MAN_DELTA_MASS", "MAN_REF_FRAME",
    "MAN_DV_1", "MAN_DV_2", "MAN_DV_3",
)
_COVARIANCE_KEYS = (
    "CX_X", "CY_X", "CY_Y", "CZ_X", "CZ_Y", "CZ_Z",
    "CX_DOT_X", "CX_DOT_Y", "CX_DOT_Z", "CX_DOT_X_DOT",
    "CY_DOT_X", "CY_DOT_Y", "CY_DOT_Z", "CY_DOT_X_DOT", "CY_DOT_Y_DOT",
    "CZ_DOT_X", "CZ_DOT_Y", "CZ_DOT_Z", "CZ_DOT_X_DOT", "CZ_DOT_Y_DOT", "CZ_DOT_Z_DOT",
)
_COVARIANCE_POSITIONS = (
    (0, 0), (1, 0), (1, 1), (2, 0), (2, 1), (2, 2),
    (3, 0), (3, 1), (3, 2), (3, 3), (4, 0), (4, 1), (4, 2), (4, 3), (4, 4),
    (5, 0), (5, 1), (5, 2), (5, 3), (5, 4), (5, 5),
)
_EXPECTED_UNITS = {
    "X": "km", "Y": "km", "Z": "km",
    "X_DOT": "km/s", "Y_DOT": "km/s", "Z_DOT": "km/s",
    "SEMI_MAJOR_AXIS": "km", "INCLINATION": "deg", "RA_OF_ASC_NODE": "deg",
    "ARG_OF_PERICENTER": "deg", "TRUE_ANOMALY": "deg", "MEAN_ANOMALY": "deg",
    "GM": "km**3/s**2", "MEAN_MOTION": "rev/day",
    "MASS": "kg", "SOLAR_RAD_AREA": "m**2", "DRAG_AREA": "m**2",
    "BSTAR": "1/ER", "MEAN_MOTION_DOT": "rev/day**2", "MEAN_MOTION_DDOT": "rev/day**3",
    "MAN_DURATION": "s", "MAN_DELTA_MASS": "kg",
    "MAN_DV_1": "km/s", "MAN_DV_2": "km/s", "MAN_DV_3": "km/s",
}
for _covariance_key, (_row, _column) in zip(_COVARIANCE_KEYS, _COVARIANCE_POSITIONS):
    if _row < 3 and _column < 3:
        _EXPECTED_UNITS[_covariance_key] = "km**2"
    elif _row >= 3 and _column >= 3:
        _EXPECTED_UNITS[_covariance_key] = "km**2/s**2"
    else:
        _EXPECTED_UNITS[_covariance_key] = "km**2/s"


class CcsdsOdmError(ValueError):
    """Raised when OPM/OMM input is outside the bounded public profile."""


@dataclass(frozen=True)
class OdmHeader:
    version: str
    creation_date: str
    originator: str
    message_id: str | None = None
    classification: str | None = None


@dataclass(frozen=True)
class OdmMetadata:
    object_name: str
    object_id: str
    center_name: str
    ref_frame: str
    time_system: str
    mean_element_theory: str | None = None


@dataclass(frozen=True)
class OdmCovariance:
    matrix: tuple[tuple[float, ...], ...]
    ref_frame: str | None = None


@dataclass(frozen=True)
class OpmMessage:
    header: OdmHeader
    metadata: OdmMetadata
    state: Mapping[str, str]
    keplerian: Mapping[str, str]
    physical: Mapping[str, str]
    maneuvers: tuple[Mapping[str, str], ...]
    covariance: OdmCovariance | None
    user_defined: Mapping[str, str]
    units: Mapping[str, str]
    comments: tuple[str, ...] = ()
    source_sha256: str | None = None


@dataclass(frozen=True)
class OmmMessage:
    header: OdmHeader
    metadata: OdmMetadata
    mean_elements: Mapping[str, str]
    tle_parameters: Mapping[str, str]
    covariance: OdmCovariance | None
    user_defined: Mapping[str, str]
    units: Mapping[str, str]
    comments: tuple[str, ...] = ()
    source_sha256: str | None = None


OdmMessage = OpmMessage | OmmMessage


def parse_odm_kvn(text: str, *, source_sha256: str | None = None) -> OdmMessage:
    if not isinstance(text, str) or "\x00" in text:
        raise CcsdsOdmError("ODM input must be NUL-free Unicode text.")
    lines = text.splitlines()
    if len(lines) > MAX_ODM_LINES:
        raise CcsdsOdmError(f"ODM input exceeds the {MAX_ODM_LINES} line limit.")
    values: dict[str, str] = {}
    units: dict[str, str] = {}
    comments: list[str] = []
    maneuvers: list[dict[str, str]] = []
    active_maneuver: dict[str, str] | None = None
    first_key: str | None = None
    last_section_rank = 0
    for line_number, raw in enumerate(lines, start=1):
        line = raw.strip()
        if not line:
            continue
        if line.startswith("COMMENT"):
            comment = line[len("COMMENT") :].strip()
            if not comment:
                raise CcsdsOdmError(f"line {line_number}: COMMENT requires text.")
            comments.append(comment)
            continue
        key, value, unit = _assignment(line, line_number)
        if first_key is None:
            first_key = key
            if first_key not in {"CCSDS_OPM_VERS", "CCSDS_OMM_VERS"}:
                raise CcsdsOdmError(
                    "The first non-comment ODM keyword must be CCSDS_OPM_VERS or CCSDS_OMM_VERS."
                )
        kind_hint = "OPM" if first_key == "CCSDS_OPM_VERS" else "OMM"
        section_rank = _keyword_section_rank(key, kind_hint)
        if section_rank < last_section_rank:
            raise CcsdsOdmError(f"line {line_number}: out-of-order {kind_hint} keyword {key}.")
        last_section_rank = max(last_section_rank, section_rank)
        if key == "MAN_EPOCH_IGNITION":
            if active_maneuver is not None:
                maneuvers.append(active_maneuver)
            active_maneuver = {}
        if key.startswith("MAN_"):
            if active_maneuver is None or key not in _MANEUVER:
                raise CcsdsOdmError(f"line {line_number}: misplaced or unsupported maneuver keyword {key}.")
            _unique(active_maneuver, key, value, line_number)
        else:
            _unique(values, key, value, line_number)
        if unit is not None:
            unit_key = f"{key}:{len(maneuvers)}" if key.startswith("MAN_") else key
            units[unit_key] = _canonical_unit(key, unit)
    if active_maneuver is not None:
        maneuvers.append(active_maneuver)
    if first_key not in {"CCSDS_OPM_VERS", "CCSDS_OMM_VERS"}:
        raise CcsdsOdmError("The first non-comment ODM keyword must be CCSDS_OPM_VERS or CCSDS_OMM_VERS.")
    kind = "OPM" if first_key == "CCSDS_OPM_VERS" else "OMM"
    allowed = _allowed_keys(kind)
    unknown = sorted(key for key in set(values) - allowed if not key.startswith("USER_DEFINED_"))
    if unknown:
        raise CcsdsOdmError(f"Unsupported {kind} keywords: {unknown}.")
    header = _header(values, kind)
    metadata = _metadata(values, kind)
    covariance = _covariance(values)
    user_defined = {key: value for key, value in values.items() if key.startswith("USER_DEFINED_")}
    if kind == "OPM":
        message: OdmMessage = OpmMessage(
            header=header,
            metadata=metadata,
            state=_select(values, _STATE, required=True, label="OPM state"),
            keplerian=_select(values, _KEPLERIAN, all_or_none=True, label="OPM Keplerian elements"),
            physical=_select(values, _PHYSICAL),
            maneuvers=tuple(maneuvers),
            covariance=covariance,
            user_defined=user_defined,
            units=units,
            comments=tuple(comments),
            source_sha256=source_sha256,
        )
    else:
        if maneuvers:
            raise CcsdsOdmError("OMM does not contain OPM maneuver blocks.")
        message = OmmMessage(
            header=header,
            metadata=metadata,
            mean_elements=_select(values, _MEAN, required=True, label="OMM mean elements"),
            tle_parameters=_select(values, _TLE),
            covariance=covariance,
            user_defined=user_defined,
            units=units,
            comments=tuple(comments),
            source_sha256=source_sha256,
        )
    validate_odm(message)
    return message


def read_odm_kvn(path: str | Path, *, max_bytes: int = MAX_ODM_BYTES) -> OdmMessage:
    source = Path(path).expanduser().resolve()
    limit = int(max_bytes)
    try:
        raw = read_bounded_bytes(source, limit=limit)
    except DataIOError as exc:
        raise CcsdsOdmError(f"ODM input exceeds the {limit} byte limit.") from exc
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise CcsdsOdmError("ODM KVN must be valid UTF-8.") from exc
    return parse_odm_kvn(text, source_sha256=hashlib.sha256(raw).hexdigest())


def validate_odm(message: OdmMessage) -> None:
    if message.header.version != "3.0":
        raise CcsdsOdmError(f"{CCSDS_ODM_PROFILE} supports CCSDS version 3.0 only.")
    for label, value in (
        ("CREATION_DATE", message.header.creation_date), ("ORIGINATOR", message.header.originator),
        ("OBJECT_NAME", message.metadata.object_name), ("OBJECT_ID", message.metadata.object_id),
        ("CENTER_NAME", message.metadata.center_name), ("REF_FRAME", message.metadata.ref_frame),
        ("TIME_SYSTEM", message.metadata.time_system),
    ):
        if not str(value).strip():
            raise CcsdsOdmError(f"{label} must not be empty.")
    _validate_epoch(message.header.creation_date, "UTC", "CREATION_DATE")
    if isinstance(message, OpmMessage):
        _validate_epoch(message.state["EPOCH"], message.metadata.time_system, "EPOCH")
        _finite_fields(message.state, _STATE[1:])
        _finite_fields(message.keplerian, _KEPLERIAN)
        _finite_fields(message.physical, _PHYSICAL)
        for index, maneuver in enumerate(message.maneuvers):
            missing = sorted(set(_MANEUVER) - set(maneuver))
            if missing:
                raise CcsdsOdmError(f"OPM maneuver {index} is missing: {missing}.")
            _validate_epoch(
                maneuver["MAN_EPOCH_IGNITION"],
                message.metadata.time_system,
                f"maneuver {index} MAN_EPOCH_IGNITION",
            )
            _finite_fields(maneuver, tuple(key for key in _MANEUVER if key != "MAN_REF_FRAME" and key != "MAN_EPOCH_IGNITION"))
    else:
        _validate_epoch(message.mean_elements["EPOCH"], message.metadata.time_system, "EPOCH")
        _finite_fields(message.mean_elements, tuple(key for key in _MEAN if key != "EPOCH"))
        eccentricity = float(message.mean_elements["ECCENTRICITY"])
        if not 0.0 <= eccentricity < 1.0 or float(message.mean_elements["MEAN_MOTION"]) <= 0.0:
            raise CcsdsOdmError("OMM requires 0 <= ECCENTRICITY < 1 and positive MEAN_MOTION.")
    if message.covariance is not None:
        _validate_covariance(message.covariance)
    _validate_message_units(message)


def inspect_odm(message_or_path: OdmMessage | str | Path) -> dict[str, Any]:
    message = read_odm_kvn(message_or_path) if not isinstance(message_or_path, (OpmMessage, OmmMessage)) else message_or_path
    validate_odm(message)
    kind = "OPM" if isinstance(message, OpmMessage) else "OMM"
    profile_issues = []
    if message.metadata.center_name.upper() != "EARTH":
        profile_issues.append("OEL mission-input use requires CENTER_NAME EARTH.")
    if kind == "OPM" and (message.metadata.ref_frame.upper(), message.metadata.time_system.upper()) != ("EME2000", "UTC"):
        profile_issues.append("Direct OPM mission-input use requires EME2000/UTC; convert explicitly first.")
    return {
        "document_type": f"ccsds_{kind.lower()}_kvn",
        "oel_profile": CCSDS_ODM_PROFILE,
        "valid": True,
        "mission_input_ready": kind == "OPM" and not profile_issues,
        "object_name": message.metadata.object_name,
        "object_id": message.metadata.object_id,
        "center_name": message.metadata.center_name,
        "ref_frame": message.metadata.ref_frame,
        "time_system": message.metadata.time_system,
        "covariance_present": message.covariance is not None,
        "maneuver_count": len(message.maneuvers) if isinstance(message, OpmMessage) else 0,
        "mean_element_theory": message.metadata.mean_element_theory,
        "source_sha256": message.source_sha256,
        "profile_issues": profile_issues,
        "non_claims": [
            "Parsing and semantic round-trip do not establish orbit or maneuver accuracy.",
            "OMM mean elements are preserved as a catalog product and are not silently converted to osculating state.",
            "OPM maneuver records are preserved but are not automatically scheduled or executed.",
        ],
    }


def serialize_odm_kvn(message: OdmMessage) -> str:
    validate_odm(message)
    kind = "OPM" if isinstance(message, OpmMessage) else "OMM"
    lines = [f"CCSDS_{kind}_VERS = {message.header.version}"]
    lines.extend(f"COMMENT  {comment}" for comment in message.comments)
    if message.header.classification is not None:
        lines.append(f"CLASSIFICATION = {message.header.classification}")
    lines.extend((f"CREATION_DATE = {message.header.creation_date}", f"ORIGINATOR = {message.header.originator}"))
    if message.header.message_id is not None:
        lines.append(f"MESSAGE_ID = {message.header.message_id}")
    metadata = {
        "OBJECT_NAME": message.metadata.object_name, "OBJECT_ID": message.metadata.object_id,
        "CENTER_NAME": message.metadata.center_name, "REF_FRAME": message.metadata.ref_frame,
        "TIME_SYSTEM": message.metadata.time_system,
    }
    if isinstance(message, OmmMessage):
        metadata["MEAN_ELEMENT_THEORY"] = str(message.metadata.mean_element_theory)
    lines.append("")
    lines.extend(f"{key} = {_with_unit(value, message.units.get(key))}" for key, value in metadata.items())
    blocks: list[Mapping[str, str]] = []
    if isinstance(message, OpmMessage):
        blocks.extend((message.state, message.keplerian, message.physical))
    else:
        blocks.extend((message.mean_elements, message.tle_parameters))
    for block in blocks:
        if block:
            lines.append("")
            lines.extend(f"{key} = {_with_unit(value, message.units.get(key))}" for key, value in block.items())
    if message.covariance is not None:
        lines.append("")
        if message.covariance.ref_frame is not None:
            lines.append(f"COV_REF_FRAME = {message.covariance.ref_frame}")
        for key, (row, column) in zip(_COVARIANCE_KEYS, _COVARIANCE_POSITIONS):
            lines.append(
                f"{key} = {_with_unit(f'{message.covariance.matrix[row][column]:.17g}', message.units.get(key))}"
            )
    if isinstance(message, OpmMessage):
        for index, maneuver in enumerate(message.maneuvers):
            lines.append("")
            lines.extend(
                f"{key} = {_with_unit(value, message.units.get(f'{key}:{index}'))}" for key, value in maneuver.items()
            )
    if message.user_defined:
        lines.append("")
        lines.extend(
            f"{key} = {_with_unit(value, message.units.get(key))}"
            for key, value in sorted(message.user_defined.items())
        )
    return "\n".join(lines) + "\n"


def write_odm_kvn(message: OdmMessage, path: str | Path, *, overwrite: bool = False) -> Path:
    target = Path(path).expanduser().resolve()
    text = serialize_odm_kvn(message)
    if target.exists() and target.read_text(encoding="utf-8") != text and not overwrite:
        raise CcsdsOdmError("ODM output exists with different content; pass overwrite=True to replace it.")
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists() or target.read_text(encoding="utf-8") != text:
        target.write_text(text, encoding="utf-8")
    return target


def compare_odm(left: OdmMessage | str | Path, right: OdmMessage | str | Path) -> dict[str, Any]:
    lhs = read_odm_kvn(left) if not isinstance(left, (OpmMessage, OmmMessage)) else left
    rhs = read_odm_kvn(right) if not isinstance(right, (OpmMessage, OmmMessage)) else right
    validate_odm(lhs)
    validate_odm(rhs)
    left_payload = _semantic_payload(lhs)
    right_payload = _semantic_payload(rhs)
    return {
        "schema": "oel.ccsds-opm-omm-semantic-comparison.v1",
        "status": "equivalent" if left_payload == right_payload else "different",
        "oel_profile": CCSDS_ODM_PROFILE,
        "left": left_payload,
        "right": right_payload,
        "non_claims": ["Semantic equality does not establish orbit accuracy or catalog freshness."],
    }


def opm_to_mission_input_packet(message_or_path: OpmMessage | str | Path) -> MissionInputPacket:
    message = read_odm_kvn(message_or_path) if not isinstance(message_or_path, OpmMessage) else message_or_path
    if not isinstance(message, OpmMessage):
        raise CcsdsOdmError("Mission-input conversion requires an OPM.")
    validate_odm(message)
    inspection = inspect_odm(message)
    if not inspection["mission_input_ready"]:
        raise CcsdsOdmError("OPM mission-input conversion requires Earth/EME2000/UTC; use explicit conversion first.")
    state = message.state
    return ingest_ephemeris_samples(
        object_id=message.metadata.object_id,
        role=message.metadata.object_name,
        samples=[{
            "time_s": 0.0,
            "position_eci_km": [float(state[key]) for key in ("X", "Y", "Z")],
            "velocity_eci_km_s": [float(state[key]) for key in ("X_DOT", "Y_DOT", "Z_DOT")],
        }],
        source_label=f"CCSDS OPM {message.header.message_id or message.metadata.object_id}",
        source_type="ccsds_opm_kvn_v3",
        frame="eci",
        source_metadata={
            "source_sha256": message.source_sha256,
            "epoch": state["EPOCH"],
            "source_ref_frame": message.metadata.ref_frame,
            "time_system": message.metadata.time_system,
            "maneuvers_preserved_not_scheduled": [dict(item) for item in message.maneuvers],
            "covariance_preserved_not_calibrated": None if message.covariance is None else [list(row) for row in message.covariance.matrix],
        },
    )


def _allowed_keys(kind: str) -> set[str]:
    common = {
        f"CCSDS_{kind}_VERS", "CLASSIFICATION", "CREATION_DATE", "ORIGINATOR", "MESSAGE_ID",
        *_METADATA, "COV_REF_FRAME", *_COVARIANCE_KEYS,
    }
    return common | (set(_STATE + _KEPLERIAN + _PHYSICAL) if kind == "OPM" else set(_MEAN + _TLE + ("MEAN_ELEMENT_THEORY",)))


def _keyword_section_rank(key: str, kind: str) -> int:
    if key == f"CCSDS_{kind}_VERS":
        return 0
    if key in {"CLASSIFICATION", "CREATION_DATE", "ORIGINATOR", "MESSAGE_ID"}:
        return 1
    if key in set(_METADATA) | {"MEAN_ELEMENT_THEORY"}:
        return 2
    if key in (_STATE if kind == "OPM" else _MEAN):
        return 3
    if kind == "OPM" and key in _KEPLERIAN:
        return 4
    if key in (_PHYSICAL if kind == "OPM" else _TLE):
        return 5
    if key == "COV_REF_FRAME" or key in _COVARIANCE_KEYS:
        return 6
    if key.startswith("MAN_"):
        return 7
    if key.startswith("USER_DEFINED_"):
        return 8
    return 9


def _header(values: Mapping[str, str], kind: str) -> OdmHeader:
    missing = sorted(set(_HEADER) - set(values))
    if missing:
        raise CcsdsOdmError(f"{kind} header is missing: {missing}.")
    return OdmHeader(values[f"CCSDS_{kind}_VERS"], values["CREATION_DATE"], values["ORIGINATOR"], values.get("MESSAGE_ID"), values.get("CLASSIFICATION"))


def _metadata(values: Mapping[str, str], kind: str) -> OdmMetadata:
    missing = sorted(set(_METADATA) - set(values))
    if missing:
        raise CcsdsOdmError(f"{kind} metadata is missing: {missing}.")
    theory = values.get("MEAN_ELEMENT_THEORY")
    if kind == "OMM" and not theory:
        raise CcsdsOdmError("OMM requires MEAN_ELEMENT_THEORY.")
    return OdmMetadata(*(values[key] for key in _METADATA), mean_element_theory=theory)


def _covariance(values: Mapping[str, str]) -> OdmCovariance | None:
    present = set(_COVARIANCE_KEYS) & set(values)
    if not present:
        return None
    missing = sorted(set(_COVARIANCE_KEYS) - set(values))
    if missing:
        raise CcsdsOdmError(f"ODM covariance is incomplete: {missing}.")
    matrix = np.zeros((6, 6), dtype=float)
    for key, (row, column) in zip(_COVARIANCE_KEYS, _COVARIANCE_POSITIONS):
        number = _float(values[key], key)
        matrix[row, column] = number
        matrix[column, row] = number
    return OdmCovariance(tuple(tuple(float(item) for item in row) for row in matrix), values.get("COV_REF_FRAME"))


def _select(
    values: Mapping[str, str],
    keys: Sequence[str],
    *,
    required: bool = False,
    all_or_none: bool = False,
    label: str = "section",
) -> dict[str, str]:
    selected = {key: values[key] for key in keys if key in values}
    if required and len(selected) != len(keys):
        raise CcsdsOdmError(f"{label} is missing: {sorted(set(keys) - set(selected))}.")
    if all_or_none and selected and len(selected) != len(keys):
        raise CcsdsOdmError(f"{label} must be supplied completely or omitted.")
    return selected


def _assignment(line: str, line_number: int) -> tuple[str, str, str | None]:
    if "=" not in line:
        raise CcsdsOdmError(f"line {line_number}: expected KEY = VALUE.")
    key, raw_value = (part.strip() for part in line.split("=", 1))
    if not key or not raw_value:
        raise CcsdsOdmError(f"line {line_number}: empty ODM key or value.")
    normalized_key = key.upper()
    match = (
        _UNIT_VALUE.fullmatch(raw_value)
        if normalized_key in _EXPECTED_UNITS or normalized_key.startswith("USER_DEFINED_")
        else None
    )
    value = raw_value if match is None else match.group("value").strip()
    unit = None if match is None else match.group("unit").strip()
    return normalized_key, value, unit


def _unique(values: dict[str, str], key: str, value: str, line_number: int) -> None:
    if key in values:
        raise CcsdsOdmError(f"line {line_number}: duplicate ODM keyword {key}.")
    values[key] = value


def _float(value: str, label: str) -> float:
    if not _NUMBER.fullmatch(str(value)):
        raise CcsdsOdmError(f"{label} must be a finite decimal number.")
    number = float(value)
    if not math.isfinite(number):
        raise CcsdsOdmError(f"{label} must be finite.")
    return number


def _finite_fields(values: Mapping[str, str], keys: Sequence[str]) -> None:
    for key in keys:
        if key in values:
            _float(values[key], key)


def _validate_covariance(covariance: OdmCovariance) -> None:
    matrix = np.asarray(covariance.matrix, dtype=float)
    if matrix.shape != (6, 6) or not np.all(np.isfinite(matrix)):
        raise CcsdsOdmError("ODM covariance must be a finite symmetric 6x6 matrix.")
    scale = float(np.max(np.abs(matrix)))
    tolerance = 64.0 * np.finfo(float).eps * scale * matrix.shape[0]
    if not np.allclose(matrix, matrix.T, atol=tolerance, rtol=0.0):
        raise CcsdsOdmError("ODM covariance must be a finite symmetric 6x6 matrix.")
    if np.any(np.diag(matrix) < 0.0):
        raise CcsdsOdmError(
            "ODM covariance must be positive semidefinite; diagonal variances must be non-negative."
        )
    if float(np.min(np.linalg.eigvalsh(matrix))) < -tolerance:
        raise CcsdsOdmError("ODM covariance must be positive semidefinite.")


def _canonical_unit(key: str, unit: str) -> str:
    supplied = str(unit or "").strip()
    if key.startswith("USER_DEFINED_"):
        if not supplied:
            raise CcsdsOdmError(f"{key} unit must not be empty.")
        return supplied
    expected = _EXPECTED_UNITS.get(key)
    if expected is None:
        raise CcsdsOdmError(f"{key} does not accept a unit annotation in {CCSDS_ODM_PROFILE}.")
    if supplied != expected:
        raise CcsdsOdmError(f"{key} unit must be {expected}; received {supplied!r}.")
    return expected


def _validate_message_units(message: OdmMessage) -> None:
    present: set[str] = set()
    if isinstance(message, OpmMessage):
        present.update(message.state)
        present.update(message.keplerian)
        present.update(message.physical)
        for index, maneuver in enumerate(message.maneuvers):
            present.update(f"{key}:{index}" for key in maneuver)
    else:
        present.update(message.mean_elements)
        present.update(message.tle_parameters)
    if message.covariance is not None:
        present.update(_COVARIANCE_KEYS)
    present.update(message.user_defined)
    for unit_key, unit in message.units.items():
        base_key, separator, index_text = str(unit_key).partition(":")
        if separator:
            if not base_key.startswith("MAN_") or not index_text.isdigit():
                raise CcsdsOdmError(f"Invalid indexed ODM unit key {unit_key!r}.")
        if str(unit_key) not in present:
            raise CcsdsOdmError(f"ODM unit key {unit_key!r} has no matching value.")
        _canonical_unit(base_key, unit)


def _validate_epoch(value: str, time_system: str, label: str) -> None:
    try:
        scale = TimeScale(str(time_system).upper())
    except ValueError as exc:
        raise CcsdsOdmError("The bounded ODM profile supports UTC, TAI, TT, and UT1 time systems.") from exc
    try:
        parse_epoch(value, scale, dut1_s=0.0 if scale is TimeScale.UT1 else None)
    except FrameTimeError as exc:
        raise CcsdsOdmError(f"{label} is not a valid {scale.value} epoch: {exc}") from exc


def _semantic_payload(message: OdmMessage) -> dict[str, Any]:
    data = {
        "kind": "OPM" if isinstance(message, OpmMessage) else "OMM",
        "header": message.header.__dict__,
        "metadata": message.metadata.__dict__,
        "user_defined": dict(message.user_defined),
        "user_defined_units": {
            key: message.units[key]
            for key in sorted(message.user_defined)
            if key in message.units
        },
        "covariance": None if message.covariance is None else message.covariance.__dict__,
    }
    if isinstance(message, OpmMessage):
        data.update(
            state=_normalized_values(message.state),
            keplerian=_normalized_values(message.keplerian),
            physical=_normalized_values(message.physical),
            maneuvers=[_normalized_values(item) for item in message.maneuvers],
        )
    else:
        data.update(
            mean_elements=_normalized_values(message.mean_elements),
            tle_parameters=_normalized_values(message.tle_parameters),
        )
    return json.loads(json.dumps(data, sort_keys=True))


def _normalized_values(values: Mapping[str, str]) -> dict[str, str | float]:
    return {
        key: float(value) if _NUMBER.fullmatch(str(value)) and key not in {"NORAD_CAT_ID", "ELEMENT_SET_NO", "REV_AT_EPOCH"} else value
        for key, value in values.items()
    }


def _with_unit(value: str, unit: str | None) -> str:
    return str(value) if unit is None else f"{value} [{unit}]"


__all__ = [
    "CCSDS_ODM_PROFILE", "CcsdsOdmError", "OdmCovariance", "OdmHeader", "OdmMessage", "OdmMetadata",
    "OmmMessage", "OpmMessage", "compare_odm", "inspect_odm", "opm_to_mission_input_packet", "parse_odm_kvn",
    "read_odm_kvn", "serialize_odm_kvn", "validate_odm", "write_odm_kvn",
]
