"""Bounded CCSDS TDM 2.0 KVN interchange for public OD workflows."""

from __future__ import annotations

import hashlib
import math
import re
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from sim.frame_time import TimeScale, parse_epoch
from sim.utils.io import SafeReadError, SafeReadLimitError, read_regular_file_nofollow

CCSDS_TDM_PROFILE = "oel.ccsds-tdm-kvn.v0.1"
MAX_TDM_BYTES = 16 * 1024 * 1024
MAX_TDM_LINES = 250_000
MAX_TDM_SEGMENTS = 10_000
MAX_TDM_RECORDS = 1_000_000

_ASSIGNMENT_KEY = re.compile(r"^[A-Z][A-Z0-9_]*$")
_NUMBER = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?$")
_PARTICIPANT = re.compile(r"^PARTICIPANT_[1-5]$")
_SUPPORTED_OBSERVABLES = {"ANGLE_1", "ANGLE_2", "RANGE"}
_UNSUPPORTED_SIGNAL_OBSERVABLES = {
    "DOPPLER_COUNT",
    "DOPPLER_INSTANTANEOUS",
    "DOPPLER_INTEGRATED",
    "RECEIVE_FREQ",
    "RECEIVE_PHASE_CT",
    "TRANSMIT_FREQ",
    "TRANSMIT_PHASE_CT",
}
_METADATA_KEYS = {
    "TRACK_ID",
    "TIME_SYSTEM",
    "START_TIME",
    "STOP_TIME",
    "MODE",
    "PATH",
    "ANGLE_TYPE",
    "RANGE_MODE",
    "RANGE_MODULUS",
    "RANGE_UNITS",
    "TIMETAG_REF",
    "DATA_QUALITY",
    "CORRECTIONS_APPLIED",
    "CORRECTION_ANGLE_1",
    "CORRECTION_ANGLE_2",
    "CORRECTION_RANGE",
}
_METADATA_ORDER = (
    "TRACK_ID",
    "TIME_SYSTEM",
    "START_TIME",
    "STOP_TIME",
    "PARTICIPANT_1",
    "PARTICIPANT_2",
    "PARTICIPANT_3",
    "PARTICIPANT_4",
    "PARTICIPANT_5",
    "MODE",
    "PATH",
    "TIMETAG_REF",
    "RANGE_MODE",
    "RANGE_MODULUS",
    "RANGE_UNITS",
    "ANGLE_TYPE",
    "DATA_QUALITY",
    "CORRECTIONS_APPLIED",
    "CORRECTION_RANGE",
    "CORRECTION_ANGLE_1",
    "CORRECTION_ANGLE_2",
)


class CcsdsTdmError(ValueError):
    """Raised when a TDM is invalid or outside OEL's bounded public profile."""


@dataclass(frozen=True)
class TdmHeader:
    version: str
    creation_date: str
    originator: str
    message_id: str | None = None


@dataclass(frozen=True)
class TdmMetadata:
    values: Mapping[str, str]

    @property
    def station_id(self) -> str:
        return str(self.values["PARTICIPANT_1"])

    @property
    def object_id(self) -> str:
        return str(self.values["PARTICIPANT_2"])


@dataclass(frozen=True)
class TdmObservation:
    keyword: str
    epoch_utc: str
    value: float
    epoch_tai_seconds: float | None = None


@dataclass(frozen=True)
class TdmSegment:
    metadata: TdmMetadata
    observations: tuple[TdmObservation, ...]
    metadata_comments: tuple[str, ...] = ()
    data_comments: tuple[str, ...] = ()


@dataclass(frozen=True)
class TdmMessage:
    header: TdmHeader
    segments: tuple[TdmSegment, ...]
    header_comments: tuple[str, ...] = ()
    source_sha256: str | None = None


def _assignment(line: str, line_number: int) -> tuple[str, str]:
    if "=" not in line:
        raise CcsdsTdmError(f"line {line_number}: expected KEY = VALUE assignment.")
    key, value = (part.strip() for part in line.split("=", 1))
    if not key or not value or not _ASSIGNMENT_KEY.fullmatch(key):
        raise CcsdsTdmError(f"line {line_number}: malformed TDM assignment.")
    return key, value


def _comment(line: str, line_number: int) -> str:
    if line == "COMMENT":
        return ""
    if not line.startswith("COMMENT "):
        raise CcsdsTdmError(f"line {line_number}: malformed COMMENT line.")
    return line[len("COMMENT ") :]


def _epoch_tai_seconds(text: str, *, field: str) -> float:
    try:
        return float(parse_epoch(text, TimeScale.UTC).tai_seconds)
    except ValueError as exc:
        raise CcsdsTdmError(f"{field} must be a valid UTC calendar or ordinal epoch.") from exc


def _validate_segment(
    metadata: Mapping[str, str],
    observations: tuple[TdmObservation, ...],
    *,
    segment_index: int,
) -> None:
    required = {"TIME_SYSTEM", "START_TIME", "STOP_TIME", "PARTICIPANT_1", "PARTICIPANT_2", "MODE", "PATH"}
    missing = sorted(required - set(metadata))
    if missing:
        raise CcsdsTdmError(f"segment {segment_index}: missing required metadata {missing}.")
    if metadata["TIME_SYSTEM"].upper() != "UTC":
        raise CcsdsTdmError(f"segment {segment_index}: the public profile requires TIME_SYSTEM = UTC.")
    if metadata["MODE"].upper() != "SEQUENTIAL":
        raise CcsdsTdmError(f"segment {segment_index}: the public profile requires MODE = SEQUENTIAL.")
    if metadata["PATH"].replace(" ", "") != "2,1":
        raise CcsdsTdmError(
            f"segment {segment_index}: the public reduced-geometric profile requires PATH = 2,1."
        )
    if not metadata["PARTICIPANT_1"].strip() or not metadata["PARTICIPANT_2"].strip():
        raise CcsdsTdmError(f"segment {segment_index}: participant identifiers must be non-empty.")
    extra_participants = sorted(
        key
        for key in metadata
        if _PARTICIPANT.fullmatch(key) and key not in {"PARTICIPANT_1", "PARTICIPANT_2"}
    )
    if extra_participants:
        raise CcsdsTdmError(
            f"segment {segment_index}: PATH = 2,1 does not permit extra participants {extra_participants}."
        )
    if not observations:
        raise CcsdsTdmError(f"segment {segment_index}: at least one tracking-data record is required.")

    start = _epoch_tai_seconds(metadata["START_TIME"], field=f"segment {segment_index} START_TIME")
    stop = _epoch_tai_seconds(metadata["STOP_TIME"], field=f"segment {segment_index} STOP_TIME")
    if stop < start:
        raise CcsdsTdmError(f"segment {segment_index}: STOP_TIME must not precede START_TIME.")
    timetag_ref = metadata.get("TIMETAG_REF")
    if timetag_ref is not None and timetag_ref.upper() != "RECEIVE":
        raise CcsdsTdmError(
            f"segment {segment_index}: the reduced-geometric profile supports only TIMETAG_REF = RECEIVE."
        )
    data_quality = metadata.get("DATA_QUALITY")
    if data_quality is not None and data_quality.upper() != "VALIDATED":
        raise CcsdsTdmError(
            f"segment {segment_index}: the reduced-geometric profile supports only DATA_QUALITY = VALIDATED."
        )
    correction_keys = sorted(key for key in metadata if key.startswith("CORRECTION_"))
    if correction_keys:
        raise CcsdsTdmError(
            f"segment {segment_index}: correction metadata {correction_keys} is unsupported; input values must already be reduced."
        )
    corrections_applied = metadata.get("CORRECTIONS_APPLIED")
    if corrections_applied is not None and corrections_applied.upper() != "YES":
        raise CcsdsTdmError(
            f"segment {segment_index}: reduced-geometric input requires CORRECTIONS_APPLIED = YES when supplied."
        )
    keywords = {item.keyword for item in observations}
    unsupported_keywords = sorted(
        str(keyword) for keyword in keywords if keyword not in _SUPPORTED_OBSERVABLES
    )
    if unsupported_keywords:
        raise CcsdsTdmError(
            f"segment {segment_index}: unsupported tracking-data keyword(s) {unsupported_keywords}."
        )
    if keywords & {"ANGLE_1", "ANGLE_2"} and metadata.get("ANGLE_TYPE", "").upper() != "AZEL":
        raise CcsdsTdmError(f"segment {segment_index}: angle records require ANGLE_TYPE = AZEL.")
    if "RANGE" in keywords:
        if metadata.get("RANGE_UNITS", "").lower() != "km":
            raise CcsdsTdmError(f"segment {segment_index}: range records require explicit RANGE_UNITS = km.")
        if metadata.get("RANGE_MODE", "").upper() != "ONE_WAY":
            raise CcsdsTdmError(f"segment {segment_index}: range records require RANGE_MODE = ONE_WAY.")
        try:
            modulus = float(metadata.get("RANGE_MODULUS", "nan"))
        except ValueError as exc:
            raise CcsdsTdmError(f"segment {segment_index}: RANGE_MODULUS must be numeric.") from exc
        if not math.isfinite(modulus) or modulus != 0.0:
            raise CcsdsTdmError(
                f"segment {segment_index}: ambiguous range is unsupported; RANGE_MODULUS must be 0."
            )

    last_epoch: float | None = None
    seen_keyword_epochs: set[tuple[str, float]] = set()
    for observation in observations:
        try:
            value = float(observation.value)
        except (TypeError, ValueError) as exc:
            raise CcsdsTdmError(
                f"segment {segment_index}: {observation.keyword} measurement must be finite numeric data."
            ) from exc
        if not math.isfinite(value):
            raise CcsdsTdmError(
                f"segment {segment_index}: {observation.keyword} measurement must be finite numeric data."
            )
        epoch = _epoch_tai_seconds(observation.epoch_utc, field=f"segment {segment_index} data epoch")
        if observation.epoch_tai_seconds is not None and observation.epoch_tai_seconds != epoch:
            raise CcsdsTdmError(
                f"segment {segment_index}: cached epoch identity does not match {observation.epoch_utc!r}."
            )
        if epoch < start or epoch > stop:
            raise CcsdsTdmError(
                f"segment {segment_index}: {observation.keyword} epoch lies outside START_TIME/STOP_TIME."
            )
        identity = (observation.keyword, epoch)
        if identity in seen_keyword_epochs:
            raise CcsdsTdmError(
                f"segment {segment_index}: duplicate {observation.keyword} timetag {observation.epoch_utc!r}."
            )
        if last_epoch is not None and epoch < last_epoch:
            raise CcsdsTdmError(
                f"segment {segment_index}: data records must be chronological."
            )
        seen_keyword_epochs.add(identity)
        last_epoch = epoch
        if observation.keyword == "ANGLE_1" and not -180.0 <= value < 360.0:
            raise CcsdsTdmError(f"segment {segment_index}: ANGLE_1 must lie within [-180, 360) deg.")
        if observation.keyword == "ANGLE_2" and not -90.0 <= value <= 90.0:
            raise CcsdsTdmError(f"segment {segment_index}: AZEL ANGLE_2 must lie within [-90, 90] deg.")
        if observation.keyword == "RANGE" and value < 0.0:
            raise CcsdsTdmError(f"segment {segment_index}: RANGE must be nonnegative.")


def parse_tdm_kvn(text: str, *, source_sha256: str | None = None) -> TdmMessage:
    """Parse strict CCSDS TDM 2.0 KVN into OEL's bounded public profile."""

    if not isinstance(text, str):
        raise TypeError("TDM KVN input must be text.")
    encoded = text.encode("utf-8")
    if len(encoded) > MAX_TDM_BYTES:
        raise CcsdsTdmError(f"TDM KVN input exceeds the {MAX_TDM_BYTES}-byte limit.")
    computed_source_sha256 = hashlib.sha256(encoded).hexdigest()
    if source_sha256 is not None and source_sha256 != computed_source_sha256:
        raise CcsdsTdmError("Supplied source_sha256 does not match the TDM KVN text.")
    lines = text.splitlines()
    if len(lines) > MAX_TDM_LINES:
        raise CcsdsTdmError(f"TDM KVN input exceeds the {MAX_TDM_LINES}-line limit.")
    for line_number, raw_line in enumerate(lines, start=1):
        try:
            raw_line.encode("ascii")
        except UnicodeEncodeError as exc:
            raise CcsdsTdmError(f"line {line_number}: TDM KVN requires printable ASCII text.") from exc
        if "\t" in raw_line or len(raw_line) > 254:
            raise CcsdsTdmError(f"line {line_number}: TDM lines must be tab-free and at most 254 characters.")

    header_values: dict[str, str] = {}
    header_comments: list[str] = []
    segments: list[TdmSegment] = []
    metadata: dict[str, str] | None = None
    observations: list[TdmObservation] | None = None
    metadata_comments: list[str] = []
    data_comments: list[str] = []
    state = "header"
    first_content_seen = False
    record_count = 0

    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line:
            continue
        if not first_content_seen:
            key, value = _assignment(line, line_number)
            if key != "CCSDS_TDM_VERS":
                raise CcsdsTdmError("CCSDS_TDM_VERS must be the first non-blank line.")
            header_values[key] = value
            first_content_seen = True
            continue

        if line == "META_START":
            if state != "header" or metadata is not None:
                raise CcsdsTdmError(f"line {line_number}: unexpected META_START.")
            if len(segments) >= MAX_TDM_SEGMENTS:
                raise CcsdsTdmError(f"TDM exceeds the {MAX_TDM_SEGMENTS}-segment limit.")
            metadata = {}
            observations = None
            metadata_comments = []
            data_comments = []
            state = "metadata"
            continue
        if line == "META_STOP":
            if state != "metadata" or metadata is None:
                raise CcsdsTdmError(f"line {line_number}: META_STOP has no matching META_START.")
            state = "await_data"
            continue
        if line == "DATA_START":
            if state != "await_data" or metadata is None:
                raise CcsdsTdmError(f"line {line_number}: DATA_START must follow META_STOP.")
            observations = []
            state = "data"
            continue
        if line == "DATA_STOP":
            if state != "data" or metadata is None or observations is None:
                raise CcsdsTdmError(f"line {line_number}: DATA_STOP has no matching DATA_START.")
            frozen_observations = tuple(observations)
            _validate_segment(metadata, frozen_observations, segment_index=len(segments))
            segments.append(
                TdmSegment(
                    metadata=TdmMetadata(dict(metadata)),
                    observations=frozen_observations,
                    metadata_comments=tuple(metadata_comments),
                    data_comments=tuple(data_comments),
                )
            )
            metadata = None
            observations = None
            state = "header"
            continue
        if line.startswith("COMMENT"):
            value = _comment(line, line_number)
            if state == "header":
                header_comments.append(value)
            elif state == "metadata":
                if metadata:
                    raise CcsdsTdmError(f"line {line_number}: metadata comments must immediately follow META_START.")
                metadata_comments.append(value)
            elif state == "data":
                data_comments.append(value)
            else:
                raise CcsdsTdmError(f"line {line_number}: COMMENT is not valid between metadata and data.")
            continue

        key, value = _assignment(line, line_number)
        if state == "header":
            if segments:
                raise CcsdsTdmError(f"line {line_number}: header content cannot appear after a completed segment.")
            if key not in {"CREATION_DATE", "ORIGINATOR", "MESSAGE_ID"}:
                raise CcsdsTdmError(f"line {line_number}: unsupported TDM header keyword {key!r}.")
            if key in header_values:
                raise CcsdsTdmError(f"line {line_number}: duplicate TDM header keyword {key!r}.")
            header_values[key] = value
        elif state == "metadata":
            assert metadata is not None
            if key not in _METADATA_KEYS and not _PARTICIPANT.fullmatch(key):
                raise CcsdsTdmError(f"line {line_number}: unsupported metadata keyword {key!r}.")
            if key in metadata:
                raise CcsdsTdmError(f"line {line_number}: duplicate metadata keyword {key!r}.")
            metadata[key] = value
        elif state == "data":
            assert observations is not None
            if key in _UNSUPPORTED_SIGNAL_OBSERVABLES or key.startswith(("DOPPLER_", "RECEIVE_FREQ_", "TRANSMIT_FREQ_")):
                raise CcsdsTdmError(
                    f"line {line_number}: {key} requires signal, path, count/integration, and sign semantics not in this profile."
                )
            if key not in _SUPPORTED_OBSERVABLES:
                raise CcsdsTdmError(f"line {line_number}: unsupported tracking-data keyword {key!r}.")
            parts = value.split()
            if len(parts) != 2 or not _NUMBER.fullmatch(parts[1]):
                raise CcsdsTdmError(f"line {line_number}: data records require 'keyword = timetag measurement'.")
            numeric = float(parts[1])
            if not math.isfinite(numeric):
                raise CcsdsTdmError(f"line {line_number}: measurement must be finite.")
            epoch_tai_seconds = _epoch_tai_seconds(parts[0], field=f"line {line_number} timetag")
            observations.append(TdmObservation(key, parts[0], numeric, epoch_tai_seconds))
            record_count += 1
            if record_count > MAX_TDM_RECORDS:
                raise CcsdsTdmError(f"TDM exceeds the {MAX_TDM_RECORDS}-record limit.")
        else:
            raise CcsdsTdmError(f"line {line_number}: expected DATA_START.")

    if not first_content_seen:
        raise CcsdsTdmError("TDM KVN input is empty.")
    if state != "header" or metadata is not None or observations is not None:
        raise CcsdsTdmError("TDM KVN ended inside an incomplete segment.")
    if header_values.get("CCSDS_TDM_VERS") != "2.0":
        raise CcsdsTdmError("The bounded public profile requires CCSDS_TDM_VERS = 2.0.")
    missing_header = sorted({"CREATION_DATE", "ORIGINATOR"} - set(header_values))
    if missing_header:
        raise CcsdsTdmError(f"TDM header is missing required fields {missing_header}.")
    _epoch_tai_seconds(header_values["CREATION_DATE"], field="CREATION_DATE")
    if not header_values["ORIGINATOR"].strip():
        raise CcsdsTdmError("ORIGINATOR must be non-empty.")
    if not segments:
        raise CcsdsTdmError("TDM requires at least one complete segment.")
    message = TdmMessage(
        header=TdmHeader(
            version="2.0",
            creation_date=header_values["CREATION_DATE"],
            originator=header_values["ORIGINATOR"],
            message_id=header_values.get("MESSAGE_ID"),
        ),
        segments=tuple(segments),
        header_comments=tuple(header_comments),
        source_sha256=computed_source_sha256,
    )
    validate_tdm(message)
    return message


def read_tdm_kvn(path: str | Path, *, max_bytes: int = MAX_TDM_BYTES) -> TdmMessage:
    source = Path(path).expanduser().absolute()
    limit = int(max_bytes)
    if limit < 0:
        raise CcsdsTdmError(f"TDM KVN input exceeds the {limit}-byte limit.")
    try:
        raw = read_regular_file_nofollow(source, max_bytes=limit)
    except SafeReadLimitError as exc:
        raise CcsdsTdmError(f"TDM KVN input exceeds the {limit}-byte limit.") from exc
    except (SafeReadError, ValueError) as exc:
        raise CcsdsTdmError(f"Could not safely read TDM KVN input {source}: {exc}") from exc
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise CcsdsTdmError("TDM KVN must be valid UTF-8 containing printable ASCII.") from exc
    return parse_tdm_kvn(text, source_sha256=hashlib.sha256(raw).hexdigest())


def _format_number(value: float) -> str:
    return format(float(value), ".16g")


def serialize_tdm_kvn(message: TdmMessage) -> str:
    validate_tdm(message)
    lines = [f"CCSDS_TDM_VERS = {message.header.version}"]
    lines.extend("COMMENT" if not item else f"COMMENT {item}" for item in message.header_comments)
    lines.extend(
        (
            f"CREATION_DATE = {message.header.creation_date}",
            f"ORIGINATOR = {message.header.originator}",
        )
    )
    if message.header.message_id is not None:
        lines.append(f"MESSAGE_ID = {message.header.message_id}")
    for segment in message.segments:
        lines.extend(("", "META_START"))
        lines.extend("COMMENT" if not item else f"COMMENT {item}" for item in segment.metadata_comments)
        values = dict(segment.metadata.values)
        for key in _METADATA_ORDER:
            if key in values:
                lines.append(f"{key} = {values.pop(key)}")
        if values:
            raise CcsdsTdmError(f"Cannot serialize unsupported metadata keys {sorted(values)}.")
        lines.extend(("META_STOP", "DATA_START"))
        lines.extend("COMMENT" if not item else f"COMMENT {item}" for item in segment.data_comments)
        for observation in segment.observations:
            lines.append(f"{observation.keyword} = {observation.epoch_utc} {_format_number(observation.value)}")
        lines.append("DATA_STOP")
    return "\n".join(lines) + "\n"


def validate_tdm(message: TdmMessage) -> None:
    if message.header.version != "2.0":
        raise CcsdsTdmError("The bounded public profile requires TDM version 2.0.")
    _epoch_tai_seconds(message.header.creation_date, field="CREATION_DATE")
    if not message.header.originator.strip():
        raise CcsdsTdmError("ORIGINATOR must be non-empty.")
    if not message.segments:
        raise CcsdsTdmError("TDM requires at least one segment.")
    seen_observations: set[tuple[str, str, str, float]] = set()
    for index, segment in enumerate(message.segments):
        _validate_segment(dict(segment.metadata.values), segment.observations, segment_index=index)
        for observation in segment.observations:
            identity = (
                segment.metadata.station_id,
                segment.metadata.object_id,
                observation.keyword,
                _epoch_tai_seconds(observation.epoch_utc, field=f"segment {index} data epoch"),
            )
            if identity in seen_observations:
                raise CcsdsTdmError(
                    f"segment {index}: duplicate observation across segments for "
                    f"{identity[0]!r}/{identity[1]!r} {identity[2]} at {observation.epoch_utc!r}."
                )
            seen_observations.add(identity)


def write_tdm_kvn(message: TdmMessage, path: str | Path) -> Path:
    target = Path(path).expanduser().resolve()
    payload = serialize_tdm_kvn(message)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        stream.write(payload)
        stream.flush()
    try:
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def _semantic_payload(message: TdmMessage) -> dict[str, Any]:
    return {
        "header": asdict(message.header),
        "segments": [
            {
                "metadata": dict(segment.metadata.values),
                "observations": [
                    {
                        "keyword": item.keyword,
                        "epoch_utc": item.epoch_utc,
                        "value": item.value,
                    }
                    for item in segment.observations
                ],
            }
            for segment in message.segments
        ],
    }


def compare_tdm(left: TdmMessage | str | Path, right: TdmMessage | str | Path) -> dict[str, Any]:
    left_message = left if isinstance(left, TdmMessage) else read_tdm_kvn(left)
    right_message = right if isinstance(right, TdmMessage) else read_tdm_kvn(right)
    left_payload = _semantic_payload(left_message)
    right_payload = _semantic_payload(right_message)
    return {
        "status": "equivalent" if left_payload == right_payload else "different",
        "equivalent": left_payload == right_payload,
        "left_source_sha256": left_message.source_sha256,
        "right_source_sha256": right_message.source_sha256,
    }


def inspect_tdm(path: str | Path) -> dict[str, Any]:
    message = read_tdm_kvn(path)
    keywords = sorted({item.keyword for segment in message.segments for item in segment.observations})
    return {
        "status": "valid",
        "profile": CCSDS_TDM_PROFILE,
        "source_sha256": message.source_sha256,
        "originator": message.header.originator,
        "message_id": message.header.message_id,
        "segment_count": len(message.segments),
        "record_count": sum(len(segment.observations) for segment in message.segments),
        "station_ids": sorted({segment.metadata.station_id for segment in message.segments}),
        "object_ids": sorted({segment.metadata.object_id for segment in message.segments}),
        "observable_keywords": keywords,
        "limitations": [
            "The public profile supports UTC sequential PATH=2,1 ANGLE_1/ANGLE_2 AZEL and unambiguous one-way RANGE in km.",
            "TDM XML, RADEC/XEYN/XSYE, Doppler/frequency/phase, media, light-time, and ambiguous or multi-way range are rejected.",
        ],
    }


__all__ = [
    "CCSDS_TDM_PROFILE",
    "CcsdsTdmError",
    "TdmHeader",
    "TdmMessage",
    "TdmMetadata",
    "TdmObservation",
    "TdmSegment",
    "compare_tdm",
    "inspect_tdm",
    "parse_tdm_kvn",
    "read_tdm_kvn",
    "serialize_tdm_kvn",
    "validate_tdm",
    "write_tdm_kvn",
]
