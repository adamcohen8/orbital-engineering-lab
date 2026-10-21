"""Preserving raw TDM IO; physical observable reduction stays in OD owners."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np

from .common import read_text, xml_root
from .contracts import MAX_SAMPLES, DataIOError
from .time import parse_time

OBSERVABLES = {
    "ANGLE_1",
    "ANGLE_2",
    "RANGE",
    "DOPPLER_COUNT",
    "DOPPLER_INSTANTANEOUS",
    "DOPPLER_INTEGRATED",
    "RECEIVE_FREQ",
    "TRANSMIT_FREQ",
    "RECEIVE_PHASE_CT",
    "TRANSMIT_PHASE_CT",
    "TRANSMIT_FREQ_RATE",
}
META = {
    "TRACK_ID",
    "DATA_TYPES",
    "TIME_SYSTEM",
    "START_TIME",
    "STOP_TIME",
    "MODE",
    "PATH",
    "PATH_1",
    "PATH_2",
    "ANGLE_TYPE",
    "REFERENCE_FRAME",
    "INTEGRATION_INTERVAL",
    "INTEGRATION_REF",
    "FREQ_OFFSET",
    "RANGE_MODE",
    "RANGE_MODULUS",
    "RANGE_UNITS",
    "TIMETAG_REF",
    "DATA_QUALITY",
    "CORRECTIONS_APPLIED",
    "DOPPLER_COUNT_BIAS",
    "DOPPLER_COUNT_SCALE",
    "DOPPLER_COUNT_ROLLOVER",
    "CORRECTION_ANGLE_1",
    "CORRECTION_ANGLE_2",
    "CORRECTION_DOPPLER",
    "CORRECTION_RANGE",
    "CORRECTION_RECEIVE",
    "CORRECTION_TRANSMIT",
}
HEADER = {"CCSDS_TDM_VERS", "CREATION_DATE", "ORIGINATOR", "MESSAGE_ID"}


@dataclass(frozen=True)
class TrackingDocument:
    header: dict
    segments: tuple[dict, ...]
    comments: tuple[str, ...] = ()


def parse_tracking(raw: str) -> TrackingDocument:
    header, segments, comments = {}, [], []
    active, data, section = header, None, "header"
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("COMMENT"):
            comments.append(line.removeprefix("COMMENT").strip())
            continue
        if line in {"META_START", "META_STOP", "DATA_START", "DATA_STOP"}:
            expected = {
                "header": "META_START",
                "complete": "META_START",
                "metadata": "META_STOP",
                "after_meta": "DATA_START",
                "data": "DATA_STOP",
            }.get(section)
            if line != expected:
                raise DataIOError("Invalid TDM section sequence.")
            if line == "META_START":
                active, data = {}, []
                segments.append({"metadata": active, "observations": data})
            section = {
                "META_START": "metadata",
                "META_STOP": "after_meta",
                "DATA_START": "data",
                "DATA_STOP": "complete",
            }[line]
            continue
        if "=" not in line:
            raise DataIOError("Malformed TDM assignment.")
        key, value = [part.strip() for part in line.split("=", 1)]
        if not re.fullmatch(r"[A-Z][A-Z0-9_]*", key) or not value:
            raise DataIOError("Invalid TDM keyword/value.")
        if section == "data":
            if key not in OBSERVABLES or len(data) >= MAX_SAMPLES:
                raise DataIOError(f"Unsupported/excessive TDM observable: {key}.")
            parts = value.split()
            if len(parts) != 2:
                raise DataIOError("TDM observable requires epoch and one scalar value.")
            epoch = parse_time(parts[0], active.get("TIME_SYSTEM", "UTC"))
            number = float(parts[1])
            if not np.isfinite(number):
                raise DataIOError("TDM observation must be finite.")
            data.append({"keyword": key, "epoch": parts[0], "value": number, "tai_seconds": epoch.tai_seconds})
        else:
            allowed = HEADER if section == "header" else META
            dynamic = (
                bool(re.fullmatch(r"(?:PARTICIPANT|TRANSMIT_DELAY|RECEIVE_DELAY)_[1-5]", key))
                if section == "metadata"
                else False
            )
            if (key not in allowed and not dynamic and not key.startswith("USER_DEFINED_")) or key in active:
                raise DataIOError(f"Unsupported or duplicate TDM keyword {key}.")
            active[key] = value
    if (
        section != "complete"
        or header.get("CCSDS_TDM_VERS") != "2.0"
        or not {"CREATION_DATE", "ORIGINATOR"} <= header.keys()
    ):
        raise DataIOError("Incomplete TDM 2.0 document.")
    for segment in segments:
        metadata, rows = segment["metadata"], segment["observations"]
        if not rows or not {"TIME_SYSTEM", "PARTICIPANT_1", "PARTICIPANT_2", "MODE"} <= metadata.keys():
            raise DataIOError("TDM segment requires time system, participants, mode, and observations.")
        if metadata.get("ANGLE_TYPE") not in {None, "AZEL", "RADEC"}:
            raise DataIOError("Only AZEL and RADEC angle conventions are supported by this IO profile.")
        identities = [(r["tai_seconds"], r["keyword"]) for r in rows]
        if len(set(identities)) != len(identities) or any(b[0] < a[0] for a, b in zip(identities, identities[1:])):
            raise DataIOError("Duplicate or nonchronological TDM observable records.")
        segment["measurement_semantics"] = "preserved_native_observables"
        segment["instantaneous_od_ready"] = False
    return TrackingDocument(header, tuple(segments), tuple(comments))


def read_tracking(path, *, format):
    raw = read_text(path)
    if format == "tdm-xml":
        root = xml_root(raw)
        if root.tag != "tdm" or root.get("version") != "2.0" or [n.tag for n in root] != ["header", "body"]:
            raise DataIOError("Expected a complete TDM 2.0 XML document.")
        lines = ["CCSDS_TDM_VERS = 2.0"]
        for child in root[0]:
            if list(child):
                raise DataIOError("Header values must be scalar.")
            lines.append(
                "COMMENT " + (child.text or "") if child.tag == "COMMENT" else f"{child.tag} = {child.text or ''}"
            )
        for segment in root[1]:
            if segment.tag != "segment" or [n.tag for n in segment] != ["metadata", "data"]:
                raise DataIOError("Invalid TDM XML segment structure.")
            lines.append("META_START")
            for child in segment[0]:
                if list(child) or child.get("units"):
                    raise DataIOError("TDM metadata must be scalar with standard implicit units.")
                lines.append(
                    "COMMENT " + (child.text or "") if child.tag == "COMMENT" else f"{child.tag} = {child.text or ''}"
                )
            lines += ["META_STOP", "DATA_START"]
            for child in segment[1]:
                if child.tag == "COMMENT":
                    lines.append("COMMENT " + (child.text or ""))
                    continue
                if (
                    child.tag != "observation"
                    or len(child) != 2
                    or child[0].tag != "EPOCH"
                    or list(child[0])
                    or list(child[1])
                ):
                    raise DataIOError("TDM XML observation requires epoch and one scalar observable.")
                if child[1].get("units"):
                    raise DataIOError("Observable units are defined by the TDM profile and metadata.")
                lines.append(f"{child[1].tag} = {child[0].text} {child[1].text}")
            lines.append("DATA_STOP")
        raw = "\n".join(lines)
    return parse_tracking(raw)


def encode_tracking(document: TrackingDocument, *, format: str):
    lines = [f"{key} = {value}" for key, value in document.header.items()]
    lines += ["COMMENT " + comment for comment in document.comments]
    for segment in document.segments:
        lines += [
            "META_START",
            *[f"{key} = {value}" for key, value in segment["metadata"].items()],
            "META_STOP",
            "DATA_START",
        ]
        lines += [f"{r['keyword']} = {r['epoch']} {r['value']:.17g}" for r in segment["observations"]]
        lines.append("DATA_STOP")
    raw = "\n".join(lines) + "\n"
    parse_tracking(raw)
    if format == "tdm-kvn":
        return raw.encode()
    if format != "tdm-xml":
        raise DataIOError("Expected TDM KVN or XML.")
    root = ET.Element("tdm", {"id": "CCSDS_TDM_VERS", "version": "2.0"})
    header, body = ET.SubElement(root, "header"), ET.SubElement(root, "body")
    for comment in document.comments:
        ET.SubElement(header, "COMMENT").text = comment
    for key, value in document.header.items():
        if key != "CCSDS_TDM_VERS":
            ET.SubElement(header, key).text = value
    for item in document.segments:
        segment = ET.SubElement(body, "segment")
        metadata, data = ET.SubElement(segment, "metadata"), ET.SubElement(segment, "data")
        for key, value in item["metadata"].items():
            ET.SubElement(metadata, key).text = value
        for row in item["observations"]:
            observation = ET.SubElement(data, "observation")
            ET.SubElement(observation, "EPOCH").text = row["epoch"]
            ET.SubElement(observation, row["keyword"]).text = format_float(row["value"])
    ET.indent(root)
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def format_float(value):
    return format(float(value), ".17g")
