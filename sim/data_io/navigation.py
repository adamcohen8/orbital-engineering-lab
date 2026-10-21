"""Bounded CCSDS syntax adapters over the existing authoritative KVN models.

XML is mapped structurally, with unknown elements rejected. This is not an
unrestricted flattening parser and never discards an unrecognized section.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import replace

from sim.interchange import ccsds_odm as odm
from sim.interchange.ccsds_cdm import parse_cdm_kvn, serialize_cdm_kvn
from sim.interchange.ccsds_oem import (
    OemCovariance,
    OemHeader,
    OemMessage,
    OemMetadata,
    OemSegment,
    OemState,
    parse_oem_kvn,
    serialize_oem_kvn,
)
from sim.interchange.ccsds_tdm import parse_tdm_kvn, serialize_tdm_kvn

from .common import digest, read_text, xml_root
from .contracts import DataIOError, Trajectory
from .time import at, creation_time, format_time, parse_time

STATE = ("EPOCH", "X", "Y", "Z", "X_DOT", "Y_DOT", "Z_DOT")
COV = odm._COVARIANCE_KEYS
META = {
    "OBJECT_NAME",
    "OBJECT_ID",
    "CENTER_NAME",
    "REF_FRAME",
    "REF_FRAME_EPOCH",
    "TIME_SYSTEM",
    "START_TIME",
    "STOP_TIME",
    "USEABLE_START_TIME",
    "USEABLE_STOP_TIME",
    "INTERPOLATION",
    "INTERPOLATION_DEGREE",
    "MEAN_ELEMENT_THEORY",
    "COMMENT",
}


def _children(node, allowed):
    if any(child.tag not in allowed for child in node):
        raise DataIOError(f"Unsupported XML element in {node.tag}: {[c.tag for c in node if c.tag not in allowed]}.")


def _leaves(node, allowed):
    _children(node, allowed)
    result = []
    seen = set()
    for child in node:
        if list(child) or not child.text or not child.text.strip():
            raise DataIOError(f"Expected nonempty scalar XML value for {child.tag}.")
        key = child.tag
        if key == "USER_DEFINED":
            key = "USER_DEFINED_" + str(child.get("parameter", ""))
        if key != "COMMENT" and key in seen:
            raise DataIOError(f"Duplicate XML keyword {key}.")
        seen.add(key)
        value = child.text.strip()
        if "\n" in value or "\r" in value:
            raise DataIOError("Multiline XML scalar cannot be mapped to KVN.")
        if child.get("units"):
            value += " [" + child.get("units") + "]"
        result.append((key, value))
    return result


def _assignments(pairs):
    return [f"{key} = {value}" if key != "COMMENT" else f"COMMENT {value}" for key, value in pairs]


def xml_to_kvn(raw: str, kind: str) -> str:
    root = xml_root(raw)
    if root.tag.lower() != kind:
        raise DataIOError(f"Expected a single {kind.upper()} XML root.")
    version = root.get("version")
    if not version:
        raise DataIOError("XML message version is required.")
    _children(root, {"header", "body"})
    if [child.tag for child in root] != ["header", "body"]:
        raise DataIOError("XML requires exactly one header followed by one body.")
    lines = [f"CCSDS_{kind.upper()}_VERS = {version}"]
    lines += _assignments(
        _leaves(root[0], {"COMMENT", "CLASSIFICATION", "CREATION_DATE", "ORIGINATOR", "MESSAGE_ID", "MESSAGE_FOR"})
    )
    if kind == "cdm":
        from sim.interchange import ccsds_cdm as cdm

        _children(root[1], {"relativeMetadataData", "segment"})
        relatives = root[1].findall("relativeMetadataData")
        segments = root[1].findall("segment")
        if len(relatives) != 1 or len(segments) != 2:
            raise DataIOError("CDM XML requires relative metadata and two object segments.")
        relative = relatives[0]
        _children(relative, {*cdm._RELATIVE_KEYS, "relativeStateVector", "COMMENT"})
        for node in relative:
            if node.tag == "relativeStateVector":
                lines += _assignments(_leaves(node, set(cdm._RELATIVE_KEYS)))
            else:
                wrapper = ET.Element("relative")
                wrapper.append(node)
                lines += _assignments(_leaves(wrapper, {*cdm._RELATIVE_KEYS, "COMMENT"}))
        for segment in segments:
            if [n.tag for n in segment] != ["metadata", "data"]:
                raise DataIOError("CDM object requires metadata followed by data.")
            lines += _assignments(_leaves(segment[0], {*cdm._OBJECT_METADATA_KEYS, "COMMENT"}))
            sections = {
                "odParameters": set(cdm._OD_KEYS),
                "additionalParameters": set(cdm._ADDITIONAL_KEYS),
                "stateVector": set(cdm._STATE_KEYS),
                "covarianceMatrix": set(cdm._COVARIANCE_KEYS),
                "userDefinedParameters": {"USER_DEFINED"},
            }
            _children(segment[1], set(sections) | {"COMMENT"})
            for node in segment[1]:
                if node.tag == "COMMENT":
                    lines.append("COMMENT " + (node.text or ""))
                else:
                    lines += _assignments(_leaves(node, sections[node.tag] | {"COMMENT"}))
        return "\n".join(lines) + "\n"
    _children(root[1], {"segment"})
    for segment in root[1]:
        if [child.tag for child in segment] != ["metadata", "data"]:
            raise DataIOError("Each XML segment requires exactly metadata then data.")
        metadata, data = segment
        if kind in {"oem", "tdm"}:
            lines.append("META_START")
        if kind == "tdm":
            from sim.interchange.ccsds_tdm import _METADATA_KEYS

            allowed = set(_METADATA_KEYS) | {"COMMENT", *[f"PARTICIPANT_{i}" for i in range(1, 6)]}
        else:
            allowed = META
        lines += _assignments(_leaves(metadata, allowed))
        if kind in {"oem", "tdm"}:
            lines.append("META_STOP")
        if kind == "oem":
            _children(data, {"stateVector", "covarianceMatrix", "COMMENT"})
            covariance_open = False
            for node in data:
                if node.tag == "COMMENT":
                    lines.append("COMMENT " + (node.text or ""))
                elif node.tag == "stateVector":
                    if covariance_open:
                        raise DataIOError("OEM state vectors must precede covariance blocks.")
                    values = dict(_leaves(node, set(STATE) | {"X_DDOT", "Y_DDOT", "Z_DDOT"}))
                    keys = STATE + (("X_DDOT", "Y_DDOT", "Z_DDOT") if "X_DDOT" in values else ())
                    # Units are normalized explicitly rather than stripped blindly.
                    units = {k: v.split(" [", 1)[1][:-1] for k, v in values.items() if " [" in v}
                    for key, unit in units.items():
                        expected = "km/s**2" if key.endswith("DDOT") else "km/s" if key.endswith("DOT") else "km"
                        if unit != expected:
                            raise DataIOError(f"Unsupported OEM XML unit for {key}: {unit}.")
                    if set(values) != set(keys):
                        raise DataIOError("OEM requires complete PV and, when supplied, complete acceleration.")
                    lines.append(" ".join(values[key].split(" [", 1)[0] for key in keys))
                else:
                    if not covariance_open:
                        lines.append("COVARIANCE_START")
                        covariance_open = True
                    values = dict(_leaves(node, set(COV) | {"EPOCH", "COV_REF_FRAME", "COMMENT"}))
                    if "EPOCH" not in values or not set(COV) <= values.keys():
                        raise DataIOError("OEM XML covariance requires epoch and complete 6x6 matrix.")
                    lines.append("EPOCH = " + values.pop("EPOCH"))
                    if "COV_REF_FRAME" in values:
                        lines.append("COV_REF_FRAME = " + values.pop("COV_REF_FRAME"))
                    for size in range(1, 7):
                        offset = size * (size - 1) // 2
                        row = []
                        for key in COV[offset : offset + size]:
                            value = values[key]
                            expected = odm._EXPECTED_UNITS[key]
                            if " [" in value and value.split(" [", 1)[1][:-1] != expected:
                                raise DataIOError(f"Unsupported covariance units for {key}.")
                            row.append(value.split(" [", 1)[0])
                        lines.append(" ".join(row))
            if covariance_open:
                lines.append("COVARIANCE_STOP")
        elif kind in {"opm", "omm"}:
            sections = {
                "stateVector": set(STATE),
                "keplerianElements": set(odm._KEPLERIAN),
                "meanElements": set(odm._MEAN),
                "tleParameters": set(odm._TLE),
                "spacecraftParameters": set(odm._PHYSICAL),
                "covarianceMatrix": set(COV) | {"COV_REF_FRAME"},
                "maneuverParameters": set(odm._MANEUVER),
                "userDefinedParameters": {"USER_DEFINED"},
            }
            _children(data, {*sections, "COMMENT"})
            for node in data:
                if node.tag == "COMMENT":
                    lines.append("COMMENT " + (node.text or ""))
                else:
                    lines += _assignments(_leaves(node, sections[node.tag] | {"COMMENT"}))
        elif kind == "tdm":
            lines.append("DATA_START")
            _children(data, {"observation", "COMMENT"})
            for node in data:
                if node.tag == "COMMENT":
                    lines.append("COMMENT " + (node.text or ""))
                    continue
                values = dict(_leaves(node, {"EPOCH", "ANGLE_1", "ANGLE_2", "RANGE"}))
                if "EPOCH" not in values or len(values) != 2:
                    raise DataIOError("TDM observation requires one epoch and one observable.")
                epoch = values.pop("EPOCH")
                key, value = next(iter(values.items()))
                if " [" in value:
                    raise DataIOError("TDM observable units must be declared through its metadata profile.")
                lines.append(f"{key} = {epoch} {value}")
            lines.append("DATA_STOP")
        else:
            raise DataIOError(f"Unsupported CCSDS XML family: {kind}.")
    return "\n".join(lines) + "\n"


def parse_navigation(raw: str, *, format: str):
    source_hash = digest(raw.encode())
    kind, syntax = format.split("-", 1)
    if syntax == "xml":
        raw = xml_to_kvn(raw, kind)
    elif syntax != "kvn":
        raise DataIOError("Expected KVN or XML navigation syntax.")
    first = next((line for line in raw.splitlines() if line.strip() and not line.startswith("COMMENT")), "")
    if kind in {"oem", "opm", "omm"} and first.strip() == f"CCSDS_{kind.upper()}_VERS = 2.0":
        # ODM2 uses the same bounded PV/mean-element subset; extensions stay rejected.
        if "MESSAGE_ID" in raw or "CLASSIFICATION =" in raw:
            raise DataIOError("Version 2.0 does not support the version 3 header additions.")
        raw = raw.replace(first, f"CCSDS_{kind.upper()}_VERS = 3.0", 1)
    parsers = {
        "oem": parse_oem_kvn,
        "opm": odm.parse_odm_kvn,
        "omm": odm.parse_odm_kvn,
        "tdm": parse_tdm_kvn,
        "cdm": parse_cdm_kvn,
    }
    if kind not in parsers:
        raise DataIOError(f"Unsupported navigation format: {format}.")
    message = replace(parsers[kind](raw), source_sha256=source_hash)
    if (
        kind == "opm"
        and not isinstance(message, odm.OpmMessage)
        or kind == "omm"
        and not isinstance(message, odm.OmmMessage)
    ):
        raise DataIOError("Navigation message kind does not match requested format.")
    return message


def read_navigation(path, *, format: str):
    return parse_navigation(read_text(path), format=format)


def _element(parent, key, value):
    units = None
    if " [" in str(value) and str(value).endswith("]"):
        value, units = str(value).rsplit(" [", 1)
        units = units[:-1]
    node = ET.SubElement(parent, key, {} if units is None else {"units": units})
    node.text = str(value)
    return node


def encode_navigation(message, *, format: str, version="3.0") -> bytes:
    kind, syntax = format.split("-", 1)
    from sim.interchange.ccsds_cdm import CdmMessage
    from sim.interchange.ccsds_tdm import TdmMessage

    expected = {"oem": OemMessage, "opm": odm.OpmMessage, "omm": odm.OmmMessage, "cdm": CdmMessage, "tdm": TdmMessage}
    if kind not in expected or not isinstance(message, expected[kind]):
        raise DataIOError(
            f"{kind.upper()} serialization requires a matching message model; a different product needs an explicit derivation."
        )
    if kind in {"oem", "opm", "omm"} and (
        version not in {"2.0", "3.0"}
        or (version == "2.0" and (message.header.message_id or message.header.classification))
    ):
        raise DataIOError("Unsupported ODM version or version 3 header fields would be lost.")
    serializers = {
        "oem": serialize_oem_kvn,
        "opm": odm.serialize_odm_kvn,
        "omm": odm.serialize_odm_kvn,
        "tdm": serialize_tdm_kvn,
        "cdm": serialize_cdm_kvn,
    }
    kvn = serializers[kind](message)
    if syntax == "kvn":
        if kind in {"oem", "opm", "omm"}:
            if version not in {"2.0", "3.0"}:
                raise DataIOError("Only ODM versions 2.0 and 3.0 are supported.")
            if version == "2.0":
                if message.header.message_id or message.header.classification:
                    raise DataIOError("ODM 2 export would lose version 3 header fields.")
                kvn = kvn.replace(f"CCSDS_{kind.upper()}_VERS = 3.0", f"CCSDS_{kind.upper()}_VERS = 2.0", 1)
        return kvn.encode()
    if syntax != "xml":
        raise DataIOError(f"Unsupported navigation output: {format}.")
    root = ET.Element(
        kind,
        {
            "id": f"CCSDS_{kind.upper()}_VERS",
            "version": version if kind in {"oem", "opm", "omm"} else "1.0" if kind == "cdm" else "2.0",
        },
    )
    header, body = ET.SubElement(root, "header"), ET.SubElement(root, "body")
    for comment in getattr(message.header, "comments", ()):
        _element(header, "COMMENT", comment)
    if kind in {"opm", "omm"}:
        for comment in message.comments:
            _element(header, "COMMENT", comment)
    if kind == "tdm":
        for comment in message.header_comments:
            _element(header, "COMMENT", comment)
    for key in ("CREATION_DATE", "ORIGINATOR", "MESSAGE_ID", "CLASSIFICATION"):
        value = getattr(message.header, key.lower(), None)
        if value:
            _element(header, key, value)
    if kind == "oem":
        for arc in message.segments:
            segment = ET.SubElement(body, "segment")
            metadata, data = ET.SubElement(segment, "metadata"), ET.SubElement(segment, "data")
            for comment in arc.metadata.comments:
                _element(metadata, "COMMENT", comment)
            for key in (
                "OBJECT_NAME",
                "OBJECT_ID",
                "CENTER_NAME",
                "REF_FRAME",
                "REF_FRAME_EPOCH",
                "TIME_SYSTEM",
                "START_TIME",
                "USEABLE_START_TIME",
                "USEABLE_STOP_TIME",
                "STOP_TIME",
                "INTERPOLATION",
                "INTERPOLATION_DEGREE",
            ):
                attr = key.lower().replace("useable", "usable")
                value = getattr(arc.metadata, attr, None)
                if value is not None:
                    _element(metadata, key, value)
            for comment in arc.comments:
                _element(data, "COMMENT", comment)
            for state in arc.states:
                node = ET.SubElement(data, "stateVector")
                for key, value in zip(STATE, (state.epoch, *state.position_km, *state.velocity_km_s)):
                    _element(node, key, value)
                if state.acceleration_km_s2 is not None:
                    for key, value in zip(("X_DDOT", "Y_DDOT", "Z_DDOT"), state.acceleration_km_s2):
                        _element(node, key, value)
            for covariance in arc.covariances:
                node = ET.SubElement(data, "covarianceMatrix")
                _element(node, "EPOCH", covariance.epoch)
                if covariance.ref_frame:
                    _element(node, "COV_REF_FRAME", covariance.ref_frame)
                for key, (i, j) in zip(COV, odm._COVARIANCE_POSITIONS):
                    _element(node, key, covariance.matrix[i][j])
    elif kind in {"opm", "omm"}:
        segment = ET.SubElement(body, "segment")
        metadata, data = ET.SubElement(segment, "metadata"), ET.SubElement(segment, "data")
        for key in ("OBJECT_NAME", "OBJECT_ID", "CENTER_NAME", "REF_FRAME", "TIME_SYSTEM", "MEAN_ELEMENT_THEORY"):
            value = getattr(message.metadata, key.lower(), None)
            if value is not None:
                _element(metadata, key, value)
        blocks = (
            [("stateVector", message.state), ("keplerianElements", message.keplerian)]
            if kind == "opm"
            else [("meanElements", message.mean_elements), ("tleParameters", message.tle_parameters)]
        )
        if kind == "opm":
            blocks.append(("spacecraftParameters", message.physical))
        if message.covariance:
            values = {key: message.covariance.matrix[i][j] for key, (i, j) in zip(COV, odm._COVARIANCE_POSITIONS)}
            if message.covariance.ref_frame:
                values = {"COV_REF_FRAME": message.covariance.ref_frame, **values}
            blocks.append(("covarianceMatrix", values))
        if kind == "opm":
            blocks.extend(("maneuverParameters", m) for m in message.maneuvers)
        for tag, values in blocks:
            if values:
                node = ET.SubElement(data, tag)
                for key, value in values.items():
                    unit = message.units.get(key)
                    _element(node, key, f"{value} [{unit}]" if unit else value)
        if message.user_defined:
            node = ET.SubElement(data, "userDefinedParameters")
            for key, value in message.user_defined.items():
                ET.SubElement(node, "USER_DEFINED", {"parameter": key.removeprefix("USER_DEFINED_")}).text = value
    elif kind == "cdm":
        from sim.interchange import ccsds_cdm as cdm

        if message.user_defined:
            raise DataIOError("Top-level CDM user fields require a separately supported XML profile.")
        # Use canonical KVN's field units/order, then rebuild the schema sections.
        relative = ET.SubElement(body, "relativeMetadataData")
        relative_state = None
        sections = {}
        for line in kvn.splitlines()[1:]:
            if line.startswith("COMMENT"):
                _element(header, "COMMENT", line.removeprefix("COMMENT").strip())
                continue
            if " = " not in line:
                continue
            key, value = line.split(" = ", 1)
            if key in cdm._HEADER_KEYS:
                if key in {"MESSAGE_FOR"}:
                    _element(header, key, value)
            elif key in cdm._RELATIVE_KEYS:
                if key.startswith(("RELATIVE_POSITION_", "RELATIVE_VELOCITY_")):
                    if relative_state is None:
                        relative_state = ET.SubElement(relative, "relativeStateVector")
                    _element(relative_state, key, value)
                else:
                    _element(relative, key, value)
            else:
                if key == "OBJECT":
                    segment = ET.SubElement(body, "segment")
                    metadata, data = ET.SubElement(segment, "metadata"), ET.SubElement(segment, "data")
                    sections = {}
                if key in cdm._OBJECT_METADATA_KEYS:
                    _element(metadata, key, value)
                    continue
                tag = (
                    "odParameters"
                    if key in cdm._OD_KEYS
                    else "additionalParameters"
                    if key in cdm._ADDITIONAL_KEYS
                    else "stateVector"
                    if key in cdm._STATE_KEYS
                    else "covarianceMatrix"
                    if key in cdm._COVARIANCE_KEYS
                    else "userDefinedParameters"
                )
                if tag not in sections:
                    sections[tag] = ET.SubElement(data, tag)
                if key.startswith("USER_DEFINED_"):
                    ET.SubElement(sections[tag], "USER_DEFINED", {"parameter": key[13:]}).text = value
                else:
                    _element(sections[tag], key, value)
    elif kind == "tdm":
        # Canonical KVN defines the authoritative bounded metadata/observation order.
        metadata = data = None
        for line in kvn.splitlines()[1:]:
            if line == "META_START":
                segment = ET.SubElement(body, "segment")
                metadata = ET.SubElement(segment, "metadata")
            elif line == "DATA_START":
                data = ET.SubElement(segment, "data")
            elif line in {"META_STOP", "DATA_STOP"} or not line.strip():
                continue
            elif " = " in line:
                key, value = line.split(" = ", 1)
                if metadata is None:
                    continue
                if data is None or key not in {"ANGLE_1", "ANGLE_2", "RANGE"}:
                    _element(metadata, key, value)
                else:
                    epoch, number = value.split()
                    node = ET.SubElement(data, "observation")
                    _element(node, "EPOCH", epoch)
                    _element(node, key, number)
    ET.indent(root)
    raw = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    parse_navigation(raw.decode(), format=format)
    return raw


def trajectory_from_oem(message: OemMessage, *, segment_index: int | None = None) -> Trajectory:
    if segment_index is None and len(message.segments) != 1:
        raise DataIOError("Select an explicit OEM segment; arcs are not silently bridged.")
    if segment_index is not None and (
        isinstance(segment_index, bool)
        or not isinstance(segment_index, int)
        or not 0 <= segment_index < len(message.segments)
    ):
        raise DataIOError("OEM segment_index must select one existing segment.")
    arc = message.segments[0 if segment_index is None else segment_index]
    if arc.metadata.ref_frame_epoch:
        raise DataIOError("Epoch-dependent reference frames require an explicit transform before trajectory import.")
    if any(state.acceleration_km_s2 is not None for state in arc.states):
        raise DataIOError(
            "PV trajectory conversion would lose supplied acceleration; preserve the OEM document instead."
        )
    epoch = parse_time(arc.states[0].epoch, arc.metadata.time_system)
    times = [parse_time(s.epoch, arc.metadata.time_system).tai_seconds - epoch.tai_seconds for s in arc.states]
    ct, cv = [], []
    for covariance in arc.covariances:
        if covariance.ref_frame and covariance.ref_frame != arc.metadata.ref_frame:
            raise DataIOError("Convert covariance frames explicitly before trajectory import.")
        ct.append(parse_time(covariance.epoch, arc.metadata.time_system).tai_seconds - epoch.tai_seconds)
        cv.append(covariance.matrix)
    return Trajectory(
        arc.metadata.object_id,
        arc.metadata.center_name,
        arc.metadata.ref_frame,
        epoch,
        times,
        [[*s.position_km, *s.velocity_km_s] for s in arc.states],
        arc.metadata.object_name,
        ct or None,
        cv or None,
        {"oem_source_sha256": message.source_sha256, "segment_index": segment_index or 0},
    )


def oem_from_trajectory(trajectory: Trajectory) -> OemMessage:
    t = trajectory
    if t.frame.startswith("SPICE:"):
        raise DataIOError(
            "OEM needs an explicit supported CCSDS frame conversion; preserve SPICE-named axes through SPK or trajectory JSON."
        )
    if t.metadata.get("discontinuity_sample_indices"):
        raise DataIOError("Select continuous trajectory segments explicitly before OEM serialization.")
    states = tuple(
        OemState(format_time(at(t.epoch, time)), tuple(pv[:3]), tuple(pv[3:])) for time, pv in zip(t.times_s, t.states)
    )
    covs = (
        ()
        if t.covariances is None
        else tuple(
            OemCovariance(epoch=format_time(at(t.epoch, time)), matrix=tuple(map(tuple, matrix)), ref_frame=t.frame)
            for time, matrix in zip(t.covariance_times_s, t.covariances)
        )
    )
    metadata = OemMetadata(
        t.object_name or t.object_id, t.object_id, t.center, t.frame, "UTC", states[0].epoch, states[-1].epoch
    )
    return OemMessage(OemHeader("3.0", creation_time(), "OEL"), (OemSegment(metadata, states, covariances=covs),))
