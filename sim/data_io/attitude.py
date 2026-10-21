"""CCSDS ADM 2 quaternion profiles and STK attitude exchange.

ADM 2 imposes Q1,Q2,Q3,QC order and A-to-B direction; the removed
ATTITUDE_DIR/QUATERNION_TYPE keywords are deliberately not accepted.
"""

from __future__ import annotations

import numpy as np

from .common import read_text
from .contracts import AttitudeHistory, DataIOError
from .stk import FRAMES, stk_epoch, stk_epoch_text
from .time import at, creation_time, format_time, parse_time

ADM_KEYS = {
    "CENTER_NAME",
    "CCSDS_AEM_VERS",
    "CCSDS_APM_VERS",
    "CREATION_DATE",
    "ORIGINATOR",
    "MESSAGE_ID",
    "OBJECT_NAME",
    "OBJECT_ID",
    "REF_FRAME_A",
    "REF_FRAME_B",
    "TIME_SYSTEM",
    "START_TIME",
    "STOP_TIME",
    "USEABLE_START_TIME",
    "USEABLE_STOP_TIME",
    "ATTITUDE_TYPE",
    "ANGVEL_FRAME",
    "INTERPOLATION_METHOD",
    "INTERPOLATION_DEGREE",
    "EPOCH",
    "Q1",
    "Q2",
    "Q3",
    "QC",
    "ANGVEL_X",
    "ANGVEL_Y",
    "ANGVEL_Z",
}


def _flatten_apm(raw):
    lines, frames, section, seen = [], {}, None, set()
    for line in raw.splitlines():
        value = line.strip()
        if value in {"QUAT_START", "ANGVEL_START"}:
            if section is not None or value in seen or (value == "ANGVEL_START" and "QUAT_STOP" not in seen):
                raise DataIOError("Invalid or repeated APM block.")
            section = value[:-6]
            seen.add(value)
            continue
        if value in {"QUAT_STOP", "ANGVEL_STOP"}:
            if section != value[:-5]:
                raise DataIOError("Unbalanced APM block.")
            section = None
            seen.add(value)
            continue
        if value and not value.startswith("COMMENT") and "=" in value:
            key, item = (part.strip() for part in value.split("=", 1))
            if key in {"REF_FRAME_A", "REF_FRAME_B"}:
                if section not in {"QUAT", "ANGVEL"}:
                    raise DataIOError("APM endpoint frames belong inside an attitude data block.")
                if section == "ANGVEL":
                    if frames.get(key) != item:
                        raise DataIOError("APM quaternion and angular-rate endpoint frames disagree.")
                    continue
                frames[key] = item
        lines.append(line)
    if section is not None or not {"QUAT_START", "QUAT_STOP"} <= seen:
        raise DataIOError("APM 2.0 requires one complete QUAT block.")
    return "\n".join(lines)


def parse_adm(raw: str, *, kind="aem") -> AttitudeHistory:
    if kind == "apm":
        raw = _flatten_apm(raw)
    fields, rows, comments = {}, [], []
    section, seen = "header", set()
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("COMMENT"):
            comments.append(line.removeprefix("COMMENT").strip())
            continue
        if line in {"META_START", "META_STOP", "DATA_START", "DATA_STOP"}:
            if kind != "aem" or line in seen:
                raise DataIOError("Only one complete ADM ephemeris segment is supported.")
            expected = {
                "header": "META_START",
                "metadata": "META_STOP",
                "after_meta": "DATA_START",
                "data": "DATA_STOP",
            }.get(section)
            if line != expected:
                raise DataIOError("Malformed ADM section ordering.")
            seen.add(line)
            section = {
                "META_START": "metadata",
                "META_STOP": "after_meta",
                "DATA_START": "data",
                "DATA_STOP": "complete",
            }[line]
        elif section == "data":
            parts = line.split()
            if len(parts) not in {5, 8}:
                raise DataIOError("AEM quaternion rows require epoch plus four components and optional three rates.")
            rows.append(parts)
        else:
            if "=" not in line:
                raise DataIOError("Malformed ADM keyword.")
            key, value = [part.strip() for part in line.split("=", 1)]
            if key not in ADM_KEYS or key in fields or section == "complete":
                raise DataIOError(f"Unsupported/duplicate ADM field {key}.")
            fields[key] = value
    required = {
        f"CCSDS_{kind.upper()}_VERS",
        "CREATION_DATE",
        "ORIGINATOR",
        "OBJECT_NAME",
        "OBJECT_ID",
        "REF_FRAME_A",
        "REF_FRAME_B",
        "TIME_SYSTEM",
    }
    if not required <= fields.keys() or fields[f"CCSDS_{kind.upper()}_VERS"] != "2.0":
        raise DataIOError("Incomplete ADM header/metadata; only version 2.0 is supported.")
    if not fields["REF_FRAME_B"].startswith("SC_BODY_") or fields["REF_FRAME_A"].startswith("SC_BODY_"):
        raise DataIOError("The bounded ADM profile requires inertial/reference frame A and spacecraft body frame B.")
    if fields["REF_FRAME_A"] in {"ECI", "ECEF", "J2000"}:
        raise DataIOError("Use explicit EME2000/GCRF/TEME/ITRF frame identifiers.")
    if kind == "aem":
        if section != "complete" or not rows or not {"START_TIME", "STOP_TIME", "ATTITUDE_TYPE"} <= fields.keys():
            raise DataIOError("AEM requires complete metadata and a nonempty data section.")
        attitude_type = fields["ATTITUDE_TYPE"]
        if attitude_type not in {"QUATERNION", "QUATERNION/ANGVEL"}:
            raise DataIOError("Only quaternion and quaternion/angular-velocity AEM data are supported.")
        rate = attitude_type == "QUATERNION/ANGVEL"
        if any(len(row) != (8 if rate else 5) for row in rows):
            raise DataIOError("AEM row shape disagrees with ATTITUDE_TYPE.")
    else:
        if not {"EPOCH", "Q1", "Q2", "Q3", "QC"} <= fields.keys():
            raise DataIOError("APM requires one complete quaternion state.")
        rate = any(key in fields for key in ("ANGVEL_X", "ANGVEL_Y", "ANGVEL_Z"))
        keys = ["EPOCH", "Q1", "Q2", "Q3", "QC"]
        if rate:
            keys += ["ANGVEL_X", "ANGVEL_Y", "ANGVEL_Z"]
        if not set(keys) <= fields.keys():
            raise DataIOError("APM angular velocity must have all three components.")
        values = []
        for key in keys:
            value = fields[key]
            if " [" in value:
                number, unit = value.rsplit(" [", 1)
                if key.startswith("ANGVEL") and unit == "deg/s]":
                    value = number
                else:
                    raise DataIOError("Unsupported APM units.")
            values.append(value)
        rows = [values]
    if rate and fields.get("ANGVEL_FRAME") != (fields["REF_FRAME_B"] if kind == "apm" else "REF_FRAME_B"):
        raise DataIOError("Angular velocity must explicitly be expressed in REF_FRAME_B.")
    epoch = parse_time(rows[0][0], fields["TIME_SYSTEM"])
    times = [parse_time(row[0], fields["TIME_SYSTEM"]).tai_seconds - epoch.tai_seconds for row in rows]
    if kind == "aem":
        if abs(parse_time(fields["START_TIME"], fields["TIME_SYSTEM"]).tai_seconds - epoch.tai_seconds) > 1e-6:
            raise DataIOError("AEM START_TIME differs from first sample.")
        if (
            abs(parse_time(fields["STOP_TIME"], fields["TIME_SYSTEM"]).tai_seconds - epoch.tai_seconds - times[-1])
            > 1e-6
        ):
            raise DataIOError("AEM STOP_TIME differs from final sample.")
    quaternions = [[float(row[4]), *map(float, row[1:4])] for row in rows]
    rates = np.deg2rad([[float(value) for value in row[5:8]] for row in rows]) if rate else None
    return AttitudeHistory(
        fields["OBJECT_ID"],
        fields["REF_FRAME_A"],
        fields["REF_FRAME_B"],
        epoch,
        times,
        quaternions,
        rates,
        {"adm_metadata": fields, "comments": comments},
    )


def encode_adm(history: AttitudeHistory, *, kind="aem") -> bytes:
    h = history
    if kind == "apm" and len(h.times_s) != 1:
        raise DataIOError("APM requires selecting exactly one attitude sample.")
    if kind not in {"aem", "apm"}:
        raise DataIOError("Expected AEM or APM.")
    fields = {f"CCSDS_{kind.upper()}_VERS": "2.0", "CREATION_DATE": creation_time(), "ORIGINATOR": "OEL"}
    metadata = {
        "OBJECT_NAME": h.object_id,
        "OBJECT_ID": h.object_id,
        "REF_FRAME_A": h.reference_frame,
        "REF_FRAME_B": h.body_frame,
        "TIME_SYSTEM": "UTC",
    }
    lines = [f"{key} = {value}" for key, value in fields.items()]
    lines += ["COMMENT " + str(c) for c in h.metadata.get("comments", [])]
    if kind == "aem":
        metadata.update(
            START_TIME=format_time(at(h.epoch, h.times_s[0])),
            STOP_TIME=format_time(at(h.epoch, h.times_s[-1])),
            ATTITUDE_TYPE="QUATERNION/ANGVEL" if h.rates_rad_s is not None else "QUATERNION",
        )
        if h.rates_rad_s is not None:
            metadata["ANGVEL_FRAME"] = "REF_FRAME_B"
        lines += ["META_START", *[f"{key} = {value}" for key, value in metadata.items()], "META_STOP", "DATA_START"]
        for i, (time, q) in enumerate(zip(h.times_s, h.quaternions)):
            values = [*q[1:], q[0]]
            if h.rates_rad_s is not None:
                values.extend(np.rad2deg(h.rates_rad_s[i]))
            lines.append(format_time(at(h.epoch, time)) + " " + " ".join(format(v, ".17g") for v in values))
        lines.append("DATA_STOP")
    else:
        lines += [f"{key} = {metadata[key]}" for key in ("OBJECT_NAME", "OBJECT_ID", "TIME_SYSTEM")]
        lines.append("EPOCH = " + format_time(at(h.epoch, h.times_s[0])))
        lines += ["QUAT_START", "REF_FRAME_A = " + h.reference_frame, "REF_FRAME_B = " + h.body_frame]
        q = h.quaternions[0]
        lines += [f"{key} = {value:.17g}" for key, value in zip(("Q1", "Q2", "Q3", "QC"), (*q[1:], q[0]))]
        lines.append("QUAT_STOP")
        if h.rates_rad_s is not None:
            lines += [
                "ANGVEL_START",
                "REF_FRAME_A = " + h.reference_frame,
                "REF_FRAME_B = " + h.body_frame,
                "ANGVEL_FRAME = " + h.body_frame,
            ]
            lines += [
                f"{key} = {value:.17g} [deg/s]"
                for key, value in zip(("ANGVEL_X", "ANGVEL_Y", "ANGVEL_Z"), np.rad2deg(h.rates_rad_s[0]))
            ]
            lines.append("ANGVEL_STOP")
    raw = "\n".join(lines) + "\n"
    parse_adm(raw, kind=kind)
    return raw.encode()


def read_stk_attitude(path, *, object_id: str, body_frame="SC_BODY_1") -> AttitudeHistory:
    lines = [
        line.strip() for line in read_text(path).splitlines() if line.strip() and not line.lstrip().startswith("#")
    ]
    if (
        len(lines) < 4
        or not lines[0].startswith("stk.v.")
        or lines[1] != "BEGIN Attitude"
        or lines[-1] != "END Attitude"
    ):
        raise DataIOError("Expected one complete STK attitude section.")
    fields, rows, in_data = {}, [], False
    rate = False
    for line in lines[2:-1]:
        if line in {"AttitudeTimeQuaternions", "AttitudeTimeQuatAngVels"}:
            if in_data:
                raise DataIOError("Duplicate STK attitude section.")
            in_data, rate = True, line.endswith("AngVels")
        elif in_data:
            row = [float(v) for v in line.split()]
            if len(row) != (8 if rate else 5):
                raise DataIOError("Incorrect STK quaternion/rate row shape.")
            rows.append(row)
        else:
            key, value = line.split(maxsplit=1)
            if key in fields:
                raise DataIOError("Duplicate STK attitude metadata.")
            fields[key] = value
    allowed = {
        "NumberOfAttitudePoints",
        "ScenarioEpoch",
        "CentralBody",
        "CoordinateAxes",
        "InterpolationOrder",
        "BlockingFactor",
    }
    if set(fields) - allowed or not rows or int(fields.get("NumberOfAttitudePoints", -1)) != len(rows):
        raise DataIOError("Unsupported STK attitude metadata or count mismatch.")
    if fields.get("CentralBody") != "Earth" or fields.get("CoordinateAxes") not in FRAMES:
        raise DataIOError("STK attitude requires explicit Earth and a supported reference frame.")
    values = np.asarray(rows)
    return AttitudeHistory(
        object_id,
        FRAMES[fields["CoordinateAxes"]],
        body_frame,
        stk_epoch(fields["ScenarioEpoch"]),
        values[:, 0],
        values[:, [4, 1, 2, 3]],
        np.deg2rad(values[:, 5:8]) if rate else None,
        {"stk_metadata": fields},
    )


def encode_stk_attitude(history: AttitudeHistory) -> bytes:
    h = history
    frames = {v: k for k, v in FRAMES.items()}
    if h.reference_frame not in frames:
        raise DataIOError("Unsupported STK attitude reference frame.")
    lines = [
        "stk.v.12.0",
        "BEGIN Attitude",
        f"NumberOfAttitudePoints {len(h.times_s)}",
        "ScenarioEpoch " + stk_epoch_text(h.epoch),
        "CentralBody Earth",
        "CoordinateAxes " + frames[h.reference_frame],
        "AttitudeTimeQuatAngVels" if h.rates_rad_s is not None else "AttitudeTimeQuaternions",
    ]
    for i, (time, q) in enumerate(zip(h.times_s, h.quaternions)):
        values = [time, *q[1:], q[0]]
        if h.rates_rad_s is not None:
            values.extend(np.rad2deg(h.rates_rad_s[i]))
        lines.append(" ".join(format(v, ".17g") for v in values))
    return ("\n".join([*lines, "END Attitude"]) + "\n").encode()
