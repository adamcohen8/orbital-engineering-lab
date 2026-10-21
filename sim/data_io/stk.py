"""STK time-position-velocity .e interchange, explicitly bounded and offline."""

from __future__ import annotations

from datetime import datetime

import numpy as np

from .common import read_text
from .contracts import DataIOError, Trajectory
from .time import format_time, parse_time

FRAMES = {"J2000": "EME2000", "ICRF": "GCRF", "TEMEOfDate": "TEME", "Fixed": "ITRF"}


def stk_epoch(value: str):
    try:
        dt = datetime.strptime(value, "%d %b %Y %H:%M:%S.%f")
    except ValueError:
        try:
            dt = datetime.strptime(value, "%d %b %Y %H:%M:%S")
        except ValueError as exc:
            raise DataIOError("STK ScenarioEpoch requires dd Mon yyyy HH:mm:ss[.ffffff] UTC.") from exc
    return parse_time(dt.isoformat())


def stk_epoch_text(epoch):
    return datetime.fromisoformat(format_time(epoch).removesuffix("Z")).strftime("%d %b %Y %H:%M:%S.%f")


def read_stk(path, *, object_id: str) -> Trajectory:
    lines = [
        line.strip() for line in read_text(path).splitlines() if line.strip() and not line.lstrip().startswith("#")
    ]
    if not lines or not lines[0].startswith("stk.v."):
        raise DataIOError("Missing STK file version.")
    if len(lines) < 4 or lines[1].lower() != "begin ephemeris" or lines[-1].lower() != "end ephemeris":
        raise DataIOError("Expected one complete STK ephemeris section.")
    metadata, rows = {}, []
    data_started = False
    for line in lines[2:-1]:
        if line == "EphemerisTimePosVel":
            if data_started:
                raise DataIOError("Duplicate STK data section.")
            data_started = True
        elif data_started:
            row = line.split()
            if len(row) != 7:
                raise DataIOError("STK EphemerisTimePosVel requires seven fields.")
            rows.append([float(value) for value in row])
        else:
            fields = line.split(maxsplit=1)
            if len(fields) != 2 or fields[0] in metadata:
                raise DataIOError("Malformed or duplicate STK metadata.")
            metadata[fields[0]] = fields[1]
    allowed = {
        "NumberOfEphemerisPoints",
        "ScenarioEpoch",
        "CentralBody",
        "CoordinateSystem",
        "DistanceUnit",
        "InterpolationMethod",
        "InterpolationOrder",
    }
    if set(metadata) - allowed or not data_started:
        raise DataIOError(f"Unsupported STK fields or data type: {sorted(set(metadata) - allowed)}.")
    required = {"NumberOfEphemerisPoints", "ScenarioEpoch", "CentralBody", "CoordinateSystem", "DistanceUnit"}
    if not required <= metadata.keys() or not rows or int(metadata["NumberOfEphemerisPoints"]) != len(rows):
        raise DataIOError("Incomplete STK metadata or point count mismatch.")
    frame = FRAMES.get(metadata["CoordinateSystem"])
    if frame is None:
        raise DataIOError("Unsupported STK coordinate system; no frame inference is performed.")
    scale = {"Kilometers": 1.0, "Meters": 0.001}.get(metadata["DistanceUnit"])
    if scale is None:
        raise DataIOError("STK DistanceUnit must be Meters or Kilometers.")
    values = np.asarray(rows)
    return Trajectory(
        object_id,
        metadata["CentralBody"].upper(),
        frame,
        stk_epoch(metadata["ScenarioEpoch"]),
        values[:, 0],
        values[:, 1:] * scale,
        metadata={"stk_metadata": metadata, "source_version": lines[0]},
    )


def encode_stk(trajectory: Trajectory) -> bytes:
    t = trajectory
    frames = {v: k for k, v in FRAMES.items()}
    if t.frame not in frames:
        raise DataIOError("STK output requires a supported named coordinate system.")
    if t.covariances is not None:
        raise DataIOError("The bounded STK .e profile cannot carry covariance; export OEM or trajectory JSON.")
    lines = [
        "stk.v.12.0",
        "BEGIN Ephemeris",
        f"NumberOfEphemerisPoints {len(t.times_s)}",
        "ScenarioEpoch " + stk_epoch_text(t.epoch),
        "InterpolationMethod Hermite",
        "InterpolationOrder 3",
        "CentralBody " + t.center.title(),
        "CoordinateSystem " + frames[t.frame],
        "DistanceUnit Kilometers",
        "EphemerisTimePosVel",
    ]
    lines += [
        " ".join(format(float(value), ".17g") for value in (time, *state)) for time, state in zip(t.times_s, t.states)
    ]
    return ("\n".join([*lines, "END Ephemeris"]) + "\n").encode()
