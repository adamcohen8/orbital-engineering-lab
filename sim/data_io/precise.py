"""User-facing SP3-c/d and Sentinel EOF adapters with explicit time/frame semantics."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

import numpy as np

from .common import read_text, xml_root
from .contracts import DataIOError, Trajectory
from .time import at, format_time, parse_time


def _epoch(fields, scale):
    if len(fields) < 6:
        raise DataIOError("Incomplete precise-orbit epoch.")
    y, m, d, h, minute = map(int, fields[:5])
    seconds = Decimal(fields[5])
    if not 0 <= seconds < 61:
        raise DataIOError("Invalid precise-orbit epoch seconds.")
    second_text = format(seconds, "f")
    whole, _, fraction = second_text.partition(".")
    fraction = fraction.rstrip("0")
    value = f"{y:04}-{m:02}-{d:02}T{h:02}:{minute:02}:{int(whole):02}" + ("." + fraction if fraction else "")
    return parse_time(value, scale)


def read_sp3(path, *, object_id: str | None = None, frame: str, derive_velocity=False) -> Trajectory:
    lines = read_text(path).splitlines()
    if not lines or lines[0][:2] not in {"#c", "#d"} or lines[0][2:3] not in {"P", "V"}:
        raise DataIOError("Expected an SP3-c/d position or PV file.")
    if not any(line.rstrip() == "EOF" for line in lines):
        raise DataIOError("Truncated SP3 file: missing EOF.")
    scale = None
    declared = []
    for line in lines:
        if line.startswith("%c"):
            candidates = [word for word in line.split() if word in {"UTC", "GPS", "TAI"}]
            if candidates:
                if scale and scale != candidates[0]:
                    raise DataIOError("Conflicting SP3 time-system declarations.")
                scale = candidates[0]
        if line.startswith("+ "):
            declared.extend(line[i : i + 3] for i in range(9, len(line), 3) if line[i : i + 3].strip() not in {"", "0"})
    if scale is None:
        raise DataIOError("SP3 requires UTC, GPS, or TAI in its header.")
    if object_id is None:
        ids = sorted({line[1:4] for line in lines if line.startswith("P")})
        if len(ids) != 1:
            raise DataIOError(f"Select an SP3 satellite explicitly: {ids}.")
        object_id = ids[0]
    epochs, positions, velocities, clocks, flags, breaks = [], [], [], [], [], []
    current = None
    epoch_seen = set()
    for line in lines:
        if line.startswith("*"):
            current = _epoch(line[1:].split(), scale)
        elif line[:1] in {"P", "V"} and line[1:4] == object_id:
            if current is None or len(line) < 60:
                raise DataIOError("SP3 record lacks epoch or required columns.")
            values = [float(line[4 + 14 * i : 18 + 14 * i]) for i in range(4)]
            if line[0] == "P":
                if current.tai_seconds in epoch_seen:
                    raise DataIOError("Duplicate SP3 object/epoch position record.")
                epoch_seen.add(current.tai_seconds)
                if any(abs(v) >= 999999 for v in values[:3]) or all(v == 0 for v in values[:3]):
                    raise DataIOError("SP3 missing/invalid position records cannot become states.")
                epochs.append(current)
                positions.append(values[:3])
                velocities.append(None)
                clocks.append({"clock_microseconds": values[3], "clock_rate_1e4_microseconds_s": None})
                flags.append(line[60:])
                if len(line) > 78 and line[78:79] == "M":
                    breaks.append(len(epochs) - 1)
            else:
                if not epochs or epochs[-1].tai_seconds != current.tai_seconds or velocities[-1] is not None:
                    raise DataIOError("SP3 velocity must follow its unique position record.")
                velocities[-1] = np.asarray(values[:3]) * 1e-4
                clocks[-1]["clock_rate_1e4_microseconds_s"] = values[3]
        elif line.startswith(("EP", "EV")):
            raise DataIOError("SP3 correlation records require a separately supported covariance profile.")
    if not epochs:
        raise DataIOError(f"No SP3 records for {object_id}.")
    times = np.asarray([e.tai_seconds - epochs[0].tai_seconds for e in epochs])
    derived = False
    if any(v is None for v in velocities):
        if not derive_velocity or len(epochs) < 3 or breaks:
            raise DataIOError(
                "SP3 lacks velocity: explicitly request finite-difference derivation on a continuous arc of at least three samples."
            )
        if any(v is not None for v in velocities):
            raise DataIOError("Mixed measured and derived SP3 velocities are not supported.")
        velocities = np.gradient(np.asarray(positions), times, axis=0, edge_order=2)
        derived = True
    return Trajectory(
        object_id,
        "EARTH",
        frame,
        epochs[0],
        times,
        np.hstack((positions, velocities)),
        metadata={
            "sp3_version": lines[0][1],
            "sp3_time_system": scale,
            "sp3_header": lines[0],
            "sp3_clock_records": clocks,
            "sp3_flags": flags,
            "discontinuity_sample_indices": breaks,
            "velocities_derived": derived,
            "derivation": "three-point finite differences" if derived else None,
        },
    )


def encode_sp3(trajectory: Trajectory, *, satellite_id: str, time_system="GPS", coordinate_system="ITRF") -> bytes:
    t = trajectory
    if t.center.upper() != "EARTH" or t.frame != "ITRF":
        raise DataIOError("SP3 output requires explicitly transformed Earth/ITRF states.")
    if len(satellite_id) != 3 or not satellite_id[0].isalpha() or not satellite_id[1:].isdigit():
        raise DataIOError("SP3 requires an explicit three-character satellite identifier, e.g. G01 or L01.")
    if t.covariances is not None:
        raise DataIOError("SP3 covariance mapping is outside this profile; use OEM/trajectory JSON.")
    if not 1 <= len(coordinate_system) <= 5:
        raise DataIOError("SP3 coordinate-system identifier must fit five columns.")
    intervals = np.diff(t.times_s)
    if intervals.size and not np.allclose(intervals, intervals[0], atol=1e-6, rtol=0):
        raise DataIOError("SP3 output requires regular samples.")
    epoch = at(t.epoch, t.times_s[0])
    stamp = format_time(epoch, time_system).removesuffix("Z")
    dt = datetime.fromisoformat(stamp)
    sec = dt.second + dt.microsecond / 1e6
    date = f"{dt.year:4} {dt.month:02} {dt.day:02} {dt.hour:02} {dt.minute:02} {sec:11.8f}"
    gps_origin = parse_time("1980-01-06T00:00:00", "GPS")
    gps_seconds = epoch.tai_seconds - gps_origin.tai_seconds
    week, sow = int(gps_seconds // 604800), gps_seconds % 604800
    # SP3 MJD and fractional day use the declared file time system.
    naive = datetime.fromisoformat(format_time(epoch, time_system).removesuffix("Z"))
    days = (naive - datetime(1858, 11, 17)).total_seconds() / 86400
    lines = [
        f"#dV{date} {len(t.times_s):7d} ORBIT {coordinate_system:5s} FIT OEL ",
        f"## {week:4d} {sow:15.8f} {(intervals[0] if intervals.size else 0):14.8f} {int(days):5d} {days % 1:15.13f}",
        "+    1   " + satellite_id + "  0" * 16,
    ]
    lines += ["+        " + "  0" * 17 for _ in range(4)]
    lines += ["++       " + "  0" * 17 for _ in range(5)]
    lines += [
        f"%c M  cc {time_system:3s} ccc cccc cccc cccc cccc ccccc ccccc ccccc ccccc",
        "%c cc cc ccc ccc cccc cccc cccc cccc ccccc ccccc ccccc ccccc",
        "%f  0.0000000  0.000000000  0.00000000000  0.000000000000000",
        "%f  0.0000000  0.000000000  0.00000000000  0.000000000000000",
        "%i    0    0    0    0      0      0      0      0         0",
        "%i    0    0    0    0      0      0      0      0         0",
    ]
    lines += ["/* OEL bounded sampled PV export; clock absent unless preserved from source."]
    lines += ["/* No orbit accuracy or clock accuracy is implied." for _ in range(3)]
    clocks = t.metadata.get("sp3_clock_records", [])
    flags = t.metadata.get("sp3_flags", [])
    for i, (time, pv) in enumerate(zip(t.times_s, t.states)):
        dt = datetime.fromisoformat(format_time(at(t.epoch, time), time_system).removesuffix("Z"))
        lines.append(
            f"*  {dt.year:4d} {dt.month:02} {dt.day:02} {dt.hour:02} {dt.minute:02} {dt.second + dt.microsecond / 1e6:11.8f}"
        )
        clock = clocks[i] if clocks else {}
        clock_rate = clock.get("clock_rate_1e4_microseconds_s")
        for key, values, offset in (
            ("P", pv[:3], clock.get("clock_microseconds", 999999.999999)),
            ("V", pv[3:] * 1e4, 999999.999999 if clock_rate is None else clock_rate),
        ):
            numbers = [*values, offset]
            formatted = [f"{v:14.6f}" for v in numbers]
            if any(len(v) != 14 for v in formatted):
                raise DataIOError("SP3 field width overflow.")
            lines.append(key + satellite_id + "".join(formatted) + (flags[i] if flags and key == "P" else ""))
    return ("\n".join([*lines, "EOF"]) + "\n").encode()


def read_sentinel_eof(path, *, frame: str, object_id: str) -> Trajectory:
    root = xml_root(read_text(path))
    nodes = root.findall(".//List_of_OSVs")
    if len(nodes) != 1:
        raise DataIOError("Sentinel EOF requires one List_of_OSVs.")
    records, epochs, auxiliary = [], [], []
    for node in nodes[0]:
        if node.tag != "OSV":
            raise DataIOError("Unsupported EOF orbit-record type.")
        fields = {}
        for child in node:
            if child.tag in fields or list(child):
                raise DataIOError("Duplicate or nonscalar EOF field.")
            fields[child.tag] = (child.text or "").strip()
            declared_unit = child.get("unit", child.get("units"))
            expected = "m/s" if child.tag in {"VX", "VY", "VZ"} else "m" if child.tag in {"X", "Y", "Z"} else None
            if expected is not None and declared_unit not in {None, expected}:
                raise DataIOError(f"EOF {child.tag} requires {expected} units.")
        required = {"UTC", "X", "Y", "Z", "VX", "VY", "VZ"}
        if not required <= fields.keys() or set(fields) - required - {"TAI", "UT1", "Absolute_Orbit", "Quality"}:
            raise DataIOError("Incomplete/unsupported EOF state fields.")
        epochs.append(parse_time(fields["UTC"].removeprefix("UTC=")))
        records.append([float(fields[k]) / 1000 for k in ("X", "Y", "Z", "VX", "VY", "VZ")])
        auxiliary.append({k: value for k, value in fields.items() if k not in {"X", "Y", "Z", "VX", "VY", "VZ"}})
    if not epochs or int(nodes[0].get("count", -1)) != len(epochs):
        raise DataIOError("EOF orbit count mismatch.")
    return Trajectory(
        object_id,
        "EARTH",
        frame,
        epochs[0],
        [e.tai_seconds - epochs[0].tai_seconds for e in epochs],
        records,
        metadata={"source_format": "sentinel-eof", "original_units": "m,m/s", "source_auxiliary_records": auxiliary},
    )
