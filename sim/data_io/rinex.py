"""RINEX 3 observation and GPS/Galileo navigation data exchange.

Native observables, loss-of-lock/signal indicators, clock terms, and all header
records are retained. Parsing does not solve a receiver orbit or reduce raw
GNSS observables into instantaneous geometric range.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_DOWN, Decimal

import numpy as np

from .common import read_text
from .contracts import MAX_SAMPLES, DataIOError
from .time import parse_time


@dataclass(frozen=True)
class RinexDocument:
    kind: str
    version: str
    header: tuple[str, ...]
    observation_types: dict[str, tuple[str, ...]]
    records: tuple[dict, ...]


def _number(value):
    value = value.strip().replace("D", "E").replace("d", "e")
    result = None if not value else float(value)
    if result is not None and not np.isfinite(result):
        raise DataIOError("RINEX values must be finite or explicitly blank.")
    return result


def parse_rinex(raw: str) -> RinexDocument:
    lines = raw.splitlines()
    if not lines or len(lines[0]) < 61 or lines[0][60:].strip() != "RINEX VERSION / TYPE":
        raise DataIOError("Missing fixed-column RINEX version/type header.")
    version, file_type = lines[0][:9].strip(), lines[0][20:21]
    if version not in {"3.02", "3.03", "3.04", "3.05"} or file_type not in {"O", "N"}:
        raise DataIOError("Supported RINEX profiles are 3.02..3.05 observation or navigation.")
    end = next((i for i, line in enumerate(lines) if line[60:].strip() == "END OF HEADER"), None)
    if end is None or end > 10_000:
        raise DataIOError("Missing or oversized RINEX header.")
    header = tuple(lines[: end + 1])
    types, counts, time_scale = {}, {}, None
    for line in header:
        label = line[60:].strip()
        if label == "SYS / # / OBS TYPES":
            system = line[0]
            if line[3:6].strip():
                if system in counts:
                    raise DataIOError("Duplicate RINEX observation-type declaration.")
                counts[system] = int(line[3:6])
                types[system] = []
            if system not in types:
                raise DataIOError("Observation-type continuation without declaration.")
            types[system].extend(line[i : i + 3] for i in range(7, 59, 4) if line[i : i + 3].strip())
        if label == "TIME OF FIRST OBS" and line[48:51].strip():
            time_scale = line[48:51].strip()
    if any(len(types[s]) != n or len(set(types[s])) != n for s, n in counts.items()):
        raise DataIOError("RINEX observation type counts disagree.")
    if file_type == "O" and not types:
        raise DataIOError("Observation types must be declared.")
    if file_type == "O" and time_scale is None:
        if lines[0][40:41] != "G":
            raise DataIOError(
                "Non-GPS/mixed RINEX observation files require an explicit TIME OF FIRST OBS time system."
            )
        time_scale = "GPS"
    records, i, previous_epoch = [], end + 1, None
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        if len(records) >= MAX_SAMPLES:
            raise DataIOError("RINEX record limit exceeded.")
        line = lines[i]
        if file_type == "O":
            parts = line.split()
            if not line.startswith(">") or not 9 <= len(parts) <= 10:
                raise DataIOError("Expected a RINEX 3 observation epoch.")
            y, m, d, h, minute = map(int, parts[1:6])
            seconds, flag, count = float(parts[6]), int(parts[7]), int(parts[8])
            if flag not in {0, 1} or not 1 <= count <= 999:
                raise DataIOError("RINEX special-event records are outside the observation profile.")
            epoch_text = f"{y:04}-{m:02}-{d:02}T{h:02}:{minute:02}:{seconds:010.7f}"
            # Epoch syntax and scale are validated; source scale stays native.
            scale = {"GAL": "GPS"}.get(time_scale, time_scale)
            # Preserve native sub-microsecond RINEX tags. Canonical validation
            # checks the calendar/scale at the enclosing microsecond; ordering
            # uses the original decimal timetag, not a rounded Epoch float.
            native_seconds = Decimal(parts[6])
            if (
                not native_seconds.is_finite()
                or not 0 <= native_seconds < 61
                or native_seconds != native_seconds.quantize(Decimal("0.0000001"))
            ):
                raise DataIOError(
                    "RINEX observation timetags require finite seconds with at most seven fractional decimal places."
                )
            enclosing = native_seconds.quantize(Decimal("0.000001"), rounding=ROUND_DOWN)
            parse_time(f"{y:04}-{m:02}-{d:02}T{h:02}:{minute:02}:{enclosing:09.6f}", scale)
            native_identity = (y, m, d, h, minute, native_seconds)
            if previous_epoch is not None and native_identity <= previous_epoch:
                raise DataIOError("RINEX observation epochs must increase strictly.")
            previous_epoch = native_identity
            satellites = []
            i += 1
            for _ in range(count):
                if i >= len(lines) or len(lines[i]) < 3:
                    raise DataIOError("Truncated RINEX satellite observations.")
                sat = lines[i][:3]
                if sat[0] not in types or not sat[1:].isdigit():
                    raise DataIOError("Undeclared RINEX satellite/system.")
                n = len(types[sat[0]])
                data = lines[i][3:]
                i += 1
                while len(data) < n * 16 and i < len(lines) and lines[i].startswith("   "):
                    data += lines[i][3:]
                    i += 1
                # Missing tail values are blank observations, not zeros.
                if len(data) > n * 16 and data[n * 16 :].strip():
                    raise DataIOError("Unexpected RINEX satellite observation fields.")
                data = data.ljust(n * 16)
                observations = {}
                for index, code in enumerate(types[sat[0]]):
                    field = data[index * 16 : (index + 1) * 16]
                    if field[14:15] not in " 01234567" or field[15:16] not in " 0123456789":
                        raise DataIOError("Invalid RINEX LLI/SSI digit.")
                    observations[code] = {"value": _number(field[:14]), "lli": field[14], "ssi": field[15]}
                if any(item["satellite"] == sat for item in satellites):
                    raise DataIOError("Duplicate satellite in RINEX epoch.")
                satellites.append({"satellite": sat, "observations": observations})
            records.append(
                {
                    "epoch": epoch_text,
                    "time_system": time_scale,
                    "flag": flag,
                    "receiver_clock_offset_s": _number(parts[9]) if len(parts) > 9 else None,
                    "satellites": satellites,
                }
            )
        else:
            if line[0:1] not in {"G", "E"} or len(line) < 23 or i + 7 >= len(lines):
                raise DataIOError("Navigation profile accepts complete eight-line GPS/Galileo ephemeris records.")
            sat = line[:3]
            parts = line[3:23].split()
            if len(parts) != 6 or not sat[1:].isdigit():
                raise DataIOError("Invalid RINEX navigation epoch/identifier.")
            y, m, d, h, minute, seconds = map(int, parts)
            epoch_text = f"{y:04}-{m:02}-{d:02}T{h:02}:{minute:02}:{seconds:02}"
            parse_time(epoch_text, "GPS")
            clock = [_number(line[23 + 19 * j : 42 + 19 * j]) for j in range(3)]
            orbit = []
            for row in lines[i + 1 : i + 8]:
                if row[:4].strip():
                    raise DataIOError("Malformed RINEX navigation continuation.")
                orbit.extend(_number(row[4 + 19 * j : 23 + 19 * j]) for j in range(4))
            if any(value is None for value in clock):
                raise DataIOError("Navigation clock coefficients must be present.")
            records.append({"satellite": sat, "epoch": epoch_text, "clock_coefficients": clock, "orbit_fields": orbit})
            i += 8
    if not records:
        raise DataIOError("RINEX contains no supported records.")
    return RinexDocument(
        "observation" if file_type == "O" else "navigation",
        version,
        header,
        {k: tuple(v) for k, v in types.items()},
        tuple(records),
    )


def read_rinex(path):
    return parse_rinex(read_text(path))


def encode_rinex(document: RinexDocument) -> bytes:
    lines = list(document.header)
    for record in document.records:
        dt, time = record["epoch"].split("T")
        y, m, d = map(int, dt.split("-"))
        h, minute, seconds = time.split(":")
        if document.kind == "observation":
            clock = record["receiver_clock_offset_s"]
            row = f"> {y:4d} {m:02} {d:02} {int(h):02} {int(minute):02} {float(seconds):10.7f}  {record['flag']:1d}{len(record['satellites']):3d}"
            if clock is not None:
                row += f"      {clock:15.12f}"
            lines.append(row)
            for satellite in record["satellites"]:
                values = []
                for code in document.observation_types[satellite["satellite"][0]]:
                    obs = satellite["observations"][code]
                    value = " " * 14 if obs["value"] is None else f"{obs['value']:14.3f}"
                    if len(value) != 14:
                        raise DataIOError("RINEX observation field width overflow.")
                    values.append(value + obs["lli"] + obs["ssi"])
                lines.append(satellite["satellite"] + "".join(values))
        elif document.kind == "navigation":

            def field(value):
                encoded = " " * 19 if value is None else f"{value:19.12E}".replace("E", "D")
                if len(encoded) != 19:
                    raise DataIOError("RINEX navigation field width overflow.")
                return encoded

            lines.append(
                f"{record['satellite']} {y:4d} {m:02} {d:02} {int(h):02} {int(minute):02} {int(float(seconds)):02}"
                + "".join(field(v) for v in record["clock_coefficients"])
            )
            for start in range(0, 28, 4):
                lines.append("    " + "".join(field(v) for v in record["orbit_fields"][start : start + 4]))
        else:
            raise DataIOError("Unknown RINEX document type.")
    raw = "\n".join(lines) + "\n"
    recovered = parse_rinex(raw)
    if recovered.records != document.records:
        raise DataIOError(
            "RINEX serialization would change native values or timetags beyond the supported fixed-column precision."
        )
    return raw.encode()
