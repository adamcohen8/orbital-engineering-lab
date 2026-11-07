"""Documented CelesTrak weather syntax and existing MSIS table mapping.

No density calculations live here. Source fields remain available for audit.
The numeric table preserves the existing consumer's adjusted centered F10.7
and previous-day observed F10.7 selection; it does not change model semantics.
"""
from __future__ import annotations

import csv
import io
import math
from datetime import date, timedelta

FIELDS = ["DATE", "BSRN", "ND", *[f"KP{i}" for i in range(1, 9)], "KP_SUM",
          *[f"AP{i}" for i in range(1, 9)], "AP_AVG", "CP", "C9", "ISN", "F10.7_OBS", "F10.7_ADJ",
          "F10.7_DATA_TYPE", "F10.7_OBS_CENTER81", "F10.7_OBS_LAST81", "F10.7_ADJ_CENTER81", "F10.7_ADJ_LAST81"]
TABLE_FIELDS = ["BSRN", "ND", *[f"KP{i}" for i in range(1, 9)], "KP_SUM", *[f"AP{i}" for i in range(1, 9)],
                "AP_AVG", "CP", "C9", "ISN", "F10.7_ADJ", "F10.7_ADJ_CENTER81", "F10.7_ADJ_LAST81",
                "F10.7_OBS", "F10.7_OBS_CENTER81", "F10.7_OBS_LAST81"]
QUALITIES = {"OBS", "INT", "PRD", "PRM", "PRF"}


def parse_celestrak_weather(text: str):
    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames != FIELDS:
        raise ValueError("Expected the documented CelesTrak space-weather CSV columns.")
    result = []
    previous = None
    for number, row in enumerate(reader, 2):
        if None in row or any(v is None for v in row.values()):
            raise ValueError(f"Invalid weather row length at {number}.")
        day = date.fromisoformat(row["DATE"])
        if previous is not None and day <= previous:
            raise ValueError("Weather dates must be unique and strictly increasing.")
        previous = day
        if row["F10.7_DATA_TYPE"] not in QUALITIES:
            raise ValueError(f"Unknown weather flux qualifier at row {number}.")
        values = dict(row)
        for field in TABLE_FIELDS:
            value = None if row[field] == "" else float(row[field])
            if value is not None and (not math.isfinite(value) or value < 0):
                raise ValueError(f"Invalid weather value at row {number}.")
            values[field] = value
        values["date"] = day
        result.append(values)
    if not result:
        raise ValueError("Weather source is empty.")
    return result


def check_msis_coverage(rows, start: date, stop: date, *, allow_predictions=False):
    """stop is the last date actually sampled, not an exclusive midnight."""
    by_date = {row["date"]: row for row in rows}
    required = {f"AP{i}" for i in range(1, 9)} | {"AP_AVG", "F10.7_OBS", "F10.7_ADJ_CENTER81"}
    day = start - timedelta(days=3)
    while day <= stop:
        row = by_date.get(day)
        if row is None or any(row[k] is None for k in required):
            raise ValueError(f"MSIS inputs missing on {day}; include three prior days and complete required indices.")
        if not allow_predictions and row["F10.7_DATA_TYPE"] not in {"OBS", "INT"}:
            raise ValueError(f"Predicted weather on {day} requires allow_predictions=true.")
        day += timedelta(days=1)


def encode_msis_table(rows) -> bytes:
    lines = ["# OEL MSIS weather table v1; source quality and unused missing values remain in the acquisition receipt/source CSV."]
    for row in rows:
        day = row["date"]
        # Unused unavailable columns remain NaN; required columns are checked by study preflight.
        values = [day.year, day.month, day.day, *[float("nan") if row[k] is None else row[k] for k in TABLE_FIELDS]]
        lines.append(" ".join(format(v, ".15g") for v in values))
    return ("\n".join(lines) + "\n").encode()
