"""Documented JSON/CSV/Parquet contracts and reproducible column mappings."""

from __future__ import annotations

import csv
import io
from pathlib import Path

import numpy as np

from .common import digest, json_bytes, read_bytes, read_text, strict_json
from .contracts import MAX_SAMPLES, DataIOError, Trajectory
from .time import format_time, parse_time

SCHEMA = "oel.trajectory.v1"
COLUMNS = ("time_s", "x_km", "y_km", "z_km", "vx_km_s", "vy_km_s", "vz_km_s")


def trajectory_document(t: Trajectory) -> dict:
    return {
        "schema": SCHEMA,
        "object_id": t.object_id,
        "object_name": t.object_name,
        "center": t.center,
        "frame": t.frame,
        "epoch": format_time(t.epoch),
        "time_system": "UTC",
        "time_basis": "elapsed_TAI_SI_seconds",
        "units": ["km"] * 3 + ["km/s"] * 3,
        "times_s": t.times_s.tolist(),
        "states": t.states.tolist(),
        "covariance_times_s": None if t.covariances is None else t.covariance_times_s.tolist(),
        "covariances": None if t.covariances is None else t.covariances.tolist(),
        "metadata": t.metadata,
    }


def trajectory_from_document(value: dict) -> Trajectory:
    allowed = {
        "schema",
        "object_id",
        "object_name",
        "center",
        "frame",
        "epoch",
        "time_system",
        "time_basis",
        "units",
        "times_s",
        "states",
        "covariance_times_s",
        "covariances",
        "metadata",
    }
    if not isinstance(value, dict) or value.get("schema") != SCHEMA or set(value) - allowed:
        raise DataIOError("Invalid/unknown trajectory JSON contract.")
    if value.get("units") != ["km"] * 3 + ["km/s"] * 3 or value.get("time_basis") != "elapsed_TAI_SI_seconds":
        raise DataIOError("Trajectory JSON requires declared km/km/s and elapsed TAI SI seconds.")
    return Trajectory(
        value["object_id"],
        value["center"],
        value["frame"],
        parse_time(value["epoch"], value["time_system"]),
        value["times_s"],
        value["states"],
        value.get("object_name", ""),
        value.get("covariance_times_s"),
        value.get("covariances"),
        value.get("metadata", {}),
    )


def encode_trajectory_table(t: Trajectory, *, format: str) -> tuple[bytes, dict]:
    document = trajectory_document(t)
    context = {key: value for key, value in document.items() if key not in {"times_s", "states"}}
    if format == "trajectory-json":
        return json_bytes(document), context
    values = np.column_stack((t.times_s, t.states))
    if format == "trajectory-csv":
        stream = io.StringIO(newline="")
        writer = csv.writer(stream)
        writer.writerow(COLUMNS)
        writer.writerows([[format_float(value) for value in row] for row in values])
        return stream.getvalue().encode(), context
    if format == "trajectory-parquet":
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise DataIOError("Parquet requires the optional 'pyarrow' IO dependency.") from exc
        table = pa.table({name: values[:, i] for i, name in enumerate(COLUMNS)})
        table = table.replace_schema_metadata({b"oel.trajectory": json_bytes(context)})
        sink = pa.BufferOutputStream()
        pq.write_table(table, sink)
        return sink.getvalue().to_pybytes(), context
    raise DataIOError(f"Unsupported table format: {format}.")


def format_float(value):
    return format(float(value), ".17g")


def read_trajectory_table(
    path, *, format: str, context: dict | None = None, column_mapping: dict | None = None
) -> Trajectory:
    if format == "trajectory-json":
        return trajectory_from_document(strict_json(read_text(path)))
    if format == "trajectory-csv":
        if context is None:
            receipt_path = Path(str(path) + ".receipt.json")
            if not receipt_path.is_file():
                raise DataIOError("CSV requires explicit frame/epoch/units context or its hash-bound OEL receipt.")
            receipt = strict_json(read_text(receipt_path))
            if receipt["output"]["sha256"] != digest(read_bytes(path)):
                raise DataIOError("CSV receipt does not match input content.")
            context = receipt["trajectory_context"]
        reader = csv.DictReader(io.StringIO(read_text(path)))
        if not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise DataIOError("CSV requires unique headers.")
        mapping = column_mapping or {key: key for key in COLUMNS}
        if set(mapping) != set(COLUMNS) or len(set(mapping.values())) != len(COLUMNS):
            raise DataIOError("Column mapping requires one unique source column for every time/PV component.")
        if set(reader.fieldnames) != set(mapping.values()):
            raise DataIOError("Unmapped or missing CSV columns; supply a precise supported mapping.")
        rows = []
        for row in reader:
            if len(rows) >= MAX_SAMPLES or None in row or any(value is None for value in row.values()):
                raise DataIOError("CSV exceeds row limits or has inconsistent columns.")
            rows.append([float(row[mapping[key]]) for key in COLUMNS])
    elif format == "trajectory-parquet":
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise DataIOError("Parquet requires the optional 'pyarrow' IO dependency.") from exc
        file = pq.ParquetFile(pa.BufferReader(read_bytes(path)))
        if file.metadata.num_rows > MAX_SAMPLES or file.metadata.num_columns != len(COLUMNS):
            raise DataIOError("Parquet table exceeds row/column limits.")
        if any(not pa.types.is_floating(field.type) for field in file.schema_arrow):
            raise DataIOError(
                "Parquet trajectory columns require floating-point types; variable-size or boolean columns are outside this profile."
            )
        table = file.read()
        if table.column_names != list(COLUMNS):
            raise DataIOError("Unexpected Parquet trajectory columns.")
        metadata = table.schema.metadata or {}
        if b"oel.trajectory" not in metadata:
            raise DataIOError("Parquet requires the OEL trajectory metadata contract.")
        context = strict_json(metadata[b"oel.trajectory"].decode())
        rows = np.column_stack([table.column(name).to_numpy() for name in COLUMNS])
    else:
        raise DataIOError(f"Unsupported table format: {format}.")
    data = np.asarray(rows, dtype=float)
    if data.ndim != 2 or data.shape[1] != 7:
        raise DataIOError("Empty or malformed trajectory table.")
    return trajectory_from_document({**context, "times_s": data[:, 0].tolist(), "states": data[:, 1:].tolist()})
