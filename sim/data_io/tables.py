"""Export any bounded review SELECT to CSV, JSON, or Parquet with its query."""

from __future__ import annotations

import csv
import io

from sim.review import ReviewWorkspace

from .common import json_bytes, publish
from .contracts import DataIOError


def export_review_table(completed_run, output, *, query: str, format="json", max_rows=200_000):
    if isinstance(max_rows, bool) or not isinstance(max_rows, int) or not 1 <= max_rows <= 200_000:
        raise DataIOError("Table export max_rows must be an integer in [1,200000].")
    with ReviewWorkspace.open(completed_run) as workspace:
        identity = workspace.evidence_identity()
        result = workspace.query(query, max_rows=max_rows, max_vm_steps=max(250_000, max_rows * 300))
        if result.truncated:
            raise DataIOError("Table selection was truncated; narrow the query or increase the explicit row limit.")
        if workspace.evidence_identity()["sha256"] != identity["sha256"]:
            raise DataIOError("Review evidence changed during table extraction.")
        rows = list(result.rows)
        columns = list(result.columns)
        schemas = workspace.table_columns()
    receipt = {
        "operation": "export-review-table",
        "format": format,
        "source_review_store": identity,
        "query": query,
        "columns": columns,
        "row_count": len(rows),
        "source_table_schemas": schemas,
    }
    if len(set(columns)) != len(columns):
        raise DataIOError("Use unique SQL aliases for exported columns.")
    if format == "json":
        raw = json_bytes({"schema": "oel.review-table.v1", "columns": columns, "rows": rows})
    elif format == "csv":
        stream = io.StringIO(newline="")
        writer = csv.writer(stream)
        writer.writerow(columns)
        for row in rows:
            writer.writerow(
                [
                    "\\N"
                    if row[col] is None
                    else "\\" + row[col]
                    if isinstance(row[col], str) and row[col].startswith("\\")
                    else row[col]
                    for col in columns
                ]
            )
        raw = stream.getvalue().encode()
        receipt["csv_null_encoding"] = "\\N is NULL; leading backslash in text is doubled"
    elif format == "parquet":
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise DataIOError("Parquet requires the optional 'pyarrow' IO dependency.") from exc
        table = pa.Table.from_pylist(rows) if rows else pa.table({key: [] for key in columns})
        table = table.replace_schema_metadata({b"oel.review-table": json_bytes(receipt)})
        sink = pa.BufferOutputStream()
        pq.write_table(table, sink)
        raw = sink.getvalue().to_pybytes()
    else:
        raise DataIOError("Review table output must be CSV, JSON, or Parquet.")
    return publish(output, raw, receipt=receipt)
