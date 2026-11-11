"""Bounded, explicit evidence retained from campaign iterations.

The child review store is an extraction source in compact modes. It is kept
until the campaign has assessed the run, so a failed run can be retained whole.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import Any

_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_REDUCERS = {"min", "max", "mean", "sum", "count", "first", "last"}


def validate_campaign_retention(policy: dict[str, Any]) -> None:
    unknown = set(policy) - {"mode", "fields", "stats", "full_iterations", "full_on_failure"}
    if unknown:
        raise ValueError(f"Unknown outputs.campaign_retention fields: {sorted(unknown)}")
    mode = policy.get("mode", "stats")
    if mode not in {"full", "fields", "stats"}:
        raise ValueError("outputs.campaign_retention.mode must be full, fields, or stats.")
    for key in ("fields", "stats"):
        entries = policy.get(key, [])
        if not isinstance(entries, list):
            raise ValueError(f"outputs.campaign_retention.{key} must be a list.")
        if key != mode and entries:
            raise ValueError(f"outputs.campaign_retention.{key} requires mode: {key}.")
        if key == mode == "fields" and not entries:
            raise ValueError("outputs.campaign_retention.fields must select at least one field.")
        names: set[str] = set()
        for index, entry in enumerate(entries):
            location = f"outputs.campaign_retention.{key}[{index}]"
            if not isinstance(entry, dict):
                raise ValueError(f"{location} must be a mapping.")
            allowed = {"name", "table", "column", "object_id", "deputy_id", "chief_id",
                       "start_s", "end_s", "unit"}
            allowed |= {"stride"} if key == "fields" else {"op"}
            if set(entry) - allowed:
                raise ValueError(f"{location} has unknown fields: {sorted(set(entry) - allowed)}")
            for field in ("name", "table", "column"):
                value = entry.get(field)
                if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
                    raise ValueError(f"{location}.{field} must be an SQL identifier.")
            if entry["name"] in names:
                raise ValueError(f"{location}.name duplicates {entry['name']!r}.")
            names.add(entry["name"])
            for selector in ("object_id", "deputy_id", "chief_id"):
                if selector in entry and (not isinstance(entry[selector], str) or not entry[selector]):
                    raise ValueError(f"{location}.{selector} must be a non-empty string.")
            if "unit" in entry and (not isinstance(entry["unit"], str) or not entry["unit"]):
                raise ValueError(f"{location}.unit must be a non-empty string.")
            for bound in ("start_s", "end_s"):
                if bound in entry and (isinstance(entry[bound], bool) or not isinstance(entry[bound], (int, float))
                                       or not math.isfinite(float(entry[bound]))):
                    raise ValueError(f"{location}.{bound} must be finite.")
            if float(entry.get("start_s", -math.inf)) > float(entry.get("end_s", math.inf)):
                raise ValueError(f"{location} has start_s after end_s.")
            if key == "fields":
                stride = entry.get("stride", 1)
                if type(stride) is not int or stride <= 0:
                    raise ValueError(f"{location}.stride must be a positive integer.")
            elif entry.get("op") not in _REDUCERS:
                raise ValueError(f"{location}.op must be one of {sorted(_REDUCERS)}.")
    iterations = policy.get("full_iterations", [])
    if not isinstance(iterations, list) or any(type(item) is not int or item < 0 for item in iterations):
        raise ValueError("outputs.campaign_retention.full_iterations must be nonnegative integers.")
    if "full_on_failure" in policy and type(policy["full_on_failure"]) is not bool:
        raise ValueError("outputs.campaign_retention.full_on_failure must be a boolean.")


def policy_from_config(config: dict[str, Any]) -> dict[str, Any]:
    return dict(dict(config.get("outputs", {}) or {}).get("campaign_retention", {}) or {})


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def preflight_campaign_retention(policy: dict[str, Any], iteration_count: int) -> None:
    if not policy:
        return
    validate_campaign_retention(policy)
    for iteration in policy.get("full_iterations", []):
        if iteration >= iteration_count:
            raise ValueError(f"outputs.campaign_retention.full_iterations contains out-of-range iteration {iteration}.")
    from sim.reporting.review_store import _create_schema

    with sqlite3.connect(":memory:") as connection:
        _create_schema(connection)
        for entry in policy.get(policy.get("mode", "stats"), []):
            table = entry["table"]
            columns = {row[1]: row[2].upper() for row in connection.execute(f'PRAGMA table_info("{table}")')}
            if not columns:
                raise ValueError(f"Campaign retention table {table!r} is not in the single-run review schema.")
            column = entry["column"]
            if column not in columns:
                raise ValueError(f"Campaign retention column {table}.{column} is not in the review schema.")
            for selector in ("object_id", "deputy_id", "chief_id"):
                if selector in entry and selector not in columns:
                    raise ValueError(f"Campaign retention table {table!r} has no {selector} column.")
            if ("start_s" in entry or "end_s" in entry) and "time_s" not in columns:
                raise ValueError(f"Campaign retention table {table!r} has no time_s column.")
            if policy.get("mode", "stats") == "stats" and entry["op"] != "count":
                if not any(token in columns[column] for token in ("REAL", "INT", "FLOAT", "NUMERIC", "DOUBLE")):
                    raise ValueError(f"Campaign statistic {entry['name']!r} requires a numeric review column.")
            if "BLOB" in columns[column]:
                raise ValueError(f"Campaign retention column {table}.{column} is binary and cannot be exported.")


def configure_iteration(config: dict[str, Any], *, policy: dict[str, Any] | None = None) -> None:
    policy = policy if policy is not None else policy_from_config(config)
    if not policy:
        return
    outputs = config.setdefault("outputs", {})
    outputs["review"] = {"enabled": True, "detail": "full", "strict": True}
    if policy.get("mode", "stats") == "full":
        return
    stats = outputs.setdefault("stats", {})
    stats.update({"save_full_log": False, "save_csv": False, "save_history_npz": False, "save_json": False,
                  "print_summary": False})
    outputs["plots"] = {"enabled": False}
    outputs["animations"] = {"enabled": False}


def _selected_rows(connection: sqlite3.Connection, spec: dict[str, Any]) -> Iterator[dict[str, Any]]:
    table, column = spec["table"], spec["column"]
    tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_schema WHERE type='table'")}
    if table not in tables:
        raise ValueError(f"Campaign retention table {table!r} is unavailable in this review store.")
    columns = {row[1] for row in connection.execute(f'PRAGMA table_info("{table}")')}
    if column not in columns:
        raise ValueError(f"Campaign retention column {table}.{column} is unavailable.")
    conditions: list[str] = []
    params: list[Any] = []
    for selector in ("object_id", "deputy_id", "chief_id"):
        if selector in spec:
            if selector not in columns:
                raise ValueError(f"Campaign retention table {table!r} has no {selector} column.")
            conditions.append(f'"{selector}" = ?')
            params.append(spec[selector])
    for key, op in (("start_s", ">="), ("end_s", "<=")):
        if key in spec:
            if "time_s" not in columns:
                raise ValueError(f"Campaign retention table {table!r} has no time_s column.")
            conditions.append(f'"time_s" {op} ?')
            params.append(float(spec[key]))
    extras = [name for name in ("time_s", "sample_index", "object_id", "deputy_id", "chief_id")
              if name in columns and name != column]
    projection = [column, *extras]
    where = " WHERE " + " AND ".join(conditions) if conditions else ""
    order = ', '.join(f'"{name}"' for name in ("time_s", "sample_index") if name in columns)
    order = order + ", rowid" if order else "rowid"
    select_columns = ", ".join('"' + name + '"' for name in projection)
    sql = f'SELECT {select_columns} FROM "{table}"{where} ORDER BY {order}'
    return (dict(zip(projection, row, strict=True)) for row in connection.execute(sql, params))


def _statistic(rows: Iterator[dict[str, Any]], spec: dict[str, Any]) -> dict[str, Any]:
    op = spec["op"]
    column = spec["column"]
    count = 0
    missing = 0
    total = 0.0
    compensation = 0.0
    selected: dict[str, Any] | None = None
    for row in rows:
        value = row[column]
        if op == "count":
            if value is None or isinstance(value, float) and not math.isfinite(value):
                missing += 1
            else:
                count += 1
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            missing += 1
            continue
        numeric = float(value)
        count += 1
        corrected = numeric - compensation
        updated = total + corrected
        compensation = (updated - total) - corrected
        total = updated
        if selected is None or op == "last" or (op == "min" and numeric < float(selected[column])) or (
            op == "max" and numeric > float(selected[column])
        ):
            selected = row
    result: float | int | None = None
    if op == "count":
        result = count
        selected = None
    elif count:
        if op in {"min", "max", "first", "last"}:
            result = float(selected[column])
        elif op == "sum":
            result = total
            selected = None
        elif op == "mean":
            result = total / count
            selected = None
    return {"value": result, "sample_count": count, "missing_count": missing,
            "time_s": selected.get("time_s") if selected else None, "reducer": op,
            "unit": spec.get("unit"), "table": spec["table"], "column": column,
            "object_id": spec.get("object_id"), "deputy_id": spec.get("deputy_id"),
            "chief_id": spec.get("chief_id"), "start_s": spec.get("start_s"), "end_s": spec.get("end_s"),
            "sampling": "stored_review_rows", "exact": True}


def extract_iteration(output_dir: str | Path, policy: dict[str, Any]) -> dict[str, Any]:
    path = Path(output_dir) / "review" / "run.sqlite"
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"Campaign iteration review store missing: {path}")
    digest = _sha256_file(path)
    retained: dict[str, Any] = {"source_review_sha256": digest}
    if policy.get("mode", "stats") == "full":
        return retained
    mode = policy.get("mode", "stats")
    with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as connection:
        values: dict[str, Any] = {}
        for spec in policy.get(mode, []):
            rows = _selected_rows(connection, spec)
            if mode == "fields":
                stride = int(spec.get("stride", 1))
                values[spec["name"]] = {"table": spec["table"], "column": spec["column"],
                                        "unit": spec.get("unit"), "object_id": spec.get("object_id"),
                                        "deputy_id": spec.get("deputy_id"), "chief_id": spec.get("chief_id"),
                                        "start_s": spec.get("start_s"), "end_s": spec.get("end_s"),
                                        "stride": stride, "rows": [row for index, row in enumerate(rows)
                                                                    if index % stride == 0]}
            else:
                values[spec["name"]] = _statistic(rows, spec)
        retained[mode] = values
    return retained


def compact_iteration_result(
    payload: dict[str, Any], result: dict[str, Any], *, policy: dict[str, Any],
    metric_paths: list[str], output_dir: str | Path,
) -> dict[str, Any]:
    """Keep only values needed by report/gates, checkpoints, and requested evidence."""
    if not policy:
        return result
    from sim.execution.metrics import extract_study_metric

    result["retained_evidence"] = extract_iteration(output_dir, policy)
    if policy.get("mode", "stats") == "full":
        return result
    result["metric_cache"] = {path: extract_study_metric(payload, path) for path in set(metric_paths)}
    result.pop("payload", None)
    summary = dict(result.get("summary", {}) or {})
    summary.pop("review_sqlite_path", None)
    summary.pop("review_outputs", None)
    summary["review_store_status"] = "campaign_retention_pending"
    artifacts = dict(summary.get("artifacts", {}) or {})
    artifacts.pop("review_store", None)
    summary["artifacts"] = artifacts
    result["summary"] = summary
    return result


def retire_iteration_review(output_dir: str | Path, *, expected_sha256: str | None) -> None:
    review_dir = Path(output_dir) / "review"
    db_path = review_dir / "run.sqlite"
    if not db_path.exists():
        return
    if db_path.is_symlink() or not db_path.is_file() or not expected_sha256:
        raise ValueError(f"Campaign review store is not eligible for retirement: {db_path}")
    if _sha256_file(db_path) != expected_sha256:
        raise ValueError(f"Campaign review store changed after extraction: {db_path}")
    db_path.unlink()
    index = Path(output_dir) / "index.md"
    if index.is_file() and not index.is_symlink():
        index.write_text("# Campaign iteration\n\nSelected evidence is in the parent campaign_dataset.jsonl.\n", encoding="utf-8")


def write_campaign_dataset(
    output_dir: str | Path, *, policy: dict[str, Any], records: list[dict[str, Any]],
) -> dict[str, str]:
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    data_path = outdir / "campaign_dataset.jsonl"
    staged = data_path.with_suffix(".jsonl.tmp")
    with staged.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(_json_safe(record), sort_keys=True, allow_nan=False) + "\n")
    staged.replace(data_path)
    manifest = {"schema": "oel.campaign_dataset.v1", "policy": policy,
                "iteration_count": len(records), "data_file": data_path.name,
                "sha256": _sha256_file(data_path)}
    manifest_path = outdir / "campaign_dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"dataset_jsonl": str(data_path), "dataset_manifest_json": str(manifest_path)}


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if hasattr(value, "item"):
        return _json_safe(value.item())
    return value
