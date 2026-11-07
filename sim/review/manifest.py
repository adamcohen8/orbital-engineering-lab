from __future__ import annotations

import json
import math
import os
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any

from sim.platform_compat import canonicalize_platform_compatibility_root as _canonicalize_platform_compatibility_root
from sim.review.generated_artifacts import clear_generated_review_artifacts
from sim.security.output_writes import (
    current_output_writer,
    mkdir_output,
    read_output_bytes,
    sha256_output,
    staged_output_file,
)
from sim.utils.io import sha256_file

WORKFLOW_REVIEW_SCHEMA_VERSION = "0.1"


def _lexical_path(value: str | Path, *, base: Path | None = None) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (base or Path.cwd()) / path
    if any(part in {"", ".", ".."} for part in path.parts[1:]):
        raise ValueError(f"Workflow path contains traversal components: {path}")
    return _canonicalize_platform_compatibility_root(Path(os.path.abspath(os.fspath(path))))


def _reject_symlink_components(path: Path, *, label: str) -> None:
    path = _canonicalize_platform_compatibility_root(path)
    if any(component.is_symlink() for component in (path, *path.parents)):
        raise ValueError(f"{label} contains a symbolic link: {path}")


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True

KNOWN_WORKFLOW_TABLE_COLUMNS: dict[str, list[str]] = {
    "bench_runs": ["variant_name", "case_name", "passed", "failure_count", "output_dir"],
    "bench_variant_summaries": ["variant_name", "run_count", "passed_runs", "pass_rate"],
    "bench_leaderboard": ["kind", "objective", "metric", "rank", "variant_name", "value", "samples"],
    "bench_failures": [
        "variant_name",
        "case_name",
        "objective",
        "metric",
        "reason",
        "failure_mode",
        "suggestion",
    ],
    "sensitivity_runs": ["run_id", "status", "parameter_path", "parameter_value", "output_dir"],
    "sensitivity_rankings": ["rank", "parameter_path", "metric_path", "method", "effect_size"],
    "campaign_runs": ["iteration", "passed", "closest_approach_km", "duration_s", "total_dv_m_s", "output_dir"],
    "campaign_metrics": ["iteration", "metric_name", "metric_value"],
    "validation_benchmarks": ["benchmark_name", "kind", "passed", "duration_s", "output_dir"],
    "validation_artifacts": ["artifact_key", "path"],
    "ground_od_systematic_parameters": [
        "parameter",
        "systematic",
        "scope",
        "station_id",
        "estimate",
        "unit",
        "data_identifiable",
        "prior_dominated",
    ],
    "ground_od_stations": ["station_id", "lat_deg", "lon_deg", "alt_km", "location_frame", "disposition"],
    "ground_od_exclusions": [
        "measurement_id",
        "station_id",
        "time_jd_utc",
        "time_s",
        "measurement_type",
        "components",
        "reason",
    ],
    "ground_od_decisions": [
        "residual_index",
        "measurement_id",
        "station_id",
        "time_s",
        "component",
        "accepted",
        "reasons",
    ],
}


def write_workflow_review(
    *,
    output_dir: str | Path,
    workflow_type: str,
    title: str | None = None,
    scenario_name: str | None = None,
    status: str = "complete",
    summary: dict[str, Any] | None = None,
    artifacts: dict[str, Any] | None = None,
    recommended_queries: list[dict[str, str]] | None = None,
    recommended_review_order: list[str] | None = None,
    source_config: str | None = None,
    provenance: dict[str, Any] | None = None,
    tables: dict[str, list[dict[str, Any]]] | None = None,
) -> dict[str, str]:
    """Write the common review doorway for a completed non-single-run workflow."""

    outdir = _lexical_path(output_dir)
    _reject_symlink_components(outdir, label="Workflow output directory")
    outdir = outdir.resolve()
    review_dir = outdir / "review"
    mkdir_output(outdir)
    mkdir_output(review_dir)

    artifact_rows = _artifact_rows(artifacts or {}, output_dir=outdir)
    missing_artifacts = [row["path"] for row in artifact_rows if not bool(row["path_exists"])]
    if str(status or "complete") == "complete" and missing_artifacts:
        raise FileNotFoundError("workflow review artifacts do not exist: " + ", ".join(missing_artifacts))
    query_rows = [dict(item) for item in list(recommended_queries or []) if isinstance(item, dict)]
    table_rows = {str(name): [dict(row) for row in rows] for name, rows in dict(tables or {}).items()}
    manifest_path = review_dir / "workflow_manifest.json"
    db_path = review_dir / "run.sqlite"
    schema_path = review_dir / "schema.json"
    saved_views_path = review_dir / "saved_views.json"
    manifest_tmp = review_dir / "workflow_manifest.json.tmp"
    db_tmp = review_dir / "run.sqlite.tmp"
    schema_tmp = review_dir / "schema.json.tmp"
    saved_views_tmp = review_dir / "saved_views.json.tmp"

    generated_utc = os.environ.get("OEL_GENERATED_UTC", "").strip()

    manifest = {
        "version": 1,
        "review_schema_version": WORKFLOW_REVIEW_SCHEMA_VERSION,
        "workflow_type": str(workflow_type or "workflow"),
        "title": str(title or scenario_name or workflow_type or "workflow"),
        "scenario_name": str(scenario_name or title or workflow_type or "workflow"),
        "generated_utc": generated_utc,
        "status": str(status or "complete"),
        "summary": _jsonable(dict(summary or {})),
        "artifacts": artifact_rows,
        "recommended_queries": query_rows,
        "recommended_review_order": [str(item) for item in list(recommended_review_order or [])],
        "source_config": str(source_config or ""),
        "provenance": _jsonable(dict(provenance or {})),
        "sqlite": "review/run.sqlite" if table_rows else "",
        "schema_json": "review/schema.json" if table_rows else "",
        "saved_views_json": "review/saved_views.json" if query_rows else "",
    }
    if current_output_writer() is not None:
        return _write_protected_review(
            manifest=manifest,
            artifact_rows=artifact_rows,
            table_rows=table_rows,
            query_rows=query_rows,
            workflow_type=str(workflow_type or "workflow"),
            manifest_path=manifest_path,
            db_path=db_path,
            schema_path=schema_path,
            saved_views_path=saved_views_path,
        )
    for tmp in (manifest_tmp, db_tmp, schema_tmp, saved_views_tmp):
        _unlink_if_exists(tmp)
    try:
        if table_rows:
            _write_workflow_sqlite(
                db_path=db_tmp,
                manifest=manifest,
                artifact_rows=artifact_rows,
                tables=table_rows,
            )
            schema = _schema_from_db(db_tmp, workflow_type=str(workflow_type or "workflow"))
            schema_tmp.write_text(json.dumps(schema, indent=2, sort_keys=True), encoding="utf-8")
        if query_rows:
            saved_views_tmp.write_text(json.dumps({"views": query_rows}, indent=2, sort_keys=True), encoding="utf-8")
        manifest_tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

        clear_generated_review_artifacts(review_dir)
        if table_rows:
            db_tmp.replace(db_path)
            schema_tmp.replace(schema_path)
        else:
            _unlink_if_exists(db_path)
            _unlink_if_exists(schema_path)
        if query_rows:
            saved_views_tmp.replace(saved_views_path)
        else:
            _unlink_if_exists(saved_views_path)
        # Publish the manifest last so readers never observe references to an
        # incomplete database/schema/view set.
        manifest_tmp.replace(manifest_path)
    finally:
        for tmp in (manifest_tmp, db_tmp, schema_tmp, saved_views_tmp):
            _unlink_if_exists(tmp)

    outputs = {"workflow_manifest_json": str(manifest_path)}
    if table_rows:
        outputs.update({"sqlite": str(db_path), "schema_json": str(schema_path)})
    if query_rows:
        outputs["saved_views_json"] = str(saved_views_path)
    return outputs


def _write_protected_review(
    *,
    manifest: dict[str, Any],
    artifact_rows: list[dict[str, Any]],
    table_rows: dict[str, list[dict[str, Any]]],
    query_rows: list[dict[str, str]],
    workflow_type: str,
    manifest_path: Path,
    db_path: Path,
    schema_path: Path,
    saved_views_path: Path,
) -> dict[str, str]:
    """Publish review files through the bound writer, preserving logical paths."""

    if table_rows:
        with staged_output_file(db_path) as db_stage:
            _write_workflow_sqlite(
                db_path=db_stage,
                manifest=manifest,
                artifact_rows=artifact_rows,
                tables=table_rows,
            )
            schema = _schema_from_db(db_stage, workflow_type=workflow_type)
        with staged_output_file(schema_path) as schema_stage:
            schema_stage.write_text(json.dumps(schema, indent=2, sort_keys=True), encoding="utf-8")
    if query_rows:
        with staged_output_file(saved_views_path) as saved_views_stage:
            saved_views_stage.write_text(
                json.dumps({"views": query_rows}, indent=2, sort_keys=True),
                encoding="utf-8",
            )
    # Publish the manifest last so readers never observe references to an
    # incomplete database/schema/view set.
    with staged_output_file(manifest_path) as manifest_stage:
        manifest_stage.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    outputs = {"workflow_manifest_json": str(manifest_path)}
    if table_rows:
        outputs.update({"sqlite": str(db_path), "schema_json": str(schema_path)})
    if query_rows:
        outputs["saved_views_json"] = str(saved_views_path)
    return outputs


def load_workflow_manifest(path: str | Path) -> dict[str, Any]:
    root = Path(path).expanduser().resolve()
    manifest_path = root if root.is_file() else root / "review" / "workflow_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Workflow review manifest not found: {manifest_path}")
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    return dict(data) if isinstance(data, dict) else {}


def workflow_manifest_path(path: str | Path) -> Path:
    root = Path(path).expanduser().resolve()
    if root.is_file():
        return root
    return root / "review" / "workflow_manifest.json"


def workflow_summary_rows(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    summary = dict(manifest.get("summary", {}) or {})
    return [
        {
            "workflow_type": manifest.get("workflow_type", ""),
            "scenario_name": manifest.get("scenario_name", ""),
            "status": manifest.get("status", ""),
            "generated_utc": manifest.get("generated_utc", ""),
            "metric_name": str(key),
            "metric_value": _scalar_text(value),
        }
        for key, value in sorted(summary.items())
        if not isinstance(value, (dict, list))
    ]


def _write_workflow_sqlite(
    *,
    db_path: Path,
    manifest: dict[str, Any],
    artifact_rows: list[dict[str, str]],
    tables: dict[str, list[dict[str, Any]]],
) -> None:
    if db_path.exists():
        db_path.unlink()
    with closing(sqlite3.connect(db_path)) as conn:
        conn.execute(
            """
            CREATE TABLE workflow_metadata (
                workflow_type TEXT,
                scenario_name TEXT,
                title TEXT,
                status TEXT,
                generated_utc TEXT,
                review_schema_version TEXT,
                source_config TEXT
            )
            """
        )
        conn.execute(
            """
            INSERT INTO workflow_metadata VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                manifest.get("workflow_type", ""),
                manifest.get("scenario_name", ""),
                manifest.get("title", ""),
                manifest.get("status", ""),
                manifest.get("generated_utc", ""),
                manifest.get("review_schema_version", ""),
                manifest.get("source_config", ""),
            ),
        )
        conn.execute(
            """
            CREATE TABLE workflow_artifacts (
                artifact_key TEXT,
                artifact_type TEXT,
                path TEXT,
                path_exists INTEGER,
                size_bytes INTEGER,
                sha256 TEXT
            )
            """
        )
        conn.executemany(
            "INSERT INTO workflow_artifacts VALUES (?, ?, ?, ?, ?, ?)",
            [
                (
                    row.get("artifact_key", ""),
                    row.get("artifact_type", ""),
                    row.get("path", ""),
                    int(bool(row.get("path_exists", False))),
                    int(row.get("size_bytes", 0) or 0),
                    row.get("sha256", ""),
                )
                for row in artifact_rows
            ],
        )
        summary_rows = workflow_summary_rows(manifest)
        conn.execute(
            """
            CREATE TABLE workflow_summary (
                workflow_type TEXT,
                scenario_name TEXT,
                status TEXT,
                generated_utc TEXT,
                metric_name TEXT,
                metric_value TEXT
            )
            """
        )
        conn.executemany(
            "INSERT INTO workflow_summary VALUES (?, ?, ?, ?, ?, ?)",
            [
                (
                    row.get("workflow_type", ""),
                    row.get("scenario_name", ""),
                    row.get("status", ""),
                    row.get("generated_utc", ""),
                    row.get("metric_name", ""),
                    row.get("metric_value", ""),
                )
                for row in summary_rows
            ],
        )
        for name, rows in tables.items():
            _write_dynamic_table(conn, name, rows)
        conn.commit()


def _unlink_if_exists(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return


def _write_dynamic_table(conn: sqlite3.Connection, name: str, rows: list[dict[str, Any]]) -> None:
    table_name = _safe_identifier(name)
    normalized_rows = []
    for row_index, row in enumerate(rows):
        normalized: dict[str, Any] = {}
        source_keys: dict[str, str] = {}
        for key, value in row.items():
            column = _safe_identifier(str(key))
            if column in normalized and source_keys[column] != str(key):
                raise ValueError(
                    f"workflow table {name!r} row {row_index} has colliding columns "
                    f"{source_keys[column]!r} and {str(key)!r}"
                )
            normalized[column] = value
            source_keys[column] = str(key)
        normalized_rows.append(normalized)
    columns = _columns_for_rows(normalized_rows)
    if not columns:
        columns = list(KNOWN_WORKFLOW_TABLE_COLUMNS.get(table_name, ["row_index"]))
    column_defs = ", ".join(f"{_quote_identifier(col)} {_sqlite_type(normalized_rows, col)}" for col in columns)
    conn.execute(f"CREATE TABLE {_quote_identifier(table_name)} ({column_defs})")
    if not normalized_rows:
        return
    placeholders = ", ".join("?" for _ in columns)
    column_sql = ", ".join(_quote_identifier(col) for col in columns)
    conn.executemany(
        f"INSERT INTO {_quote_identifier(table_name)} ({column_sql}) VALUES ({placeholders})",
        [[_sqlite_value(row.get(col)) for col in columns] for row in normalized_rows],
    )


def _schema_from_db(db_path: Path, *, workflow_type: str) -> dict[str, Any]:
    with closing(sqlite3.connect(db_path)) as conn:
        tables = [
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_schema WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
            ).fetchall()
        ]
        columns = {
            table: [
                {
                    "name": str(row[1]),
                    "type": str(row[2]),
                    "notnull": bool(row[3]),
                    "primary_key": bool(row[5]),
                }
                for row in conn.execute(f"PRAGMA table_info({_quote_identifier(table)})").fetchall()
            ]
            for table in tables
        }
    return {
        "schema_version": WORKFLOW_REVIEW_SCHEMA_VERSION,
        "kind": "workflow_review",
        "workflow_type": workflow_type,
        "tables": tables,
        "columns": columns,
    }


def _artifact_rows(artifacts: dict[str, Any], *, output_dir: Path) -> list[dict[str, Any]]:
    writer = current_output_writer()
    rows: list[dict[str, Any]] = []
    for key, value in _flatten_items(artifacts):
        path_text = str(value or "").strip()
        if not path_text:
            continue
        candidate = _lexical_path(path_text, base=output_dir)
        _reject_symlink_components(candidate, label=f"Workflow artifact {key!r}")
        resolved = candidate.resolve()
        if writer is None and not _is_relative_to(resolved, output_dir.resolve()):
            raise ValueError(
                f"Workflow artifact {key!r} must remain inside the output directory: {path_text}"
            )
        if writer is not None:
            try:
                payload = read_output_bytes(resolved)
            except FileNotFoundError:
                exists = False
                payload = b""
            else:
                exists = True
        else:
            payload = b""
            exists = resolved.is_file()
        rows.append(
            {
                "artifact_key": str(key),
                "artifact_type": _artifact_type(key, path_text),
                "path": _relative_path(path_text, output_dir=output_dir),
                "path_exists": exists,
                "size_bytes": len(payload) if writer is not None and exists else int(resolved.stat().st_size) if exists else 0,
                "sha256": sha256_output(resolved) if writer is not None and exists else sha256_file(resolved) if exists else "",
            }
        )
    return rows


def _flatten_items(value: Any, *, prefix: str = "") -> list[tuple[str, Any]]:
    if isinstance(value, dict):
        rows: list[tuple[str, Any]] = []
        for key, child in sorted(value.items()):
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            rows.extend(_flatten_items(child, prefix=child_prefix))
        return rows
    if isinstance(value, list):
        return [(prefix or "list", json.dumps(_jsonable(value), sort_keys=True))]
    return [(prefix or "artifact", value)]


def _columns_for_rows(rows: list[dict[str, Any]]) -> list[str]:
    columns: list[str] = []
    for row in rows:
        for key in row.keys():
            col = _safe_identifier(str(key))
            if col not in columns:
                columns.append(col)
    return columns


def _sqlite_type(rows: list[dict[str, Any]], column: str) -> str:
    for row in rows:
        value = row.get(column)
        if value is None:
            continue
        if isinstance(value, bool):
            return "INTEGER"
        if isinstance(value, int):
            return "INTEGER"
        if isinstance(value, float):
            return "REAL"
        return "TEXT"
    return "TEXT"


def _sqlite_value(value: Any) -> Any:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float)):
        return value
    return json.dumps(_jsonable(value), sort_keys=True)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_jsonable(child) for child in value]
    if isinstance(value, tuple):
        return [_jsonable(child) for child in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _scalar_text(value: Any) -> str:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(_jsonable(value), sort_keys=True)
    if value is None:
        return ""
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    return str(value)


def _safe_identifier(value: str) -> str:
    out = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(value or "table"))
    out = "_".join(part for part in out.split("_") if part)
    if not out:
        out = "table"
    if out[0].isdigit():
        out = f"t_{out}"
    return out


def _quote_identifier(value: str) -> str:
    return '"' + str(value).replace('"', '""') + '"'


def _artifact_type(key: str, path_text: str) -> str:
    suffix = Path(path_text).suffix.lower().lstrip(".")
    if suffix in {"png", "jpg", "jpeg", "svg", "pdf"}:
        return "figure" if suffix != "pdf" else "document"
    if suffix in {"csv", "json", "sqlite", "npz"}:
        return "data"
    if suffix in {"md", "txt"}:
        return "report"
    key_lower = str(key).lower()
    if "plot" in key_lower or "figure" in key_lower:
        return "figure"
    if "report" in key_lower or "brief" in key_lower:
        return "report"
    return "artifact"


def _relative_path(path_text: str, *, output_dir: Path) -> str:
    path = _lexical_path(path_text, base=output_dir)
    _reject_symlink_components(path, label="Workflow artifact")
    resolved = path.resolve()
    try:
        return resolved.relative_to(output_dir.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"Workflow artifact must remain inside the output directory: {path_text}") from exc
