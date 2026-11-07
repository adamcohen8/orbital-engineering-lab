"""Create an explicitly partial, queryable excerpt of one completed run."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sqlite3
import tempfile
from contextlib import closing
from pathlib import Path
from typing import Any, Sequence

from sim.review.workspace import ReviewWorkspace

SLICE_SCHEMA = "oel.single_run_review_slice.v1"
SLICE_DATABASE = "review_slice.sqlite"
SLICE_MANIFEST = "slice_manifest.json"
FAMILIES = frozenset({"state", "relative", "control", "fsw", "access", "events", "global_metrics"})

_TABLE_FAMILY = {
    "time_samples": "state", "object_state": "state", "object_state_covariance": "state",
    "object_orbital_elements": "state", "attitude_error": "state",
    "spacecraft_resources": "state",
    "relative_state": "relative",
    "thrust": "control", "impulsive_maneuvers": "control",
    "controller_decisions": "control", "mission_modes": "control",
    "mission_transitions": "control", "command_gates": "control",
    "fsw_invocations": "fsw", "fsw_input_events": "fsw", "fsw_load_events": "fsw",
    "fsw_objectives": "fsw", "fsw_task_timing": "fsw", "actuator_commands": "fsw",
    "actuator_command_receipts": "fsw", "actuator_realization": "fsw",
    "actuator_device_state": "fsw", "fsw_diagnostics": "fsw",
    "fsw_diagnostic_fields": "fsw", "safety_requirement_evidence": "fsw",
    "fsw_snapshots": "fsw",
    "ground_access": "access", "ground_access_windows": "access",
    "events": "events", "metrics": "global_metrics",
    "coverage_summary": None, "coverage_samples": None, "coverage_intervals": None,
    "coverage_transitions": None, "link_summary": None, "link_samples": None,
    "link_windows": None, "link_transitions": None,
    "game_input_events": None, "game_observer_samples": None, "game_scoring_events": None,
    "mission_recovery_summary": None, "mission_recovery_elements": None,
    "mission_recovery_candidates": None, "mission_recovery_burns": None,
    "mission_recovery_candidate_elements": None,
    "artifacts": None,
}
_METADATA_TABLES = frozenset({
    "run_metadata", "objects", "frame_provenance", "object_propagation",
    "object_initialization", "object_state_frame", "time_samples",
})
_FSW_INVOCATION_CHILDREN = frozenset({
    "fsw_input_events", "fsw_load_events", "fsw_objectives", "fsw_task_timing",
    "actuator_commands", "fsw_diagnostics", "fsw_diagnostic_fields",
    "safety_requirement_evidence", "fsw_snapshots",
})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_sha256(document: dict[str, Any]) -> str:
    payload = {key: value for key, value in document.items() if key != "manifest_sha256"}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _tables(connection: sqlite3.Connection) -> list[str]:
    return [str(row[0]) for row in connection.execute(
        "SELECT name FROM sqlite_schema WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
    )]


def _columns(connection: sqlite3.Connection, table: str) -> set[str]:
    return {str(row[1]) for row in connection.execute(f'PRAGMA table_info("{table}")')}


def _object_predicate(columns: set[str], selected: tuple[str, ...]) -> tuple[str, list[Any]]:
    if not selected:
        return "1", []
    ids = "(" + ",".join("?" for _ in selected) + ")"
    if {"deputy_id", "chief_id"} <= columns:
        return f"deputy_id IN {ids} AND chief_id IN {ids}", [*selected, *selected]
    for name in ("object_id", "source_object_id"):
        if name in columns:
            return f"{name} IN {ids}", list(selected)
    return "1", []


def _time_predicate(table: str, columns: set[str], start: float, end: float) -> tuple[str, list[Any]]:
    if table == "ground_access_windows":
        return "start_s <= ? AND end_s >= ?", [end, start]
    if "time_s" in columns:
        return "time_s >= ? AND time_s <= ?", [start, end]
    if table == "actuator_realization":
        return "interval_start_ns <= ? AND interval_end_ns >= ?", [round(end * 1e9), round(start * 1e9)]
    for name in ("invocation_time_ns", "delivery_time_ns", "received_time_ns", "interval_start_ns"):
        if name in columns:
            return f"{name} >= ? AND {name} <= ?", [round(start * 1e9), round(end * 1e9)]
    return "1", []


def create_review_slice(
    source: str | Path,
    destination: str | Path,
    *,
    start_s: float,
    end_s: float,
    object_ids: Sequence[str] = (),
    families: Sequence[str] = ("state", "relative", "control", "events"),
    staging_parent: str | Path | None = None,
) -> dict[str, Any]:
    """Write a new slice directory; never modify or replace the source run."""

    start, end = float(start_s), float(end_s)
    if not math.isfinite(start) or not math.isfinite(end) or start > end:
        raise ValueError("Slice time bounds must be finite and start_s <= end_s.")
    selected = tuple(sorted(set(map(str, object_ids))))
    chosen = tuple(sorted(set(map(str, families))))
    if any(not value for value in selected) or not chosen or not set(chosen) <= FAMILIES:
        raise ValueError(f"Slice objects must be non-empty and families must be selected from {sorted(FAMILIES)}.")
    target = Path(destination).expanduser().resolve()
    if target.exists() or target.is_symlink():
        raise FileExistsError("Review slice destination must be new.")
    if not target.parent.is_dir():
        raise FileNotFoundError("Review slice destination parent does not exist.")

    temporary_parent = (
        target.parent
        if staging_parent is None
        else Path(staging_parent).expanduser().absolute()
    )
    current = temporary_parent
    components: list[Path] = []
    while True:
        components.append(current)
        if current.parent == current:
            break
        current = current.parent
    for component in reversed(components):
        if component.is_symlink():
            raise ValueError(f"Review slice staging path cannot contain symbolic links: {component}")
    if not temporary_parent.is_dir():
        raise FileNotFoundError("Review slice staging parent does not exist.")
    temporary_parent = temporary_parent.resolve(strict=True)
    if temporary_parent.stat().st_dev != target.parent.stat().st_dev:
        raise ValueError("Review slice staging and destination must share a filesystem.")

    with ReviewWorkspace.open(source) as workspace:
        logical = workspace.logical_db_path
        if logical.resolve() == target or logical.name != "run.sqlite":
            raise ValueError("A review slice requires a complete single-run review/run.sqlite source.")
        source_identity = workspace.evidence_identity()
        with closing(workspace._connect(integrity=True)) as original:
            quick_check = original.execute("PRAGMA quick_check").fetchone()
            if quick_check is None or quick_check[0] != "ok":
                raise ValueError("Source review store failed SQLite quick_check.")
            tables = _tables(original)
            unsupported_schema = original.execute(
                "SELECT name, type FROM sqlite_schema WHERE type IN ('view', 'trigger') LIMIT 1"
            ).fetchone()
            if unsupported_schema is not None:
                raise ValueError(f"Review slice does not support source views or triggers: {unsupported_schema!r}")
            unknown = set(tables) - set(_TABLE_FAMILY) - _METADATA_TABLES
            if unknown or "run_metadata" not in tables or "time_samples" not in tables:
                raise ValueError(f"Unsupported single-run review table inventory: {sorted(unknown)}")
            if original.execute("SELECT COUNT(*) FROM run_metadata").fetchone()[0] != 1:
                raise ValueError("Slice source must contain exactly one run_metadata row.")
            run_id, config_sha = original.execute(
                "SELECT run_id, config_sha256 FROM run_metadata"
            ).fetchone()
            bounds = original.execute("SELECT MIN(time_s), MAX(time_s) FROM time_samples").fetchone()
            if bounds[0] is None or start < bounds[0] or end > bounds[1]:
                raise ValueError(f"Slice window must lie within sampled run bounds {bounds!r}.")
            available = {str(row[0]) for row in original.execute("SELECT object_id FROM objects")}
            if not set(selected) <= available:
                raise ValueError(f"Unknown slice object ids: {sorted(set(selected) - available)}")
            before = {table: original.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0] for table in tables}

            with tempfile.TemporaryDirectory(prefix=f".{target.name}.", dir=temporary_parent) as temporary:
                staged = Path(temporary) / target.name
                staged.mkdir()
                database = staged / SLICE_DATABASE
                with sqlite3.connect(database) as sliced:
                    original.backup(sliced)
                    if workspace.evidence_identity()["sha256"] != source_identity["sha256"]:
                        raise ValueError("Source review store changed while the slice was being created.")
                    sliced.execute(
                        "UPDATE run_metadata SET output_dir = NULL, config_path = NULL, "
                        "summary_json_path = NULL, run_log_json_path = NULL"
                    )
                    sliced.execute("CREATE TEMP TABLE keep_fsw AS SELECT object_id, invocation_id "
                                   "FROM fsw_invocations WHERE invocation_time_ns BETWEEN ? AND ?",
                                   (round(start * 1e9), round(end * 1e9)))
                    if selected:
                        ids = ",".join("?" for _ in selected)
                        sliced.execute(f"DELETE FROM keep_fsw WHERE object_id NOT IN ({ids})", selected)
                    sliced.execute("CREATE TEMP TABLE keep_commands AS SELECT object_id, command_source_id, "
                                   "command_boot_id, command_sequence FROM actuator_commands "
                                   "WHERE EXISTS (SELECT 1 FROM keep_fsw k WHERE k.object_id = actuator_commands.object_id "
                                   "AND k.invocation_id = actuator_commands.invocation_id)")
                    for table in tables:
                        if table == "run_metadata":
                            continue
                        family = _TABLE_FAMILY.get(table)
                        if table in _METADATA_TABLES:
                            family = "metadata"
                        if family is None or (family != "metadata" and family not in chosen):
                            sliced.execute(f'DELETE FROM "{table}"')
                            continue
                        columns = _columns(sliced, table)
                        object_sql, object_values = _object_predicate(columns, selected)
                        if table == "metrics":
                            # This family explicitly carries full-run metrics.
                            object_sql, object_values = "1", []
                        time_sql, time_values = _time_predicate(table, columns, start, end)
                        if table in _FSW_INVOCATION_CHILDREN or table == "fsw_invocations":
                            time_sql = ("EXISTS (SELECT 1 FROM keep_fsw k WHERE k.object_id = "
                                        f'"{table}".object_id AND k.invocation_id = "{table}".invocation_id)')
                            time_values = []
                        elif table == "actuator_command_receipts":
                            time_sql = ("EXISTS (SELECT 1 FROM keep_commands k WHERE k.object_id = "
                                        "actuator_command_receipts.object_id AND k.command_source_id = "
                                        "actuator_command_receipts.command_source_id AND k.command_boot_id = "
                                        "actuator_command_receipts.command_boot_id AND k.command_sequence = "
                                        "actuator_command_receipts.command_sequence)")
                            time_values = []
                        sliced.execute(f'DELETE FROM "{table}" WHERE NOT ({object_sql} AND {time_sql})',
                                       [*object_values, *time_values])
                    sliced.commit()
                    sliced.execute("VACUUM")
                    if sliced.execute("PRAGMA quick_check").fetchone() != ("ok",):
                        raise ValueError("Sliced review database failed SQLite quick_check.")
                    after = {table: sliced.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0] for table in tables}
                manifest = {
                    "schema": SLICE_SCHEMA,
                    "partial": True,
                    "read_only_review": True,
                    "continuation_eligible": False,
                    "source": {
                        "logical_path": "review/run.sqlite",
                        "sha256": source_identity["sha256"],
                        "bytes": source_identity["size_bytes"],
                        "run_id": run_id,
                        "config_sha256": config_sha,
                    },
                    "selection": {
                        "start_s": start, "end_s": end,
                        "object_ids": list(selected), "families": list(chosen),
                    },
                    "global_metadata_retained": True,
                    "global_metrics_retained": "global_metrics" in chosen,
                    "row_counts_before": before,
                    "row_counts_after": after,
                    "database": {"path": SLICE_DATABASE, "sha256": _sha256(database), "bytes": database.stat().st_size},
                }
                manifest["manifest_sha256"] = _manifest_sha256(manifest)
                (staged / SLICE_MANIFEST).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                staged.rename(target)
                return manifest


def verify_review_slice(path: str | Path) -> dict[str, Any]:
    """Verify the portable slice's self identity and database bytes."""
    directory = Path(path).expanduser().resolve(strict=True)
    manifest = json.loads((directory / SLICE_MANIFEST).read_text(encoding="utf-8"))
    database = directory / SLICE_DATABASE
    if (not isinstance(manifest, dict) or manifest.get("schema") != SLICE_SCHEMA
            or manifest.get("partial") is not True
            or manifest.get("manifest_sha256") != _manifest_sha256(manifest)):
        raise ValueError("Invalid review slice manifest.")
    declared = manifest.get("database")
    if (not isinstance(declared, dict) or declared.get("path") != SLICE_DATABASE
            or database.is_symlink() or database.stat().st_size != declared.get("bytes")
            or _sha256(database) != declared.get("sha256")):
        raise ValueError("Review slice database digest mismatch.")
    return manifest


def query_review_slice(path: str | Path, sql: str, *, max_rows: int = 1000) -> Any:
    """Query a verified slice without presenting it as a complete ReviewWorkspace."""

    from sim.review.workspace import _validate_select_sql

    verify_review_slice(path)
    directory = Path(path).expanduser().resolve(strict=True)
    database = directory / SLICE_DATABASE
    with ReviewWorkspace.open(database) as workspace:
        return workspace.query(_validate_select_sql(sql), max_rows=max_rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    command = parser.add_subparsers(dest="command", required=True)
    create = command.add_parser("create", help="Create a partial single-run review artifact.")
    create.add_argument("source")
    create.add_argument("destination")
    create.add_argument("--start-s", type=float, required=True)
    create.add_argument("--end-s", type=float, required=True)
    create.add_argument("--object", action="append", default=[])
    create.add_argument("--family", action="append", choices=sorted(FAMILIES))
    query = command.add_parser("query", help="Run a read-only query on a verified slice.")
    query.add_argument("slice_dir")
    query.add_argument("--sql", required=True)
    query.add_argument("--max-rows", type=int, default=50)
    args = parser.parse_args(argv)
    try:
        if args.command == "create":
            result = create_review_slice(args.source, args.destination,
                                         start_s=args.start_s, end_s=args.end_s,
                                         object_ids=args.object,
                                         families=args.family or ("state", "relative", "control", "events"))
        else:
            selected = query_review_slice(args.slice_dir, args.sql, max_rows=args.max_rows)
            result = {"columns": selected.columns, "rows": selected.rows,
                      "row_count": selected.row_count, "truncated": selected.truncated}
    except (OSError, ValueError, sqlite3.Error) as exc:
        parser.exit(2, f"review slice failed: {exc}\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
