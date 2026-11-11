from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sim.scale.store_models import (
    CORE_TABLES,
    StoreSummary,
)


def summarize_store(path: str | Path) -> StoreSummary:
    db_path = Path(path).expanduser().resolve()
    if not db_path.is_file():
        raise FileNotFoundError(f"OEL Scale store does not exist: {db_path}")
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        table_counts = {table: _table_count(conn, table) for table in CORE_TABLES}
        metadata = conn.execute("SELECT schema_version FROM store_metadata LIMIT 1").fetchone()
        invalidation_rows = conn.execute(
            """
            SELECT reason, COUNT(*) AS count
            FROM invalidation_queue
            WHERE status IN ('open', 'planned', 'claimed')
            GROUP BY reason
            ORDER BY reason
            """
        ).fetchall()
        catalog_object_ids = tuple(
            str(row[0]) for row in conn.execute("SELECT object_id FROM catalog_objects ORDER BY object_id").fetchall()
        )
        current_object_ids = tuple(
            str(row[0])
            for row in conn.execute("SELECT object_id FROM current_object_state ORDER BY object_id").fetchall()
        )
    invalidation_by_reason = {str(row["reason"]): int(row["count"]) for row in invalidation_rows}
    return StoreSummary(
        path=db_path,
        schema_version=str(metadata["schema_version"] if metadata else ""),
        table_counts=table_counts,
        catalog_object_ids=catalog_object_ids,
        current_object_ids=current_object_ids,
        catalog_object_count=table_counts.get("catalog_objects", 0),
        current_object_state_count=table_counts.get("current_object_state", 0),
        conjunction_candidate_count=table_counts.get("conjunction_candidates", 0),
        open_invalidation_count=sum(invalidation_by_reason.values()),
        invalidation_by_reason=invalidation_by_reason,
    )
def _table_count(conn: sqlite3.Connection, table: str) -> int:
    try:
        row = conn.execute(f"SELECT COUNT(*) FROM {_quote_identifier(table)}").fetchone()
    except sqlite3.Error:
        return 0
    return int(row[0] if row else 0)


def _quote_identifier(value: str) -> str:
    safe = str(value).replace('"', '""')
    return f'"{safe}"'


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True)


def _loads(value: Any, *, default: Any) -> Any:
    if value in (None, ""):
        return default
    try:
        return json.loads(str(value))
    except json.JSONDecodeError:
        return default


def _float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)
