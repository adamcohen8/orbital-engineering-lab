from __future__ import annotations

import sqlite3
import uuid
from pathlib import Path
from typing import Any

from sim.scale.store_catalog import (
    _ensure_catalog_object,
    _insert_od_solution_record,
    _latest_od_solution_for_object,
    _upsert_current_od_solution,
)
from sim.scale.store_connection import (
    _json,
    _loads,
    _utc_stamp,
)
from sim.scale.store_measurements import (
    _complete_observation_invalidations_for_object,
    _insert_od_solution_changed_invalidation,
)
from sim.scale.store_models import (
    ManeuverFlagRecord,
    OdSolutionRecord,
    OdSolutionSummary,
)
from sim.scale.store_schema import (
    initialize_store,
)


def load_recent_od_solutions(
    path: str | Path,
    *,
    object_ids: tuple[str, ...] = (),
    limit_per_object: int = 2,
) -> dict[str, list[OdSolutionSummary]]:
    db_path = Path(path).expanduser().resolve()
    if not db_path.is_file():
        raise FileNotFoundError(f"OEL Scale store does not exist: {db_path}")
    resolved_limit = max(int(limit_per_object), 0)
    if resolved_limit <= 0:
        return {}
    params: list[Any] = []
    where = ""
    if object_ids:
        placeholders = ", ".join("?" for _ in object_ids)
        where = f"WHERE object_id IN ({placeholders})"
        params.extend(object_ids)
    with sqlite3.connect(db_path) as conn:
        current_where = ""
        current_params: list[Any] = []
        if object_ids:
            placeholders = ", ".join("?" for _ in object_ids)
            current_where = f"WHERE cos.object_id IN ({placeholders})"
            current_params.extend(object_ids)
        current_rows = conn.execute(
            f"""
            SELECT cos.object_id, cos.od_solution_id
            FROM current_object_state cos
            {current_where}
            ORDER BY cos.object_id
            """,
            current_params,
        ).fetchall()
        rows = conn.execute(
            f"""
            SELECT od_solution_id, object_id, method, input_digest, epoch_jd_utc,
                   mean_elements_json, status, supersedes_product_id
            FROM od_solutions
            {where}
            ORDER BY object_id, epoch_jd_utc DESC, created_utc DESC, rowid DESC
            """,
            params,
        ).fetchall()
    summaries: list[OdSolutionSummary] = []
    for row in rows:
        summaries.append(
            OdSolutionSummary(
                od_solution_id=str(row[0]),
                object_id=str(row[1]),
                method=str(row[2] or ""),
                input_digest=str(row[3] or ""),
                epoch_jd_utc=float(row[4]),
                mean_elements=dict(_loads(row[5], default={})),
                status=str(row[6] or ""),
                supersedes_product_id=str(row[7] or ""),
            )
        )
    by_id = {item.od_solution_id: item for item in summaries}
    by_object: dict[str, list[OdSolutionSummary]] = {}
    for item in summaries:
        by_object.setdefault(item.object_id, []).append(item)
    out: dict[str, list[OdSolutionSummary]] = {}
    seeded_objects: set[str] = set()
    for object_id_raw, current_od_solution_id_raw in current_rows:
        object_id = str(object_id_raw)
        current_od_solution_id = str(current_od_solution_id_raw)
        seeded_objects.add(object_id)
        bucket = out.setdefault(object_id, [])
        seen_ids: set[str] = set()
        next_od_solution_id = current_od_solution_id
        while next_od_solution_id and len(bucket) < resolved_limit and next_od_solution_id not in seen_ids:
            seen_ids.add(next_od_solution_id)
            summary = by_id.get(next_od_solution_id)
            if summary is None:
                break
            bucket.append(summary)
            next_od_solution_id = summary.supersedes_product_id
        for summary in by_object.get(object_id, []):
            if len(bucket) >= resolved_limit:
                break
            if summary.od_solution_id in seen_ids:
                continue
            bucket.append(summary)
            seen_ids.add(summary.od_solution_id)
    for object_id, summaries_for_object in by_object.items():
        if object_id in seeded_objects:
            continue
        out[object_id] = summaries_for_object[:resolved_limit]
    return out


def write_od_solution_records(
    path: str | Path,
    *,
    solutions: list[OdSolutionRecord],
    update_current: bool = True,
) -> dict[str, Any]:
    initialize_store(path, description="OEL Scale operational catalog store")
    db_path = Path(path).expanduser().resolve()
    now = _utc_stamp()
    inserted = 0
    current_updated = 0
    invalidations_inserted = 0
    observation_invalidations_completed = 0
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        for solution in solutions:
            if _ensure_catalog_object(conn, object_id=solution.object_id, now=now, source="od_solution_write"):
                pass
            supersedes = _latest_od_solution_for_object(conn, solution.object_id)
            if not _insert_od_solution_record(conn, solution=solution, supersedes_product_id=supersedes, now=now):
                continue
            inserted += 1
            # Retain rejected OGP fits for review without replacing an accepted
            # state, consuming observations, or invalidating downstream products.
            if solution.method == "ogp_mean_elements_fit" and solution.status != "current_candidate":
                continue
            if update_current:
                _upsert_current_od_solution(
                    conn,
                    object_id=solution.object_id,
                    od_solution_id=solution.od_solution_id,
                    current_policy=(
                        "latest_ogp_od_or_tle"
                        if solution.method in {"ogp_interchange", "ogp_mean_elements_fit"}
                        else "latest_sgp4_od_or_tle"
                    ),
                    selection_reason=f"latest {solution.method} product selected",
                    warnings=list(solution.quality_flags.get("warnings", [])),
                    now=now,
                )
                current_updated += 1
                observation_invalidations_completed += _complete_observation_invalidations_for_object(
                    conn,
                    object_id=solution.object_id,
                    now=now,
                )
            invalidations_inserted += _insert_od_solution_changed_invalidation(
                conn,
                object_id=solution.object_id,
                epoch_jd_utc=solution.epoch_jd_utc,
                stale_product_id=solution.od_solution_id,
                now=now,
            )
        conn.commit()
    return {
        "store_path": str(db_path),
        "od_solutions_seen": len(solutions),
        "od_solutions_inserted": inserted,
        "current_states_updated": current_updated,
        "invalidations_inserted": invalidations_inserted,
        "observation_invalidations_completed": observation_invalidations_completed,
    }


def write_maneuver_flags(path: str | Path, *, flags: list[ManeuverFlagRecord]) -> dict[str, Any]:
    initialize_store(path, description="OEL Scale operational catalog store")
    db_path = Path(path).expanduser().resolve()
    now = _utc_stamp()
    flags_inserted = 0
    invalidations_inserted = 0
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        for flag in flags:
            if _insert_maneuver_flag(conn, flag=flag, now=now):
                flags_inserted += 1
                invalidations_inserted += _insert_maneuver_invalidation(conn, flag=flag, now=now)
        conn.commit()
    return {
        "store_path": str(db_path),
        "flags_seen": len(flags),
        "flags_inserted": flags_inserted,
        "invalidations_inserted": invalidations_inserted,
    }


def _insert_maneuver_flag(conn: sqlite3.Connection, *, flag: ManeuverFlagRecord, now: str) -> bool:
    try:
        conn.execute(
            """
            INSERT INTO maneuver_flags (
                maneuver_flag_id, object_id, source_od_solution_id,
                start_jd_utc, end_jd_utc, method, confidence,
                status, summary_json, created_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                flag.maneuver_flag_id,
                flag.object_id,
                flag.source_od_solution_id,
                flag.start_jd_utc,
                flag.end_jd_utc,
                flag.method,
                flag.confidence,
                flag.status,
                _json(flag.summary),
                now,
            ),
        )
        return True
    except sqlite3.IntegrityError:
        return False


def _insert_maneuver_invalidation(conn: sqlite3.Connection, *, flag: ManeuverFlagRecord, now: str) -> int:
    conn.execute(
        """
        INSERT INTO invalidation_queue (
            invalidation_id, object_id, deputy_id, chief_id,
            time_start_jd_utc, time_end_jd_utc, stale_product_id,
            stale_product_type, reason, priority, status, created_utc,
            claimed_utc, completed_utc
        ) VALUES (?, ?, NULL, NULL, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL)
        """,
        (
            f"invalidation:maneuver_detected:{flag.maneuver_flag_id}:{uuid.uuid4()}",
            flag.object_id,
            flag.start_jd_utc,
            flag.end_jd_utc,
            flag.source_od_solution_id,
            "maneuver_flag",
            "maneuver_detected",
            120.0,
            "open",
            now,
        ),
    )
    return 1


def load_exact_od_assessment_inputs(path, *, od_solution_id=None, observation_ids=()):
    """Read exact immutable fit/observation rows; never select current or create a store."""
    db = Path(path).expanduser().resolve()
    if not db.is_file():
        raise FileNotFoundError(db)
    ids = tuple(str(value) for value in observation_ids)
    if len(ids) > 10000 or len(ids) != len(set(ids)):
        raise ValueError("Observation IDs must be distinct and bounded")
    with sqlite3.connect(db.as_uri() + "?mode=ro", uri=True) as conn:
        conn.row_factory = sqlite3.Row
        # Bind the fit and every source row to one SQLite read snapshot.
        conn.execute("BEGIN")
        solution = None
        if od_solution_id is not None:
            row = conn.execute("SELECT * FROM od_solutions WHERE od_solution_id=?", (od_solution_id,)).fetchone()
            if row is None:
                raise ValueError("Unknown exact OD solution ID")
            solution = dict(row)
            if ids:
                raise ValueError("Select a solution or observation IDs, not both")
            ids = tuple(_loads(solution["source_input_ids_json"], default=[]))
            if not ids or len(ids) != len(set(ids)) or len(ids) > 10000:
                raise ValueError("Fit must retain distinct source observation IDs")
        rows = []
        for start in range(0, len(ids), 500):
            batch = ids[start : start + 500]
            placeholders = ",".join("?" for _ in batch)
            rows.extend(
                dict(row)
                for row in conn.execute(
                    f"SELECT * FROM track_observations WHERE observation_id IN ({placeholders})", batch
                ).fetchall()
            )
        if len(rows) != len(ids):
            raise ValueError("Unknown or missing exact source observation IDs")
    return {
        "solution": solution,
        "observations": sorted(rows, key=lambda row: (row["time_jd_utc"], row["observation_id"])),
    }
