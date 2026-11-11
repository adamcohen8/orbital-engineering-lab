from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from sim.scale.store_connection import (
    _json,
    _utc_stamp,
)
from sim.scale.store_models import (
    ConjunctionCandidateRecord,
    ScreeningWindowRecord,
)
from sim.scale.store_schema import (
    initialize_store,
)


def write_screening_results(
    path: str | Path,
    *,
    windows: list[ScreeningWindowRecord],
    candidates: list[ConjunctionCandidateRecord],
    supersede_prior_candidates: bool = False,
) -> dict[str, Any]:
    initialize_store(path, description="OEL Scale operational catalog store")
    db_path = Path(path).expanduser().resolve()
    now = _utc_stamp()
    windows_inserted = 0
    candidates_inserted = 0
    candidates_superseded = 0
    hpop_packets_marked_stale = 0
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        for window in windows:
            if _insert_screening_window(conn, window=window, now=now):
                windows_inserted += 1
        for candidate in candidates:
            superseded_candidate_id = (
                _current_candidate_for_pair_window(conn, candidate=candidate) if supersede_prior_candidates else None
            )
            if _insert_conjunction_candidate(
                conn,
                candidate=candidate,
                now=now,
                supersedes_candidate_id=superseded_candidate_id,
            ):
                candidates_inserted += 1
                if superseded_candidate_id:
                    candidates_superseded += _mark_candidate_superseded(
                        conn,
                        candidate_id=superseded_candidate_id,
                    )
                    hpop_packets_marked_stale += _mark_hpop_packets_stale_for_candidate(
                        conn,
                        candidate_id=superseded_candidate_id,
                    )
        conn.commit()
    return {
        "store_path": str(db_path),
        "windows_seen": len(windows),
        "windows_inserted": windows_inserted,
        "candidates_seen": len(candidates),
        "candidates_inserted": candidates_inserted,
        "candidates_superseded": candidates_superseded,
        "hpop_packets_marked_stale": hpop_packets_marked_stale,
    }


def count_current_conjunction_candidates(path: str | Path, *, object_ids: tuple[str, ...] = ()) -> int:
    db_path = Path(path).expanduser().resolve()
    if not db_path.is_file():
        raise FileNotFoundError(f"OEL Scale store does not exist: {db_path}")
    params: list[Any] = []
    where = """
        WHERE cc.promotion_status != 'superseded'
          AND NOT EXISTS (
              SELECT 1
              FROM conjunction_candidates newer
              WHERE newer.supersedes_candidate_id = cc.candidate_id
          )
    """
    if object_ids:
        placeholders = ", ".join("?" for _ in object_ids)
        where += f" AND (cc.deputy_id IN ({placeholders}) OR cc.chief_id IN ({placeholders}))"
        params.extend(object_ids)
        params.extend(object_ids)
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(f"SELECT COUNT(*) FROM conjunction_candidates cc {where}", params).fetchone()
    return int(row[0] if row else 0)


def count_stale_hpop_handoff_packets(path: str | Path, *, object_ids: tuple[str, ...] = ()) -> int:
    db_path = Path(path).expanduser().resolve()
    if not db_path.is_file():
        raise FileNotFoundError(f"OEL Scale store does not exist: {db_path}")
    params: list[Any] = []
    where = "WHERE status = 'stale'"
    if object_ids:
        placeholders = ", ".join("?" for _ in object_ids)
        where += f" AND (deputy_id IN ({placeholders}) OR chief_id IN ({placeholders}))"
        params.extend(object_ids)
        params.extend(object_ids)
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(f"SELECT COUNT(*) FROM hpop_handoff_packets {where}", params).fetchone()
    return int(row[0] if row else 0)
def _insert_screening_window(
    conn: sqlite3.Connection,
    *,
    window: ScreeningWindowRecord,
    now: str,
) -> bool:
    try:
        conn.execute(
            """
            INSERT INTO screening_windows (
                screening_window_id, screening_run_id, deputy_id, chief_id,
                window_start_jd_utc, window_end_jd_utc, method,
                source_product_ids_json, input_digest, min_sample_distance_km,
                min_sample_time_jd_utc, relative_speed_km_s, pruning_reason,
                status, created_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                window.screening_window_id,
                window.screening_run_id,
                window.deputy_id,
                window.chief_id,
                window.window_start_jd_utc,
                window.window_end_jd_utc,
                window.method,
                _json(list(window.source_product_ids)),
                window.input_digest,
                window.min_sample_distance_km,
                window.min_sample_time_jd_utc,
                window.relative_speed_km_s,
                window.pruning_reason,
                window.status,
                now,
            ),
        )
        return True
    except sqlite3.IntegrityError:
        return False


def _insert_conjunction_candidate(
    conn: sqlite3.Connection,
    *,
    candidate: ConjunctionCandidateRecord,
    now: str,
    supersedes_candidate_id: str | None = None,
) -> bool:
    try:
        conn.execute(
            """
            INSERT INTO conjunction_candidates (
                candidate_id, screening_run_id, screening_window_id, deputy_id,
                chief_id, tca_jd_utc, miss_distance_km, relative_speed_km_s,
                ranking_score, ranking_features_json, promotion_status,
                warnings_json, created_utc, supersedes_candidate_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                candidate.candidate_id,
                candidate.screening_run_id,
                candidate.screening_window_id,
                candidate.deputy_id,
                candidate.chief_id,
                candidate.tca_jd_utc,
                candidate.miss_distance_km,
                candidate.relative_speed_km_s,
                candidate.ranking_score,
                _json(candidate.ranking_features),
                candidate.promotion_status,
                _json(list(candidate.warnings)),
                now,
                supersedes_candidate_id,
            ),
        )
        return True
    except sqlite3.IntegrityError:
        return False


def _current_candidate_for_pair_window(
    conn: sqlite3.Connection,
    *,
    candidate: ConjunctionCandidateRecord,
) -> str | None:
    row = conn.execute(
        """
        SELECT cc.candidate_id
        FROM conjunction_candidates cc
        JOIN screening_windows sw ON sw.screening_window_id = cc.screening_window_id
        WHERE cc.deputy_id = ?
          AND cc.chief_id = ?
          AND sw.window_start_jd_utc = (
              SELECT new_sw.window_start_jd_utc
              FROM screening_windows new_sw
              WHERE new_sw.screening_window_id = ?
          )
          AND sw.window_end_jd_utc = (
              SELECT new_sw.window_end_jd_utc
              FROM screening_windows new_sw
              WHERE new_sw.screening_window_id = ?
          )
          AND cc.promotion_status != 'superseded'
          AND NOT EXISTS (
              SELECT 1
              FROM conjunction_candidates newer
              WHERE newer.supersedes_candidate_id = cc.candidate_id
          )
        ORDER BY cc.created_utc DESC, cc.candidate_id DESC
        LIMIT 1
        """,
        (
            candidate.deputy_id,
            candidate.chief_id,
            candidate.screening_window_id,
            candidate.screening_window_id,
        ),
    ).fetchone()
    return str(row[0]) if row and row[0] else None


def _mark_candidate_superseded(conn: sqlite3.Connection, *, candidate_id: str) -> int:
    cursor = conn.execute(
        """
        UPDATE conjunction_candidates
        SET promotion_status = 'superseded'
        WHERE candidate_id = ?
          AND promotion_status != 'superseded'
        """,
        (candidate_id,),
    )
    return int(cursor.rowcount if cursor.rowcount is not None else 0)


def _mark_hpop_packets_stale_for_candidate(conn: sqlite3.Connection, *, candidate_id: str) -> int:
    cursor = conn.execute(
        """
        UPDATE hpop_handoff_packets
        SET status = 'stale'
        WHERE candidate_id = ?
          AND status IN ('planned', 'claimed', 'completed')
        """,
        (candidate_id,),
    )
    conn.execute(
        """
        UPDATE onp_refinement_cases
        SET status = 'stale', updated_utc = ?
        WHERE candidate_id = ?
          AND status IN ('materialized', 'validated', 'validation_failed')
        """,
        (_utc_stamp(), candidate_id),
    )
    return int(cursor.rowcount if cursor.rowcount is not None else 0)
