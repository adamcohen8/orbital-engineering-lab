from __future__ import annotations

import sqlite3
import uuid
from pathlib import Path
from typing import Any

from sim.dynamics.orbit.tle import (
    TLEElements,
    ogp_mean_elements_from_mapping,
    ogp_mean_elements_to_mapping,
    parse_tle_lines,
    with_tle_checksum,
)
from sim.scale.store_connection import (
    _json,
    _loads,
    _utc_stamp,
)
from sim.scale.store_models import (
    CurrentTLEState,
    OdSolutionRecord,
    TLEIngestRecord,
    TLEIngestResult,
)
from sim.scale.store_schema import (
    initialize_store,
)


def ingest_tle_records(
    path: str | Path,
    *,
    records: list[TLEIngestRecord],
    source_label: str,
    source_digest: str,
    normalized_by: str = "sim.scale.ingest_tle",
) -> TLEIngestResult:
    store_info = initialize_store(path, description="OEL Scale operational catalog store")
    db_path = Path(store_info["path"])
    batch_id = f"tle_batch:{uuid.uuid4()}"
    now = _utc_stamp()
    objects_created = 0
    tle_inserted = 0
    od_inserted = 0
    current_updated = 0
    invalidations_inserted = 0
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute(
            """
            INSERT INTO ingest_batches (
                batch_id, source_label, source_type, source_digest, normalized_by,
                created_utc, metadata_json, warnings_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                batch_id,
                source_label,
                "tle",
                source_digest,
                normalized_by,
                now,
                _json({"record_count": len(records)}),
                _json([]),
            ),
        )
        for record in records:
            if _insert_or_update_catalog_object(conn, record=record, now=now):
                objects_created += 1
            tle_id = f"tle:{record.digest}"
            inserted = _insert_tle_record(
                conn, record=record, batch_id=batch_id, tle_id=tle_id, source_label=source_label, now=now
            )
            if not inserted:
                continue
            tle_inserted += 1
            od_solution_id = f"od:latest_tle:{record.digest}"
            _insert_latest_tle_od_solution(conn, record=record, tle_id=tle_id, od_solution_id=od_solution_id, now=now)
            od_inserted += 1
            if _upsert_current_object_state(conn, record=record, od_solution_id=od_solution_id, now=now):
                current_updated += 1
                invalidations_inserted += _insert_tle_invalidations(
                    conn,
                    record=record,
                    stale_product_id=od_solution_id,
                    now=now,
                )
        conn.commit()
    return TLEIngestResult(
        store_path=db_path,
        batch_id=batch_id,
        source_label=source_label,
        source_digest=source_digest,
        records_seen=len(records),
        objects_created=objects_created,
        tle_records_inserted=tle_inserted,
        od_solutions_inserted=od_inserted,
        current_states_updated=current_updated,
        invalidations_inserted=invalidations_inserted,
    )
def load_current_tle_states(
    path: str | Path,
    *,
    object_ids: tuple[str, ...] = (),
    tags: tuple[str, ...] = (),
) -> list[CurrentTLEState]:
    db_path = Path(path).expanduser().resolve()
    if not db_path.is_file():
        raise FileNotFoundError(f"OEL Scale store does not exist: {db_path}")
    params: list[Any] = []
    where = ""
    if object_ids:
        placeholders = ", ".join("?" for _ in object_ids)
        where = f"WHERE cos.object_id IN ({placeholders})"
        params.extend(object_ids)
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            f"""
            SELECT
                cos.object_id,
                cos.od_solution_id,
                od.input_digest,
                od.method,
                od.mean_elements_json,
                od.quality_flags_json,
                od.epoch_jd_utc,
                obj.tags_json
            FROM current_object_state cos
            JOIN od_solutions od ON od.od_solution_id = cos.od_solution_id
            JOIN catalog_objects obj ON obj.object_id = cos.object_id
            {where}
            ORDER BY cos.object_id
            """,
            params,
        ).fetchall()
        tle_rows = conn.execute("SELECT digest, line1, line2, tle_epoch_jd_utc FROM tle_history").fetchall()
    tle_by_digest = {str(row[0]): (str(row[1]), str(row[2]), float(row[3])) for row in tle_rows}
    states: list[CurrentTLEState] = []
    for row in rows:
        if tags and not set(tags).issubset(set(_loads(row[7], default=[]))):
            continue
        method = str(row[3] or "")
        quality = dict(_loads(row[5], default={}))
        mean_elements = dict(_loads(row[4], default={}))
        if method in {"ogp_interchange", "ogp_mean_elements_fit"}:
            normalized = ogp_mean_elements_to_mapping(ogp_mean_elements_from_mapping(mean_elements))
            states.append(
                CurrentTLEState(
                    object_id=str(row[0]),
                    od_solution_id=str(row[1]),
                    input_digest=str(row[2]),
                    line1="",
                    line2="",
                    tle_epoch_jd_utc=float(row[6]),
                    mean_elements=normalized,
                    source_product_id=str(quality.get("source_product_id", "") or ""),
                    source_provenance=dict(quality.get("source_provenance", {}) or {}),
                )
            )
            continue
        template_digest = str(row[2] if method == "latest_tle" else quality.get("template_tle_digest", ""))
        template = tle_by_digest.get(template_digest)
        if template is None:
            continue
        line1, line2, tle_epoch = template
        if method != "latest_tle":
            line2 = _tle_line2_from_mean_elements(line2, dict(_loads(row[4], default={})))
            tle_epoch = float(row[6])
        states.append(
            CurrentTLEState(
                object_id=str(row[0]),
                od_solution_id=str(row[1]),
                input_digest=str(row[2]),
                line1=line1,
                line2=line2,
                tle_epoch_jd_utc=tle_epoch,
                mean_elements=ogp_mean_elements_to_mapping(parse_tle_lines(line1, line2)),
                source_product_id=str(quality.get("source_product_id", "") or ""),
                source_provenance=dict(quality.get("source_provenance", {}) or {}),
            )
        )
    return states


def _tle_line2_from_mean_elements(line2: str, elements: dict[str, Any]) -> str:
    required = (
        "inclination_deg",
        "raan_deg",
        "eccentricity",
        "argp_deg",
        "mean_anomaly_deg",
        "mean_motion_rev_per_day",
    )
    if any(name not in elements for name in required):
        raise ValueError("Current SGP4 orbit determination product is missing required mean elements.")
    eccentricity_digits = f"{float(elements['eccentricity']):.7f}".split(".", maxsplit=1)[1]
    updated = (
        f"{line2[:8]}{float(elements['inclination_deg']):8.4f} "
        f"{float(elements['raan_deg']):8.4f} {eccentricity_digits:>7} "
        f"{float(elements['argp_deg']):8.4f} {float(elements['mean_anomaly_deg']):8.4f} "
        f"{float(elements['mean_motion_rev_per_day']):11.8f}{line2[63:]}"
    )
    return with_tle_checksum(updated)


def load_latest_tle_states(path: str | Path, *, object_ids: tuple[str, ...] = ()) -> dict[str, CurrentTLEState]:
    db_path = Path(path).expanduser().resolve()
    if not db_path.is_file():
        raise FileNotFoundError(f"OEL Scale store does not exist: {db_path}")
    params: list[Any] = []
    where = ""
    if object_ids:
        placeholders = ", ".join("?" for _ in object_ids)
        where = f"WHERE object_id IN ({placeholders})"
        params.extend(object_ids)
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            f"""
            SELECT object_id, digest, line1, line2, tle_epoch_jd_utc
            FROM tle_history
            {where}
            ORDER BY object_id, tle_epoch_jd_utc DESC, created_utc DESC, rowid DESC
            """,
            params,
        ).fetchall()
    latest: dict[str, CurrentTLEState] = {}
    for row in rows:
        object_id = str(row[0])
        if object_id in latest:
            continue
        latest[object_id] = CurrentTLEState(
            object_id=object_id,
            od_solution_id="",
            input_digest=str(row[1]),
            line1=str(row[2]),
            line2=str(row[3]),
            tle_epoch_jd_utc=float(row[4]),
            mean_elements=ogp_mean_elements_to_mapping(parse_tle_lines(str(row[2]), str(row[3]))),
        )
    return latest


def current_ogp_elements(state: CurrentTLEState) -> TLEElements:
    """Resolve current OGP elements from native interchange or legacy TLE storage."""

    if state.mean_elements:
        return ogp_mean_elements_from_mapping(state.mean_elements)
    return parse_tle_lines(state.line1, state.line2)


def _ensure_catalog_object(conn: sqlite3.Connection, *, object_id: str, now: str, source: str) -> bool:
    oid = str(object_id or "").strip()
    if not oid:
        raise ValueError("object_id must be non-empty.")
    existing = conn.execute("SELECT object_id FROM catalog_objects WHERE object_id = ?", (oid,)).fetchone()
    if existing:
        conn.execute("UPDATE catalog_objects SET updated_utc = ? WHERE object_id = ?", (now, oid))
        return False
    conn.execute(
        """
        INSERT INTO catalog_objects (
            object_id, primary_name, object_class, norad_id, cospar_id,
            aliases_json, tags_json, priority, metadata_json, created_utc, updated_utc
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            oid,
            oid,
            "unknown",
            "",
            "",
            _json([]),
            _json([]),
            0.0,
            _json({"source": source}),
            now,
            now,
        ),
    )
    return True


def _insert_or_update_catalog_object(conn: sqlite3.Connection, *, record: TLEIngestRecord, now: str) -> bool:
    existing = conn.execute("SELECT object_id FROM catalog_objects WHERE object_id = ?", (record.object_id,)).fetchone()
    if existing:
        conn.execute(
            """
            UPDATE catalog_objects
            SET primary_name = COALESCE(NULLIF(?, ''), primary_name),
                norad_id = COALESCE(NULLIF(?, ''), norad_id),
                cospar_id = COALESCE(NULLIF(?, ''), cospar_id),
                aliases_json = ?,
                updated_utc = ?
            WHERE object_id = ?
            """,
            (
                record.primary_name,
                record.norad_id,
                record.cospar_id,
                _json(_aliases(record)),
                now,
                record.object_id,
            ),
        )
        return False
    conn.execute(
        """
        INSERT INTO catalog_objects (
            object_id, primary_name, object_class, norad_id, cospar_id,
            aliases_json, tags_json, priority, metadata_json, created_utc, updated_utc
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            record.object_id,
            record.primary_name,
            "satellite",
            record.norad_id,
            record.cospar_id,
            _json(_aliases(record)),
            _json([]),
            0.0,
            _json({"source": "tle_ingest"}),
            now,
            now,
        ),
    )
    return True
def _insert_tle_record(
    conn: sqlite3.Connection,
    *,
    record: TLEIngestRecord,
    batch_id: str,
    tle_id: str,
    source_label: str,
    now: str,
) -> bool:
    try:
        conn.execute(
            """
            INSERT INTO tle_history (
                tle_id, object_id, batch_id, line1, line2, tle_epoch_jd_utc,
                checksum_valid, parsed_elements_json, source_label, digest,
                created_utc, warnings_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                tle_id,
                record.object_id,
                batch_id,
                record.line1,
                record.line2,
                record.tle_epoch_jd_utc,
                None,
                _json(record.parsed_elements),
                source_label,
                record.digest,
                now,
                _json(list(record.warnings)),
            ),
        )
        return True
    except sqlite3.IntegrityError:
        return False


def _insert_od_solution_record(
    conn: sqlite3.Connection,
    *,
    solution: OdSolutionRecord,
    supersedes_product_id: str | None,
    now: str,
) -> bool:
    try:
        conn.execute(
            """
            INSERT INTO od_solutions (
                od_solution_id, object_id, method, source_input_ids_json,
                input_digest, epoch_jd_utc, native_frame, state_json,
                mean_elements_json, residual_summary_json, quality_flags_json,
                status, created_utc, supersedes_product_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                solution.od_solution_id,
                solution.object_id,
                solution.method,
                _json(list(solution.source_input_ids)),
                solution.input_digest,
                float(solution.epoch_jd_utc),
                solution.native_frame,
                _json(solution.state),
                _json(solution.mean_elements),
                _json(solution.residual_summary),
                _json(solution.quality_flags),
                solution.status,
                now,
                supersedes_product_id or "",
            ),
        )
        return True
    except sqlite3.IntegrityError:
        return False


def _insert_latest_tle_od_solution(
    conn: sqlite3.Connection,
    *,
    record: TLEIngestRecord,
    tle_id: str,
    od_solution_id: str,
    now: str,
) -> None:
    conn.execute(
        """
        INSERT INTO od_solutions (
            od_solution_id, object_id, method, source_input_ids_json,
            input_digest, epoch_jd_utc, native_frame, state_json,
            mean_elements_json, residual_summary_json, quality_flags_json,
            status, created_utc, supersedes_product_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            od_solution_id,
            record.object_id,
            "latest_tle",
            _json([tle_id]),
            record.digest,
            record.tle_epoch_jd_utc,
            "teme",
            _json({}),
            _json(record.parsed_elements),
            _json({}),
            _json({"source": "tle", "warnings": list(record.warnings)}),
            "current_candidate",
            now,
            _latest_od_solution_for_object(conn, record.object_id),
        ),
    )


def _latest_od_solution_for_object(conn: sqlite3.Connection, object_id: str) -> str | None:
    row = conn.execute(
        """
        SELECT od_solution_id
        FROM current_object_state
        WHERE object_id = ?
        """,
        (object_id,),
    ).fetchone()
    return str(row[0]) if row and row[0] else None


def _upsert_current_od_solution(
    conn: sqlite3.Connection,
    *,
    object_id: str,
    od_solution_id: str,
    current_policy: str,
    selection_reason: str,
    warnings: list[str],
    now: str,
) -> None:
    conn.execute(
        """
        INSERT INTO current_object_state (
            object_id, current_policy, od_solution_id, selected_utc,
            selection_reason, warnings_json
        ) VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(object_id) DO UPDATE SET
            current_policy = excluded.current_policy,
            od_solution_id = excluded.od_solution_id,
            selected_utc = excluded.selected_utc,
            selection_reason = excluded.selection_reason,
            warnings_json = excluded.warnings_json
        """,
        (
            object_id,
            current_policy,
            od_solution_id,
            now,
            selection_reason,
            _json(warnings),
        ),
    )


def _upsert_current_object_state(
    conn: sqlite3.Connection,
    *,
    record: TLEIngestRecord,
    od_solution_id: str,
    now: str,
) -> bool:
    current = conn.execute(
        """
        SELECT od.epoch_jd_utc
        FROM current_object_state AS cos
        LEFT JOIN od_solutions AS od ON od.od_solution_id = cos.od_solution_id
        WHERE cos.object_id = ?
        """,
        (record.object_id,),
    ).fetchone()
    if current is not None and current[0] is not None and float(record.tle_epoch_jd_utc) < float(current[0]):
        return False
    conn.execute(
        """
        INSERT INTO current_object_state (
            object_id, current_policy, od_solution_id, selected_utc,
            selection_reason, warnings_json
        ) VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(object_id) DO UPDATE SET
            current_policy = excluded.current_policy,
            od_solution_id = excluded.od_solution_id,
            selected_utc = excluded.selected_utc,
            selection_reason = excluded.selection_reason,
            warnings_json = excluded.warnings_json
        """,
        (
            record.object_id,
            "latest_tle",
            od_solution_id,
            now,
            "latest ingested TLE selected for Phase 1A baseline",
            _json(list(record.warnings)),
        ),
    )
    return True


def _insert_tle_invalidations(
    conn: sqlite3.Connection,
    *,
    record: TLEIngestRecord,
    stale_product_id: str,
    now: str,
) -> int:
    rows = [
        ("new_tle", 100.0),
        ("od_solution_changed", 90.0),
    ]
    for reason, priority in rows:
        conn.execute(
            """
            INSERT INTO invalidation_queue (
                invalidation_id, object_id, deputy_id, chief_id,
                time_start_jd_utc, time_end_jd_utc, stale_product_id,
                stale_product_type, reason, priority, status, created_utc,
                claimed_utc, completed_utc
            ) VALUES (?, ?, NULL, NULL, ?, NULL, ?, ?, ?, ?, ?, ?, NULL, NULL)
            """,
            (
                f"invalidation:{reason}:{record.digest}:{uuid.uuid4()}",
                record.object_id,
                record.tle_epoch_jd_utc,
                stale_product_id,
                "od_solution",
                reason,
                priority,
                "open",
                now,
            ),
        )
    return len(rows)
def _aliases(record: TLEIngestRecord) -> list[dict[str, str]]:
    aliases = [{"kind": "object_id", "value": record.object_id}]
    if record.norad_id:
        aliases.append({"kind": "norad", "value": record.norad_id})
    if record.cospar_id:
        aliases.append({"kind": "cospar", "value": record.cospar_id})
    if record.primary_name:
        aliases.append({"kind": "name", "value": record.primary_name})
    return aliases
