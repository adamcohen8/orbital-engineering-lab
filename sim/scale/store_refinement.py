from __future__ import annotations

import hashlib
import sqlite3
import uuid
from pathlib import Path
from typing import Any

from sim.scale.store_connection import (
    _json,
    _loads,
    _utc_stamp,
)
from sim.scale.store_models import (
    CandidateForRefinement,
    CandidateRefinementRecord,
    HpopHandoffCase,
    HpopHandoffPacketRecord,
    OnpRefinementCaseRecord,
    OnpRefinementRunRecord,
)
from sim.scale.store_schema import (
    initialize_store,
)


def load_pending_candidates_for_refinement(
    path: str | Path,
    *,
    object_ids: tuple[str, ...] = (),
    limit: int = 0,
) -> list[CandidateForRefinement]:
    db_path = Path(path).expanduser().resolve()
    if not db_path.is_file():
        raise FileNotFoundError(f"OEL Scale store does not exist: {db_path}")
    params: list[Any] = []
    where = """
        WHERE NOT EXISTS (
            SELECT 1
            FROM candidate_refinements cr
            WHERE cr.candidate_id = cc.candidate_id
              AND cr.status = 'completed'
        )
        AND cc.promotion_status != 'superseded'
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
    limit_clause = ""
    if limit > 0:
        limit_clause = "LIMIT ?"
        params.append(int(limit))
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            f"""
            SELECT
                cc.candidate_id,
                cc.screening_run_id,
                cc.screening_window_id,
                cc.deputy_id,
                cc.chief_id,
                sw.window_start_jd_utc,
                sw.window_end_jd_utc,
                sw.source_product_ids_json,
                cc.tca_jd_utc,
                cc.miss_distance_km,
                cc.relative_speed_km_s,
                cc.ranking_features_json
            FROM conjunction_candidates cc
            JOIN screening_windows sw ON sw.screening_window_id = cc.screening_window_id
            {where}
            ORDER BY cc.miss_distance_km ASC, cc.relative_speed_km_s ASC, cc.candidate_id ASC
            {limit_clause}
            """,
            params,
        ).fetchall()
    return [
        CandidateForRefinement(
            candidate_id=str(row[0]),
            screening_run_id=str(row[1] or ""),
            screening_window_id=str(row[2]),
            deputy_id=str(row[3]),
            chief_id=str(row[4]),
            window_start_jd_utc=float(row[5]),
            window_end_jd_utc=float(row[6]),
            source_product_ids=tuple(str(item) for item in _loads(row[7], default=[])),
            sampled_tca_jd_utc=float(row[8]),
            sampled_miss_distance_km=float(row[9]),
            sampled_relative_speed_km_s=float(row[10]),
            ranking_features=dict(_loads(row[11], default={})),
        )
        for row in rows
    ]


def pending_candidate_refinement_count(path: str | Path, *, object_ids: tuple[str, ...] = ()) -> int:
    return len(load_pending_candidates_for_refinement(path, object_ids=object_ids))


def write_candidate_refinements(
    path: str | Path,
    *,
    refinements: list[CandidateRefinementRecord],
) -> dict[str, Any]:
    initialize_store(path, description="OEL Scale operational catalog store")
    db_path = Path(path).expanduser().resolve()
    now = _utc_stamp()
    inserted = 0
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        for refinement in refinements:
            if _insert_candidate_refinement(conn, refinement=refinement, now=now):
                inserted += 1
                _update_candidate_with_refinement(conn, refinement=refinement)
        conn.commit()
    return {
        "store_path": str(db_path),
        "refinements_seen": len(refinements),
        "refinements_inserted": inserted,
    }


def load_pending_hpop_handoff_cases(
    path: str | Path,
    *,
    object_ids: tuple[str, ...] = (),
    limit: int = 0,
) -> list[HpopHandoffCase]:
    db_path = Path(path).expanduser().resolve()
    if not db_path.is_file():
        raise FileNotFoundError(f"OEL Scale store does not exist: {db_path}")
    params: list[Any] = []
    where = """
        WHERE cc.promotion_status = 'refined_for_hpop'
          AND cr.status = 'completed'
          AND NOT EXISTS (
              SELECT 1
              FROM conjunction_candidates newer
              WHERE newer.supersedes_candidate_id = cc.candidate_id
          )
          AND NOT EXISTS (
              SELECT 1
              FROM hpop_handoff_packets hp
              WHERE hp.candidate_id = cc.candidate_id
                AND hp.status IN ('planned', 'claimed', 'completed')
          )
    """
    if object_ids:
        placeholders = ", ".join("?" for _ in object_ids)
        where += f" AND (cc.deputy_id IN ({placeholders}) OR cc.chief_id IN ({placeholders}))"
        params.extend(object_ids)
        params.extend(object_ids)
    limit_clause = ""
    if limit > 0:
        limit_clause = "LIMIT ?"
        params.append(int(limit))
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            f"""
            SELECT
                cc.candidate_id,
                cr.refinement_id,
                cc.screening_window_id,
                cc.deputy_id,
                cc.chief_id,
                cr.refined_tca_jd_utc,
                cr.refined_miss_distance_km,
                cr.refined_relative_speed_km_s,
                cr.ranking_score,
                cr.ranking_features_json,
                cr.source_product_ids_json
            FROM conjunction_candidates cc
            JOIN candidate_refinements cr ON cr.candidate_id = cc.candidate_id
            {where}
            ORDER BY cr.refined_miss_distance_km ASC,
                     cr.refined_relative_speed_km_s ASC,
                     cc.candidate_id ASC
            {limit_clause}
            """,
            params,
        ).fetchall()
    return [
        HpopHandoffCase(
            candidate_id=str(row[0]),
            refinement_id=str(row[1]),
            screening_window_id=str(row[2]),
            deputy_id=str(row[3]),
            chief_id=str(row[4]),
            refined_tca_jd_utc=float(row[5]),
            refined_miss_distance_km=float(row[6]),
            refined_relative_speed_km_s=float(row[7]),
            ranking_score=float(row[8]),
            ranking_features=dict(_loads(row[9], default={})),
            source_product_ids=tuple(str(item) for item in _loads(row[10], default=[])),
        )
        for row in rows
    ]


def pending_hpop_handoff_count(path: str | Path, *, object_ids: tuple[str, ...] = ()) -> int:
    return len(load_pending_hpop_handoff_cases(path, object_ids=object_ids))


def load_pending_onp_handoff_cases(
    path: str | Path, *, object_ids: tuple[str, ...] = (), limit: int = 0
) -> list[HpopHandoffCase]:
    return load_pending_hpop_handoff_cases(path, object_ids=object_ids, limit=limit)


def pending_onp_handoff_count(path: str | Path, *, object_ids: tuple[str, ...] = ()) -> int:
    return pending_hpop_handoff_count(path, object_ids=object_ids)


def write_hpop_handoff_packets(
    path: str | Path,
    *,
    packets: list[HpopHandoffPacketRecord],
) -> dict[str, Any]:
    initialize_store(path, description="OEL Scale operational catalog store")
    db_path = Path(path).expanduser().resolve()
    now = _utc_stamp()
    inserted = 0
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        for packet in packets:
            if _insert_hpop_handoff_packet(conn, packet=packet, now=now):
                inserted += 1
        conn.commit()
    return {
        "store_path": str(db_path),
        "packets_seen": len(packets),
        "packets_inserted": inserted,
    }


def write_onp_handoff_packets(
    path: str | Path, *, packets: list[HpopHandoffPacketRecord]
) -> dict[str, Any]:
    return write_hpop_handoff_packets(path, packets=packets)


def load_hpop_handoff_packets(
    path: str | Path,
    *,
    object_ids: tuple[str, ...] = (),
) -> list[HpopHandoffPacketRecord]:
    db_path = Path(path).expanduser().resolve()
    if not db_path.is_file():
        raise FileNotFoundError(f"OEL Scale store does not exist: {db_path}")
    params: list[Any] = []
    where = "WHERE status IN ('planned', 'claimed', 'completed')"
    if object_ids:
        placeholders = ", ".join("?" for _ in object_ids)
        where += f" AND (deputy_id IN ({placeholders}) OR chief_id IN ({placeholders}))"
        params.extend(object_ids)
        params.extend(object_ids)
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            f"""
            SELECT
                packet_id, candidate_id, refinement_id, screening_window_id,
                deputy_id, chief_id, force_model_profile, case_start_jd_utc,
                case_end_jd_utc, tca_jd_utc, miss_distance_km,
                relative_speed_km_s, ranking_score, output_dir, packet_path,
                packet_json, input_digest, status, warnings_json
            FROM hpop_handoff_packets
            {where}
            ORDER BY miss_distance_km ASC, relative_speed_km_s ASC, packet_id ASC
            """,
            params,
        ).fetchall()
    return [
        HpopHandoffPacketRecord(
            packet_id=str(row[0]),
            candidate_id=str(row[1]),
            refinement_id=str(row[2] or ""),
            screening_window_id=str(row[3] or ""),
            deputy_id=str(row[4]),
            chief_id=str(row[5]),
            force_model_profile=str(row[6] or ""),
            case_start_jd_utc=float(row[7]),
            case_end_jd_utc=float(row[8]),
            tca_jd_utc=float(row[9]),
            miss_distance_km=float(row[10]),
            relative_speed_km_s=float(row[11]),
            ranking_score=float(row[12]),
            output_dir=str(row[13] or ""),
            packet_path=str(row[14] or ""),
            packet_json=dict(_loads(row[15], default={})),
            input_digest=str(row[16] or ""),
            status=str(row[17] or ""),
            warnings=tuple(str(item) for item in _loads(row[18], default=[])),
        )
        for row in rows
    ]


def load_onp_handoff_packets(
    path: str | Path, *, object_ids: tuple[str, ...] = ()
) -> list[HpopHandoffPacketRecord]:
    return load_hpop_handoff_packets(path, object_ids=object_ids)


def load_pending_onp_refinement_packets(
    path: str | Path,
    *,
    object_ids: tuple[str, ...] = (),
    limit: int | None = None,
    materialization_signature: str | None = None,
) -> list[HpopHandoffPacketRecord]:
    db_path = Path(path).expanduser().resolve()
    if not db_path.is_file():
        raise FileNotFoundError(f"OEL Scale store does not exist: {db_path}")
    params: list[Any] = []
    object_clause = ""
    if object_ids:
        placeholders = ", ".join("?" for _ in object_ids)
        object_clause = f" AND (hp.deputy_id IN ({placeholders}) OR hp.chief_id IN ({placeholders}))"
        params.extend(object_ids)
        params.extend(object_ids)
    limit_clause = ""
    if limit is not None:
        limit_clause = " LIMIT ?"
    signature_clause = ""
    if materialization_signature is not None:
        signature_clause = " AND oc.materialization_signature = ?"
        params.append(str(materialization_signature))
    if limit is not None:
        params.append(max(int(limit), 0))
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            f"""
            SELECT
                hp.packet_id, hp.candidate_id, hp.refinement_id, hp.screening_window_id,
                hp.deputy_id, hp.chief_id, hp.force_model_profile, hp.case_start_jd_utc,
                hp.case_end_jd_utc, hp.tca_jd_utc, hp.miss_distance_km,
                hp.relative_speed_km_s, hp.ranking_score, hp.output_dir, hp.packet_path,
                hp.packet_json, hp.input_digest, hp.status, hp.warnings_json
            FROM hpop_handoff_packets hp
            WHERE hp.status IN ('planned', 'claimed', 'completed')
              {object_clause}
              AND NOT EXISTS (
                  SELECT 1
                  FROM onp_refinement_cases oc
                  WHERE oc.packet_id = hp.packet_id
                    AND oc.packet_input_digest = hp.input_digest
                    {signature_clause}
                    AND oc.status IN ('materialized', 'validated')
              )
            ORDER BY hp.miss_distance_km ASC, hp.relative_speed_km_s ASC, hp.packet_id ASC
            {limit_clause}
            """,
            params,
        ).fetchall()
    return [
        HpopHandoffPacketRecord(
            packet_id=str(row[0]),
            candidate_id=str(row[1]),
            refinement_id=str(row[2] or ""),
            screening_window_id=str(row[3] or ""),
            deputy_id=str(row[4]),
            chief_id=str(row[5]),
            force_model_profile=str(row[6] or ""),
            case_start_jd_utc=float(row[7]),
            case_end_jd_utc=float(row[8]),
            tca_jd_utc=float(row[9]),
            miss_distance_km=float(row[10]),
            relative_speed_km_s=float(row[11]),
            ranking_score=float(row[12]),
            output_dir=str(row[13] or ""),
            packet_path=str(row[14] or ""),
            packet_json=dict(_loads(row[15], default={})),
            input_digest=str(row[16] or ""),
            status=str(row[17] or ""),
            warnings=tuple(str(item) for item in _loads(row[18], default=[])),
        )
        for row in rows
    ]


def pending_onp_refinement_materialization_count(
    path: str | Path,
    *,
    object_ids: tuple[str, ...] = (),
    materialization_signature: str | None = None,
) -> int:
    return len(
        load_pending_onp_refinement_packets(
            path,
            object_ids=object_ids,
            materialization_signature=materialization_signature,
        )
    )


def write_onp_refinement_cases(
    path: str | Path,
    *,
    cases: list[OnpRefinementCaseRecord],
) -> dict[str, Any]:
    initialize_store(path, description="OEL Scale operational catalog store")
    db_path = Path(path).expanduser().resolve()
    now = _utc_stamp()
    inserted = 0
    updated = 0
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        for case in cases:
            exists = conn.execute(
                "SELECT 1 FROM onp_refinement_cases WHERE input_digest = ?",
                (case.input_digest,),
            ).fetchone()
            _upsert_onp_refinement_case(conn, case=case, now=now)
            if exists is None:
                inserted += 1
            else:
                updated += 1
        conn.commit()
    return {
        "store_path": str(db_path),
        "cases_seen": len(cases),
        "cases_inserted": inserted,
        "cases_updated": updated,
    }


def load_onp_refinement_cases(
    path: str | Path,
    *,
    object_ids: tuple[str, ...] = (),
    include_stale: bool = False,
) -> list[OnpRefinementCaseRecord]:
    db_path = Path(path).expanduser().resolve()
    if not db_path.is_file():
        raise FileNotFoundError(f"OEL Scale store does not exist: {db_path}")
    params: list[Any] = []
    where = "WHERE 1 = 1" if include_stale else "WHERE status != 'stale'"
    if object_ids:
        placeholders = ", ".join("?" for _ in object_ids)
        where += f" AND (deputy_id IN ({placeholders}) OR chief_id IN ({placeholders}))"
        params.extend(object_ids)
        params.extend(object_ids)
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            f"""
            SELECT
                case_id, packet_id, packet_input_digest, candidate_id,
                refinement_id, screening_window_id, deputy_id, chief_id,
                force_model_profile, analysis_start_jd_utc, analysis_end_jd_utc,
                initial_state_frame, frame_transform, source_product_ids_json,
                source_state_provenance_json, generated_config_path, config_digest,
                scenario_output_dir, validation_status, validation_report_json,
                materialization_signature, input_digest, status, warnings_json
            FROM onp_refinement_cases
            {where}
            ORDER BY analysis_start_jd_utc ASC, case_id ASC
            """,
            params,
        ).fetchall()
    return [
        OnpRefinementCaseRecord(
            case_id=str(row[0]),
            packet_id=str(row[1]),
            packet_input_digest=str(row[2] or ""),
            candidate_id=str(row[3]),
            refinement_id=str(row[4] or ""),
            screening_window_id=str(row[5] or ""),
            deputy_id=str(row[6]),
            chief_id=str(row[7]),
            force_model_profile=str(row[8] or ""),
            analysis_start_jd_utc=float(row[9]),
            analysis_end_jd_utc=float(row[10]),
            initial_state_frame=str(row[11] or ""),
            frame_transform=str(row[12] or ""),
            source_product_ids=tuple(str(item) for item in _loads(row[13], default=[])),
            source_state_provenance=dict(_loads(row[14], default={})),
            generated_config_path=str(row[15] or ""),
            config_digest=str(row[16] or ""),
            scenario_output_dir=str(row[17] or ""),
            validation_status=str(row[18] or ""),
            validation_report=dict(_loads(row[19], default={})),
            materialization_signature=str(row[20] or ""),
            input_digest=str(row[21] or ""),
            status=str(row[22] or ""),
            warnings=tuple(str(item) for item in _loads(row[23], default=[])),
        )
        for row in rows
    ]


def recover_incomplete_onp_refinement_runs(
    path: str | Path,
    *,
    case_ids: tuple[str, ...] = (),
) -> int:
    """Mark abandoned durable claims as resumable after explicit operator recovery."""

    initialize_store(path, description="OEL Scale operational catalog store")
    db_path = Path(path).expanduser().resolve()
    now = _utc_stamp()
    case_clause = ""
    params: list[Any] = [now, now]
    if case_ids:
        placeholders = ", ".join("?" for _ in case_ids)
        case_clause = f" AND case_id IN ({placeholders})"
        params.extend(case_ids)
    with sqlite3.connect(db_path) as conn:
        cursor = conn.execute(
            f"""
            UPDATE onp_refinement_runs
            SET status = 'interrupted',
                error = CASE
                    WHEN error IS NULL OR error = '' THEN 'Recovered incomplete execution claim.'
                    ELSE error
                END,
                completed_utc = ?, updated_utc = ?
            WHERE status IN ('claimed', 'running')
              {case_clause}
            """,
            params,
        )
        conn.commit()
    return int(cursor.rowcount if cursor.rowcount is not None else 0)


def claim_onp_refinement_runs(
    path: str | Path,
    *,
    cases: list[OnpRefinementCaseRecord],
    batch_run_id: str,
    backend: str,
    retry_failed: bool = False,
    retry_interrupted: bool = False,
    limit: int | None = None,
) -> list[OnpRefinementRunRecord]:
    """Atomically claim validated, non-stale cases that still need execution.

    Thread-pool interrupted attempts are deliberately held until an operator
    explicitly requests recovery.  A current marker from
    :func:`recover_incomplete_onp_refinement_runs` is also an explicit
    recovery decision.  Process-pool claims retain their historical retry
    behavior for compatibility.
    """

    initialize_store(path, description="OEL Scale operational catalog store")
    db_path = Path(path).expanduser().resolve()
    now = _utc_stamp()
    claimed: list[OnpRefinementRunRecord] = []
    maximum = None if limit is None else max(int(limit), 0)
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("BEGIN IMMEDIATE")
        for case in cases:
            if maximum is not None and len(claimed) >= maximum:
                break
            if case.status != "validated" or case.validation_status != "ok":
                continue
            latest = conn.execute(
                """
                SELECT attempt, status, error, backend
                FROM onp_refinement_runs
                WHERE case_id = ? AND config_digest = ?
                ORDER BY attempt DESC
                LIMIT 1
                """,
                (case.case_id, case.config_digest),
            ).fetchone()
            if latest is not None:
                latest_status = str(latest[1] or "")
                if latest_status in {"claimed", "running", "completed"}:
                    continue
                if latest_status == "interrupted":
                    if str(latest[3] or "") == "thread_pool":
                        recovered_marker = str(latest[2] or "").startswith(
                            "Recovered incomplete execution claim."
                        )
                        if not (retry_interrupted or recovered_marker):
                            continue
                if latest_status in {"error", "blocked"} and not retry_failed:
                    continue
                attempt = int(latest[0]) + 1
            else:
                attempt = 1
            execution_id = str(uuid.uuid4())
            input_digest = hashlib.sha256(
                _json(
                    {
                        "backend": backend,
                        "case_id": case.case_id,
                        "config_digest": case.config_digest,
                        "attempt": attempt,
                    }
                ).encode("utf-8")
            ).hexdigest()
            conn.execute(
                """
                INSERT INTO onp_refinement_runs (
                    execution_id, batch_run_id, case_id, config_digest, backend,
                    attempt, status, output_dir, review_store_path,
                    run_summary_json, comparison_json, input_digest, error,
                    warnings_json, claimed_utc, started_utc, completed_utc,
                    updated_utc
                ) VALUES (?, ?, ?, ?, ?, ?, 'claimed', '', '', '{}', '{}', ?, '', '[]', ?, '', '', ?)
                """,
                (
                    execution_id,
                    batch_run_id,
                    case.case_id,
                    case.config_digest,
                    backend,
                    attempt,
                    input_digest,
                    now,
                    now,
                ),
            )
            claimed.append(
                OnpRefinementRunRecord(
                    execution_id=execution_id,
                    batch_run_id=batch_run_id,
                    case_id=case.case_id,
                    config_digest=case.config_digest,
                    backend=backend,
                    attempt=attempt,
                    status="claimed",
                    input_digest=input_digest,
                )
            )
        conn.commit()
    return claimed


def mark_onp_refinement_run_running(path: str | Path, *, execution_id: str) -> None:
    db_path = Path(path).expanduser().resolve()
    now = _utc_stamp()
    with sqlite3.connect(db_path) as conn:
        cursor = conn.execute(
            """
            UPDATE onp_refinement_runs
            SET status = 'running', started_utc = ?, updated_utc = ?
            WHERE execution_id = ? AND status = 'claimed'
            """,
            (now, now, execution_id),
        )
        if cursor.rowcount != 1:
            raise RuntimeError(f"ONP refinement execution claim is no longer runnable: {execution_id}")
        conn.commit()


def finish_onp_refinement_run(
    path: str | Path,
    *,
    execution_id: str,
    status: str,
    backend: str = "",
    output_dir: str = "",
    review_store_path: str = "",
    run_summary: dict[str, Any] | None = None,
    comparison: dict[str, Any] | None = None,
    error: str = "",
    warnings: tuple[str, ...] = (),
) -> None:
    if status not in {"completed", "error", "blocked", "interrupted"}:
        raise ValueError(f"Unsupported ONP refinement terminal status: {status}")
    db_path = Path(path).expanduser().resolve()
    now = _utc_stamp()
    with sqlite3.connect(db_path) as conn:
        cursor = conn.execute(
            """
            UPDATE onp_refinement_runs
            SET status = ?, backend = CASE WHEN ? = '' THEN backend ELSE ? END,
                output_dir = ?, review_store_path = ?,
                run_summary_json = ?, comparison_json = ?, error = ?,
                warnings_json = ?, completed_utc = ?, updated_utc = ?
            WHERE execution_id = ? AND status IN ('claimed', 'running')
            """,
            (
                status,
                backend,
                backend,
                output_dir,
                review_store_path,
                _json(run_summary or {}),
                _json(comparison or {}),
                error,
                _json(list(warnings)),
                now,
                now,
                execution_id,
            ),
        )
        if cursor.rowcount != 1:
            raise RuntimeError(f"ONP refinement execution is not active: {execution_id}")
        conn.commit()


def load_onp_refinement_runs(
    path: str | Path,
    *,
    batch_run_id: str = "",
    case_id: str = "",
) -> list[OnpRefinementRunRecord]:
    db_path = Path(path).expanduser().resolve()
    if not db_path.is_file():
        raise FileNotFoundError(f"OEL Scale store does not exist: {db_path}")
    clauses: list[str] = []
    params: list[Any] = []
    if batch_run_id:
        clauses.append("batch_run_id = ?")
        params.append(batch_run_id)
    if case_id:
        clauses.append("case_id = ?")
        params.append(case_id)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            f"""
            SELECT execution_id, batch_run_id, case_id, config_digest, backend,
                   attempt, status, output_dir, review_store_path,
                   run_summary_json, comparison_json, input_digest, error,
                   warnings_json
            FROM onp_refinement_runs
            {where}
            ORDER BY claimed_utc ASC, execution_id ASC
            """,
            params,
        ).fetchall()
    return [
        OnpRefinementRunRecord(
            execution_id=str(row[0]),
            batch_run_id=str(row[1]),
            case_id=str(row[2]),
            config_digest=str(row[3] or ""),
            backend=str(row[4] or ""),
            attempt=int(row[5]),
            status=str(row[6] or ""),
            output_dir=str(row[7] or ""),
            review_store_path=str(row[8] or ""),
            run_summary=dict(_loads(row[9], default={})),
            comparison=dict(_loads(row[10], default={})),
            input_digest=str(row[11] or ""),
            error=str(row[12] or ""),
            warnings=tuple(str(item) for item in _loads(row[13], default=[])),
        )
        for row in rows
    ]


def summarize_onp_refinement_execution_queue(
    path: str | Path,
    *,
    object_ids: tuple[str, ...] = (),
) -> dict[str, int]:
    cases = [
        case
        for case in load_onp_refinement_cases(path, object_ids=object_ids)
        if case.status == "validated" and case.validation_status == "ok"
    ]
    latest_by_case: dict[str, OnpRefinementRunRecord] = {}
    for run in load_onp_refinement_runs(path):
        case = latest_by_case.get(run.case_id)
        if case is None or run.attempt > case.attempt:
            latest_by_case[run.case_id] = run
    counts = {
        "validated_case_count": len(cases),
        "pending_execution_count": 0,
        "active_execution_count": 0,
        "completed_execution_count": 0,
        "retryable_failed_count": 0,
    }
    for case in cases:
        latest = latest_by_case.get(case.case_id)
        if latest is None or latest.status == "interrupted":
            counts["pending_execution_count"] += 1
        elif latest.status in {"claimed", "running"}:
            counts["active_execution_count"] += 1
        elif latest.status == "completed":
            counts["completed_execution_count"] += 1
        elif latest.status in {"error", "blocked"}:
            counts["retryable_failed_count"] += 1
    return counts
def _insert_candidate_refinement(
    conn: sqlite3.Connection,
    *,
    refinement: CandidateRefinementRecord,
    now: str,
) -> bool:
    try:
        conn.execute(
            """
            INSERT INTO candidate_refinements (
                refinement_id, candidate_id, screening_run_id, screening_window_id,
                deputy_id, chief_id, method, source_product_ids_json, input_digest,
                sampled_tca_jd_utc, sampled_miss_distance_km,
                sampled_relative_speed_km_s, refinement_start_jd_utc,
                refinement_end_jd_utc, refinement_step_s, refined_tca_jd_utc,
                refined_miss_distance_km, refined_relative_speed_km_s,
                ranking_score, ranking_features_json, status, warnings_json,
                created_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                refinement.refinement_id,
                refinement.candidate_id,
                refinement.screening_run_id,
                refinement.screening_window_id,
                refinement.deputy_id,
                refinement.chief_id,
                refinement.method,
                _json(list(refinement.source_product_ids)),
                refinement.input_digest,
                refinement.sampled_tca_jd_utc,
                refinement.sampled_miss_distance_km,
                refinement.sampled_relative_speed_km_s,
                refinement.refinement_start_jd_utc,
                refinement.refinement_end_jd_utc,
                refinement.refinement_step_s,
                refinement.refined_tca_jd_utc,
                refinement.refined_miss_distance_km,
                refinement.refined_relative_speed_km_s,
                refinement.ranking_score,
                _json(refinement.ranking_features),
                refinement.status,
                _json(list(refinement.warnings)),
                now,
            ),
        )
        return True
    except sqlite3.IntegrityError:
        return False


def _update_candidate_with_refinement(conn: sqlite3.Connection, *, refinement: CandidateRefinementRecord) -> None:
    conn.execute(
        """
        UPDATE conjunction_candidates
        SET tca_jd_utc = ?,
            miss_distance_km = ?,
            relative_speed_km_s = ?,
            ranking_score = ?,
            ranking_features_json = ?,
            promotion_status = ?
        WHERE candidate_id = ?
        """,
        (
            refinement.refined_tca_jd_utc,
            refinement.refined_miss_distance_km,
            refinement.refined_relative_speed_km_s,
            refinement.ranking_score,
            _json(refinement.ranking_features),
            "refined_for_hpop" if refinement.status == "completed" else refinement.status,
            refinement.candidate_id,
        ),
    )


def _insert_hpop_handoff_packet(
    conn: sqlite3.Connection,
    *,
    packet: HpopHandoffPacketRecord,
    now: str,
) -> bool:
    try:
        conn.execute(
            """
            INSERT INTO hpop_handoff_packets (
                packet_id, candidate_id, refinement_id, screening_window_id,
                deputy_id, chief_id, force_model_profile, case_start_jd_utc,
                case_end_jd_utc, tca_jd_utc, miss_distance_km,
                relative_speed_km_s, ranking_score, output_dir, packet_path,
                packet_json, input_digest, status, warnings_json, created_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                packet.packet_id,
                packet.candidate_id,
                packet.refinement_id,
                packet.screening_window_id,
                packet.deputy_id,
                packet.chief_id,
                packet.force_model_profile,
                packet.case_start_jd_utc,
                packet.case_end_jd_utc,
                packet.tca_jd_utc,
                packet.miss_distance_km,
                packet.relative_speed_km_s,
                packet.ranking_score,
                packet.output_dir,
                packet.packet_path,
                _json(packet.packet_json),
                packet.input_digest,
                packet.status,
                _json(list(packet.warnings)),
                now,
            ),
        )
        return True
    except sqlite3.IntegrityError:
        return False


def _upsert_onp_refinement_case(
    conn: sqlite3.Connection,
    *,
    case: OnpRefinementCaseRecord,
    now: str,
) -> None:
    conn.execute(
        """
        INSERT INTO onp_refinement_cases (
            case_id, packet_id, packet_input_digest, candidate_id,
            refinement_id, screening_window_id, deputy_id, chief_id,
            force_model_profile, analysis_start_jd_utc, analysis_end_jd_utc,
            initial_state_frame, frame_transform, source_product_ids_json,
            source_state_provenance_json, generated_config_path, config_digest,
            scenario_output_dir, validation_status, validation_report_json,
            materialization_signature, input_digest, status, warnings_json, created_utc, updated_utc
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(input_digest) DO UPDATE SET
            generated_config_path = excluded.generated_config_path,
            config_digest = excluded.config_digest,
            scenario_output_dir = excluded.scenario_output_dir,
            validation_status = excluded.validation_status,
            validation_report_json = excluded.validation_report_json,
            materialization_signature = excluded.materialization_signature,
            status = excluded.status,
            warnings_json = excluded.warnings_json,
            updated_utc = excluded.updated_utc
        """,
        (
            case.case_id,
            case.packet_id,
            case.packet_input_digest,
            case.candidate_id,
            case.refinement_id,
            case.screening_window_id,
            case.deputy_id,
            case.chief_id,
            case.force_model_profile,
            case.analysis_start_jd_utc,
            case.analysis_end_jd_utc,
            case.initial_state_frame,
            case.frame_transform,
            _json(list(case.source_product_ids)),
            _json(case.source_state_provenance),
            case.generated_config_path,
            case.config_digest,
            case.scenario_output_dir,
            case.validation_status,
            _json(case.validation_report),
            case.materialization_signature,
            case.input_digest,
            case.status,
            _json(list(case.warnings)),
            now,
            now,
        ),
    )
