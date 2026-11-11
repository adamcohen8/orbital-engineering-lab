from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from pathlib import Path
from typing import Any

from sim.scale.store_catalog import (
    _ensure_catalog_object,
)
from sim.scale.store_connection import (
    _float_or_none,
    _json,
    _loads,
    _utc_stamp,
)
from sim.scale.store_models import (
    GroundStationMeasurementIngestResult,
    GroundStationMeasurementQualityRecord,
    GroundStationMeasurementRecord,
    TrackObservationIngestResult,
    TrackObservationRecord,
)
from sim.scale.store_schema import (
    initialize_store,
)


def ingest_track_observation_records(
    path: str | Path,
    *,
    records: list[TrackObservationRecord],
    source_label: str,
    source_digest: str,
    normalized_by: str = "sim.scale.ingest_track_observations",
) -> TrackObservationIngestResult:
    store_info = initialize_store(path, description="OEL Scale operational catalog store")
    db_path = Path(store_info["path"])
    batch_id = f"track_batch:{uuid.uuid4()}"
    now = _utc_stamp()
    objects_created = 0
    observations_inserted = 0
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
                "track_observations",
                source_digest,
                normalized_by,
                now,
                _json({"record_count": len(records)}),
                _json([]),
            ),
        )
        for record in records:
            if _ensure_catalog_object(conn, object_id=record.object_id, now=now, source="track_observation_ingest"):
                objects_created += 1
            observation_id = f"obs:{record.digest or _track_observation_digest(record)}"
            inserted = _insert_track_observation_record(
                conn,
                record=record,
                batch_id=batch_id,
                observation_id=observation_id,
                source_label=source_label,
                now=now,
            )
            if not inserted:
                continue
            observations_inserted += 1
            invalidations_inserted += _insert_observation_invalidation(
                conn,
                object_id=record.object_id,
                time_jd_utc=record.time_jd_utc,
                stale_product_id=observation_id,
                now=now,
            )
        conn.commit()
    return TrackObservationIngestResult(
        store_path=db_path,
        batch_id=batch_id,
        source_label=source_label,
        source_digest=source_digest,
        records_seen=len(records),
        objects_created=objects_created,
        observations_inserted=observations_inserted,
        invalidations_inserted=invalidations_inserted,
    )


def ingest_ground_station_measurement_records(
    path: str | Path,
    *,
    records: list[GroundStationMeasurementRecord],
    source_label: str,
    source_digest: str,
    normalized_by: str = "sim.scale.ingest_ground_station_measurements",
) -> GroundStationMeasurementIngestResult:
    store_info = initialize_store(path, description="OEL Scale operational catalog store")
    db_path = Path(store_info["path"])
    batch_id = f"ground_measurement_batch:{uuid.uuid4()}"
    now = _utc_stamp()
    objects_created = 0
    stations_recorded = 0
    measurements_inserted = 0
    invalidations_inserted = 0
    seen_stations: set[str] = set()
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
                "ground_station_measurements",
                source_digest,
                normalized_by,
                now,
                _json({"record_count": len(records)}),
                _json([]),
            ),
        )
        for record in records:
            if _ensure_catalog_object(
                conn, object_id=record.object_id, now=now, source="ground_station_measurement_ingest"
            ):
                objects_created += 1
            station_key = f"{batch_id}:{record.station_id}"
            if station_key not in seen_stations:
                if _insert_ground_station_measurement_station(conn, record=record, batch_id=batch_id, now=now):
                    stations_recorded += 1
                seen_stations.add(station_key)
            measurement_id = f"ground_measurement:{record.digest or _ground_station_measurement_digest(record)}"
            inserted = _insert_ground_station_measurement_record(
                conn,
                record=record,
                batch_id=batch_id,
                measurement_id=measurement_id,
                source_label=source_label,
                now=now,
            )
            if not inserted:
                continue
            measurements_inserted += 1
            invalidations_inserted += _insert_observation_invalidation(
                conn,
                object_id=record.object_id,
                time_jd_utc=record.time_jd_utc,
                stale_product_id=measurement_id,
                stale_product_type="ground_station_measurement",
                now=now,
            )
        conn.commit()
    return GroundStationMeasurementIngestResult(
        store_path=db_path,
        batch_id=batch_id,
        source_label=source_label,
        source_digest=source_digest,
        records_seen=len(records),
        objects_created=objects_created,
        stations_recorded=stations_recorded,
        measurements_inserted=measurements_inserted,
        invalidations_inserted=invalidations_inserted,
    )
def load_track_observation_rows(
    path: str | Path,
    *,
    object_ids: tuple[str, ...] = (),
    frame: str = "teme",
    min_observations: int = 2,
) -> dict[str, list[dict[str, Any]]]:
    db_path = Path(path).expanduser().resolve()
    if not db_path.is_file():
        raise FileNotFoundError(f"OEL Scale store does not exist: {db_path}")
    params: list[Any] = [str(frame).lower()]
    object_where = ""
    if object_ids:
        placeholders = ", ".join("?" for _ in object_ids)
        object_where = f"AND object_id IN ({placeholders})"
        params.extend(object_ids)
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            f"""
            SELECT observation_id, object_id, time_jd_utc, frame,
                   position_x_km, position_y_km, position_z_km,
                   velocity_x_km_s, velocity_y_km_s, velocity_z_km_s,
                   covariance_json, quality_flags_json, digest
            FROM track_observations
            WHERE lower(frame) = ?
              AND position_x_km IS NOT NULL
              AND position_y_km IS NOT NULL
              AND position_z_km IS NOT NULL
              AND velocity_x_km_s IS NOT NULL
              AND velocity_y_km_s IS NOT NULL
              AND velocity_z_km_s IS NOT NULL
              {object_where}
            ORDER BY object_id, time_jd_utc, observation_id
            """,
            params,
        ).fetchall()
    by_object: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_object.setdefault(str(row[1]), []).append(
            {
                "observation_id": str(row[0]),
                "object_id": str(row[1]),
                "time_jd_utc": float(row[2]),
                "frame": str(row[3]),
                "position_teme_km": [float(row[4]), float(row[5]), float(row[6])],
                "velocity_teme_km_s": [float(row[7]), float(row[8]), float(row[9])],
                "covariance": _loads(row[10], default={}),
                "quality_flags": _loads(row[11], default={}),
                "digest": str(row[12] or ""),
            }
        )
    minimum = max(int(min_observations), 0)
    return {object_id: items for object_id, items in by_object.items() if len(items) >= minimum}


def load_ground_station_measurement_rows(
    path: str | Path,
    *,
    object_ids: tuple[str, ...] = (),
    station_ids: tuple[str, ...] = (),
    min_measurements: int = 1,
) -> dict[str, list[dict[str, Any]]]:
    db_path = Path(path).expanduser().resolve()
    if not db_path.is_file():
        raise FileNotFoundError(f"OEL Scale store does not exist: {db_path}")
    params: list[Any] = []
    where = "WHERE 1 = 1"
    if object_ids:
        placeholders = ", ".join("?" for _ in object_ids)
        where += f" AND gsm.object_id IN ({placeholders})"
        params.extend(object_ids)
    if station_ids:
        placeholders = ", ".join("?" for _ in station_ids)
        where += f" AND gsm.station_id IN ({placeholders})"
        params.extend(station_ids)
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            f"""
            SELECT
                gsm.measurement_id,
                gsm.object_id,
                gsm.station_id,
                gsm.batch_id,
                gsm.time_jd_utc,
                gsm.measurement_type,
                gsm.components_json,
                gsm.vector_json,
                gsm.sigma_json,
                gs.station_metadata_json,
                gsm.quality_flags_json,
                gsm.truth_geometry_json,
                gsm.source_label,
                gsm.digest
            FROM ground_station_measurements gsm
            LEFT JOIN ground_station_measurement_stations gs
              ON gs.batch_id = gsm.batch_id
             AND gs.station_id = gsm.station_id
            {where}
            ORDER BY gsm.object_id, gsm.station_id, gsm.time_jd_utc, gsm.measurement_id
            """,
            params,
        ).fetchall()
    by_object: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_object.setdefault(str(row[1]), []).append(
            {
                "measurement_id": str(row[0]),
                "object_id": str(row[1]),
                "station_id": str(row[2]),
                "batch_id": str(row[3] or ""),
                "time_jd_utc": float(row[4]),
                "measurement_type": str(row[5] or ""),
                "components": [str(item) for item in _loads(row[6], default=[])],
                "vector": [float(item) for item in _loads(row[7], default=[])],
                "sigma": [float(item) for item in _loads(row[8], default=[])],
                "station_metadata": dict(_loads(row[9], default={})),
                "quality_flags": dict(_loads(row[10], default={})),
                "truth_geometry": dict(_loads(row[11], default={})),
                "source_label": str(row[12] or ""),
                "digest": str(row[13] or ""),
            }
        )
    minimum = max(int(min_measurements), 0)
    return {object_id: items for object_id, items in by_object.items() if len(items) >= minimum}


def write_ground_station_measurement_quality_records(
    path: str | Path,
    *,
    records: list[GroundStationMeasurementQualityRecord],
) -> dict[str, Any]:
    initialize_store(path, description="OEL Scale operational catalog store")
    db_path = Path(path).expanduser().resolve()
    now = _utc_stamp()
    inserted = 0
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        for record in records:
            if _insert_ground_station_measurement_quality_record(conn, record=record, now=now):
                inserted += 1
        conn.commit()
    return {
        "store_path": str(db_path),
        "quality_records_seen": len(records),
        "quality_records_inserted": inserted,
    }


def load_ground_station_measurement_quality_rows(
    path: str | Path,
    *,
    assessment_id: str = "",
    object_ids: tuple[str, ...] = (),
    station_ids: tuple[str, ...] = (),
) -> list[dict[str, Any]]:
    db_path = Path(path).expanduser().resolve()
    if not db_path.is_file():
        raise FileNotFoundError(f"OEL Scale store does not exist: {db_path}")
    params: list[Any] = []
    where = "WHERE 1 = 1"
    if assessment_id:
        where += " AND assessment_id = ?"
        params.append(str(assessment_id))
    if object_ids:
        placeholders = ", ".join("?" for _ in object_ids)
        where += f" AND object_id IN ({placeholders})"
        params.extend(object_ids)
    if station_ids:
        placeholders = ", ".join("?" for _ in station_ids)
        where += f" AND station_id IN ({placeholders})"
        params.extend(station_ids)
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            f"""
            SELECT
                quality_id, assessment_id, measurement_id, object_id, station_id,
                od_manifest_path, od_report_json_path, residual_kind, time_s,
                position_error_m, normalized_position_error, status, flags_json,
                metrics_json, thresholds_json, created_utc
            FROM ground_station_measurement_quality
            {where}
            ORDER BY object_id, station_id, time_s, measurement_id, quality_id
            """,
            params,
        ).fetchall()
    return [
        {
            "quality_id": str(row[0]),
            "assessment_id": str(row[1]),
            "measurement_id": str(row[2]),
            "object_id": str(row[3]),
            "station_id": str(row[4]),
            "od_manifest_path": str(row[5] or ""),
            "od_report_json_path": str(row[6] or ""),
            "residual_kind": str(row[7] or ""),
            "time_s": float(row[8]),
            "position_error_m": float(row[9]),
            "normalized_position_error": _float_or_none(row[10]),
            "status": str(row[11] or ""),
            "flags": [str(item) for item in _loads(row[12], default=[])],
            "metrics": dict(_loads(row[13], default={})),
            "thresholds": dict(_loads(row[14], default={})),
            "created_utc": str(row[15] or ""),
        }
        for row in rows
    ]
def _insert_track_observation_record(
    conn: sqlite3.Connection,
    *,
    record: TrackObservationRecord,
    batch_id: str,
    observation_id: str,
    source_label: str,
    now: str,
) -> bool:
    pos = record.position_km
    vel = record.velocity_km_s
    digest = record.digest or _track_observation_digest(record)
    try:
        conn.execute(
            """
            INSERT INTO track_observations (
                observation_id, object_id, batch_id, time_jd_utc, frame,
                position_x_km, position_y_km, position_z_km,
                velocity_x_km_s, velocity_y_km_s, velocity_z_km_s,
                covariance_json, quality_flags_json, source_label, digest,
                created_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                observation_id,
                record.object_id,
                batch_id,
                float(record.time_jd_utc),
                str(record.frame or ""),
                None if pos is None else float(pos[0]),
                None if pos is None else float(pos[1]),
                None if pos is None else float(pos[2]),
                None if vel is None else float(vel[0]),
                None if vel is None else float(vel[1]),
                None if vel is None else float(vel[2]),
                _json(record.covariance or {}),
                _json(record.quality_flags or {}),
                source_label,
                digest,
                now,
            ),
        )
        return True
    except sqlite3.IntegrityError:
        return False


def _insert_ground_station_measurement_station(
    conn: sqlite3.Connection,
    *,
    record: GroundStationMeasurementRecord,
    batch_id: str,
    now: str,
) -> bool:
    try:
        conn.execute(
            """
            INSERT INTO ground_station_measurement_stations (
                batch_id, station_id, station_metadata_json, source_label, created_utc
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                batch_id,
                record.station_id,
                _json(record.station_metadata or {}),
                str((record.station_metadata or {}).get("source_label", "")),
                now,
            ),
        )
        return True
    except sqlite3.IntegrityError:
        return False


def _insert_ground_station_measurement_record(
    conn: sqlite3.Connection,
    *,
    record: GroundStationMeasurementRecord,
    batch_id: str,
    measurement_id: str,
    source_label: str,
    now: str,
) -> bool:
    digest = record.digest or _ground_station_measurement_digest(record)
    try:
        conn.execute(
            """
            INSERT INTO ground_station_measurements (
                measurement_id, object_id, station_id, batch_id, time_jd_utc,
                measurement_type, components_json, vector_json, sigma_json,
                quality_flags_json, truth_geometry_json, source_label, digest,
                created_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                measurement_id,
                record.object_id,
                record.station_id,
                batch_id,
                float(record.time_jd_utc),
                str(record.measurement_type or ""),
                _json(list(record.components)),
                _json([float(item) for item in record.vector]),
                _json([float(item) for item in record.sigma]),
                _json(record.quality_flags or {}),
                _json(record.truth_geometry or {}),
                source_label,
                digest,
                now,
            ),
        )
        return True
    except sqlite3.IntegrityError:
        return False


def _insert_ground_station_measurement_quality_record(
    conn: sqlite3.Connection,
    *,
    record: GroundStationMeasurementQualityRecord,
    now: str,
) -> bool:
    try:
        conn.execute(
            """
            INSERT INTO ground_station_measurement_quality (
                quality_id, assessment_id, measurement_id, object_id, station_id,
                od_manifest_path, od_report_json_path, residual_kind, time_s,
                position_error_m, normalized_position_error, status, flags_json,
                metrics_json, thresholds_json, created_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.quality_id,
                record.assessment_id,
                record.measurement_id,
                record.object_id,
                record.station_id,
                record.od_manifest_path,
                record.od_report_json_path,
                record.residual_kind,
                float(record.time_s),
                float(record.position_error_m),
                None if record.normalized_position_error is None else float(record.normalized_position_error),
                record.status,
                _json(list(record.flags)),
                _json(record.metrics),
                _json(record.thresholds),
                now,
            ),
        )
        return True
    except sqlite3.IntegrityError:
        return False
def _insert_observation_invalidation(
    conn: sqlite3.Connection,
    *,
    object_id: str,
    time_jd_utc: float,
    stale_product_id: str,
    stale_product_type: str = "track_observation",
    now: str,
) -> int:
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
            f"invalidation:new_observation:{stale_product_id}:{uuid.uuid4()}",
            object_id,
            float(time_jd_utc),
            stale_product_id,
            stale_product_type,
            "new_observation",
            95.0,
            "open",
            now,
        ),
    )
    return 1


def _insert_od_solution_changed_invalidation(
    conn: sqlite3.Connection,
    *,
    object_id: str,
    epoch_jd_utc: float,
    stale_product_id: str,
    now: str,
) -> int:
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
            f"invalidation:od_solution_changed:{stale_product_id}:{uuid.uuid4()}",
            object_id,
            float(epoch_jd_utc),
            stale_product_id,
            "od_solution",
            "od_solution_changed",
            90.0,
            "open",
            now,
        ),
    )
    return 1


def _complete_observation_invalidations_for_object(
    conn: sqlite3.Connection,
    *,
    object_id: str,
    now: str,
) -> int:
    cur = conn.execute(
        """
        UPDATE invalidation_queue
        SET status = 'completed',
            completed_utc = ?
        WHERE object_id = ?
          AND reason = 'new_observation'
          AND status IN ('open', 'planned', 'claimed')
        """,
        (now, object_id),
    )
    return int(cur.rowcount if cur.rowcount is not None else 0)
def _track_observation_digest(record: TrackObservationRecord) -> str:
    payload = {
        "object_id": record.object_id,
        "time_jd_utc": float(record.time_jd_utc),
        "frame": record.frame,
        "position_km": list(record.position_km) if record.position_km is not None else None,
        "velocity_km_s": list(record.velocity_km_s) if record.velocity_km_s is not None else None,
        "covariance": record.covariance or {},
        "quality_flags": record.quality_flags or {},
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _ground_station_measurement_digest(record: GroundStationMeasurementRecord) -> str:
    payload = {
        "object_id": record.object_id,
        "station_id": record.station_id,
        "time_jd_utc": float(record.time_jd_utc),
        "measurement_type": record.measurement_type,
        "components": list(record.components),
        "vector": [float(item) for item in record.vector],
        "sigma": [float(item) for item in record.sigma],
        "station_metadata": record.station_metadata or {},
        "quality_flags": record.quality_flags or {},
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
