from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from sim.scale.store_connection import (
    _float_or_none,
    _json,
    _loads,
    _utc_stamp,
)
from sim.scale.store_models import (
    PropagationProductRecord,
    PropagationProductSummary,
    PropagationSampleRecord,
)
from sim.scale.store_schema import (
    initialize_store,
)


def write_propagation_products(
    path: str | Path,
    *,
    products: list[PropagationProductRecord],
    complete_invalidations: bool = True,
) -> dict[str, Any]:
    initialize_store(path, description="OEL Scale operational catalog store")
    db_path = Path(path).expanduser().resolve()
    now = _utc_stamp()
    products_inserted = 0
    samples_inserted = 0
    objects_completed: set[str] = set()
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        for product in products:
            inserted = _insert_propagation_product(conn, product=product, now=now)
            if not inserted:
                continue
            products_inserted += 1
            samples_inserted += _insert_propagation_samples(conn, product=product)
            if product.status == "completed":
                objects_completed.add(product.object_id)
        invalidations_completed = (
            _complete_propagation_invalidations(conn, object_ids=objects_completed, now=now)
            if complete_invalidations
            else 0
        )
        conn.commit()
    return {
        "store_path": str(db_path),
        "products_seen": len(products),
        "products_inserted": products_inserted,
        "samples_inserted": samples_inserted,
        "objects_completed": len(objects_completed),
        "invalidations_completed": invalidations_completed,
    }


def complete_propagation_invalidations(path: str | Path, *, object_ids: tuple[str, ...]) -> int:
    initialize_store(path, description="OEL Scale operational catalog store")
    db_path = Path(path).expanduser().resolve()
    now = _utc_stamp()
    with sqlite3.connect(db_path) as conn:
        completed = _complete_propagation_invalidations(conn, object_ids=set(object_ids), now=now)
        conn.commit()
    return completed


def load_completed_propagation_products(
    path: str | Path,
    *,
    object_ids: tuple[str, ...] = (),
) -> list[PropagationProductSummary]:
    db_path = Path(path).expanduser().resolve()
    if not db_path.is_file():
        raise FileNotFoundError(f"OEL Scale store does not exist: {db_path}")
    params: list[Any] = []
    where = "WHERE status = 'completed'"
    if object_ids:
        placeholders = ", ".join("?" for _ in object_ids)
        where += f" AND object_id IN ({placeholders})"
        params.extend(object_ids)
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            f"""
            SELECT
                propagation_product_id, object_id, model, backend, output_frame,
                start_jd_utc, end_jd_utc, step_s, sample_count, input_digest,
                ogp_regime, runtime_backend, metadata_json
            FROM propagation_products
            {where}
            ORDER BY start_jd_utc, end_jd_utc, object_id, propagation_product_id
            """,
            params,
        ).fetchall()
    return [
        PropagationProductSummary(
            propagation_product_id=str(row[0]),
            object_id=str(row[1]),
            model=str(row[2] or ""),
            backend=str(row[3] or ""),
            output_frame=str(row[4] or ""),
            start_jd_utc=float(row[5]),
            end_jd_utc=float(row[6]),
            step_s=float(row[7]),
            sample_count=int(row[8] or 0),
            input_digest=str(row[9] or ""),
            ogp_regime=str(row[10] or ""),
            runtime_backend=str(row[11] or ""),
            metadata=dict(_loads(row[12], default={})),
        )
        for row in rows
    ]


def load_product_samples(path: str | Path, *, propagation_product_id: str) -> list[PropagationSampleRecord]:
    db_path = Path(path).expanduser().resolve()
    if not db_path.is_file():
        raise FileNotFoundError(f"OEL Scale store does not exist: {db_path}")
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT
                sample_index, jd_utc, t_s, pos_x_km, pos_y_km, pos_z_km,
                vel_x_km_s, vel_y_km_s, vel_z_km_s, error
            FROM propagation_samples
            WHERE propagation_product_id = ?
            ORDER BY sample_index
            """,
            (propagation_product_id,),
        ).fetchall()
    return [_propagation_sample_from_row(row) for row in rows]


def load_product_samples_by_id(
    path: str | Path,
    *,
    propagation_product_ids: tuple[str, ...],
) -> dict[str, list[PropagationSampleRecord]]:
    """Load a product set in one connection using ordered bind-safe chunks."""

    product_ids = tuple(dict.fromkeys(str(value) for value in propagation_product_ids))
    if not product_ids:
        return {}
    db_path = Path(path).expanduser().resolve()
    if not db_path.is_file():
        raise FileNotFoundError(f"OEL Scale store does not exist: {db_path}")
    with sqlite3.connect(db_path) as conn:
        bind_limit = (
            int(conn.getlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER))
            if hasattr(conn, "getlimit")
            else 999
        )
        chunk_size = max(1, min(bind_limit, 900))
        rows: list[tuple[Any, ...]] = []
        for start in range(0, len(product_ids), chunk_size):
            chunk = product_ids[start : start + chunk_size]
            placeholders = ", ".join("?" for _ in chunk)
            rows.extend(
                conn.execute(
                    f"""
                    SELECT
                        propagation_product_id,
                        sample_index, jd_utc, t_s, pos_x_km, pos_y_km, pos_z_km,
                        vel_x_km_s, vel_y_km_s, vel_z_km_s, error
                    FROM propagation_samples
                    WHERE propagation_product_id IN ({placeholders})
                    ORDER BY propagation_product_id, sample_index
                    """,
                    chunk,
                ).fetchall()
            )
    result = {product_id: [] for product_id in product_ids}
    for row in rows:
        result[str(row[0])].append(_propagation_sample_from_row(row[1:]))
    return result


def _propagation_sample_from_row(row: tuple[Any, ...]) -> PropagationSampleRecord:
    return PropagationSampleRecord(
        sample_index=int(row[0]),
        jd_utc=float(row[1]),
        t_s=float(row[2]),
        pos_x_km=_float_or_none(row[3]),
        pos_y_km=_float_or_none(row[4]),
        pos_z_km=_float_or_none(row[5]),
        vel_x_km_s=_float_or_none(row[6]),
        vel_y_km_s=_float_or_none(row[7]),
        vel_z_km_s=_float_or_none(row[8]),
        error=str(row[9] or ""),
    )


def estimate_pair_window_count(path: str | Path, *, object_ids: tuple[str, ...] = ()) -> int:
    products = load_completed_propagation_products(path, object_ids=object_ids)
    by_window: dict[tuple[float, float, float], set[str]] = {}
    for product in products:
        key = (product.start_jd_utc, product.end_jd_utc, product.step_s)
        by_window.setdefault(key, set()).add(product.object_id)
    return sum((len(objects) * (len(objects) - 1)) // 2 for objects in by_window.values())


def load_matching_current_propagation_products(
    path: str | Path,
    *,
    object_ids: tuple[str, ...] = (),
    windows: tuple[tuple[float, float, float], ...] = (),
    model: str = "sgp4",
    backend: str = "scalar",
    output_frame: str = "teme",
) -> list[PropagationProductSummary]:
    db_path = Path(path).expanduser().resolve()
    if not db_path.is_file():
        raise FileNotFoundError(f"OEL Scale store does not exist: {db_path}")
    params: list[Any] = [model, backend, output_frame]
    where = """
        WHERE pp.status = 'completed'
          AND pp.model = ?
          AND pp.backend = ?
          AND pp.output_frame = ?
          AND pp.source_od_solution_id = cos.od_solution_id
    """
    if object_ids:
        placeholders = ", ".join("?" for _ in object_ids)
        where += f" AND pp.object_id IN ({placeholders})"
        params.extend(object_ids)
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            f"""
            SELECT
                pp.propagation_product_id,
                pp.object_id,
                pp.model,
                pp.backend,
                pp.output_frame,
                pp.start_jd_utc,
                pp.end_jd_utc,
                pp.step_s,
                pp.sample_count,
                pp.input_digest,
                pp.ogp_regime,
                pp.runtime_backend,
                pp.metadata_json
            FROM propagation_products pp
            JOIN current_object_state cos ON cos.object_id = pp.object_id
            {where}
            ORDER BY pp.start_jd_utc, pp.end_jd_utc, pp.object_id, pp.propagation_product_id
            """,
            params,
        ).fetchall()
    product_rows = [
        PropagationProductSummary(
            propagation_product_id=str(row[0]),
            object_id=str(row[1]),
            model=str(row[2] or ""),
            backend=str(row[3] or ""),
            output_frame=str(row[4] or ""),
            start_jd_utc=float(row[5]),
            end_jd_utc=float(row[6]),
            step_s=float(row[7]),
            sample_count=int(row[8] or 0),
            input_digest=str(row[9] or ""),
            ogp_regime=str(row[10] or ""),
            runtime_backend=str(row[11] or ""),
            metadata=dict(_loads(row[12], default={})),
        )
        for row in rows
    ]
    if not windows:
        return product_rows
    expected = {(float(start), float(end), float(step)) for start, end, step in windows}
    return [
        product for product in product_rows if (product.start_jd_utc, product.end_jd_utc, product.step_s) in expected
    ]


def count_matching_current_propagation_products(
    path: str | Path,
    *,
    object_ids: tuple[str, ...] = (),
    windows: tuple[tuple[float, float, float], ...] = (),
    model: str = "sgp4",
    backend: str = "scalar",
    output_frame: str = "teme",
) -> int:
    products = load_matching_current_propagation_products(
        path,
        object_ids=object_ids,
        windows=windows,
        model=model,
        backend=backend,
        output_frame=output_frame,
    )
    return len({(product.object_id, product.start_jd_utc, product.end_jd_utc, product.step_s) for product in products})


def estimate_matching_pair_window_count(
    path: str | Path,
    *,
    object_ids: tuple[str, ...] = (),
    windows: tuple[tuple[float, float, float], ...] = (),
    model: str = "sgp4",
    backend: str = "scalar",
    output_frame: str = "teme",
) -> int:
    products = load_matching_current_propagation_products(
        path,
        object_ids=object_ids,
        windows=windows,
        model=model,
        backend=backend,
        output_frame=output_frame,
    )
    by_window: dict[tuple[float, float, float], set[str]] = {}
    for product in products:
        key = (product.start_jd_utc, product.end_jd_utc, product.step_s)
        by_window.setdefault(key, set()).add(product.object_id)
    return sum((len(objects) * (len(objects) - 1)) // 2 for objects in by_window.values())


def count_screening_windows_for_current_products(
    path: str | Path,
    *,
    object_ids: tuple[str, ...] = (),
    windows: tuple[tuple[float, float, float], ...] = (),
    model: str = "sgp4",
    backend: str = "scalar",
    output_frame: str = "teme",
) -> int:
    matching_products = load_matching_current_propagation_products(
        path,
        object_ids=object_ids,
        windows=windows,
        model=model,
        backend=backend,
        output_frame=output_frame,
    )
    matching_product_ids = {product.propagation_product_id for product in matching_products}
    expected_windows = {(float(start), float(end)) for start, end, _ in windows}
    if not matching_product_ids:
        return 0
    db_path = Path(path).expanduser().resolve()
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT screening_window_id, window_start_jd_utc, window_end_jd_utc, source_product_ids_json
            FROM screening_windows
            """
        ).fetchall()
    matching_window_ids: set[str] = set()
    for row in rows:
        if expected_windows and (float(row[1]), float(row[2])) not in expected_windows:
            continue
        source_product_ids = {str(item) for item in _loads(row[3], default=[])}
        if source_product_ids and source_product_ids.issubset(matching_product_ids):
            matching_window_ids.add(str(row[0]))
    return len(matching_window_ids)
def _insert_propagation_product(
    conn: sqlite3.Connection,
    *,
    product: PropagationProductRecord,
    now: str,
) -> bool:
    existing = conn.execute(
        "SELECT 1 FROM propagation_products WHERE propagation_product_id = ? OR input_digest = ? LIMIT 1",
        (product.propagation_product_id, product.input_digest),
    ).fetchone()
    if existing is not None:
        return False
    conn.execute(
        """
            INSERT INTO propagation_products (
                propagation_product_id, object_id, source_od_solution_id,
                model, backend, ogp_regime, runtime_backend, precision, native_frame, output_frame,
                frame_transform, start_jd_utc, end_jd_utc, step_s,
                sample_count, input_digest, storage_kind, storage_uri,
                storage_digest, shape_json, metadata_json, status, error, created_utc,
                supersedes_product_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
        (
            product.propagation_product_id,
            product.object_id,
            product.source_od_solution_id,
            product.model,
            product.backend,
            product.ogp_regime,
            product.runtime_backend,
            product.precision,
            product.native_frame,
            product.output_frame,
            product.frame_transform,
            product.start_jd_utc,
            product.end_jd_utc,
            product.step_s,
            product.sample_count,
            product.input_digest,
            "sqlite",
            "",
            "",
            _json({"rows": product.sample_count, "columns": 8}),
            _json(product.metadata),
            product.status,
            product.error,
            now,
            _latest_propagation_product_for_object(conn, product.object_id, product.start_jd_utc, product.end_jd_utc),
        ),
    )
    return True


def _insert_propagation_samples(conn: sqlite3.Connection, *, product: PropagationProductRecord) -> int:
    rows = [
        (
            product.propagation_product_id,
            int(sample["sample_index"]),
            float(sample["jd_utc"]),
            float(sample["t_s"]),
            _float_or_none(sample.get("pos_x_km")),
            _float_or_none(sample.get("pos_y_km")),
            _float_or_none(sample.get("pos_z_km")),
            _float_or_none(sample.get("vel_x_km_s")),
            _float_or_none(sample.get("vel_y_km_s")),
            _float_or_none(sample.get("vel_z_km_s")),
            str(sample.get("error", "") or ""),
        )
        for sample in product.samples
    ]
    conn.executemany(
        """
        INSERT INTO propagation_samples (
            propagation_product_id, sample_index, jd_utc, t_s,
            pos_x_km, pos_y_km, pos_z_km,
            vel_x_km_s, vel_y_km_s, vel_z_km_s, error
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    return len(rows)


def _complete_propagation_invalidations(conn: sqlite3.Connection, *, object_ids: set[str], now: str) -> int:
    if not object_ids:
        return 0
    placeholders = ", ".join("?" for _ in object_ids)
    cursor = conn.execute(
        f"""
        UPDATE invalidation_queue
        SET status = 'completed', completed_utc = ?
        WHERE status IN ('open', 'planned', 'claimed')
          AND reason IN ('new_tle', 'od_solution_changed', 'maneuver_detected')
          AND object_id IN ({placeholders})
        """,
        [now, *sorted(object_ids)],
    )
    return int(cursor.rowcount if cursor.rowcount is not None else 0)


def _latest_propagation_product_for_object(
    conn: sqlite3.Connection,
    object_id: str,
    start_jd_utc: float,
    end_jd_utc: float,
) -> str | None:
    row = conn.execute(
        """
        SELECT propagation_product_id
        FROM propagation_products
        WHERE object_id = ?
          AND start_jd_utc = ?
          AND end_jd_utc = ?
        ORDER BY created_utc DESC
        LIMIT 1
        """,
        (object_id, float(start_jd_utc), float(end_jd_utc)),
    ).fetchone()
    return str(row[0]) if row and row[0] else None
