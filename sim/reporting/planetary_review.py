"""Additive planetary evidence schema; Earth ECI tables stay empty."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path


def write_review(*, payload, context, artifacts, config_json, config_sha256):
    from sim.reporting import review_store as shared

    if not context.cfg.outputs.review.enabled:
        return {}
    directory = Path(context.outdir) / "review"
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "run.sqlite"
    temporary = directory / "run.sqlite.tmp"
    if temporary.exists():
        temporary.unlink()
    summary = payload["summary"]
    stamp = summary["generated_utc"]
    with sqlite3.connect(temporary) as conn:
        conn.execute("PRAGMA foreign_keys=ON")
        shared._create_schema(conn)
        conn.executescript("""
            CREATE TABLE planetary_environment (
              origin_center_id INTEGER NOT NULL, central_body TEXT NOT NULL,
              inertial_axes TEXT NOT NULL, environment_sha256 TEXT NOT NULL,
              provenance_json TEXT NOT NULL);
            CREATE TABLE planetary_state (
              object_id TEXT NOT NULL REFERENCES objects(object_id), sample_index INTEGER NOT NULL,
              time_s REAL NOT NULL, origin_center_id INTEGER NOT NULL, inertial_axes TEXT NOT NULL,
              x_inertial_km REAL NOT NULL,y_inertial_km REAL NOT NULL,z_inertial_km REAL NOT NULL,
              vx_inertial_km_s REAL NOT NULL,vy_inertial_km_s REAL NOT NULL,vz_inertial_km_s REAL NOT NULL,
              mass_kg REAL NOT NULL, altitude_km REAL NOT NULL,density_kg_m3 REAL NOT NULL,
              relative_speed_m_s REAL NOT NULL,within_atmosphere_domain INTEGER NOT NULL,
              forces_json TEXT NOT NULL, PRIMARY KEY(object_id,sample_index));
        """)
        shared._insert_run_metadata(
            conn,
            cfg=context.cfg,
            summary=summary,
            outdir=context.outdir,
            generated_utc=stamp,
            config_json=config_json,
            config_sha256=config_sha256,
        )
        shared._insert_objects(conn, cfg=context.cfg, summary=summary)
        shared._insert_object_initialization(conn, payload=payload)
        shared._insert_object_propagation(conn, payload=payload)
        shared._insert_object_state_frame(conn, payload=payload)
        model = payload["planetary_environment"]
        conn.execute(
            "INSERT INTO planetary_environment VALUES (?,?,?,?,?)",
            (
                model["origin_center_id"],
                model["central_body"],
                "J2000",
                model["environment_sha256"],
                json.dumps(model, sort_keys=True, allow_nan=False),
            ),
        )
        if str(context.cfg.outputs.review.detail) != "compact":
            shared._insert_time_samples(conn, t_s=context.t_s)
            for oid, history in context.truth_hist.items():
                rows = []
                for i, (t, row, d) in enumerate(
                    zip(
                        context.t_s,
                        history,
                        payload["planetary_diagnostics_by_object"][oid],
                    )
                ):
                    rows.append(
                        (
                            oid,
                            i,
                            float(t),
                            model["origin_center_id"],
                            "J2000",
                            *map(float, row[:6]),
                            float(row[13]),
                            d["altitude_km"],
                            d["density_kg_m3"],
                            d["relative_speed_m_s"],
                            int(d["within_atmosphere_domain"]),
                            json.dumps(
                                d["forces_km_s2"], sort_keys=True, allow_nan=False
                            ),
                        )
                    )
                conn.executemany(
                    "INSERT INTO planetary_state VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    rows,
                )
        shared._insert_events(conn, t_s=context.t_s, summary=summary, thrust_hist={})
        shared._insert_artifacts(
            conn, artifacts=artifacts, outdir=context.outdir, generated_utc=stamp
        )
    from sim.review.generated_artifacts import clear_generated_review_artifacts

    clear_generated_review_artifacts(directory)
    temporary.replace(target)
    schema = directory / "schema.json"
    shared._write_schema_json(schema, generated_utc=stamp, db_path=target)
    return {"sqlite": str(target), "schema_json": str(schema)}
