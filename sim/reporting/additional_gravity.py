"""Optional external-field provenance and sample-indexed force evidence."""

from __future__ import annotations

import json


def attach_earth_evidence(payload, engine, parts):
    models, rows = {}, []
    for oid, agent in engine.agents.items():
        additional = agent.dynamics.orbit_propagator.additional_gravity
        additional.verify_resources()
        models[oid] = additional.metadata()
        for i, (t, state) in enumerate(zip(parts.t_s, parts.truth_hist[oid])):
            for name, acceleration in additional.components(t, state[:6]).items():
                rows.append(
                    {
                        "object_id": oid,
                        "sample_index": i,
                        "time_s": float(t),
                        "force": name,
                        "acceleration_km_s2": acceleration,
                    }
                )
    payload["additional_gravity"] = models
    payload["additional_gravity_samples"] = rows
    payload["summary"]["additional_gravity"] = models


def insert_review(conn, payload, *, include_samples):
    conn.executescript("""
        CREATE TABLE additional_gravity_environment (
          object_id TEXT PRIMARY KEY REFERENCES objects(object_id),
          origin_center_id INTEGER NOT NULL, inertial_axes TEXT NOT NULL,
          environment_sha256 TEXT NOT NULL, provenance_json TEXT NOT NULL);
        CREATE TABLE additional_gravity_acceleration (
          object_id TEXT NOT NULL REFERENCES objects(object_id), sample_index INTEGER NOT NULL,
          time_s REAL NOT NULL, force TEXT NOT NULL,
          ax_km_s2 REAL NOT NULL, ay_km_s2 REAL NOT NULL, az_km_s2 REAL NOT NULL,
          PRIMARY KEY(object_id,sample_index,force));
    """)
    for oid, model in payload["additional_gravity"].items():
        conn.execute(
            "INSERT INTO additional_gravity_environment VALUES (?,?,?,?,?)",
            (
                oid,
                model["origin_center_id"],
                model["inertial_axes"],
                model["environment_sha256"],
                json.dumps(model, sort_keys=True, allow_nan=False),
            ),
        )
    if include_samples:
        conn.executemany(
            "INSERT INTO additional_gravity_acceleration VALUES (?,?,?,?,?,?,?)",
            [
                (r["object_id"], r["sample_index"], r["time_s"], r["force"], *r["acceleration_km_s2"])
                for r in payload["additional_gravity_samples"]
            ],
        )
