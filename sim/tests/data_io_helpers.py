"""Small synthetic review stores for IO contract tests, not orbit evidence."""

import hashlib
import json
import sqlite3

import numpy as np


def make_review(root, *, times=None, states=None, frame="ECI", covariance=False):
    root.mkdir(parents=True)
    review = root / "review"
    review.mkdir()
    if times is None:
        times = np.arange(3) * 30.0
    if states is None:
        states = [[7000, float(t) * 7.5, 0, 0, 7.5, 0] for t in times]
    config = json.dumps(
        {
            "scenario_name": "io-test",
            "simulator": {"initial_jd_utc": 2460310.5, "dynamics": {"attitude": {"enabled": True}}},
        },
        sort_keys=True,
    )
    db = review / "run.sqlite"
    with sqlite3.connect(db) as conn:
        from sim.reporting.review_store import _create_schema

        _create_schema(conn)
        conn.execute(
            "INSERT INTO run_metadata (run_id, scenario_name, config_json, config_sha256) VALUES (?,?,?,?)",
            ("io-test-run", "io-test", config, hashlib.sha256(config.encode()).hexdigest()),
        )
        conn.execute("INSERT INTO object_state_frame VALUES (?,?)", ("SAT", frame))
        conn.execute("INSERT INTO objects (object_id, object_type, enabled) VALUES ('SAT','satellite',1)")
        for index, (time, state) in enumerate(zip(times, states)):
            conn.execute(
                "INSERT INTO object_state VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (index, float(time), "SAT", *map(float, state), 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0),
            )
            if covariance:
                matrix = np.eye(6) * 1e-6
                matrix[0, 3] = matrix[3, 0] = 2e-7
                conn.execute(
                    "INSERT INTO object_state_covariance (sample_index,time_s,object_id,frame,component_order_json,units_json,covariance_json,mathematically_valid,calibrated,calibration_scope) VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (
                        index,
                        float(time),
                        "SAT",
                        "ECI",
                        json.dumps(["x", "y", "z", "vx", "vy", "vz"]),
                        json.dumps(["km"] * 3 + ["km/s"] * 3),
                        json.dumps(matrix.tolist()),
                        1,
                        0,
                        None,
                    ),
                )
    return root
