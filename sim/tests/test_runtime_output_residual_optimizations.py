"""Preserve rocket arithmetic and transactional review-store output contracts."""

from __future__ import annotations

import re
import sqlite3
from types import SimpleNamespace

import numpy as np
import pytest

import sim.reporting.review_store as review_owner
import sim.runtime.rocket_factory as rocket_owner
from sim.config import scenario_config_from_dict
from sim.single_run import _SingleRunEngine


def _legacy_orbital_elements(r_km, v_km_s, mu=rocket_owner.EARTH_MU_KM3_S2):
    r = float(np.linalg.norm(r_km))
    v2 = float(np.dot(v_km_s, v_km_s))
    if r <= 0.0:
        return np.inf, np.inf
    eps = 0.5 * v2 - mu / r
    a = np.inf if abs(eps) < 1e-14 else float(-mu / (2.0 * eps))
    h = np.cross(r_km, v_km_s)
    e_vec = np.cross(v_km_s, h) / mu - r_km / r
    return a, float(np.linalg.norm(e_vec))


def test_rocket_orbital_elements_exact_three_vector_envelope():
    rng = np.random.default_rng(8237)
    states = [
        (rng.normal(size=3) * 7000, rng.normal(size=3) * 7.5)
        for _ in range(128)
    ]
    states += [
        (np.zeros(3), np.zeros(3)),
        (np.array([7000.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0])),
        (np.array([7000.0, 0.0, 0.0]), np.array([0.0, np.sqrt(2 * rocket_owner.EARTH_MU_KM3_S2 / 7000), 0.0])),
        (np.array([1e-150, -2e-150, 3e-150]), np.array([1e-150, 2e-150, -3e-150])),
        (np.array([1e100, -2e100, 3e100]), np.array([1e100, 2e100, -3e100])),
    ]
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for r, v in states:
            for readonly, stride in ((False, 1), (True, 2)):
                r_input, v_input = np.zeros(3 * stride)[::stride], np.zeros(3 * stride)[::stride]
                r_input[:], v_input[:] = r, v
                r_input.flags.writeable = v_input.flags.writeable = not readonly
                expected = _legacy_orbital_elements(r_input, v_input)
                actual = rocket_owner._orbital_elements_basic(r_input, v_input)
                np.testing.assert_array_equal(np.asarray(actual).view(np.uint64), np.asarray(expected).view(np.uint64))
                np.testing.assert_array_equal(r_input, r)
                np.testing.assert_array_equal(v_input, v)


@pytest.mark.parametrize("envelope", ["float32", "integer", "two", "four", "list", "subclass"])
def test_rocket_orbital_elements_unsupported_inputs_retain_fallback(monkeypatch, envelope):
    r, v = np.array([7000.0, 1.0, 2.0]), np.array([0.0, 7.5, 0.2])
    if envelope == "float32":
        r, v = r.astype(np.float32), v.astype(np.float32)
    elif envelope == "integer":
        r, v = r.astype(np.int64), v.astype(np.int64)
    elif envelope == "two":
        r, v = r[:2], v[:2]
    elif envelope == "four":
        r, v = np.append(r, 0), np.append(v, 0)
    elif envelope == "list":
        r, v = r.tolist(), v.tolist()
    elif envelope == "subclass":
        class Vector(np.ndarray):
            pass

        r, v = r.view(Vector), v.view(Vector)

    def forbidden_dispatch(*args):
        pytest.fail("unsupported input reached fixed three-vector owner")

    monkeypatch.setattr(rocket_owner, "_navigation_orbital_elements_basic", forbidden_dispatch)
    try:
        expected = _legacy_orbital_elements(r, v)
    except (TypeError, ValueError) as exc:
        with pytest.raises(type(exc), match=re.escape(str(exc))):
            rocket_owner._orbital_elements_basic(r, v)
    else:
        np.testing.assert_array_equal(rocket_owner._orbital_elements_basic(r, v), expected)


def test_review_schema_transaction_preserves_schema_and_helper_exit_state():
    class CapturedConnection(sqlite3.Connection):
        def executescript(self, script):
            self.schema_script = script
            return super().executescript(script)

    with sqlite3.connect(":memory:", factory=CapturedConnection) as candidate, sqlite3.connect(":memory:") as reference:
        trace = []
        candidate.set_trace_callback(trace.append)
        review_owner._create_schema(candidate)
        assert not candidate.in_transaction
        reference.executescript(re.sub(r"^\s*(BEGIN|COMMIT);\s*$", "", candidate.schema_script, flags=re.MULTILINE))
        query = "SELECT type, name, tbl_name, sql FROM sqlite_master ORDER BY type, name"
        assert candidate.execute(query).fetchall() == reference.execute(query).fetchall()
        assert sum(statement.strip() == "BEGIN;" for statement in trace) == 1
        assert sum(statement.strip() == "COMMIT;" for statement in trace) == 1
        candidate.execute("INSERT INTO time_samples VALUES (?, ?)", (0, 1.25))
        assert candidate.in_transaction
        candidate.commit()
        assert not candidate.in_transaction
        assert candidate.execute("SELECT * FROM time_samples").fetchall() == [(0, 1.25)]


def test_review_schema_failure_rolls_back_and_connection_can_retry():
    with sqlite3.connect(":memory:") as conn:
        def deny_late_table(action, name, *_):
            return sqlite3.SQLITE_DENY if action == sqlite3.SQLITE_CREATE_TABLE and name == "artifacts" else sqlite3.SQLITE_OK

        conn.set_authorizer(deny_late_table)
        with pytest.raises(sqlite3.DatabaseError, match="not authorized"):
            review_owner._create_schema(conn)
        assert not conn.in_transaction
        assert conn.execute("SELECT name FROM sqlite_master").fetchall() == []
        conn.set_authorizer(None)
        review_owner._create_schema(conn)
        assert not conn.in_transaction
        assert conn.execute("SELECT name FROM sqlite_master WHERE name='artifacts'").fetchone() == ("artifacts",)


def test_review_writer_transaction_preserves_all_rows_and_schema_sidecar(tmp_path, monkeypatch):
    cfg = scenario_config_from_dict({
        "scenario_name": "review_transaction_parity",
        "objects": {"satellite": {
            "enabled": True, "specs": {"mass_kg": 100.0},
            "initial_state": {"position_eci_km": [7000, 0, 0], "velocity_eci_km_s": [0, 7.5, 0]},
            "flight_software": {"profile": "fsw.profile.coast_monitor.v1"},
        }},
        "simulator": {
            "duration_s": 2.0, "dt_s": 1.0,
            "dynamics": {"attitude": {"enabled": False}},
            "termination": {"earth_impact_enabled": False},
        },
        "outputs": {
            "output_dir": str(tmp_path), "plots": {"enabled": False},
            "review": {"enabled": True, "detail": "full"},
            "stats": {"save_json": False, "save_full_log": False, "print_summary": False},
        },
    })
    engine = _SingleRunEngine(cfg)
    payload = engine.run()
    context = SimpleNamespace(cfg=cfg, outdir=tmp_path, t_s=engine.t_s,
                              truth_hist=engine.truth_hist, thrust_hist=engine.thrust_hist)
    original = review_owner._create_schema
    captured = []

    class CapturedConnection(sqlite3.Connection):
        def executescript(self, script):
            captured.append(script)
            return super().executescript(script)

    with sqlite3.connect(":memory:", factory=CapturedConnection) as conn:
        original(conn)
    legacy_sql = re.sub(r"^\s*(BEGIN|COMMIT);\s*$", "", captured[0], flags=re.MULTILINE)

    def saved_rows():
        with sqlite3.connect(tmp_path / "review" / "run.sqlite") as conn:
            schema = conn.execute("SELECT type, name, tbl_name, sql FROM sqlite_master ORDER BY type, name").fetchall()
            tables = [row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
            rows = {name: conn.execute(f'SELECT * FROM "{name}"').fetchall() for name in tables}
        return schema, rows, (tmp_path / "review" / "schema.json").read_bytes()

    with monkeypatch.context() as patcher:
        patcher.setattr(review_owner, "_create_schema", lambda conn: conn.executescript(legacy_sql))
        review_owner.write_single_run_review_store(payload=payload, context=context, artifacts={})
        reference = saved_rows()
    review_owner.write_single_run_review_store(payload=payload, context=context, artifacts={})
    actual = saved_rows()
    assert actual == reference
    assert actual[1]["fsw_invocations"]
    assert actual[1]["fsw_input_events"]
    assert actual[1]["fsw_snapshots"]


@pytest.mark.parametrize("failure", ["schema", "insert"])
def test_review_writer_failure_preserves_previous_store_and_cleans_temporary(tmp_path, monkeypatch, failure):
    directory = tmp_path / "review"
    directory.mkdir()
    previous = directory / "run.sqlite"
    previous.write_bytes(b"previous completed store")
    sidecar = directory / "schema.json"
    sidecar.write_text("previous schema")
    connections = []
    real_connect = sqlite3.connect

    def connect(path):
        conn = real_connect(path)
        connections.append(conn)
        if failure == "schema":
            conn.set_authorizer(lambda action, name, *_: sqlite3.SQLITE_DENY if action == sqlite3.SQLITE_CREATE_TABLE and name == "artifacts" else sqlite3.SQLITE_OK)
        return conn

    def fail_insert(conn, **kwargs):
        conn.execute("INSERT INTO time_samples VALUES (?, ?)", (0, 1.0))
        raise RuntimeError("insert failure")

    monkeypatch.setattr(review_owner.sqlite3, "connect", connect)
    if failure == "insert":
        monkeypatch.setattr(review_owner, "_insert_run_metadata", fail_insert)
    context = SimpleNamespace(
        cfg=SimpleNamespace(
            simulator=SimpleNamespace(central_body="earth"),
            outputs=SimpleNamespace(review=SimpleNamespace(enabled=True, detail="standard")),
        ),
        outdir=tmp_path, t_s=np.array([0.0]), truth_hist={}, thrust_hist={},
    )
    error, message = (sqlite3.DatabaseError, "not authorized") if failure == "schema" else (RuntimeError, "insert failure")
    with pytest.raises(error, match=message):
        review_owner.write_single_run_review_store(payload={"summary": {}}, context=context, artifacts={})
    assert previous.read_bytes() == b"previous completed store"
    assert sidecar.read_text() == "previous schema"
    assert list(directory.glob("run.sqlite.tmp*")) == []
    for conn in connections:
        with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
            conn.execute("SELECT 1")
