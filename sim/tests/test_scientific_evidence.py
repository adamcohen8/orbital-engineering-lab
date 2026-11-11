"""Recorded evidence semantics with independently specified rows."""

import hashlib
import json
import sqlite3
import subprocess
import sys

import pytest

from sim.reporting.review_store import _create_schema, refresh_review_schema
from sim.review import ReviewWorkspace, read_scientific_evidence, scientific_evidence


@pytest.fixture
def saved(tmp_path):
    review = tmp_path / "review"
    review.mkdir()
    db = review / "run.sqlite"
    config = json.dumps({"simulator": {"initial_jd_utc": 2462000.5}})
    with sqlite3.connect(db) as conn:
        _create_schema(conn)
        conn.execute("INSERT INTO run_metadata(run_id, oel_version, review_schema_version, config_sha256, config_json) "
                     "VALUES ('frozen-run', '1.0.0', 'test', ?, ?)",
                     (hashlib.sha256(config.encode()).hexdigest(), config))
        conn.execute("INSERT INTO objects(object_id) VALUES ('target')")
        conn.execute("INSERT INTO object_state_frame VALUES ('target', 'eci')")
        conn.execute("INSERT INTO frame_provenance(scope, model, time_scale_model) VALUES ('scenario', 'simple', 'utc')")
        conn.executemany("INSERT INTO object_state(sample_index, time_s, object_id, pos_x_eci_km) "
                         "VALUES (?, ?, 'target', ?)", [(0, 0, 6.0), (1, 7, -4.0), (2, 13, -4.0), (3, 23, 3.0)])
    refresh_review_schema(db)
    return tmp_path


def request(operation, **values):
    return {"operation": operation, "quantity_ids": ["pos_x_eci_km"], **values}


def test_brackets_ties_and_scope(saved):
    exact = read_scientific_evidence(saved, request("exact", time_s=10))
    assert exact["blocker"]["code"] == "epoch_not_recorded"
    assert exact["blocker"]["facts"]["object_id"] == "target"
    assert exact["blocker"]["facts"]["neighbor_times_s"] == [7, 13]
    absent_index = read_scientific_evidence(saved, request("exact", sample_index=99))
    assert absent_index["blocker"]["code"] == "sample_not_recorded"
    assert absent_index["blocker"]["facts"] == {"object_id": "target", "sample_index": 99}
    bracket = read_scientific_evidence(saved, request("bracket", time_s=10))
    assert [r["sample_index"] for r in bracket["rows"]] == [1, 2]
    for tie, time in [("earlier", 7), ("later", 13)]:
        result = read_scientific_evidence(saved, request("nearest", time_s=10, tie=tie))
        assert result["rows"][0]["time_s"] == time
        assert result["sampling"]["separation_s"] == 3
        assert result["sampling"]["sample_minus_requested_s"] == time - 10
    on_sample = read_scientific_evidence(saved, request("nearest", time_s=13, tie="earlier"))
    assert on_sample["sampling"]["sample_minus_requested_s"] == 0
    limited = read_scientific_evidence(saved, request("nearest", time_s=10, tie="earlier", max_separation_s=2))
    assert limited["blocker"]["code"] == "nearest_outside_tolerance"
    minimum = read_scientific_evidence(saved, request("extremum", start_s=7, stop_s=23, direction="min", tie="later"))
    assert minimum["rows"][0]["time_s"] == 13
    assert minimum["sampling"]["samples"] == 3
    assert minimum["sampling"]["kind"] == "saved_samples"
    assert minimum["quantities"]["pos_x_eci_km"]["recorded_unit"] == "km"


def test_signed_difference_uses_attributed_compatible_reference(saved):
    spec = request("difference", time_s=7, output_units="m", order="reference_minus_saved")
    assert read_scientific_evidence(saved, spec)["blocker"]["code"] == "reference_missing"
    reference = {"source": "synthetic control", "quantity_id": "pos_x_eci_km",
                 "value": 500, "units": "m", "object_id": "target", "frame": "ECI",
                 "frame_model": "simple", "epoch_jd_utc": 2462000.5 + 7 / 86400}
    result = read_scientific_evidence(saved, {**spec, "reference": reference})
    assert result["rows"][0]["difference"] == 4500
    assert result["rows"][0]["reference"]["verification"] == "caller_supplied"
    assert result["source"]["run_id"] == "frozen-run"
    assert result["rows"][0]["epoch_jd_utc"] == reference["epoch_jd_utc"]
    for field, value, code in [
        ("frame", "TEME", "reference_frame_mismatch"),
        ("epoch_jd_utc", 2462000.5, "reference_epoch_mismatch"),
        ("units", "m/s", "incompatible_units"),
    ]:
        reply = read_scientific_evidence(saved, {**spec, "reference": {**reference, field: value}})
        assert reply["blocker"]["code"] == code


def test_ambiguous_selection_and_incomplete_interval(saved):
    db = saved / "review/run.sqlite"
    with sqlite3.connect(db) as conn:
        conn.execute("INSERT INTO object_state(sample_index, time_s, object_id) VALUES (4, 7, 'target')")
    assert read_scientific_evidence(saved, request("exact", time_s=7))["blocker"]["code"] == "ambiguous_sample"
    assert read_scientific_evidence(saved, request("exact", sample_index=1))["rows"][0]["pos_x_eci_km"] == -4
    result = read_scientific_evidence(saved, request("extremum", direction="max", tie="later"))
    assert result["blocker"]["code"] == "incomplete_quantity"
    with sqlite3.connect(db) as conn:
        conn.execute("INSERT INTO objects(object_id) VALUES ('other')")
    assert read_scientific_evidence(saved, request("exact", time_s=0))["blocker"]["code"] == "object_ambiguous"


def test_cli_api_parity_and_changed_snapshot(saved):
    spec = request("exact", sample_index=2)
    result = read_scientific_evidence(saved, spec)
    cli = subprocess.check_output([sys.executable, "-m", "sim.review", str(saved), "--scientific", json.dumps(spec)])
    assert json.loads(cli) == result
    with ReviewWorkspace.open(saved) as workspace:
        workspace.freeze_source()
        with sqlite3.connect(saved / "review/run.sqlite") as conn:
            conn.execute("UPDATE object_state SET pos_x_eci_km = 55 WHERE sample_index = 2")
        reply = scientific_evidence(workspace, spec)
        assert reply["blocker"]["code"] == "evidence_changed"
        assert reply["rows"] == []


def test_public_local_mcp_returns_same_receipt(saved):
    from integrations.oel_mcp import PublicOELMCPHandlers

    spec = request("nearest", time_s=10, tie="earlier")
    handlers = PublicOELMCPHandlers(profile="public_local", read_roots=(saved,), write_roots=(saved,))
    reply = handlers.call("oel.review_scientific.v1", {
        "output_dir": str(saved), "request": spec,
        "handling": {"marking": "PUBLIC", "release_scope": "local_only"},
    })
    assert reply["status"] == "completed", reply
    assert reply["result"] == read_scientific_evidence(saved, spec)
    missing = handlers.call("oel.review_scientific.v1", {
        "output_dir": str(saved), "request": request("exact", time_s=10),
        "handling": {"marking": "PUBLIC", "release_scope": "local_only"},
    })
    assert missing["status"] == "partial"
    assert missing["result"]["blocker"]["code"] == "epoch_not_recorded"
