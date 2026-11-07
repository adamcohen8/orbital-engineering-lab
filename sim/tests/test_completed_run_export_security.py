from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

import sim.interchange.adapters.review_store as review_store_adapter


def _write_completed_run(root: Path) -> Path:
    review = root / "review"
    review.mkdir(parents=True)
    config = {
        "metadata": {
            "owner": "public",
            "export_review": {"approved_for_public_export": True},
        },
        "objects": {
            "sat": {
                "kind": "satellite",
                "role": "target",
                "specs": {"mass_kg": 10.0},
            }
        },
        "simulator": {
            "initial_jd_utc": 2461254.5,
            "dynamics": {"orbit": {"model": "two_body"}, "attitude": {"enabled": False}},
        },
    }
    config_json = json.dumps(config, sort_keys=True, separators=(",", ":"))
    config_sha256 = hashlib.sha256(config_json.encode("utf-8")).hexdigest()
    with sqlite3.connect(review / "run.sqlite") as connection:
        connection.executescript(
            """
            CREATE TABLE run_metadata (
                run_id TEXT, scenario_name TEXT, oel_version TEXT,
                review_schema_version TEXT, generated_utc TEXT,
                config_path TEXT, config_sha256 TEXT, config_json TEXT
            );
            CREATE TABLE objects (object_id TEXT, object_type TEXT, enabled INTEGER, role TEXT);
            CREATE TABLE object_state (
                sample_index INTEGER, time_s REAL, object_id TEXT,
                pos_x_eci_km REAL, pos_y_eci_km REAL, pos_z_eci_km REAL,
                vel_x_eci_km_s REAL, vel_y_eci_km_s REAL, vel_z_eci_km_s REAL,
                mass_kg REAL
            );
            CREATE TABLE object_state_frame (object_id TEXT, state_frame TEXT);
            CREATE TABLE events (
                event_id TEXT, time_s REAL, sample_index INTEGER, object_id TEXT,
                event_type TEXT, severity TEXT, message TEXT, source TEXT
            );
            """
        )
        connection.execute(
            "INSERT INTO run_metadata VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "original-run",
                "security-fixture",
                "0.36.0",
                "1",
                "",
                "fixture.yaml",
                config_sha256,
                config_json,
            ),
        )
        connection.execute("INSERT INTO objects VALUES (?, ?, ?, ?)", ("sat", "satellite", 1, "target"))
        connection.executemany(
            "INSERT INTO object_state VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (0, 0.0, "sat", 7000.0, 0.0, 0.0, 0.0, 7.5, 0.0, 10.0),
                (1, 1.0, "sat", 6999.9, 7.5, 0.0, -0.01, 7.5, 0.0, 10.0),
            ],
        )
        connection.execute("INSERT INTO object_state_frame VALUES (?, ?)", ("sat", "ECI"))
    (review / "schema.json").write_text(
        json.dumps({"generated_utc": "2026-10-23T01:02:03Z"}),
        encoding="utf-8",
    )
    return root


def test_completed_run_export_uses_retained_source_after_visible_root_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _write_completed_run(tmp_path / "source")
    original = tmp_path / "moved-source"
    replacement = tmp_path / "source"
    output = tmp_path / "state.json"
    original_db = source / "review" / "run.sqlite"
    original_schema = source / "review" / "schema.json"
    expected_hash = hashlib.sha256(original_db.read_bytes()).hexdigest()
    expected_schema_hash = hashlib.sha256(original_schema.read_bytes()).hexdigest()
    original_open = review_store_adapter.ReviewWorkspace.open

    def open_then_replace(path: str | Path, **kwargs: object):
        workspace = original_open(path, **kwargs)
        source.rename(original)
        (replacement / "review").mkdir(parents=True)
        (replacement / "review" / "run.sqlite").write_bytes(b"attacker-visible-database")
        (replacement / "review" / "schema.json").write_text(
            json.dumps({"generated_utc": "2099-01-01T00:00:00Z"}),
            encoding="utf-8",
        )
        return workspace

    monkeypatch.setattr(review_store_adapter.ReviewWorkspace, "open", open_then_replace)

    product = review_store_adapter.build_completed_run_state_product(
        source,
        output_path=output,
        object_id="sat",
        selector="final",
    )

    assert product["payload"]["source_run"]["run_id"] == "original-run"
    assert product["payload"]["source_run"]["generated_utc"] == "2026-10-23T01:02:03Z"
    assert product["payload"]["source_run"]["review_db_sha256"] == expected_hash
    assert product["provenance"]["source_artifacts"][0]["sha256"] == expected_hash
    schema_artifact = next(
        artifact
        for artifact in product["provenance"]["source_artifacts"]
        if artifact["artifact_id"] == "completed_run_review_schema"
    )
    assert schema_artifact["sha256"] == expected_schema_hash


def test_completed_run_export_rejects_in_place_source_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _write_completed_run(tmp_path / "source")
    database = source / "review" / "run.sqlite"
    original_query = review_store_adapter.ReviewWorkspace.query
    mutated = False

    def query_then_mutate(workspace, *args: object, **kwargs: object):
        nonlocal mutated
        result = original_query(workspace, *args, **kwargs)
        if not mutated:
            with database.open("r+b") as stream:
                stream.seek(-1, 2)
                value = stream.read(1)
                stream.seek(-1, 2)
                stream.write(bytes([value[0] ^ 0x01]))
                stream.flush()
            mutated = True
        return result

    monkeypatch.setattr(review_store_adapter.ReviewWorkspace, "query", query_then_mutate)

    with pytest.raises(
        review_store_adapter.CompletedRunStateExportError,
        match="Review store or schema changed",
    ):
        review_store_adapter.build_completed_run_state_product(
            source,
            output_path=tmp_path / "state.json",
            object_id="sat",
            selector="final",
        )
