from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from sim.interchange.adapters.maneuver_detection import (
    build_maneuver_detection_product,
    export_maneuver_detection_product,
)
from sim.review import ReviewWorkspace


def _write_maneuver_run(root: Path, *, run_id: str, generated_utc: str) -> None:
    review = root / "review"
    review.mkdir(parents=True, exist_ok=True)
    config = {
        "metadata": {"owner": "public", "public_surface": True},
        "objects": {
            "chaser": {
                "knowledge": {
                    "targets": ["target"],
                    "estimation": {
                        "type": "ekf",
                        "ekf": {"meas_noise_diag": [1.0] * 6},
                        "maneuver_detection": {"enabled": True, "threshold": 3.0},
                    },
                }
            }
        },
        "simulator": {},
    }
    db_path = review / "run.sqlite"
    with sqlite3.connect(db_path) as connection:
        connection.executescript(
            """
            CREATE TABLE run_metadata (
                run_id TEXT, scenario_name TEXT, oel_version TEXT,
                review_schema_version TEXT, generated_utc TEXT, config_json TEXT
            );
            CREATE TABLE events (
                event_id TEXT, time_s REAL, sample_index INTEGER, object_id TEXT,
                event_type TEXT, severity TEXT, message TEXT, source TEXT
            );
            """
        )
        connection.execute(
            "INSERT INTO run_metadata VALUES (?, ?, ?, ?, ?, ?)",
            (run_id, "synthetic-maneuver", "0.36.0", "1", generated_utc, json.dumps(config)),
        )
        connection.execute(
            "INSERT INTO events VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("event-1", 5.0, 10, "target", "maneuver_detection_confirmed", "info", "confirmed", "test"),
        )
    (review / "schema.json").write_text("{}\n", encoding="utf-8")
    (root / "master_run_summary.json").write_text(
        json.dumps(
            {
                "knowledge_consistency_by_observer": {
                    "chaser": {
                        "target": {
                            "maneuver_confirmed_event_count": 1,
                            "maneuver_first_confirmed_t_s": 5.0,
                            "innovation_norm": 4.0,
                        }
                    }
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_maneuver_export_uses_retained_source_after_visible_root_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    _write_maneuver_run(source, run_id="original-run", generated_utc="")
    original_db = hashlib.sha256((source / "review" / "run.sqlite").read_bytes()).hexdigest()
    original_summary = hashlib.sha256((source / "master_run_summary.json").read_bytes()).hexdigest()

    original_open = ReviewWorkspace.open

    def open_then_replace(
        cls: type[ReviewWorkspace], path: str | Path, *, protected: bool | None = None, **kwargs: object
    ) -> ReviewWorkspace:
        workspace = original_open(path, protected=protected, **kwargs)
        source.rename(tmp_path / "retained-source")
        _write_maneuver_run(source, run_id="attacker-run", generated_utc="1970-01-01T00:00:00Z")
        return workspace

    monkeypatch.setattr(ReviewWorkspace, "open", classmethod(open_then_replace))
    product_path = tmp_path / "product.json"
    result = export_maneuver_detection_product(
        source,
        output_path=product_path,
        export_created_utc="2026-10-23T01:02:03Z",
    )

    product = json.loads(product_path.read_text(encoding="utf-8"))
    assert result["valid"] is True
    assert product["producer"]["run_id"] == "original-run"
    assert product["created_utc"] == "2026-10-23T01:02:03Z"
    hashes = {item["artifact_id"]: item["sha256"] for item in product["provenance"]["source_artifacts"]}
    assert hashes["completed_run_review_store"] == original_db
    assert hashes["completed_run_summary"] == original_summary
    with sqlite3.connect(source / "review" / "run.sqlite") as connection:
        assert connection.execute("SELECT run_id FROM run_metadata").fetchone()[0] == "attacker-run"


def test_maneuver_export_reports_missing_source_timestamp_without_epoch_fabrication(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    _write_maneuver_run(source, run_id="timestampless-run", generated_utc="")

    product = build_maneuver_detection_product(
        source,
        output_path=tmp_path / "product.json",
        export_created_utc="2026-10-23T01:02:03Z",
    )

    assert product["created_utc"] == "2026-10-23T01:02:03Z"
    assert product["created_utc"] != "1970-01-01T00:00:00Z"
    assert product["freshness"]["age_status"] == "not_evaluated"
    assert product["freshness"]["evaluated_utc"] is None
    assert any("generated_utc" in warning for warning in product["quality"]["warnings"])
