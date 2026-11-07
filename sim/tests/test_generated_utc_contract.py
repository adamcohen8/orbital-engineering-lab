from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from sim.reporting.review_store import _utc_stamp
from sim.review.manifest import load_workflow_manifest, write_workflow_review


def _write_tiny_workflow_review(root: Path) -> dict[str, str]:
    summary_path = root / "summary.json"
    summary_path.write_text('{"ok": true}\n', encoding="utf-8")
    return write_workflow_review(
        output_dir=root,
        workflow_type="timestamp_contract",
        title="Timestamp contract",
        scenario_name="timestamp_contract",
        summary={"row_count": 1},
        artifacts={"summary_json": str(summary_path)},
        tables={"validation_benchmarks": [{"benchmark_name": "tiny", "passed": True}]},
    )


def test_generated_utc_defaults_empty_and_review_is_repeatable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OEL_GENERATED_UTC", raising=False)

    first_outputs = _write_tiny_workflow_review(tmp_path)
    first_bytes = {
        name: Path(first_outputs[name]).read_bytes()
        for name in ("workflow_manifest_json", "sqlite", "schema_json")
    }

    second_outputs = _write_tiny_workflow_review(tmp_path)
    second_bytes = {
        name: Path(second_outputs[name]).read_bytes()
        for name in ("workflow_manifest_json", "sqlite", "schema_json")
    }

    assert first_bytes == second_bytes
    assert load_workflow_manifest(tmp_path)["generated_utc"] == ""
    assert _utc_stamp() == ""
    with sqlite3.connect(tmp_path / "review" / "run.sqlite") as connection:
        assert connection.execute("SELECT generated_utc FROM workflow_metadata").fetchone() == ("",)


def test_generated_utc_uses_explicit_operator_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected = "2026-10-23T00:00:00+00:00"
    monkeypatch.setenv("OEL_GENERATED_UTC", expected)

    _write_tiny_workflow_review(tmp_path)

    assert load_workflow_manifest(tmp_path)["generated_utc"] == expected
    assert _utc_stamp() == expected
    with sqlite3.connect(tmp_path / "review" / "run.sqlite") as connection:
        assert connection.execute("SELECT generated_utc FROM workflow_metadata").fetchone() == (expected,)
