"""Focused regressions for bounded review evidence and artifact provenance."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
import tempfile
from contextlib import closing
from pathlib import Path

import pytest

from sim.review import evidence_capsule
from sim.review.evidence_capsule import (
    EvidenceCapsuleError,
    create_evidence_capsule,
    restore_evidence_capsule,
)
from sim.review.manifest import write_workflow_review
from sim.review.workspace import ReviewQueryError, ReviewWorkspace, execute_bounded_review_query


def _review_fixture(root: Path) -> Path:
    review_dir = root / "review"
    review_dir.mkdir(parents=True, exist_ok=True)
    database = review_dir / "run.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE evidence (value BLOB)")
        connection.executemany("INSERT INTO evidence VALUES (?)", [(b"1234",), (b"5678",)])
    return database


def _make_symlink(link: Path, target: Path) -> None:
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("Symlinks are unavailable on this platform.")


def test_review_query_enforces_value_aggregate_row_and_vm_caps(tmp_path: Path) -> None:
    _review_fixture(tmp_path)

    with ReviewWorkspace.open(tmp_path) as workspace:
        with pytest.raises(ReviewQueryError, match="SQLite size budget"):
            workspace.query("SELECT value FROM evidence", max_value_bytes=3, max_result_bytes=100)

        with pytest.raises(ReviewQueryError, match="aggregate"):
            workspace.query("SELECT value FROM evidence ORDER BY rowid", max_value_bytes=128, max_result_bytes=16)

        with closing(workspace._connect(authorize=True)) as connection:
            with pytest.raises(ReviewQueryError, match="row limit"):
                execute_bounded_review_query(
                    connection,
                    "SELECT value FROM evidence",
                    max_rows=1,
                    max_value_bytes=128,
                    max_result_bytes=32,
                )

            with pytest.raises(ReviewQueryError, match="step budget"):
                execute_bounded_review_query(
                    connection,
                    "WITH RECURSIVE counter(value) AS ("
                    "SELECT 1 UNION ALL SELECT value + 1 FROM counter LIMIT 100000) "
                    "SELECT max(value) FROM counter",
                    max_vm_steps=1,
                    max_value_bytes=64,
                    max_result_bytes=128,
                )


def test_capsule_verification_query_count_is_bounded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    database = _review_fixture(tmp_path)
    monkeypatch.setattr(evidence_capsule, "MAX_CAPSULE_VERIFICATION_QUERIES", 0)

    with pytest.raises(EvidenceCapsuleError, match="query count"):
        create_evidence_capsule(database, verification_queries=["SELECT value FROM evidence"])


def test_corrupt_sqlite_capsule_raises_typed_error(tmp_path: Path) -> None:
    database = tmp_path / "review" / "run.sqlite"
    database.parent.mkdir(parents=True)
    database.write_bytes(b"this is not a sqlite database\n")

    with pytest.raises(EvidenceCapsuleError, match="SQLite verification failed"):
        create_evidence_capsule(database)


def test_capsule_rejects_static_symlinks_for_source_provenance_and_restore(tmp_path: Path) -> None:
    database = _review_fixture(tmp_path)

    source_alias = tmp_path / "source-alias.sqlite"
    _make_symlink(source_alias, database)
    with pytest.raises(EvidenceCapsuleError, match="unsafe"):
        create_evidence_capsule(source_alias)

    provenance = tmp_path / "provenance.txt"
    provenance.write_text("public provenance\n", encoding="utf-8")
    provenance_alias = tmp_path / "provenance-alias.txt"
    _make_symlink(provenance_alias, provenance)
    with pytest.raises(EvidenceCapsuleError, match="symbolic link"):
        create_evidence_capsule(database, provenance_paths=[provenance_alias])

    create_evidence_capsule(database, remove_original=True)
    restore_target = tmp_path / "restore-target.sqlite"
    restore_target.write_bytes(b"outside restore target\n")
    _make_symlink(database, restore_target)
    with pytest.raises(EvidenceCapsuleError, match="symbolic link"):
        restore_evidence_capsule(database, overwrite=True)
    assert restore_target.read_bytes() == b"outside restore target\n"


def test_review_workspace_accepts_valid_macos_tmp_alias() -> None:
    if sys.platform == "darwin" and Path("/tmp").resolve() == Path("/private/tmp"):
        with tempfile.TemporaryDirectory(prefix="oel-review-alias-", dir="/private/tmp") as raw_root:
            physical_root = Path(raw_root)
            _review_fixture(physical_root)
            alias_root = Path("/tmp") / physical_root.relative_to(Path("/private/tmp"))
            with ReviewWorkspace.open(alias_root) as workspace:
                assert workspace.query("SELECT COUNT(*) AS count FROM evidence").rows == [{"count": 2}]
        return

    with tempfile.TemporaryDirectory(prefix="oel-review-alias-") as raw_root:
        root = Path(raw_root)
        _review_fixture(root)
        with ReviewWorkspace.open(root) as workspace:
            assert workspace.query("SELECT COUNT(*) AS count FROM evidence").rows == [{"count": 2}]


def test_workflow_artifacts_are_contained_and_hashed(tmp_path: Path) -> None:
    output_dir = tmp_path / "workflow"
    output_dir.mkdir()
    artifact = output_dir / "summary.json"
    artifact.write_text('{"status":"ok"}\n', encoding="utf-8")

    write_workflow_review(
        output_dir=output_dir,
        workflow_type="focused_regression",
        status="complete",
        artifacts={"summary_json": str(artifact)},
    )

    manifest = json.loads((output_dir / "review" / "workflow_manifest.json").read_text(encoding="utf-8"))
    row = manifest["artifacts"][0]
    assert row["path"] == "summary.json"
    assert row["path_exists"] is True
    assert row["size_bytes"] == artifact.stat().st_size
    assert row["sha256"] == hashlib.sha256(artifact.read_bytes()).hexdigest()

    external = tmp_path / "external.json"
    external.write_text('{"status":"outside"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="inside the output directory"):
        write_workflow_review(
            output_dir=tmp_path / "workflow-external",
            workflow_type="focused_regression",
            status="partial",
            artifacts={"external": str(external)},
        )

    symlinked = output_dir / "external-alias.json"
    _make_symlink(symlinked, external)
    with pytest.raises(ValueError, match="symbolic link"):
        write_workflow_review(
            output_dir=output_dir,
            workflow_type="focused_regression",
            status="partial",
            artifacts={"symlinked": str(symlinked)},
        )
