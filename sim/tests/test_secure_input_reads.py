"""Focused coverage for descriptor-bound input and control-record primitives."""

from __future__ import annotations

import os
import shutil
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

import sim.review.workspace as review_workspace_module
from sim.review.evidence_capsule import create_evidence_capsule
from sim.review.workspace import ReviewStoreNotFoundError
from sim.security import (
    DescriptorStore,
    SecureReadChanged,
    SecureReadError,
    SecureReadRoot,
    input_reads,
)


def _root(tmp_path: Path) -> SecureReadRoot:
    root = tmp_path / "records"
    root.mkdir()
    return SecureReadRoot.open(root, approved_roots=(tmp_path,))


def test_snapshot_compatibility_backend_keeps_only_memory_database(tmp_path, monkeypatch):
    source = tmp_path / "source.sqlite"
    with sqlite3.connect(source) as database:
        database.execute("CREATE TABLE evidence(value TEXT)")
        database.execute("INSERT INTO evidence VALUES ('verified')")
    payload = source.read_bytes()
    source.unlink()
    monkeypatch.setattr(input_reads, "_sqlite_deserialize_available", lambda: False)
    with closing(input_reads.sqlite_connection_from_snapshot(payload)) as connection:
        assert connection.execute("SELECT value FROM evidence").fetchall() == [("verified",)]
        assert connection.execute("PRAGMA database_list").fetchall() == [(0, "main", "")]


def test_snapshot_compatibility_backend_preserves_empty_and_invalid_database_semantics(monkeypatch):
    monkeypatch.setattr(input_reads, "_sqlite_deserialize_available", lambda: False)
    with closing(input_reads.sqlite_connection_from_snapshot(b"")) as connection:
        assert connection.execute("SELECT name FROM sqlite_schema").fetchall() == []
    with pytest.raises(sqlite3.DatabaseError):
        input_reads.sqlite_connection_from_snapshot(b"invalid SQLite snapshot")


def test_read_root_can_create_missing_descendants_below_approved_root(tmp_path: Path) -> None:
    existing = tmp_path / "existing"
    existing.mkdir()
    existing.chmod(0o750)
    root_path = existing / "records" / "nested"
    with SecureReadRoot.open(
        root_path,
        approved_roots=(tmp_path,),
        create=True,
        mode=0o700,
    ) as root:
        assert root.path == root_path
        assert root_path.is_dir()
        assert (root_path.stat().st_mode & 0o777) == 0o700
        assert (root_path.parent.stat().st_mode & 0o777) == 0o700
        assert (existing.stat().st_mode & 0o777) == 0o750


def test_read_root_stays_anchored_after_visible_root_rename(tmp_path: Path) -> None:
    root_path = tmp_path / "records"
    root_path.mkdir()
    (root_path / "payload.txt").write_bytes(b"anchored")

    with SecureReadRoot.open(root_path, approved_roots=(tmp_path,)) as root:
        moved = tmp_path / "moved-records"
        root_path.rename(moved)
        replacement = tmp_path / "records"
        replacement.mkdir()
        (replacement / "payload.txt").write_bytes(b"replacement")

        assert root.read_bytes("payload.txt") == b"anchored"


def test_read_root_rejects_symlink_child(tmp_path: Path) -> None:
    outside = tmp_path / "outside.txt"
    outside.write_bytes(b"outside")
    root = _root(tmp_path)
    try:
        (root.path / "alias.txt").symlink_to(outside)
        with pytest.raises(SecureReadError):
            root.read_bytes("alias.txt")
    finally:
        root.close()


def test_descriptor_store_rejects_hard_link_update(tmp_path: Path) -> None:
    root = _root(tmp_path)
    try:
        (root.path / "record").write_bytes(b"old")
        os.link(root.path / "record", root.path / "outside-alias")
        store = DescriptorStore(root)
        with pytest.raises(SecureReadError):
            store.write_existing("record", b"new")
        assert (root.path / "record").read_bytes() == b"old"
    finally:
        root.close()


def test_descriptor_store_lock_rejects_name_replacement_after_acquisition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _root(tmp_path)
    try:
        store = DescriptorStore(root)
        real_flock = input_reads.fcntl.flock

        def replace_after_exclusive(fd: int, operation: int) -> None:
            real_flock(fd, operation)
            if operation == input_reads.fcntl.LOCK_EX:
                os.rename(root.path / "control.lock", root.path / "control.lock.old")
                (root.path / "control.lock").write_bytes(b"replacement")

        monkeypatch.setattr(input_reads.fcntl, "flock", replace_after_exclusive)
        with pytest.raises(SecureReadChanged):
            with store.lock("control.lock"):
                pass
    finally:
        root.close()


def test_descriptor_store_atomic_failure_retains_uncertain_temporary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _root(tmp_path)
    try:
        store = DescriptorStore(root)
        (root.path / "record").write_bytes(b"old")

        def fail_replace(*_args: object, **_kwargs: object) -> None:
            raise OSError("publication outcome is unknown")

        monkeypatch.setattr(input_reads.os, "replace", fail_replace)
        with pytest.raises(OSError, match="unknown"):
            store.write_atomic("record", b"new")
        assert any(name.startswith(".record.") and name.endswith(".tmp") for name in os.listdir(root.path))
    finally:
        root.close()


def test_secure_read_limit_rejects_non_integer_values(tmp_path: Path) -> None:
    root = _root(tmp_path)
    try:
        (root.path / "payload").write_bytes(b"data")
        with pytest.raises(TypeError):
            root.read_bytes("payload", max_bytes=True)
        with pytest.raises(TypeError):
            root.read_bytes("payload", max_bytes=1.5)  # type: ignore[arg-type]
    finally:
        root.close()


def test_protected_review_sqlite_snapshot_survives_visible_root_symlink_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    review_dir = tmp_path / "review"
    review_dir.mkdir()
    database = review_dir / "run.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE items (value TEXT NOT NULL)")
        connection.execute("INSERT INTO items(value) VALUES ('anchored')")

    with review_workspace_module.ReviewWorkspace.open(tmp_path) as workspace:
        real_connect = sqlite3.connect
        seen_databases: list[object] = []
        swapped = False

        def connect(database_arg: object, *args: object, **kwargs: object) -> sqlite3.Connection:
            nonlocal swapped
            seen_databases.append(database_arg)
            if database_arg == ":memory:" and not swapped:
                swapped = True
                moved = tmp_path / "review-moved"
                review_dir.rename(moved)
                hostile_review = tmp_path / "hostile" / "review"
                hostile_review.mkdir(parents=True)
                hostile_database = hostile_review / "run.sqlite"
                with real_connect(hostile_database) as hostile:
                    hostile.execute("CREATE TABLE items (value TEXT NOT NULL)")
                    hostile.execute("INSERT INTO items(value) VALUES ('redirected')")
                review_dir.symlink_to(hostile_review, target_is_directory=True)
            return real_connect(database_arg, *args, **kwargs)

        monkeypatch.setattr(review_workspace_module.sqlite3, "connect", connect)
        result = workspace.query("SELECT value FROM items")

    assert result.rows == [{"value": "anchored"}]
    assert swapped
    assert seen_databases == [":memory:"]


def test_review_workspace_rejects_root_replacement_during_namespace_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    review_dir = source / "review"
    review_dir.mkdir(parents=True)
    with sqlite3.connect(review_dir / "run.sqlite") as connection:
        connection.execute("CREATE TABLE items (value TEXT NOT NULL)")
        connection.execute("INSERT INTO items(value) VALUES ('original')")

    replacement = tmp_path / "replacement"
    replacement_review = replacement / "review"
    replacement_review.mkdir(parents=True)
    with sqlite3.connect(replacement_review / "run.sqlite") as connection:
        connection.execute("CREATE TABLE items (value TEXT NOT NULL)")
        connection.execute("INSERT INTO items(value) VALUES ('replacement')")

    real_open = review_workspace_module.SecureReadRoot.open
    open_count = 0

    def replace_after_output_open(
        cls: type[SecureReadRoot],
        path: str | Path,
        *,
        approved_roots: tuple[str | Path, ...],
        **kwargs: object,
    ) -> SecureReadRoot:
        nonlocal open_count
        reader = real_open(path, approved_roots=approved_roots, **kwargs)
        open_count += 1
        if open_count == 1:
            source.rename(tmp_path / "moved-source")
            replacement.rename(source)
        return reader

    monkeypatch.setattr(
        review_workspace_module.SecureReadRoot,
        "open",
        classmethod(replace_after_output_open),
    )
    with pytest.raises(ReviewStoreNotFoundError):
        review_workspace_module.ReviewWorkspace.open(source, protected=True)
    assert open_count >= 2


def test_capsule_review_workspace_binds_hydrated_db_to_retained_manifest(tmp_path: Path) -> None:
    source = tmp_path / "capsule-source"
    review_dir = source / "review"
    review_dir.mkdir(parents=True)
    database = review_dir / "run.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE items (value TEXT NOT NULL)")
        connection.execute("INSERT INTO items(value) VALUES ('capsule-original')")
    create_evidence_capsule(database, remove_original=True)

    with review_workspace_module.ReviewWorkspace.open(source, protected=True) as workspace:
        assert workspace.query("SELECT value FROM items").rows == [{"value": "capsule-original"}]


def test_capsule_review_workspace_rejects_root_replacement_during_materialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "capsule-source"
    review_dir = source / "review"
    review_dir.mkdir(parents=True)
    database = review_dir / "run.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE items (value TEXT NOT NULL)")
        connection.execute("INSERT INTO items(value) VALUES ('capsule-original')")
    create_evidence_capsule(database, remove_original=True)
    replacement = tmp_path / "capsule-replacement"
    shutil.copytree(source, replacement)

    real_open = review_workspace_module.SecureReadRoot.open
    open_count = 0

    def replace_after_output_open(
        cls: type[SecureReadRoot],
        path: str | Path,
        *,
        approved_roots: tuple[str | Path, ...],
        **kwargs: object,
    ) -> SecureReadRoot:
        nonlocal open_count
        reader = real_open(path, approved_roots=approved_roots, **kwargs)
        open_count += 1
        if open_count == 1:
            source.rename(tmp_path / "moved-capsule-source")
            replacement.rename(source)
        return reader

    monkeypatch.setattr(
        review_workspace_module.SecureReadRoot,
        "open",
        classmethod(replace_after_output_open),
    )
    with pytest.raises(ReviewStoreNotFoundError):
        review_workspace_module.ReviewWorkspace.open(source, protected=True)
    assert open_count >= 2


def test_protected_empty_sqlite_store_preserves_pristine_read_only_semantics(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    review_dir.mkdir()
    (review_dir / "run.sqlite").touch()

    with review_workspace_module.ReviewWorkspace.open(tmp_path, protected=True) as workspace:
        assert workspace.query("SELECT name FROM sqlite_master").rows == []
        with closing(workspace._connect()) as connection:
            with pytest.raises(sqlite3.OperationalError, match="read.?only"):
                connection.execute("CREATE TABLE items (value TEXT NOT NULL)")


def test_sqlite_snapshot_keeps_nested_parent_descriptor_after_visible_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _root(tmp_path)
    nested = root.path / "nested"
    nested.mkdir()
    database = nested / "run.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE items (value TEXT NOT NULL)")
        connection.execute("INSERT INTO items(value) VALUES ('anchored-nested')")

    real_stat = input_reads.os.stat
    swapped = False

    def stat(path: object, *args: object, **kwargs: object) -> os.stat_result:
        nonlocal swapped
        if path == "run.sqlite" and kwargs.get("dir_fd") is not None and not swapped:
            swapped = True
            nested.rename(tmp_path / "nested-moved")
            hostile_nested = tmp_path / "nested"
            hostile_nested.mkdir()
            hostile_database = hostile_nested / "run.sqlite"
            with sqlite3.connect(hostile_database) as hostile:
                hostile.execute("CREATE TABLE items (value TEXT NOT NULL)")
                hostile.execute("INSERT INTO items(value) VALUES ('redirected-nested')")
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(input_reads.os, "stat", stat)
    try:
        payload = input_reads.sqlite_snapshot_bytes(
            root,
            "nested/run.sqlite",
            max_bytes=1_000_000,
        )
    finally:
        root.close()

    memory = sqlite3.connect(":memory:")
    try:
        memory.deserialize(payload)
        assert memory.execute("SELECT value FROM items").fetchone() == ("anchored-nested",)
    finally:
        memory.close()
    assert swapped


def test_sqlite_snapshot_copies_real_wal_without_source_links(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _root(tmp_path)
    database = root.path / "run.sqlite"
    source = sqlite3.connect(database)
    try:
        source.execute("PRAGMA journal_mode=WAL")
        source.execute("PRAGMA wal_autocheckpoint=0")
        source.execute("CREATE TABLE items (value TEXT NOT NULL)")
        source.execute("INSERT INTO items(value) VALUES ('wal-anchored')")
        source.commit()
        assert (root.path / "run.sqlite-wal").is_file()
        assert (root.path / "run.sqlite-shm").is_file()

        def forbidden_link(*_args: object, **_kwargs: object) -> None:
            raise AssertionError("protected WAL snapshots must copy bytes, not link source inodes")

        monkeypatch.setattr(input_reads.os, "link", forbidden_link)
        payload = input_reads.sqlite_snapshot_bytes(
            root,
            "run.sqlite",
            max_bytes=1_000_000,
        )
    finally:
        source.close()
        root.close()

    memory = sqlite3.connect(":memory:")
    try:
        memory.deserialize(payload)
        assert memory.execute("SELECT value FROM items").fetchone() == ("wal-anchored",)
    finally:
        memory.close()


def test_review_workspace_legacy_mode_remains_available_without_posix_read_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    review_dir = tmp_path / "review"
    review_dir.mkdir()
    database = review_dir / "run.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE items (value TEXT NOT NULL)")
        connection.execute("INSERT INTO items(value) VALUES ('legacy')")

    monkeypatch.setattr(
        review_workspace_module,
        "secure_read_primitives_available",
        lambda: False,
    )
    with review_workspace_module.ReviewWorkspace.open(tmp_path, protected=False) as legacy:
        assert legacy.query("SELECT value FROM items").rows == [{"value": "legacy"}]
    with review_workspace_module.ReviewWorkspace.open(tmp_path) as default:
        assert not default._protected
        assert default.query("SELECT value FROM items").rows == [{"value": "legacy"}]
    def unavailable(*_args: object, **_kwargs: object) -> object:
        raise input_reads.SecureReadUnavailable("simulated POSIX descriptor gap")

    monkeypatch.setattr(review_workspace_module.SecureReadRoot, "open", unavailable)
    with pytest.raises(ReviewStoreNotFoundError):
        review_workspace_module.ReviewWorkspace.open(tmp_path, protected=True)
