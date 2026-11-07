"""Bounded identity and source-stability regressions for secure output writes."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

import sim.security.output_writes as output_writes
from sim.security.output_writes import (
    SecureWriteConflict,
    SecureWriteError,
    SecureWriteRoot,
    staged_output,
)


def test_remove_tree_retains_replaced_root_name(tmp_path: Path) -> None:
    output = tmp_path / "output"
    with SecureWriteRoot.open(output, approved_roots=(tmp_path,), require_empty=True) as writer:
        writer.write_text("original.txt", "original\n")
        moved = tmp_path / "moved-output"
        output.rename(moved)
        replacement = tmp_path / "output"
        replacement.mkdir()

        writer.remove_tree()

    assert replacement.is_dir()
    assert not any(moved.iterdir())


def test_remove_tree_retains_nested_name_replaced_during_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "output"
    with SecureWriteRoot.open(output, approved_roots=(tmp_path,), require_empty=True) as writer:
        writer.write_text("nested/original.txt", "original\n")
        nested = output / "nested"
        moved = tmp_path / "moved-nested"
        original_remove = output_writes._remove_owned_entry
        replaced = False

        def replace_before_nested_remove(
            parent_fd: int,
            name: str,
            expected_identity: tuple[int, int] | None,
            *,
            directory: bool,
        ) -> bool:
            nonlocal replaced
            if directory and name == "nested" and not replaced:
                nested.rename(moved)
                nested.mkdir()
                replaced = True
            return original_remove(
                parent_fd,
                name,
                expected_identity,
                directory=directory,
            )

        monkeypatch.setattr(output_writes, "_remove_owned_entry", replace_before_nested_remove)

        writer.remove_tree()

    assert nested.is_dir()
    assert not any(nested.iterdir())
    assert not any(moved.iterdir())


def test_private_staging_cleanup_retains_replaced_parent(tmp_path: Path) -> None:
    output = tmp_path / "output"
    writer = SecureWriteRoot.open(output, approved_roots=(tmp_path,), require_empty=True)
    staging = writer.create_staging(prefix=".oel-stability-")
    private_parent = staging._private_parent
    assert private_parent is not None
    moved = private_parent.with_name(private_parent.name + "-moved")
    private_parent.rename(moved)
    replacement = private_parent
    replacement.mkdir()

    try:
        staging.remove_tree()
        assert replacement.is_dir()
        assert (moved / "payload").is_dir()
        assert not any((moved / "payload").iterdir())
    finally:
        writer.close()
        shutil.rmtree(moved, ignore_errors=True)
        shutil.rmtree(replacement, ignore_errors=True)


def test_publish_file_rejects_inode_replacement_after_checked_stat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"original")
    output = tmp_path / "output"
    target = output / "published.bin"
    original_stat = output_writes.os.stat
    original_supports_dir_fd = output_writes.os.supports_dir_fd
    swapped = False

    def replace_after_check(path: object, *args: object, **kwargs: object) -> object:
        nonlocal swapped
        result = original_stat(path, *args, **kwargs)
        if path == source.name and kwargs.get("dir_fd") is not None and not swapped:
            source.unlink()
            source.write_bytes(b"replacement")
            swapped = True
        return result

    with SecureWriteRoot.open(output, approved_roots=(tmp_path,), require_empty=True) as writer:
        monkeypatch.setattr(output_writes.os, "stat", replace_after_check)
        # ``secure_write_primitives_available`` compares these callables by
        # identity.  Keep the capability check true while replacing only the
        # stat seam used to inject the checked-stat/opened-inode race.
        monkeypatch.setattr(
            output_writes.os,
            "supports_dir_fd",
            set(original_supports_dir_fd) | {replace_after_check},
        )
        with pytest.raises(SecureWriteError, match="changed before"):
            writer.publish_file("published.bin", source)

    assert not target.exists()
    assert source.read_bytes() == b"replacement"


@pytest.mark.parametrize("same_metadata", [False, True])
def test_publish_file_retains_partial_output_when_source_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, same_metadata: bool
) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"original")
    output = tmp_path / "output"
    target = output / "published.bin"
    original_write_all = output_writes._write_all
    mutated = False

    def mutate_after_copy(fd: int, payload: bytes) -> None:
        nonlocal mutated
        original_write_all(fd, payload)
        if not mutated:
            source.write_bytes(b"changed!")
            mutated = True

    with SecureWriteRoot.open(output, approved_roots=(tmp_path,), require_empty=True) as writer:
        monkeypatch.setattr(output_writes, "_write_all", mutate_after_copy)
        if same_metadata:
            # Force the coarse-clock case even on a nanosecond filesystem.
            monkeypatch.setattr(output_writes, "_entry_matches_snapshot", lambda *_args: True)
        with pytest.raises(SecureWriteError, match="changed during"):
            writer.publish_file("published.bin", source)
        assert target.read_bytes() == b"original"
        with pytest.raises(SecureWriteConflict):
            writer.publish_file("published.bin", source)


def test_staged_publication_rejects_late_directory_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "output"
    original_listdir = output_writes.os.listdir
    injected = False
    private_parent: Path | None = None
    staging_fd: int | None = None

    with pytest.raises(SecureWriteError, match="directory entries changed"):
        with staged_output(output, approved_roots=(tmp_path,)) as transaction:
            private_parent = transaction.staging._private_parent
            staging_fd = transaction.staging.fd
            (transaction.staging_path / "first.txt").write_bytes(b"first")

            def add_late_entry(fd: int | str | Path) -> list[str]:
                nonlocal injected
                names = original_listdir(fd)
                if fd == staging_fd and not injected:
                    (transaction.staging_path / "late.txt").write_bytes(b"late")
                    injected = True
                return names

            monkeypatch.setattr(output_writes.os, "listdir", add_late_entry)

    # The staging descriptor is closed by the context manager.  Restore the
    # listdir seam before cleanup and the final output-directory assertion.
    monkeypatch.undo()
    if private_parent is not None:
        shutil.rmtree(private_parent, ignore_errors=True)
    assert output.is_dir()
    assert not any(output.iterdir())
