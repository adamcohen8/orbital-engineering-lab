from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from integrations.oel_mcp.execution import complete_manifest
from sim.security.output_writes import SecureWriteError, SecureWriteRoot, staged_output


def test_manifest_hashes_retained_directory_after_path_substitution(tmp_path: Path) -> None:
    output = tmp_path / "output"
    replacement = tmp_path / "replacement"
    replacement.mkdir()
    payload = b"approved evidence\n"
    with SecureWriteRoot.open(output, approved_roots=(tmp_path,)) as writer:
        writer.write_bytes("result.bin", payload)
        writer.write_bytes("nested/second.bin", b"second")
        output.rename(tmp_path / "moved")
        output.symlink_to(replacement, target_is_directory=True)
        (replacement / "result.bin").write_bytes(b"substituted")
        assert writer.list_files() == (Path("nested/second.bin"), Path("result.bin"))
        manifest: dict = {}
        complete_manifest(
            manifest, status="completed", artifacts=[str(output / "result.bin")], writer=writer
        )
    assert manifest["artifacts_complete"]
    assert manifest["artifact_records"] == [{
        "path": str(output / "result.bin"), "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }]


def test_inventory_rejects_symlink_in_output(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    with SecureWriteRoot.open(tmp_path / "output", approved_roots=(tmp_path,)) as writer:
        (writer.root / "escape").symlink_to(outside, target_is_directory=True)
        with pytest.raises(SecureWriteError, match="regular files and directories"):
            writer.list_files()


def test_evidence_hash_rejects_fifo_without_waiting_for_writer(tmp_path: Path) -> None:
    with SecureWriteRoot.open(tmp_path / "output", approved_roots=(tmp_path,)) as writer:
        fifo = writer.root / "fifo"
        os.mkfifo(fifo)
        try:
            with pytest.raises(SecureWriteError, match="regular file"):
                writer.file_record("fifo")
            with pytest.raises(SecureWriteError, match="regular file"):
                writer.read_bytes("fifo")
        finally:
            fifo.unlink()


@pytest.mark.parametrize("directory_publication", [False, True])
def test_private_publication_preserves_only_owner_execute_permission(
    tmp_path: Path, directory_publication: bool
) -> None:
    source = tmp_path / "source"
    source.write_bytes(b"#!/bin/sh\nexit 0\n")
    source.chmod(0o777)
    output = tmp_path / "output"
    if directory_publication:
        with staged_output(output, approved_roots=(tmp_path,)) as transaction:
            staged = transaction.staging_path / "executable"
            staged.write_bytes(source.read_bytes())
            staged.chmod(0o777)
    else:
        with SecureWriteRoot.open(output, approved_roots=(tmp_path,)) as writer:
            writer.publish_file("executable", source)
    assert (output / "executable").stat().st_mode & 0o777 == 0o700
