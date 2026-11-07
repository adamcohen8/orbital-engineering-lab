from __future__ import annotations

import os
from pathlib import Path

import pytest

import sim.security.output_writes as shared_output_writes
from integrations.oel_mcp.execution import complete_manifest
from integrations.oel_mcp.secure_writes import (
    SecureWriteConflict,
    SecureWriteRoot,
    SecureWriteUnavailable,
    bind_output_writer,
    copy_output_file,
    current_output_writer,
    mkdir_output,
    open_output,
    read_output_bytes,
    staged_output,
    staged_output_file,
    update_output_text,
    write_output_text,
)


def test_mcp_exports_use_shared_engine_writer() -> None:
    assert SecureWriteRoot is shared_output_writes.SecureOutputRoot
    assert shared_output_writes.SecureOutputWriter is SecureWriteRoot


def test_staging_parent_is_always_server_private(tmp_path: Path) -> None:
    with pytest.raises(PermissionError, match="server-private"):
        with staged_output(
            tmp_path / "output",
            approved_roots=(tmp_path,),
            staging_parent=tmp_path,
        ):
            pass


def test_failed_exclusive_write_removes_partial_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Historical node name; failed entries now remain fail-closed and exclusive."""

    output = tmp_path / "output"
    with SecureWriteRoot.open(output, approved_roots=(tmp_path,)) as writer:
        original = shared_output_writes._write_all

        def fail_after_open(fd: int, payload: bytes) -> None:
            original(fd, payload[:1])
            raise OSError("simulated producer write failure")

        monkeypatch.setattr(shared_output_writes, "_write_all", fail_after_open)
        with pytest.raises(OSError, match="simulated producer write failure"):
            writer.write_bytes("partial.bin", b"partial")
        assert (output / "partial.bin").read_bytes() == b"p"
        with pytest.raises(SecureWriteConflict):
            writer.write_bytes("partial.bin", b"retry")


def test_writer_contract_preserves_logical_root_and_descriptor_file_api(tmp_path: Path) -> None:
    output = tmp_path / "output"
    source = tmp_path / "private-stage.bin"
    source.write_bytes(b"published\n")
    with SecureWriteRoot.open(output, approved_roots=(tmp_path,), require_empty=True) as writer:
        assert writer.root == output
        assert writer.logical_root == output
        with bind_output_writer(writer):
            assert current_output_writer() is writer
        writer.mkdir("nested")
        with writer.open("nested/data.bin", "wb") as stream:
            stream.write(b"writer\n")
        assert writer.read_bytes("nested/data.bin") == b"writer\n"
        assert writer.sha256("nested/data.bin")
        writer.publish_file("published.bin", source)
        writer.write_text("operation.json", "initial\n")
        writer.update_existing_bytes("operation.json", b"final\n")
    assert (output / "published.bin").read_bytes() == b"published\n"
    assert (output / "operation.json").read_bytes() == b"final\n"


def test_staged_writer_keeps_final_logical_root(tmp_path: Path) -> None:
    output = tmp_path / "output"
    with staged_output(output, approved_roots=(tmp_path,)) as transaction:
        assert transaction.staging.logical_root == output
        transaction.staging.write_text("result.txt", "ok\n")


def test_writer_carries_operation_identity_into_staging(tmp_path: Path) -> None:
    output = tmp_path / "output"
    with staged_output(
        output,
        approved_roots=(tmp_path,),
        active_opid="op-test-123",
    ) as transaction:
        assert transaction.active_opid == "op-test-123"
        assert transaction.final.active_opid == "op-test-123"
        assert transaction.staging.active_opid == "op-test-123"


def test_manifest_hashes_physical_staging_source_under_logical_path(tmp_path: Path) -> None:
    logical = tmp_path / "published" / "result.json"
    physical = tmp_path / "private-stage" / "result.json"
    physical.parent.mkdir()
    physical.write_bytes(b'{"ok":true}\n')

    manifest = complete_manifest(
        {},
        status="completed",
        artifacts=[str(logical)],
        artifact_files=[str(physical)],
    )

    assert manifest["artifacts"] == [str(logical)]
    assert manifest["artifact_records"][0]["path"] == str(logical)
    assert manifest["artifact_records"][0]["bytes"] == physical.stat().st_size
    assert manifest["artifact_records"][0]["sha256"]
    assert manifest["artifacts_complete"] is True


def test_context_helpers_bind_logical_paths_and_preserve_single_file_bytes(tmp_path: Path) -> None:
    output = tmp_path / "output"
    source = tmp_path / "source.bin"
    source.write_bytes(b"copied\n")
    with SecureWriteRoot.open(output, approved_roots=(tmp_path,), require_empty=True) as writer:
        with bind_output_writer(writer):
            write_output_text(output / "nested" / "result.txt", "helper\n")
            assert read_output_bytes(output / "nested" / "result.txt") == b"helper\n"
            copy_output_file(source, output / "copied.bin")
            with open_output(output / "stream.bin", "wb") as stream:
                stream.write(b"stream\n")
            write_output_text(output / "operation.json", "initial\n")
            update_output_text(output / "operation.json", "final\n")
            with staged_output_file(output / "single.bin") as physical:
                physical.write_bytes(b"content-bound\n")
            with pytest.raises(PermissionError, match="outside the bound logical root"):
                write_output_text(tmp_path / "outside.txt", "rejected\n")
    assert (output / "copied.bin").read_bytes() == b"copied\n"
    assert (output / "operation.json").read_bytes() == b"final\n"
    assert (output / "single.bin").read_bytes() == b"content-bound\n"


def test_helpers_keep_cli_fallback_without_bound_writer(tmp_path: Path) -> None:
    target = tmp_path / "fallback" / "result.txt"
    write_output_text(target, "fallback\n")
    write_output_text(target, "overwritten\n")
    assert target.read_text(encoding="utf-8") == "overwritten\n"


def test_owned_manifest_update_rejects_inode_swap_and_hardlink(tmp_path: Path) -> None:
    output = tmp_path / "output"
    with SecureWriteRoot.open(output, approved_roots=(tmp_path,), require_empty=True) as writer:
        writer.write_text("manifest.json", "owned\n")
        manifest = output / "manifest.json"
        manifest.unlink()
        manifest.write_text("replacement\n", encoding="utf-8")
        with pytest.raises(PermissionError, match="inode"):
            writer.update_existing_bytes("manifest.json", b"must reject\n")
        assert manifest.read_text(encoding="utf-8") == "replacement\n"

    output = tmp_path / "hardlink-output"
    with SecureWriteRoot.open(output, approved_roots=(tmp_path,), require_empty=True) as writer:
        writer.write_text("manifest.json", "owned\n")
        os.link(output / "manifest.json", output / "manifest-hardlink.json")
        with pytest.raises(PermissionError, match="inode"):
            writer.update_existing_bytes("manifest.json", b"must reject\n")
        assert (output / "manifest.json").read_text(encoding="utf-8") == "owned\n"


def test_owned_inode_descriptors_close_with_the_writer(tmp_path: Path) -> None:
    with SecureWriteRoot.open(tmp_path / "output", approved_roots=(tmp_path,)) as writer:
        writer.write_bytes("one", b"one")
        with writer.open("two", "xb") as stream:
            stream.write(b"two")
        retained = tuple(writer._owned_file_fds.values())
        assert len(retained) == 2
        for descriptor in retained:
            assert os.fstat(descriptor).st_nlink == 1
    for descriptor in retained:
        with pytest.raises(OSError):
            os.fstat(descriptor)


def test_mkdir_helper_accepts_already_open_logical_root(tmp_path: Path) -> None:
    output = tmp_path / "output"
    with SecureWriteRoot.open(output, approved_roots=(tmp_path,), require_empty=True) as writer:
        with bind_output_writer(writer):
            assert mkdir_output(output) == output
            assert mkdir_output(output / "nested") == output / "nested"


def test_secure_root_requires_explicit_approved_root(tmp_path: Path) -> None:
    with pytest.raises(PermissionError, match="approved write root"):
        SecureWriteRoot.open(tmp_path / "output", approved_roots=())


def test_secure_root_creates_exclusive_nested_files(tmp_path: Path) -> None:
    with SecureWriteRoot.open(
        tmp_path / "output", approved_roots=(tmp_path,), require_empty=True
    ) as writer:
        target = writer.write_text("nested/result.json", "{}\n")
        assert target.read_text(encoding="utf-8") == "{}\n"
        with pytest.raises(SecureWriteConflict):
            writer.write_text("nested/result.json", "changed\n")


def test_persistent_handle_is_not_rerouted_by_path_replacement(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    replacement = tmp_path / "replacement"
    replacement.mkdir()

    with SecureWriteRoot.open(output, approved_roots=(tmp_path,)) as writer:
        moved = tmp_path / "moved-output"
        output.rename(moved)
        output.symlink_to(replacement, target_is_directory=True)
        written = writer.write_text("survives.txt", "descriptor-bound\n")

    assert written == output / "survives.txt"
    assert (moved / "survives.txt").read_text(encoding="utf-8") == "descriptor-bound\n"
    assert not (replacement / "survives.txt").exists()


def test_staged_output_rejects_symlinks_and_publishes_regular_files(tmp_path: Path) -> None:
    outside = tmp_path / "outside.txt"
    outside.write_text("must remain unchanged\n", encoding="utf-8")
    output = tmp_path / "output"

    with pytest.raises(PermissionError, match="symlink"):
        with staged_output(output, approved_roots=(tmp_path,)) as transaction:
            (transaction.staging_path / "safe.txt").write_text("safe\n", encoding="utf-8")
            (transaction.staging_path / "escape").symlink_to(outside)

    assert not (output / "safe.txt").exists()
    assert outside.read_text(encoding="utf-8") == "must remain unchanged\n"


def test_staged_output_publishes_without_overwriting_existing_entries(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    (output / "existing.txt").write_text("original\n", encoding="utf-8")

    with pytest.raises(SecureWriteConflict):
        with staged_output(output, approved_roots=(tmp_path,), require_empty=False) as transaction:
            (transaction.staging_path / "existing.txt").write_text("replacement\n", encoding="utf-8")

    assert (output / "existing.txt").read_text(encoding="utf-8") == "original\n"


def test_publish_file_rejects_symlink_source(tmp_path: Path) -> None:
    output = tmp_path / "output"
    outside = tmp_path / "outside.txt"
    outside.write_text("outside\n", encoding="utf-8")
    source = tmp_path / "source.txt"
    source.symlink_to(outside)
    with SecureWriteRoot.open(output, approved_roots=(tmp_path,), require_empty=True) as writer:
        with pytest.raises(PermissionError, match="regular source"):
            writer.publish_file("copied.txt", source)
    assert not (output / "copied.txt").exists()


@pytest.mark.skipif(os.name != "posix", reason="secure descriptor primitives are POSIX-only")
def test_secure_writer_is_available_on_supported_posix_hosts(tmp_path: Path) -> None:
    try:
        with SecureWriteRoot.open(tmp_path / "output", approved_roots=(tmp_path,)):
            pass
    except SecureWriteUnavailable as exc:  # pragma: no cover - host capability guard
        pytest.fail(str(exc))
