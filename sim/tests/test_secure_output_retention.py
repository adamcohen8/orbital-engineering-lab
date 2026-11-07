"""Fail-closed retention regressions for descriptor-bound output writes."""

from __future__ import annotations

from pathlib import Path

import pytest

import sim.security.output_writes as output_writes
from sim.security.output_writes import SecureWriteConflict, SecureWriteRoot, staged_output


def test_direct_write_retains_replacement_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "output"
    target = output / "direct.bin"

    def fail_after_replacement(_fd: int, _payload: bytes) -> None:
        target.unlink()
        target.write_bytes(b"replacement")
        raise OSError("simulated direct-write failure")

    with SecureWriteRoot.open(output, approved_roots=(tmp_path,), require_empty=True) as writer:
        monkeypatch.setattr(output_writes, "_write_all", fail_after_replacement)
        with pytest.raises(OSError, match="simulated direct-write failure"):
            writer.write_bytes("direct.bin", b"partial")
        assert target.read_bytes() == b"replacement"
        with pytest.raises(SecureWriteConflict):
            writer.write_bytes("direct.bin", b"silent retry")


def test_stream_write_retains_replacement_on_failure(tmp_path: Path) -> None:
    output = tmp_path / "output"
    target = output / "stream.bin"

    with SecureWriteRoot.open(output, approved_roots=(tmp_path,), require_empty=True) as writer:
        with pytest.raises(RuntimeError, match="simulated stream failure"):
            with writer.open("stream.bin", "xb") as stream:
                stream.write(b"partial")
                target.unlink()
                target.write_bytes(b"replacement")
                raise RuntimeError("simulated stream failure")
        assert target.read_bytes() == b"replacement"
        with pytest.raises(SecureWriteConflict):
            with writer.open("stream.bin", "xb"):
                pass


def test_single_file_publication_retains_replacement_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"source")
    output = tmp_path / "output"
    target = output / "published.bin"

    def fail_after_replacement(_fd: int, _payload: bytes) -> None:
        target.unlink()
        target.write_bytes(b"replacement")
        raise OSError("simulated publication failure")

    with SecureWriteRoot.open(output, approved_roots=(tmp_path,), require_empty=True) as writer:
        monkeypatch.setattr(output_writes, "_write_all", fail_after_replacement)
        with pytest.raises(OSError, match="simulated publication failure"):
            writer.publish_file("published.bin", source)
        assert target.read_bytes() == b"replacement"
        with pytest.raises(SecureWriteConflict):
            writer.publish_file("published.bin", source)


def test_multi_file_publication_retains_replacement_after_later_collision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "output"
    output.mkdir()
    collision = output / "second.bin"
    collision.write_bytes(b"existing")
    replacement = output / "first.bin"
    swapped = False
    original_write_all = output_writes._write_all

    def replace_first_after_copy(fd: int, payload: bytes) -> None:
        nonlocal swapped
        original_write_all(fd, payload)
        if not swapped:
            replacement.unlink()
            replacement.write_bytes(b"replacement")
            swapped = True

    with pytest.raises(SecureWriteConflict):
        with staged_output(output, approved_roots=(tmp_path,), require_empty=False) as transaction:
            (transaction.staging_path / "first.bin").write_bytes(b"first")
            (transaction.staging_path / "second.bin").write_bytes(b"second")
            monkeypatch.setattr(output_writes, "_write_all", replace_first_after_copy)

    assert replacement.read_bytes() == b"replacement"
    assert collision.read_bytes() == b"existing"
    with SecureWriteRoot.open(output, approved_roots=(tmp_path,), require_empty=False) as writer:
        with pytest.raises(SecureWriteConflict):
            writer.write_bytes("first.bin", b"silent retry")
