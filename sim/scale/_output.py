"""Logical Scale output paths bound to a descriptor-backed writer.

Scale producers keep their existing logical paths in payloads and manifests while
the caller controls the physical destination through ``OutputWriter``.  The
legacy path-only behavior remains available when ``writer`` is ``None``.
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any, Iterator

from sim.security.output_writes import (
    OutputWriter,
    SecureWriteConflict,
    copy_output_file,
    rewrite_staged_paths,
    sha256_output,
    write_output_bytes,
    write_output_text,
)
from sim.security.output_writes import (
    bind_output_writer as bind_writer_context,
)


def _absolute_lexical(value: str | Path) -> Path:
    """Normalize an absolute path without following producer symlinks."""

    path = Path(value).expanduser()
    if not path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts[1:]):
        raise ValueError("Scale output paths must be absolute and normalized.")
    return path


class ScaleOutputBinding:
    """Map Scale logical paths below a writer's logical root to its fd root."""

    def __init__(self, writer: OutputWriter, logical_root: str | Path) -> None:
        self.writer = writer
        self.logical_root = _absolute_lexical(logical_root)
        self.writer_logical_root = _absolute_lexical(writer.logical_root)
        self.physical_root = _absolute_lexical(writer.path)
        try:
            self.logical_root.relative_to(self.writer_logical_root)
        except ValueError as exc:
            raise ValueError(
                "Scale output logical root must be within the bound writer logical root."
            ) from exc

    def _relative(self, logical_path: str | Path) -> Path:
        path = _absolute_lexical(logical_path)
        try:
            return path.relative_to(self.writer_logical_root)
        except ValueError as exc:
            raise ValueError("Scale output path is outside the bound writer logical root.") from exc

    def physical(self, logical_path: str | Path) -> Path:
        return self.physical_root / self._relative(logical_path)

    def write_bytes(self, logical_path: str | Path, payload: bytes) -> Path:
        path = _absolute_lexical(logical_path)
        relative = self._relative(path)
        with bind_writer_context(self.writer):
            try:
                write_output_bytes(path, payload)
            except SecureWriteConflict:
                self.writer.update_existing_bytes(relative, payload)
        return path

    def write_text(self, logical_path: str | Path, text: str, *, encoding: str = "utf-8") -> Path:
        path = _absolute_lexical(logical_path)
        relative = self._relative(path)
        with bind_writer_context(self.writer):
            try:
                write_output_text(path, text, encoding=encoding)
            except SecureWriteConflict:
                self.writer.update_existing_bytes(relative, text.encode(encoding))
        return path

    def sha256(self, logical_path: str | Path) -> str:
        with bind_writer_context(self.writer):
            return sha256_output(_absolute_lexical(logical_path))

    def publish_file(self, logical_path: str | Path, source: str | Path) -> Path:
        path = _absolute_lexical(logical_path)
        with bind_writer_context(self.writer):
            copy_output_file(source, path)
        return path


def bind_output_writer(
    writer: OutputWriter | None,
    logical_root: str | Path,
) -> ScaleOutputBinding | None:
    if writer is None:
        return None
    return ScaleOutputBinding(writer, logical_root)


def write_bytes(
    path: str | Path,
    payload: bytes,
    *,
    writer: ScaleOutputBinding | None = None,
) -> Path:
    if writer is not None:
        return writer.write_bytes(path, payload)
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)
    return target


def write_text(
    path: str | Path,
    text: str,
    *,
    writer: ScaleOutputBinding | None = None,
    encoding: str = "utf-8",
) -> Path:
    return write_bytes(path, text.encode(encoding), writer=writer)


def physical_path(path: str | Path, *, writer: ScaleOutputBinding | None = None) -> Path:
    if writer is None:
        return Path(path).expanduser().resolve()
    return writer.physical(path)


def logicalize(value: Any, *, writer: ScaleOutputBinding | None = None) -> Any:
    if writer is None:
        return value
    return rewrite_staged_paths(value, writer.physical_root, writer.writer_logical_root)


def store_read_path(path: str | Path, *, writer: OutputWriter | None = None) -> Path:
    """Resolve a store for reads without redirecting source stores.

    A writer-bound logical path is mapped to its physical descriptor root. A
    source path outside that root remains an ordinary read-only path; mutation
    callers must use :func:`staged_sqlite_store`, which rejects that case.
    """

    if writer is None:
        return Path(path).expanduser().resolve()
    logical = _absolute_lexical(path)
    binding = ScaleOutputBinding(writer, writer.logical_root)
    try:
        return binding.physical(logical)
    except ValueError:
        return logical


@contextlib.contextmanager
def staged_sqlite_store(
    path: str | Path,
    *,
    writer: OutputWriter | None = None,
) -> Iterator[Path]:
    """Mutate one derived SQLite store in private staging, then publish bytes.

    The logical path stays in Scale configs and returned metadata. When a
    writer is bound, only a store below its logical root is eligible for
    mutation. Existing output bytes are seeded into a private SQLite file and
    refreshed through the writer's owned-inode update check; a new store is
    published exclusively. An input store outside the writer root is never
    opened for mutation.
    """

    if writer is None:
        yield Path(path).expanduser().resolve()
        return

    logical = _absolute_lexical(path)
    binding = ScaleOutputBinding(writer, writer.logical_root)
    relative = binding._relative(logical)
    staging = writer.create_staging(prefix=".oel-scale-store-")
    stage_path = staging.path / relative
    stage_path.parent.mkdir(parents=True, exist_ok=True)
    existing = False
    try:
        try:
            seed = writer.read_bytes(relative)
        except FileNotFoundError:
            seed = None
        if seed is not None:
            stage_path.write_bytes(seed)
            existing = True
        yield stage_path
        payload = stage_path.read_bytes()
        if existing:
            replace = getattr(writer, "replace_owned_bytes", None)
            if not callable(replace):
                replace = writer.update_existing_bytes
            replace(relative, payload)
        else:
            writer.publish_file(relative, stage_path)
    finally:
        if not staging._closed:
            staging.remove_tree()
