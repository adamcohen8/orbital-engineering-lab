"""Fail-closed, descriptor-bound output writes for OEL producers.

The caller answers *which* roots an operation may use.  This module answers
*how* bytes are written after that decision: an approved directory is
opened component by component with no-follow flags, the resulting descriptor
is retained for the operation, and new files are created with ``O_EXCL``.

Producers that only accept a filesystem path can use :func:`staged_output`.
They write into a private child directory and the transaction publishes the
result through the retained destination descriptor.  Publication rejects
symlinks, special files, path collisions, and traversal attempts.

The descriptor primitives are intentionally fail-closed.  They currently
require POSIX ``dir_fd`` support, ``O_DIRECTORY``, and ``O_NOFOLLOW``.  A
caller must not silently fall back to path-only checks on a platform that does
not provide those primitives.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import shutil
import stat
import tempfile
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO, ContextManager, Iterator, Mapping, Protocol, Sequence


class SecureWriteError(PermissionError):
    """Base error for a rejected or unavailable secure write operation."""


class SecureWriteUnavailable(SecureWriteError):
    """The host does not expose the descriptor primitives required here."""


class SecureWriteConflict(FileExistsError, SecureWriteError):
    """A publication would replace an existing destination entry."""


class OutputWriter(Protocol):
    """Minimal producer contract for protected logical/physical output."""

    path: Path
    active_opid: str | None

    @property
    def root(self) -> Path: ...

    @property
    def logical_root(self) -> Path: ...

    def write_bytes(
        self,
        relative: str | Path,
        payload: bytes,
        *,
        mode: int = 0o600,
        parents: bool = True,
    ) -> Path: ...

    def write_text(
        self, relative: str | Path, text: str, *, encoding: str = "utf-8"
    ) -> Path: ...

    def open(
        self, relative: str | Path, mode: str = "xb", *, permissions: int = 0o600
    ) -> ContextManager[BinaryIO]: ...

    def mkdir(self, relative: str | Path) -> Path: ...

    def read_bytes(self, relative: str | Path, *, max_bytes: int | None = None) -> bytes: ...

    def sha256(self, relative: str | Path) -> str: ...

    def file_record(self, relative: str | Path) -> dict[str, Any]: ...

    def list_files(self) -> tuple[Path, ...]: ...

    def publish_file(self, relative: str | Path, source: str | Path) -> Path: ...

    def update_existing_bytes(self, relative: str | Path, payload: bytes) -> Path: ...

    def adopt_existing(self, relative: str | Path) -> Path: ...


_ACTIVE_OUTPUT_WRITER: ContextVar[OutputWriter | None] = ContextVar(
    "oel_active_output_writer",
    default=None,
)

# Private staging handles are kept only long enough to associate a successful
# single-file publication with the stage that supplied its source.  This lets
# path-facing helpers clean their normal successful stage without weakening
# identity checks for unknown entries.
_STAGING_HANDLES: dict[Path, Any] = {}


@contextlib.contextmanager
def bind_output_writer(writer: OutputWriter) -> Iterator[OutputWriter]:
    """Bind one protected writer for strict producer helpers in this context."""

    token = _ACTIVE_OUTPUT_WRITER.set(writer)
    try:
        yield writer
    finally:
        _ACTIVE_OUTPUT_WRITER.reset(token)


def current_output_writer() -> OutputWriter | None:
    """Return the context-bound writer, if an owner supplied one."""

    return _ACTIVE_OUTPUT_WRITER.get()


def require_output_writer(*, logical_root: str | Path | None = None) -> OutputWriter:
    """Require a context-bound writer and optionally verify its logical root."""

    writer = current_output_writer()
    if writer is None:
        raise SecureWriteUnavailable("This producer requires a descriptor-bound output writer.")
    if logical_root is not None and _absolute_lexical(logical_root) != _absolute_lexical(writer.logical_root):
        raise SecureWriteError("The producer logical output root does not match the bound writer.")
    return writer


def secure_write_primitives_available() -> bool:
    """Return whether this host exposes every primitive used by the writer."""

    required = ("O_DIRECTORY", "O_NOFOLLOW")
    if os.name != "posix" or any(not hasattr(os, name) for name in required):
        return False
    dir_fd_functions = getattr(os, "supports_dir_fd", set())
    return all(
        function in dir_fd_functions
        for function in (os.open, os.mkdir, os.stat, os.unlink, os.rmdir)
    )


def require_secure_write_primitives() -> None:
    if not secure_write_primitives_available():
        raise SecureWriteUnavailable(
            "Secure output writes require POSIX directory-descriptor, "
            "O_DIRECTORY, O_NOFOLLOW, and dir_fd support on this host."
        )


def _absolute_lexical(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError("Secure output paths must be absolute.")
    if any(part in {"", ".", ".."} for part in path.parts[1:]):
        raise ValueError("Secure output paths must be normalized and contain no traversal components.")
    return path


def _relative_parts(value: str | Path) -> tuple[str, ...]:
    path = Path(value)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("Secure output child paths must be relative and contain no traversal components.")
    return tuple(path.parts)


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _logical_path(value: str | Path, *, root: str | Path | None = None) -> Path:
    """Normalize a producer path without resolving symlinks."""

    path = Path(value).expanduser()
    if not path.is_absolute():
        if root is None:
            path = Path.cwd() / path
        else:
            path = Path(root).joinpath(path)
    return _absolute_lexical(path)


def _writer_relative_path(writer: OutputWriter, value: str | Path) -> Path:
    """Map one logical producer path to a writer-relative path."""

    logical_root = _absolute_lexical(writer.logical_root)
    path = _logical_path(value, root=logical_root)
    try:
        relative = path.relative_to(logical_root)
    except ValueError as exc:
        raise SecureWriteError(
            f"Output path {path} is outside the bound logical root {logical_root}."
        ) from exc
    if not relative.parts:
        raise SecureWriteError("A producer must target a file or child directory below its logical root.")
    return relative


def _active_writer_target(value: str | Path) -> tuple[OutputWriter | None, Path, Path]:
    """Return the active writer, relative target, and legacy absolute target."""

    writer = current_output_writer()
    if writer is None:
        return None, Path(), _logical_path(value)
    relative = _writer_relative_path(writer, value)
    return writer, relative, writer.logical_root.joinpath(relative)


def write_output_bytes(
    path: str | Path,
    payload: bytes,
    *,
    mode: int = 0o600,
    parents: bool = True,
) -> Path:
    """Create one output, using the bound descriptor writer when present.

    With no bound writer this keeps the ordinary local/CLI overwrite behavior.
    MCP handlers bind a writer before calling producer code, so an out-of-root
    path is rejected rather than falling back to a path-only write.
    """

    writer, relative, legacy = _active_writer_target(path)
    if writer is not None:
        return writer.write_bytes(relative, payload, mode=mode, parents=parents)
    target = legacy
    if parents:
        target.parent.mkdir(parents=True, exist_ok=True)
    try:
        # The unbound helper is the legacy CLI path.  Keep its historical
        # overwrite behavior; an active MCP writer always takes the exclusive
        # descriptor path above.
        with target.open("wb") as stream:
            stream.write(bytes(payload))
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as exc:
        raise SecureWriteConflict(f"Output already exists: {target}") from exc
    return target


def write_output_text(
    path: str | Path,
    text: str,
    *,
    encoding: str = "utf-8",
    mode: int = 0o600,
    parents: bool = True,
) -> Path:
    """Create one encoded output through the active writer boundary."""

    writer = current_output_writer()
    if writer is not None:
        relative = _writer_relative_path(writer, path)
        return writer.write_bytes(relative, text.encode(encoding), mode=mode, parents=parents)
    return write_output_bytes(path, text.encode(encoding), mode=mode, parents=parents)


def update_output_bytes(path: str | Path, payload: bytes) -> Path:
    """Update only an inode created by the active writer operation."""

    writer = current_output_writer()
    if writer is not None:
        return writer.update_existing_bytes(_writer_relative_path(writer, path), payload)
    target = _logical_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(bytes(payload))
    return target


def update_output_text(
    path: str | Path,
    text: str,
    *,
    encoding: str = "utf-8",
) -> Path:
    """Update one operation-owned text manifest through the active writer."""

    return update_output_bytes(path, text.encode(encoding))


@contextlib.contextmanager
def open_output(
    path: str | Path,
    mode: str = "xb",
    *,
    permissions: int = 0o600,
) -> Iterator[BinaryIO]:
    """Open one binary output through the active writer boundary."""

    writer = current_output_writer()
    if writer is not None:
        relative = _writer_relative_path(writer, path)
        with writer.open(relative, mode, permissions=permissions) as stream:
            yield stream
        return

    target = _logical_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with target.open(mode) as stream:
            yield stream
    except FileExistsError as exc:
        raise SecureWriteConflict(f"Output already exists: {target}") from exc


def mkdir_output(path: str | Path) -> Path:
    """Create one child output directory through the active writer boundary."""

    writer = current_output_writer()
    if writer is not None:
        logical_root = _absolute_lexical(writer.logical_root)
        target = _logical_path(path, root=logical_root)
        if target == logical_root:
            return writer.root
        return writer.mkdir(_writer_relative_path(writer, target))
    target = _logical_path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def read_output_bytes(path: str | Path, *, max_bytes: int | None = None) -> bytes:
    """Read an output through the descriptor writer when one is bound."""

    writer = current_output_writer()
    if writer is not None:
        return writer.read_bytes(_writer_relative_path(writer, path), max_bytes=max_bytes)
    target = _logical_path(path)
    if max_bytes is not None and target.stat().st_size > max_bytes:
        raise ValueError("Secure output file exceeds the requested read limit.")
    return target.read_bytes()


def sha256_output(path: str | Path) -> str:
    """Hash an output through the descriptor writer when one is bound."""

    writer = current_output_writer()
    if writer is not None:
        return writer.sha256(_writer_relative_path(writer, path))
    digest = hashlib.sha256()
    with _logical_path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_output_file(source: str | Path, destination: str | Path) -> Path:
    """Publish one producer file without replacing an active-writer output."""

    writer = current_output_writer()
    if writer is not None:
        return writer.publish_file(_writer_relative_path(writer, destination), _logical_path(source))

    source_path = _logical_path(source)
    destination_path = _logical_path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with source_path.open("rb") as source_stream, destination_path.open("wb") as destination_stream:
            shutil.copyfileobj(source_stream, destination_stream)
            destination_stream.flush()
            os.fsync(destination_stream.fileno())
    except FileExistsError as exc:
        raise SecureWriteConflict(f"Output already exists: {destination_path}") from exc
    return destination_path


@contextlib.contextmanager
def staged_output_file(logical_path: str | Path) -> Iterator[Path]:
    """Yield a private per-file path and publish its unchanged bytes on success.

    The producer receives the logical path separately when it builds manifests
    or content-bound metadata.  Only the file bytes cross the private staging
    boundary; this helper never rewrites paths or hashes inside those bytes.
    """

    writer = current_output_writer()
    if writer is None:
        yield _logical_path(logical_path)
        return

    relative = _writer_relative_path(writer, logical_path)
    create_staging = getattr(writer, "create_staging", None)
    if not callable(create_staging):
        raise SecureWriteUnavailable("The bound writer does not support private per-file staging.")
    staging = create_staging(prefix=".oel-output-file-")
    physical = staging.path / relative.name
    try:
        yield physical
        # Seal the source identity for cleanup before the final writer copies
        # it.  If publication fails or the producer replaces the source after
        # copying, cleanup retains the mismatched stage name.
        cleanup_identities = _snapshot_identities(
            _validate_staging_tree(staging.fd, staging.path)
        )
        staging._cleanup_identities = cleanup_identities
        writer.publish_file(relative, physical)
    finally:
        if not staging._closed:
            staging.remove_tree()


def _select_approved_root(path: Path, approved_roots: Sequence[str | Path]) -> Path:
    roots = tuple(_absolute_lexical(root) for root in approved_roots)
    if not roots:
        raise SecureWriteError("Secure output requires at least one explicit approved write root.")
    matches = [root for root in roots if _is_relative_to(path, root)]
    if not matches:
        raise SecureWriteError("Output path is outside the explicit approved write roots.")
    return max(matches, key=lambda root: len(root.parts))


def _directory_flags() -> int:
    return os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW


def _open_directory_chain(path: Path, *, create: bool) -> int:
    """Open ``path`` one component at a time without following symlinks."""

    require_secure_write_primitives()
    path = _absolute_lexical(path)
    current_fd = os.open(os.path.sep, _directory_flags())
    try:
        for part in path.parts[1:]:
            try:
                next_fd = os.open(part, _directory_flags(), dir_fd=current_fd)
            except FileNotFoundError:
                if not create:
                    raise
                try:
                    os.mkdir(part, mode=0o700, dir_fd=current_fd)
                except FileExistsError:
                    pass
                else:
                    _fsync_directory(current_fd)
                next_fd = os.open(part, _directory_flags(), dir_fd=current_fd)
            os.close(current_fd)
            current_fd = next_fd
        return current_fd
    except BaseException:
        os.close(current_fd)
        raise


def _open_relative_directory(base_fd: int, parts: Sequence[str], *, create: bool) -> int:
    current_fd = os.dup(base_fd)
    try:
        for part in parts:
            try:
                next_fd = os.open(part, _directory_flags(), dir_fd=current_fd)
            except FileNotFoundError:
                if not create:
                    raise
                try:
                    os.mkdir(part, mode=0o700, dir_fd=current_fd)
                except FileExistsError:
                    pass
                else:
                    _fsync_directory(current_fd)
                next_fd = os.open(part, _directory_flags(), dir_fd=current_fd)
            os.close(current_fd)
            current_fd = next_fd
        return current_fd
    except BaseException:
        os.close(current_fd)
        raise


def _close_quietly(fd: int | None) -> None:
    if fd is None:
        return
    try:
        os.close(fd)
    except OSError:
        pass


def _fsync_directory(fd: int) -> None:
    """Make a newly created directory entry durable or fail closed."""

    os.fsync(fd)


def _new_file_identity(fd: int) -> tuple[int, int]:
    """Validate and return the identity of one newly created output inode."""

    file_stat = os.fstat(fd)
    if not stat.S_ISREG(file_stat.st_mode):
        raise SecureWriteError("Secure output creation requires a regular file.")
    if file_stat.st_nlink != 1:
        raise SecureWriteError("Secure output creation rejects multiply linked files.")
    return file_stat.st_dev, file_stat.st_ino


def _directory_identity(fd: int) -> tuple[int, int]:
    """Return the device/inode identity of an open directory descriptor."""

    directory_stat = os.fstat(fd)
    if not stat.S_ISDIR(directory_stat.st_mode):
        raise SecureWriteError("Secure output cleanup requires a directory descriptor.")
    return directory_stat.st_dev, directory_stat.st_ino


def _entry_snapshot(metadata: os.stat_result) -> tuple[str, int, int, int, int, int, int]:
    """Capture the identity and mutation-sensitive metadata of one entry.

    Access time is deliberately excluded because opening and reading a source
    can update it without changing the bytes being published.  Size and
    change/modify times cover ordinary producer mutations; the descriptor
    identity covers replacement by another inode.
    """

    if stat.S_ISDIR(metadata.st_mode):
        kind = "directory"
    elif stat.S_ISREG(metadata.st_mode):
        kind = "file"
    elif stat.S_ISLNK(metadata.st_mode):
        kind = "symlink"
    else:
        kind = "special"
    return (
        kind,
        int(metadata.st_dev),
        int(metadata.st_ino),
        int(metadata.st_size),
        int(metadata.st_mtime_ns),
        int(metadata.st_ctime_ns),
        int(metadata.st_nlink),
    )


def _entry_matches_snapshot(
    expected: tuple[str, int, int, int, int, int, int],
    metadata: os.stat_result,
) -> bool:
    return expected == _entry_snapshot(metadata)


class _RootOrFileOpen:
    """Dispatch ``SecureWriteRoot.open`` by access context.

    The class-level form opens a descriptor-bound output root.  The instance
    form opens one exclusive file below an already-open root.
    """

    def __get__(self, instance: Any, owner: type[SecureWriteRoot]) -> Any:
        if instance is None:
            return owner._open_root
        return instance._open_file


@dataclass
class SecureWriteRoot:
    """A validated output directory held open by descriptor for its lifetime."""

    path: Path
    _fd: int
    approved_root: Path
    _logical_root: Path | None = None
    active_opid: str | None = None
    _owned_files: set[tuple[str, ...]] = field(default_factory=set)
    _owned_inodes: dict[tuple[str, ...], tuple[int, int]] = field(default_factory=dict)
    _owned_file_fds: dict[tuple[str, ...], int] = field(default_factory=dict, repr=False)
    _owned_directories: dict[tuple[str, ...], tuple[int, int]] = field(default_factory=dict)
    _closed: bool = False
    _private_parent: Path | None = None
    _root_identity: tuple[int, int] | None = None
    _private_parent_identity: tuple[int, int] | None = None
    _cleanup_identities: dict[tuple[str, ...], tuple[int, int]] | None = None

    @classmethod
    def _open_root(
        cls,
        path: str | Path,
        *,
        approved_roots: Sequence[str | Path],
        create: bool = True,
        require_empty: bool = False,
        logical_root: str | Path | None = None,
        active_opid: str | None = None,
    ) -> SecureWriteRoot:
        target = _absolute_lexical(path)
        approved_root = _select_approved_root(target, approved_roots)
        root_fd = _open_directory_chain(approved_root, create=False)
        try:
            relative = target.relative_to(approved_root)
            target_fd = _open_relative_directory(root_fd, relative.parts, create=create)
        finally:
            os.close(root_fd)
        try:
            root_identity = _directory_identity(target_fd)
        except BaseException:
            _close_quietly(target_fd)
            raise
        logical = target if logical_root is None else _absolute_lexical(logical_root)
        writer = cls(
            target,
            target_fd,
            approved_root,
            _logical_root=logical,
            active_opid=str(active_opid) if active_opid else None,
            _root_identity=root_identity,
        )
        writer._owned_directories[()] = root_identity
        try:
            if require_empty and os.listdir(target_fd):
                raise FileExistsError("The approved output directory must be new or empty.")
        except BaseException:
            writer.close()
            raise
        return writer

    @property
    def fd(self) -> int:
        if self._closed:
            raise ValueError("The secure output handle is closed.")
        return self._fd

    @property
    def root(self) -> Path:
        return self.path

    @property
    def logical_root(self) -> Path:
        return self.path if self._logical_root is None else self._logical_root

    def close(self) -> None:
        if not self._closed:
            for file_fd in self._owned_file_fds.values():
                _close_quietly(file_fd)
            self._owned_file_fds.clear()
            _close_quietly(self._fd)
            self._closed = True
            for staging_path, staging in tuple(_STAGING_HANDLES.items()):
                if staging is self:
                    _STAGING_HANDLES.pop(staging_path, None)

    def __enter__(self) -> SecureWriteRoot:
        if self._closed:
            raise ValueError("The secure output handle is closed.")
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _tb: Any) -> None:
        self.close()

    def child(self, relative: str | Path, *, create: bool = True) -> SecureWriteRoot:
        parts = _relative_parts(relative)
        child_fd = _open_relative_directory(self.fd, parts, create=create)
        try:
            child_identity = _directory_identity(child_fd)
            return SecureWriteRoot(
                self.path.joinpath(*parts),
                child_fd,
                self.approved_root,
                _logical_root=self.logical_root.joinpath(*parts),
                active_opid=self.active_opid,
                _root_identity=child_identity,
                _owned_directories={(): child_identity},
            )
        except BaseException:
            _close_quietly(child_fd)
            raise

    def _remember_directories(self, parts: Sequence[str]) -> None:
        """Remember identities for directories opened by this writer."""

        if not parts:
            return
        directory_fd: int | None = None
        try:
            directory_fd = os.dup(self.fd)
            for index, part in enumerate(parts):
                next_fd = os.open(part, _directory_flags(), dir_fd=directory_fd)
                _close_quietly(directory_fd)
                directory_fd = next_fd
                self._owned_directories[tuple(parts[: index + 1])] = _directory_identity(
                    directory_fd
                )
        except OSError:
            # The actual operation already holds its own descriptor.  If a
            # name moved before this advisory bookkeeping pass, leave that
            # path unowned so cleanup retains it rather than guessing.
            return
        finally:
            _close_quietly(directory_fd)

    def ensure_empty(self) -> None:
        if os.listdir(self.fd):
            raise FileExistsError("The approved output directory must be new or empty.")

    def write_bytes(
        self,
        relative: str | Path,
        payload: bytes,
        *,
        mode: int = 0o600,
        parents: bool = True,
    ) -> Path:
        """Create a new regular file exclusively below this descriptor."""

        parts = _relative_parts(relative)
        parent_fd: int | None = None
        file_fd: int | None = None
        try:
            parent_fd = _open_relative_directory(self.fd, parts[:-1], create=parents)
            self._remember_directories(parts[:-1])
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
            file_fd = os.open(parts[-1], flags, mode, dir_fd=parent_fd)
            identity = _new_file_identity(file_fd)
            _write_all(file_fd, payload)
            os.fsync(file_fd)
            _fsync_directory(parent_fd)
            self._remember_owned_file(parts, file_fd, identity)
            return self.path.joinpath(*parts)
        except FileExistsError as exc:
            raise SecureWriteConflict(f"Output already exists: {self.path.joinpath(*parts)}") from exc
        except BaseException:
            # Retain the failed entry.  Cleanup by mutable pathname could
            # delete a concurrent replacement after this descriptor was opened.
            raise
        finally:
            _close_quietly(file_fd)
            _close_quietly(parent_fd)

    def replace_owned_bytes(self, relative: str | Path, payload: bytes) -> Path:
        """Update a file created by this handle, without path replacement."""

        parts = _relative_parts(relative)
        parent_fd: int | None = None
        file_fd: int | None = None
        try:
            parent_fd = _open_relative_directory(self.fd, parts[:-1], create=False)
            file_fd = self._open_owned_for_update(parts, parent_fd)
            os.ftruncate(file_fd, 0)
            _write_all(file_fd, payload)
            os.fsync(file_fd)
            return self.path.joinpath(*parts)
        finally:
            _close_quietly(file_fd)
            _close_quietly(parent_fd)

    def update_existing_bytes(self, relative: str | Path, payload: bytes) -> Path:
        """Refresh an existing regular file without replacing its directory entry.

        This is reserved for operation manifests that are created and then
        finalized by the same MCP call.  New producer artifacts must use
        :meth:`write_bytes`, which refuses an existing entry.
        """

        parts = _relative_parts(relative)
        parent_fd: int | None = None
        file_fd: int | None = None
        try:
            parent_fd = _open_relative_directory(self.fd, parts[:-1], create=False)
            file_fd = self._open_owned_for_update(parts, parent_fd)
            os.ftruncate(file_fd, 0)
            _write_all(file_fd, payload)
            os.fsync(file_fd)
            return self.path.joinpath(*parts)
        finally:
            _close_quietly(file_fd)
            _close_quietly(parent_fd)

    def adopt_existing(self, relative: str | Path) -> Path:
        """Bind one existing regular file for identity-checked updates."""

        parts = _relative_parts(relative)
        parent_fd: int | None = None
        file_fd: int | None = None
        try:
            parent_fd = _open_relative_directory(self.fd, parts[:-1], create=False)
            file_fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW, dir_fd=parent_fd)
            identity = _new_file_identity(file_fd)
            self._remember_owned_file(parts, file_fd, identity)
            return self.path.joinpath(*parts)
        finally:
            _close_quietly(file_fd)
            _close_quietly(parent_fd)

    def _remember_owned_file(self, parts: tuple[str, ...], file_fd: int, identity: tuple[int, int]) -> None:
        # Keep the original inode alive. Linux may otherwise immediately reuse
        # its number after unlink(), making a replacement pass a dev/ino check.
        retained = os.dup(file_fd)
        previous = self._owned_file_fds.get(parts)
        self._owned_file_fds[parts] = retained
        self._owned_files.add(parts)
        self._owned_inodes[parts] = identity
        _close_quietly(previous)

    def _forget_owned_file(self, parts: tuple[str, ...]) -> None:
        self._owned_files.discard(parts)
        self._owned_inodes.pop(parts, None)
        _close_quietly(self._owned_file_fds.pop(parts, None))

    def _open_owned_for_update(self, parts: tuple[str, ...], parent_fd: int) -> int:
        expected = self._owned_inodes.get(parts)
        if parts not in self._owned_files or expected is None:
            raise SecureWriteError("Existing output updates require a file created by this secure handle.")
        file_fd: int | None = None
        try:
            # Open without O_TRUNC.  The descriptor identity is checked before
            # ftruncate so a renamed or replaced path can never truncate an
            # inode that this operation did not create.
            file_fd = os.open(parts[-1], os.O_WRONLY | os.O_NOFOLLOW, dir_fd=parent_fd)
            actual = os.fstat(file_fd)
            identity = (actual.st_dev, actual.st_ino)
            if not stat.S_ISREG(actual.st_mode) or actual.st_nlink != 1 or identity != expected:
                raise SecureWriteError("Existing output inode is no longer owned by this secure handle.")
            return file_fd
        except BaseException:
            _close_quietly(file_fd)
            raise

    def write_text(self, relative: str | Path, text: str, *, encoding: str = "utf-8") -> Path:
        return self.write_bytes(relative, text.encode(encoding), parents=True)

    @contextlib.contextmanager
    def _open_file(
        self,
        relative: str | Path,
        mode: str = "xb",
        *,
        permissions: int = 0o600,
    ) -> Iterator[BinaryIO]:
        """Open one new binary file relative to this root.

        ``xb`` and ``wb`` are accepted as producer-facing spellings, and both
        are exclusive here.  Existing entries must be updated through the
        explicit manifest-only update method.
        """

        if mode not in {"xb", "wb"}:
            raise ValueError("Secure output open accepts only exclusive binary xb or wb mode.")
        parts = _relative_parts(relative)
        parent_fd: int | None = None
        file_fd: int | None = None
        try:
            parent_fd = _open_relative_directory(self.fd, parts[:-1], create=True)
            self._remember_directories(parts[:-1])
            file_fd = os.open(
                parts[-1],
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                permissions,
                dir_fd=parent_fd,
            )
            identity = _new_file_identity(file_fd)
            _fsync_directory(parent_fd)
            self._remember_owned_file(parts, file_fd, identity)
            with os.fdopen(file_fd, "wb") as stream:
                file_fd = None
                try:
                    yield stream
                    stream.flush()
                    os.fsync(stream.fileno())
                except BaseException:
                    try:
                        stream.close()
                    finally:
                        self._forget_owned_file(parts)
                    raise
        except FileExistsError as exc:
            raise SecureWriteConflict(f"Output already exists: {self.path.joinpath(*parts)}") from exc
        except BaseException:
            self._forget_owned_file(parts)
            # Retain the failed entry for the same reason as write_bytes().
            raise
        finally:
            _close_quietly(file_fd)
            _close_quietly(parent_fd)

    open = _RootOrFileOpen()

    def mkdir(self, relative: str | Path) -> Path:
        """Create a descriptor-bound directory below this root."""

        parts = _relative_parts(relative)
        directory_fd = _open_relative_directory(self.fd, parts, create=True)
        self._remember_directories(parts)
        _close_quietly(directory_fd)
        return self.path.joinpath(*parts)

    def read_bytes(self, relative: str | Path, *, max_bytes: int | None = None) -> bytes:
        """Read one regular file below this root without following a symlink."""

        parts = _relative_parts(relative)
        parent_fd: int | None = None
        file_fd: int | None = None
        try:
            parent_fd = _open_relative_directory(self.fd, parts[:-1], create=False)
            if not stat.S_ISREG(os.stat(
                parts[-1], dir_fd=parent_fd, follow_symlinks=False
            ).st_mode):
                raise SecureWriteError("Secure output reads require a regular file.")
            file_fd = os.open(
                parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent_fd
            )
            existing = os.fstat(file_fd)
            if not stat.S_ISREG(existing.st_mode):
                raise SecureWriteError("Secure output reads require a regular file.")
            if max_bytes is not None and existing.st_size > max_bytes:
                raise ValueError("Secure output file exceeds the requested read limit.")
            chunks: list[bytes] = []
            remaining = None if max_bytes is None else max_bytes
            while True:
                size = 1024 * 1024 if remaining is None else min(1024 * 1024, remaining)
                if size <= 0:
                    break
                chunk = os.read(file_fd, size)
                if not chunk:
                    break
                chunks.append(chunk)
                if remaining is not None:
                    remaining -= len(chunk)
            return b"".join(chunks)
        finally:
            _close_quietly(file_fd)
            _close_quietly(parent_fd)

    def sha256(self, relative: str | Path) -> str:
        return str(self.file_record(relative)["sha256"])

    def file_record(self, relative: str | Path) -> dict[str, Any]:
        """Hash and size one regular file through the retained root descriptor."""

        parts = _relative_parts(relative)
        parent_fd = _open_relative_directory(self.fd, parts[:-1], create=False)
        file_fd: int | None = None
        try:
            file_fd = os.open(
                parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent_fd
            )
            before = os.fstat(file_fd)
            if not stat.S_ISREG(before.st_mode):
                raise SecureWriteError("Secure output reads require a regular file.")
            digest = hashlib.sha256()
            size = 0
            while chunk := os.read(file_fd, 1024 * 1024):
                digest.update(chunk)
                size += len(chunk)
            after = os.fstat(file_fd)
            if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
                after.st_size, after.st_mtime_ns, after.st_ctime_ns
            ) or size != after.st_size:
                raise SecureWriteError("Output changed while its evidence record was being hashed.")
            return {"bytes": size, "sha256": digest.hexdigest()}
        finally:
            _close_quietly(file_fd)
            _close_quietly(parent_fd)

    def list_files(self) -> tuple[Path, ...]:
        """Inventory regular files below the retained descriptor without following links."""

        files: list[Path] = []

        def visit(directory_fd: int, prefix: Path) -> None:
            for name in sorted(os.listdir(directory_fd)):
                metadata = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
                relative = prefix / name
                if stat.S_ISDIR(metadata.st_mode):
                    child_fd = os.open(name, _directory_flags(), dir_fd=directory_fd)
                    try:
                        visit(child_fd, relative)
                    finally:
                        os.close(child_fd)
                elif stat.S_ISREG(metadata.st_mode):
                    files.append(relative)
                else:
                    raise SecureWriteError("Secure output inventories require regular files and directories.")

        visit(self.fd, Path())
        return tuple(files)

    def publish_file(self, relative: str | Path, source: str | Path) -> Path:
        """Copy one regular file from a trusted private stage exclusively.

        The source descriptor is checked against the no-follow path stat before
        copying and against its size, identity, link count, and timestamps
        after copying, including a second descriptor-bound content hash.
        This rejects ordinary producer mutations while the
        copy is in flight; it is not a filesystem lock or an atomic snapshot
        against a hostile same-account writer.
        """

        parts = _relative_parts(relative)
        # Keep the source lexical.  Resolving here would silently traverse a
        # producer-created symlink before the no-follow descriptor checks run.
        source_path = _absolute_lexical(source)
        source_parent_fd: int | None = None
        source_fd: int | None = None
        destination_parent_fd: int | None = None
        destination_fd: int | None = None
        try:
            source_parent_fd = _open_directory_chain(source_path.parent, create=False)
            checked_source_stat = os.stat(
                source_path.name, dir_fd=source_parent_fd, follow_symlinks=False
            )
            if not stat.S_ISREG(checked_source_stat.st_mode):
                raise SecureWriteError("Secure publication requires a regular source file.")
            checked_source = _entry_snapshot(checked_source_stat)
            source_fd = os.open(
                source_path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=source_parent_fd
            )
            source_stat = os.fstat(source_fd)
            if not stat.S_ISREG(source_stat.st_mode) or not _entry_matches_snapshot(
                checked_source, source_stat
            ):
                raise SecureWriteError("Secure publication source changed before it was opened.")
            destination_parent_fd = _open_relative_directory(self.fd, parts[:-1], create=True)
            self._remember_directories(parts[:-1])
            destination_fd = os.open(
                parts[-1],
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600 | (source_stat.st_mode & stat.S_IXUSR),
                dir_fd=destination_parent_fd,
            )
            copied_digest = hashlib.sha256()
            while True:
                chunk = os.read(source_fd, 1024 * 1024)
                if not chunk:
                    break
                copied_digest.update(chunk)
                _write_all(destination_fd, chunk)
            os.fsync(destination_fd)
            # Same-size rewrites can share mtime/ctime within one Linux clock
            # tick. Verify the still-open source bytes as well as its metadata.
            os.lseek(source_fd, 0, os.SEEK_SET)
            current_digest = hashlib.sha256()
            while chunk := os.read(source_fd, 1024 * 1024):
                current_digest.update(chunk)
            after_source_stat = os.fstat(source_fd)
            if (not _entry_matches_snapshot(checked_source, after_source_stat)
                    or copied_digest.digest() != current_digest.digest()):
                raise SecureWriteError("Secure publication source changed during publication.")
            identity = _new_file_identity(destination_fd)
            _fsync_directory(destination_parent_fd)
            self._remember_owned_file(parts, destination_fd, identity)
            staging_binding = _staging_source_binding(source_path)
            if staging_binding is not None:
                staging, source_relative = staging_binding
                staging._remember_directories(source_relative[:-1])
                if staging._cleanup_identities is None:
                    staging._cleanup_identities = {}
                staging._cleanup_identities[source_relative] = (
                    checked_source[1],
                    checked_source[2],
                )
            return self.path.joinpath(*parts)
        except FileExistsError as exc:
            raise SecureWriteConflict(f"Output already exists: {self.path.joinpath(*parts)}") from exc
        except BaseException:
            # Retain partial publication; path-based rollback could remove a
            # concurrent replacement at this name.
            raise
        finally:
            _close_quietly(source_fd)
            _close_quietly(destination_fd)
            _close_quietly(source_parent_fd)
            _close_quietly(destination_parent_fd)

    def create_staging(
        self,
        *,
        prefix: str = ".oel-staging-",
        parent: str | Path | None = None,
    ) -> SecureWriteRoot:
        if parent is not None:
            raise SecureWriteError(
                "Staging is always created below a server-private temporary "
                "directory; caller-selected staging parents are unsupported."
            )
        if not prefix or any(part in {"", ".", ".."} for part in Path(prefix).parts):
            raise ValueError("Invalid secure staging prefix.")
        # macOS commonly exposes the process temp directory through ``/var``
        # -> ``/private/var``.  Resolve that trusted system location before
        # opening it with no-follow flags; the created child itself remains
        # private and descriptor-validated.
        private_root = Path(tempfile.gettempdir()).resolve(strict=True)
        private_parent = Path(tempfile.mkdtemp(prefix=prefix, dir=private_root))
        private_parent_stat = os.stat(private_parent, follow_symlinks=False)
        if not stat.S_ISDIR(private_parent_stat.st_mode):
            _remove_private_parent(private_parent)
            raise SecureWriteError("Secure staging parent must be a directory.")
        private_parent_identity = (private_parent_stat.st_dev, private_parent_stat.st_ino)
        try:
            staging = SecureWriteRoot.open(
                private_parent / "payload",
                approved_roots=(private_parent,),
                create=True,
                require_empty=True,
                logical_root=self.logical_root,
                active_opid=self.active_opid,
            )
            staging._private_parent = private_parent
            staging._private_parent_identity = private_parent_identity
            _STAGING_HANDLES[staging.path] = staging
            return staging
        except BaseException:
            _remove_private_parent(private_parent, private_parent_identity)
            raise

    def publish_staging(self, staging: SecureWriteRoot) -> tuple[Path, ...]:
        """Copy a staged tree into this root using exclusive descriptor writes.

        The complete private tree is checked before publication.  If a later
        collision or write failure interrupts publication, entries already
        visible in this root are deliberately retained; pathname rollback
        could remove an unrelated replacement that appeared after creation.
        Source identities, metadata, and directory entry sets are checked
        again during and after publication.  These checks reject ordinary
        producer races but do not serialize a hostile same-account writer
        after the final check.
        """

        self._require_staging(staging)
        # Validate the whole private tree before creating any visible entry.
        # Publication may still fail later (for example on a collision), but
        # failed entries are retained instead of being removed by pathname.
        staging_snapshot = _validate_staging_tree(staging.fd, staging.path)
        staging._cleanup_identities = _snapshot_identities(staging_snapshot)
        published: list[Path] = []
        _publish_directory(
            staging.fd,
            self.fd,
            self.path,
            published,
            staging_snapshot,
        )
        # Re-snapshot before removing the private tree.  This catches entries
        # added or replaced after the initial validation and retains any
        # already-published destination entries on failure.
        if _validate_staging_tree(staging.fd, staging.path) != staging_snapshot:
            raise SecureWriteError("Secure staging changed during publication.")
        staging.remove_tree()
        return tuple(published)

    def remove_tree(self) -> None:
        """Remove this handle's tree while retaining entries with changed identity.

        The descriptor keeps cleanup of the original tree detached from later
        pathname changes.  The final name check is intentionally best effort:
        it detects an already-replaced name, but is not an atomic
        unlink-by-inode primitive against a hostile same-account rename
        between the check and the unlink.  Only identities recorded by this
        handle or by a publication seal are eligible for recursive cleanup;
        unknown or late names remain in the private tree.
        """

        parent = self.path.parent
        name = self.path.name
        parent_fd: int | None = None
        cleanup_identities = dict(self._owned_directories)
        cleanup_identities.update(self._owned_inodes)
        if self._cleanup_identities is not None:
            cleanup_identities.update(self._cleanup_identities)
        try:
            parent_fd = _open_directory_chain(parent, create=False)
            _remove_directory_fd(
                self.fd,
                expected_identities=cleanup_identities,
            )
            _remove_owned_entry(parent_fd, name, self._root_identity, directory=True)
        finally:
            _close_quietly(parent_fd)
            self.close()
        if self._private_parent is not None:
            _remove_private_parent(self._private_parent, self._private_parent_identity)

    def _require_staging(self, child: SecureWriteRoot) -> None:
        if child.path == self.path or child._private_parent is None:
            raise SecureWriteError("Staging must be an isolated private directory.")
        if child._closed:
            raise SecureWriteError("Staging handle is closed.")


@dataclass
class SecureOutputTransaction:
    """Final output handle plus an isolated path-facing producer directory."""

    final: SecureWriteRoot
    staging: SecureWriteRoot
    _published: bool = False

    @property
    def path(self) -> Path:
        return self.final.path

    @property
    def staging_path(self) -> Path:
        return self.staging.path

    @property
    def active_opid(self) -> str | None:
        """Operation identity carried by both final and staging handles."""

        return self.final.active_opid

    def publish(self) -> tuple[Path, ...]:
        if self._published:
            return ()
        created = self.final.publish_staging(self.staging)
        self._published = True
        return created

    def discard(self) -> None:
        if not self.staging._closed:
            self.staging.remove_tree()

    def rewrite(self, value: Any) -> Any:
        return rewrite_staged_paths(value, self.staging_path, self.path)

    def close(self) -> None:
        self.staging.close()
        self.final.close()


@contextlib.contextmanager
def staged_output(
    path: str | Path,
    *,
    approved_roots: Sequence[str | Path],
    require_empty: bool = True,
    staging_parent: str | Path | None = None,
    active_opid: str | None = None,
) -> Iterator[SecureOutputTransaction]:
    """Run a path-facing producer in private staging and publish on success."""

    if staging_parent is not None:
        raise SecureWriteError(
            "Staging is always server-private; caller-selected staging "
            "parents are unsupported."
        )

    final = SecureWriteRoot.open(
        path,
        approved_roots=approved_roots,
        create=True,
        require_empty=require_empty,
        active_opid=active_opid,
    )
    try:
        staging = final.create_staging()
    except BaseException:
        final.close()
        raise
    transaction = SecureOutputTransaction(final=final, staging=staging)
    try:
        yield transaction
        transaction.publish()
    except BaseException:
        try:
            transaction.discard()
        except BaseException:
            pass
        raise
    finally:
        transaction.close()


def rewrite_staged_paths(value: Any, staging: Path, final: Path) -> Any:
    """Rewrite producer result paths after staged publication."""

    staging_text = str(staging)
    final_text = str(final)
    if isinstance(value, Path):
        try:
            return final / value.relative_to(staging)
        except ValueError:
            return value
    if isinstance(value, str):
        if value == staging_text:
            return final_text
        prefix = staging_text + os.sep
        if value.startswith(prefix):
            return final_text + value[len(staging_text) :]
        return value
    if isinstance(value, dict):
        return {key: rewrite_staged_paths(item, staging, final) for key, item in value.items()}
    if isinstance(value, list):
        return [rewrite_staged_paths(item, staging, final) for item in value]
    if isinstance(value, tuple):
        return tuple(rewrite_staged_paths(item, staging, final) for item in value)
    return value


def _write_all(fd: int, payload: bytes) -> None:
    view = memoryview(bytes(payload))
    while view:
        written = os.write(fd, view)
        if written <= 0:
            raise OSError("Secure output write made no progress.")
        view = view[written:]


def _validate_staging_tree(
    source_fd: int,
    source_path: Path,
) -> dict[Path, tuple[str, int, int, int, int, int, int]]:
    """Reject unsafe entries and snapshot the private tree for publication."""

    snapshot: dict[Path, tuple[str, int, int, int, int, int, int]] = {}

    def visit(directory_fd: int, prefix: Path, path: Path) -> None:
        for name in sorted(os.listdir(directory_fd)):
            if name in {".", ".."}:
                raise SecureWriteError("Invalid directory entry in secure staging.")
            source_stat = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            relative = prefix / name
            target = path / name
            entry = _entry_snapshot(source_stat)
            if stat.S_ISLNK(source_stat.st_mode):
                raise SecureWriteError(f"Secure staging contains a symlink: {target}")
            if stat.S_ISDIR(source_stat.st_mode):
                snapshot[relative] = entry
                child_fd: int | None = None
                try:
                    child_fd = os.open(name, _directory_flags(), dir_fd=directory_fd)
                    opened_stat = os.fstat(child_fd)
                    if not _entry_matches_snapshot(entry, opened_stat):
                        raise SecureWriteError(
                            f"Secure staging entry changed during validation: {target}"
                        )
                    visit(child_fd, relative, target)
                    after_stat = os.fstat(child_fd)
                    if not _entry_matches_snapshot(entry, after_stat):
                        raise SecureWriteError(
                            f"Secure staging entry changed during validation: {target}"
                        )
                finally:
                    _close_quietly(child_fd)
            elif stat.S_ISREG(source_stat.st_mode):
                snapshot[relative] = entry
            else:
                raise SecureWriteError(f"Secure staging contains a non-regular file: {target}")

    visit(source_fd, Path(), source_path)
    return snapshot


def _snapshot_identities(
    snapshot: Mapping[Path, tuple[str, int, int, int, int, int, int]],
) -> dict[tuple[str, ...], tuple[int, int]]:
    """Convert a relative tree snapshot into cleanup identity guards."""

    return {
        tuple(relative.parts): (entry[1], entry[2])
        for relative, entry in snapshot.items()
    }


def _staging_source_binding(
    source_path: Path,
) -> tuple[SecureWriteRoot, tuple[str, ...]] | None:
    """Find a live private stage that owns a lexical source path."""

    for staging_path, staging in tuple(_STAGING_HANDLES.items()):
        if staging._closed:
            _STAGING_HANDLES.pop(staging_path, None)
            continue
        try:
            relative = source_path.relative_to(staging_path)
        except ValueError:
            continue
        if relative.parts:
            return staging, tuple(relative.parts)
    return None


def _publish_directory(
    source_fd: int,
    destination_fd: int,
    destination_path: Path,
    published: list[Path],
    expected: dict[Path, tuple[str, int, int, int, int, int, int]],
    prefix: Path = Path(),
) -> None:
    names = sorted(os.listdir(source_fd))
    expected_names = {
        relative.name for relative in expected if relative.parent == prefix
    }
    if set(names) != expected_names:
        raise SecureWriteError(
            f"Secure staging directory entries changed during publication: {destination_path}"
        )
    for name in names:
        if name in {".", ".."}:
            raise SecureWriteError("Invalid directory entry in secure staging.")
        source_stat = os.stat(name, dir_fd=source_fd, follow_symlinks=False)
        relative = prefix / name
        expected_entry = expected.get(relative)
        if expected_entry is None or not _entry_matches_snapshot(expected_entry, source_stat):
            raise SecureWriteError(
                f"Secure staging entry changed during publication: {destination_path / name}"
            )
        target = destination_path / name
        if stat.S_ISLNK(source_stat.st_mode):
            raise SecureWriteError(f"Secure staging contains a symlink: {target}")
        if stat.S_ISDIR(source_stat.st_mode):
            source_child: int | None = None
            destination_child: int | None = None
            try:
                source_child = os.open(name, _directory_flags(), dir_fd=source_fd)
                opened_stat = os.fstat(source_child)
                if not _entry_matches_snapshot(expected_entry, opened_stat):
                    raise SecureWriteError(
                        f"Secure staging entry changed during publication: {target}"
                    )
                try:
                    os.mkdir(name, mode=0o700, dir_fd=destination_fd)
                except FileExistsError as exc:
                    raise SecureWriteConflict(f"Output already exists: {target}") from exc
                _fsync_directory(destination_fd)
                published.append(target)
                destination_child = os.open(name, _directory_flags(), dir_fd=destination_fd)
                _publish_directory(
                    source_child,
                    destination_child,
                    target,
                    published,
                    expected,
                    relative,
                )
                after_stat = os.fstat(source_child)
                if not _entry_matches_snapshot(expected_entry, after_stat):
                    raise SecureWriteError(
                        f"Secure staging entry changed during publication: {target}"
                    )
            finally:
                _close_quietly(source_child)
                _close_quietly(destination_child)
            continue
        if not stat.S_ISREG(source_stat.st_mode):
            raise SecureWriteError(f"Secure staging contains a non-regular file: {target}")
        source_file: int | None = None
        destination_file: int | None = None
        try:
            source_file = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=source_fd)
            opened_stat = os.fstat(source_file)
            if not stat.S_ISREG(opened_stat.st_mode) or not _entry_matches_snapshot(
                expected_entry, opened_stat
            ):
                raise SecureWriteError(f"Secure staging entry changed during publication: {target}")
            destination_file = os.open(
                name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600 | (opened_stat.st_mode & stat.S_IXUSR),
                dir_fd=destination_fd,
            )
            while True:
                chunk = os.read(source_file, 1024 * 1024)
                if not chunk:
                    break
                _write_all(destination_file, chunk)
            os.fsync(destination_file)
            after_stat = os.fstat(source_file)
            if not _entry_matches_snapshot(expected_entry, after_stat):
                raise SecureWriteError(f"Secure staging entry changed during publication: {target}")
            _fsync_directory(destination_fd)
            published.append(target)
        except FileExistsError as exc:
            raise SecureWriteConflict(f"Output already exists: {target}") from exc
        finally:
            _close_quietly(source_file)
            _close_quietly(destination_file)

    if set(os.listdir(source_fd)) != expected_names:
        raise SecureWriteError(
            f"Secure staging directory entries changed during publication: {destination_path}"
        )


def _remove_owned_entry(
    parent_fd: int,
    name: str,
    expected_identity: tuple[int, int] | None,
    *,
    directory: bool,
) -> bool:
    """Remove a name only when its current no-follow identity still matches.

    This is a narrow ownership check.  POSIX has no portable atomic
    unlink-by-inode operation for a name, so a hostile same-account actor can
    still win a rename race after the check.  On any mismatch or removal
    uncertainty the name is retained.
    """

    if expected_identity is None:
        return False
    try:
        current = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except OSError:
        return False
    if (current.st_dev, current.st_ino) != expected_identity:
        return False
    try:
        if directory:
            os.rmdir(name, dir_fd=parent_fd)
        else:
            os.unlink(name, dir_fd=parent_fd)
    except OSError:
        # A producer may have added content or another actor may have changed
        # the name after the check.  Do not retry by pathname.
        return False
    return True


def _remove_directory_fd(
    directory_fd: int,
    *,
    expected_identities: Mapping[tuple[str, ...], tuple[int, int]] | None = None,
    prefix: tuple[str, ...] = (),
) -> None:
    for name in os.listdir(directory_fd):
        try:
            source_stat = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        except OSError:
            continue
        relative = prefix + (name,)
        if expected_identities is None:
            expected_identity = (source_stat.st_dev, source_stat.st_ino)
        else:
            expected_identity = expected_identities.get(relative)
            if expected_identity is None:
                # The cleanup snapshot did not include this name.  It may be
                # a late producer entry or a replacement, so retain it.
                continue
        if stat.S_ISDIR(source_stat.st_mode) and not stat.S_ISLNK(source_stat.st_mode):
            child_fd: int | None = None
            try:
                child_fd = os.open(name, _directory_flags(), dir_fd=directory_fd)
                if _directory_identity(child_fd) != expected_identity:
                    # The name changed between the no-follow stat and open;
                    # do not recurse through the replacement descriptor.
                    continue
                _remove_directory_fd(
                    child_fd,
                    expected_identities=expected_identities,
                    prefix=relative,
                )
            except OSError:
                # Retain a name that could no longer be opened as the
                # originally observed directory.
                continue
            finally:
                _close_quietly(child_fd)
            _remove_owned_entry(
                directory_fd,
                name,
                expected_identity,
                directory=True,
            )
        else:
            _remove_owned_entry(
                directory_fd,
                name,
                expected_identity,
                directory=False,
            )


def _remove_private_parent(
    path: Path,
    expected_identity: tuple[int, int] | None = None,
) -> None:
    """Remove an empty private staging parent without following symlinks."""

    parent_fd: int | None = None
    try:
        parent_fd = _open_directory_chain(path.parent, create=False)
        if expected_identity is None:
            current = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
            expected_identity = (current.st_dev, current.st_ino)
        _remove_owned_entry(parent_fd, path.name, expected_identity, directory=True)
    except OSError:
        # A failed cleanup leaves only a 0700 private staging directory.  It
        # is never treated as published output and is reported by the caller's
        # normal temporary-artifact retention policy.
        pass
    finally:
        _close_quietly(parent_fd)


# Stable generic names for engine owners.  The longer ``SecureWrite*`` names
# remain available for compatibility with the MCP adapter.
SecureOutputRoot = SecureWriteRoot
SecureOutputWriter = SecureWriteRoot
SecureOutputError = SecureWriteError
SecureOutputUnavailable = SecureWriteUnavailable
SecureOutputConflict = SecureWriteConflict


__all__ = [
    "OutputWriter",
    "bind_output_writer",
    "copy_output_file",
    "current_output_writer",
    "mkdir_output",
    "open_output",
    "read_output_bytes",
    "require_output_writer",
    "sha256_output",
    "staged_output_file",
    "update_output_bytes",
    "update_output_text",
    "write_output_bytes",
    "write_output_text",
    "SecureOutputConflict",
    "SecureOutputError",
    "SecureOutputRoot",
    "SecureOutputUnavailable",
    "SecureOutputWriter",
    "SecureOutputTransaction",
    "SecureWriteConflict",
    "SecureWriteError",
    "SecureWriteRoot",
    "SecureWriteUnavailable",
    "require_secure_write_primitives",
    "rewrite_staged_paths",
    "secure_write_primitives_available",
    "staged_output",
]
