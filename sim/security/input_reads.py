"""Descriptor-bound reads and small anchored record-store primitives.

The path policy decides which directory a caller may inspect.  This module
keeps that directory open and performs all child operations relative to its
descriptor.  A caller therefore does not turn a checked ``Path`` back into a
mutable path lookup between authorization and use.

The implementation is deliberately fail-closed.  It requires POSIX
directory-descriptor operations, ``O_DIRECTORY`` and ``O_NOFOLLOW``.  The
documented root-owned macOS ``/tmp`` and ``/var`` compatibility aliases are
rewritten before traversal; descendants are never resolved by pathname.

The store helper is intentionally small.  It provides descriptor-bound
exclusive creation, same-inode updates, and an anchored advisory lock for
control records.  A lock does not constrain a hostile same-account process
that ignores it, and there is no portable unlink-or-rename-by-inode primitive;
those limits are part of the API contract.

``SecureReadRoot.open(..., create=True)`` is the narrow initialization escape
hatch for an already-approved parent: missing directory components are created
relative to the retained parent descriptor and permissioned through their
opened descriptors. The default remains non-creating.
"""

from __future__ import annotations

import contextlib
import hashlib
import operator
import os
import secrets
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterator, Sequence

from sim.platform_compat import (
    canonicalize_platform_compatibility_root as _canonicalize_platform_root,
)

try:  # pragma: no cover - unavailable on non-POSIX hosts, which fail closed
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None  # type: ignore[assignment]


READ_CHUNK_BYTES = 1024 * 1024
DEFAULT_MAX_READ_BYTES = 64 * 1024 * 1024


class SecureReadError(PermissionError):
    """Base error for rejected or unavailable descriptor-bound reads."""


class SecureReadUnavailable(SecureReadError):
    """The current host cannot provide the required descriptor primitives."""


class SecureReadChanged(SecureReadError):
    """A regular file changed while a bounded read was in progress."""


def secure_read_primitives_available() -> bool:
    """Return whether the host exposes every primitive used by this module."""

    if os.name != "posix" or not hasattr(os, "O_DIRECTORY") or not hasattr(os, "O_NOFOLLOW"):
        return False
    dir_fd_functions = getattr(os, "supports_dir_fd", set())
    fd_functions = getattr(os, "supports_fd", set())
    return all(
        function in dir_fd_functions
        for function in (os.open, os.stat, os.unlink)
    ) and os.listdir in fd_functions


def require_secure_read_primitives() -> None:
    if not secure_read_primitives_available():
        raise SecureReadUnavailable(
            "Secure input reads require POSIX directory-descriptor, "
            "O_DIRECTORY, O_NOFOLLOW, and fd-relative listing support on this host."
        )


def _require_secure_create_primitives() -> None:
    require_secure_read_primitives()
    if (
        os.mkdir not in getattr(os, "supports_dir_fd", set())
        or not hasattr(os, "fchmod")
        or not hasattr(os, "fsync")
    ):
        raise SecureReadUnavailable(
            "Secure directory creation requires mkdirat, fchmod, and fsync support."
        )


def _absolute_lexical(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    path = _canonicalize_platform_root(path)
    if any(part in {"", ".", ".."} for part in path.parts[1:]):
        raise SecureReadError(f"Secure input paths must be normalized: {path}")
    return path


def _relative_parts(value: str | Path) -> tuple[str, ...]:
    path = Path(value)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise SecureReadError(
            "Secure input child paths must be relative and contain no traversal components."
        )
    return tuple(path.parts)


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _select_approved_root(path: Path, approved_roots: Sequence[str | Path]) -> Path:
    roots = tuple(_absolute_lexical(root) for root in approved_roots)
    if not roots:
        raise SecureReadError("Secure input reads require an explicit approved root.")
    matches = [root for root in roots if _is_relative_to(path, root)]
    if not matches:
        raise SecureReadError("Input path is outside the explicit approved read roots.")
    return max(matches, key=lambda root: len(root.parts))


def _directory_flags() -> int:
    return os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW


def _directory_mode(mode: int) -> int:
    if isinstance(mode, bool):
        raise TypeError("Secure directory modes must be integers, not booleans.")
    try:
        value = operator.index(mode)
    except TypeError as exc:
        raise TypeError("Secure directory modes must be integers.") from exc
    if value < 0 or value > 0o7777:
        raise ValueError("Secure directory modes must fit in the POSIX mode bits.")
    return value


def _open_directory_component(
    current_fd: int,
    part: str,
    *,
    create: bool,
    mode: int,
) -> int:
    """Open one child directory, optionally creating it under ``current_fd``."""

    created = False
    try:
        child_fd = os.open(part, _directory_flags(), dir_fd=current_fd)
    except FileNotFoundError:
        if not create:
            raise
        try:
            os.mkdir(part, mode, dir_fd=current_fd)
        except FileExistsError:
            # A concurrent creator may have won.  The no-follow open below
            # still decides whether the resulting entry is an acceptable
            # directory; a symlink or replacement is rejected there.
            pass
        else:
            # Persist the new directory entry before handing its descriptor to
            # the caller. Hosts without directory fsync support fail closed.
            os.fsync(current_fd)
            created = True
        child_fd = os.open(part, _directory_flags(), dir_fd=current_fd)
    if created:
        try:
            os.fchmod(child_fd, mode)
            os.fsync(child_fd)
        except BaseException:
            _close_quietly(child_fd)
            raise
    return child_fd


def _open_directory_chain(
    path: Path,
    *,
    create: bool = False,
    mode: int = 0o700,
) -> int:
    """Open an absolute directory one component at a time without links."""

    require_secure_read_primitives()
    path = _absolute_lexical(path)
    mode = _directory_mode(mode)
    current_fd = os.open(os.path.sep, _directory_flags())
    try:
        for part in path.parts[1:]:
            next_fd = _open_directory_component(
                current_fd,
                part,
                create=create,
                mode=mode,
            )
            os.close(current_fd)
            current_fd = next_fd
        return current_fd
    except BaseException:
        _close_quietly(current_fd)
        raise


def _open_relative_directory(
    base_fd: int,
    parts: Sequence[str],
    *,
    create: bool = False,
    mode: int = 0o700,
) -> int:
    mode = _directory_mode(mode)
    current_fd = os.dup(base_fd)
    try:
        for part in parts:
            next_fd = _open_directory_component(
                current_fd,
                part,
                create=create,
                mode=mode,
            )
            os.close(current_fd)
            current_fd = next_fd
        return current_fd
    except BaseException:
        _close_quietly(current_fd)
        raise


def _close_quietly(fd: int | None) -> None:
    if fd is None:
        return
    try:
        os.close(fd)
    except OSError:
        pass


def _directory_identity(fd: int) -> tuple[int, int]:
    metadata = os.fstat(fd)
    if not stat.S_ISDIR(metadata.st_mode):
        raise SecureReadError("Secure input root is not a directory.")
    return int(metadata.st_dev), int(metadata.st_ino)


def _snapshot(metadata: os.stat_result) -> tuple[str, int, int, int, int, int, int]:
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


def _same_snapshot(
    expected: tuple[str, int, int, int, int, int, int],
    metadata: os.stat_result,
) -> bool:
    return expected == _snapshot(metadata)


def _bounded_limit(value: int | None) -> int:
    if value is None:
        limit = DEFAULT_MAX_READ_BYTES
    elif isinstance(value, bool):
        raise TypeError("Secure input read limits must be integers, not booleans.")
    else:
        try:
            limit = operator.index(value)
        except TypeError as exc:
            raise TypeError("Secure input read limits must be integers.") from exc
    if limit < 0:
        raise ValueError("Secure input read limits must be non-negative.")
    return limit


def _metadata_snapshot(metadata: os.stat_result) -> tuple[str, int, int, int, int, int, int]:
    return _snapshot(metadata)


def _sqlite_entry_metadata(
    directory_fd: int,
    name: str,
) -> tuple[str, int, int, int, int, int, int] | None:
    """Stat one SQLite entry relative to an already-retained parent FD."""

    try:
        metadata = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        return None
    snapshot = _metadata_snapshot(metadata)
    if snapshot[0] != "file":
        raise SecureReadError(f"SQLite snapshot entry {name!r} is not a regular file.")
    return snapshot


def _read_sqlite_entry_bytes(
    directory_fd: int,
    name: str,
    expected: tuple[str, int, int, int, int, int, int],
    *,
    max_bytes: int,
) -> bytes:
    """Read one checked SQLite entry without reopening its visible path."""

    source_fd: int | None = None
    stream: BinaryIO | None = None
    try:
        source_fd = os.open(
            name,
            os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
            dir_fd=directory_fd,
        )
        opened = os.fstat(source_fd)
        if not _same_snapshot(expected, opened):
            raise SecureReadChanged(f"SQLite snapshot entry {name!r} changed before it was opened.")
        stream = os.fdopen(os.dup(source_fd), "rb")
        payload = bytearray()
        while True:
            chunk = stream.read(min(READ_CHUNK_BYTES, max_bytes - len(payload) + 1))
            if not chunk:
                break
            payload.extend(chunk)
            if len(payload) > max_bytes:
                raise SecureReadError(f"SQLite snapshot exceeds the {max_bytes}-byte read budget.")
        after = os.fstat(source_fd)
        if not _same_snapshot(expected, after) or len(payload) != int(after.st_size):
            raise SecureReadChanged(f"SQLite snapshot entry {name!r} changed during its read.")
        return bytes(payload)
    except FileNotFoundError as exc:
        raise SecureReadChanged(f"SQLite snapshot entry {name!r} disappeared during its read.") from exc
    except OSError as exc:
        if isinstance(exc, SecureReadError):
            raise
        raise SecureReadError(f"SQLite snapshot entry {name!r} could not be read securely.") from exc
    finally:
        if stream is not None:
            stream.close()
        _close_quietly(source_fd)


def _copy_sqlite_entry(
    directory_fd: int,
    name: str,
    expected: tuple[str, int, int, int, int, int, int],
    destination: Path,
    *,
    max_bytes: int,
) -> None:
    """Copy one descriptor-bound SQLite entry into a private snapshot tree."""

    payload = _read_sqlite_entry_bytes(
        directory_fd,
        name,
        expected,
        max_bytes=max_bytes,
    )
    try:
        with destination.open("xb") as stream:
            stream.write(payload)
    except FileExistsError as exc:
        raise SecureReadError("Private SQLite snapshot destination unexpectedly exists.") from exc
    except OSError as exc:
        raise SecureReadError("Private SQLite snapshot could not be written.") from exc


def _read_private_sqlite_snapshot(path: Path, *, max_bytes: int) -> bytes:
    """Read a completed private SQLite file with the same bounded checks."""

    try:
        before = path.stat()
        expected = _metadata_snapshot(before)
        if expected[0] != "file":
            raise SecureReadError("Private SQLite snapshot is not a regular file.")
        if expected[3] > max_bytes:
            raise SecureReadError(f"SQLite snapshot exceeds the {max_bytes}-byte read budget.")
        with path.open("rb") as stream:
            payload = stream.read(max_bytes + 1)
        after = path.stat()
        if not _same_snapshot(expected, after) or len(payload) != int(after.st_size):
            raise SecureReadChanged("Private SQLite snapshot changed during its read.")
        if len(payload) > max_bytes:
            raise SecureReadError(f"SQLite snapshot exceeds the {max_bytes}-byte read budget.")
        return payload
    except SecureReadError:
        raise
    except OSError as exc:
        raise SecureReadError("Private SQLite snapshot could not be read securely.") from exc


def _sqlite_deserialize_available() -> bool:
    import sqlite3

    return hasattr(sqlite3.Connection, "deserialize")


def sqlite_connection_from_snapshot(payload: bytes):
    """Load verified snapshot bytes into memory, including on Python 3.10.

    The compatibility reader opens only our private byte copy, never a source
    database pathname. Its directory is owner-only, and the copy is read-only
    and immutable to SQLite. As with the WAL snapshot path, this does not claim
    isolation from a hostile process running under the same account.
    """
    import sqlite3
    import tempfile

    memory = sqlite3.connect(":memory:")
    try:
        if payload:
            if _sqlite_deserialize_available():
                memory.deserialize(payload)
            else:
                with tempfile.TemporaryDirectory(prefix="oel-sqlite-memory-") as temporary:
                    snapshot = Path(temporary) / "snapshot.sqlite"
                    descriptor = os.open(snapshot, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                    with os.fdopen(descriptor, "wb") as stream:
                        stream.write(payload)
                        stream.flush()
                        os.fsync(stream.fileno())
                    source = sqlite3.connect(f"{snapshot.as_uri()}?mode=ro&immutable=1", uri=True)
                    try:
                        source.execute("PRAGMA query_only = ON")
                        source.backup(memory)
                    finally:
                        source.close()
        return memory
    except BaseException:
        memory.close()
        raise


def sqlite_snapshot_bytes(
    root: SecureReadRoot,
    relative: str | Path,
    *,
    max_bytes: int | None = None,
) -> bytes:
    """Return a bounded SQLite snapshot without reopening the visible path.

    A completed store is read directly from the descriptor-bound main file.
    If a WAL is present, bounded byte copies of the main database, WAL, and
    shared-memory sidecar are written into a private temporary directory.
    SQLite is opened only on those private copies, backed up into memory,
    vacuumed into a standalone private database, and read back as bounded
    bytes; no source inode is linked, written, checkpointed, or removed. The
    source entry metadata is checked before and after every copy and after the
    SQLite vacuum. Rollback-journal stores are rejected because their visible
    path is not a completed snapshot.

    This bounds the aggregate main/WAL/SHM input and closes the checked-path
    to SQLite-open race. It does not claim protection from a same-account
    actor mutating an already-open inode in place.
    """

    import sqlite3
    import tempfile

    limit = _bounded_limit(max_bytes)
    parts = _relative_parts(relative)
    parent_relative = Path(*parts[:-1]) if len(parts) > 1 else Path(".")
    main_name = parts[-1]
    wal_name = main_name + "-wal"
    shm_name = main_name + "-shm"
    journal_name = main_name + "-journal"
    with root.open_directory(parent_relative) as directory_fd:
        metadata = {
            name: _sqlite_entry_metadata(directory_fd, name)
            for name in (main_name, wal_name, shm_name, journal_name)
        }
        main_snapshot = metadata[main_name]
        if main_snapshot is None:
            raise FileNotFoundError(main_name)
        if metadata[journal_name] is not None:
            raise SecureReadChanged(
                "SQLite rollback journal is present; protected snapshot is not complete."
            )
        wal_snapshot = metadata[wal_name]
        shm_snapshot = metadata[shm_name]
        if wal_snapshot is not None and shm_snapshot is None:
            raise SecureReadChanged("SQLite WAL snapshot requires its shared-memory sidecar.")
        total_bytes = sum(
            int(snapshot[3])
            for snapshot in metadata.values()
            if snapshot is not None
        )
        if total_bytes > limit:
            raise SecureReadError(f"SQLite snapshot exceeds the {limit}-byte read budget.")

        if wal_snapshot is None:
            payload = _read_sqlite_entry_bytes(
                directory_fd,
                main_name,
                main_snapshot,
                max_bytes=limit,
            )
            after = {
                name: _sqlite_entry_metadata(directory_fd, name)
                for name in (main_name, wal_name, shm_name, journal_name)
            }
            if after != metadata:
                raise SecureReadChanged("SQLite entries changed during protected snapshot.")
            return payload

        with tempfile.TemporaryDirectory(prefix="oel-sqlite-snapshot-") as temporary:
            snapshot_dir = Path(temporary)
            try:
                _copy_sqlite_entry(
                    directory_fd,
                    main_name,
                    main_snapshot,
                    snapshot_dir / main_name,
                    max_bytes=limit,
                )
                _copy_sqlite_entry(
                    directory_fd,
                    wal_name,
                    wal_snapshot,
                    snapshot_dir / wal_name,
                    max_bytes=limit,
                )
                _copy_sqlite_entry(
                    directory_fd,
                    shm_name,
                    shm_snapshot,
                    snapshot_dir / shm_name,
                    max_bytes=limit,
                )
                after_copy = {
                    name: _sqlite_entry_metadata(directory_fd, name)
                    for name in (main_name, wal_name, shm_name, journal_name)
                }
                if after_copy != metadata:
                    raise SecureReadChanged("SQLite entries changed during protected snapshot copies.")
                source_uri = f"{(snapshot_dir / main_name).as_uri()}?mode=ro"
                canonical_path = snapshot_dir / "canonical.sqlite"
                source: sqlite3.Connection | None = None
                memory: sqlite3.Connection | None = None
                canonical: sqlite3.Connection | None = None
                try:
                    source = sqlite3.connect(source_uri, uri=True)
                    source.execute("PRAGMA query_only = ON")
                    source.execute("BEGIN")
                    memory = sqlite3.connect(":memory:")
                    source.backup(memory)
                    # VACUUM INTO is SQLite's supported way to materialize a
                    # standalone DELETE-mode database from the in-memory WAL
                    # snapshot. It writes only the new private destination;
                    # the descriptor-bound source is never checkpointed or
                    # changed, and no serialized WAL header is returned.
                    memory.commit()
                    quoted_target = memory.execute(
                        "SELECT quote(?)",
                        (str(canonical_path),),
                    ).fetchone()[0]
                    if not isinstance(quoted_target, str):
                        raise SecureReadError("SQLite could not quote its private vacuum destination.")
                    memory.execute(f"VACUUM INTO {quoted_target}")
                    after_backup = {
                        name: _sqlite_entry_metadata(directory_fd, name)
                        for name in (main_name, wal_name, shm_name, journal_name)
                    }
                    if after_backup != metadata:
                        raise SecureReadChanged("SQLite entries changed during protected snapshot vacuum.")
                    canonical = sqlite3.connect(
                        f"{canonical_path.as_uri()}?mode=ro",
                        uri=True,
                    )
                    canonical.execute("PRAGMA query_only = ON")
                    canonical.execute("SELECT name FROM sqlite_schema LIMIT 1").fetchone()
                    return _read_private_sqlite_snapshot(canonical_path, max_bytes=limit)
                except sqlite3.Error as exc:
                    raise SecureReadError("SQLite WAL snapshot could not be opened securely.") from exc
                finally:
                    if canonical is not None:
                        canonical.close()
                    if memory is not None:
                        memory.close()
                    if source is not None:
                        source.close()
            except SecureReadError:
                raise
            except OSError as exc:
                raise SecureReadError("SQLite WAL snapshot could not be copied securely.") from exc


@dataclass
class SecureReadRoot:
    """An approved directory held open for descriptor-relative input reads."""

    path: Path
    _fd: int
    approved_root: Path
    _root_identity: tuple[int, int]
    _closed: bool = False

    @classmethod
    def open(
        cls,
        path: str | Path,
        *,
        approved_roots: Sequence[str | Path],
        create: bool = False,
        mode: int = 0o700,
    ) -> SecureReadRoot:
        """Retain an approved directory, optionally creating descendants safely.

        ``create=True`` creates only missing components below an already
        existing approved root. Each component is created through its retained
        parent descriptor, reopened with ``O_NOFOLLOW|O_DIRECTORY``, and the
        final directory is chmod'ed through its descriptor. The default never
        creates or changes a path.
        """

        if create:
            _require_secure_create_primitives()
        target = _absolute_lexical(path)
        approved_root = _select_approved_root(target, approved_roots)
        anchor_fd = _open_directory_chain(approved_root)
        target_fd: int | None = None
        try:
            relative = target.relative_to(approved_root)
            target_fd = _open_relative_directory(
                anchor_fd,
                relative.parts,
                create=create,
                mode=mode,
            )
            if create:
                os.fchmod(target_fd, _directory_mode(mode))
            identity = _directory_identity(target_fd)
        except BaseException:
            _close_quietly(target_fd)
            raise
        finally:
            _close_quietly(anchor_fd)
        return cls(target, target_fd, approved_root, identity)

    @property
    def fd(self) -> int:
        if self._closed:
            raise ValueError("The secure input handle is closed.")
        return self._fd

    def dup_fd(self) -> int:
        """Duplicate the retained root descriptor for a cooperating consumer."""

        return os.dup(self.fd)

    def close(self) -> None:
        if not self._closed:
            _close_quietly(self._fd)
            self._closed = True

    def __del__(self) -> None:
        """Best-effort fallback for callers that forget explicit ``close``."""

        try:
            if not getattr(self, "_closed", True):
                _close_quietly(getattr(self, "_fd", None))
                self._closed = True
        except Exception:
            # Finalizers must not raise during interpreter shutdown.  Normal
            # callers still own the explicit close/context-manager contract.
            pass

    def __enter__(self) -> SecureReadRoot:
        if self._closed:
            raise ValueError("The secure input handle is closed.")
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def _parent_fd(self, parts: tuple[str, ...]) -> int:
        return _open_relative_directory(self.fd, parts[:-1])

    def _open_regular_fd(
        self,
        relative: str | Path,
        *,
        max_bytes: int | None,
    ) -> tuple[int, int, tuple[str, int, int, int, int, int, int]]:
        parts = _relative_parts(relative)
        parent_fd = self._parent_fd(parts)
        file_fd: int | None = None
        try:
            checked = os.stat(parts[-1], dir_fd=parent_fd, follow_symlinks=False)
            expected = _snapshot(checked)
            if expected[0] != "file":
                raise SecureReadError("Secure input reads require a regular file.")
            limit = _bounded_limit(max_bytes)
            if expected[3] > limit:
                raise SecureReadError(f"Secure input exceeds the {limit}-byte read budget.")
            file_fd = os.open(
                parts[-1],
                os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                dir_fd=parent_fd,
            )
            opened = os.fstat(file_fd)
            if not _same_snapshot(expected, opened):
                raise SecureReadChanged("Secure input changed before it was opened.")
            return parent_fd, file_fd, expected
        except BaseException:
            _close_quietly(file_fd)
            _close_quietly(parent_fd)
            raise

    @contextlib.contextmanager
    def open_file(
        self,
        relative: str | Path,
        *,
        max_bytes: int | None = None,
    ) -> Iterator[BinaryIO]:
        """Yield a checked regular-file stream and verify its inode after use."""

        parent_fd, file_fd, expected = self._open_regular_fd(relative, max_bytes=max_bytes)
        stream: BinaryIO | None = None
        try:
            stream = os.fdopen(os.dup(file_fd), "rb")
            yield stream
            after = os.fstat(file_fd)
            if not _same_snapshot(expected, after):
                raise SecureReadChanged("Secure input changed during its read.")
        finally:
            if stream is not None:
                stream.close()
            _close_quietly(file_fd)
            _close_quietly(parent_fd)

    def read_bytes(self, relative: str | Path, *, max_bytes: int | None = None) -> bytes:
        limit = _bounded_limit(max_bytes)
        with self.open_file(relative, max_bytes=limit) as stream:
            payload = stream.read(limit + 1)
        if len(payload) > limit:
            raise SecureReadError(f"Secure input exceeds the {limit}-byte read budget.")
        return payload

    def read_text(
        self,
        relative: str | Path,
        *,
        encoding: str = "utf-8",
        max_bytes: int | None = None,
    ) -> str:
        try:
            return self.read_bytes(relative, max_bytes=max_bytes).decode(encoding)
        except UnicodeDecodeError as exc:
            raise SecureReadError("Secure input is not valid text for the requested encoding.") from exc

    def sha256(self, relative: str | Path, *, max_bytes: int | None = None) -> str:
        limit = _bounded_limit(max_bytes)
        digest = hashlib.sha256()
        read = 0
        with self.open_file(relative, max_bytes=limit) as stream:
            while True:
                chunk = stream.read(READ_CHUNK_BYTES)
                if not chunk:
                    break
                read += len(chunk)
                if read > limit:
                    raise SecureReadError(f"Secure input exceeds the {limit}-byte read budget.")
                digest.update(chunk)
        return digest.hexdigest()

    def file_record(self, relative: str | Path, *, max_bytes: int | None = None) -> dict[str, int | str]:
        limit = _bounded_limit(max_bytes)
        digest = hashlib.sha256()
        size = 0
        parent_fd, file_fd, expected = self._open_regular_fd(relative, max_bytes=limit)
        stream: BinaryIO | None = None
        try:
            stream = os.fdopen(os.dup(file_fd), "rb")
            while True:
                chunk = stream.read(READ_CHUNK_BYTES)
                if not chunk:
                    break
                size += len(chunk)
                if size > limit:
                    raise SecureReadError(f"Secure input exceeds the {limit}-byte read budget.")
                digest.update(chunk)
            after = os.fstat(file_fd)
            if not _same_snapshot(expected, after) or size != int(after.st_size):
                raise SecureReadChanged("Secure input changed during its evidence read.")
            return {
                "bytes": int(size),
                "sha256": digest.hexdigest(),
                "st_dev": int(after.st_dev),
                "st_ino": int(after.st_ino),
                "mtime_ns": int(after.st_mtime_ns),
                "ctime_ns": int(after.st_ctime_ns),
                "nlink": int(after.st_nlink),
            }
        finally:
            if stream is not None:
                stream.close()
            _close_quietly(file_fd)
            _close_quietly(parent_fd)

    def stat(self, relative: str | Path) -> os.stat_result:
        """Return no-follow metadata for one anchored regular file."""

        parts = _relative_parts(relative)
        parent_fd = self._parent_fd(parts)
        file_fd: int | None = None
        try:
            checked = os.stat(parts[-1], dir_fd=parent_fd, follow_symlinks=False)
            if not stat.S_ISREG(checked.st_mode):
                raise SecureReadError("Secure input metadata requires a regular file.")
            file_fd = os.open(
                parts[-1],
                os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                dir_fd=parent_fd,
            )
            opened = os.fstat(file_fd)
            if not _same_snapshot(_snapshot(checked), opened):
                raise SecureReadChanged("Secure input changed before metadata was read.")
            return opened
        finally:
            _close_quietly(file_fd)
            _close_quietly(parent_fd)

    def entry_stat(self, relative: str | Path) -> os.stat_result:
        """Return no-follow metadata for any anchored directory entry.

        This narrow variant is for callers that must distinguish a missing
        entry from an existing special file before choosing a read path.  It
        never opens the entry and therefore cannot block on a FIFO/device.
        """

        parts = _relative_parts(relative)
        parent_fd = self._parent_fd(parts)
        try:
            return os.stat(parts[-1], dir_fd=parent_fd, follow_symlinks=False)
        finally:
            _close_quietly(parent_fd)

    def list_names(self, relative: str | Path = ".") -> tuple[str, ...]:
        """List one anchored directory and reject a changed directory identity."""

        if str(relative) == ".":
            parts: tuple[str, ...] = ()
        else:
            parts = _relative_parts(relative)
        directory_fd = _open_relative_directory(self.fd, parts)
        identity = _directory_identity(directory_fd)
        try:
            names = tuple(sorted(str(name) for name in os.listdir(directory_fd)))
            if _directory_identity(directory_fd) != identity:
                raise SecureReadChanged("Secure input directory changed during listing.")
            return names
        finally:
            _close_quietly(directory_fd)

    @contextlib.contextmanager
    def open_directory(self, relative: str | Path = ".") -> Iterator[int]:
        """Yield a retained child directory descriptor for a cooperating reader."""

        parts = () if str(relative) == "." else _relative_parts(relative)
        directory_fd = _open_relative_directory(self.fd, parts)
        identity = _directory_identity(directory_fd)
        try:
            yield directory_fd
            if _directory_identity(directory_fd) != identity:
                raise SecureReadChanged("Secure input directory changed during use.")
        finally:
            _close_quietly(directory_fd)


class DescriptorStore:
    """Small record store anchored to a :class:`SecureReadRoot` descriptor."""

    def __init__(self, root: SecureReadRoot, *, max_record_bytes: int = 4_000_000) -> None:
        self.root = root
        self.max_record_bytes = _bounded_limit(max_record_bytes)

    def dup_fd(self) -> int:
        return self.root.dup_fd()

    def read_bytes(self, relative: str | Path) -> bytes:
        return self.root.read_bytes(relative, max_bytes=self.max_record_bytes)

    def stat(self, relative: str | Path) -> os.stat_result:
        return self.root.stat(relative)

    def entry_stat(self, relative: str | Path) -> os.stat_result:
        return self.root.entry_stat(relative)

    def list_names(self, relative: str | Path = ".") -> tuple[str, ...]:
        return self.root.list_names(relative)

    def _parent_fd(self, parts: tuple[str, ...]) -> int:
        return _open_relative_directory(self.root.fd, parts[:-1])

    def write_new(
        self,
        relative: str | Path,
        payload: bytes,
        *,
        mode: int = 0o600,
    ) -> os.stat_result:
        """Create a regular record exclusively and return its opened identity."""

        data = bytes(payload)
        if len(data) > self.max_record_bytes:
            raise SecureReadError("Descriptor store record exceeds its size budget.")
        parts = _relative_parts(relative)
        parent_fd = self._parent_fd(parts)
        file_fd: int | None = None
        try:
            file_fd = os.open(
                parts[-1],
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                mode,
                dir_fd=parent_fd,
            )
            metadata = os.fstat(file_fd)
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                raise SecureReadError("Descriptor store requires a regular record.")
            _write_all(file_fd, data)
            os.fsync(file_fd)
            os.fsync(parent_fd)
            after = os.fstat(file_fd)
            if not stat.S_ISREG(after.st_mode) or after.st_nlink != 1:
                raise SecureReadChanged("Descriptor store record gained an unexpected link.")
            return after
        finally:
            _close_quietly(file_fd)
            _close_quietly(parent_fd)

    def write_existing(
        self,
        relative: str | Path,
        payload: bytes,
        *,
        expected_identity: tuple[int, int] | None = None,
    ) -> os.stat_result:
        """Update the currently opened inode without following a replacement."""

        data = bytes(payload)
        if len(data) > self.max_record_bytes:
            raise SecureReadError("Descriptor store record exceeds its size budget.")
        parts = _relative_parts(relative)
        parent_fd = self._parent_fd(parts)
        file_fd: int | None = None
        try:
            file_fd = os.open(
                parts[-1], os.O_WRONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent_fd
            )
            before = os.fstat(file_fd)
            if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
                raise SecureReadError("Descriptor store requires a regular record.")
            identity = (int(before.st_dev), int(before.st_ino))
            if expected_identity is not None and identity != expected_identity:
                raise SecureReadChanged("Descriptor store record changed before update.")
            os.ftruncate(file_fd, 0)
            _write_all(file_fd, data)
            os.fsync(file_fd)
            after = os.fstat(file_fd)
            if (
                (int(after.st_dev), int(after.st_ino)) != identity
                or not stat.S_ISREG(after.st_mode)
                or after.st_nlink != 1
            ):
                raise SecureReadChanged("Descriptor store record changed during update.")
            os.fsync(parent_fd)
            return after
        finally:
            _close_quietly(file_fd)
            _close_quietly(parent_fd)

    def write_atomic(
        self,
        relative: str | Path,
        payload: bytes,
        *,
        mode: int = 0o600,
        expected_identity: tuple[int, int] | None = None,
    ) -> os.stat_result:
        """Publish a complete replacement under the retained parent descriptor.

        The temporary record is created and fsynced through the anchored
        descriptor.  ``os.replace`` is atomic for ordinary cooperating
        writers, but it cannot atomically assert a target inode on portable
        POSIX APIs.  Callers coordinating ordinary writers may hold
        :meth:`lock` and use an expected identity.  The lock is advisory and
        does not constrain a hostile same-account process; the residual rename
        race remains an explicit limitation.  If publication fails before
        ``os.replace`` returns, the temporary name is retained.  Once replace
        returns, later fsync or identity failures leave the published entry in
        place and are reported without attempting a race-prone rollback.
        """

        data = bytes(payload)
        if len(data) > self.max_record_bytes:
            raise SecureReadError("Descriptor store record exceeds its size budget.")
        parts = _relative_parts(relative)
        parent_fd = self._parent_fd(parts)
        temporary_name = f".{parts[-1]}.{secrets.token_hex(12)}.tmp"
        temporary_fd: int | None = None
        try:
            if expected_identity is not None:
                current_fd = os.open(
                    parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent_fd
                )
                try:
                    current = os.fstat(current_fd)
                    if (
                        not stat.S_ISREG(current.st_mode)
                        or current.st_nlink != 1
                        or (int(current.st_dev), int(current.st_ino)) != expected_identity
                    ):
                        raise SecureReadChanged("Descriptor store record changed before replacement.")
                finally:
                    _close_quietly(current_fd)
            temporary_fd = os.open(
                temporary_name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                mode,
                dir_fd=parent_fd,
            )
            temporary_stat = os.fstat(temporary_fd)
            if not stat.S_ISREG(temporary_stat.st_mode) or temporary_stat.st_nlink != 1:
                raise SecureReadError("Descriptor store requires a regular temporary record.")
            _write_all(temporary_fd, data)
            os.fsync(temporary_fd)
            if os.fstat(temporary_fd).st_nlink != 1:
                raise SecureReadChanged("Descriptor store temporary record gained an unexpected link.")
            os.replace(temporary_name, parts[-1], src_dir_fd=parent_fd, dst_dir_fd=parent_fd)
            os.fsync(parent_fd)
            current_fd = os.open(
                parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent_fd
            )
            try:
                current = os.fstat(current_fd)
                if not stat.S_ISREG(current.st_mode) or current.st_nlink != 1:
                    raise SecureReadError("Descriptor store requires a regular published record.")
                return current
            finally:
                _close_quietly(current_fd)
        finally:
            _close_quietly(temporary_fd)
            # A failed pre-publication rename leaves a mutable name.  There is
            # no portable unlink-by-inode primitive, so retaining that private
            # temporary is safer than checking then unlinking a name that a
            # same-account actor could replace between the two calls.  After a
            # successful replace the temporary name is gone; later errors do
            # not attempt to roll the published entry back.
            _close_quietly(parent_fd)

    @contextlib.contextmanager
    def lock(self, relative: str | Path = ".control.lock") -> Iterator[int]:
        """Hold an advisory lock on a verified, single-link regular record.

        The lock serializes cooperating writers only.  A same-account process
        can replace the lock name while this descriptor remains open, so the
        named identity is checked after acquisition and callers must treat a
        mismatch as an unknown outcome.
        """

        if fcntl is None:
            raise SecureReadUnavailable("Anchored record locks require POSIX fcntl support.")
        parts = _relative_parts(relative)
        parent_fd = self._parent_fd(parts)
        lock_fd: int | None = None
        try:
            lock_fd = os.open(
                parts[-1],
                os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK,
                0o600,
                dir_fd=parent_fd,
            )
            before = os.fstat(lock_fd)
            if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
                raise SecureReadError("Descriptor store locks require a single-link regular record.")
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            after = os.fstat(lock_fd)
            named = os.stat(parts[-1], dir_fd=parent_fd, follow_symlinks=False)
            if (
                not stat.S_ISREG(after.st_mode)
                or after.st_nlink != 1
                or (int(after.st_dev), int(after.st_ino))
                != (int(before.st_dev), int(before.st_ino))
                or (int(named.st_dev), int(named.st_ino))
                != (int(after.st_dev), int(after.st_ino))
            ):
                raise SecureReadChanged("Descriptor store lock name changed during acquisition.")
            yield lock_fd
        finally:
            if lock_fd is not None:
                try:
                    fcntl.flock(lock_fd, fcntl.LOCK_UN)
                except OSError:
                    pass
            _close_quietly(lock_fd)
            _close_quietly(parent_fd)


def _write_all(fd: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(fd, view)
        if written <= 0:
            raise OSError("Descriptor store write made no progress.")
        view = view[written:]


__all__ = [
    "DEFAULT_MAX_READ_BYTES",
    "DescriptorStore",
    "READ_CHUNK_BYTES",
    "SecureReadChanged",
    "SecureReadError",
    "SecureReadRoot",
    "SecureReadUnavailable",
    "require_secure_read_primitives",
    "secure_read_primitives_available",
]
