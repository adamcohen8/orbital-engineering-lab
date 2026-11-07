"""Atomic state and local transaction locking for OEL installations."""

from __future__ import annotations

import json
import os
import tempfile
import time
import uuid
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

from sim.platform_compat import canonicalize_platform_compatibility_root as _canonicalize_platform_root
from sim.security.output_writes import SecureWriteRoot, secure_write_primitives_available

from .contracts import INSTALLATION_STATE_SCHEMA


def secure_state_write_available() -> bool:
    """Return whether descriptor-relative state replacement is available."""

    return secure_write_primitives_available() and os.rename in getattr(os, "supports_dir_fd", set())


def _atomic_write_text_descriptor_bound(
    path: str | Path,
    text: str,
    *,
    approved_root: str | Path,
) -> Path:
    """Replace one installer-owned file through a retained parent descriptor.

    This is intentionally opt-in for installer state and launchers.  The
    legacy path-only helpers below remain available for workspace/CLI callers
    and for hosts without POSIX descriptor primitives.
    """

    destination = Path(path)
    if not secure_state_write_available():
        return atomic_write_text(destination, text)
    secure_destination = _canonicalize_platform_root(destination)
    secure_approved_root = _canonicalize_platform_root(Path(approved_root))
    temporary_name: str | None = None
    temporary_fd: int | None = None
    with SecureWriteRoot.open(
        secure_destination.parent,
        approved_roots=(secure_approved_root,),
        create=True,
    ) as parent:
        try:
            temporary_name = f".{destination.name}.{uuid.uuid4().hex}.tmp"
            temporary_fd = os.open(
                temporary_name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
                0o600,
                dir_fd=parent.fd,
            )
            with os.fdopen(temporary_fd, "w", encoding="utf-8") as stream:
                temporary_fd = None
                stream.write(text)
                stream.flush()
                os.fsync(stream.fileno())
            # ``rename`` is in supports_dir_fd on POSIX and has the required
            # same-directory replacement semantics without following a
            # destination symlink.
            os.rename(
                temporary_name,
                secure_destination.name,
                src_dir_fd=parent.fd,
                dst_dir_fd=parent.fd,
            )
            os.fsync(parent.fd)
            temporary_name = None
            return destination
        finally:
            if temporary_fd is not None:
                try:
                    os.close(temporary_fd)
                except OSError:
                    pass
            if temporary_name is not None:
                try:
                    os.unlink(temporary_name, dir_fd=parent.fd)
                except OSError:
                    pass


def atomic_write_text_descriptor_bound(
    path: str | Path,
    text: str,
    *,
    approved_root: str | Path,
) -> Path:
    """Write installer-owned text with a retained parent when supported."""

    return _atomic_write_text_descriptor_bound(path, text, approved_root=approved_root)


def atomic_write_json_descriptor_bound(
    path: str | Path,
    payload: dict[str, Any],
    *,
    approved_root: str | Path,
) -> Path:
    """Serialize installer-owned JSON and replace it through a retained parent."""

    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    return _atomic_write_text_descriptor_bound(path, text, approved_root=approved_root)


def unlink_descriptor_bound(path: str | Path, *, approved_root: str | Path) -> None:
    """Remove one installer-owned leaf relative to a retained parent."""

    destination = Path(path)
    if not secure_state_write_available():
        destination.unlink(missing_ok=True)
        return
    secure_destination = _canonicalize_platform_root(destination)
    secure_approved_root = _canonicalize_platform_root(Path(approved_root))
    try:
        with SecureWriteRoot.open(
            secure_destination.parent,
            approved_roots=(secure_approved_root,),
            create=False,
        ) as parent:
            try:
                os.unlink(secure_destination.name, dir_fd=parent.fd)
            except FileNotFoundError:
                pass
    except FileNotFoundError:
        pass


def atomic_write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(destination)
    except Exception:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise
    return destination


def atomic_write_text(path: str | Path, text: str) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(destination)
    except Exception:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise
    return destination


def read_state(path: str | Path, *, default: dict[str, Any] | None = None) -> dict[str, Any]:
    source = Path(path)
    if not source.exists():
        return dict(default or {})
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"State file must contain a JSON object: {source}")
    return value


def empty_installation_state() -> dict[str, Any]:
    return {"schema_version": INSTALLATION_STATE_SCHEMA, "installations": {}, "history": []}


class StateLock(AbstractContextManager["StateLock"]):
    def __init__(self, path: str | Path, *, operation: str, stale_after_s: float = 3600.0) -> None:
        self.path = Path(path)
        self.operation = operation
        self.stale_after_s = float(stale_after_s)
        self.transaction_id = str(uuid.uuid4())
        self._owned = False

    def __enter__(self) -> StateLock:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": "oel.update-lock.v1",
            "transaction_id": self.transaction_id,
            "operation": self.operation,
            "pid": os.getpid(),
            "created_unix_s": time.time(),
        }
        try:
            descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError as exc:
            existing = self._existing()
            age = time.time() - float(existing.get("created_unix_s", time.time()))
            hint = " Run `oel update status` and use the documented recovery flow; do not delete an active lock."
            if age > self.stale_after_s:
                hint = " The lock appears stale; run `oel update recover-lock` after confirming no update process is active."
            raise RuntimeError(f"Another OEL state transaction is active: {existing}.{hint}") from exc
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, sort_keys=True)
            stream.write("\n")
        self._owned = True
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if self._owned:
            try:
                current = self._existing()
                if current.get("transaction_id") == self.transaction_id:
                    self.path.unlink()
            finally:
                self._owned = False

    def _existing(self) -> dict[str, Any]:
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {"path": str(self.path), "status": "unreadable"}
        return value if isinstance(value, dict) else {"path": str(self.path), "status": "invalid"}


def recover_stale_lock(path: str | Path, *, stale_after_s: float = 3600.0) -> dict[str, Any]:
    lock = Path(path)
    if not lock.exists():
        return {"status": "ready", "removed": False, "path": str(lock)}
    try:
        payload = json.loads(lock.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Cannot safely recover unreadable update lock: {lock}") from exc
    age = time.time() - float(payload.get("created_unix_s", time.time()))
    if age <= float(stale_after_s):
        raise RuntimeError(f"Update lock is not stale ({age:.1f} seconds old): {lock}")
    lock.unlink()
    return {"status": "ready", "removed": True, "path": str(lock), "previous": payload}
