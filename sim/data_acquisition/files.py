"""Bounded snapshots and atomic publications in user-owned acquisition storage."""
from __future__ import annotations

import os
import shutil
import stat
import tempfile
from pathlib import Path

from sim.platform_compat import canonicalize_platform_compatibility_root

from .contracts import AcquisitionError, canonical


def private_dir(path: Path):
    path = canonicalize_platform_compatibility_root(path)
    # Reject symlinks at every existing component; these are owned storage roots.
    for part in (path, *path.parents):
        if part.is_symlink():
            raise AcquisitionError("UNSAFE_PATH", "Acquisition storage must not traverse symbolic links.")
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    return path


def headroom(path: Path, growth: int):
    existing = path
    while not existing.exists():
        existing = existing.parent
    if shutil.disk_usage(existing).free - growth < 5 * 1024**3:
        raise AcquisitionError("STORAGE_LIMIT", "Acquisition must preserve at least 5 GiB free disk space.")


def read_snapshot(path: Path, max_bytes: int) -> bytes:
    try:
        if path.is_symlink():
            raise AcquisitionError("UNSAFE_PATH", "Select a regular file, not a symbolic link.")
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
        with os.fdopen(os.open(path, flags), "rb") as stream:
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise AcquisitionError("UNSAFE_PATH", "Select a regular file.")
            if before.st_size > max_bytes:
                raise AcquisitionError("SIZE_LIMIT", "File exceeds max_bytes.")
            data = stream.read(max_bytes + 1)
            after = os.fstat(stream.fileno())
        def identity(s):
            return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns
        if identity(before) != identity(after) or identity(after) != identity(path.stat()) or len(data) != before.st_size:
            raise AcquisitionError("SOURCE_CHANGED", "Source changed during registration; retry with a stable file.")
        if len(data) > max_bytes:
            raise AcquisitionError("SIZE_LIMIT", "File exceeds max_bytes.")
        return data
    except OSError as exc:
        raise AcquisitionError("FILE_ERROR", "Unable to snapshot the selected regular file.") from exc


def publish(path: Path, data: bytes, *, replace: bool = False):
    path = canonicalize_platform_compatibility_root(path)
    private_dir(path.parent)
    headroom(path.parent, len(data))
    if path.is_symlink():
        raise AcquisitionError("UNSAFE_PATH", "Publication target must not be a symbolic link.")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".acquire-", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        if replace:
            os.replace(temporary, path)
        else:
            try:
                os.link(temporary, path)
            except FileExistsError as exc:
                raise AcquisitionError("DESTINATION_EXISTS", "Destination exists; select a new path.") from exc
    finally:
        if temporary:
            temporary.unlink(missing_ok=True)


def publish_json(path, value, *, replace=False):
    publish(path, canonical(value), replace=replace)
