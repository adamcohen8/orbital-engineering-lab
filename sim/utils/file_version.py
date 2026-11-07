"""File change tokens for numeric resource caches on coarse timestamp clocks."""

from __future__ import annotations

import hashlib
import os
import stat
import time
from collections import OrderedDict
from threading import RLock

_stat = os.stat
_digests: OrderedDict[tuple, str] = OrderedDict()
_lock = RLock()


def file_version(path: str | os.PathLike[str]) -> tuple:
    """Disambiguate writes within a filesystem timestamp tick using content.

    Linux can give two same-size writes identical mtime and ctime values.
    Hash recently changed files rather than assuming nanosecond fields imply
    nanosecond resolution. Once a timestamp has settled, a later ordinary
    write changes metadata, retaining the cheap steady-state cache lookup.
    Future timestamps remain ambiguous and are hashed as well.
    """
    metadata = _stat(path)
    if not stat.S_ISREG(metadata.st_mode):
        raise OSError("Numeric resource must be a regular file.")
    metadata_key = (metadata.st_ino, metadata.st_size, metadata.st_mtime_ns, metadata.st_ctime_ns)
    key = (os.path.abspath(path), *metadata_key)
    with _lock:
        digest = _digests.get(key)
    if time.time_ns() - max(metadata.st_mtime_ns, metadata.st_ctime_ns) <= 2_000_000_000:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_BINARY", 0))
        with os.fdopen(descriptor, "rb") as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                raise OSError("Numeric resource must be a regular file.")
            hasher = hashlib.sha256()
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                hasher.update(chunk)
            digest = hasher.hexdigest()
        with _lock:
            _digests[key] = digest
            _digests.move_to_end(key)
            while len(_digests) > 256:
                _digests.popitem(last=False)
    return (*metadata_key, digest)
