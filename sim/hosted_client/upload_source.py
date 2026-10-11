"""Public-client source validation before bounded network staging."""
import os
import stat
from pathlib import Path


def read_scenario(value):
    path = Path(value).expanduser().absolute()
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("Upload path contains a symbolic link")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(descriptor, "rb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ValueError("Upload requires a regular file")
        payload = stream.read(1024 * 1024 + 1)
    if not 0 < len(payload) <= 1024 * 1024:
        raise ValueError("Network scenario must contain between 1 byte and 1 MiB")
    return payload
