"""Small, dependency-free operating-system compatibility helpers."""

from __future__ import annotations

import os
import platform
import stat
import subprocess
import sys
import time
from multiprocessing.context import BaseContext
from pathlib import Path
from typing import Any, Callable

try:
    import resource as _resource
except ImportError:  # Windows does not provide the Unix ``resource`` module.
    _resource = None  # type: ignore[assignment]


def canonicalize_platform_compatibility_root(path: Path) -> Path:
    """Expand only immutable macOS compatibility roots before no-follow traversal.

    macOS exposes ``/tmp`` and ``/var`` as root-owned compatibility symlinks
    into ``/private``. Rejecting those two platform aliases makes ordinary
    ``tempfile`` paths unusable, while resolving the complete caller path would
    silently follow untrusted descendant symlinks. Rewrite only the known
    system root and leave every remaining component for descriptor-anchored
    ``O_NOFOLLOW`` traversal.
    """

    if sys.platform != "darwin" or len(path.parts) < 2:
        return path
    alias = Path(os.sep) / path.parts[1]
    expected = {Path("/tmp"): Path("/private/tmp"), Path("/var"): Path("/private/var")}.get(alias)
    if expected is None:
        return path
    try:
        metadata = os.lstat(alias)
        resolved = Path(os.path.realpath(alias))
    except OSError:
        return path
    if not stat.S_ISLNK(metadata.st_mode) or metadata.st_uid != 0 or resolved != expected:
        return path
    return expected.joinpath(*path.parts[2:])


def open_folder(
    path: str | Path,
    *,
    platform_name: str | None = None,
    popen: Callable[[list[str]], Any] | None = None,
    startfile: Callable[[str], Any] | None = None,
) -> Path:
    """Open a local folder with the native OS shell and return its resolved path."""
    folder = Path(path).expanduser()
    if not folder.is_absolute():
        folder = (Path.cwd() / folder).resolve()
    if folder.is_file():
        folder = folder.parent
    if not folder.is_dir():
        raise FileNotFoundError(f"Folder does not exist: {folder}")

    target_platform = str(platform_name or sys.platform).lower()
    if target_platform.startswith("win"):
        opener = startfile if startfile is not None else getattr(os, "startfile", None)
        if opener is None:
            raise OSError("The Windows folder opener is unavailable.")
        opener(str(folder))
    else:
        launch = popen if popen is not None else subprocess.Popen
        executable = "open" if target_platform == "darwin" else "xdg-open"
        launch([executable, str(folder)])
    return folder


def process_context(*, start_method: str | None = None) -> BaseContext:
    """Return the multiprocessing context used by OEL object workers."""
    from multiprocessing import get_context

    method = start_method
    if method is None and sys.platform.startswith("win"):
        method = "spawn"
    return get_context(method)


def max_rss_bytes() -> int | None:
    """Return peak process RSS in bytes when the host exposes that metric."""
    if _resource is None:
        return None
    try:
        usage = _resource.getrusage(_resource.RUSAGE_SELF)
        value = int(getattr(usage, "ru_maxrss", 0))
    except Exception:
        return None
    if value <= 0:
        return None
    # Linux and the BSDs report KiB; macOS reports bytes.
    return value if sys.platform == "darwin" else value * 1024


def max_rss_mb() -> float | None:
    value = max_rss_bytes()
    return None if value is None else float(value) / (1024.0 * 1024.0)


def process_cpu_time_seconds(*, include_children: bool = True) -> float:
    """Return process CPU time, including child processes when supported."""
    if _resource is None:
        return float(time.process_time())
    try:
        own = _resource.getrusage(_resource.RUSAGE_SELF)
        total = float(own.ru_utime + own.ru_stime)
        if include_children:
            children = _resource.getrusage(_resource.RUSAGE_CHILDREN)
            total += float(children.ru_utime + children.ru_stime)
        return total
    except Exception:
        return float(time.process_time())


def available_memory_bytes() -> int | None:
    """Return an estimate of immediately available physical memory."""
    if platform.system() == "Windows":
        return _windows_available_memory_bytes()

    names = ("SC_AVPHYS_PAGES", "SC_PAGE_SIZE")
    if hasattr(os, "sysconf") and all(name in os.sysconf_names for name in names):
        try:
            pages = int(os.sysconf("SC_AVPHYS_PAGES"))
            page_size = int(os.sysconf("SC_PAGE_SIZE"))
            if pages > 0 and page_size > 0:
                return pages * page_size
        except (OSError, ValueError):
            pass
    if sys.platform == "darwin":
        return _macos_available_memory_bytes()
    return None


def _windows_available_memory_bytes() -> int | None:
    try:
        import ctypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MEMORYSTATUSEX()
        status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return None
        value = int(status.ullAvailPhys)
        return value if value > 0 else None
    except Exception:
        return None


def _macos_available_memory_bytes() -> int | None:
    try:
        output = subprocess.check_output(
            ["vm_stat"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        page_size = 4096
        available_pages = 0
        for line in output.splitlines():
            text = line.strip()
            if "page size of" in text:
                page_size = int(text.split("page size of", 1)[1].strip().split(" ", 1)[0])
            elif text.startswith(("Pages free:", "Pages inactive:", "Pages speculative:")):
                available_pages += int(text.split(":", 1)[1].strip().rstrip("."))
        return available_pages * page_size if available_pages > 0 else None
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


__all__ = [
    "available_memory_bytes",
    "canonicalize_platform_compatibility_root",
    "max_rss_bytes",
    "max_rss_mb",
    "open_folder",
    "process_context",
    "process_cpu_time_seconds",
]
