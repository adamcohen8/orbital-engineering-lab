"""Safe extraction for installable OEL release artifacts."""

from __future__ import annotations

import os
import shutil
import stat
import tarfile
import zipfile
from pathlib import Path, PurePosixPath


class UnsafeArchiveError(ValueError):
    """Raised when an archive contains an unsafe or ambiguous member."""


def _validated_relative(name: str) -> PurePosixPath:
    normalized = str(name).replace("\\", "/")
    path = PurePosixPath(normalized)
    if not normalized or not path.parts or normalized in {".", "./"} or path.is_absolute() or ".." in path.parts:
        raise UnsafeArchiveError(f"Archive member escapes the destination: {name!r}")
    if path.parts and ":" in path.parts[0]:
        raise UnsafeArchiveError(f"Archive member uses a drive-qualified path: {name!r}")
    return path


def _is_os_path_alias(path: Path) -> bool:
    """Allow only the standard macOS temporary-directory aliases."""

    if os.name == "nt":
        return False
    aliases = {Path("/tmp"): Path("/private/tmp"), Path("/var"): Path("/private/var")}
    target = aliases.get(path)
    return target is not None and path.resolve(strict=False) == target


def _validated_destination(destination: str | Path) -> Path:
    """Reject symlink routing while preserving macOS ``/tmp`` and ``/var`` aliases."""

    requested = Path(destination).expanduser().absolute()
    for node in (requested, *requested.parents):
        if node.is_symlink() and not _is_os_path_alias(node):
            raise UnsafeArchiveError(f"Extraction destination may not use a symbolic-link parent: {node}")
    return requested.resolve(strict=False)


def safe_extract(archive_path: str | Path, destination: str | Path, *, max_bytes: int = 8_000_000_000) -> Path:
    archive = Path(archive_path).expanduser().resolve()
    target = _validated_destination(destination)
    target.mkdir(parents=True, exist_ok=True)
    if tarfile.is_tarfile(archive):
        _extract_tar(archive, target, max_bytes=max_bytes)
    elif zipfile.is_zipfile(archive):
        _extract_zip(archive, target, max_bytes=max_bytes)
    else:
        raise UnsafeArchiveError(f"Unsupported release archive: {archive}")
    return target


def _prepare_member_output(destination: Path, relative: PurePosixPath, *, directory: bool) -> Path:
    """Create one archive member without following pre-existing links."""

    current = destination
    parts = relative.parts
    for index, part in enumerate(parts):
        current = current / part
        if current.is_symlink():
            raise UnsafeArchiveError(f"Archive extraction path contains a symbolic link: {current}")
        final = index == len(parts) - 1
        if current.exists():
            if final:
                if directory and not current.is_dir():
                    raise UnsafeArchiveError(f"Archive directory collides with a non-directory: {current}")
                if not directory:
                    # Do not clobber an existing file or special node in a
                    # caller-provided destination.
                    raise UnsafeArchiveError(f"Archive file would overwrite an existing path: {current}")
            elif not current.is_dir():
                raise UnsafeArchiveError(f"Archive extraction parent is not a directory: {current}")
        elif final and directory:
            current.mkdir()
        elif not final:
            current.mkdir()
    return current


def _extract_tar(archive: Path, destination: Path, *, max_bytes: int) -> None:
    total = 0
    names: set[str] = set()
    with tarfile.open(archive, "r:*") as source:
        members = source.getmembers()
        for member in members:
            relative = _validated_relative(member.name)
            name = relative.as_posix()
            if name in names:
                raise UnsafeArchiveError(f"Archive contains duplicate member: {name}")
            names.add(name)
            if member.issym() or member.islnk() or member.isdev() or member.isfifo():
                raise UnsafeArchiveError(f"Archive contains unsupported special member: {name}")
            if not (member.isdir() or member.isfile()):
                raise UnsafeArchiveError(f"Archive contains unsupported member type: {name}")
            total += max(0, int(member.size))
            if total > max_bytes:
                raise UnsafeArchiveError(f"Archive expands beyond the {max_bytes} byte safety limit.")
        for member in members:
            relative = _validated_relative(member.name)
            if member.isdir():
                _prepare_member_output(destination, relative, directory=True)
                continue
            output = _prepare_member_output(destination, relative, directory=False)
            stream = source.extractfile(member)
            if stream is None:
                raise UnsafeArchiveError(f"Could not read archive member: {member.name}")
            with stream, output.open("wb") as sink:
                shutil.copyfileobj(stream, sink)
            output.chmod(0o755 if member.mode & 0o111 else 0o644)


def _extract_zip(archive: Path, destination: Path, *, max_bytes: int) -> None:
    total = 0
    names: set[str] = set()
    with zipfile.ZipFile(archive) as source:
        members = source.infolist()
        for member in members:
            relative = _validated_relative(member.filename)
            name = relative.as_posix()
            if name in names:
                raise UnsafeArchiveError(f"Archive contains duplicate member: {name}")
            names.add(name)
            mode = member.external_attr >> 16
            if stat.S_ISLNK(mode) or stat.S_ISCHR(mode) or stat.S_ISBLK(mode) or stat.S_ISFIFO(mode):
                raise UnsafeArchiveError(f"Archive contains unsupported special member: {name}")
            total += max(0, int(member.file_size))
            if total > max_bytes:
                raise UnsafeArchiveError(f"Archive expands beyond the {max_bytes} byte safety limit.")
        for member in members:
            relative = _validated_relative(member.filename)
            if member.is_dir():
                _prepare_member_output(destination, relative, directory=True)
                continue
            output = _prepare_member_output(destination, relative, directory=False)
            with source.open(member) as stream, output.open("wb") as sink:
                shutil.copyfileobj(stream, sink)
            mode = member.external_attr >> 16
            output.chmod(0o755 if mode & 0o111 else 0o644)
