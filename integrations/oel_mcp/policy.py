from __future__ import annotations

import contextlib
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

from integrations.oel_mcp.contracts import DEPLOYMENT_PROFILES
from integrations.oel_mcp.recovery import annotate_error
from sim.platform_compat import canonicalize_platform_compatibility_root
from sim.security.input_reads import (
    DEFAULT_MAX_READ_BYTES,
    SecureReadRoot,
)

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class BoundReadHandle:
    """A retained MCP input root for cooperating path-only consumers."""

    logical_root: Path
    reader: SecureReadRoot

    def _relative(self, value: str | Path) -> Path:
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = self.logical_root / path
        path = canonicalize_platform_compatibility_root(Path(os.path.abspath(path)))
        try:
            relative = path.relative_to(self.logical_root)
        except ValueError as exc:
            raise PermissionError("Input path is outside its retained MCP read root.") from exc
        if not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
            raise PermissionError("Input path is not a normalized child of its retained MCP read root.")
        return relative

    def read_bytes(self, value: str | Path, *, max_bytes: int | None = DEFAULT_MAX_READ_BYTES) -> bytes:
        return self.reader.read_bytes(self._relative(value), max_bytes=max_bytes)

    def read_text(
        self,
        value: str | Path,
        *,
        encoding: str = "utf-8",
        max_bytes: int | None = DEFAULT_MAX_READ_BYTES,
    ) -> str:
        return self.reader.read_text(self._relative(value), encoding=encoding, max_bytes=max_bytes)

    def file_record(
        self,
        value: str | Path,
        *,
        max_bytes: int | None = DEFAULT_MAX_READ_BYTES,
    ) -> dict[str, int | str]:
        return self.reader.file_record(self._relative(value), max_bytes=max_bytes)

    def stat(self, value: str | Path) -> os.stat_result:
        return self.reader.stat(self._relative(value))

    def is_file(self, value: str | Path) -> bool:
        try:
            self.stat(value)
        except (FileNotFoundError, NotADirectoryError):
            return False
        return True


@dataclass(frozen=True)
class MCPPathPolicy:
    read_roots: tuple[Path, ...]
    write_roots: tuple[Path, ...]

    @classmethod
    def configured(
        cls,
        *,
        read_roots: tuple[str | Path, ...] | None = None,
        write_roots: tuple[str | Path, ...] | None = None,
    ) -> MCPPathPolicy:
        legacy_roots = _legacy_roots()
        reads = (
            tuple(read_roots)
            if read_roots is not None
            else _roots_from_environment("OEL_MCP_READ_ROOTS") or legacy_roots or (ROOT,)
        )
        writes = (
            tuple(write_roots)
            if write_roots is not None
            else _roots_from_environment("OEL_MCP_WRITE_ROOTS") or legacy_roots
        )
        return cls(
            read_roots=tuple(Path(root).expanduser().resolve() for root in reads),
            write_roots=tuple(Path(root).expanduser().resolve() for root in writes),
        )

    def resolve_read(self, value: str | Path, *, kind: str = "any") -> Path:
        path = self._resolve(value)
        self._require_root(path, self.read_roots)
        if kind == "file" and not path.is_file():
            raise FileNotFoundError("Required authorized input file was not found.")
        if kind == "directory" and not path.is_dir():
            raise FileNotFoundError("Required authorized input directory was not found.")
        if kind == "any" and not path.exists():
            raise FileNotFoundError("Required authorized input path was not found.")
        return path

    @contextlib.contextmanager
    def retain_read_root(self, value: str | Path) -> Iterator[tuple[Path, SecureReadRoot]]:
        """Retain an approved input directory while a consumer uses it.

        ``resolve_read`` remains a compatibility path for callers that only
        need a logical name.  Security-sensitive readers must keep the
        directory descriptor open and perform child reads relative to it so a
        replacement or symlink cannot turn a prior path check into a different
        input.  The yielded ``Path`` is the logical, normalized source name.
        """

        logical = self._resolve_read_lexical(value)
        self._require_root(logical, self.read_roots)
        with SecureReadRoot.open(logical, approved_roots=self.read_roots) as reader:
            yield logical, reader

    @contextlib.contextmanager
    def bind_read_root(self, value: str | Path) -> Iterator[BoundReadHandle]:
        """Yield a descriptor-bound reader with logical-path provenance."""

        with self.retain_read_root(value) as (logical, reader):
            yield BoundReadHandle(logical_root=logical, reader=reader)

    def read_authorized_bytes(
        self,
        value: str | Path,
        *,
        maximum: int | None = DEFAULT_MAX_READ_BYTES,
    ) -> tuple[Path, bytes]:
        """Read one approved file through a retained parent descriptor.

        The returned path is provenance only.  Callers must consume the
        returned bytes rather than reopening that path.
        """

        logical = self._resolve_read_lexical(value)
        self._require_root(logical, self.read_roots)
        with SecureReadRoot.open(logical.parent, approved_roots=self.read_roots) as reader:
            return logical, reader.read_bytes(logical.name, max_bytes=maximum)

    def authorized_read_path(self, value: str | Path, *, kind: str = "any") -> Path:
        """Return a logical input name after descriptor-bound type checking."""

        logical = self._resolve_read_lexical(value)
        self._require_root(logical, self.read_roots)
        if kind == "directory":
            with SecureReadRoot.open(logical, approved_roots=self.read_roots):
                return logical
        if kind == "file":
            with SecureReadRoot.open(logical.parent, approved_roots=self.read_roots) as reader:
                reader.stat(logical.name)
            return logical
        if kind != "any":
            raise ValueError("Authorized read kind must be any, file, or directory.")
        try:
            with SecureReadRoot.open(logical, approved_roots=self.read_roots):
                return logical
        except (NotADirectoryError, OSError):
            with SecureReadRoot.open(logical.parent, approved_roots=self.read_roots) as reader:
                reader.stat(logical.name)
            return logical

    def read_authorized_text(
        self,
        value: str | Path,
        *,
        encoding: str = "utf-8",
        maximum: int | None = DEFAULT_MAX_READ_BYTES,
    ) -> tuple[Path, str]:
        logical, payload = self.read_authorized_bytes(value, maximum=maximum)
        try:
            return logical, payload.decode(encoding)
        except UnicodeDecodeError as exc:
            raise ValueError("Authorized input is not valid text for the requested encoding.") from exc

    def authorized_file_record(
        self,
        value: str | Path,
        *,
        maximum: int | None = DEFAULT_MAX_READ_BYTES,
    ) -> tuple[Path, dict[str, int | str]]:
        """Return a bounded descriptor-bound hash/identity record."""

        logical = self._resolve_read_lexical(value)
        self._require_root(logical, self.read_roots)
        with SecureReadRoot.open(logical.parent, approved_roots=self.read_roots) as reader:
            return logical, reader.file_record(logical.name, max_bytes=maximum)

    @contextlib.contextmanager
    def snapshot_read_files(
        self,
        root: str | Path,
        relative_paths: Sequence[str | Path],
        *,
        maximum_file_bytes: int | None = DEFAULT_MAX_READ_BYTES,
        maximum_total_bytes: int | None = DEFAULT_MAX_READ_BYTES,
        optional: bool = True,
    ) -> Iterator[tuple[Path, Path]]:
        """Seal selected approved bytes for a path-only downstream API.

        The source directory remains retained for the whole copy.  Every
        selected file is read through its descriptor and copied into a private
        temporary tree; downstream libraries receive only that sealed bounded
        snapshot.  ``logical_root`` is returned separately so callers can keep
        the original evidence identity in results and manifests.
        """

        logical_root = self._resolve_read_lexical(root)
        self._require_root(logical_root, self.read_roots)
        total_limit = DEFAULT_MAX_READ_BYTES if maximum_total_bytes is None else int(maximum_total_bytes)
        if total_limit < 0:
            raise ValueError("Authorized input snapshot total limit must be non-negative.")
        total = 0
        with self.retain_read_root(logical_root) as (_logical, reader):
            with tempfile.TemporaryDirectory(prefix="oel-mcp-input-") as temporary:
                snapshot = Path(temporary)
                for raw_relative in relative_paths:
                    relative = _safe_relative_input(raw_relative)
                    try:
                        metadata = reader.stat(relative)
                    except (FileNotFoundError, NotADirectoryError):
                        if optional:
                            continue
                        raise
                    size = int(metadata.st_size)
                    if maximum_file_bytes is not None and size > int(maximum_file_bytes):
                        raise ValueError("Authorized input exceeds the snapshot file-size budget.")
                    total += size
                    if total > total_limit:
                        raise ValueError("Authorized input exceeds the snapshot aggregate size budget.")
                    payload = reader.read_bytes(relative, max_bytes=maximum_file_bytes)
                    target = snapshot / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(payload)
                    os.chmod(target, 0o600)
                    os.utime(target, ns=(int(metadata.st_atime_ns), int(metadata.st_mtime_ns)))
                yield logical_root, snapshot

    @contextlib.contextmanager
    def snapshot_read_bytes(
        self,
        value: str | Path,
        payload: bytes,
        *,
        maximum: int | None = DEFAULT_MAX_READ_BYTES,
    ) -> Iterator[tuple[Path, Path]]:
        """Seal already-authorized bytes for a path-only downstream API.

        This is the handoff for callers that have already consumed a file
        through :meth:`read_authorized_bytes`.  It deliberately does not
        reopen the visible source path, so a replacement between authorization
        and a legacy parser cannot change the bytes being consumed.
        """

        logical = self._resolve_read_lexical(value)
        self._require_root(logical, self.read_roots)
        data = bytes(payload)
        if maximum is not None and len(data) > int(maximum):
            raise ValueError("Authorized input exceeds the snapshot file-size budget.")
        with tempfile.TemporaryDirectory(prefix="oel-mcp-input-") as temporary:
            snapshot = Path(temporary)
            target = snapshot / logical.name
            target.write_bytes(data)
            os.chmod(target, 0o600)
            yield logical, snapshot

    def resolve_read_child(
        self,
        root: str | Path,
        *parts: str,
        kind: str = "any",
        required: bool = True,
    ) -> Path:
        """Resolve a derived input while keeping it inside its authorized parent."""

        parent = self.resolve_read(root, kind="directory")
        path = parent.joinpath(*parts).resolve()
        self._require_root(path, (parent,))
        self._require_root(path, self.read_roots)
        if required:
            if kind == "file" and not path.is_file():
                raise FileNotFoundError("Required authorized input file was not found.")
            if kind == "directory" and not path.is_dir():
                raise FileNotFoundError("Required authorized input directory was not found.")
            if kind == "any" and not path.exists():
                raise FileNotFoundError("Required authorized input path was not found.")
        elif path.exists():
            if kind == "file" and not path.is_file():
                raise FileNotFoundError("Authorized optional input is not a file.")
            if kind == "directory" and not path.is_dir():
                raise FileNotFoundError("Authorized optional input is not a directory.")
        return path

    def resolve_write(self, value: str | Path) -> Path:
        # Keep the write path lexical.  ``Path.resolve`` follows an existing
        # symlink and would turn a path precheck into authorization for the
        # symlink target; the descriptor-bound writer performs the final
        # no-follow traversal.
        path = self._resolve_write_lexical(value)
        self._require_root(path, self.write_roots)
        cursor = path
        while True:
            if cursor.is_symlink():
                raise PermissionError("Path is not authorized for this operation.")
            if cursor == cursor.parent:
                break
            cursor = cursor.parent
        return path

    def open_secure_writer(
        self,
        value: str | Path,
        *,
        create: bool = True,
        require_empty: bool = False,
        active_opid: str | None = None,
    ):
        """Open an output directory through the descriptor-bound MCP writer."""

        from sim.security.output_writes import SecureOutputRoot

        path = self.resolve_write(value)
        return SecureOutputRoot.open(
            path,
            approved_roots=self.write_roots,
            create=create,
            require_empty=require_empty,
            active_opid=active_opid,
        )

    @staticmethod
    def _resolve(value: str | Path) -> Path:
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = ROOT / path
        return path.resolve()

    @staticmethod
    def _resolve_read_lexical(value: str | Path) -> Path:
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = ROOT / path
        return canonicalize_platform_compatibility_root(Path(os.path.abspath(path)))

    @staticmethod
    def _resolve_write_lexical(value: str | Path) -> Path:
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = ROOT / path
        # ``abspath`` normalizes ``.``/``..`` without following symlinks.
        # On macOS, canonicalize only the immutable /tmp and /var aliases so
        # that a configured root such as /tmp/oel-output matches the resolved
        # /private/tmp/oel-output root.  Descendant components remain lexical;
        # the loop in ``resolve_write`` and the descriptor-bound writer still
        # reject symlink traversal.
        return canonicalize_platform_compatibility_root(Path(os.path.abspath(path)))

    @staticmethod
    def _require_root(path: Path, roots: tuple[Path, ...]) -> None:
        if not any(_is_relative_to(path, root) for root in roots):
            raise annotate_error(PermissionError("Path is not authorized for this operation."), "path.not_authorized")


def validate_handling(profile: str, handling: dict[str, Any] | None) -> dict[str, str]:
    if profile not in DEPLOYMENT_PROFILES:
        raise PermissionError("Deployment profile is not authorized.")
    if not isinstance(handling, dict):
        raise PermissionError("Handling metadata is required for this operation.")
    marking = str(handling.get("marking", "")).strip()
    release_scope = str(handling.get("release_scope", "")).strip()
    owner = str(handling.get("owner", "")).strip()
    if not marking or release_scope not in {"public", "local_only", "frontier_eligible"}:
        raise PermissionError("Handling metadata is missing or conflicting; review is required.")
    if profile == "direct_frontier_restricted" and release_scope not in {"public", "frontier_eligible"}:
        raise PermissionError("Data is not eligible for the active frontier deployment profile.")
    return {"marking": marking, "release_scope": release_scope, "owner": owner}


def _roots_from_environment(name: str) -> tuple[Path, ...]:
    value = os.environ.get(name, "").strip()
    return tuple(Path(item) for item in value.split(os.pathsep) if item.strip())


def _legacy_roots() -> tuple[Path, ...]:
    return _roots_from_environment("OEL_MCP_ALLOWED_ROOTS")


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _safe_relative_input(value: str | Path) -> Path:
    relative = Path(value)
    if relative.is_absolute() or not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
        raise ValueError("Authorized input snapshot paths must be normalized and relative.")
    return relative
