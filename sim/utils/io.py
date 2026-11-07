from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
import stat
from functools import lru_cache
from importlib import import_module
from json import dumps
from json.encoder import encode_basestring_ascii
from numbers import Integral, Real
from pathlib import Path
from typing import Any

from sim.platform_compat import canonicalize_platform_compatibility_root as _canonicalize_platform_compatibility_root
from sim.security.output_writes import (
    SecureWriteConflict,
    current_output_writer,
    update_output_text,
    write_output_text,
)

_JSON_SCALAR_TYPES = frozenset((str, int, float, bool, type(None)))
_JSON_LEAF_BATCH_LIMIT = 2048


class SafeReadError(OSError):
    """Raised when a bounded no-follow read cannot be completed safely."""


def read_regular_file_nofollow(
    path: str | Path,
    *,
    max_bytes: int,
    min_bytes: int = 0,
) -> bytes:
    """Read one stable regular file without following path-component symlinks."""

    maximum = int(max_bytes)
    minimum = int(min_bytes)
    if maximum < 0 or minimum < 0 or minimum > maximum:
        raise ValueError("read bounds must satisfy 0 <= min_bytes <= max_bytes.")
    lexical = Path(os.path.abspath(os.fspath(Path(path).expanduser())))
    lexical = _canonicalize_platform_compatibility_root(lexical)
    try:
        if os.name == "posix" and os.open in os.supports_dir_fd:
            return _read_posix_nofollow(lexical, minimum=minimum, maximum=maximum)
        return _read_portable_nofollow(lexical, minimum=minimum, maximum=maximum)
    except SafeReadError:
        raise
    except OSError as exc:
        raise SafeReadError(f"Could not safely read regular file {lexical}: {exc}") from exc


def sha256_regular_file_nofollow(
    path: str | Path,
    *,
    max_bytes: int,
    min_bytes: int = 0,
) -> tuple[str, int]:
    """Stream one stable regular file and return its SHA-256 and byte count.

    This is the bounded, memory-efficient companion to
    :func:`read_regular_file_nofollow`. Path components and the final file are
    opened without following symlinks where the platform supports descriptor-
    relative ``O_NOFOLLOW`` traversal.
    """

    maximum = int(max_bytes)
    minimum = int(min_bytes)
    if maximum < 0 or minimum < 0 or minimum > maximum:
        raise ValueError("hash bounds must satisfy 0 <= min_bytes <= max_bytes.")
    lexical = Path(os.path.abspath(os.fspath(Path(path).expanduser())))
    lexical = _canonicalize_platform_compatibility_root(lexical)
    try:
        if os.name == "posix" and os.open in os.supports_dir_fd:
            return _hash_posix_nofollow(lexical, minimum=minimum, maximum=maximum)
        return _hash_portable_nofollow(lexical, minimum=minimum, maximum=maximum)
    except SafeReadError:
        raise
    except OSError as exc:
        raise SafeReadError(f"Could not safely hash regular file {lexical}: {exc}") from exc


def _read_posix_nofollow(path: Path, *, minimum: int, maximum: int) -> bytes:
    parts = path.parts
    if not parts or parts[0] != os.sep or len(parts) < 2:
        raise SafeReadError(f"Expected an absolute file path: {path}")
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    descriptors: list[int] = []
    try:
        current = os.open(os.sep, directory_flags)
        descriptors.append(current)
        for component in parts[1:-1]:
            if component in {"", ".", ".."}:
                raise SafeReadError(f"Unsafe path component in {path}")
            current = os.open(component, directory_flags | nofollow, dir_fd=current)
            descriptors.append(current)
        name = parts[-1]
        if name in {"", ".", ".."}:
            raise SafeReadError(f"Unsafe file name in {path}")
        descriptor = os.open(name, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | nofollow, dir_fd=current)
        descriptors.append(descriptor)
        return _read_stable_descriptor(descriptor, path=path, minimum=minimum, maximum=maximum)
    finally:
        for descriptor in reversed(descriptors):
            try:
                os.close(descriptor)
            except OSError:
                pass


def _read_portable_nofollow(path: Path, *, minimum: int, maximum: int) -> bytes:
    current = Path(path.anchor)
    for component in path.parts[1:]:
        current = current / component
        if current.is_symlink():
            raise SafeReadError(f"Symbolic links are not permitted in evidence paths: {current}")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
    try:
        return _read_stable_descriptor(descriptor, path=path, minimum=minimum, maximum=maximum)
    finally:
        os.close(descriptor)


def _hash_posix_nofollow(path: Path, *, minimum: int, maximum: int) -> tuple[str, int]:
    parts = path.parts
    if not parts or parts[0] != os.sep or len(parts) < 2:
        raise SafeReadError(f"Expected an absolute file path: {path}")
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    descriptors: list[int] = []
    try:
        current = os.open(os.sep, directory_flags)
        descriptors.append(current)
        for component in parts[1:-1]:
            if component in {"", ".", ".."}:
                raise SafeReadError(f"Unsafe path component in {path}")
            current = os.open(component, directory_flags | nofollow, dir_fd=current)
            descriptors.append(current)
        name = parts[-1]
        if name in {"", ".", ".."}:
            raise SafeReadError(f"Unsafe file name in {path}")
        descriptor = os.open(name, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | nofollow, dir_fd=current)
        descriptors.append(descriptor)
        return _hash_stable_descriptor(descriptor, path=path, minimum=minimum, maximum=maximum)
    finally:
        for descriptor in reversed(descriptors):
            try:
                os.close(descriptor)
            except OSError:
                pass


def _hash_portable_nofollow(path: Path, *, minimum: int, maximum: int) -> tuple[str, int]:
    current = Path(path.anchor)
    for component in path.parts[1:]:
        current = current / component
        if current.is_symlink():
            raise SafeReadError(f"Symbolic links are not permitted in evidence paths: {current}")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
    try:
        return _hash_stable_descriptor(descriptor, path=path, minimum=minimum, maximum=maximum)
    finally:
        os.close(descriptor)


def _hash_stable_descriptor(
    descriptor: int,
    *,
    path: Path,
    minimum: int,
    maximum: int,
) -> tuple[str, int]:
    before = os.fstat(descriptor)
    if not stat.S_ISREG(before.st_mode):
        raise SafeReadError(f"Evidence input must be a regular file: {path}")
    if before.st_size < minimum or before.st_size > maximum:
        raise SafeReadError(f"Evidence input must contain between {minimum} and {maximum} bytes: {path}")
    digest = hashlib.sha256()
    total = 0
    while True:
        chunk = os.read(descriptor, 1024 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if total > maximum:
            raise SafeReadError(f"Evidence input exceeds {maximum} bytes: {path}")
        digest.update(chunk)
    after = os.fstat(descriptor)
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
    if identity_after != identity_before or total != before.st_size:
        raise SafeReadError(f"Evidence input changed while it was being hashed: {path}")
    if total < minimum or total > maximum:
        raise SafeReadError(f"Evidence input must contain between {minimum} and {maximum} bytes: {path}")
    return digest.hexdigest(), total


def _read_stable_descriptor(descriptor: int, *, path: Path, minimum: int, maximum: int) -> bytes:
    before = os.fstat(descriptor)
    if not stat.S_ISREG(before.st_mode):
        raise SafeReadError(f"Evidence input must be a regular file: {path}")
    if before.st_size < minimum or before.st_size > maximum:
        raise SafeReadError(f"Evidence input must contain between {minimum} and {maximum} bytes: {path}")
    chunks: list[bytes] = []
    remaining = maximum + 1
    while remaining > 0:
        chunk = os.read(descriptor, min(1024 * 1024, remaining))
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    content = b"".join(chunks)
    after = os.fstat(descriptor)
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
    if identity_after != identity_before or len(content) != before.st_size:
        raise SafeReadError(f"Evidence input changed while it was being read: {path}")
    if len(content) < minimum or len(content) > maximum:
        raise SafeReadError(f"Evidence input must contain between {minimum} and {maximum} bytes: {path}")
    return content


def canonical_evidence_artifact_bytes(
    path: str | Path,
    *,
    root: str | Path,
    max_bytes: int,
) -> bytes:
    """Return deterministic comparison bytes for a retained evidence artifact.

    Ordinary artifacts retain exact byte semantics. JSON artifacts normalize
    only absolute references to their own evidence root, and SQLite artifacts
    are reduced to ordered schema/table content so harmless database-page
    layout differences do not defeat authoritative replay.
    """

    source = Path(path)
    evidence_root = Path(root).absolute()
    artifact_paths = _unique_artifact_reference_suffixes(
        tuple(item.relative_to(evidence_root).as_posix() for item in evidence_root.rglob("*") if item.is_file())
    )
    content = read_regular_file_nofollow(source, max_bytes=max_bytes)
    if source.suffix.lower() == ".sqlite":
        return _canonical_sqlite_bytes(content, evidence_root, artifact_paths)
    if source.suffix.lower() == ".json":
        try:
            value = json.loads(
                content.decode("utf-8"),
                object_pairs_hook=_reject_duplicate_json_fields,
                parse_constant=_reject_nonfinite_json_constant,
            )
        except (UnicodeDecodeError, ValueError):
            return content
        normalized = _normalize_evidence_root(value, evidence_root, artifact_paths)
        normalized = _normalize_nested_manifest_receipts(normalized)
        return json.dumps(normalized, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return content


def _canonical_sqlite_bytes(content: bytes, root: Path, artifact_paths: tuple[str, ...]) -> bytes:
    connection = sqlite3.connect(":memory:")
    try:
        connection.deserialize(content)
        schema_rows = connection.execute(
            "SELECT type, name, tbl_name, sql FROM sqlite_master "
            "WHERE name NOT LIKE 'sqlite_%' ORDER BY type, name"
        ).fetchall()
        tables: dict[str, Any] = {}
        for kind, name, _table_name, _sql in schema_rows:
            if kind != "table":
                continue
            quoted = str(name).replace('"', '""')
            rows = connection.execute(f'SELECT * FROM "{quoted}"').fetchall()
            normalized_rows = [_normalize_evidence_root(list(row), root, artifact_paths) for row in rows]
            normalized_rows.sort(
                key=lambda row: json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False)
            )
            tables[str(name)] = normalized_rows
        payload = {
            "schema": _normalize_evidence_root([list(row) for row in schema_rows], root, artifact_paths),
            "tables": tables,
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (sqlite3.DatabaseError, ValueError, TypeError) as exc:
        raise SafeReadError(f"Could not canonicalize SQLite evidence artifact: {exc}") from exc
    finally:
        connection.close()


def _unique_artifact_reference_suffixes(artifact_paths: tuple[str, ...]) -> tuple[str, ...]:
    counts: dict[str, int] = {}
    for relative in artifact_paths:
        parts = relative.split("/")
        for index in range(len(parts)):
            suffix = "/".join(parts[index:])
            counts[suffix] = counts.get(suffix, 0) + 1
    return tuple(sorted((suffix for suffix, count in counts.items() if count == 1), key=len, reverse=True))


def _reject_duplicate_json_fields(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"Duplicate JSON field: {key}")
        value[key] = item
    return value


def _reject_nonfinite_json_constant(value: str) -> Any:
    raise ValueError(f"Non-finite JSON constant: {value}")


def _normalize_nested_manifest_receipts(value: Any) -> Any:
    """Remove unstable byte-layout receipts from an already-expanded nested manifest.

    The enclosing manifest still binds the retained nested manifest byte for
    byte, while canonical replay independently compares every artifact named
    by it.  Reducing a nested receipt to its path prevents SQLite page layout
    and atomic-build path text from making an otherwise identical replay fail.
    """

    if not isinstance(value, dict) or not str(value.get("schema_version", "")).endswith("manifest.v1"):
        return value
    files = value.get("files")
    if not isinstance(files, list) or not all(
        isinstance(item, dict) and set(item) == {"path", "bytes", "sha256"} for item in files
    ):
        return value
    return {**value, "files": [{"path": item["path"]} for item in files]}


def _normalize_evidence_root(value: Any, root: Path, artifact_paths: tuple[str, ...]) -> Any:
    if isinstance(value, str):
        prefix = str(root)
        atomic_prefix = str(root.parent / f".{root.name}.building-")
        is_owned_path = value == prefix or value.startswith(prefix + os.sep) or value.startswith(atomic_prefix)
        if os.path.isabs(value) and is_owned_path:
            for relative in artifact_paths:
                suffix = os.sep + relative.replace("/", os.sep)
                if value.endswith(suffix):
                    return "$EVIDENCE_ROOT/" + relative
        if value == prefix or value.startswith(prefix + os.sep):
            return "$EVIDENCE_ROOT" + value[len(prefix) :]
        return value
    if isinstance(value, dict):
        return {
            str(key): _normalize_evidence_root(item, root, artifact_paths)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_normalize_evidence_root(item, root, artifact_paths) for item in value]
    if isinstance(value, bytes):
        return {"$bytes_hex": value.hex()}
    return value


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    item = getattr(value, "item", None)
    if callable(item):
        try:
            scalar = item()
        except Exception:
            scalar = value
        if scalar is not value:
            return json_safe(scalar)
    return value


def write_json(path: str, payload: dict[str, Any]) -> None:
    if current_output_writer() is not None:
        encoder = _native_json_encoder()
        chunks = _iter_json_safe(payload, indent=2) if encoder is None else encoder(payload, _native_json_fallback)
        serialized = "".join(chunks)
        try:
            write_output_text(path, serialized)
        except SecureWriteConflict:
            update_output_text(path, serialized)
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    try:
        with tmp.open("w", encoding="utf-8") as handle:
            # Preserve the historical json_safe + json.dump byte contract while
            # sanitizing values as the encoder visits them. Large run logs no
            # longer require a second full-size Python list/dict tree.
            encoder = _native_json_encoder()
            chunks = _iter_json_safe(payload, indent=2) if encoder is None else encoder(payload, _native_json_fallback)
            for chunk in chunks:
                handle.write(chunk)
        tmp.replace(out)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


@lru_cache(maxsize=1)
def _native_json_encoder():
    """Use the optional text encoder without requiring a numerical backend."""
    try:
        return getattr(import_module("oel_rust_orbit"), "json_safe_chunks", None)
    except ImportError:
        return None


def _native_json_fallback(value: Any, level: int):
    # The fallback owns uncommon Python types and established coercion/errors.
    # Rebase only actual formatting newlines; JSON string newlines are escaped.
    prefix = "  " * level
    for chunk in _iter_json_safe(value, indent=2):
        yield chunk.replace("\n", "\n" + prefix) if level else chunk


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Hash a file without allocating a full-size bytes object."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(int(chunk_size)), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _iter_json_safe(value: Any, *, indent: int | str | None = None):
    """Yield the exact encoding of ``json.dump(json_safe(value))`` lazily."""

    indent_text = None if indent is None else indent if isinstance(indent, str) else " " * int(indent)
    markers: dict[int, Any] = {}

    def scalar(item: Any) -> Any:
        if item is None or isinstance(item, (str, bool, int, float, list, tuple, dict)):
            return item
        if isinstance(item, Path):
            return str(item)
        if isinstance(item, Integral):
            return int(item)
        if isinstance(item, Real):
            numeric = float(item)
            return numeric if math.isfinite(numeric) else None
        convert = getattr(item, "item", None)
        if callable(convert):
            try:
                converted = convert()
            except Exception:
                converted = item
            if converted is not item:
                return scalar(converted)
        return item

    def float_text(item: float) -> str:
        return float.__repr__(item) if math.isfinite(item) else "null"

    def encode(item: Any, level: int):
        item = scalar(item)
        if isinstance(item, str):
            yield encode_basestring_ascii(item)
        elif item is None:
            yield "null"
        elif item is True:
            yield "true"
        elif item is False:
            yield "false"
        elif isinstance(item, int):
            yield int.__repr__(item)
        elif isinstance(item, float):
            yield float_text(item)
        elif isinstance(item, (list, tuple)):
            if 0 < len(item) <= _JSON_LEAF_BATCH_LIMIT and all(type(child) in _JSON_SCALAR_TYPES for child in item):
                # Numeric histories and typed packet vectors have many short
                # primitive rows. Encode each bounded row in the C JSON
                # encoder instead of recursively yielding/writing each cell.
                clean = [
                    None if type(child) is float and not math.isfinite(child) else child
                    for child in item
                ]
                if indent_text is None:
                    yield dumps(clean, separators=(",", ":"), allow_nan=False)
                else:
                    prefix = "\n" + indent_text * (level + 1)
                    encoded = dumps(clean, separators=("," + prefix, ": "), allow_nan=False)
                    yield "[" + prefix + encoded[1:-1] + "\n" + indent_text * level + "]"
                return
            marker = id(item)
            if marker in markers:
                raise ValueError("Circular reference detected")
            markers[marker] = item
            try:
                if not item:
                    yield "[]"
                    return
                yield "["
                child_level = level + 1
                separator = "," if indent_text is None else ",\n" + indent_text * child_level
                if indent_text is not None:
                    yield "\n" + indent_text * child_level
                for index, child in enumerate(item):
                    if index:
                        yield separator
                    yield from encode(child, child_level)
                if indent_text is not None:
                    yield "\n" + indent_text * level
                yield "]"
            finally:
                markers.pop(marker, None)
        elif isinstance(item, dict):
            marker = id(item)
            if marker in markers:
                raise ValueError("Circular reference detected")
            markers[marker] = item
            try:
                # json_safe historically constructs a new dict with string
                # keys. Preserve its last-value-wins collision behavior while
                # retaining the first key's insertion position.
                normalized_items: dict[str, Any] = {}
                for key, child in item.items():
                    normalized_items[str(key)] = child
                if not normalized_items:
                    yield "{}"
                    return
                if len(normalized_items) <= _JSON_LEAF_BATCH_LIMIT and all(
                    type(child) in _JSON_SCALAR_TYPES for child in normalized_items.values()
                ):
                    clean = {
                        key: None if type(child) is float and not math.isfinite(child) else child
                        for key, child in normalized_items.items()
                    }
                    if indent_text is None:
                        yield dumps(clean, separators=(",", ":"), allow_nan=False)
                    else:
                        prefix = "\n" + indent_text * (level + 1)
                        encoded = dumps(clean, separators=("," + prefix, ": "), allow_nan=False)
                        yield "{" + prefix + encoded[1:-1] + "\n" + indent_text * level + "}"
                    return
                yield "{"
                child_level = level + 1
                separator = "," if indent_text is None else ",\n" + indent_text * child_level
                if indent_text is not None:
                    yield "\n" + indent_text * child_level
                for index, (key, child) in enumerate(normalized_items.items()):
                    if index:
                        yield separator
                    yield encode_basestring_ascii(key)
                    yield ": " if indent_text is not None else ":"
                    yield from encode(child, child_level)
                if indent_text is not None:
                    yield "\n" + indent_text * level
                yield "}"
            finally:
                markers.pop(marker, None)
        else:
            raise TypeError(f"Object of type {item.__class__.__name__} is not JSON serializable")

    yield from encode(value, 0)
