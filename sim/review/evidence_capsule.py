"""Content-bound compression and temporary hydration for derived evidence files."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import shutil
import sqlite3
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence

from sim.platform_compat import canonicalize_platform_compatibility_root as _canonicalize_platform_compatibility_root
from sim.security.input_reads import (
    DEFAULT_MAX_READ_BYTES,
    SecureReadError,
    SecureReadRoot,
    sqlite_snapshot_bytes,
)

EVIDENCE_CAPSULE_SCHEMA = "oel.evidence_capsule.v1"
EVIDENCE_CAPSULE_MANIFEST = "evidence_capsule.json"
_CHUNK_BYTES = 1024 * 1024
MAX_REVIEW_HYDRATION_BYTES = 1_000_000_000
MAX_CAPSULE_VERIFICATION_QUERIES = 32
MAX_CAPSULE_SOURCE_BYTES = 2 * 1024**3


class EvidenceCapsuleError(ValueError):
    """Raised when compressed evidence cannot be trusted or hydrated."""


def _sha256(path: Path) -> str:
    reader, name = _open_reader_for_file(path)
    try:
        return reader.sha256(name, max_bytes=MAX_CAPSULE_SOURCE_BYTES)
    finally:
        reader.close()


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_sqlite(path: Path) -> bool:
    return path.name.endswith(".sqlite")


def _lexical_path(value: str | Path, *, base: Path | None = None) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (base or Path.cwd()) / path
    if any(part in {"", ".", ".."} for part in path.parts[1:]):
        raise EvidenceCapsuleError(f"Evidence path contains traversal components: {path}")
    return _canonicalize_platform_compatibility_root(Path(os.path.abspath(os.fspath(path))))


def _open_reader_for_file(path: str | Path) -> tuple[SecureReadRoot, str]:
    """Open a file's containing directory with an explicit local boundary."""

    candidate = _lexical_path(path)
    try:
        reader = SecureReadRoot.open(
            candidate.parent,
            approved_roots=(candidate.parent.parent,),
        )
    except (OSError, SecureReadError) as exc:
        raise EvidenceCapsuleError(f"Evidence path could not be opened securely: {candidate}") from exc
    return reader, candidate.name


def _read_manifest_bytes(reader: SecureReadRoot, name: str) -> dict[str, Any]:
    try:
        payload = json.loads(reader.read_text(name, max_bytes=DEFAULT_MAX_READ_BYTES))
    except (OSError, SecureReadError, json.JSONDecodeError) as exc:
        raise EvidenceCapsuleError("Evidence capsule manifest is unreadable.") from exc
    if not isinstance(payload, dict):
        raise EvidenceCapsuleError("Evidence capsule manifest must contain a JSON object.")
    return payload


def _reject_symlink_components(path: Path, *, label: str) -> None:
    path = _canonicalize_platform_compatibility_root(path)
    if any(component.is_symlink() for component in (path, *path.parents)):
        raise EvidenceCapsuleError(f"{label} contains a symbolic link: {path}")


def _safe_logical_path(value: str | Path) -> Path:
    path = _lexical_path(value)
    _reject_symlink_components(path, label="Evidence path")
    # Keep the lexical path.  SecureReadRoot reopens its containing directory
    # with no-follow components and anchors subsequent reads to that handle;
    # resolving here would reintroduce a check-then-use pathname lookup.
    return path


def _verification_query_rows(connection: sqlite3.Connection, query: str) -> list[list[Any]]:
    # Hydration consumes manifest-supplied SQL, so enforce the same read-only
    # boundary as ordinary review queries (query_only alone allows ATTACH).
    from sim.review.workspace import (
        _read_only_authorizer,
        _validate_select_sql,
        execute_bounded_review_query,
    )

    connection.set_authorizer(_read_only_authorizer)
    try:
        statement = _validate_select_sql(query)
        rows = execute_bounded_review_query(connection, statement)
        return [list(row) for row in rows]
    except (ValueError, sqlite3.Error) as exc:
        raise EvidenceCapsuleError(f"Invalid read-only evidence verification query: {exc}") from exc


def _sqlite_verification_connection(
    connection: sqlite3.Connection,
    path: Path,
) -> dict[str, Any]:
    try:
        quick_check = connection.execute("PRAGMA quick_check").fetchone()
        if quick_check != ("ok",):
            raise EvidenceCapsuleError(f"SQLite quick_check failed for {path}: {quick_check!r}")
        tables = [
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_schema "
                "WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
            )
        ]
        row_counts = {
            table: int(connection.execute(f'SELECT COUNT(*) FROM "{table.replace(chr(34), chr(34) * 2)}"').fetchone()[0])
            for table in tables
        }
        run_metadata: dict[str, Any] = {}
        if "run_metadata" in tables:
            columns = {
                str(row[1]) for row in connection.execute("PRAGMA table_info(run_metadata)").fetchall()
            }
            selected = [
                name
                for name in ("oel_version", "review_schema_version", "config_source_path", "config_sha256")
                if name in columns
            ]
            if selected:
                row = connection.execute(f"SELECT {', '.join(selected)} FROM run_metadata LIMIT 1").fetchone()
                if row is not None:
                    run_metadata = dict(zip(selected, row, strict=True))
        return {"quick_check": "ok", "row_counts": row_counts, "run_metadata": run_metadata}
    except EvidenceCapsuleError:
        raise
    except sqlite3.DatabaseError as exc:
        raise EvidenceCapsuleError(f"SQLite verification failed for {path}: {exc}") from exc


def _legacy_sha256(path: Path) -> str:
    """Hash a trusted local compatibility path without POSIX descriptor APIs."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _legacy_load_manifest(logical_path: Path) -> tuple[Path, dict[str, Any]]:
    logical_path = _safe_logical_path(logical_path)
    manifest_path = _manifest_path(logical_path)
    _reject_symlink_components(manifest_path, label="Evidence capsule manifest")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvidenceCapsuleError(f"Evidence capsule manifest is unreadable: {manifest_path}") from exc
    if not isinstance(payload, dict) or payload.get("schema") != EVIDENCE_CAPSULE_SCHEMA:
        raise EvidenceCapsuleError(f"Evidence capsule manifest has an unsupported schema: {manifest_path}")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, list):
        raise EvidenceCapsuleError(f"Evidence capsule manifest has no artifact list: {manifest_path}")
    matches = [
        item for item in artifacts
        if isinstance(item, dict) and item.get("logical_path") == logical_path.name
    ]
    if len(matches) != 1:
        raise EvidenceCapsuleError(
            f"Evidence capsule manifest must contain exactly one entry for {logical_path.name}: {manifest_path}"
        )
    entry = dict(matches[0])
    compressed_name = str(entry.get("compressed_path", "") or "")
    candidate = Path(compressed_name)
    if candidate.is_absolute() or len(candidate.parts) != 1 or candidate.name != compressed_name:
        raise EvidenceCapsuleError(f"Unsafe compressed evidence path in {manifest_path}: {compressed_name!r}")
    return manifest_path, entry


def _legacy_compressed_entry(
    logical_path: Path,
    *,
    verify_compressed: bool,
) -> tuple[Path, dict[str, Any]]:
    logical_path = _safe_logical_path(logical_path)
    _manifest, entry = _legacy_load_manifest(logical_path)
    compressed = logical_path.parent / str(entry["compressed_path"])
    _reject_symlink_components(compressed, label="Compressed evidence")
    if not compressed.is_file() or compressed.is_symlink():
        raise EvidenceCapsuleError(f"Compressed evidence is missing or unsafe: {compressed}")
    if int(entry.get("compressed_bytes", -1)) != compressed.stat().st_size:
        raise EvidenceCapsuleError(f"Compressed evidence size does not match its manifest: {compressed}")
    if verify_compressed and _legacy_sha256(compressed) != str(entry.get("compressed_sha256", "")):
        raise EvidenceCapsuleError(f"Compressed evidence digest does not match its manifest: {compressed}")
    return compressed, entry


def _legacy_sqlite_verification(path: Path) -> dict[str, Any]:
    """Verify a trusted local compatibility SQLite path."""

    try:
        uri = f"{path.resolve().as_uri()}?mode=ro"
        with sqlite3.connect(uri, uri=True) as connection:
            connection.execute("PRAGMA query_only = ON")
            return _sqlite_verification_connection(connection, path)
    except EvidenceCapsuleError:
        raise
    except sqlite3.DatabaseError as exc:
        raise EvidenceCapsuleError(f"SQLite verification failed for {path}: {exc}") from exc


@contextmanager
def _protected_sqlite_connection(
    reader: SecureReadRoot,
    name: str,
    path: Path,
) -> Iterator[sqlite3.Connection]:
    """Yield a bounded in-memory SQLite snapshot read from a retained root."""

    try:
        payload = sqlite_snapshot_bytes(
            reader,
            name,
            max_bytes=MAX_CAPSULE_SOURCE_BYTES,
        )
        from sim.security.input_reads import sqlite_connection_from_snapshot

        connection = sqlite_connection_from_snapshot(payload)
        try:
            connection.execute("PRAGMA query_only = ON")
            yield connection
        finally:
            connection.close()
    except (SecureReadError, sqlite3.DatabaseError) as exc:
        raise EvidenceCapsuleError(
            f"Protected SQLite snapshot could not be opened securely: {path}"
        ) from exc


def _sqlite_verification_from_reader(
    reader: SecureReadRoot,
    name: str,
    path: Path,
) -> dict[str, Any]:
    with _protected_sqlite_connection(reader, name, path) as connection:
        return _sqlite_verification_connection(connection, path)


def _manifest_path(logical_path: Path) -> Path:
    return logical_path.parent / EVIDENCE_CAPSULE_MANIFEST


def compressed_evidence_path(logical_path: str | Path) -> Path:
    path = Path(logical_path)
    return path.with_name(f"{path.name}.gz")


def _load_manifest(
    logical_path: Path,
    *,
    reader: SecureReadRoot | None = None,
) -> tuple[Path, dict[str, Any]]:
    logical_path = _safe_logical_path(logical_path)
    manifest_path = _manifest_path(logical_path)
    _reject_symlink_components(manifest_path, label="Evidence capsule manifest")
    owned_reader = reader is None
    if reader is None:
        reader, _manifest_name = _open_reader_for_file(manifest_path)
    try:
        payload = _read_manifest_bytes(reader, EVIDENCE_CAPSULE_MANIFEST)
        if payload.get("schema") != EVIDENCE_CAPSULE_SCHEMA:
            raise EvidenceCapsuleError(f"Evidence capsule manifest has an unsupported schema: {manifest_path}")
        artifacts = payload.get("artifacts")
        if not isinstance(artifacts, list):
            raise EvidenceCapsuleError(f"Evidence capsule manifest has no artifact list: {manifest_path}")
        matches = [
            item for item in artifacts
            if isinstance(item, dict) and item.get("logical_path") == logical_path.name
        ]
        if len(matches) != 1:
            raise EvidenceCapsuleError(
                f"Evidence capsule manifest must contain exactly one entry for {logical_path.name}: {manifest_path}"
            )
        entry = dict(matches[0])
        compressed_name = str(entry.get("compressed_path", "") or "")
        candidate = Path(compressed_name)
        if candidate.is_absolute() or len(candidate.parts) != 1 or candidate.name != compressed_name:
            raise EvidenceCapsuleError(f"Unsafe compressed evidence path in {manifest_path}: {compressed_name!r}")
        return manifest_path, entry
    finally:
        if owned_reader:
            reader.close()


def _compressed_entry(
    logical_path: Path,
    *,
    verify_compressed: bool,
    reader: SecureReadRoot | None = None,
) -> tuple[Path, dict[str, Any]]:
    logical_path = _safe_logical_path(logical_path)
    owned_reader = reader is None
    if reader is None:
        reader, _name = _open_reader_for_file(logical_path)
    try:
        _manifest, entry = _load_manifest(logical_path, reader=reader)
        compressed_name = str(entry["compressed_path"])
        compressed = reader.path / compressed_name
        _reject_symlink_components(compressed, label="Compressed evidence")
        metadata = reader.stat(compressed_name)
        if int(metadata.st_size) > MAX_CAPSULE_SOURCE_BYTES:
            raise EvidenceCapsuleError(
                f"Compressed evidence exceeds the {MAX_CAPSULE_SOURCE_BYTES}-byte read budget: {compressed}"
            )
        if int(entry.get("compressed_bytes", -1)) != int(metadata.st_size):
            raise EvidenceCapsuleError(f"Compressed evidence size does not match its manifest: {compressed}")
        if verify_compressed and reader.sha256(
            compressed_name,
            max_bytes=MAX_CAPSULE_SOURCE_BYTES,
        ) != str(entry.get("compressed_sha256", "")):
            raise EvidenceCapsuleError(f"Compressed evidence digest does not match its manifest: {compressed}")
        return compressed, entry
    except (FileNotFoundError, NotADirectoryError, SecureReadError) as exc:
        raise EvidenceCapsuleError(f"Compressed evidence is missing or unsafe: {compressed}") from exc
    finally:
        if owned_reader:
            reader.close()


def _capsule_original_identity(
    logical_path: str | Path,
    *,
    reader: SecureReadRoot,
) -> tuple[str, int, int]:
    """Return a capsule's retained-source content identity.

    The hydrated SQLite file is deliberately temporary, so its filesystem
    mtime is unsuitable for a stable plan or manifest identity.  Read the
    original digest, byte count, and recorded source mtime from the capsule
    manifest through the already-retained source descriptor.  Rechecking the
    compressed entry's descriptor-bound size and digest makes this a source
    identity operation rather than a visible-path reopen.
    """

    path = _safe_logical_path(logical_path)
    _, entry = _compressed_entry(path, verify_compressed=True, reader=reader)
    digest = str(entry.get("original_sha256", "") or "")
    if len(digest) != 64:
        raise EvidenceCapsuleError(f"Evidence capsule has an invalid original digest: {_manifest_path(path)}")
    try:
        bytes.fromhex(digest)
    except ValueError as exc:
        raise EvidenceCapsuleError(f"Evidence capsule has an invalid original digest: {_manifest_path(path)}") from exc
    original_bytes = entry.get("original_bytes")
    original_mtime_ns = entry.get("original_mtime_ns")
    if (
        isinstance(original_bytes, bool)
        or not isinstance(original_bytes, int)
        or original_bytes < 0
        or isinstance(original_mtime_ns, bool)
        or not isinstance(original_mtime_ns, int)
        or original_mtime_ns < 0
    ):
        raise EvidenceCapsuleError(
            f"Evidence capsule has invalid original identity metadata: {_manifest_path(path)}"
        )
    return digest, int(original_bytes), int(original_mtime_ns)


def evidence_file_exists(logical_path: str | Path) -> bool:
    try:
        path = _safe_logical_path(logical_path)
    except EvidenceCapsuleError:
        return False
    try:
        reader, name = _open_reader_for_file(path)
    except (EvidenceCapsuleError, OSError, SecureReadError):
        return False
    try:
        try:
            reader.stat(name)
            return True
        except (FileNotFoundError, NotADirectoryError, SecureReadError):
            _compressed_entry(path, verify_compressed=False, reader=reader)
            return True
    except EvidenceCapsuleError:
        return False
    finally:
        reader.close()


def evidence_file_sha256(logical_path: str | Path) -> str:
    path = _safe_logical_path(logical_path)
    reader, name = _open_reader_for_file(path)
    try:
        try:
            reader.stat(name)
            return reader.sha256(name, max_bytes=MAX_CAPSULE_SOURCE_BYTES)
        except (FileNotFoundError, NotADirectoryError, SecureReadError):
            _compressed, entry = _compressed_entry(path, verify_compressed=True, reader=reader)
    finally:
        reader.close()
    digest = str(entry.get("original_sha256", "") or "")
    if len(digest) != 64:
        raise EvidenceCapsuleError(f"Evidence capsule has an invalid original digest: {_manifest_path(path)}")
    materialized = materialize_evidence(
        path, prefer_capsule=True,
        max_original_bytes=MAX_REVIEW_HYDRATION_BYTES if _is_sqlite(path) else None,
    )
    try:
        if _sha256(materialized.path) != digest:
            raise EvidenceCapsuleError(f"Evidence capsule original digest verification failed: {path}")
    finally:
        materialized.close()
    return digest


def evidence_file_mtime_ns(logical_path: str | Path, *, secure: bool = True) -> int:
    path = _safe_logical_path(logical_path)
    if not secure:
        if path.is_file() and not path.is_symlink():
            return int(path.stat().st_mtime_ns)
        _compressed, entry = _legacy_compressed_entry(path, verify_compressed=True)
        value = entry.get("original_mtime_ns")
        if not isinstance(value, int) or value < 0:
            raise EvidenceCapsuleError(f"Evidence capsule has no valid original mtime: {_manifest_path(path)}")
        return value
    reader, name = _open_reader_for_file(path)
    try:
        try:
            return int(reader.stat(name).st_mtime_ns)
        except (FileNotFoundError, NotADirectoryError, SecureReadError):
            _compressed, entry = _compressed_entry(path, verify_compressed=True, reader=reader)
    finally:
        reader.close()
    value = entry.get("original_mtime_ns")
    if not isinstance(value, int) or value < 0:
        raise EvidenceCapsuleError(f"Evidence capsule has no valid original mtime: {_manifest_path(path)}")
    return value


@dataclass
class MaterializedEvidence:
    logical_path: Path
    path: Path
    _temporary: tempfile.TemporaryDirectory[str] | None = None

    @property
    def hydrated(self) -> bool:
        return self._temporary is not None

    def close(self) -> None:
        if self._temporary is not None:
            self._temporary.cleanup()
            self._temporary = None

    def __enter__(self) -> Path:
        return self.path

    def __exit__(self, *_args: object) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()


def _legacy_materialize_evidence(
    logical_path: str | Path,
    *,
    prefer_capsule: bool,
    max_original_bytes: int | None,
    secure_copy: bool,
) -> MaterializedEvidence:
    """Materialize through trusted pathname APIs for explicit compatibility callers."""

    path = _safe_logical_path(logical_path)
    if path.is_file() and not path.is_symlink() and not prefer_capsule and not secure_copy:
        return MaterializedEvidence(logical_path=path, path=path)
    if path.is_file() and not path.is_symlink() and not prefer_capsule and secure_copy:
        original_bytes = int(path.stat().st_size)
        copy_limit = max_original_bytes if max_original_bytes is not None else MAX_CAPSULE_SOURCE_BYTES
        if original_bytes > copy_limit:
            raise EvidenceCapsuleError(
                f"Evidence exceeds the {int(copy_limit)}-byte size budget: {path}"
            )
        temporary = tempfile.TemporaryDirectory(prefix="oel-evidence-")
        hydrated = Path(temporary.name) / path.name
        try:
            with path.open("rb") as source, hydrated.open("wb") as destination:
                shutil.copyfileobj(source, destination, length=_CHUNK_BYTES)
            if hydrated.stat().st_size != original_bytes:
                raise EvidenceCapsuleError(f"Hydrated evidence size changed during read: {path}")
            return MaterializedEvidence(logical_path=path, path=hydrated, _temporary=temporary)
        except Exception:
            temporary.cleanup()
            raise
    compressed, entry = _legacy_compressed_entry(path, verify_compressed=True)
    try:
        original_bytes = int(entry.get("original_bytes", -1))
    except (TypeError, ValueError) as exc:
        raise EvidenceCapsuleError(f"Evidence capsule has an invalid original size: {path}") from exc
    if original_bytes < 0:
        raise EvidenceCapsuleError(f"Evidence capsule has an invalid original size: {path}")
    if max_original_bytes is not None and original_bytes > int(max_original_bytes):
        raise EvidenceCapsuleError(
            f"Hydrated evidence exceeds the {int(max_original_bytes)}-byte size budget: {path}"
        )
    temporary = tempfile.TemporaryDirectory(prefix="oel-evidence-")
    hydrated = Path(temporary.name) / path.name
    try:
        with gzip.open(compressed, "rb") as source, hydrated.open("wb") as destination:
            written = 0
            while written < original_bytes:
                remaining = original_bytes - written
                chunk = source.read(min(_CHUNK_BYTES, remaining + 1))
                if not chunk:
                    break
                if len(chunk) > remaining:
                    raise EvidenceCapsuleError(
                        f"Hydrated evidence exceeds its declared size before completion: {path}"
                    )
                destination.write(chunk)
                written += len(chunk)
            if written == original_bytes and source.read(1):
                raise EvidenceCapsuleError(
                    f"Hydrated evidence exceeds its declared size before completion: {path}"
                )
        if hydrated.stat().st_size != int(entry.get("original_bytes", -1)):
            raise EvidenceCapsuleError(f"Hydrated evidence size does not match its manifest: {path}")
        if _legacy_sha256(hydrated) != str(entry.get("original_sha256", "")):
            raise EvidenceCapsuleError(f"Hydrated evidence digest does not match its manifest: {path}")
        if str(entry.get("kind")) == "sqlite":
            verification = _legacy_sqlite_verification(hydrated)
            expected = dict(entry.get("verification", {}) or {})
            if isinstance(expected.get("queries"), list):
                if len(expected["queries"]) > MAX_CAPSULE_VERIFICATION_QUERIES:
                    raise EvidenceCapsuleError(
                        "Evidence capsule verification query count exceeds the configured limit."
                    )
                query_results: list[dict[str, Any]] = []
                uri = f"{hydrated.resolve().as_uri()}?mode=ro"
                with sqlite3.connect(uri, uri=True) as connection:
                    connection.execute("PRAGMA query_only = ON")
                    for item in expected["queries"]:
                        if not isinstance(item, dict) or not isinstance(item.get("query"), str):
                            raise EvidenceCapsuleError(
                                f"Invalid verification query in capsule manifest: {path}"
                            )
                        query_results.append(
                            {
                                "query": item["query"],
                                "rows": _verification_query_rows(connection, item["query"]),
                            }
                        )
                verification["queries"] = query_results
            if verification != expected:
                raise EvidenceCapsuleError(f"Hydrated SQLite verification does not match its manifest: {path}")
    except Exception:
        temporary.cleanup()
        raise
    return MaterializedEvidence(logical_path=path, path=hydrated, _temporary=temporary)


def materialize_evidence(
    logical_path: str | Path,
    *,
    prefer_capsule: bool = False,
    max_original_bytes: int | None = None,
    secure_copy: bool = False,
    secure: bool = True,
) -> MaterializedEvidence:
    """Return direct evidence or a bounded capsule hydration.

    ``secure=True`` is the descriptor-bound route used by protected review
    handlers. ``secure=False`` is an explicit trusted local compatibility
    route for hosts without the POSIX descriptor primitives.
    """

    path = _safe_logical_path(logical_path)
    if not secure:
        return _legacy_materialize_evidence(
            path,
            prefer_capsule=prefer_capsule,
            max_original_bytes=max_original_bytes,
            secure_copy=secure_copy,
        )
    reader, name = _open_reader_for_file(path)
    try:
        try:
            original_metadata = reader.stat(name)
            original_present = True
        except (FileNotFoundError, NotADirectoryError, SecureReadError):
            original_present = False
            original_metadata = None
        if original_present and not prefer_capsule:
            if not secure_copy:
                return MaterializedEvidence(logical_path=path, path=reader.path / name)
            original_bytes = int(original_metadata.st_size)
            copy_limit = max_original_bytes if max_original_bytes is not None else MAX_CAPSULE_SOURCE_BYTES
            if original_bytes > int(copy_limit):
                raise EvidenceCapsuleError(
                    f"Evidence exceeds the {int(copy_limit)}-byte size budget: {path}"
                )
            temporary = tempfile.TemporaryDirectory(prefix="oel-evidence-")
            hydrated = Path(temporary.name) / path.name
            try:
                with reader.open_file(name, max_bytes=int(copy_limit)) as source, hydrated.open("wb") as destination:
                    shutil.copyfileobj(source, destination, length=_CHUNK_BYTES)
                if hydrated.stat().st_size != original_bytes:
                    raise EvidenceCapsuleError(f"Hydrated evidence size changed during read: {path}")
                return MaterializedEvidence(logical_path=path, path=hydrated, _temporary=temporary)
            except Exception:
                temporary.cleanup()
                raise

        compressed, entry = _compressed_entry(path, verify_compressed=True, reader=reader)
        try:
            original_bytes = int(entry.get("original_bytes", -1))
        except (TypeError, ValueError) as exc:
            raise EvidenceCapsuleError(f"Evidence capsule has an invalid original size: {path}") from exc
        if original_bytes < 0:
            raise EvidenceCapsuleError(f"Evidence capsule has an invalid original size: {path}")
        if max_original_bytes is not None and original_bytes > int(max_original_bytes):
            raise EvidenceCapsuleError(
                f"Hydrated evidence exceeds the {int(max_original_bytes)}-byte size budget: {path}"
            )
        temporary = tempfile.TemporaryDirectory(prefix="oel-evidence-")
        hydrated = Path(temporary.name) / path.name
        try:
            compressed_limit = max(
                int(entry.get("compressed_bytes", -1)),
                DEFAULT_MAX_READ_BYTES,
            )
            if compressed_limit > MAX_CAPSULE_SOURCE_BYTES:
                raise EvidenceCapsuleError(
                    f"Compressed evidence exceeds the {MAX_CAPSULE_SOURCE_BYTES}-byte read budget: {path}"
                )
            with reader.open_file(compressed.name, max_bytes=compressed_limit) as compressed_stream:
                with gzip.GzipFile(fileobj=compressed_stream, mode="rb") as source, hydrated.open("wb") as destination:
                    written = 0
                    while written < original_bytes:
                        remaining = original_bytes - written
                        chunk = source.read(min(_CHUNK_BYTES, remaining + 1))
                        if not chunk:
                            break
                        if len(chunk) > remaining:
                            raise EvidenceCapsuleError(
                                f"Hydrated evidence exceeds its declared size before completion: {path}"
                            )
                        destination.write(chunk)
                        written += len(chunk)
                    if written == original_bytes and source.read(1):
                        raise EvidenceCapsuleError(
                            f"Hydrated evidence exceeds its declared size before completion: {path}"
                        )
            if hydrated.stat().st_size != int(entry.get("original_bytes", -1)):
                raise EvidenceCapsuleError(f"Hydrated evidence size does not match its manifest: {path}")
            if _sha256(hydrated) != str(entry.get("original_sha256", "")):
                raise EvidenceCapsuleError(f"Hydrated evidence digest does not match its manifest: {path}")
            if str(entry.get("kind")) == "sqlite":
                hydrated_reader, hydrated_name = _open_reader_for_file(hydrated)
                try:
                    verification = _sqlite_verification_from_reader(
                        hydrated_reader,
                        hydrated_name,
                        path,
                    )
                    expected = dict(entry.get("verification", {}) or {})
                    if isinstance(expected.get("queries"), list):
                        if len(expected["queries"]) > MAX_CAPSULE_VERIFICATION_QUERIES:
                            raise EvidenceCapsuleError(
                                "Evidence capsule verification query count exceeds the configured limit."
                            )
                        query_results: list[dict[str, Any]] = []
                        with _protected_sqlite_connection(
                            hydrated_reader,
                            hydrated_name,
                            path,
                        ) as connection:
                            for item in expected["queries"]:
                                if not isinstance(item, dict) or not isinstance(item.get("query"), str):
                                    raise EvidenceCapsuleError(
                                        f"Invalid verification query in capsule manifest: {path}"
                                    )
                                query_results.append(
                                    {
                                        "query": item["query"],
                                        "rows": _verification_query_rows(connection, item["query"]),
                                    }
                                )
                        verification["queries"] = query_results
                finally:
                    hydrated_reader.close()
                if verification != expected:
                    raise EvidenceCapsuleError(f"Hydrated SQLite verification does not match its manifest: {path}")
        except Exception:
            temporary.cleanup()
            raise
        return MaterializedEvidence(logical_path=path, path=hydrated, _temporary=temporary)
    finally:
        reader.close()


@contextmanager
def materialized_evidence_file(logical_path: str | Path) -> Iterator[Path]:
    materialized = materialize_evidence(
        logical_path,
        max_original_bytes=MAX_REVIEW_HYDRATION_BYTES if _is_sqlite(Path(logical_path)) else None,
    )
    try:
        yield materialized.path
    finally:
        materialized.close()


def create_evidence_capsule(
    logical_path: str | Path,
    *,
    remove_original: bool = False,
    compression_level: int = 1,
    verification_queries: Sequence[str] = (),
    provenance_paths: Sequence[str | Path] = (),
) -> dict[str, Any]:
    """Create and verify a gzip capsule; optionally remove the exact source file.

    Source and SQLite verification reads are descriptor-bound. The capsule
    publication transaction remains a local compatibility writer and is not a
    protected MCP handler route.
    """

    source_path = _lexical_path(logical_path)
    try:
        _reject_symlink_components(source_path, label="Evidence source")
    except EvidenceCapsuleError as exc:
        raise EvidenceCapsuleError(f"Evidence source is missing or unsafe: {source_path}") from exc
    path = source_path
    source_reader, source_name = _open_reader_for_file(path)
    source_verification: dict[str, Any] | None = None
    try:
        source_record = source_reader.file_record(
            source_name,
            max_bytes=MAX_CAPSULE_SOURCE_BYTES,
        )
        if _is_sqlite(path):
            source_verification = _sqlite_verification_from_reader(
                source_reader,
                source_name,
                path,
            )
    except (FileNotFoundError, NotADirectoryError, SecureReadError) as exc:
        raise EvidenceCapsuleError(f"Evidence source is missing or unsafe: {path}") from exc
    finally:
        source_reader.close()
    if path.name.endswith(("-wal", "-shm")):
        raise EvidenceCapsuleError(f"SQLite sidecars cannot be capsule sources: {path}")
    if _is_sqlite(path):
        sidecar_reader, _sidecar_name = _open_reader_for_file(path)
        try:
            for suffix in ("-wal", "-shm"):
                try:
                    sidecar_reader.entry_stat(path.name + suffix)
                except (FileNotFoundError, NotADirectoryError, SecureReadError):
                    continue
                raise EvidenceCapsuleError(
                    f"SQLite evidence has a WAL/SHM sidecar: {path.name + suffix}"
                )
        finally:
            sidecar_reader.close()
    if not 0 <= int(compression_level) <= 9:
        raise EvidenceCapsuleError("gzip compression level must be between 0 and 9")

    manifest_path = _manifest_path(path)
    _reject_symlink_components(manifest_path, label="Evidence capsule manifest")
    manifest_reader, _manifest_name = _open_reader_for_file(manifest_path)
    try:
        try:
            manifest_reader.entry_stat(EVIDENCE_CAPSULE_MANIFEST)
        except (FileNotFoundError, NotADirectoryError, SecureReadError):
            existing_payload = None
        else:
            existing_payload = _read_manifest_bytes(manifest_reader, EVIDENCE_CAPSULE_MANIFEST)
    finally:
        manifest_reader.close()
    if existing_payload is not None:
        try:
            valid_manifest = (
                isinstance(existing_payload, dict)
                and existing_payload.get("schema") == EVIDENCE_CAPSULE_SCHEMA
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise EvidenceCapsuleError(f"Refusing to replace an unreadable manifest: {manifest_path}") from exc
        if not valid_manifest:
            raise EvidenceCapsuleError(f"Refusing to replace an unrelated manifest: {manifest_path}")

    original_sha256 = str(source_record["sha256"])
    verification = source_verification if _is_sqlite(path) else {"byte_round_trip": True}
    query_results: list[dict[str, Any]] = []
    if verification_queries:
        if not _is_sqlite(path):
            raise EvidenceCapsuleError("verification queries are supported only for SQLite evidence")
        if len(verification_queries) > MAX_CAPSULE_VERIFICATION_QUERIES:
            raise EvidenceCapsuleError(
                "Evidence capsule verification query count exceeds the configured limit."
            )
        query_reader, query_name = _open_reader_for_file(path)
        try:
            with _protected_sqlite_connection(query_reader, query_name, path) as connection:
                for query in verification_queries:
                    query_results.append(
                        {"query": str(query), "rows": _verification_query_rows(connection, str(query))}
                    )
        finally:
            query_reader.close()
        verification["queries"] = query_results
    provenance: list[dict[str, Any]] = []
    for raw_provenance_path in provenance_paths:
        provenance_path = _lexical_path(raw_provenance_path)
        _reject_symlink_components(provenance_path, label="Evidence provenance input")
        provenance_reader, provenance_name = _open_reader_for_file(provenance_path)
        try:
            provenance_record = provenance_reader.file_record(
                provenance_name,
                max_bytes=MAX_CAPSULE_SOURCE_BYTES,
            )
        except (FileNotFoundError, NotADirectoryError, SecureReadError) as exc:
            raise EvidenceCapsuleError(
                f"Evidence provenance input is missing or unsafe: {provenance_path}"
            ) from exc
        finally:
            provenance_reader.close()
        provenance.append(
            {
                "path": provenance_path.as_posix(),
                "bytes": int(provenance_record["bytes"]),
                "sha256": str(provenance_record["sha256"]),
            }
        )

    compressed = compressed_evidence_path(path)
    staged = compressed.with_name(f".{compressed.name}.tmp-{os.getpid()}")
    source_reader, source_name = _open_reader_for_file(path)
    try:
        with source_reader.open_file(
            source_name,
            max_bytes=MAX_CAPSULE_SOURCE_BYTES,
        ) as source, staged.open("wb") as raw_target:
            with gzip.GzipFile(
                filename="", mode="wb", fileobj=raw_target, compresslevel=int(compression_level), mtime=0
            ) as target:
                shutil.copyfileobj(source, target, length=_CHUNK_BYTES)
        final_source_record = source_reader.file_record(
            source_name,
            max_bytes=MAX_CAPSULE_SOURCE_BYTES,
        )
    except (FileNotFoundError, NotADirectoryError, SecureReadError) as exc:
        raise EvidenceCapsuleError(f"Evidence source changed during capsule creation: {path}") from exc
    finally:
        source_reader.close()
    if final_source_record != source_record:
        raise EvidenceCapsuleError(f"Evidence source changed during capsule creation: {path}")

    generated_utc = _now_utc()
    entry = {
        "logical_path": path.name,
        "compressed_path": compressed.name,
        "kind": "sqlite" if _is_sqlite(path) else "file",
        "compression": {"algorithm": "gzip", "level": int(compression_level)},
        "original_bytes": int(source_record["bytes"]),
        "original_sha256": original_sha256,
        "original_mtime_ns": int(source_record["mtime_ns"]),
        "compressed_bytes": staged.stat().st_size,
        "compressed_sha256": _sha256(staged),
        "verified_utc": generated_utc,
        "verification": verification,
        "provenance": provenance,
        "restore_command": f"python tools/evidence_capsules.py restore {path.as_posix()}",
    }
    staged.replace(compressed)

    if existing_payload is not None:
        payload = existing_payload
        artifacts = [
            item for item in list(payload.get("artifacts", []) or [])
            if isinstance(item, dict) and item.get("logical_path") != path.name
        ]
    else:
        artifacts = []
    artifacts.append(entry)
    payload = {"schema": EVIDENCE_CAPSULE_SCHEMA, "generated_utc": generated_utc, "artifacts": artifacts}
    staged_manifest = manifest_path.with_name(f".{manifest_path.name}.tmp-{os.getpid()}")
    staged_manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    staged_manifest.replace(manifest_path)

    materialized = materialize_evidence(path, prefer_capsule=True)
    try:
        hydrated = materialized.path
        if _sha256(hydrated) != original_sha256:
            raise EvidenceCapsuleError(f"Post-commit capsule verification failed: {path}")
    finally:
        materialized.close()
    if remove_original:
        source_reader, source_name = _open_reader_for_file(path)
        try:
            current_record = source_reader.file_record(
                source_name,
                max_bytes=MAX_CAPSULE_SOURCE_BYTES,
            )
            if current_record != source_record:
                raise EvidenceCapsuleError(
                    f"Evidence source changed before removal; retaining it: {path}"
                )
            # The descriptor anchors the parent directory and the identity is
            # checked immediately beforehand.  POSIX has no portable atomic
            # unlink-by-inode primitive, so an adversarial same-account rename
            # after this check remains outside the guarantee; retain the file
            # whenever the identity check is uncertain.
            os.unlink(source_name, dir_fd=source_reader.fd)
        except (FileNotFoundError, NotADirectoryError, SecureReadError) as exc:
            raise EvidenceCapsuleError(
                f"Evidence source could not be removed safely; retaining it: {path}"
            ) from exc
        finally:
            source_reader.close()
    return entry


def restore_evidence_capsule(logical_path: str | Path, *, overwrite: bool = False) -> Path:
    """Restore a capsule through the local compatibility writer route.

    Protected MCP handlers do not dispatch to this path-facing restoration
    helper; they consume descriptor-bound evidence readers instead.
    """

    path = _safe_logical_path(logical_path)
    if path.exists() and not overwrite:
        raise EvidenceCapsuleError(f"Evidence destination already exists: {path}")
    _compressed, entry = _compressed_entry(path, verify_compressed=True)
    expected_sha256 = str(entry.get("original_sha256", ""))
    backup = path.with_name(f".{path.name}.backup-{os.getpid()}")
    if path.exists():
        path.replace(backup)
    try:
        materialized = materialize_evidence(
            path,
            prefer_capsule=True,
            secure=False,
        )
        try:
            staged = path.with_name(f".{path.name}.restore-{os.getpid()}")
            shutil.copyfile(materialized.path, staged)
            staged.replace(path)
        finally:
            materialized.close()
        if _sha256(path) != expected_sha256:
            raise EvidenceCapsuleError(f"Restored evidence digest check failed: {path}")
    except Exception:
        path.unlink(missing_ok=True)
        if backup.exists():
            backup.replace(path)
        raise
    backup.unlink(missing_ok=True)
    if _sha256(path) != expected_sha256:
        raise EvidenceCapsuleError(f"Restored evidence digest check failed: {path}")
    return path
