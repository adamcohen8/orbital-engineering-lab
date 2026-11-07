from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import stat
from contextlib import closing
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from sim.review.evidence_capsule import (
    MAX_REVIEW_HYDRATION_BYTES,
    EvidenceCapsuleError,
    MaterializedEvidence,
    _capsule_original_identity,
    _safe_logical_path,
    evidence_file_mtime_ns,
    materialize_evidence,
)
from sim.security.input_reads import (
    DEFAULT_MAX_READ_BYTES,
    SecureReadChanged,
    SecureReadError,
    SecureReadRoot,
    secure_read_primitives_available,
    sqlite_connection_from_snapshot,
    sqlite_snapshot_bytes,
)

REVIEW_QUERY_DEFAULT_MAX_VALUE_BYTES = 1_000_000
REVIEW_QUERY_DEFAULT_MAX_RESULT_BYTES = 4_000_000


def _protected_review_capabilities_available() -> bool:
    """Return whether the default review route can use its secure backend."""

    return secure_read_primitives_available()


def _optional_descriptor_record(
    reader: SecureReadRoot,
    name: str,
    *,
    max_bytes: int,
) -> dict[str, int | str] | None:
    try:
        return dict(reader.file_record(name, max_bytes=max_bytes))
    except (FileNotFoundError, NotADirectoryError):
        return None


def _legacy_optional_file_record(
    path: Path,
    *,
    max_bytes: int,
) -> dict[str, int | str] | None:
    try:
        return _legacy_file_record(path, max_bytes=max_bytes)
    except (FileNotFoundError, NotADirectoryError):
        return None


def _legacy_file_record(path: Path, *, max_bytes: int) -> dict[str, int | str]:
    if path.is_symlink():
        raise SecureReadError(f"Legacy review source must not be a symbolic link: {path}")
    before = path.stat()
    if not stat.S_ISREG(before.st_mode):
        raise SecureReadError(f"Legacy review source must be a regular file: {path}")
    if int(before.st_size) > int(max_bytes):
        raise SecureReadError(f"Legacy review source exceeds the {int(max_bytes)}-byte read budget: {path}")
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            if size > int(max_bytes):
                raise SecureReadError(f"Legacy review source exceeds the {int(max_bytes)}-byte read budget: {path}")
            digest.update(chunk)
    after = path.stat()
    if (
        (
            int(before.st_dev),
            int(before.st_ino),
            int(before.st_size),
            int(before.st_mtime_ns),
            int(before.st_ctime_ns),
            int(before.st_nlink),
        )
        != (
            int(after.st_dev),
            int(after.st_ino),
            int(after.st_size),
            int(after.st_mtime_ns),
            int(after.st_ctime_ns),
            int(after.st_nlink),
        )
        or size != int(after.st_size)
    ):
        raise SecureReadChanged(f"Legacy review source changed during its evidence read: {path}")
    return {
        "bytes": int(size),
        "sha256": digest.hexdigest(),
        "st_dev": int(after.st_dev),
        "st_ino": int(after.st_ino),
        "mtime_ns": int(after.st_mtime_ns),
        "ctime_ns": int(after.st_ctime_ns),
        "nlink": int(after.st_nlink),
    }


def _secure_directory_identity(reader: SecureReadRoot) -> tuple[int, int]:
    metadata = os.fstat(reader.fd)
    if not stat.S_ISDIR(metadata.st_mode):
        raise SecureReadError("Review source namespace is not a directory.")
    return int(metadata.st_dev), int(metadata.st_ino)


def _secure_parent_directory_identity(reader: SecureReadRoot) -> tuple[int, int]:
    parent_fd: int | None = None
    try:
        parent_fd = os.open(
            "..",
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
            dir_fd=reader.fd,
        )
        metadata = os.fstat(parent_fd)
        if not stat.S_ISDIR(metadata.st_mode):
            raise SecureReadError("Review source namespace parent is not a directory.")
        return int(metadata.st_dev), int(metadata.st_ino)
    finally:
        if parent_fd is not None:
            os.close(parent_fd)


def _verify_review_source_namespace(
    *,
    output_reader: SecureReadRoot,
    review_reader: SecureReadRoot,
    db_reader: SecureReadRoot,
    materialized: MaterializedEvidence,
    db_path: Path,
) -> None:
    """Reject descriptors opened from different visible source-root inodes."""

    output_identity = _secure_directory_identity(output_reader)
    review_identity = _secure_directory_identity(review_reader)
    if review_reader.path == output_reader.path:
        if review_identity != output_identity:
            raise SecureReadChanged("Review and output descriptors were opened from different source namespaces.")
    elif _secure_parent_directory_identity(review_reader) != output_identity:
        raise SecureReadChanged("Review and output descriptors were opened from different source namespaces.")

    if not materialized.hydrated:
        if _secure_directory_identity(db_reader) != review_identity:
            raise SecureReadChanged("Database and review descriptors were opened from different source namespaces.")
        return

    # Capsule hydration intentionally uses a private temporary DB directory.
    # Bind its bytes back to the capsule manifest read through the retained
    # source review descriptor so a swap during materialization cannot pair a
    # database from one namespace with schema/summary files from another.
    try:
        original_sha256, original_bytes, _original_mtime_ns = _capsule_original_identity(
            db_path,
            reader=review_reader,
        )
        hydrated_record = db_reader.file_record(
            materialized.path.name,
            max_bytes=MAX_REVIEW_HYDRATION_BYTES,
        )
    except (FileNotFoundError, NotADirectoryError, SecureReadError, EvidenceCapsuleError) as exc:
        raise SecureReadChanged("Capsule database could not be bound to the retained source namespace.") from exc
    if (
        str(hydrated_record.get("sha256", "")) != original_sha256
        or int(hydrated_record.get("bytes", -1)) != int(original_bytes)
    ):
        raise SecureReadChanged("Capsule database bytes do not match the retained source namespace.")


class ReviewStoreNotFoundError(FileNotFoundError):
    """Raised when a review SQLite store cannot be found."""


class ReviewQueryError(ValueError):
    """Raised when a review query is unsafe or cannot be executed."""


@dataclass(frozen=True)
class ReviewQueryResult:
    columns: list[str]
    rows: list[dict[str, Any]]
    row_count: int
    truncated: bool = False


@dataclass
class ReviewWorkspace:
    output_dir: Path
    db_path: Path
    schema_path: Path
    saved_views_path: Path
    _materialized: MaterializedEvidence | None = field(default=None, repr=False, compare=False)
    _db_reader: SecureReadRoot | None = field(default=None, repr=False, compare=False)
    _review_reader: SecureReadRoot | None = field(default=None, repr=False, compare=False)
    _output_reader: SecureReadRoot | None = field(default=None, repr=False, compare=False)
    _db_name: str = field(default="run.sqlite", repr=False, compare=False)
    _source_frozen: bool = field(default=False, repr=False, compare=False)
    _frozen_db_snapshot: bytes | None = field(default=None, repr=False, compare=False)
    _frozen_schema_bytes: bytes | None = field(default=None, repr=False, compare=False)
    _frozen_db_mtime_ns: int | None = field(default=None, repr=False, compare=False)
    _frozen_db_records: dict[str, dict[str, int | str] | None] = field(
        default_factory=dict,
        repr=False,
        compare=False,
    )
    _frozen_schema_record: dict[str, int | str] | None = field(default=None, repr=False, compare=False)
    # Direct dataclass construction is retained for compatibility with local
    # callers.  ``open`` sets this explicitly so all public review handlers
    # use the descriptor-bound path.
    _protected: bool = field(default=False, repr=False, compare=False)

    @classmethod
    def open(
        cls,
        path: str | Path,
        *,
        protected: bool | None = None,
        logical_output_dir: str | Path | None = None,
    ) -> ReviewWorkspace:
        """Open a review store, retaining its directory descriptors.

        ``protected=None`` selects the descriptor-bound route when the host
        provides its required primitives and otherwise preserves the trusted
        local pathname route for compatibility. ``protected=True`` forces the
        descriptor-bound route and fails closed when it is unavailable.
        ``protected=False`` is an explicit compatibility route for trusted
        local callers that require the historical visible-path SQLite
        connection. MCP and other public review handlers must pass
        ``protected=True``.
        """

        if protected is not None and not isinstance(protected, bool):
            raise TypeError("ReviewWorkspace protected must be a bool or None.")
        use_protected = (
            _protected_review_capabilities_available()
            if protected is None
            else bool(protected)
        )

        root = _safe_logical_path(path)
        if root.name in {"run.sqlite", "run.sqlite.gz"}:
            if root.name.endswith(".gz"):
                root = root.with_name(root.name.removesuffix(".gz"))
            db_path = root
            output_dir = root.parent.parent
        elif root.suffix == ".sqlite":
            # Portable review slices and other explicitly supplied SQLite
            # files use their containing directory as the output boundary.
            db_path = root
            output_dir = root.parent
        else:
            output_dir = root
            db_path = output_dir / "review" / "run.sqlite"
        logical_output = (
            _safe_logical_path(logical_output_dir)
            if logical_output_dir is not None
            else output_dir
        )
        source_output_dir = output_dir
        logical_db = logical_output / "review" / "run.sqlite" if logical_output_dir is not None else db_path
        output_reader: SecureReadRoot | None = None
        if use_protected:
            try:
                # Pin the containing output namespace before materialization.
                # The later review/database descriptors are checked against
                # this inode, so a root replacement during opening cannot
                # produce a mixed source namespace.
                output_reader = SecureReadRoot.open(
                    source_output_dir,
                    approved_roots=(source_output_dir.parent,),
                )
            except (OSError, SecureReadError) as exc:
                raise ReviewStoreNotFoundError(
                    f"Review output namespace could not be opened securely: {source_output_dir}"
                ) from exc
        try:
            materialized = materialize_evidence(
                db_path,
                max_original_bytes=MAX_REVIEW_HYDRATION_BYTES,
                secure=use_protected,
            )
        except (FileNotFoundError, ValueError, sqlite3.DatabaseError) as exc:
            if output_reader is not None:
                output_reader.close()
            raise ReviewStoreNotFoundError(f"Review store not found or invalid: {db_path}") from exc
        except BaseException:
            # Materialization can surface a descriptor-bound change directly
            # (for example, while hydrating a capsule).  Do not leave the
            # namespace pin open on an exceptional path that predates the
            # workspace cleanup block below.
            if output_reader is not None:
                output_reader.close()
            raise
        # A protected path-only consumer may receive a private sealed input
        # tree while its outputs and provenance must remain under the caller's
        # logical run directory.  Keep the materialized/descriptor paths for
        # reads, but expose the explicitly supplied logical namespace to
        # renderers and artifact records.
        if logical_output_dir is not None:
            materialized.logical_path = logical_db
            output_dir = logical_output
        review_dir = db_path.parent
        if not use_protected:
            return cls(
                output_dir=output_dir,
                db_path=materialized.path,
                schema_path=review_dir / "schema.json",
                saved_views_path=review_dir / "saved_views.json",
                _materialized=materialized,
                _protected=False,
            )
        db_reader: SecureReadRoot | None = None
        review_reader: SecureReadRoot | None = None
        try:
            # Keep both the logical path contract and descriptor-bound
            # handles. Protected SQLite consumers hydrate from ``db_reader``;
            # ``db_path`` remains a compatibility identity for callers that
            # explicitly request the legacy route.
            db_anchor = materialized.path.parent
            db_reader = SecureReadRoot.open(db_anchor, approved_roots=(db_anchor.parent,))
            review_reader = SecureReadRoot.open(review_dir, approved_roots=(review_dir.parent,))
            assert output_reader is not None
            _verify_review_source_namespace(
                output_reader=output_reader,
                review_reader=review_reader,
                db_reader=db_reader,
                materialized=materialized,
                db_path=db_path,
            )
            return cls(
                output_dir=output_dir,
                db_path=materialized.path,
                schema_path=review_dir / "schema.json",
                saved_views_path=review_dir / "saved_views.json",
                _materialized=materialized,
                _db_reader=db_reader,
                _review_reader=review_reader,
                _output_reader=output_reader,
                _db_name=materialized.path.name,
                _protected=use_protected,
            )
        except (OSError, SecureReadError, EvidenceCapsuleError) as exc:
            if db_reader is not None:
                db_reader.close()
            if review_reader is not None:
                review_reader.close()
            if output_reader is not None:
                output_reader.close()
            materialized.close()
            raise ReviewStoreNotFoundError(f"Review store could not be opened securely: {db_path}") from exc

    def close(self) -> None:
        if self._db_reader is not None:
            self._db_reader.close()
            self._db_reader = None
        if self._review_reader is not None:
            self._review_reader.close()
            self._review_reader = None
        if self._output_reader is not None:
            self._output_reader.close()
            self._output_reader = None
        if self._materialized is not None:
            self._materialized.close()
            self._materialized = None

    def freeze_source(self) -> None:
        """Bind protected reads to one bounded source snapshot for this workspace.

        Protected review queries already hydrate from retained directory
        descriptors.  Exporters that need to bind provenance to the same
        content can opt into this one-shot snapshot before their first query;
        subsequent queries and identity checks then use the captured bytes
        instead of reopening the mutable visible path.
        """

        if self._source_frozen:
            return
        if self._protected:
            if self._db_reader is None:
                raise ReviewStoreNotFoundError(
                    "Protected review access requires a retained database directory descriptor."
                )
            try:
                db_records_before = self._capture_source_records()
                self._frozen_db_snapshot = sqlite_snapshot_bytes(
                    self._db_reader,
                    self._db_name,
                    max_bytes=MAX_REVIEW_HYDRATION_BYTES,
                )
                db_records_after = self._capture_source_records()
                if db_records_after != db_records_before:
                    raise SecureReadChanged("Review store changed while its protected source snapshot was captured.")
                self._frozen_db_records = db_records_before
                self._frozen_db_mtime_ns = int(self._db_reader.stat(self._db_name).st_mtime_ns)
                try:
                    self._frozen_db_mtime_ns = max(
                        self._frozen_db_mtime_ns,
                        int(self._db_reader.stat(f"{self._db_name}-wal").st_mtime_ns),
                    )
                except (FileNotFoundError, NotADirectoryError, SecureReadError):
                    pass
                if self._review_reader is not None:
                    schema_record_before = _optional_descriptor_record(
                        self._review_reader,
                        "schema.json",
                        max_bytes=DEFAULT_MAX_READ_BYTES,
                    )
                    try:
                        self._frozen_schema_bytes = self._review_reader.read_bytes(
                            "schema.json",
                            max_bytes=DEFAULT_MAX_READ_BYTES,
                        )
                    except FileNotFoundError:
                        self._frozen_schema_bytes = None
                    schema_record_after = _optional_descriptor_record(
                        self._review_reader,
                        "schema.json",
                        max_bytes=DEFAULT_MAX_READ_BYTES,
                    )
                    if schema_record_after != schema_record_before:
                        raise SecureReadChanged(
                            "Review schema changed while its protected source snapshot was captured."
                        )
                    if self._frozen_schema_bytes is not None:
                        digest = hashlib.sha256(self._frozen_schema_bytes).hexdigest()
                        if schema_record_before is None or digest != str(schema_record_before["sha256"]):
                            raise SecureReadChanged(
                                "Review schema bytes did not match their retained source record."
                            )
                    self._frozen_schema_record = schema_record_before
            except (FileNotFoundError, NotADirectoryError, SecureReadError, sqlite3.DatabaseError) as exc:
                raise ReviewStoreNotFoundError(
                    "Protected review source snapshot could not be opened securely."
                ) from exc
        else:
            # Explicit legacy compatibility route for trusted local callers.
            # Public/protected callers take the descriptor-bound branch above.
            try:
                db_records_before = self._capture_source_records()
                self._frozen_db_snapshot = self.db_path.read_bytes()
                self._frozen_db_mtime_ns = int(self.db_path.stat().st_mtime_ns)
                db_records_after = self._capture_source_records()
                if db_records_after != db_records_before:
                    raise SecureReadChanged("Review store changed while its source snapshot was captured.")
                self._frozen_db_records = db_records_before
            except OSError as exc:
                raise ReviewStoreNotFoundError(
                    f"Review store source snapshot could not be read: {self.db_path}"
                ) from exc
            try:
                schema_record_before = _legacy_optional_file_record(
                    self.schema_path,
                    max_bytes=DEFAULT_MAX_READ_BYTES,
                )
                self._frozen_schema_bytes = self.schema_path.read_bytes()
                schema_record_after = _legacy_file_record(
                    self.schema_path,
                    max_bytes=DEFAULT_MAX_READ_BYTES,
                )
                if schema_record_after != schema_record_before:
                    raise SecureReadChanged("Review schema changed while its source snapshot was captured.")
                if hashlib.sha256(self._frozen_schema_bytes).hexdigest() != str(schema_record_before["sha256"]):
                    raise SecureReadChanged("Review schema bytes did not match their retained source record.")
                self._frozen_schema_record = schema_record_before
            except FileNotFoundError:
                self._frozen_schema_bytes = None
                self._frozen_schema_record = None
            except OSError as exc:
                raise ReviewStoreNotFoundError(
                    f"Review schema source snapshot could not be read: {self.schema_path}"
                ) from exc
        self._source_frozen = True

    def source_schema_bytes(self) -> bytes | None:
        """Return the exact schema bytes bound to this workspace, if present."""

        self.freeze_source()
        return self._frozen_schema_bytes

    def source_db_record(self) -> dict[str, int | str]:
        """Return the raw main-database record bound to this workspace."""

        self.freeze_source()
        record = self._frozen_db_records.get(self._db_name)
        if record is None:
            raise ReviewStoreNotFoundError("Review store source record is unavailable.")
        return dict(record)

    def source_schema_record(self) -> dict[str, int | str] | None:
        """Return the retained schema record, if the source contains one."""

        self.freeze_source()
        return None if self._frozen_schema_record is None else dict(self._frozen_schema_record)

    def has_live_wal(self) -> bool:
        """Return whether the retained source includes an uncheckpointed WAL."""

        wal_name = f"{self._db_name}-wal"
        if self._source_frozen:
            return self._frozen_db_records.get(wal_name) is not None
        if self._protected and self._db_reader is not None:
            try:
                metadata = self._db_reader.entry_stat(wal_name)
            except (FileNotFoundError, NotADirectoryError):
                return False
            if not stat.S_ISREG(metadata.st_mode):
                raise SecureReadError("SQLite WAL entry is not a regular file.")
            return True
        return self.db_path.with_name(wal_name).is_file()

    def assert_source_unchanged(self) -> None:
        """Reject in-place source mutations while retaining the original root.

        The comparison is descriptor-bound. Renaming or replacing the visible
        parent leaves these retained handles pinned to the original directory
        and therefore remains valid; changing an opened source entry in place,
        replacing its inode, or adding/removing a WAL or schema is rejected.
        """

        self.freeze_source()
        current_db_records = self._capture_source_records()
        if current_db_records != self._frozen_db_records:
            raise SecureReadChanged("Review store changed after its source snapshot was captured.")
        if self._protected and self._review_reader is not None:
            current_schema_record = _optional_descriptor_record(
                self._review_reader,
                "schema.json",
                max_bytes=DEFAULT_MAX_READ_BYTES,
            )
        else:
            current_schema_record = (
                None
                if self._frozen_schema_record is None
                else _legacy_file_record(self.schema_path, max_bytes=DEFAULT_MAX_READ_BYTES)
            )
        if current_schema_record != self._frozen_schema_record:
            raise SecureReadChanged("Review schema changed after its source snapshot was captured.")

    def _capture_source_records(self) -> dict[str, dict[str, int | str] | None]:
        names = (
            self._db_name,
            f"{self._db_name}-wal",
            f"{self._db_name}-shm",
            f"{self._db_name}-journal",
        )
        if self._protected and self._db_reader is not None:
            return {
                name: _optional_descriptor_record(
                    self._db_reader,
                    name,
                    max_bytes=MAX_REVIEW_HYDRATION_BYTES,
                )
                for name in names
            }
        return {
            name: _legacy_optional_file_record(
                self.db_path.with_name(name),
                max_bytes=MAX_REVIEW_HYDRATION_BYTES,
            )
            for name in names
        }

    @property
    def logical_db_path(self) -> Path:
        """Stable review-store identity, independent of capsule hydration."""

        if self._materialized is not None:
            return self._materialized.logical_path
        return self.output_dir / "review" / "run.sqlite"

    @property
    def review_dir(self) -> Path:
        """Durable review artifact directory for the logical output."""

        return self.logical_db_path.parent

    @property
    def source_output_reader(self) -> SecureReadRoot | None:
        """Return the retained source-output descriptor, when protected."""

        return self._output_reader

    def evidence_identity(self) -> dict[str, Any]:
        """Return content identity without exposing a temporary hydration path."""

        logical = self.logical_db_path
        try:
            relative_path = str(logical.relative_to(self.output_dir))
        except ValueError:
            relative_path = str(logical)
        if self._source_frozen and self._frozen_db_snapshot is not None:
            digest = hashlib.sha256(self._frozen_db_snapshot).hexdigest()
            size_bytes = len(self._frozen_db_snapshot)
            mtime_ns = int(self._frozen_db_mtime_ns or 0)
        elif self._protected and self._db_reader is not None:
            if self._materialized is not None and self._materialized.hydrated and self._review_reader is not None:
                # Capsule hydration is intentionally temporary.  Bind the
                # public identity to the retained source manifest instead of
                # the freshly-created hydration file whose mtime changes on
                # every open.  The helper verifies the compressed entry via
                # the retained source descriptor before trusting its
                # content-bound original record.
                digest, size_bytes, mtime_ns = _capsule_original_identity(
                    self._review_reader.path / self._db_name,
                    reader=self._review_reader,
                )
            else:
                digest, size_bytes = _sqlite_evidence_digest_from_reader(
                    self._db_reader,
                    self._db_name,
                )
                mtime_ns = int(self._db_reader.stat(self._db_name).st_mtime_ns)
        else:
            # Explicit legacy compatibility route.  Protected callers never
            # hash a checked path and then reopen that path below.
            digest, size_bytes = _sqlite_evidence_digest(self.db_path)
            mtime_ns = int(evidence_file_mtime_ns(logical, secure=False))
        wal_name = f"{self._db_name}-wal"
        if self._protected and self._db_reader is not None and not (
            self._materialized is not None and self._materialized.hydrated
        ):
            try:
                mtime_ns = max(mtime_ns, int(self._db_reader.stat(wal_name).st_mtime_ns))
            except (FileNotFoundError, NotADirectoryError, SecureReadError):
                pass
        return {
            "relative_path": relative_path,
            "size_bytes": size_bytes,
            "mtime_ns": mtime_ns,
            "sha256": digest,
            "storage": (
                "sqlite"
                if self._materialized is None or not self._materialized.hydrated
                else "sqlite_gzip_capsule"
            ),
        }

    def __enter__(self) -> ReviewWorkspace:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def tables(self) -> list[str]:
        with closing(self._connect()) as conn:
            rows = conn.execute(
                """
                SELECT name
                FROM sqlite_schema
                WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
                ORDER BY name
                """
            ).fetchall()
        return [str(row["name"]) for row in rows]

    def schema(self) -> dict[str, Any]:
        schema: Any = {}
        if self._source_frozen:
            if self._frozen_schema_bytes is not None:
                try:
                    schema = json.loads(self._frozen_schema_bytes.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    schema = {}
        elif self._review_reader is not None:
            try:
                schema = json.loads(
                    self._review_reader.read_text("schema.json", max_bytes=DEFAULT_MAX_READ_BYTES)
                )
            except (FileNotFoundError, NotADirectoryError, SecureReadError, json.JSONDecodeError):
                schema = {}
        elif not self._protected:
            try:
                schema = json.loads(self.schema_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                schema = {}
        out = dict(schema)
        out["columns"] = self.table_columns()
        return out

    def table_columns(self) -> dict[str, list[dict[str, Any]]]:
        columns: dict[str, list[dict[str, Any]]] = {}
        with closing(self._connect()) as conn:
            tables = [
                str(row["name"])
                for row in conn.execute(
                    """
                    SELECT name
                    FROM sqlite_schema
                    WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
                    ORDER BY name
                    """
                ).fetchall()
            ]
            for table in tables:
                rows = conn.execute(f"PRAGMA table_info({_quote_identifier(table)})").fetchall()
                columns[table] = [
                    {
                        "name": str(row["name"]),
                        "type": str(row["type"] or ""),
                        "notnull": bool(row["notnull"]),
                        "primary_key": bool(row["pk"]),
                    }
                    for row in rows
                ]
        return columns

    def saved_views(self) -> list[dict[str, Any]]:
        if self._review_reader is not None:
            try:
                data = json.loads(
                    self._review_reader.read_text("saved_views.json", max_bytes=DEFAULT_MAX_READ_BYTES)
                )
            except (FileNotFoundError, NotADirectoryError, SecureReadError, json.JSONDecodeError):
                return []
        elif not self._protected:
            try:
                data = json.loads(self.saved_views_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return []
        else:
            return []
        if isinstance(data, list):
            return [dict(item) for item in data if isinstance(item, dict)]
        if isinstance(data, dict) and isinstance(data.get("views"), list):
            return [dict(item) for item in data["views"] if isinstance(item, dict)]
        return []

    def query(
        self,
        sql: str,
        parameters: Any = None,
        *,
        max_rows: int = 1000,
        max_vm_steps: int = 250_000,
        max_value_bytes: int | None = REVIEW_QUERY_DEFAULT_MAX_VALUE_BYTES,
        max_result_bytes: int | None = REVIEW_QUERY_DEFAULT_MAX_RESULT_BYTES,
    ) -> ReviewQueryResult:
        """Run a read-only query under bounded local defaults.

        Callers with an explicitly approved larger result may pass larger
        ``max_value_bytes``/``max_result_bytes`` values. ``None`` remains an
        explicit opt-out for trusted internal callers that already enforce a
        stronger outer budget.
        """
        statement = _validate_select_sql(sql)
        limit = int(max(max_rows, 1))
        step_budget = int(max(max_vm_steps, 1))
        params = () if parameters is None else parameters
        try:
            with closing(self._connect(authorize=True)) as conn:
                if max_value_bytes is not None and hasattr(conn, "setlimit"):
                    conn.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, max(1, int(max_value_bytes)))
                steps = 0

                def _abort_long_query() -> int:
                    nonlocal steps
                    steps += 1000
                    return 1 if steps > step_budget else 0

                conn.set_progress_handler(_abort_long_query, 1000)
                cursor = conn.execute(statement, params)
                columns = [str(item[0]) for item in (cursor.description or [])]
                if len(set(columns)) != len(columns):
                    raise ReviewQueryError(
                        "Review query column names must be unique; use explicit aliases for duplicate columns."
                    )
                rows: list[dict[str, Any]] = []
                result_bytes = 0
                truncated = False
                for _index in range(limit + 1):
                    row = cursor.fetchone()
                    if row is None:
                        break
                    if _index == limit:
                        truncated = True
                        break
                    converted: dict[str, Any] = {}
                    for column_index, column in enumerate(columns):
                        # sqlite3.Row name lookup is case-insensitive; positional
                        # lookup preserves distinct aliases such as Range/range.
                        value = row[column_index]
                        value_bytes = _value_size_bytes(value)
                        if max_value_bytes is not None and value_bytes > int(max_value_bytes):
                            raise ReviewQueryError("Review query value exceeds the per-value byte budget.")
                        result_bytes += len(column.encode("utf-8")) + value_bytes
                        if max_result_bytes is not None and result_bytes > int(max_result_bytes):
                            raise ReviewQueryError("Review query result exceeds the response-size budget (aggregate bytes).")
                        converted[column] = value
                    rows.append(converted)
                conn.set_progress_handler(None, 0)
        except sqlite3.Error as exc:
            message = _bounded_sqlite_error_message(exc, value_budget_enabled=max_value_bytes is not None)
            raise ReviewQueryError(message) from exc
        return ReviewQueryResult(
            columns=columns,
            rows=rows,
            row_count=len(rows),
            truncated=truncated,
        )

    def _connect(
        self,
        *,
        authorize: bool = False,
        integrity: bool = False,
    ) -> sqlite3.Connection:
        if authorize and integrity:
            raise ValueError("Review connections cannot combine public and internal authorizers.")
        if self._protected:
            if self._db_reader is None:
                raise ReviewStoreNotFoundError(
                    "Protected review access requires a retained database directory descriptor."
                )
            conn: sqlite3.Connection | None = None
            try:
                if self._source_frozen and self._frozen_db_snapshot is not None:
                    payload = self._frozen_db_snapshot
                else:
                    payload = sqlite_snapshot_bytes(
                        self._db_reader,
                        self._db_name,
                        max_bytes=MAX_REVIEW_HYDRATION_BYTES,
                    )
                # SQLite represents a zero-byte database as a valid empty
                # store, but ``Connection.deserialize(b"")`` raises
                # ``MemoryError`` on supported Python builds.  Preserve the
                # empty-store semantics with a pristine in-memory database;
                # the normal read-only/query authorizer setup below still
                # applies and missing tables become the bounded query error
                # expected by review callers.
                conn = sqlite_connection_from_snapshot(payload)
            except (SecureReadError, sqlite3.DatabaseError) as exc:
                if conn is not None:
                    conn.close()
                raise ReviewStoreNotFoundError(
                    "Protected review SQLite snapshot could not be opened securely."
                ) from exc
        else:
            # Explicit compatibility path for trusted local callers.  This
            # is intentionally separate from the protected MCP route.
            uri = f"{self.db_path.resolve().as_uri()}?mode=ro"
            conn = sqlite3.connect(uri, uri=True)
        assert conn is not None
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA query_only = ON")
        if integrity:
            conn.set_authorizer(_slice_integrity_authorizer)
        elif authorize:
            conn.set_authorizer(_read_only_authorizer)
        return conn


def _sqlite_evidence_digest(db_path: Path) -> tuple[str, int]:
    """Hash a consistent logical DB snapshot when live content resides in WAL."""

    resolved = _safe_logical_path(db_path)
    uri = f"{resolved.as_uri()}?mode=ro"
    with closing(sqlite3.connect(uri, uri=True)) as source:
        source.execute("PRAGMA query_only = ON")
        source.execute("BEGIN")
        source.execute("SELECT rootpage FROM sqlite_schema LIMIT 1").fetchone()
        journal_mode = str(source.execute("PRAGMA journal_mode").fetchone()[0] or "").lower()
        if journal_mode != "wal":
            # Compatibility-only path. Protected callers use the retained
            # descriptor helper above and never reopen this visible path.
            digest = hashlib.sha256()
            with resolved.open("rb") as reader:
                for chunk in iter(lambda: reader.read(1024 * 1024), b""):
                    digest.update(chunk)
            return digest.hexdigest(), int(resolved.stat().st_size)
        with TemporaryDirectory(prefix="oel-review-identity-") as temporary_dir:
            snapshot_path = Path(temporary_dir) / "snapshot.sqlite"
            with closing(sqlite3.connect(snapshot_path)) as destination:
                source.backup(destination)
            digest = hashlib.sha256()
            with snapshot_path.open("rb") as reader:
                for chunk in iter(lambda: reader.read(1024 * 1024), b""):
                    digest.update(chunk)
            return digest.hexdigest(), int(snapshot_path.stat().st_size)


def _sqlite_evidence_digest_from_reader(
    reader: SecureReadRoot,
    name: str,
) -> tuple[str, int]:
    """Hash the bounded SQLite snapshot obtained from a retained descriptor."""

    payload = sqlite_snapshot_bytes(
        reader,
        name,
        max_bytes=MAX_REVIEW_HYDRATION_BYTES,
    )
    return hashlib.sha256(payload).hexdigest(), len(payload)


def _value_size_bytes(value: Any) -> int:
    if value is None:
        return 4
    if isinstance(value, bytes):
        return len(value)
    if isinstance(value, str):
        return len(value.encode("utf-8"))
    return len(str(value).encode("utf-8"))


def _bounded_sqlite_error_message(exc: sqlite3.Error, *, value_budget_enabled: bool) -> str:
    """Translate SQLite limit failures into the stable review-query contract."""

    message = str(exc)
    lowered = message.lower()
    if "interrupted" in lowered:
        return "Review query exceeded the execution step budget."
    if value_budget_enabled and "string or blob too big" in lowered:
        # SQLite applies SQLITE_LIMIT_LENGTH to the encoded value/row record,
        # so this error cannot identify whether payload bytes or record
        # overhead crossed the cap. Keep the public message generic.
        return "Review query exceeded the configured SQLite size budget."
    return message


def execute_bounded_review_query(
    connection: sqlite3.Connection,
    statement: str,
    *,
    max_rows: int = 10_000,
    max_vm_steps: int = 250_000,
    max_value_bytes: int = REVIEW_QUERY_DEFAULT_MAX_VALUE_BYTES,
    max_result_bytes: int = REVIEW_QUERY_DEFAULT_MAX_RESULT_BYTES,
) -> list[Any]:
    """Execute a read-only review query with CPU, row, and byte bounds.

    Qualification and maturation gates consume SQLite rows directly rather
    than through ``ReviewWorkspace.query``.  Keep that gate-facing path under
    the same bounded execution envelope so a recursive CTE or wide result
    cannot run indefinitely or accumulate an unbounded list.
    """

    statement = _validate_select_sql(statement)
    row_limit = max(int(max_rows), 1)
    step_budget = max(int(max_vm_steps), 1)
    value_budget = max(int(max_value_bytes), 1)
    result_budget = max(int(max_result_bytes), 1)
    # ``PRAGMA query_only`` is not a complete SQL capability boundary (for
    # example, it does not by itself reject ATTACH or extension loading).
    # Install the same authorizer used by the public workspace query path so
    # callers that provide their own read-only connection get the same safety
    # contract as protected review callers.
    connection.set_authorizer(_read_only_authorizer)
    if hasattr(connection, "setlimit"):
        connection.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, value_budget)
    steps = 0

    def _abort_long_query() -> int:
        nonlocal steps
        steps += 1000
        return 1 if steps > step_budget else 0

    connection.set_progress_handler(_abort_long_query, 1000)
    try:
        cursor = connection.execute(statement)
        rows: list[Any] = []
        result_bytes = 0
        for index in range(row_limit + 1):
            row = cursor.fetchone()
            if row is None:
                break
            if index >= row_limit:
                raise ReviewQueryError("Review query exceeded the row limit.")
            for value in row:
                value_bytes = _value_size_bytes(value)
                if value_bytes > value_budget:
                    raise ReviewQueryError("Review query value exceeds the per-value byte budget.")
                result_bytes += value_bytes
                if result_bytes > result_budget:
                    raise ReviewQueryError("Review query result exceeds the aggregate byte budget.")
            rows.append(row)
        return rows
    except sqlite3.Error as exc:
        raise ReviewQueryError(_bounded_sqlite_error_message(exc, value_budget_enabled=True)) from exc
    finally:
        connection.set_progress_handler(None, 0)

def _validate_select_sql(sql: str) -> str:
    statement = str(sql or "").strip()
    if not statement:
        raise ReviewQueryError("Review query cannot be empty.")
    if "\x00" in statement:
        raise ReviewQueryError("Review query cannot contain null bytes.")
    statement = _strip_leading_comments(statement).strip()
    statement = _strip_single_trailing_semicolon(statement)
    first_token = _first_sql_token(statement)
    if first_token not in {"select", "with"}:
        raise ReviewQueryError("Review queries are read-only and must start with SELECT or WITH.")
    return statement


def _strip_single_trailing_semicolon(statement: str) -> str:
    semicolon_count = 0
    in_single = False
    in_double = False
    for char in statement:
        if char == "'" and not in_double:
            in_single = not in_single
        elif char == '"' and not in_single:
            in_double = not in_double
        elif char == ";" and not in_single and not in_double:
            semicolon_count += 1
    if semicolon_count == 0:
        return statement
    stripped = statement.rstrip()
    if semicolon_count == 1 and stripped.endswith(";"):
        return stripped[:-1].rstrip()
    raise ReviewQueryError("Review queries must contain exactly one read-only statement.")


def _strip_leading_comments(statement: str) -> str:
    out = statement.lstrip()
    while out.startswith("--") or out.startswith("/*"):
        if out.startswith("--"):
            _, _, remainder = out.partition("\n")
            out = remainder.lstrip()
            continue
        end = out.find("*/")
        if end < 0:
            raise ReviewQueryError("Review query has an unterminated block comment.")
        out = out[end + 2 :].lstrip()
    return out


def _first_sql_token(statement: str) -> str:
    match = re.match(r"([A-Za-z_][A-Za-z0-9_]*)", statement)
    return match.group(1).lower() if match else ""


def _read_only_authorizer(action_code: int, param1: str | None, param2: str | None, *_args: Any) -> int:
    dangerous = {
        getattr(sqlite3, "SQLITE_ALTER_TABLE", -1),
        getattr(sqlite3, "SQLITE_ATTACH", -1),
        getattr(sqlite3, "SQLITE_CREATE_INDEX", -1),
        getattr(sqlite3, "SQLITE_CREATE_TABLE", -1),
        getattr(sqlite3, "SQLITE_CREATE_TEMP_INDEX", -1),
        getattr(sqlite3, "SQLITE_CREATE_TEMP_TABLE", -1),
        getattr(sqlite3, "SQLITE_CREATE_TEMP_TRIGGER", -1),
        getattr(sqlite3, "SQLITE_CREATE_TEMP_VIEW", -1),
        getattr(sqlite3, "SQLITE_CREATE_TRIGGER", -1),
        getattr(sqlite3, "SQLITE_CREATE_VIEW", -1),
        getattr(sqlite3, "SQLITE_DELETE", -1),
        getattr(sqlite3, "SQLITE_DETACH", -1),
        getattr(sqlite3, "SQLITE_DROP_INDEX", -1),
        getattr(sqlite3, "SQLITE_DROP_TABLE", -1),
        getattr(sqlite3, "SQLITE_DROP_TEMP_INDEX", -1),
        getattr(sqlite3, "SQLITE_DROP_TEMP_TABLE", -1),
        getattr(sqlite3, "SQLITE_DROP_TEMP_TRIGGER", -1),
        getattr(sqlite3, "SQLITE_DROP_TEMP_VIEW", -1),
        getattr(sqlite3, "SQLITE_DROP_TRIGGER", -1),
        getattr(sqlite3, "SQLITE_DROP_VIEW", -1),
        getattr(sqlite3, "SQLITE_INSERT", -1),
        getattr(sqlite3, "SQLITE_PRAGMA", -1),
        getattr(sqlite3, "SQLITE_REINDEX", -1),
        getattr(sqlite3, "SQLITE_SAVEPOINT", -1),
        getattr(sqlite3, "SQLITE_TRANSACTION", -1),
        getattr(sqlite3, "SQLITE_UPDATE", -1),
    }
    if action_code in dangerous:
        return sqlite3.SQLITE_DENY
    if action_code == getattr(sqlite3, "SQLITE_FUNCTION", -2):
        function_name = str(param1 or param2 or "").lower()
        if function_name == "load_extension":
            return sqlite3.SQLITE_DENY
    return sqlite3.SQLITE_OK


def _slice_integrity_authorizer(action_code: int, param1: str | None, param2: str | None, *_args: Any) -> int:
    """Allow only the bounded source checks needed while creating a slice.

    Slice creation is an internal transformation, but its source connection
    still reads from an untrusted review store.  Keep this authorizer narrower
    than the public read-only query route: it permits schema/table reads,
    ``COUNT``/``MIN``/``MAX``/``LIKE`` inventory checks, and the two integrity
    pragmas used by the slice preflight.  It never grants general-purpose
    PRAGMA, function, transaction, or mutation access.
    """

    if action_code in {
        getattr(sqlite3, "SQLITE_READ", -1),
        getattr(sqlite3, "SQLITE_SELECT", -1),
    }:
        return sqlite3.SQLITE_OK
    if action_code == getattr(sqlite3, "SQLITE_PRAGMA", -2):
        pragma_name = str(param1 or "").lower()
        if pragma_name in {"quick_check", "table_info"}:
            return sqlite3.SQLITE_OK
        return sqlite3.SQLITE_DENY
    if action_code == getattr(sqlite3, "SQLITE_FUNCTION", -3):
        function_name = str(param1 or param2 or "").lower()
        if function_name in {"count", "min", "max", "like"}:
            return sqlite3.SQLITE_OK
    return sqlite3.SQLITE_DENY


def _quote_identifier(value: str) -> str:
    return '"' + str(value).replace('"', '""') + '"'
