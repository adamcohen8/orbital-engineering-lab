"""Immutable artifacts and cross-process acquisition/rate coordination."""
from __future__ import annotations

import os
import re
import sqlite3
import uuid
from contextlib import contextmanager
from pathlib import Path

from .contracts import MAX_BYTES, RECEIPT, AcquisitionError, canonical, digest, stamp, strict_json, utcnow
from .files import headroom, private_dir, publish, read_snapshot


class Store:
    def __init__(self, root: Path):
        self.root = Path(root).absolute()

    @contextmanager
    def connect(self):
        private_dir(self.root)
        os.chmod(self.root, 0o700)
        if (self.root / "index.sqlite").is_symlink():
            raise AcquisitionError("UNSAFE_PATH", "Acquisition index cannot be a symbolic link.")
        db = sqlite3.connect(self.root / "index.sqlite", timeout=2)
        try:
            db.row_factory = sqlite3.Row
            db.execute("CREATE TABLE IF NOT EXISTS artifacts (id TEXT PRIMARY KEY, receipt BLOB NOT NULL, receipt_sha TEXT NOT NULL, request_key TEXT, scope TEXT NOT NULL)")
            db.execute("CREATE INDEX IF NOT EXISTS requests ON artifacts(request_key)")
            db.execute("CREATE TABLE IF NOT EXISTS attempts (scope TEXT, product TEXT, time REAL)")
            db.execute("CREATE TABLE IF NOT EXISTS backoff (scope TEXT PRIMARY KEY, until REAL)")
            db.execute("CREATE TABLE IF NOT EXISTS pins (artifact TEXT, owner TEXT, PRIMARY KEY(artifact, owner))")
            db.commit()
            yield db
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    @contextmanager
    def operation(self, scope):
        locks = private_dir(self.root / "locks")
        path = locks / f"{digest(scope.encode())}.sqlite"
        if path.is_symlink():
            raise AcquisitionError("UNSAFE_PATH", "Acquisition lock cannot be a symbolic link.")
        db = sqlite3.connect(path, timeout=0.1)
        try:
            try:
                db.execute("BEGIN EXCLUSIVE")
            except sqlite3.OperationalError as exc:
                raise AcquisitionError("BUSY", "Another OEL process is acquiring data for this account/provider; retry later.", retry_after_s=2) from exc
            yield
        finally:
            db.rollback()
            db.close()

    def reserve(self, scope, product, *, now: float, retrieval_interval: int = 0):
        """Count attempts, including login/retries, under a database transaction."""
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            blocked = db.execute("SELECT until FROM backoff WHERE scope=?", (scope,)).fetchone()
            wait = max(0.0, blocked[0] - now) if blocked else 0.0
            attempts = [r[0] for r in db.execute("SELECT time FROM attempts WHERE scope=? AND time>? ORDER BY time", (scope, now - 3600))]
            minute = [t for t in attempts if t > now - 60]
            if len(minute) >= 20:
                wait = max(wait, minute[0] + 60 - now)
            if len(attempts) >= 180:
                wait = max(wait, attempts[0] + 3600 - now)
            if retrieval_interval:
                row = db.execute("SELECT MAX(time) FROM attempts WHERE scope=? AND product=?", (scope, product)).fetchone()
                if row[0] is not None:
                    wait = max(wait, row[0] + retrieval_interval - now)
            if wait > 0:
                raise AcquisitionError("RATE_LIMIT", "Provider/account retrieval policy requires waiting; reuse retained data or retry later.", retry_after_s=wait)
            db.execute("INSERT INTO attempts VALUES (?, ?, ?)", (scope, product, now))
            db.commit()

    def backoff(self, scope, until):
        with self.connect() as db:
            db.execute("INSERT INTO backoff VALUES (?, ?) ON CONFLICT(scope) DO UPDATE SET until=MAX(until,excluded.until)", (scope, until))

    def put(self, raw: bytes, metadata: dict, *, scope: str, request_key: str | None = None):
        headroom(self.root, len(raw) * 2)
        sha = digest(raw)
        relative = f"objects/{digest(scope.encode())}/{sha}"
        path = self.root / relative
        if path.exists():
            if digest(read_snapshot(path, MAX_BYTES)) != sha:
                raise AcquisitionError("INTEGRITY_ERROR", "Retained artifact is corrupt; restore it from an approved archive.")
        else:
            try:
                publish(path, raw)
            except AcquisitionError as exc:
                if exc.code != "DESTINATION_EXISTS" or digest(read_snapshot(path, MAX_BYTES)) != sha:
                    raise
        artifact = uuid.uuid4().hex
        value = dict(metadata, schema=RECEIPT, artifact_id=artifact, sha256=sha,
                     bytes=len(raw), stored_path=relative, scope=scope, retrieved_at=stamp(utcnow()))
        blob = canonical(value)
        with self.connect() as db:
            db.execute("INSERT INTO artifacts VALUES (?, ?, ?, ?, ?)", (artifact, blob, digest(blob), request_key, scope))
        return value

    def get(self, artifact):
        if not isinstance(artifact, str) or not re.fullmatch(r"[a-f0-9]{32}", artifact):
            raise AcquisitionError("INVALID_ARTIFACT", "Use a returned artifact_id.")
        with self.connect() as db:
            row = db.execute("SELECT receipt, receipt_sha FROM artifacts WHERE id=?", (artifact,)).fetchone()
        if not row:
            raise AcquisitionError("ARTIFACT_MISSING", "The selected artifact is not retained in this store.")
        if digest(row[0]) != row[1]:
            raise AcquisitionError("INTEGRITY_ERROR", "Acquisition receipt identity changed.")
        value = strict_json(row[0])
        path = self.root / value["stored_path"]
        if path.resolve().is_relative_to(self.root.resolve()) is False:
            raise AcquisitionError("UNSAFE_PATH", "Artifact escaped its store.")
        raw = read_snapshot(path, MAX_BYTES)
        if len(raw) != value["bytes"] or digest(raw) != value["sha256"]:
            raise AcquisitionError("INTEGRITY_ERROR", "Retained artifact identity changed; restore the exact original bytes.")
        return value, raw

    def lookup(self, key):
        with self.connect() as db:
            rows = db.execute("SELECT id FROM artifacts WHERE request_key=? ORDER BY rowid DESC", (key,)).fetchall()
        return [self.get(row[0])[0] for row in rows]

    def receipts(self, scope=None):
        with self.connect() as db:
            rows = db.execute("SELECT id FROM artifacts" + (" WHERE scope=?" if scope else ""), (scope,) if scope else ()).fetchall()
        return [self.get(row[0])[0] for row in rows]

    def pin(self, artifact, owner):
        self.get(artifact)
        with self.connect() as db:
            db.execute("INSERT OR IGNORE INTO pins VALUES (?, ?)", (artifact, str(owner)))

    def retention(self):
        with self.connect() as db:
            rows = db.execute("SELECT artifact, owner FROM pins ORDER BY artifact,owner").fetchall()
        pins = {}
        for row in rows:
            pins.setdefault(row[0], []).append(row[1])
        values = self.receipts()
        return {"artifacts": [{"artifact_id": v["artifact_id"], "bytes": v["bytes"], "references": pins.get(v["artifact_id"], [])} for v in values],
                "automatic_eviction": False, "artifact_count": len(values), "logical_bytes": sum(v["bytes"] for v in values)}
