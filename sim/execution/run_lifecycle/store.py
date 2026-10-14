from __future__ import annotations

import json
import os
import tempfile
import time
import uuid
from contextlib import AbstractContextManager, contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable, Iterator

if os.name == "nt":
    import msvcrt
else:
    import fcntl

from .models import (
    RUN_LOCATOR_SCHEMA,
    ExecutionOwner,
    MalformedRunStateError,
    RunEvent,
    RunIdentity,
    RunManifest,
    RunNotFoundError,
    RunPhase,
    RunPolicyError,
    RunState,
    canonical_json_bytes,
    local_host_sha256,
    sha256_json,
    utc_now,
    validate_state_phase,
    validate_transition,
)

MANIFEST_NAME = "run_manifest.json"
EVENTS_NAME = "run_events.jsonl"
OWNER_NAME = "execution_owner.json"
LIFECYCLE_DIR_NAME = "lifecycle"
_MALFORMED_LOCK_GRACE_S = 1.0


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(json.dumps(payload, indent=2, sort_keys=True).encode("utf-8"))
            stream.write(b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except Exception:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MalformedRunStateError(f"Lifecycle JSON is unreadable: {path.name}.") from exc
    if not isinstance(value, dict):
        raise MalformedRunStateError(f"Lifecycle JSON must contain an object: {path.name}.")
    return value


def _process_alive(pid: int) -> bool:
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except (PermissionError, OSError):
        return True
    return True


class _RunLock(AbstractContextManager["_RunLock"]):
    def __init__(self, path: Path, *, timeout_s: float = 10.0) -> None:
        self.path = path
        self.timeout_s = float(timeout_s)
        self.token = str(uuid.uuid4())
        self._owned = False

    def __enter__(self) -> _RunLock:
        deadline = time.monotonic() + self.timeout_s
        payload = canonical_json_bytes(
            {
                "token": self.token,
                "pid": os.getpid(),
                "host_sha256": local_host_sha256(),
                "created_at": utc_now(),
            }
        ) + b"\n"
        while True:
            with self._publication_guard(deadline):
                temporary_descriptor, temporary_name = tempfile.mkstemp(
                    prefix=".run-lock-", dir=self.path.parent
                )
                temporary = Path(temporary_name)
                linked = False
                try:
                    with os.fdopen(temporary_descriptor, "wb") as stream:
                        stream.write(payload)
                        stream.flush()
                        os.fsync(stream.fileno())
                    try:
                        # Publish only a fully written and fsynced lock. Atomic
                        # hard-link creation cannot replace another owner's lock.
                        os.link(temporary, self.path)
                        linked = True
                    except FileExistsError:
                        pass
                finally:
                    try:
                        temporary.unlink()
                    except FileNotFoundError:
                        pass
                reclaimed = not linked and self._reclaim_orphaned_lock()
            if not linked:
                if reclaimed:
                    continue
                if time.monotonic() >= deadline:
                    raise RunPolicyError("Timed out waiting for the lifecycle run lock.") from None
                time.sleep(0.05)
                continue
            _fsync_directory(self.path.parent)
            self._owned = True
            return self

    @contextmanager
    def _publication_guard(self, deadline: float) -> Iterator[None]:
        # Keep this inode permanently: replacing/unlinking a guard lets two
        # processes lock different inodes. OS locks release after a crash.
        path = self.path.with_name(self.path.name + ".guard")
        descriptor = os.open(path, os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0), 0o600)
        locked = False
        try:
            if os.name == "nt" and os.fstat(descriptor).st_size == 0:
                os.write(descriptor, b"\0")
            while not locked:
                try:
                    if os.name == "nt":
                        os.lseek(descriptor, 0, os.SEEK_SET)
                        msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
                    else:
                        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    locked = True
                except OSError:
                    if time.monotonic() >= deadline:
                        raise RunPolicyError("Timed out waiting for the lifecycle run lock.") from None
                    time.sleep(0.05)
            yield
        finally:
            if locked:
                if os.name == "nt":
                    os.lseek(descriptor, 0, os.SEEK_SET)
                    msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    def _reclaim_orphaned_lock(self) -> bool:
        """Remove only a content-stable lock owned by a dead process on this host."""
        try:
            observed = _read_json_object(self.path)
            token = str(observed.get("token", ""))
            pid = int(observed.get("pid", 0))
        except (MalformedRunStateError, TypeError, ValueError):
            return self._reclaim_stable_malformed_lock()
        if not token or pid < 1 or str(observed.get("host_sha256", "")) != local_host_sha256():
            return False
        if _process_alive(pid):
            return False
        try:
            current = _read_json_object(self.path)
        except MalformedRunStateError:
            return False
        if current != observed:
            return False
        try:
            self.path.unlink()
        except FileNotFoundError:
            return True
        _fsync_directory(self.path.parent)
        return True

    def _reclaim_stable_malformed_lock(self) -> bool:
        """Recover a legacy torn lock after a short, content-stable grace period."""

        try:
            before = self.path.stat()
        except FileNotFoundError:
            return True
        if time.time() - before.st_mtime < _MALFORMED_LOCK_GRACE_S:
            return False
        try:
            after = self.path.stat()
        except FileNotFoundError:
            return True
        if (
            before.st_ino != after.st_ino
            or before.st_mtime_ns != after.st_mtime_ns
            or before.st_size != after.st_size
        ):
            return False
        try:
            self.path.unlink()
        except FileNotFoundError:
            return True
        _fsync_directory(self.path.parent)
        return True

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if not self._owned:
            return
        try:
            with self._publication_guard(time.monotonic() + self.timeout_s):
                current = _read_json_object(self.path)
                if str(current.get("token", "")) == self.token:
                    self.path.unlink(missing_ok=True)
        finally:
            self._owned = False



class LifecycleStore:
    """Durable local lifecycle truth plus a non-authoritative run locator."""

    def __init__(
        self,
        *,
        state_root: str | Path,
        allowed_output_roots: Iterable[str | Path],
    ) -> None:
        self.state_root = Path(state_root).expanduser().resolve()
        self.locator_root = self.state_root / "locators"
        self.allowed_output_roots = tuple(
            Path(root).expanduser().resolve() for root in allowed_output_roots
        )
        if not self.allowed_output_roots:
            raise ValueError("At least one authorized lifecycle output root is required.")

    def resolve_output_dir(self, value: str | Path) -> Path:
        raw = Path(value).expanduser()
        if not raw.is_absolute():
            raw = self.allowed_output_roots[0] / raw
        lexical = Path(os.path.abspath(raw))
        self._reject_symlink_path(lexical)
        output = raw.resolve()
        matching_root = next(
            (root for root in self.allowed_output_roots if _is_relative_to(output, root)),
            None,
        )
        if matching_root is None or output == matching_root:
            raise RunPolicyError("Lifecycle output must be a child of an authorized output root.")
        return output

    def prepare_output_dir(self, value: str | Path) -> Path:
        output = self.resolve_output_dir(value)
        if output.exists() and any(output.iterdir()):
            raise FileExistsError("Lifecycle output directory must be new or empty.")
        output.mkdir(parents=True, exist_ok=True)
        lifecycle = output / LIFECYCLE_DIR_NAME
        lifecycle.mkdir(mode=0o700, exist_ok=False)
        return output

    @staticmethod
    def manifest_ref(run_id: str, output_dir: str | Path) -> str:
        digest = sha256_json(
            {"run_id": str(run_id), "output_dir": str(Path(output_dir).resolve())}
        )
        return f"oel-run-ref:{digest}"

    def create(self, identity: RunIdentity) -> RunManifest:
        output = self.resolve_output_dir(identity.output_dir)
        lifecycle = output / LIFECYCLE_DIR_NAME
        if not lifecycle.is_dir():
            raise RunPolicyError("Lifecycle output was not prepared before manifest creation.")
        manifest_path = lifecycle / MANIFEST_NAME
        events_path = lifecycle / EVENTS_NAME
        lock_path = lifecycle / ".run.lock"
        created = utc_now()
        manifest = RunManifest(
            identity=identity,
            state=RunState.ACCEPTED,
            sequence=1,
            phase=RunPhase.ACCEPTED,
            created_at=created,
            updated_at=created,
            execution={"completed": False, "status": "accepted"},
            artifacts={"complete": False, "files": []},
            review={"available": False, "evidence_complete": "not_evaluated"},
        ).with_digest()
        with _RunLock(lock_path):
            if manifest_path.exists() or events_path.exists():
                raise FileExistsError("Lifecycle artifacts already exist in the output directory.")
            _atomic_write_json(manifest_path, manifest.to_dict())
            self._append_event(events_path, self._event_for(manifest, event="accepted"))
            self._write_locator(manifest_path, manifest)
        return manifest

    def transition(
        self,
        run_id: str,
        requested: RunState,
        *,
        phase: RunPhase,
        event: str,
        execution: dict[str, Any] | None = None,
        artifacts: dict[str, Any] | None = None,
        review: dict[str, Any] | None = None,
        error: dict[str, Any] | None = None,
    ) -> RunManifest:
        manifest_path = self._manifest_path(run_id)
        lifecycle = manifest_path.parent
        with _RunLock(lifecycle / ".run.lock"):
            current = self._read_manifest_path(manifest_path, expected_run_id=run_id)
            validate_transition(current.state, requested)
            validate_state_phase(requested, phase)
            now = utc_now()
            terminal_at = now if requested in {
                RunState.COMPLETED,
                RunState.FAILED,
                RunState.CANCELLED,
                RunState.INTERRUPTED,
            } else None
            started_at = current.started_at
            if requested in {RunState.STARTING, RunState.RUNNING} and started_at is None:
                started_at = now
            updated = replace(
                current,
                state=requested,
                sequence=current.sequence + 1,
                phase=phase,
                updated_at=now,
                started_at=started_at,
                terminal_at=terminal_at,
                execution=dict(current.execution if execution is None else execution),
                artifacts=dict(current.artifacts if artifacts is None else artifacts),
                review=dict(current.review if review is None else review),
                error=None if error is None else dict(error),
                manifest_sha256="",
            ).with_digest()
            _atomic_write_json(manifest_path, updated.to_dict())
            self._append_event(lifecycle / EVENTS_NAME, self._event_for(updated, event=event))
            self._write_locator(manifest_path, updated)
            return updated

    def read_manifest(self, run_id: str) -> RunManifest:
        return self._read_manifest_path(self._manifest_path(run_id), expected_run_id=run_id)

    def create_owner(
        self,
        run_id: str,
        *,
        owner_token: str,
        pid: int,
        host_sha256: str,
        heartbeat_interval_s: float,
        stale_after_s: float,
    ) -> ExecutionOwner:
        manifest_path = self._manifest_path(run_id)
        now = utc_now()
        owner = ExecutionOwner(
            run_id=run_id,
            owner_token=owner_token,
            pid=int(pid),
            host_sha256=host_sha256,
            status="active",
            started_at=now,
            updated_at=now,
            heartbeat_interval_s=float(heartbeat_interval_s),
            stale_after_s=float(stale_after_s),
        ).with_digest()
        owner_path = manifest_path.parent / OWNER_NAME
        with _RunLock(owner_path.parent / ".run.lock"):
            if owner_path.exists():
                raise FileExistsError("Lifecycle execution owner already exists.")
            _atomic_write_json(owner_path, owner.to_dict())
        return owner

    def heartbeat_owner(
        self,
        run_id: str,
        *,
        owner_token: str,
        status: str = "active",
    ) -> ExecutionOwner:
        owner_path = self._manifest_path(run_id).parent / OWNER_NAME
        with _RunLock(owner_path.parent / ".run.lock"):
            current = ExecutionOwner.from_dict(_read_json_object(owner_path))
            if current.run_id != run_id or current.owner_token != owner_token:
                raise RunPolicyError("Lifecycle execution-owner identity does not match.")
            updated = replace(
                current,
                status=status,
                updated_at=utc_now(),
                owner_sha256="",
            ).with_digest()
            _atomic_write_json(owner_path, updated.to_dict())
        return updated

    def read_owner(self, run_id: str) -> ExecutionOwner | None:
        owner_path = self._manifest_path(run_id).parent / OWNER_NAME
        if not owner_path.is_file():
            return None
        owner = ExecutionOwner.from_dict(_read_json_object(owner_path))
        if owner.run_id != run_id:
            raise MalformedRunStateError("Execution-owner run identity does not match.")
        return owner

    def read_events(self, run_id: str, *, after_sequence: int = 0, limit: int = 100) -> tuple[RunEvent, ...]:
        if limit < 1 or limit > 1000:
            raise ValueError("Lifecycle event limit must be between 1 and 1000.")
        manifest_path = self._manifest_path(run_id)
        events_path = manifest_path.parent / EVENTS_NAME
        try:
            lines = events_path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise MalformedRunStateError("Lifecycle event log is unreadable.") from exc
        events: list[RunEvent] = []
        previous = 0
        for line in lines:
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise MalformedRunStateError("Lifecycle event log contains malformed JSON.") from exc
            if not isinstance(value, dict):
                raise MalformedRunStateError("Lifecycle event log contains a non-object record.")
            item = RunEvent.from_dict(value)
            if item.run_id != run_id or item.sequence != previous + 1:
                raise MalformedRunStateError("Lifecycle event ordering or run identity is invalid.")
            previous = item.sequence
            if item.sequence > int(after_sequence):
                events.append(item)
                if len(events) >= limit:
                    break
        return tuple(events)

    def _manifest_path(self, run_id: str) -> Path:
        try:
            normalized = str(uuid.UUID(str(run_id)))
        except ValueError as exc:
            raise RunNotFoundError("Run identity was not found.") from exc
        locator_path = self.locator_root / f"{normalized}.json"
        if not locator_path.is_file():
            raise RunNotFoundError("Run identity was not found.")
        locator = _read_json_object(locator_path)
        if locator.get("schema_version") != RUN_LOCATOR_SCHEMA:
            raise MalformedRunStateError("Run locator schema_version is unsupported.")
        unsigned = dict(locator)
        recorded = str(unsigned.pop("locator_sha256", ""))
        if recorded != sha256_json(unsigned):
            raise MalformedRunStateError("Run locator digest does not match its canonical content.")
        if str(locator.get("run_id", "")) != normalized:
            raise MalformedRunStateError("Run locator identity does not match its filename.")
        manifest_path = Path(str(locator.get("manifest_path", ""))).expanduser().resolve()
        if not any(_is_relative_to(manifest_path, root) for root in self.allowed_output_roots):
            raise MalformedRunStateError("Run locator points outside authorized output roots.")
        if manifest_path.name != MANIFEST_NAME or manifest_path.parent.name != LIFECYCLE_DIR_NAME:
            raise MalformedRunStateError("Run locator does not point to a lifecycle manifest.")
        return manifest_path

    def _read_manifest_path(self, path: Path, *, expected_run_id: str) -> RunManifest:
        manifest = RunManifest.from_dict(_read_json_object(path))
        if manifest.identity.run_id != str(expected_run_id):
            raise MalformedRunStateError("Run manifest identity does not match the requested run.")
        expected_ref = self.manifest_ref(manifest.identity.run_id, manifest.identity.output_dir)
        if manifest.identity.manifest_ref != expected_ref:
            raise MalformedRunStateError("Run manifest reference does not match its authorized output identity.")
        if path != Path(manifest.identity.output_dir).resolve() / LIFECYCLE_DIR_NAME / MANIFEST_NAME:
            raise MalformedRunStateError("Run manifest output identity does not match its locator path.")
        return manifest

    def _write_locator(self, manifest_path: Path, manifest: RunManifest) -> None:
        unsigned = {
            "schema_version": RUN_LOCATOR_SCHEMA,
            "run_id": manifest.identity.run_id,
            "manifest_ref": manifest.identity.manifest_ref,
            "manifest_path": str(manifest_path),
            "manifest_sha256": manifest.manifest_sha256,
            "updated_at": manifest.updated_at,
        }
        locator = {**unsigned, "locator_sha256": sha256_json(unsigned)}
        _atomic_write_json(self.locator_root / f"{manifest.identity.run_id}.json", locator)

    @staticmethod
    def _append_event(path: Path, event: RunEvent) -> None:
        payload = canonical_json_bytes(event.to_dict()) + b"\n"
        descriptor = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
        with os.fdopen(descriptor, "ab") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        _fsync_directory(path.parent)

    @staticmethod
    def _event_for(manifest: RunManifest, *, event: str) -> RunEvent:
        return RunEvent(
            run_id=manifest.identity.run_id,
            sequence=manifest.sequence,
            event=event,
            state=manifest.state,
            phase=manifest.phase,
            emitted_at=manifest.updated_at,
            manifest_sha256=manifest.manifest_sha256,
        )

    @staticmethod
    def _reject_symlink_path(path: Path) -> None:
        current = Path(path.anchor)
        for part in path.parts[1:]:
            current = current / part
            if current.exists() and current.is_symlink():
                raise RunPolicyError("Lifecycle output paths cannot contain symlink components.")
