from __future__ import annotations

import json
import os
import stat
import time
import uuid
from contextlib import AbstractContextManager, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Iterator

if os.name == "nt":  # pragma: no cover - secure lifecycle paths fail closed on Windows
    fcntl = None
else:
    import fcntl

from sim.platform_compat import canonicalize_platform_compatibility_root
from sim.security.input_reads import (
    DescriptorStore,
    SecureReadError,
    SecureReadRoot,
    SecureReadUnavailable,
)
from sim.security.output_writes import SecureWriteError, SecureWriteRoot, SecureWriteUnavailable

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
_MAX_LIFECYCLE_RECORD_BYTES = 4_000_000


def _canonical_path(value: str | Path) -> Path:
    """Return an absolute lexical path without resolving any symlink."""

    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    path = canonicalize_platform_compatibility_root(path)
    if any(part in {"", ".", ".."} for part in path.parts[1:]):
        raise RunPolicyError("Lifecycle paths must be normalized and contain no traversal components.")
    return Path(os.path.abspath(os.fspath(path)))


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _reject_symlink_path(path: Path) -> None:
    """Reject every symlink component before a path is used as a route."""

    current = Path(path.anchor)
    for part in path.parts[1:]:
        current = current / part
        try:
            if current.is_symlink():
                raise RunPolicyError("Lifecycle paths cannot contain symlink components.")
        except OSError as exc:
            raise RunPolicyError("Lifecycle path components could not be inspected safely.") from exc


def _directory_identity(fd: int) -> tuple[int, int]:
    metadata = os.fstat(fd)
    if not stat.S_ISDIR(metadata.st_mode):
        raise RunPolicyError("Lifecycle secure root is not a directory.")
    return int(metadata.st_dev), int(metadata.st_ino)


def _visible_directory_identity(path: Path) -> tuple[int, int]:
    try:
        metadata = os.stat(path, follow_symlinks=False)
    except OSError as exc:
        raise RunPolicyError("Lifecycle root identity could not be verified safely.") from exc
    if not stat.S_ISDIR(metadata.st_mode):
        raise RunPolicyError("Lifecycle root identity is no longer a directory.")
    return int(metadata.st_dev), int(metadata.st_ino)


def _process_alive(pid: int) -> bool:
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except (PermissionError, OSError):
        return True
    return True


@dataclass(frozen=True)
class _BoundPath:
    store: DescriptorStore
    relative: str


_ACTIVE_BOUND_PATHS: ContextVar[dict[Path, _BoundPath] | None] = ContextVar(
    "oel_active_lifecycle_bound_paths",
    default=None,
)


@contextmanager
def _bind_path(path: Path, store: DescriptorStore, relative: str) -> Iterator[None]:
    current = _ACTIVE_BOUND_PATHS.get()
    mapping = {} if current is None else dict(current)
    mapping[_canonical_path(path)] = _BoundPath(store=store, relative=relative)
    token = _ACTIVE_BOUND_PATHS.set(mapping)
    try:
        yield
    finally:
        _ACTIVE_BOUND_PATHS.reset(token)


def _bound_path(path: Path) -> _BoundPath:
    mapping = _ACTIVE_BOUND_PATHS.get() or {}
    bound = mapping.get(_canonical_path(path))
    if bound is None:
        raise RunPolicyError(
            "Lifecycle record access was not bound to a retained directory descriptor."
        )
    return bound


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, indent=2, sort_keys=True).encode("utf-8") + b"\n"


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write one bound JSON record while retaining the legacy helper signature."""

    bound = _bound_path(path)
    encoded = _json_bytes(payload)
    try:
        try:
            current = bound.store.stat(bound.relative)
        except FileNotFoundError:
            bound.store.write_new(bound.relative, encoded, mode=0o600)
        else:
            bound.store.write_atomic(
                bound.relative,
                encoded,
                mode=0o600,
                expected_identity=(int(current.st_dev), int(current.st_ino)),
            )
    except FileExistsError:
        # Exclusive creation is part of the lifecycle race contract.
        raise
    except (SecureReadError, SecureWriteError, OSError) as exc:
        raise RunPolicyError("Lifecycle JSON publication could not be bound safely.") from exc


def _fsync_directory(path: Path) -> None:
    """Synchronize a bound directory without reopening a mutable pathname."""

    bound = _bound_path(path)
    try:
        with bound.store.root.open_directory(".") as descriptor:
            os.fsync(descriptor)
    except (SecureReadError, OSError) as exc:
        raise RunPolicyError("Lifecycle directory durability could not be verified.") from exc


def _read_json_object(path: Path) -> dict[str, Any]:
    """Read a JSON object through the active descriptor binding."""

    bound = _bound_path(path)
    try:
        raw = bound.store.read_bytes(bound.relative)
        value = json.loads(raw.decode("utf-8"))
    except FileNotFoundError as exc:
        raise MalformedRunStateError(f"Lifecycle JSON is missing: {path.name}.") from exc
    except (SecureReadError, OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise MalformedRunStateError(f"Lifecycle JSON is unreadable: {path.name}.") from exc
    if not isinstance(value, dict):
        raise MalformedRunStateError(f"Lifecycle JSON must contain an object: {path.name}.")
    return value


def _record_exists(store: DescriptorStore, relative: str) -> bool:
    try:
        store.stat(relative)
    except FileNotFoundError:
        return False
    return True


def _append_bound_bytes(bound: _BoundPath, payload: bytes) -> None:
    relative = Path(bound.relative)
    parts = relative.parts
    if not parts or any(part in {"", ".", ".."} for part in parts):
        raise RunPolicyError("Lifecycle event paths must be normalized and relative.")
    parent_relative = "." if len(parts) == 1 else str(Path(*parts[:-1]))
    descriptor: int | None = None
    try:
        with bound.store.root.open_directory(parent_relative) as parent_descriptor:
            descriptor = os.open(
                parts[-1],
                os.O_WRONLY
                | os.O_APPEND
                | os.O_CREAT
                | getattr(os, "O_NOFOLLOW", 0),
                0o600,
                dir_fd=parent_descriptor,
            )
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode):
                raise RunPolicyError("Lifecycle event log must be a regular file.")
            if metadata.st_size + len(payload) > _MAX_LIFECYCLE_RECORD_BYTES:
                raise RunPolicyError("Lifecycle event log exceeds its bounded record budget.")
            view = memoryview(payload)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise RunPolicyError("Lifecycle event publication made no progress.")
                view = view[written:]
            os.fsync(descriptor)
            os.fsync(parent_descriptor)
    except (SecureReadError, OSError) as exc:
        raise RunPolicyError("Lifecycle event publication could not be bound safely.") from exc
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass


def _append_event(path: Path, event: RunEvent) -> None:
    _append_bound_bytes(_bound_path(path), canonical_json_bytes(event.to_dict()) + b"\n")


@dataclass
class _RootBinding:
    path: Path
    writer: SecureWriteRoot
    reader: SecureReadRoot
    store: DescriptorStore
    identity: tuple[int, int]


@dataclass
class _LifecycleHandle:
    path: Path
    binding: _RootBinding
    reader: SecureReadRoot
    store: DescriptorStore
    identity: tuple[int, int]


def _nearest_existing_ancestor(path: Path) -> Path:
    candidate = path
    while not candidate.exists():
        parent = candidate.parent
        if parent == candidate:
            raise RunPolicyError("Lifecycle secure root has no existing approved ancestor.")
        candidate = parent
    _reject_symlink_path(candidate)
    if not candidate.is_dir():
        raise RunPolicyError("Lifecycle secure root ancestor is not a directory.")
    return candidate


def _open_root_binding(path: Path) -> _RootBinding:
    path = _canonical_path(path)
    _reject_symlink_path(path)
    approved = _nearest_existing_ancestor(path)
    writer: SecureWriteRoot | None = None
    reader: SecureReadRoot | None = None
    try:
        writer = SecureWriteRoot.open(path, approved_roots=(approved,), create=True)
        writer_identity = _directory_identity(writer.fd)
        reader = SecureReadRoot.open(path, approved_roots=(writer.approved_root,))
        reader_identity = _directory_identity(reader.fd)
        if writer_identity != reader_identity or _visible_directory_identity(path) != writer_identity:
            raise RunPolicyError("Lifecycle root identity changed while it was being opened.")
        return _RootBinding(
            path=path,
            writer=writer,
            reader=reader,
            store=DescriptorStore(reader, max_record_bytes=_MAX_LIFECYCLE_RECORD_BYTES),
            identity=writer_identity,
        )
    except (SecureReadUnavailable, SecureWriteUnavailable):
        if reader is not None:
            reader.close()
        if writer is not None:
            writer.close()
        raise
    except (SecureReadError, SecureWriteError, OSError) as exc:
        if reader is not None:
            reader.close()
        if writer is not None:
            writer.close()
        raise RunPolicyError("Lifecycle secure root could not be opened safely.") from exc
    except BaseException:
        if reader is not None:
            reader.close()
        if writer is not None:
            writer.close()
        raise


class _RunLock(AbstractContextManager["_RunLock"]):
    """Descriptor-bound lifecycle lock preserving the existing lock schema."""

    def __init__(self, path: Path, *, timeout_s: float = 10.0) -> None:
        self.path = _canonical_path(path)
        _reject_symlink_path(self.path.parent)
        try:
            self._root = SecureReadRoot.open(
                self.path.parent,
                approved_roots=(self.path.parent,),
            )
        except SecureReadUnavailable:
            raise
        except (SecureReadError, OSError) as exc:
            raise RunPolicyError("Lifecycle lock parent could not be retained safely.") from exc
        self._store = DescriptorStore(self._root, max_record_bytes=_MAX_LIFECYCLE_RECORD_BYTES)
        self._parent_identity = _directory_identity(self._root.fd)
        self.timeout_s = float(timeout_s)
        self.token = str(uuid.uuid4())
        self._lock_name = self.path.name
        self._guard_name = f"{self._lock_name}.guard"
        self._owned = False
        self._closed = False

    def _assert_parent_visible(self) -> None:
        if _visible_directory_identity(self.path.parent) != self._parent_identity:
            raise RunPolicyError("Lifecycle lock parent identity changed; refusing replacement.")

    def _close(self) -> None:
        if not self._closed:
            self._root.close()
            self._closed = True

    @contextmanager
    def _publication_guard(self, deadline: float) -> Iterator[None]:
        if fcntl is None:
            self._close()
            raise SecureReadUnavailable("Lifecycle lock guards require POSIX fcntl support.")
        self._assert_parent_visible()
        descriptor: int | None = None
        locked = False
        try:
            while True:
                retry_guard_open = False
                with self._root.open_directory(".") as parent_descriptor:
                    try:
                        descriptor = os.open(
                            self._guard_name,
                            os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
                            0o600,
                            dir_fd=parent_descriptor,
                        )
                    except FileNotFoundError:
                        # Treat one transient name-open ENOENT as retryable while
                        # retaining the parent descriptor. Never fall back to a
                        # pathname reopen or admit the lock without the guard.
                        retry_guard_open = True
                    if retry_guard_open:
                        pass
                    else:
                        before = os.fstat(descriptor)
                        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
                            raise RunPolicyError("Lifecycle lock guard must be a single-link regular file.")
                        while not locked:
                            try:
                                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                                locked = True
                            except OSError:
                                if time.monotonic() >= deadline:
                                    raise RunPolicyError(
                                        "Timed out waiting for the lifecycle run lock."
                                    ) from None
                                time.sleep(0.05)
                        after = os.fstat(descriptor)
                        named = os.stat(
                            self._guard_name,
                            dir_fd=parent_descriptor,
                            follow_symlinks=False,
                        )
                        if (
                            not stat.S_ISREG(after.st_mode)
                            or after.st_nlink != 1
                            or (int(after.st_dev), int(after.st_ino))
                            != (int(before.st_dev), int(before.st_ino))
                            or (int(named.st_dev), int(named.st_ino))
                            != (int(after.st_dev), int(after.st_ino))
                        ):
                            raise RunPolicyError("Lifecycle lock guard identity changed during acquisition.")
                        yield
                        break
                if retry_guard_open:
                    if time.monotonic() >= deadline:
                        raise RunPolicyError(
                            "Timed out waiting for the lifecycle run lock."
                        ) from None
                    time.sleep(0.01)
        except (SecureReadError, OSError) as exc:
            if isinstance(exc, RunPolicyError):
                raise
            raise RunPolicyError("Lifecycle lock publication could not be bound safely.") from exc
        finally:
            if locked and descriptor is not None:
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
                except OSError:
                    pass
            if descriptor is not None:
                try:
                    os.close(descriptor)
                except OSError:
                    pass
        self._assert_parent_visible()

    def _read_lock_record(self) -> tuple[bytes, dict[str, Any], tuple[int, int]]:
        try:
            before = self._store.stat(self._lock_name)
            raw = self._store.read_bytes(self._lock_name)
        except FileNotFoundError as exc:
            raise RunNotFoundError("Lifecycle lock is absent.") from exc
        except SecureReadError as exc:
            raise RunPolicyError("Lifecycle lock is not a safe regular file.") from exc
        try:
            value = json.loads(raw.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise MalformedRunStateError("Lifecycle lock contains malformed JSON.") from exc
        if not isinstance(value, dict):
            raise MalformedRunStateError("Lifecycle lock must contain an object.")
        try:
            after = self._store.stat(self._lock_name)
        except FileNotFoundError as exc:
            raise RunPolicyError("Lifecycle lock disappeared during verification.") from exc
        identity = (int(before.st_dev), int(before.st_ino))
        if identity != (int(after.st_dev), int(after.st_ino)):
            raise RunPolicyError("Lifecycle lock identity changed during verification.")
        return raw, value, identity

    def _remove_lock(self, *, expected_identity: tuple[int, int] | None = None) -> bool:
        try:
            with self._root.open_directory(".") as parent_descriptor:
                try:
                    descriptor = os.open(
                        self._lock_name,
                        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                        dir_fd=parent_descriptor,
                    )
                except FileNotFoundError:
                    return True
                try:
                    current = os.fstat(descriptor)
                    identity = (int(current.st_dev), int(current.st_ino))
                    if (
                        not stat.S_ISREG(current.st_mode)
                        or current.st_nlink != 1
                        or (expected_identity is not None and identity != expected_identity)
                    ):
                        raise RunPolicyError("Lifecycle lock identity changed before removal.")
                finally:
                    os.close(descriptor)
                os.unlink(self._lock_name, dir_fd=parent_descriptor)
                os.fsync(parent_descriptor)
                return True
        except SecureReadError as exc:
            raise RunPolicyError("Lifecycle lock removal could not be bound safely.") from exc
        except OSError as exc:
            raise RunPolicyError("Lifecycle lock removal failed safely.") from exc

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
        try:
            while True:
                linked = False
                with self._publication_guard(deadline):
                    try:
                        self._store.write_new(self._lock_name, payload, mode=0o600)
                        linked = True
                    except FileExistsError:
                        pass
                    reclaimed = not linked and self._reclaim_orphaned_lock()
                if linked:
                    self._assert_parent_visible()
                    self._owned = True
                    return self
                if reclaimed:
                    continue
                if time.monotonic() >= deadline:
                    raise RunPolicyError("Timed out waiting for the lifecycle run lock.") from None
                time.sleep(0.05)
        except BaseException:
            self._close()
            raise

    def _reclaim_orphaned_lock(self) -> bool:
        """Remove only a content-stable lock owned by a dead process on this host."""

        try:
            observed_raw, observed, observed_identity = self._read_lock_record()
        except RunNotFoundError:
            return True
        except MalformedRunStateError:
            return self._reclaim_stable_malformed_lock()
        token = str(observed.get("token", ""))
        try:
            pid = int(observed.get("pid", 0))
        except (TypeError, ValueError):
            return self._reclaim_stable_malformed_lock()
        if not token or pid < 1 or str(observed.get("host_sha256", "")) != local_host_sha256():
            return False
        if _process_alive(pid):
            return False
        try:
            current_raw, _current, current_identity = self._read_lock_record()
        except RunNotFoundError:
            return True
        except MalformedRunStateError:
            return False
        if current_raw != observed_raw or current_identity != observed_identity:
            return False
        return self._remove_lock(expected_identity=observed_identity)

    def _reclaim_stable_malformed_lock(self) -> bool:
        """Recover a legacy torn lock after a short, content-stable grace period."""

        try:
            before = self._store.stat(self._lock_name)
        except FileNotFoundError:
            return True
        except SecureReadError as exc:
            raise RunPolicyError("Malformed lifecycle lock metadata is unsafe.") from exc
        if time.time() - before.st_mtime < _MALFORMED_LOCK_GRACE_S:
            return False
        try:
            after = self._store.stat(self._lock_name)
        except FileNotFoundError:
            return True
        except SecureReadError as exc:
            raise RunPolicyError("Malformed lifecycle lock metadata is unsafe.") from exc
        if (
            before.st_dev != after.st_dev
            or before.st_ino != after.st_ino
            or before.st_mtime_ns != after.st_mtime_ns
            or before.st_size != after.st_size
        ):
            return False
        return self._remove_lock(
            expected_identity=(int(after.st_dev), int(after.st_ino))
        )

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if not self._owned:
            self._close()
            return
        try:
            with self._publication_guard(time.monotonic() + self.timeout_s):
                try:
                    _raw, current, current_identity = self._read_lock_record()
                except RunNotFoundError:
                    return
                if str(current.get("token", "")) == self.token:
                    self._remove_lock(expected_identity=current_identity)
        finally:
            self._owned = False
            self._close()


class LifecycleStore:
    """Durable local lifecycle truth plus a non-authoritative run locator."""

    def __init__(
        self,
        *,
        state_root: str | Path,
        allowed_output_roots: Iterable[str | Path],
    ) -> None:
        self.state_root = _canonical_path(state_root)
        self.allowed_output_roots = tuple(_canonical_path(root) for root in allowed_output_roots)
        if not self.allowed_output_roots:
            raise ValueError("At least one authorized lifecycle output root is required.")
        self._lifecycle_handles: dict[Path, _LifecycleHandle] = {}
        self._closed = False
        self._state = _open_root_binding(self.state_root)
        self.locator_root = self.state_root / "locators"
        try:
            self._state.writer.mkdir("locators")
            self._locator_reader = SecureReadRoot.open(
                self.locator_root,
                approved_roots=(self._state.writer.approved_root,),
            )
            self._locator_identity = _directory_identity(self._locator_reader.fd)
            if _visible_directory_identity(self.locator_root) != self._locator_identity:
                raise RunPolicyError("Lifecycle locator root identity changed while opening.")
            self._locator = _RootBinding(
                path=self.locator_root,
                writer=self._state.writer,
                reader=self._locator_reader,
                store=DescriptorStore(
                    self._locator_reader,
                    max_record_bytes=_MAX_LIFECYCLE_RECORD_BYTES,
                ),
                identity=self._locator_identity,
            )
            opened_outputs: list[_RootBinding] = []
            try:
                for root in self.allowed_output_roots:
                    opened_outputs.append(_open_root_binding(root))
            except BaseException:
                for binding in opened_outputs:
                    binding.reader.close()
                    binding.writer.close()
                raise
            self._outputs = tuple(opened_outputs)
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for handle in self._lifecycle_handles.values():
            handle.reader.close()
        self._lifecycle_handles.clear()
        locator = getattr(self, "_locator_reader", None)
        if locator is not None:
            locator.close()
        state = getattr(self, "_state", None)
        if state is not None:
            state.reader.close()
            state.writer.close()
        for binding in getattr(self, "_outputs", ()):
            binding.reader.close()
            binding.writer.close()

    def __enter__(self) -> LifecycleStore:
        if self._closed:
            raise RunPolicyError("Lifecycle store is closed.")
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def __del__(self) -> None:  # pragma: no cover - best-effort descriptor cleanup
        try:
            self.close()
        except BaseException:
            pass

    @staticmethod
    def _assert_binding_visible(binding: _RootBinding) -> None:
        if _visible_directory_identity(binding.path) != binding.identity:
            raise RunPolicyError("Lifecycle root identity changed; refusing replacement.")

    def _assert_locator_visible(self) -> None:
        self._assert_binding_visible(self._state)
        try:
            visible_identity = _visible_directory_identity(self.locator_root)
        except RunPolicyError as exc:
            raise RunPolicyError(
                "Lifecycle locator root identity changed; refusing replacement."
            ) from exc
        if visible_identity != self._locator_identity:
            raise RunPolicyError("Lifecycle locator root identity changed; refusing replacement.")

    def _output_binding(self, path: Path) -> tuple[_RootBinding, Path]:
        path = _canonical_path(path)
        _reject_symlink_path(path)
        matches = [
            (binding, path.relative_to(binding.path))
            for binding in self._outputs
            if _is_relative_to(path, binding.path) and path != binding.path
        ]
        if not matches:
            raise RunPolicyError("Lifecycle output must be a child of an authorized output root.")
        return max(matches, key=lambda item: len(item[0].path.parts))

    def _lifecycle_handle(self, path: Path) -> _LifecycleHandle:
        path = _canonical_path(path)
        binding, _relative = self._output_binding(path)
        self._assert_binding_visible(binding)
        expected = self._lifecycle_handles.get(path)
        if expected is not None:
            if _visible_directory_identity(path) != expected.identity:
                raise RunPolicyError("Lifecycle directory identity changed; refusing replacement.")
            return expected
        reader: SecureReadRoot | None = None
        try:
            reader = SecureReadRoot.open(path, approved_roots=(binding.writer.approved_root,))
            identity = _directory_identity(reader.fd)
            visible = _visible_directory_identity(path)
            if identity != visible:
                raise RunPolicyError("Lifecycle directory identity changed while opening.")
            handle = _LifecycleHandle(
                path=path,
                binding=binding,
                reader=reader,
                store=DescriptorStore(
                    reader,
                    max_record_bytes=_MAX_LIFECYCLE_RECORD_BYTES,
                ),
                identity=identity,
            )
            self._lifecycle_handles[path] = handle
            return handle
        except BaseException:
            if reader is not None:
                reader.close()
            raise

    @contextmanager
    def _lifecycle_scope(self, path: Path) -> Iterator[DescriptorStore]:
        handle = self._lifecycle_handle(path)
        self._assert_binding_visible(handle.binding)
        if _visible_directory_identity(handle.path) != handle.identity:
            raise RunPolicyError("Lifecycle directory identity changed; refusing replacement.")
        try:
            yield handle.store
        finally:
            self._assert_binding_visible(handle.binding)
            if _visible_directory_identity(handle.path) != handle.identity:
                raise RunPolicyError("Lifecycle directory identity changed during use.")

    @contextmanager
    def _locator_scope(self) -> Iterator[DescriptorStore]:
        self._assert_locator_visible()
        try:
            yield self._locator.store
        finally:
            self._assert_locator_visible()

    def resolve_output_dir(self, value: str | Path) -> Path:
        raw = Path(value).expanduser()
        if not raw.is_absolute():
            raw = self.allowed_output_roots[0] / raw
        output = _canonical_path(raw)
        binding, _relative = self._output_binding(output)
        self._assert_binding_visible(binding)
        return output

    def prepare_output_dir(self, value: str | Path) -> Path:
        output = self.resolve_output_dir(value)
        binding, relative = self._output_binding(output)
        output_writer: SecureWriteRoot | None = None
        try:
            output_writer = binding.writer.child(relative, create=True)
            if os.listdir(output_writer.fd):
                raise FileExistsError("Lifecycle output directory must be new or empty.")
            try:
                os.mkdir(LIFECYCLE_DIR_NAME, mode=0o700, dir_fd=output_writer.fd)
            except FileExistsError:
                raise FileExistsError("Lifecycle output directory must be new or empty.") from None
            os.fsync(output_writer.fd)
        except FileExistsError:
            raise
        except (SecureWriteError, OSError) as exc:
            raise RunPolicyError("Lifecycle output could not be prepared through its retained root.") from exc
        finally:
            if output_writer is not None:
                output_writer.close()
        self._assert_binding_visible(binding)
        return output

    @staticmethod
    def manifest_ref(run_id: str, output_dir: str | Path) -> str:
        digest = sha256_json(
            {"run_id": str(run_id), "output_dir": str(_canonical_path(output_dir))}
        )
        return f"oel-run-ref:{digest}"

    def create(self, identity: RunIdentity) -> RunManifest:
        output = self.resolve_output_dir(identity.output_dir)
        lifecycle = output / LIFECYCLE_DIR_NAME
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
        with self._lifecycle_scope(lifecycle) as store:
            with _RunLock(lock_path):
                if _record_exists(store, MANIFEST_NAME) or _record_exists(store, EVENTS_NAME):
                    raise FileExistsError("Lifecycle artifacts already exist in the output directory.")
                with _bind_path(manifest_path, store, MANIFEST_NAME):
                    _atomic_write_json(manifest_path, manifest.to_dict())
                with _bind_path(events_path, store, EVENTS_NAME):
                    _append_event(events_path, self._event_for(manifest, event="accepted"))
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
        events_path = lifecycle / EVENTS_NAME
        with self._lifecycle_scope(lifecycle) as store:
            with _RunLock(lifecycle / ".run.lock"):
                with _bind_path(manifest_path, store, MANIFEST_NAME), _bind_path(
                    events_path, store, EVENTS_NAME
                ):
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
                    _append_event(events_path, self._event_for(updated, event=event))
                self._write_locator(manifest_path, updated)
                return updated

    def read_manifest(self, run_id: str) -> RunManifest:
        manifest_path = self._manifest_path(run_id)
        with self._lifecycle_scope(manifest_path.parent) as store:
            with _bind_path(manifest_path, store, MANIFEST_NAME):
                return self._read_manifest_path(manifest_path, expected_run_id=run_id)

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
        owner_path = manifest_path.parent / OWNER_NAME
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
        with self._lifecycle_scope(owner_path.parent) as store:
            with _RunLock(owner_path.parent / ".run.lock"):
                if _record_exists(store, OWNER_NAME):
                    raise FileExistsError("Lifecycle execution owner already exists.")
                with _bind_path(owner_path, store, OWNER_NAME):
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
        with self._lifecycle_scope(owner_path.parent) as store:
            with _RunLock(owner_path.parent / ".run.lock"):
                with _bind_path(owner_path, store, OWNER_NAME):
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
        with self._lifecycle_scope(owner_path.parent) as store:
            if not _record_exists(store, OWNER_NAME):
                return None
            with _bind_path(owner_path, store, OWNER_NAME):
                owner = ExecutionOwner.from_dict(_read_json_object(owner_path))
        if owner.run_id != run_id:
            raise MalformedRunStateError("Execution-owner run identity does not match.")
        return owner

    def read_events(
        self,
        run_id: str,
        *,
        after_sequence: int = 0,
        limit: int = 100,
    ) -> tuple[RunEvent, ...]:
        if limit < 1 or limit > 1000:
            raise ValueError("Lifecycle event limit must be between 1 and 1000.")
        manifest_path = self._manifest_path(run_id)
        events_path = manifest_path.parent / EVENTS_NAME
        with self._lifecycle_scope(events_path.parent) as store:
            with _bind_path(events_path, store, EVENTS_NAME):
                try:
                    raw = store.read_bytes(EVENTS_NAME)
                    lines = raw.decode("utf-8").splitlines()
                except (FileNotFoundError, SecureReadError, UnicodeError) as exc:
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
        with self._locator_scope() as locator_store:
            if not _record_exists(locator_store, locator_path.name):
                raise RunNotFoundError("Run identity was not found.")
            with _bind_path(locator_path, locator_store, locator_path.name):
                locator = _read_json_object(locator_path)
        if locator.get("schema_version") != RUN_LOCATOR_SCHEMA:
            raise MalformedRunStateError("Run locator schema_version is unsupported.")
        unsigned = dict(locator)
        recorded = str(unsigned.pop("locator_sha256", ""))
        try:
            expected = sha256_json(unsigned)
        except (TypeError, ValueError) as exc:
            raise MalformedRunStateError("Run locator contains non-canonical JSON values.") from exc
        if recorded != expected:
            raise MalformedRunStateError("Run locator digest does not match its canonical content.")
        if str(locator.get("run_id", "")) != normalized:
            raise MalformedRunStateError("Run locator identity does not match its filename.")
        try:
            manifest_path = _canonical_path(str(locator.get("manifest_path", "")))
            _reject_symlink_path(manifest_path)
        except RunPolicyError as exc:
            raise MalformedRunStateError("Run locator contains an unsafe manifest path.") from exc
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
            raise MalformedRunStateError(
                "Run manifest reference does not match its authorized output identity."
            )
        expected_path = (
            _canonical_path(manifest.identity.output_dir)
            / LIFECYCLE_DIR_NAME
            / MANIFEST_NAME
        )
        if _canonical_path(path) != expected_path:
            raise MalformedRunStateError("Run manifest output identity does not match its locator path.")
        return manifest

    def _write_locator(self, manifest_path: Path, manifest: RunManifest) -> None:
        unsigned = {
            "schema_version": RUN_LOCATOR_SCHEMA,
            "run_id": manifest.identity.run_id,
            "manifest_ref": manifest.identity.manifest_ref,
            "manifest_path": str(_canonical_path(manifest_path)),
            "manifest_sha256": manifest.manifest_sha256,
            "updated_at": manifest.updated_at,
        }
        locator = {**unsigned, "locator_sha256": sha256_json(unsigned)}
        locator_path = self.locator_root / f"{manifest.identity.run_id}.json"
        with self._locator_scope() as locator_store:
            with _bind_path(locator_path, locator_store, locator_path.name):
                _atomic_write_json(locator_path, locator)

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


__all__ = [
    "EVENTS_NAME",
    "LifecycleStore",
    "LIFECYCLE_DIR_NAME",
    "MANIFEST_NAME",
    "OWNER_NAME",
    "_RunLock",
]
