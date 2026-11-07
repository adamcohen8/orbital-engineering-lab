"""Bounded local transport for the public typed FSW boundary.

This owner transports records, never physics. Trusted native code retains its
host permissions; stdio is not an operating-system security sandbox.
"""

from __future__ import annotations

import json
import math
import os
import queue
import subprocess
import threading
import time
from collections import deque
from hashlib import sha256
from pathlib import Path
from typing import Any

from sim.flight_software import (
    BootEvent,
    FlightSoftwareCheckpoint,
    FlightSoftwareInputBatch,
    FlightSoftwareOutput,
    FlightSoftwareSnapshot,
    NativeFlightSoftwareSnapshot,
    ShutdownEvent,
    StackIdentity,
    canonical_json_bytes,
)
from sim.flight_software.schemas import from_primitive, to_primitive

PROTOCOL = "oel.fsw.stdio.v1"
MAX_MESSAGE_BYTES = 262144
MAX_SEQUENCE = 2**64 - 1


class NativeBridgeFailure(RuntimeError):
    """Host/transport failure, never a modeled spacecraft fault."""


def _verified_native_implementation_hash(candidate: Any, build: dict[str, Any]) -> str:
    """Bind a native checkpoint to the host-verified artifact identity.

    ``build`` is returned only after ``inspect_native_build`` has checked the
    current manifest/source/verification digests and the executable bytes.
    The candidate never supplies an implementation hash; this digest is
    computed by the host from those verified values and the typed contract.
    """

    source = dict(candidate.normalized_manifest.get("source", {}) or {})
    material = {
        "schema": "oel.fsw.native_implementation_identity.v1",
        "stack_id": candidate.candidate_id,
        "stack_version": candidate.revision,
        "contract_major": 1,
        "onboard_contract": candidate.onboard_contract,
        "hardware_profile": candidate.hardware_profile,
        "native_language": str(source.get("language", "")),
        "source_sha256": str(build["source_sha256"]),
        "manifest_sha256": str(build["manifest_sha256"]),
        "verification_sha256": str(build["verification_sha256"]),
        "executable_sha256": str(build["executable_sha256"]),
    }
    return sha256(canonical_json_bytes(material)).hexdigest()


def start_local_process(
    artifact: Path,
    *,
    stdin: Any = subprocess.PIPE,
    stdout: Any = subprocess.PIPE,
    stderr: Any = subprocess.PIPE,
    cwd: Path | None = None,
) -> subprocess.Popen:
    # Raw pipes prevent close() waiting for a BufferedReader/Writer lock held
    # by a blocked IO worker after the response deadline has expired.
    return subprocess.Popen(
        [str(artifact)],
        cwd=artifact.parent if cwd is None else cwd,
        stdin=stdin,
        stdout=stdout,
        stderr=stderr,
        bufsize=0,
        start_new_session=os.name != "nt",
    )


def _signal_process(process: subprocess.Popen, *, force: bool = False) -> None:
    if os.name != "nt":
        import signal

        try:
            os.killpg(process.pid, signal.SIGKILL if force else signal.SIGTERM)
            return
        except ProcessLookupError:
            return
        except PermissionError:
            pass
    if process.poll() is None:
        try:
            process.kill() if force else process.terminate()
        except ProcessLookupError:
            pass


def stop_local_process(process: subprocess.Popen | None) -> None:
    if process is None:
        return
    try:
        process.wait(timeout=0.1)
    except subprocess.TimeoutExpired:
        _signal_process(process)
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            _signal_process(process, force=True)
            process.wait(timeout=1)
    # Also retire inherited Unix pipes when the direct child exited first.
    if os.name != "nt":
        _signal_process(process)
    for pipe in (process.stdin, process.stdout, process.stderr):
        if pipe is not None:
            pipe.close()


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON field: {key}")
        result[key] = value
    return result


def _nonfinite(value: str) -> None:
    raise ValueError(f"Nonfinite JSON constant: {value}")


class StdioEndpoint:
    def __init__(self, artifact: Path, *, timeout_s: float = 5.0) -> None:
        if not math.isfinite(timeout_s) or not 0 < timeout_s <= 60:
            raise ValueError("Native response deadline must be finite and in (0, 60] seconds.")
        self.process = start_local_process(artifact)
        self.timeout_s = timeout_s
        self.sequence = 0
        self.responses: queue.Queue = queue.Queue(maxsize=2)
        self.diagnostics: deque[bytes] = deque(maxlen=16)
        self.lock = threading.Lock()
        self.closed = False
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.stderr_reader = threading.Thread(target=self._drain_stderr, daemon=True)
        self.reader.start()
        self.stderr_reader.start()

    def _read(self) -> None:
        pending = bytearray()
        try:
            while not self.closed:
                data = self.process.stdout.read(8192)
                if not data:
                    self.responses.put(NativeBridgeFailure("Native endpoint closed its response pipe."), timeout=0.1)
                    return
                pending.extend(data)
                while b"\n" in pending:
                    end = pending.index(b"\n") + 1
                    if end > MAX_MESSAGE_BYTES:
                        raise ValueError("Native response exceeds the message bound.")
                    self.responses.put(bytes(pending[:end]), timeout=0.1)
                    del pending[:end]
                if len(pending) >= MAX_MESSAGE_BYTES:
                    raise ValueError("Native response exceeds the message bound.")
        except (OSError, ValueError) as exc:
            if not self.closed:
                try:
                    self.responses.put(NativeBridgeFailure(str(exc)), timeout=0.1)
                except queue.Full:
                    pass
        except queue.Full:
            return

    def _drain_stderr(self) -> None:
        try:
            while not self.closed:
                data = self.process.stderr.read(4096)
                if not data:
                    return
                self.diagnostics.append(data)
        except (OSError, ValueError):
            return

    def exchange(self, operation: str, payload: Any) -> Any:
        with self.lock:
            if self.closed:
                raise NativeBridgeFailure("Native endpoint is closed.")
            if self.sequence >= MAX_SEQUENCE or not self.responses.empty():
                self.close()
                raise NativeBridgeFailure("Native sequence exhausted or unsolicited output received.")
            sequence = self.sequence + 1
            request = {"protocol": PROTOCOL, "sequence": sequence, "operation": operation, "payload": payload}
            data = (json.dumps(request, allow_nan=False, separators=(",", ":")) + "\n").encode()
            if len(data) > MAX_MESSAGE_BYTES:
                self.close()
                raise NativeBridgeFailure("Native request exceeds the message bound.")
            self.sequence = sequence
            deadline = time.monotonic() + self.timeout_s
            written: queue.Queue = queue.Queue(maxsize=1)

            def write() -> None:
                try:
                    remaining = memoryview(data)
                    while remaining and not self.closed:
                        count = self.process.stdin.write(remaining)
                        if not count:
                            raise BrokenPipeError("Native request pipe closed during write.")
                        remaining = remaining[count:]
                    if remaining:
                        raise BrokenPipeError("Native request was cancelled.")
                    written.put(None)
                except (OSError, ValueError) as exc:
                    written.put(exc)

            threading.Thread(target=write, daemon=True).start()
            try:
                error = written.get(timeout=max(0.001, deadline - time.monotonic()))
                if error:
                    raise NativeBridgeFailure("Native request pipe failed.") from error
                response = self.responses.get(timeout=max(0.001, deadline - time.monotonic()))
                if isinstance(response, Exception):
                    raise response
                packet = json.loads(response, parse_constant=_nonfinite, object_pairs_hook=_unique_object)
                if (
                    not isinstance(packet, dict)
                    or packet.get("protocol") != PROTOCOL
                    or type(packet.get("sequence")) is not int
                    or packet["sequence"] != sequence
                    or packet.get("operation") != operation
                    or packet.get("ok") is not True
                    or "payload" not in packet
                ):
                    raise NativeBridgeFailure("Native response identity, version, or status mismatch.")
                return packet["payload"]
            except (queue.Empty, ValueError, RecursionError, NativeBridgeFailure) as exc:
                self.close()
                raise NativeBridgeFailure(f"Native {operation} failed: {exc}") from exc

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            stop_local_process(self.process)


class NativeFlightSoftwareAdapter:
    """An executable presented as ordinary SatelliteFlightSoftware."""

    def __init__(
        self,
        manifest_path: str,
        workspace_root: str,
        satellite_id: str = "target",
        reference_object_id: str = "target",
        allow_legacy_snapshot_restore: bool = False,
    ) -> None:
        from .candidate import load_candidate
        from .native import inspect_native_build

        self.candidate = load_candidate(manifest_path, workspace_root=workspace_root)
        if self.candidate.native is None:
            raise ValueError("Native adapter requires a native_stack candidate.")
        self.build = inspect_native_build(self.candidate)
        self.satellite_id = satellite_id
        self._allow_legacy_snapshot_restore = bool(allow_legacy_snapshot_restore)
        self.endpoint: StdioEndpoint | None = None
        self._active_boot_id: str | None = None
        self._implementation_hash = _verified_native_implementation_hash(self.candidate, self.build)
        # The old adapter used the verified executable digest directly as its
        # identity.  Keep that host-derived value only for explicit migration
        # of pre-envelope checkpoints; new snapshots use ``_implementation_hash``.
        self._legacy_implementation_hash = str(self.build["executable_sha256"])
        self._identity = StackIdentity(
            self.candidate.candidate_id,
            self.candidate.revision,
            1,
            self._implementation_hash,
            self.candidate.native["checkpointable"],
        )

    @property
    def identity(self) -> StackIdentity:
        return self._identity

    def boot(self, event: BootEvent) -> None:
        from .native import inspect_native_build

        if self.endpoint is not None:
            raise NativeBridgeFailure("Native stack is already booted.")
        if event.satellite_id != self.satellite_id:
            raise NativeBridgeFailure("Native boot satellite does not match the adapter.")
        current_build = inspect_native_build(self.candidate)
        if current_build != self.build or _verified_native_implementation_hash(self.candidate, current_build) != self._implementation_hash:
            raise NativeBridgeFailure("Native build changed before boot.")
        self.endpoint = StdioEndpoint(Path(self.build["executable"]), timeout_s=self.candidate.native["timeout_s"])
        try:
            hello = self.endpoint.exchange("hello", {"boundary": "oel.fsw.boundary.v1"})
            if (
                not isinstance(hello, dict)
                or type(hello.get("checkpointable")) is not bool
                or hello != {"boundary": "oel.fsw.boundary.v1", "checkpointable": self.identity.checkpointable}
            ):
                raise NativeBridgeFailure("Native capabilities do not match the declared contract.")
            self._ack("boot", event)
            self._active_boot_id = event.boot_id
        except Exception:
            self._active_boot_id = None
            self.endpoint.close()
            raise

    def _exchange(self, operation: str, payload: Any) -> Any:
        if self.endpoint is None:
            raise NativeBridgeFailure("Native stack is not booted.")
        return self.endpoint.exchange(operation, to_primitive(payload))

    def _ack(self, operation: str, payload: Any) -> None:
        if self._exchange(operation, payload) != {}:
            self.endpoint.close()
            raise NativeBridgeFailure(f"Native {operation} acknowledgement must be an empty object.")

    def step(self, batch: FlightSoftwareInputBatch) -> FlightSoftwareOutput:
        try:
            if batch.satellite_id != self.satellite_id:
                raise ValueError("Native input targets a different satellite.")
            output = from_primitive(FlightSoftwareOutput, self._exchange("step", batch))
            if output.satellite_id != batch.satellite_id or output.invocation_id != batch.invocation_id:
                raise ValueError("Native output does not match its invocation.")
            return output
        except Exception as exc:
            if self.endpoint:
                self.endpoint.close()
            raise NativeBridgeFailure(f"Invalid native FSW output: {exc}") from exc

    def snapshot(self) -> NativeFlightSoftwareSnapshot:
        if not self.identity.checkpointable:
            raise NativeBridgeFailure("Native candidate declares checkpointable=false.")
        try:
            snapshot = from_primitive(FlightSoftwareSnapshot, self._exchange("snapshot", {}))
            self._check_snapshot_identity(snapshot)
            return NativeFlightSoftwareSnapshot(
                stack_id=snapshot.stack_id,
                stack_version=snapshot.stack_version,
                boot_id=snapshot.boot_id,
                invocation_id=snapshot.invocation_id,
                active_load_id=snapshot.active_load_id,
                active_load_revision=snapshot.active_load_revision,
                state_bytes=snapshot.state_bytes,
                state_hash_sha256=snapshot.state_hash_sha256,
                host_implementation_hash=self._implementation_hash,
            )
        except Exception as exc:
            if self.endpoint:
                self.endpoint.close()
            raise NativeBridgeFailure(f"Invalid native checkpoint: {exc}") from exc

    def restore(self, snapshot: FlightSoftwareCheckpoint) -> None:
        if not self.identity.checkpointable:
            raise NativeBridgeFailure("Native candidate declares checkpointable=false.")
        self._check_snapshot_identity(snapshot)
        if isinstance(snapshot, NativeFlightSoftwareSnapshot):
            if snapshot.host_implementation_hash != self._implementation_hash:
                raise NativeBridgeFailure("Native checkpoint implementation identity does not match the verified artifact.")
            payload = snapshot.as_flight_software_snapshot()
        else:
            raise NativeBridgeFailure(
                "Legacy native checkpoint restore is blocked; call migrate_legacy_snapshot with an independently "
                "verified executable digest before restore."
            )
        self._ack("restore", payload)

    def migrate_legacy_snapshot(
        self,
        snapshot: FlightSoftwareSnapshot,
        *,
        expected_executable_sha256: str,
    ) -> NativeFlightSoftwareSnapshot:
        """Promote an old v1 checkpoint after explicit host-side identity review.

        The expected executable digest must come from the enclosing trusted
        checkpoint evidence.  It is compared with the digest captured from the
        verified build receipt; candidate metadata cannot satisfy this check.
        The returned envelope is the only legacy result accepted by restore().
        """

        if not self.identity.checkpointable:
            raise NativeBridgeFailure("Native candidate declares checkpointable=false.")
        if not self._allow_legacy_snapshot_restore:
            raise NativeBridgeFailure(
                "Legacy native checkpoint migration requires allow_legacy_snapshot_restore=true."
            )
        if isinstance(snapshot, NativeFlightSoftwareSnapshot):
            raise NativeBridgeFailure("Native checkpoint is already host-bound; restore it directly.")
        self._check_snapshot_identity(snapshot)
        if (
            not isinstance(expected_executable_sha256, str)
            or len(expected_executable_sha256) != 64
            or any(ch not in "0123456789abcdef" for ch in expected_executable_sha256)
        ):
            raise NativeBridgeFailure("Legacy migration expected executable digest must be a lowercase SHA-256 value.")
        if expected_executable_sha256 != self._legacy_implementation_hash:
            raise NativeBridgeFailure(
                "Legacy migration executable digest does not match the currently verified native artifact."
            )
        return NativeFlightSoftwareSnapshot(
            stack_id=snapshot.stack_id,
            stack_version=snapshot.stack_version,
            boot_id=snapshot.boot_id,
            invocation_id=snapshot.invocation_id,
            active_load_id=snapshot.active_load_id,
            active_load_revision=snapshot.active_load_revision,
            state_bytes=snapshot.state_bytes,
            state_hash_sha256=snapshot.state_hash_sha256,
            host_implementation_hash=self._implementation_hash,
        )

    def _check_snapshot_identity(self, snapshot: FlightSoftwareCheckpoint) -> None:
        if (snapshot.stack_id, snapshot.stack_version) != (self.identity.stack_id, self.identity.stack_version):
            raise NativeBridgeFailure("Native checkpoint belongs to a different stack or version.")
        if self._active_boot_id is None:
            raise NativeBridgeFailure("Native checkpoint does not belong to an active boot.")
        if snapshot.boot_id != self._active_boot_id:
            raise NativeBridgeFailure("Native checkpoint belongs to a different boot.")

    def shutdown(self, event: ShutdownEvent) -> None:
        if self.endpoint is not None:
            try:
                if not self.endpoint.closed:
                    self._ack("shutdown", event)
            finally:
                self.endpoint.close()
                self._active_boot_id = None
