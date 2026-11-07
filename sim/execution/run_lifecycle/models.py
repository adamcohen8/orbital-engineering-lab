from __future__ import annotations

import hashlib
import json
import math
import platform
import uuid
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from typing import Any

RUN_HANDLE_SCHEMA = "oel.run-handle.v1"
RUN_STATE_SCHEMA = "oel.run-state.v1"
RUN_EVENT_SCHEMA = "oel.run-event.v1"
RUN_MANIFEST_SCHEMA = "oel.run-manifest.v1"
RUN_LOCATOR_SCHEMA = "oel.run-locator.v1"
AWAIT_RESULT_SCHEMA = "oel.await-result.v1"
INSPECT_RESULT_SCHEMA = "oel.inspect-result.v1"
FOREGROUND_RESULT_SCHEMA = "oel.foreground-result.v1"
EVENTS_RESULT_SCHEMA = "oel.run-events-result.v1"
EXECUTION_OWNER_SCHEMA = "oel.execution-owner.v1"
RECONCILE_RESULT_SCHEMA = "oel.reconcile-result.v1"


class LifecycleError(RuntimeError):
    """Base error for the transport-neutral local lifecycle contract."""


class InvalidTransitionError(LifecycleError):
    """Raised when a transition is outside the frozen lifecycle state machine."""


class MalformedRunStateError(LifecycleError):
    """Raised when durable lifecycle state cannot be verified."""


class RunNotFoundError(LifecycleError):
    """Raised when an opaque run identity has no authorized locator."""


class RunIdentityMismatchError(LifecycleError):
    """Raised when a caller-provided content identity does not match the run."""


class RunPolicyError(LifecycleError):
    """Raised when a requested path, workflow, or resource posture is not allowed."""


class RunState(str, Enum):
    ACCEPTED = "accepted"
    STARTING = "starting"
    RUNNING = "running"
    FINALIZING = "finalizing"
    CANCELLING = "cancelling"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    INTERRUPTED = "interrupted"


class RunPhase(str, Enum):
    ACCEPTED = "accepted"
    MATERIALIZING = "materializing"
    EXECUTING = "executing"
    COLLECTING_ARTIFACTS = "collecting_artifacts"
    COMMITTING_TERMINAL_STATE = "committing_terminal_state"


TERMINAL_STATES = frozenset(
    {
        RunState.COMPLETED,
        RunState.FAILED,
        RunState.CANCELLED,
        RunState.INTERRUPTED,
    }
)

ALLOWED_TRANSITIONS: dict[RunState, frozenset[RunState]] = {
    RunState.ACCEPTED: frozenset(
        {RunState.STARTING, RunState.FAILED, RunState.CANCELLING, RunState.INTERRUPTED}
    ),
    RunState.STARTING: frozenset(
        {RunState.RUNNING, RunState.FAILED, RunState.CANCELLING, RunState.INTERRUPTED}
    ),
    RunState.RUNNING: frozenset(
        {RunState.FINALIZING, RunState.FAILED, RunState.CANCELLING, RunState.INTERRUPTED}
    ),
    RunState.CANCELLING: frozenset(
        {RunState.CANCELLED, RunState.FAILED, RunState.INTERRUPTED}
    ),
    RunState.FINALIZING: frozenset(
        {RunState.COMPLETED, RunState.FAILED, RunState.INTERRUPTED}
    ),
    RunState.COMPLETED: frozenset(),
    RunState.FAILED: frozenset(),
    RunState.CANCELLED: frozenset(),
    RunState.INTERRUPTED: frozenset(),
}

STATE_PHASES: dict[RunState, frozenset[RunPhase]] = {
    RunState.ACCEPTED: frozenset({RunPhase.ACCEPTED}),
    RunState.STARTING: frozenset({RunPhase.MATERIALIZING}),
    RunState.RUNNING: frozenset({RunPhase.EXECUTING}),
    RunState.FINALIZING: frozenset({RunPhase.COLLECTING_ARTIFACTS}),
    RunState.CANCELLING: frozenset({RunPhase.EXECUTING, RunPhase.COLLECTING_ARTIFACTS}),
    RunState.COMPLETED: frozenset({RunPhase.COMMITTING_TERMINAL_STATE}),
    RunState.FAILED: frozenset({RunPhase.COMMITTING_TERMINAL_STATE}),
    RunState.CANCELLED: frozenset({RunPhase.COMMITTING_TERMINAL_STATE}),
    RunState.INTERRUPTED: frozenset({RunPhase.COMMITTING_TERMINAL_STATE}),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def sha256_json(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def local_host_sha256() -> str:
    return sha256_json({"hostname": platform.node().strip().lower() or "unknown"})


def validate_transition(current: RunState, requested: RunState) -> None:
    if requested not in ALLOWED_TRANSITIONS[current]:
        raise InvalidTransitionError(f"Lifecycle transition {current.value!r} -> {requested.value!r} is not allowed.")


def validate_state_phase(state: RunState, phase: RunPhase) -> None:
    if phase not in STATE_PHASES[state]:
        raise InvalidTransitionError(
            f"Lifecycle state {state.value!r} cannot use phase {phase.value!r}."
        )


@dataclass(frozen=True)
class RunIdentity:
    run_id: str
    manifest_ref: str
    normalized_config_sha256: str
    validation_id: str
    source_config_sha256: str
    source_config_ref: str
    output_dir: str
    resource_profile: str
    resource_plan_sha256: str
    engine_version: str
    engine_edition: str
    installation_disposition: str
    source_revision: str
    authorization_disposition: str
    plugin_trust_disposition: str
    handling_label: str = "public"
    release_scope: str = "public"
    execution_mode: str = "foreground"

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "manifest_ref": self.manifest_ref,
            "normalized_config_sha256": self.normalized_config_sha256,
            "validation_id": self.validation_id,
            "source_config_sha256": self.source_config_sha256,
            "source_config_ref": self.source_config_ref,
            "output_dir": self.output_dir,
            "resource_profile": self.resource_profile,
            "resource_plan_sha256": self.resource_plan_sha256,
            "engine_version": self.engine_version,
            "engine_edition": self.engine_edition,
            "installation_disposition": self.installation_disposition,
            "source_revision": self.source_revision,
            "authorization_disposition": self.authorization_disposition,
            "plugin_trust_disposition": self.plugin_trust_disposition,
            "handling_label": self.handling_label,
            "release_scope": self.release_scope,
            "execution_mode": self.execution_mode,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> RunIdentity:
        try:
            identity = cls(**{name: str(value[name]) for name in cls.__dataclass_fields__})
        except (KeyError, TypeError, ValueError) as exc:
            raise MalformedRunStateError("Run identity is missing required fields.") from exc
        if not identity.run_id or not identity.manifest_ref.startswith("oel-run-ref:"):
            raise MalformedRunStateError("Run identity has an invalid run_id or manifest_ref.")
        try:
            if str(uuid.UUID(identity.run_id)) != identity.run_id:
                raise ValueError
        except ValueError as exc:
            raise MalformedRunStateError("Run identity has an invalid canonical UUID.") from exc
        for field_name in ("normalized_config_sha256", "source_config_sha256", "resource_plan_sha256"):
            digest = str(getattr(identity, field_name))
            if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
                raise MalformedRunStateError(f"Run identity has an invalid {field_name}.")
        if identity.execution_mode != "foreground":
            raise MalformedRunStateError("This lifecycle version supports only foreground execution.")
        return identity


@dataclass(frozen=True)
class RunManifest:
    identity: RunIdentity
    state: RunState
    sequence: int
    phase: RunPhase
    created_at: str
    updated_at: str
    started_at: str | None = None
    terminal_at: str | None = None
    execution: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, Any] = field(default_factory=dict)
    review: dict[str, Any] = field(default_factory=dict)
    error: dict[str, Any] | None = None
    manifest_sha256: str = ""

    @property
    def is_terminal(self) -> bool:
        return self.state in TERMINAL_STATES

    def unsigned_dict(self) -> dict[str, Any]:
        return {
            "schema_version": RUN_MANIFEST_SCHEMA,
            "identity": self.identity.to_dict(),
            "state": self.state.value,
            "sequence": int(self.sequence),
            "phase": self.phase.value,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "started_at": self.started_at,
            "terminal_at": self.terminal_at,
            "execution": dict(self.execution),
            "artifacts": dict(self.artifacts),
            "review": dict(self.review),
            "error": None if self.error is None else dict(self.error),
        }

    def with_digest(self) -> RunManifest:
        return replace(self, manifest_sha256=sha256_json(self.unsigned_dict()))

    def to_dict(self) -> dict[str, Any]:
        payload = self.unsigned_dict()
        payload["manifest_sha256"] = self.manifest_sha256 or sha256_json(payload)
        return payload

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": RUN_STATE_SCHEMA,
            "run_id": self.identity.run_id,
            "manifest_ref": self.identity.manifest_ref,
            "normalized_config_sha256": self.identity.normalized_config_sha256,
            "validation_id": self.identity.validation_id,
            "resource_profile": self.identity.resource_profile,
            "state": self.state.value,
            "terminal": self.is_terminal,
            "sequence": self.sequence,
            "phase": self.phase.value,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "started_at": self.started_at,
            "terminal_at": self.terminal_at,
            "execution": dict(self.execution),
            "artifacts": dict(self.artifacts),
            "review": dict(self.review),
            "error": None if self.error is None else dict(self.error),
            "manifest_sha256": self.manifest_sha256,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> RunManifest:
        if value.get("schema_version") != RUN_MANIFEST_SCHEMA:
            raise MalformedRunStateError("Run manifest schema_version is unsupported.")
        try:
            manifest = cls(
                identity=RunIdentity.from_dict(dict(value["identity"])),
                state=RunState(str(value["state"])),
                sequence=int(value["sequence"]),
                phase=RunPhase(str(value["phase"])),
                created_at=str(value["created_at"]),
                updated_at=str(value["updated_at"]),
                started_at=None if value.get("started_at") is None else str(value["started_at"]),
                terminal_at=None if value.get("terminal_at") is None else str(value["terminal_at"]),
                execution=dict(value.get("execution", {}) or {}),
                artifacts=dict(value.get("artifacts", {}) or {}),
                review=dict(value.get("review", {}) or {}),
                error=None if value.get("error") is None else dict(value["error"]),
                manifest_sha256=str(value.get("manifest_sha256", "")),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise MalformedRunStateError("Run manifest is missing or has invalid required fields.") from exc
        if manifest.sequence < 1:
            raise MalformedRunStateError("Run manifest sequence must be positive.")
        try:
            validate_state_phase(manifest.state, manifest.phase)
        except InvalidTransitionError as exc:
            raise MalformedRunStateError("Run manifest state and phase are inconsistent.") from exc
        try:
            expected = sha256_json(manifest.unsigned_dict())
        except (TypeError, ValueError) as exc:
            raise MalformedRunStateError("Run manifest contains non-canonical JSON values.") from exc
        if manifest.manifest_sha256 != expected:
            raise MalformedRunStateError("Run manifest digest does not match its canonical content.")
        if manifest.is_terminal and not manifest.terminal_at:
            raise MalformedRunStateError("Terminal run manifest is missing terminal_at.")
        if not manifest.is_terminal and manifest.terminal_at is not None:
            raise MalformedRunStateError("Nonterminal run manifest cannot contain terminal_at.")
        return manifest


@dataclass(frozen=True)
class RunHandle:
    identity: RunIdentity
    state: RunState
    sequence: int
    created_at: str

    @classmethod
    def from_manifest(cls, manifest: RunManifest) -> RunHandle:
        return cls(manifest.identity, manifest.state, manifest.sequence, manifest.created_at)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": RUN_HANDLE_SCHEMA,
            "run_id": self.identity.run_id,
            "state": self.state.value,
            "sequence": self.sequence,
            "created_at": self.created_at,
            "manifest_ref": self.identity.manifest_ref,
            "normalized_config_sha256": self.identity.normalized_config_sha256,
            "validation_id": self.identity.validation_id,
            "resource_profile": self.identity.resource_profile,
            "authorization_disposition": self.identity.authorization_disposition,
            "execution_mode": self.identity.execution_mode,
        }


@dataclass(frozen=True)
class RunEvent:
    run_id: str
    sequence: int
    event: str
    state: RunState
    phase: RunPhase
    emitted_at: str
    manifest_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": RUN_EVENT_SCHEMA,
            "run_id": self.run_id,
            "sequence": self.sequence,
            "event": self.event,
            "state": self.state.value,
            "phase": self.phase.value,
            "emitted_at": self.emitted_at,
            "manifest_sha256": self.manifest_sha256,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> RunEvent:
        if value.get("schema_version") != RUN_EVENT_SCHEMA:
            raise MalformedRunStateError("Run event schema_version is unsupported.")
        try:
            event = cls(
                run_id=str(value["run_id"]),
                sequence=int(value["sequence"]),
                event=str(value["event"]),
                state=RunState(str(value["state"])),
                phase=RunPhase(str(value["phase"])),
                emitted_at=str(value["emitted_at"]),
                manifest_sha256=str(value["manifest_sha256"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise MalformedRunStateError("Run event is missing or has invalid required fields.") from exc
        if len(event.manifest_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in event.manifest_sha256
        ):
            raise MalformedRunStateError("Run event has an invalid manifest digest.")
        return event


@dataclass(frozen=True)
class ExecutionOwner:
    run_id: str
    owner_token: str
    pid: int
    host_sha256: str
    status: str
    started_at: str
    updated_at: str
    heartbeat_interval_s: float
    stale_after_s: float
    owner_sha256: str = ""

    def unsigned_dict(self) -> dict[str, Any]:
        return {
            "schema_version": EXECUTION_OWNER_SCHEMA,
            "run_id": self.run_id,
            "owner_token": self.owner_token,
            "pid": self.pid,
            "host_sha256": self.host_sha256,
            "status": self.status,
            "started_at": self.started_at,
            "updated_at": self.updated_at,
            "heartbeat_interval_s": self.heartbeat_interval_s,
            "stale_after_s": self.stale_after_s,
        }

    def with_digest(self) -> ExecutionOwner:
        return replace(self, owner_sha256=sha256_json(self.unsigned_dict()))

    def to_dict(self) -> dict[str, Any]:
        payload = self.unsigned_dict()
        payload["owner_sha256"] = self.owner_sha256 or sha256_json(payload)
        return payload

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ExecutionOwner:
        if value.get("schema_version") != EXECUTION_OWNER_SCHEMA:
            raise MalformedRunStateError("Execution-owner schema_version is unsupported.")
        try:
            owner = cls(
                run_id=str(value["run_id"]),
                owner_token=str(value["owner_token"]),
                pid=int(value["pid"]),
                host_sha256=str(value["host_sha256"]),
                status=str(value["status"]),
                started_at=str(value["started_at"]),
                updated_at=str(value["updated_at"]),
                heartbeat_interval_s=float(value["heartbeat_interval_s"]),
                stale_after_s=float(value["stale_after_s"]),
                owner_sha256=str(value["owner_sha256"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise MalformedRunStateError("Execution-owner state is missing required fields.") from exc
        if isinstance(value.get("pid"), bool) or owner.pid < 1 or owner.status not in {"active", "exited"}:
            raise MalformedRunStateError("Execution-owner pid or status is invalid.")
        if (
            not math.isfinite(owner.heartbeat_interval_s)
            or not math.isfinite(owner.stale_after_s)
            or owner.heartbeat_interval_s <= 0
            or owner.stale_after_s <= owner.heartbeat_interval_s
        ):
            raise MalformedRunStateError("Execution-owner heartbeat timing is invalid.")
        try:
            expected = sha256_json(owner.unsigned_dict())
        except (TypeError, ValueError) as exc:
            raise MalformedRunStateError("Execution-owner contains non-canonical JSON values.") from exc
        if owner.owner_sha256 != expected:
            raise MalformedRunStateError("Execution-owner digest does not match its canonical content.")
        return owner


@dataclass(frozen=True)
class RunInspection:
    outcome: str
    run_id: str
    state: dict[str, Any] | None = None
    error: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": INSPECT_RESULT_SCHEMA,
            "outcome": self.outcome,
            "run_id": self.run_id,
            "state": self.state,
            "error": self.error,
            "run_state_changed": False,
        }


@dataclass(frozen=True)
class AwaitResult:
    outcome: str
    run_id: str
    wait_requested_s: float
    wait_observed_s: float
    state: dict[str, Any] | None = None
    event: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    owner: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": AWAIT_RESULT_SCHEMA,
            "outcome": self.outcome,
            "run_id": self.run_id,
            "wait_requested_s": round(float(self.wait_requested_s), 6),
            "wait_observed_s": round(float(self.wait_observed_s), 6),
            "state": self.state,
            "event": self.event,
            "error": self.error,
            "owner": self.owner,
            "run_state_changed": False,
        }


@dataclass(frozen=True)
class ReconcileResult:
    outcome: str
    run_id: str
    state: dict[str, Any] | None = None
    owner: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    run_state_changed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": RECONCILE_RESULT_SCHEMA,
            "outcome": self.outcome,
            "run_id": self.run_id,
            "state": self.state,
            "owner": self.owner,
            "error": self.error,
            "run_state_changed": self.run_state_changed,
        }


@dataclass(frozen=True)
class ForegroundResult:
    handle: RunHandle
    terminal_manifest: RunManifest

    @property
    def succeeded(self) -> bool:
        return self.terminal_manifest.state is RunState.COMPLETED

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": FOREGROUND_RESULT_SCHEMA,
            "status": "completed" if self.succeeded else "failed",
            "handle": self.handle.to_dict(),
            "state": self.terminal_manifest.state_dict(),
        }
