
import hashlib
import json
import math
import os
import re
import stat
import weakref
from collections.abc import Iterable
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping

from sim.security.input_reads import DescriptorStore, SecureReadError, SecureReadRoot

_SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")
_GRANT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,159}$")
_OPERATION_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,159}$")
_MAX_LEDGER_RECORD_BYTES = 4_000_000


class ApprovalGrantError(PermissionError):
    """Base class for fail-closed host approval grant errors."""


class ApprovalGrantExpired(ApprovalGrantError):
    """Raised when a bounded grant is past its configured expiry."""


class ApprovalGrantRevoked(ApprovalGrantError):
    """Raised when a host has revoked a bounded grant."""


class ApprovalGrantMismatch(ApprovalGrantError):
    """Raised when a grant does not match the requested operation."""


class OperationLedgerError(RuntimeError):
    """Base class for durable operation ledger errors."""


class OperationConflictError(OperationLedgerError):
    """Raised when an operation ID is reused with different request identity."""

    oel_recovery_code = "operation.conflict"
    oel_reconciliation_required = True
    oel_retryable = False


class UnknownOperationOutcome(OperationLedgerError):
    """Raised when a previous operation began but has no committed terminal result."""

    oel_recovery_code = "operation.unknown_outcome"
    oel_reconciliation_required = True
    oel_retryable = False


@dataclass(frozen=True)
class ContentIdentity:
    """A host-configured content binding checked against current inputs."""

    argument: str
    sha256: str
    kind: str = "file"

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], *, default_argument: str = "") -> "ContentIdentity":
        argument = str(value.get("argument", value.get("field", default_argument))).strip()
        digest = str(value.get("sha256", value.get("content_sha256", ""))).strip().lower()
        kind = str(value.get("kind", "file")).strip()
        if not argument:
            raise ValueError("Approval grant content identity requires an argument field.")
        if not _SHA256.fullmatch(digest):
            raise ValueError("Approval grant content identity requires a SHA-256 digest.")
        if kind not in {"file", "review_digest"}:
            raise ValueError("Approval grant content identity kind is unsupported.")
        return cls(argument=argument, sha256=digest, kind=kind)


@dataclass(frozen=True)
class ApprovalGrant:
    """An operator-configured write, execute, or trust grant.

    A grant is configuration, never request data. A bounded grant must bind
    at least one reviewed tool, argument digest, plan operation, content
    identity, or resource ceiling and must carry an expiry.
    """

    grant_id: str
    scopes: frozenset[str]
    tool_ids: frozenset[str] = frozenset()
    reviewed_arguments_sha256: str = ""
    operation_id: str = ""
    content_identities: tuple[ContentIdentity, ...] = ()
    resource_ceiling: Mapping[str, float | int] | None = None
    expires_at: datetime | None = None
    revoked: bool = False
    purpose: str = ""

    def __post_init__(self) -> None:
        if not _GRANT_ID.fullmatch(self.grant_id):
            raise ValueError("Approval grant id is invalid.")
        if not self.scopes or not self.scopes <= {"trust", "write", "execute"}:
            raise ValueError("Approval grant scope is invalid.")
        if self.reviewed_arguments_sha256 and not _SHA256.fullmatch(self.reviewed_arguments_sha256):
            raise ValueError("Approval grant reviewed argument digest is invalid.")
        if self.operation_id and not _OPERATION_ID.fullmatch(self.operation_id):
            raise ValueError("Approval grant operation id is invalid.")
        ceiling = dict(self.resource_ceiling or {})
        for key, value in ceiling.items():
            if not str(key).strip() or isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("Approval grant resource ceilings must be finite numbers.")
            if not math.isfinite(float(value)) or float(value) < 0:
                raise ValueError("Approval grant resource ceilings must be finite and non-negative.")
        object.__setattr__(self, "resource_ceiling", dict(sorted((str(k), v) for k, v in ceiling.items())))
        if self.expires_at is not None and self.expires_at.tzinfo is None:
            raise ValueError("Approval grant expiry must include a timezone.")
        if self.bounded:
            if not self.tool_ids:
                raise ValueError("Bounded approval grants require an exact tool identity.")
            if not self.reviewed_arguments_sha256 and not (self.operation_id and self.content_identities):
                raise ValueError(
                    "Bounded approval grants require reviewed arguments or a reviewed plan operation with content identities."
                )
            if self.expires_at is None:
                raise ValueError("Bounded approval grants require an expiry.")

    @property
    def bounded(self) -> bool:
        return bool(
            self.tool_ids
            or self.reviewed_arguments_sha256
            or self.operation_id
            or self.content_identities
            or self.resource_ceiling
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ApprovalGrant":
        raw = dict(value)
        grant_id = str(raw.get("grant_id", "")).strip()
        scopes_value = raw.get("scopes", raw.get("scope"))
        if isinstance(scopes_value, str):
            scopes = frozenset({scopes_value.strip()})
        elif isinstance(scopes_value, (list, tuple, set)):
            scopes = frozenset(str(item).strip() for item in scopes_value if str(item).strip())
        else:
            scopes = frozenset()
        tool_value = raw.get("tool_ids", raw.get("tool_id", ()))
        if isinstance(tool_value, str):
            tool_ids = frozenset({tool_value.strip()})
        elif isinstance(tool_value, (list, tuple, set)):
            tool_ids = frozenset(str(item).strip() for item in tool_value if str(item).strip())
        else:
            tool_ids = frozenset()
        digest = str(
            raw.get(
                "reviewed_arguments_sha256",
                raw.get("reviewed_args_sha256", raw.get("arguments_sha256", "")),
            )
        ).strip().lower()
        operation_id = str(
            raw.get("operation_id", raw.get("plan_operation_id", raw.get("plan_operation", "")))
        ).strip()
        content_raw = raw.get("content_identities", raw.get("content_identity", ()))
        identities: list[ContentIdentity] = []
        if isinstance(content_raw, Mapping):
            for argument, item in content_raw.items():
                if isinstance(item, str):
                    item = {"sha256": item}
                if not isinstance(item, Mapping):
                    raise ValueError("Approval grant content identities must be mappings.")
                identities.append(ContentIdentity.from_mapping(item, default_argument=str(argument)))
        elif isinstance(content_raw, (list, tuple)):
            for item in content_raw:
                if not isinstance(item, Mapping):
                    raise ValueError("Approval grant content identities must be mappings.")
                identities.append(ContentIdentity.from_mapping(item))
        elif content_raw:
            raise ValueError("Approval grant content identities must be a mapping or list.")
        ceiling_raw = raw.get("resource_ceiling", raw.get("resource_limits", {}))
        if not isinstance(ceiling_raw, Mapping):
            raise ValueError("Approval grant resource ceiling must be a mapping.")
        ceiling: dict[str, float | int] = {}
        for key, item in ceiling_raw.items():
            if isinstance(item, bool) or not isinstance(item, (int, float)):
                raise ValueError("Approval grant resource ceilings must be finite numbers.")
            ceiling[str(key)] = item
        expires_raw = raw.get("expires_at")
        expires_at = _parse_expiry(expires_raw) if expires_raw else None
        return cls(
            grant_id=grant_id,
            scopes=scopes,
            tool_ids=tool_ids,
            reviewed_arguments_sha256=digest,
            operation_id=operation_id,
            content_identities=tuple(identities),
            resource_ceiling=ceiling,
            expires_at=expires_at,
            revoked=bool(raw.get("revoked", False)),
            purpose=str(raw.get("purpose", "")).strip(),
        )

    def validate(
        self,
        *,
        scope: str,
        tool_id: str,
        arguments: Mapping[str, Any],
        requested_operation_id: str,
        path_policy: Any = None,
        now: datetime | None = None,
        content_identity_resolver: Callable[[Mapping[str, Any], ContentIdentity], str] | None = None,
        resource_metrics: Mapping[str, Any] | None = None,
        defer_resource_ceiling: bool = False,
    ) -> None:
        if self.revoked:
            raise ApprovalGrantRevoked("The configured MCP approval grant has been revoked.")
        current = now or datetime.now(timezone.utc)
        if self.expires_at is not None and current.astimezone(timezone.utc) >= self.expires_at.astimezone(timezone.utc):
            raise ApprovalGrantExpired("The configured MCP approval grant has expired.")
        if scope not in self.scopes:
            raise ApprovalGrantMismatch("The approval grant does not authorize this operation scope.")
        if self.tool_ids and tool_id not in self.tool_ids:
            raise ApprovalGrantMismatch("The approval grant does not authorize this tool.")
        if self.operation_id and requested_operation_id != self.operation_id:
            raise ApprovalGrantMismatch("The approval operation id does not match the reviewed plan operation.")
        if self.reviewed_arguments_sha256:
            actual = canonical_request_digest(tool_id, arguments)
            if actual != self.reviewed_arguments_sha256:
                raise ApprovalGrantMismatch("The request does not match the reviewed MCP arguments.")
        if self.content_identities:
            for identity in self.content_identities:
                actual = _content_identity(
                    arguments,
                    identity,
                    path_policy=path_policy,
                    resolver=content_identity_resolver,
                )
                if actual != identity.sha256:
                    raise ApprovalGrantMismatch("A reviewed MCP content identity no longer matches.")
        if self.resource_ceiling and not defer_resource_ceiling:
            self.validate_resource_metrics(resource_metrics)

    def validate_resource_metrics(self, metrics: Mapping[str, Any] | None) -> None:
        """Validate host-computed metrics after the request has been prepared.

        Request arguments remain bound by ``reviewed_arguments_sha256`` but are
        never accepted as evidence for a resource ceiling.  Callers must pass
        the resource plan or estimate produced by the trusted OEL preparation
        path; an absent plan or metric fails closed.
        """

        if not self.resource_ceiling:
            return
        if not isinstance(metrics, Mapping):
            raise ApprovalGrantMismatch(
                "A host-computed resource plan is required for this approval ceiling."
            )
        for name, ceiling in self.resource_ceiling.items():
            observed = _resource_value(metrics, name)
            if observed is None:
                raise ApprovalGrantMismatch(
                    f"Host-computed MCP resource value {name} is missing from the prepared plan."
                )
            if observed > float(ceiling):
                raise ApprovalGrantMismatch(
                    f"The prepared MCP resource plan exceeds the configured {name} ceiling."
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "grant_id": self.grant_id,
            "scopes": sorted(self.scopes),
            "tool_ids": sorted(self.tool_ids),
            "reviewed_arguments_sha256": self.reviewed_arguments_sha256,
            "operation_id": self.operation_id,
            "content_identities": [
                {"argument": item.argument, "sha256": item.sha256, "kind": item.kind}
                for item in self.content_identities
            ],
            "resource_ceiling": dict(self.resource_ceiling or {}),
            "expires_at": self.expires_at.astimezone(timezone.utc).isoformat() if self.expires_at else None,
            "revoked": self.revoked,
            "purpose": self.purpose,
        }


@dataclass(frozen=True)
class ApprovalDecision:
    approval_id: str
    scope: str
    grant_id: str | None
    operation_id: str | None
    request_digest: str
    bounded: bool
    tool_id: str = ""


class ApprovalGrantStore:
    """Host grant catalog with explicit in-process revocation."""

    def __init__(
        self,
        grants: Mapping[str, ApprovalGrant] | list[ApprovalGrant] | tuple[ApprovalGrant, ...] = (),
        *,
        revoked_ids: set[str] | frozenset[str] = frozenset(),
        now: Callable[[], datetime] | None = None,
        content_identity_resolver: Callable[[Mapping[str, Any], ContentIdentity], str] | None = None,
    ) -> None:
        rows = grants.values() if isinstance(grants, Mapping) else grants
        catalog: dict[str, ApprovalGrant] = {}
        for grant in rows:
            item = grant if isinstance(grant, ApprovalGrant) else ApprovalGrant.from_mapping(grant)  # type: ignore[arg-type]
            if not item.bounded:
                raise ValueError("MCP approval grant catalog entries must be bounded.")
            if item.grant_id in catalog:
                raise ValueError("Duplicate MCP approval grant id.")
            catalog[item.grant_id] = item
        self._grants = catalog
        self._revoked_ids = set(str(item) for item in revoked_ids)
        self._now = now or (lambda: datetime.now(timezone.utc))
        self._content_identity_resolver = content_identity_resolver

    @property
    def grants(self) -> tuple[ApprovalGrant, ...]:
        return tuple(self._grants.values())

    @property
    def revoked_ids(self) -> frozenset[str]:
        return frozenset(self._revoked_ids)

    @classmethod
    def from_environment(cls) -> "ApprovalGrantStore":
        raw = os.environ.get("OEL_MCP_APPROVAL_GRANTS_JSON", os.environ.get("OEL_MCP_APPROVAL_GRANTS", "")).strip()
        if not raw:
            grants: list[ApprovalGrant] = []
        else:
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError("OEL_MCP_APPROVAL_GRANTS_JSON must contain valid JSON.") from exc
            if isinstance(parsed, Mapping):
                parsed = parsed.get("grants", [])
            if not isinstance(parsed, list):
                raise ValueError("OEL_MCP_APPROVAL_GRANTS_JSON must contain a grants list.")
            grants = [ApprovalGrant.from_mapping(item) for item in parsed if isinstance(item, Mapping)]
            if len(grants) != len(parsed):
                raise ValueError("OEL_MCP_APPROVAL_GRANTS_JSON grants must be mappings.")
        revoked = {
            item.strip()
            for item in os.environ.get("OEL_MCP_REVOKED_APPROVAL_GRANTS", "").split(",")
            if item.strip()
        }
        return cls(grants, revoked_ids=revoked)

    def get(self, grant_id: str) -> ApprovalGrant | None:
        return self._grants.get(str(grant_id))

    def revoke(self, grant_id: str) -> None:
        if str(grant_id) not in self._grants:
            raise KeyError(f"Unknown MCP approval grant: {grant_id}")
        self._revoked_ids.add(str(grant_id))

    def restore(self, grant_id: str) -> None:
        self._revoked_ids.discard(str(grant_id))

    def authorize(
        self,
        *,
        grant_id: str,
        scope: str,
        tool_id: str,
        arguments: Mapping[str, Any],
        operation_id: str = "",
        path_policy: Any = None,
        resource_metrics: Mapping[str, Any] | None = None,
        defer_resource_ceiling: bool = False,
    ) -> ApprovalGrant:
        grant = self.get(grant_id)
        if grant is None:
            raise ApprovalGrantMismatch("The MCP approval grant is not configured by the host.")
        if str(grant_id) in self._revoked_ids:
            raise ApprovalGrantRevoked("The MCP approval grant has been revoked by the host.")
        grant.validate(
            scope=scope,
            tool_id=tool_id,
            arguments=arguments,
            requested_operation_id=operation_id,
            path_policy=path_policy,
            now=self._now(),
            content_identity_resolver=self._content_identity_resolver,
            resource_metrics=resource_metrics,
            defer_resource_ceiling=defer_resource_ceiling,
        )
        return grant

    def enforce_resource_ceiling(
        self,
        *,
        grant_id: str,
        metrics: Mapping[str, Any] | None,
    ) -> None:
        grant = self.get(grant_id)
        if grant is None:
            raise ApprovalGrantMismatch("The MCP approval grant is not configured by the host.")
        if str(grant_id) in self._revoked_ids:
            raise ApprovalGrantRevoked("The MCP approval grant has been revoked by the host.")
        now = self._now()
        if grant.expires_at is not None and now.astimezone(timezone.utc) >= grant.expires_at.astimezone(timezone.utc):
            raise ApprovalGrantExpired("The configured MCP approval grant has expired.")
        grant.validate_resource_metrics(metrics)


def canonical_request_digest(tool_id: str, arguments: Mapping[str, Any]) -> str:
    """Hash exact request data while excluding mutable authorization metadata."""

    clean = {
        str(key): value
        for key, value in arguments.items()
        if str(key) not in {"approval", "trust_approval"}
    }
    payload = {"tool_id": str(tool_id), "arguments": _canonical_value(clean)}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    ).hexdigest()


def derive_operation_id(*, tool_id: str, authority_id: str, request_digest: str) -> str:
    seed = f"{tool_id}\0{authority_id}\0{request_digest}".encode()
    return "oel-op-" + hashlib.sha256(seed).hexdigest()[:48]


@dataclass(frozen=True)
class OperationBegin:
    operation_id: str
    request_digest: str
    replay_envelope: dict[str, Any] | None = None


class DurableOperationLedger:
    """Crash-safe one-record-per-operation ledger.

    A begun record is deliberately never treated as retryable. If the owner
    dies after reservation and before terminal commit, callers receive
    UnknownOperationOutcome and must inspect local evidence or reconcile
    explicitly instead of rerunning the effect.
    """

    def __init__(
        self,
        root: str | os.PathLike[str],
        *,
        now: Callable[[], datetime] | None = None,
        artifact_roots: Iterable[str | os.PathLike[str]] = (),
    ) -> None:
        self.root = _absolute_normalized_path(Path(root).expanduser())
        self.artifact_roots = _normalized_roots(artifact_roots)
        self._now = now or (lambda: datetime.now(timezone.utc))
        _ensure_directory_nofollow(self.root)
        # Keep the authorized directory alive across the whole ledger lifetime.
        # Reopening its visible name on every request would let a replacement
        # directory hide reservations and turn an unknown outcome into a retry.
        self._read_root = SecureReadRoot.open(self.root, approved_roots=(self.root,))
        self._store = DescriptorStore(self._read_root, max_record_bytes=_MAX_LEDGER_RECORD_BYTES)
        self._close_anchor = weakref.finalize(self, self._read_root.close)

    def close(self) -> None:
        """Release this ledger's retained directory capability."""
        self._close_anchor()

    @contextmanager
    def _transaction(self, path: Path) -> Iterator[None]:
        with ExitStack() as stack:
            try:
                stack.enter_context(self._store.lock(f".{path.name}.lock"))
            except (OSError, ValueError) as exc:
                raise OperationLedgerError("MCP operation ledger transaction could not be opened securely.") from exc
            yield

    def _read_record(self, path: Path) -> dict[str, Any]:
        try:
            data = json.loads(self._store.read_bytes(path.name))
        except (OSError, SecureReadError, UnicodeError, ValueError) as exc:
            raise OperationLedgerError("MCP operation ledger record is unreadable or malformed.") from exc
        if not isinstance(data, dict):
            raise OperationLedgerError("MCP operation ledger record is malformed.")
        return data

    @staticmethod
    def _record_bytes(value: Mapping[str, Any]) -> bytes:
        return (
            json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False) + "\n"
        ).encode("utf-8")

    def begin(
        self,
        *,
        operation_id: str,
        tool_id: str,
        request_digest: str,
        authority_id: str,
    ) -> OperationBegin:
        operation_id = _validated_operation_id(operation_id)
        _validated_digest(request_digest)
        path = self._record_path(operation_id)
        expected = {
            "schema_version": 1,
            "operation_id": operation_id,
            "tool_id": str(tool_id),
            "request_digest": request_digest,
            "authority_id": str(authority_id),
            "status": "begun",
            "started_utc": _timestamp(self._now()),
            "completed_utc": None,
            "envelope": None,
        }
        with self._transaction(path):
            try:
                self._store.write_new(path.name, self._record_bytes(expected))
            except FileExistsError:
                record = self._read_record(path)
                _check_identity(
                    record,
                    operation_id=operation_id,
                    tool_id=tool_id,
                    request_digest=request_digest,
                    authority_id=authority_id,
                )
                status = str(record.get("status", ""))
                if status in {"completed", "failed", "partial"} and isinstance(record.get("envelope"), dict):
                    replay = _replay_envelope(
                        record,
                        dict(record["envelope"]),
                        artifact_roots=self.artifact_roots,
                    )
                    return OperationBegin(operation_id, request_digest, replay)
                if status == "begun":
                    raise UnknownOperationOutcome(
                        "This MCP operation began but has no committed terminal result; inspect its evidence before retrying."
                    ) from None
                raise OperationLedgerError("The MCP operation ledger contains an invalid operation record.") from None
            return OperationBegin(operation_id, request_digest)

    def finish(
        self,
        *,
        operation_id: str,
        tool_id: str,
        request_digest: str,
        authority_id: str,
        envelope: Mapping[str, Any],
        status: str,
    ) -> None:
        if status not in {"completed", "failed", "partial"}:
            raise ValueError("MCP operation ledger terminal status is invalid.")
        path = self._record_path(_validated_operation_id(operation_id))
        with self._transaction(path):
            record = self._read_record(path)
            _check_identity(
                record,
                operation_id=operation_id,
                tool_id=tool_id,
                request_digest=request_digest,
                authority_id=authority_id,
            )
            if str(record.get("status")) != "begun":
                if str(record.get("status")) in {"completed", "failed", "partial"}:
                    return
                raise OperationLedgerError("The MCP operation ledger is not in a reservable state.")
            saved_envelope = _canonical_value(dict(envelope))
            artifact_paths = _artifact_paths(saved_envelope)
            artifact_identities, artifact_verification = _capture_artifact_identities(
                artifact_paths,
                artifact_roots=self.artifact_roots,
            )
            updated = dict(record)
            updated.update(
                {
                    "status": status,
                    "completed_utc": _timestamp(self._now()),
                    "envelope": saved_envelope,
                    "artifact_verification": artifact_verification,
                    "artifact_identities": artifact_identities,
                }
            )
            self._store.write_atomic(path.name, self._record_bytes(updated))

    def inspect(self, operation_id: str) -> dict[str, Any]:
        return self._read_record(self._record_path(_validated_operation_id(operation_id)))

    def _record_path(self, operation_id: str) -> Path:
        return self.root / (hashlib.sha256(operation_id.encode("utf-8")).hexdigest() + ".json")


_ARTIFACT_PATH_KEYS = frozenset(
    {
        "path",
        "output_path",
        "output_dir",
        "manifest_path",
        "packet_path",
        "product_path",
        "scenario_path",
        "operation_manifest_path",
        "source_path",
        "review_sqlite_path",
        "contact_sheet_path",
        "quality_receipt_path",
        "candidate_output_dir",
        "base_output_dir",
        "audit_output_dir",
        "brief_path",
        "audit_json_path",
        "audit_markdown_path",
        "mcp_manifest_path",
        "state_product_path",
        "report_path",
    }
)


def _artifact_paths(value: Any) -> tuple[str, ...]:
    candidates: set[str] = set()

    def visit(item: Any, key: str = "") -> None:
        if isinstance(item, Mapping):
            for child_key, child in item.items():
                name = str(child_key)
                if name in _ARTIFACT_PATH_KEYS or name in {"artifacts", "resolved_path"}:
                    visit(child, name)
                elif isinstance(child, (Mapping, list, tuple)):
                    visit(child, name)
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child, key)
        elif isinstance(item, str) and (
            key in _ARTIFACT_PATH_KEYS or key in {"artifacts", "resolved_path"}
        ):
            candidate = item.strip()
            if candidate:
                candidates.add(candidate)

    visit(value)
    return tuple(sorted(candidates))


def _capture_artifact_identities(
    paths: tuple[str, ...],
    *,
    artifact_roots: tuple[Path, ...],
) -> tuple[list[dict[str, Any]], str]:
    if not paths:
        return [], "none"
    identities: list[dict[str, Any]] = []
    saw_directory = False
    unverified = False
    for raw in paths:
        try:
            path = _absolute_normalized_path(Path(raw).expanduser())
        except (TypeError, ValueError):
            unverified = True
            continue
        if not _path_is_under_roots(path, artifact_roots):
            unverified = True
            continue
        try:
            details = os.stat(path, follow_symlinks=False)
        except OSError:
            unverified = True
            continue
        if stat.S_ISLNK(details.st_mode):
            unverified = True
            continue
        if stat.S_ISDIR(details.st_mode):
            # A directory alone is not an immutable artifact identity.  Keep
            # processing listed files from the same result.
            saw_directory = True
            continue
        if not stat.S_ISREG(details.st_mode):
            unverified = True
            continue
        try:
            digest, size = _file_identity_nofollow(path)
        except (OSError, ApprovalGrantMismatch):
            unverified = True
            continue
        identities.append({"path": str(path), "sha256": digest, "size": int(size)})
    if unverified or (saw_directory and not identities):
        return identities, "unverified"
    return identities, "verified" if identities else "none"


def _replay_envelope(
    record: Mapping[str, Any],
    envelope: dict[str, Any],
    *,
    artifact_roots: tuple[Path, ...],
) -> dict[str, Any]:
    verification = str(record.get("artifact_verification", ""))
    if verification == "none":
        return envelope
    if verification == "verified":
        identities = record.get("artifact_identities")
        if isinstance(identities, list) and _artifact_identities_match(
            identities,
            artifact_roots=artifact_roots,
        ):
            return envelope
        message = "The stored MCP result references an artifact that is missing or has changed."
    elif verification == "unverified":
        message = "The stored MCP result has no verifiable current artifact identity."
    else:
        message = "The stored MCP result has an invalid artifact-verification state."
    return _mark_stale_envelope(envelope, message)


def _artifact_identities_match(identities: list[Any], *, artifact_roots: tuple[Path, ...]) -> bool:
    for item in identities:
        if not isinstance(item, Mapping):
            return False
        try:
            path = _absolute_normalized_path(Path(str(item.get("path", ""))).expanduser())
        except (TypeError, ValueError):
            return False
        expected = str(item.get("sha256", "")).strip().lower()
        if not _path_is_under_roots(path, artifact_roots) or not _SHA256.fullmatch(expected):
            return False
        try:
            digest, size = _file_identity_nofollow(path)
            if size != int(item.get("size", -1)) or digest != expected:
                return False
        except (OSError, ValueError, ApprovalGrantMismatch):
            return False
    return True


def _mark_stale_envelope(envelope: dict[str, Any], message: str) -> dict[str, Any]:
    replay = dict(envelope)
    replay["status"] = "partial" if str(replay.get("status")) == "completed" else replay.get("status", "partial")
    evidence = dict(replay.get("evidence") or {})
    evidence["complete"] = False
    replay["evidence"] = evidence
    replay["error"] = {"type": "StoredArtifactStale", "message": message}
    audit = replay.get("audit")
    if isinstance(audit, Mapping):
        updated_audit = dict(audit)
        updated_audit["status"] = replay["status"]
        replay["audit"] = updated_audit
    return replay


def _parse_expiry(value: Any) -> datetime:
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        result = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError("Approval grant expires_at must be an ISO-8601 timestamp.") from exc
    if result.tzinfo is None:
        raise ValueError("Approval grant expires_at must include a timezone.")
    return result.astimezone(timezone.utc)


def _validated_operation_id(value: str) -> str:
    text = str(value).strip()
    if not _OPERATION_ID.fullmatch(text):
        raise ValueError("MCP operation id is invalid.")
    return text


def _validated_digest(value: str) -> str:
    text = str(value).strip().lower()
    if not _SHA256.fullmatch(text):
        raise ValueError("MCP request digest is invalid.")
    return text


def _timestamp(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).replace(microsecond=0).isoformat()


def _canonical_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _canonical_value(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [_canonical_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("MCP request values must be finite.")
        return value
    return str(value)


def _lookup(mapping: Mapping[str, Any], dotted: str) -> Any:
    current: Any = mapping
    for part in str(dotted).split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def _content_identity(
    arguments: Mapping[str, Any],
    identity: ContentIdentity,
    *,
    path_policy: Any,
    resolver: Callable[[Mapping[str, Any], ContentIdentity], str] | None,
) -> str:
    value = _lookup(arguments, identity.argument)
    if value is None:
        raise ApprovalGrantMismatch(f"Reviewed content argument {identity.argument} is missing.")
    if identity.kind == "review_digest":
        if resolver is None:
            raise ApprovalGrantMismatch("The host did not provide a trusted review-digest resolver.")
        actual = str(resolver(arguments, identity)).strip().lower()
        if not _SHA256.fullmatch(actual):
            raise ApprovalGrantMismatch("The host review-digest resolver returned an invalid digest.")
        return actual
    if path_policy is None:
        raise ApprovalGrantMismatch("A content-bound approval requires an authorized read-root policy.")
    raw_path = Path(value).expanduser()
    lexical_path = raw_path if raw_path.is_absolute() else raw_path.absolute()
    if any(component.is_symlink() for component in _path_components(lexical_path)):
        raise ApprovalGrantMismatch("The reviewed content path contains a symlink.")
    try:
        path = path_policy.resolve_read(value, kind="file")
    except Exception as exc:
        raise ApprovalGrantMismatch("The reviewed content path is not authorized.") from exc
    return _sha256_file_nofollow(path)


def _sha256_file_nofollow(path: Path) -> str:
    digest, _size = _file_identity_nofollow(path)
    return digest


def _file_identity_nofollow(path: Path) -> tuple[str, int]:
    path = _absolute_normalized_path(path)
    if path.name in {"", ".", ".."}:
        raise ApprovalGrantMismatch("The reviewed content path is not a regular file.")
    try:
        with SecureReadRoot.open(path.parent, approved_roots=(path.parent,)) as reader:
            # Hash in bounded chunks; the captured size is the growth ceiling,
            # not a request to materialize a potentially large artifact.
            size = reader.stat(path.name).st_size
            record = reader.file_record(path.name, max_bytes=size)
            return str(record["sha256"]), int(record["bytes"])
    except (OSError, SecureReadError) as exc:
        raise ApprovalGrantMismatch("The reviewed content could not be read securely.") from exc


def _resource_value(arguments: Mapping[str, Any], name: str) -> float | None:
    value = _lookup(arguments, name)
    if value is None and "." not in str(name):
        matches: list[Any] = []

        def visit(item: Any) -> None:
            if isinstance(item, Mapping):
                for key, child in item.items():
                    if str(key) == str(name):
                        matches.append(child)
                    visit(child)
            elif isinstance(item, list):
                for child in item:
                    visit(child)

        visit(arguments)
        value = matches[0] if matches else None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    if isinstance(value, (list, tuple)) and str(name).endswith(("_count", "_rows", "_bytes")):
        return float(len(value))
    return None


def _absolute_normalized_path(path: Path) -> Path:
    if not path.is_absolute():
        raise ValueError("MCP path must be absolute.")
    return Path(os.path.abspath(os.fspath(path)))


def _normalized_roots(roots: Iterable[str | os.PathLike[str]]) -> tuple[Path, ...]:
    normalized: list[Path] = []
    for raw in roots:
        root = _absolute_normalized_path(Path(raw).expanduser())
        if root not in normalized:
            normalized.append(root)
    return tuple(normalized)


def _path_is_under_roots(path: Path, roots: tuple[Path, ...]) -> bool:
    return any(_is_relative_to(path, root) for root in roots)


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _path_components(path: Path) -> list[Path]:
    absolute = path if path.is_absolute() else path.absolute()
    current = Path(absolute.root)
    components: list[Path] = []
    for part in absolute.parts[1:]:
        current = current / part
        components.append(current)
    return components


def _ensure_directory_nofollow(path: Path) -> None:
    path = _absolute_normalized_path(path)
    if path == Path(path.anchor):
        raise PermissionError("MCP operation ledger root must not be the filesystem root.")
    try:
        directory_fd = _open_directory_nofollow(path, create=True)
    except OSError as exc:
        raise PermissionError("MCP operation ledger path could not be opened securely.") from exc
    try:
        details = os.fstat(directory_fd)
        if not stat.S_ISDIR(details.st_mode):
            raise PermissionError("MCP operation ledger path is not a directory.")
        try:
            owner = os.getuid()
        except AttributeError:
            owner = None
        if owner is not None and details.st_uid != owner:
            raise PermissionError("MCP operation ledger directory is not owned by the current host user.")
        os.fchmod(directory_fd, 0o700)
        os.fsync(directory_fd)
        if os.fstat(directory_fd).st_mode & 0o077:
            raise PermissionError("MCP operation ledger directory is accessible by another user.")
    finally:
        os.close(directory_fd)


def _open_directory_nofollow(path: Path, *, create: bool) -> int:
    path = _absolute_normalized_path(path)
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    current_fd = os.open(os.path.sep, flags)
    try:
        for part in path.parts[1:]:
            try:
                next_fd = os.open(part, flags, dir_fd=current_fd)
            except FileNotFoundError:
                if not create:
                    raise
                try:
                    os.mkdir(part, mode=0o700, dir_fd=current_fd)
                except FileExistsError:
                    pass
                else:
                    os.fsync(current_fd)
                next_fd = os.open(part, flags, dir_fd=current_fd)
            os.close(current_fd)
            current_fd = next_fd
        return current_fd
    except BaseException:
        os.close(current_fd)
        raise


def _check_identity(
    record: Mapping[str, Any],
    *,
    operation_id: str,
    tool_id: str,
    request_digest: str,
    authority_id: str,
) -> None:
    if (
        str(record.get("operation_id", "")) != operation_id
        or str(record.get("tool_id", "")) != str(tool_id)
        or str(record.get("request_digest", "")) != request_digest
        or str(record.get("authority_id", "")) != str(authority_id)
    ):
        raise OperationConflictError("The MCP operation id was reused with changed tool, authority, or arguments.")


__all__ = [
    "ApprovalDecision",
    "ApprovalGrant",
    "ApprovalGrantError",
    "ApprovalGrantExpired",
    "ApprovalGrantMismatch",
    "ApprovalGrantRevoked",
    "ApprovalGrantStore",
    "ContentIdentity",
    "DurableOperationLedger",
    "OperationBegin",
    "OperationConflictError",
    "OperationLedgerError",
    "UnknownOperationOutcome",
    "canonical_request_digest",
    "derive_operation_id",
]
