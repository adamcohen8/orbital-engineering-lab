from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import secrets
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Event
from typing import Any, Callable, Mapping

import yaml

from integrations.oel_mcp.approval_grants import (
    ApprovalDecision,
    ApprovalGrant,
    ApprovalGrantStore,
    canonical_request_digest,
)
from integrations.oel_mcp.policy import MCPPathPolicy
from integrations.oel_mcp.recovery import annotate_error
from integrations.oel_mcp.secure_writes import (
    SecureOutputTransaction,
    SecureWriteConflict,
    SecureWriteRoot,
    bind_output_writer,
    staged_output,
)
from sim.api import SimulationWorkspace
from sim.config import scenario_config_from_dict
from sim.resource_limits import apply_resource_profile_to_config_dict, estimate_resource_requirements
from sim.security import ConfigPathPolicy
from sim.security.sealed_mode import validate_sealed_mode

MAX_SCENARIO_BYTES = 2_000_000
M4_RESOURCE_PROFILES = ("laptop-safe", "standard")
EXECUTION_MANIFEST_NAME = "mcp_execution_manifest.json"
_SAFE_ARTIFACT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$")


class MCPExecutionCancelled(RuntimeError):
    """Raised at a deterministic workflow callback boundary after MCP cancellation."""


@dataclass(frozen=True)
class ExecutionApprovalPolicy:
    """Standing IDs plus host-configured, content-bound approval grants.

    The three ID fields remain the v1 constructor and environment contract.
    Bounded grants are additional server configuration; request data can only
    select a preconfigured grant and can never define one.
    """

    write_approval_ids: frozenset[str] = frozenset()
    execution_approval_ids: frozenset[str] = frozenset()
    trust_approval_ids: frozenset[str] = frozenset()
    bounded_grants: tuple[ApprovalGrant, ...] = ()
    revoked_grant_ids: frozenset[str] = frozenset()
    grant_store: ApprovalGrantStore | None = None

    def __post_init__(self) -> None:
        if self.grant_store is None:
            object.__setattr__(
                self,
                "grant_store",
                ApprovalGrantStore(self.bounded_grants, revoked_ids=self.revoked_grant_ids),
            )

    @classmethod
    def configured(cls) -> ExecutionApprovalPolicy:
        store = ApprovalGrantStore.from_environment()
        return cls(
            write_approval_ids=_approval_ids("OEL_MCP_WRITE_APPROVAL_IDS"),
            execution_approval_ids=_approval_ids("OEL_MCP_EXECUTION_APPROVAL_IDS"),
            trust_approval_ids=_approval_ids("OEL_MCP_TRUST_APPROVAL_IDS"),
            bounded_grants=tuple(store.grants),
            revoked_grant_ids=frozenset(store.revoked_ids),
            grant_store=store,
        )

    def authorize(
        self,
        approval: dict[str, Any] | None,
        *,
        executes: bool,
        tool_id: str = "",
        arguments: dict[str, Any] | None = None,
        path_policy: MCPPathPolicy | None = None,
    ) -> ApprovalDecision:
        if not isinstance(approval, dict):
            raise annotate_error(
                PermissionError("Operator approval metadata is required for this operation."),
                "approval.required",
            )
        approval_id = str(approval.get("approval_id", approval.get("grant_id", ""))).strip()
        explicit_grant_id = str(approval.get("grant_id", "")).strip()
        grant_id = explicit_grant_id or approval_id
        scope = str(approval.get("scope", "")).strip()
        expected_scope = "execute" if executes else "write"
        requested_operation_id = str(approval.get("operation_id", "")).strip()
        request_arguments = dict(arguments or {})
        store = self.grant_store
        grant = store.get(grant_id) if store is not None and grant_id else None
        if explicit_grant_id and grant is None:
            raise annotate_error(
                PermissionError("The explicitly selected MCP approval grant is not configured by the host."),
                "approval.required",
            )
        if grant is not None:
            if scope != expected_scope:
                raise annotate_error(
                    PermissionError("The approval scope does not match this operation effect."),
                    "approval.required",
                )
            if approval_id and approval_id != grant_id:
                raise annotate_error(
                    PermissionError("The approval id and configured grant id must match."),
                    "approval.required",
                )
            try:
                store.authorize(
                    grant_id=grant_id,
                    scope=scope,
                    tool_id=str(tool_id),
                    arguments=request_arguments,
                    # A host-configured operation is a reviewed plan identity;
                    # the request must supply it and the grant checks exact equality.
                    operation_id=requested_operation_id,
                    path_policy=path_policy,
                    # Preparation occurs inside the handler operation.  Defer
                    # only the host-metric ceiling; exact tool/argument/content
                    # binding is still checked before entering the operation.
                    defer_resource_ceiling=True,
                )
            except PermissionError as exc:
                annotate_error(exc, "approval.required")
                raise
            return ApprovalDecision(
                approval_id=grant_id,
                scope=scope,
                grant_id=grant_id,
                operation_id=requested_operation_id or None,
                request_digest=canonical_request_digest(tool_id, request_arguments),
                bounded=grant.bounded,
                tool_id=str(tool_id),
            )
        allowed = self.execution_approval_ids if executes else self.write_approval_ids
        if scope != expected_scope or not approval_id or approval_id not in allowed:
            raise annotate_error(
                PermissionError("The operation is not enabled by the server's operator approval policy."),
                "approval.required",
            )
        return ApprovalDecision(
            approval_id=approval_id,
            scope=scope,
            grant_id=None,
            operation_id=requested_operation_id or None,
            request_digest=canonical_request_digest(tool_id, request_arguments),
            bounded=False,
            tool_id=str(tool_id),
        )

    def authorize_trust(
        self,
        approval: dict[str, Any] | None,
        *,
        tool_id: str = "",
        arguments: dict[str, Any] | None = None,
        path_policy: MCPPathPolicy | None = None,
    ) -> ApprovalDecision:
        if not isinstance(approval, dict):
            raise annotate_error(
                PermissionError("Operator trust approval is required before importing scenario plugins."),
                "trust.required",
            )
        approval_id = str(approval.get("approval_id", approval.get("grant_id", ""))).strip()
        explicit_grant_id = str(approval.get("grant_id", "")).strip()
        grant_id = explicit_grant_id or approval_id
        requested_operation_id = str(approval.get("operation_id", "")).strip()
        scope = str(approval.get("scope", "")).strip()
        store = self.grant_store
        grant = store.get(grant_id) if store is not None and grant_id else None
        if explicit_grant_id and grant is None:
            raise annotate_error(
                PermissionError("The explicitly selected MCP approval grant is not configured by the host."),
                "trust.required",
            )
        if grant is not None:
            if scope != "trust":
                raise annotate_error(
                    PermissionError("The approval scope does not authorize plugin trust."),
                    "trust.required",
                )
            if approval_id and approval_id != grant_id:
                raise annotate_error(
                    PermissionError("The approval id and configured grant id must match."),
                    "trust.required",
                )
            try:
                store.authorize(
                    grant_id=grant_id,
                    scope="trust",
                    tool_id=str(tool_id),
                    arguments=dict(arguments or {}),
                    operation_id=requested_operation_id,
                    path_policy=path_policy,
                    # Preparation occurs inside the handler operation.  Defer
                    # only the host-metric ceiling; exact tool/argument/content
                    # binding is still checked before entering the operation.
                    defer_resource_ceiling=True,
                )
            except PermissionError as exc:
                annotate_error(exc, "trust.required")
                raise
            return ApprovalDecision(
                approval_id=grant_id,
                scope="trust",
                grant_id=grant_id,
                operation_id=requested_operation_id or None,
                request_digest=canonical_request_digest(tool_id, dict(arguments or {})),
                bounded=grant.bounded,
                tool_id=str(tool_id),
            )
        if str(approval.get("scope", "")).strip() != "trust" or not approval_id or approval_id not in self.trust_approval_ids:
            raise annotate_error(
                PermissionError("Plugin trust is not enabled by the server's operator approval policy."),
                "trust.required",
            )
        return ApprovalDecision(
            approval_id=approval_id,
            scope="trust",
            grant_id=None,
            operation_id=requested_operation_id or None,
            request_digest=canonical_request_digest(tool_id, dict(arguments or {})),
            bounded=False,
            tool_id=str(tool_id),
        )

    def enforce_resource_ceiling(
        self,
        decision: ApprovalDecision | None,
        metrics: Mapping[str, Any] | None,
    ) -> None:
        """Enforce a configured grant ceiling against trusted host metrics.

        Bounded request authorization intentionally defers this check because
        preparation may be required to compute the actual resource plan.
        Handlers must call this before the first effect; unsupported paths pass
        no metrics and fail closed.
        """

        if decision is None or not decision.grant_id:
            return
        store = self.grant_store
        if store is None:
            raise PermissionError("Host resource-ceiling configuration is unavailable.")
        store.enforce_resource_ceiling(grant_id=decision.grant_id, metrics=metrics)

    def require(
        self,
        approval: dict[str, Any] | None,
        *,
        executes: bool,
        tool_id: str = "",
        arguments: dict[str, Any] | None = None,
        path_policy: MCPPathPolicy | None = None,
    ) -> str:
        return self.authorize(
            approval,
            executes=executes,
            tool_id=tool_id,
            arguments=arguments,
            path_policy=path_policy,
        ).approval_id

    def require_trust(
        self,
        approval: dict[str, Any] | None,
        *,
        tool_id: str = "",
        arguments: dict[str, Any] | None = None,
        path_policy: MCPPathPolicy | None = None,
    ) -> str:
        return self.authorize_trust(
            approval,
            tool_id=tool_id,
            arguments=arguments,
            path_policy=path_policy,
        ).approval_id


@dataclass(frozen=True)
class PreparedScenario:
    source_path: Path
    output_dir: Path
    resource_profile: str
    raw_sha256: str
    normalized_sha256: str
    validation_id: str
    config: Any
    config_dict: dict[str, Any]


def bounded_config_path_policy(
    *,
    config_path: str | Path,
    path_policy: MCPPathPolicy,
) -> ConfigPathPolicy:
    """Build a config policy from only the MCP adapter's explicit roots.

    ``ConfigPathPolicy.default`` adds repository, workspace, and config-directory
    roots for ordinary CLI use. MCP callers have a narrower grant, so those
    implicit roots must not be inherited during config loading or execution.
    """

    source = Path(config_path).expanduser().resolve()
    return ConfigPathPolicy(
        config_path=source,
        workspace_root=Path(__file__).resolve().parents[2],
        read_roots=tuple(path_policy.read_roots),
        write_roots=tuple(path_policy.write_roots),
        allow_external_config_paths=False,
        allow_external_ai_prompt_files=False,
        restrict_ai_prompt_to_read_roots=True,
        # MCP is a hostile-boundary adapter.  It must fail closed on hosts
        # without descriptor-bound input primitives instead of inheriting the
        # trusted local CLI's bounded path-compatibility fallback.
        require_secure_reads=True,
    )


def prepare_scenario(
    *,
    config_path: str | Path,
    output_dir: str | Path,
    resource_profile: str,
    path_policy: MCPPathPolicy,
) -> PreparedScenario:
    source, raw_bytes = path_policy.read_authorized_bytes(
        config_path,
        maximum=MAX_SCENARIO_BYTES,
    )
    output = path_policy.resolve_write(output_dir)
    profile = str(resource_profile).strip()
    if profile not in M4_RESOURCE_PROFILES:
        raise ValueError("M4 requires the laptop-safe or standard resource profile.")
    raw = yaml.safe_load(raw_bytes)
    if not isinstance(raw, dict):
        raise ValueError("Scenario config must be a YAML mapping.")
    prepared = apply_resource_profile_to_config_dict(raw, profile)
    outputs = dict(prepared.get("outputs", {}) or {})
    outputs["output_dir"] = str(output)
    review = dict(outputs.get("review", {}) or {})
    review["enabled"] = True
    review.setdefault("detail", "standard")
    outputs["review"] = review
    stats = dict(outputs.get("stats", {}) or {})
    stats["print_summary"] = False
    outputs["stats"] = stats
    prepared["outputs"] = outputs
    config_policy = bounded_config_path_policy(config_path=source, path_policy=path_policy)
    config = scenario_config_from_dict(prepared, source_path=source, path_policy=config_policy)
    sealed_errors = validate_sealed_mode(config)
    ai_report = dict(config.outputs.ai_report or {})
    if bool(ai_report.get("enabled", False)) and not bool(ai_report.get("dry_run", False)):
        sealed_errors.append(
            "outputs.ai_report.enabled: MCP execution forbids all AI-provider calls; "
            "disable the report or use dry_run: true."
        )
    ai_config = dict(config.outputs.ai_config or {})
    if ai_config and bool(ai_config.get("enabled", True)) and not bool(ai_config.get("dry_run", False)):
        sealed_errors.append(
            "outputs.ai_config.enabled: MCP execution forbids all AI-provider calls; "
            "disable the assistant or use dry_run: true."
        )
    if sealed_errors:
        # CLI retention switches are not arguments of the MCP contract.
        sealed_errors = [message.replace(
            " or pass --allow-high-detail-outputs for approved retention", ""
        ) for message in sealed_errors]
        raise annotate_error(
            ValueError("MCP sealed execution policy failed:\n- " + "\n- ".join(sealed_errors)),
            "scenario.policy_blocked",
        )
    normalized = config.to_dict()
    normalized_sha256 = _sha256_json(normalized)
    validation_id = f"oel-m4-validation-v1:{normalized_sha256}"
    return PreparedScenario(
        source_path=source,
        output_dir=output,
        resource_profile=profile,
        raw_sha256=hashlib.sha256(raw_bytes).hexdigest(),
        normalized_sha256=normalized_sha256,
        validation_id=validation_id,
        config=config,
        config_dict=prepared,
    )


def validate_prepared_scenario(
    prepared: PreparedScenario,
    *,
    path_policy: MCPPathPolicy,
    trust_plugins: bool,
) -> dict[str, Any]:
    workspace = SimulationWorkspace(
        workspace_root=Path(__file__).resolve().parents[2],
        read_roots=path_policy.read_roots,
        write_roots=path_policy.write_roots,
        allow_config_dir_writes=False,
    )
    safe = workspace.validate(prepared.config, import_plugins=False)
    trusted: dict[str, Any] = {
        "ok": None,
        "status": "not_run",
        "reason": "safe_validation_failed" if not bool(safe.get("ok")) else "trust_not_granted",
    }
    if bool(safe.get("ok")) and trust_plugins:
        trusted = workspace.validate(prepared.config, import_plugins=True)
    trusted_ok = bool(trusted.get("ok")) if trust_plugins else False
    return {
        "status": "validated" if trusted_ok else "safe_only" if bool(safe.get("ok")) else "failed",
        "safe_validation": safe,
        "trusted_validation": trusted,
        "execution_ready": trusted_ok,
        "validation_id": prepared.validation_id if trusted_ok else "",
    }


def resource_estimate(prepared: PreparedScenario) -> dict[str, Any]:
    estimate = asdict(estimate_resource_requirements(prepared.config))
    estimate["action"] = (
        "refuse"
        if estimate["risk"] == "unsafe"
        else ("advisory" if estimate["risk"] in {"moderate", "heavy"} else "proceed")
    )
    return estimate


def require_safe_resource_estimate(estimate: dict[str, Any]) -> None:
    if str(estimate.get("risk")) == "unsafe" or str(estimate.get("action")) == "refuse":
        raise ValueError("The scenario failed the active resource-safety preflight.")


def ensure_new_output_dir(
    path: Path,
    *,
    approved_roots: tuple[str | Path, ...] | None = None,
) -> None:
    if approved_roots is None:
        _ensure_directory_nofollow(path, require_empty=True)
        return
    with SecureWriteRoot.open(
        path,
        approved_roots=approved_roots,
        create=True,
        require_empty=True,
    ):
        return


def write_materialized_config(
    prepared: PreparedScenario,
    *,
    writer: SecureWriteRoot | None = None,
    output_dir: Path | None = None,
) -> Path:
    target_dir = Path(output_dir) if output_dir is not None else prepared.output_dir
    config_dict = dict(prepared.config_dict)
    outputs = dict(config_dict.get("outputs", {}) or {})
    outputs["output_dir"] = str(target_dir)
    config_dict["outputs"] = outputs
    serialized = yaml.safe_dump(config_dict, sort_keys=False, allow_unicode=False).encode("utf-8")
    if writer is not None:
        return writer.write_bytes("mcp_execution_config.yaml", serialized)
    target = target_dir / "mcp_execution_config.yaml"
    _write_child_nofollow(target_dir, target.name, serialized)
    return target


def cancellation_callback(cancel_event: Event | None) -> Callable[..., None]:
    def check(*_args: Any, **_kwargs: Any) -> None:
        if cancel_event is not None and cancel_event.is_set():
            raise MCPExecutionCancelled("The MCP request was cancelled at a deterministic workflow boundary.")

    return check


def write_execution_manifest(
    output_dir: Path,
    payload: dict[str, Any],
    *,
    filename: str = EXECUTION_MANIFEST_NAME,
    approved_roots: tuple[str | Path, ...] | None = None,
    writer: SecureWriteRoot | None = None,
    allow_update: bool = True,
) -> Path:
    serialized = (json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n").encode("utf-8")
    if writer is not None:
        try:
            return writer.write_bytes(filename, serialized)
        except SecureWriteConflict:
            if not allow_update:
                raise
            return writer.update_existing_bytes(filename, serialized)
    roots = approved_roots if approved_roots is not None else (output_dir,)
    with SecureWriteRoot.open(output_dir, approved_roots=roots, create=False) as handle:
        try:
            return handle.write_bytes(filename, serialized)
        except SecureWriteConflict:
            if not allow_update:
                raise
            return handle.update_existing_bytes(filename, serialized)


def secure_staged_output(
    path_policy: MCPPathPolicy,
    path: str | Path,
    *,
    require_empty: bool = True,
    staging_parent: str | Path | None = None,
    active_opid: str | None = None,
):
    """Create a private staging transaction under an explicit MCP write grant."""

    target = path_policy.resolve_write(path)
    return staged_output(
        target,
        approved_roots=path_policy.write_roots,
        require_empty=require_empty,
        staging_parent=staging_parent,
        active_opid=active_opid,
    )


@contextlib.contextmanager
def secure_staged_file(
    path_policy: MCPPathPolicy,
    path: str | Path,
    *,
    active_opid: str | None = None,
):
    """Stage a path-facing producer that creates one file.

    The final parent may already contain unrelated evidence, so the
    transaction is rooted at that parent with ``require_empty=False``.  The
    producer receives a private staging path; publication still refuses an
    existing destination entry and rejects symlinks or special files.
    """

    target = path_policy.resolve_write(path)
    with secure_staged_output(
        path_policy,
        target.parent,
        require_empty=False,
        active_opid=active_opid,
    ) as transaction:
        yield transaction, transaction.staging_path / target.name


@contextlib.contextmanager
def secure_writer_context(
    path_policy: MCPPathPolicy,
    path: str | Path,
    *,
    create: bool = True,
    require_empty: bool = False,
    active_opid: str | None = None,
):
    """Open an approved writer and bind it for helpers that preserve logical paths."""

    with path_policy.open_secure_writer(
        path,
        create=create,
        require_empty=require_empty,
        active_opid=active_opid,
    ) as writer:
        with bind_output_writer(writer):
            yield writer


def _write_child_nofollow(
    directory: Path,
    filename: str,
    payload: bytes,
    *,
    atomic: bool = False,
) -> None:
    if Path(filename).name != filename:
        raise ValueError("MCP artifact filename must be a single path component.")
    directory_fd = _open_directory_nofollow(directory, create=False)
    temporary_name = filename
    try:
        if atomic:
            temporary_name = f".{filename}.{secrets.token_hex(12)}.tmp"
        file_flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_NOFOLLOW", 0)
        if atomic:
            file_flags |= os.O_EXCL
        file_fd = os.open(temporary_name, file_flags, 0o600, dir_fd=directory_fd)
        with os.fdopen(file_fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if atomic:
            os.replace(
                temporary_name,
                filename,
                src_dir_fd=directory_fd,
                dst_dir_fd=directory_fd,
            )
    finally:
        if atomic:
            try:
                os.unlink(temporary_name, dir_fd=directory_fd)
            except FileNotFoundError:
                pass
        os.close(directory_fd)


def _open_directory_nofollow(directory: Path, *, create: bool) -> int:
    """Open an absolute directory one component at a time without symlinks."""

    path = Path(directory)
    if not path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts[1:]):
        raise ValueError("MCP artifact directory must be an absolute normalized path")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    current_fd = os.open(os.path.sep, flags)
    try:
        for part in path.parts[1:]:
            try:
                next_fd = os.open(part, flags, dir_fd=current_fd)
            except FileNotFoundError:
                if not create:
                    raise
                os.mkdir(part, mode=0o700, dir_fd=current_fd)
                next_fd = os.open(part, flags, dir_fd=current_fd)
            os.close(current_fd)
            current_fd = next_fd
        return current_fd
    except Exception:
        os.close(current_fd)
        raise


def _ensure_directory_nofollow(directory: Path, *, require_empty: bool) -> None:
    directory_fd = _open_directory_nofollow(directory, create=True)
    try:
        if require_empty and os.listdir(directory_fd):
            raise FileExistsError("The approved output directory must be new or empty.")
    finally:
        os.close(directory_fd)


def manifest_base(
    *,
    tool_id: str,
    approval_id: str,
    prepared: PreparedScenario | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "tool_id": tool_id,
        "approval_id": approval_id,
        "status": "running",
        "started_utc": _utc_now(),
        "completed_utc": None,
        "cancelled": False,
        "artifacts_complete": False,
    }
    if prepared is not None:
        payload.update(
            {
                "source_config_sha256": prepared.raw_sha256,
                "normalized_config_sha256": prepared.normalized_sha256,
                "validation_id": prepared.validation_id,
                "resource_profile": prepared.resource_profile,
                "output_dir": str(prepared.output_dir),
            }
        )
    return payload


def complete_manifest(
    manifest: dict[str, Any],
    *,
    status: str,
    cancelled: bool = False,
    artifacts: list[str] | None = None,
    artifact_files: list[str] | None = None,
    writer: Any | None = None,
    error_type: str = "",
) -> dict[str, Any]:
    """Complete a manifest while keeping logical paths separate from hash sources.

    Protected MCP producers write into private staging roots, so the path that
    belongs in the returned manifest can differ from the physical file that is
    hashed.  ``artifact_files`` is an optional one-to-one companion list for
    that case.  The public ``artifacts``-only form retains its historical
    behavior and hashes the paths it reports.  A supplied ``writer`` instead
    inventories each logical artifact through its retained directory
    descriptor, keeping hashes bound across visible path replacement.
    """

    logical_inputs = [str(path) for path in list(artifacts or []) if str(path)]
    physical_inputs = (
        [str(path) for path in list(artifact_files)]
        if artifact_files is not None
        else logical_inputs
    )
    if len(logical_inputs) != len(physical_inputs):
        raise ValueError("Logical and physical artifact lists must have the same length.")

    # Preserve the old duplicate suppression while retaining the physical
    # source paired with the first occurrence of each logical artifact.
    artifact_pairs: list[tuple[str, str]] = []
    seen_logical: set[str] = set()
    for logical_path, physical_path in zip(logical_inputs, physical_inputs):
        if logical_path in seen_logical:
            continue
        seen_logical.add(logical_path)
        artifact_pairs.append((logical_path, physical_path))
    artifact_paths = [logical_path for logical_path, _ in artifact_pairs]
    artifact_records: list[dict[str, Any]] = []
    for logical_path, physical_path in artifact_pairs:
        if writer is not None:
            logical = Path(os.path.abspath(Path(logical_path).expanduser()))
            relative = logical.relative_to(writer.logical_root)
            try:
                record = writer.file_record(relative)
            except FileNotFoundError:
                continue
            artifact_records.append({"path": logical_path, **record})
            continue
        path = Path(physical_path)
        if not path.is_file() or path.is_symlink():
            continue
        artifact_records.append(
            {
                "path": logical_path,
                "bytes": int(path.stat().st_size),
                "sha256": _sha256_file(path),
            }
        )
    manifest.update(
        {
            "status": status,
            "completed_utc": _utc_now(),
            "cancelled": cancelled,
            "artifacts": artifact_paths,
            "artifact_records": artifact_records,
            "artifacts_complete": (
                status == "completed"
                and not cancelled
                and bool(artifact_paths)
                and len(artifact_records) == len(artifact_paths)
            ),
        }
    )
    if error_type:
        manifest["error_type"] = error_type
    return manifest


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_artifact_id(value: str) -> str:
    artifact_id = str(value or "").strip()
    if not _SAFE_ARTIFACT_ID.fullmatch(artifact_id):
        raise ValueError("Artifact id must contain only letters, numbers, dot, underscore, or hyphen.")
    return artifact_id


def _approval_ids(name: str) -> frozenset[str]:
    return frozenset(item.strip() for item in os.environ.get(name, "").split(",") if item.strip())


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


__all__ = [
    "EXECUTION_MANIFEST_NAME",
    "M4_RESOURCE_PROFILES",
    "MAX_SCENARIO_BYTES",
    "ExecutionApprovalPolicy",
    "MCPExecutionCancelled",
    "PreparedScenario",
    "bounded_config_path_policy",
    "cancellation_callback",
    "complete_manifest",
    "ensure_new_output_dir",
    "manifest_base",
    "prepare_scenario",
    "require_safe_resource_estimate",
    "resource_estimate",
    "safe_artifact_id",
    "secure_staged_file",
    "secure_staged_output",
    "secure_writer_context",
    "SecureOutputTransaction",
    "validate_prepared_scenario",
    "write_execution_manifest",
    "write_materialized_config",
]
