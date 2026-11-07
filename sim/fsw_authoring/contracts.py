"""Public contracts for the OEL FSW Authoring Kit.

The authoring kit stops at local component checks and one deterministic smoke
scenario.  Controller comparison, qualification, baseline promotion, mission-specific
integrations, cFS/SIL, and evidence packaging belong to the private FSWDK layer.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

AUTHORING_CONTRACT_VERSION = 1
CANDIDATE_SCHEMA_ID = "oel.fsw_authoring.candidate.v1"
CAPABILITIES_SCHEMA_ID = "oel.fsw_authoring.capabilities.v1"
SCAFFOLD_RECEIPT_SCHEMA_ID = "oel.fsw_authoring.scaffold_receipt.v1"
WORK_ORDER_SCHEMA_ID = "oel.fsw_authoring.work_order.v1"
EXECUTION_PLAN_SCHEMA_ID = "oel.fsw_authoring.execution_plan.v1"
VALIDATION_RECEIPT_SCHEMA_ID = "oel.fsw_authoring.validation_receipt.v1"
TEST_RESULT_SCHEMA_ID = "oel.fsw_authoring.test_results.v1"
RUN_MANIFEST_SCHEMA_ID = "oel.fsw_authoring.run_manifest.v1"

AUTHORING_STATUSES = frozenset({"ready", "invalid", "passed", "failed", "incomplete", "cancelled"})
AUTHORING_NON_CLAIMS = (
    "Results apply only to the exact candidate, component tests, scenario, and deterministic OEL models recorded.",
    "The OEL FSW Authoring Kit does not perform Controller Bench comparison, tuning, or qualification.",
    "Authoring receipts are not flight qualification, certification, hardware readiness, or operational approval.",
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_value(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tree(path: str | Path, *, suffixes: frozenset[str] | None = None) -> str:
    root = Path(path).resolve()
    if root.is_file():
        return sha256_file(root)
    rows: list[dict[str, str]] = []
    ignored = {".oel", "__pycache__", ".pytest_cache", ".ruff_cache"}
    candidates: list[Path] = []

    def onerror(error: OSError) -> None:
        # Match pathlib traversal: unreadable directories are skipped, while
        # unexpected I/O failures must not produce a misleading partial hash.
        if not isinstance(error, PermissionError):
            raise error

    walk = os.walk(root, followlinks=False, onerror=onerror) if root.is_dir() else ()
    for directory, directories, files in walk:
        directories[:] = [name for name in directories if name not in ignored]
        candidates.extend(Path(directory) / name for name in (*directories, *files) if name not in ignored)
    for item in sorted(candidates):
        relative = item.relative_to(root)
        if item.is_symlink():
            raise ValueError(f"Content-bound candidate trees may not contain symbolic links: {item}")
        if not item.is_file() or suffixes is not None and item.suffix.lower() not in suffixes:
            continue
        rows.append({"path": relative.as_posix(), "sha256": sha256_file(item)})
    return sha256_value(rows)


def generated_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True, slots=True)
class CandidateEntrypoint:
    module: str
    class_name: str

    def to_dict(self) -> dict[str, str]:
        return {"module": self.module, "class_name": self.class_name}


@dataclass(frozen=True, slots=True)
class CandidateSource:
    root: Path
    revision_id: str
    entrypoint: CandidateEntrypoint | None


@dataclass(frozen=True, slots=True)
class CandidateVerification:
    component_suite: Path
    smoke_case: Path


@dataclass(frozen=True, slots=True)
class FlightSoftwareCandidate:
    candidate_id: str
    revision: str
    manifest_path: Path
    workspace_root: Path
    source: CandidateSource
    onboard_contract: str
    hardware_profile: str
    task_period_s: float
    intended_use: str
    verification: CandidateVerification
    handling: Mapping[str, Any]
    normalized_manifest: Mapping[str, Any]
    manifest_sha256: str
    source_sha256: str
    verification_sha256: str
    candidate_sha256: str
    native: Mapping[str, Any] | None = None

    @property
    def kind(self) -> str:
        return "native_stack" if self.native is not None else "python_stack"

    def identity(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "revision": self.revision,
            "kind": self.kind,
            "manifest_sha256": self.manifest_sha256,
            "source_sha256": self.source_sha256,
            "verification_sha256": self.verification_sha256,
            "candidate_sha256": self.candidate_sha256,
            "onboard_contract": self.onboard_contract,
            "hardware_profile": self.hardware_profile,
            "task_period_s": self.task_period_s,
        }


@dataclass(frozen=True, slots=True)
class AuthoringIssue:
    code: str
    message: str
    path: str = ""
    severity: str = "error"
    next_step: str = ""

    def to_dict(self) -> dict[str, str]:
        return {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "path": self.path,
            "next_step": self.next_step,
        }


@dataclass(frozen=True, slots=True)
class AuthoringReceipt:
    schema: str
    status: str
    operation: str
    candidate: Mapping[str, Any] = field(default_factory=dict)
    effects: Mapping[str, bool] = field(default_factory=dict)
    artifacts: tuple[Mapping[str, Any], ...] = ()
    issues: tuple[AuthoringIssue, ...] = ()
    result: Mapping[str, Any] = field(default_factory=dict)
    non_claims: tuple[str, ...] = AUTHORING_NON_CLAIMS
    generated_at: str = field(default_factory=generated_utc)

    def __post_init__(self) -> None:
        if self.status not in AUTHORING_STATUSES:
            raise ValueError(f"Unsupported FSW authoring status: {self.status!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "contract_version": AUTHORING_CONTRACT_VERSION,
            "generated_utc": self.generated_at,
            "status": self.status,
            "operation": self.operation,
            "candidate": dict(self.candidate),
            "effects": dict(self.effects),
            "artifacts": [dict(item) for item in self.artifacts],
            "issues": [item.to_dict() for item in self.issues],
            "result": dict(self.result),
            "non_claims": list(self.non_claims),
        }


def effects(*, writes: bool = False, executes: bool = False) -> dict[str, bool]:
    return {
        "reads": True,
        "writes": bool(writes),
        "executes": bool(executes),
        "external_communication": False,
    }


__all__ = [
    "AUTHORING_CONTRACT_VERSION",
    "AUTHORING_NON_CLAIMS",
    "AuthoringIssue",
    "AuthoringReceipt",
    "CANDIDATE_SCHEMA_ID",
    "CAPABILITIES_SCHEMA_ID",
    "CandidateEntrypoint",
    "CandidateSource",
    "CandidateVerification",
    "EXECUTION_PLAN_SCHEMA_ID",
    "FlightSoftwareCandidate",
    "RUN_MANIFEST_SCHEMA_ID",
    "SCAFFOLD_RECEIPT_SCHEMA_ID",
    "TEST_RESULT_SCHEMA_ID",
    "VALIDATION_RECEIPT_SCHEMA_ID",
    "WORK_ORDER_SCHEMA_ID",
    "canonical_json",
    "effects",
    "generated_utc",
    "sha256_file",
    "sha256_tree",
    "sha256_value",
]
