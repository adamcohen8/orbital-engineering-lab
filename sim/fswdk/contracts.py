"""Versioned data contracts for the OEL Flight Software Development Kit.

This module contains data identity and receipt types only.  It deliberately
does not implement simulation, controller scoring, or qualification logic.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

FSWDK_CONTRACT_VERSION = 1
CANDIDATE_SCHEMA_ID = "oel.fswdk.candidate.v1"
WORK_ORDER_SCHEMA_ID = "oel.fswdk.work_order.v1"
EXECUTION_PLAN_SCHEMA_ID = "oel.fswdk.execution_plan.v1"
VALIDATION_RECEIPT_SCHEMA_ID = "oel.fswdk.validation_receipt.v1"
TEST_RESULT_SCHEMA_ID = "oel.fswdk.test_results.v1"
RUN_MANIFEST_SCHEMA_ID = "oel.fswdk.run_manifest.v1"
COMPARISON_PACKET_SCHEMA_ID = "oel.fswdk.comparison_packet.v1"
QUALIFICATION_PACKET_SCHEMA_ID = "oel.fswdk.qualification_packet.v1"
AGENT_EVIDENCE_PACKET_SCHEMA_ID = "oel.fswdk.agent_evidence_packet.v1"
PACKAGE_MANIFEST_SCHEMA_ID = "oel.fswdk.package_manifest.v1"

FSWDK_STATUSES = frozenset(
    {
        "ready",
        "clarification_required",
        "blocked",
        "invalid",
        "passed",
        "failed",
        "incomplete",
        "cancelled",
    }
)

FSWDK_NON_CLAIMS = (
    "Evidence is limited to the exact candidate, models, cases, and declared simulation envelope.",
    "OEL FSWDK does not establish flight qualification, certification, operational approval, or hardware readiness.",
    "Controller comparison does not establish global optimality outside the named variants, objectives, and cases.",
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


def sha256_tree(
    path: str | Path,
    *,
    suffixes: frozenset[str] | None = None,
) -> str:
    root = Path(path).resolve()
    if root.is_file():
        return sha256_file(root)
    rows = []
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
    entrypoint: CandidateEntrypoint | None = None
    artifact: Path | None = None


@dataclass(frozen=True, slots=True)
class CandidateVerification:
    component_suite: Path | None = None
    smoke_case: Path | None = None
    bench_suite: Path | None = None
    qualification_spec: Path | None = None


@dataclass(frozen=True, slots=True)
class FlightSoftwareCandidate:
    candidate_id: str
    revision: str
    kind: str
    manifest_path: Path
    workspace_root: Path
    source: CandidateSource
    onboard_contract: str
    hardware_profile: str
    task_period_s: float
    intended_use: str
    declared_envelope: Path | None
    verification: CandidateVerification
    handling: Mapping[str, Any]
    normalized_manifest: Mapping[str, Any]
    manifest_sha256: str
    source_sha256: str
    verification_sha256: str
    candidate_sha256: str

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
class FSWDKIssue:
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
class FSWDKReceipt:
    schema: str
    status: str
    operation: str
    candidate: Mapping[str, Any] = field(default_factory=dict)
    effects: Mapping[str, bool] = field(default_factory=dict)
    artifacts: tuple[Mapping[str, Any], ...] = ()
    issues: tuple[FSWDKIssue, ...] = ()
    result: Mapping[str, Any] = field(default_factory=dict)
    non_claims: tuple[str, ...] = FSWDK_NON_CLAIMS
    generated_at: str = field(default_factory=generated_utc)

    def __post_init__(self) -> None:
        if self.status not in FSWDK_STATUSES:
            raise ValueError(f"Unsupported FSWDK status: {self.status!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "contract_version": FSWDK_CONTRACT_VERSION,
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


def effects(*, writes: bool = False, executes: bool = False, external: bool = False) -> dict[str, bool]:
    return {
        "reads": True,
        "writes": bool(writes),
        "executes": bool(executes),
        "external_communication": bool(external),
    }
