from __future__ import annotations

import json
import re
from collections.abc import Sequence
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from sim.security.input_reads import SecureReadError, SecureReadRoot
from sim.security.output_writes import current_output_writer, staged_output_file

SCHEMA_VERSION = 1
SCHEMA_ID = "oel-workflow-evidence-v1"
_ARTIFACT_KEYS = {"artifact_id", "path", "media_type", "required", "exists", "sha256"}
_SHA256_RE = re.compile(r"^[a-fA-F0-9]{64}$")
_MAX_WORKFLOW_EVIDENCE_BYTES = 64 * 1024 * 1024


def _lexical_file_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    return candidate


def _open_file_reader(path: str | Path) -> tuple[SecureReadRoot, Path]:
    """Open the containing directory and return its one-component child."""

    candidate = _lexical_file_path(path)
    reader = SecureReadRoot.open(
        candidate.parent,
        approved_roots=(candidate.parent.parent,),
    )
    return reader, candidate.name


def _writer_relative(path: Path, writer: Any) -> tuple[Path, Path] | None:
    """Return logical and physical paths when ``path`` is writer-owned."""

    logical_root = Path(writer.logical_root).expanduser()
    physical_root = Path(writer.root).expanduser()
    for candidate, mapped_root in ((logical_root, logical_root), (physical_root, logical_root)):
        try:
            relative = path.relative_to(candidate)
        except ValueError:
            continue
        return mapped_root / relative, physical_root / relative
    return None


def _artifact_state(path: str | Path) -> tuple[Path, bool, str | None]:
    """Read existence and hash through the retained writer when applicable."""

    raw = Path(path).expanduser()
    if not raw.is_absolute():
        raw = Path.cwd() / raw
    writer = current_output_writer()
    if writer is not None:
        mapped = _writer_relative(raw, writer)
        if mapped is not None:
            logical, _physical = mapped
            try:
                digest = writer.sha256(logical.relative_to(Path(writer.logical_root)))
            except (FileNotFoundError, OSError, ValueError):
                return logical, False, None
            return logical, True, digest
    try:
        reader, name = _open_file_reader(raw)
        try:
            record = reader.file_record(name, max_bytes=_MAX_WORKFLOW_EVIDENCE_BYTES)
            return reader.path / name, True, str(record["sha256"])
        finally:
            reader.close()
    except (FileNotFoundError, NotADirectoryError, SecureReadError, OSError):
        return _lexical_file_path(raw), False, None


def sha256_file(path: str | Path) -> str:
    reader, name = _open_file_reader(path)
    try:
        return reader.sha256(name, max_bytes=2 * 1024**3)
    finally:
        reader.close()


def artifact_reference(
    path: str | Path,
    *,
    artifact_id: str,
    media_type: str,
    required: bool = True,
) -> dict[str, Any]:
    resolved, exists, digest = _artifact_state(path)
    return {
        "artifact_id": str(artifact_id),
        "path": str(resolved),
        "media_type": str(media_type),
        "required": bool(required),
        "exists": exists,
        "sha256": digest if exists else None,
    }


def build_workflow_evidence(
    *,
    workflow_id: str,
    status: str,
    disposition: str,
    inputs: Sequence[Mapping[str, Any]] = (),
    artifacts: Sequence[Mapping[str, Any]] = (),
    quality_gates: Mapping[str, Any] | None = None,
    warnings: Sequence[str] = (),
    failures: Sequence[Mapping[str, Any]] = (),
    provenance: Mapping[str, Any] | None = None,
    non_claims: Sequence[str] = (),
    domain_summary: Mapping[str, Any] | None = None,
    data_markings: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    normalized_artifacts = [
        artifact_reference(
            str(item.get("path", "")),
            artifact_id=str(item.get("artifact_id", "")),
            media_type=str(item.get("media_type", "")),
            required=bool(item.get("required", False)),
        )
        for item in artifacts
    ]
    required_missing = [
        str(item.get("artifact_id", ""))
        for item in normalized_artifacts
        if bool(item.get("required", False)) and not bool(item.get("exists", False))
    ]
    payload = {
        "schema_id": SCHEMA_ID,
        "schema_version": SCHEMA_VERSION,
        "workflow_id": str(workflow_id),
        "status": str(status),
        "disposition": str(disposition),
        "generated_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "inputs": [deepcopy(dict(item)) for item in inputs],
        "quality_gates": deepcopy(dict(quality_gates or {})),
        "warnings": list(dict.fromkeys(str(item) for item in warnings if str(item))),
        "failures": [deepcopy(dict(item)) for item in failures],
        "artifacts": normalized_artifacts,
        "artifact_integrity": {
            "required_artifacts_present": not required_missing,
            "missing_required_artifact_ids": required_missing,
        },
        "provenance": deepcopy(dict(provenance or {})),
        "data_markings": deepcopy(dict(data_markings or {})),
        "non_claims": [str(item) for item in non_claims],
        "domain_summary": deepcopy(dict(domain_summary or {})),
    }
    _validate_workflow_evidence(payload, verify_artifacts=True)
    return payload


def write_workflow_evidence(path: str | Path, payload: Mapping[str, Any]) -> Path:
    target, _exists, _digest = _artifact_state(path)
    normalized = dict(payload)
    _validate_workflow_evidence(normalized, verify_artifacts=True)
    serialized = json.dumps(normalized, indent=2, sort_keys=True, allow_nan=False) + "\n"
    writer = current_output_writer()
    if writer is None:
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(target.name + ".tmp")
        tmp.write_text(serialized, encoding="utf-8")
        tmp.replace(target)
    else:
        with staged_output_file(target) as staged:
            staged.write_text(serialized, encoding="utf-8")
    return target


def load_workflow_evidence(path: str | Path) -> dict[str, Any]:
    raw = _lexical_file_path(path)
    writer = current_output_writer()
    if writer is not None:
        mapped = _writer_relative(raw if raw.is_absolute() else Path.cwd() / raw, writer)
    else:
        mapped = None
    if mapped is None:
        reader, name = _open_file_reader(raw)
        try:
            payload = json.loads(
                reader.read_text(name, max_bytes=_MAX_WORKFLOW_EVIDENCE_BYTES)
            )
        finally:
            reader.close()
    else:
        logical, _physical = mapped
        payload = json.loads(
            writer.read_bytes(logical.relative_to(Path(writer.logical_root))).decode("utf-8")
        )
    if not isinstance(payload, dict):
        raise ValueError("OEL workflow evidence must contain a JSON object.")
    if payload.get("schema_id") != SCHEMA_ID or payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported OEL workflow-evidence schema.")
    _validate_workflow_evidence(payload, verify_artifacts=True)
    return payload


def _validate_workflow_evidence(payload: Mapping[str, Any], *, verify_artifacts: bool) -> None:
    required_keys = {
        "schema_id",
        "schema_version",
        "workflow_id",
        "status",
        "disposition",
        "generated_utc",
        "inputs",
        "quality_gates",
        "warnings",
        "failures",
        "artifacts",
        "artifact_integrity",
        "provenance",
        "data_markings",
        "non_claims",
        "domain_summary",
    }
    unknown = set(payload) - required_keys
    missing = required_keys - set(payload)
    if missing or unknown:
        raise ValueError(f"Invalid workflow evidence fields; missing={sorted(missing)}, unknown={sorted(unknown)}")
    for key in ("workflow_id", "status", "disposition", "generated_utc"):
        if not str(payload.get(key, "") or "").strip():
            raise ValueError(f"Workflow evidence {key} must be non-empty.")
    missing_required: list[str] = []
    for index, raw in enumerate(list(payload.get("artifacts", []) or [])):
        if not isinstance(raw, Mapping) or set(raw) != _ARTIFACT_KEYS:
            raise ValueError(f"Workflow evidence artifact[{index}] does not match the v1 artifact contract.")
        artifact_id = str(raw.get("artifact_id", "") or "").strip()
        path_text = str(raw.get("path", "") or "").strip()
        media_type = str(raw.get("media_type", "") or "").strip()
        if not artifact_id or not path_text or not media_type:
            raise ValueError(f"Workflow evidence artifact[{index}] identifiers, path, and media_type are required.")
        required = raw.get("required")
        exists = raw.get("exists")
        if not isinstance(required, bool) or not isinstance(exists, bool):
            raise ValueError(f"Workflow evidence artifact[{index}] required and exists must be booleans.")
        path, actual_exists, actual_sha = _artifact_state(path_text)
        declared_sha = raw.get("sha256")
        if declared_sha is not None and not _SHA256_RE.fullmatch(str(declared_sha)):
            raise ValueError(f"Workflow evidence artifact[{index}] has an invalid sha256.")
        if verify_artifacts and (exists != actual_exists or declared_sha != actual_sha):
            raise ValueError(f"Workflow evidence artifact integrity mismatch for '{artifact_id}'.")
        if required and not actual_exists:
            missing_required.append(artifact_id)
    integrity = dict(payload.get("artifact_integrity", {}) or {})
    expected = {
        "required_artifacts_present": not missing_required,
        "missing_required_artifact_ids": missing_required,
    }
    if integrity != expected:
        raise ValueError("Workflow evidence artifact_integrity does not match the artifact list.")
