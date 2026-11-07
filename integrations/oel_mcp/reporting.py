from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from integrations.oel_mcp.contracts import MAX_REVIEW_RESULT_BYTES, MAX_REVIEW_VALUE_BYTES
from integrations.oel_mcp.execution import complete_manifest, manifest_base, write_execution_manifest
from sim.agent_task.runner import inspect_output

REPORT_PACKET_SCHEMA = "oel.mcp_report_evidence.v2"
LEGACY_REPORT_PACKET_SCHEMA = "oel.mcp_report_evidence.v1"
REPORT_AUDIT_SCHEMA = "oel.mcp_report_audit.v1"
MAX_REPORT_SOURCE_BYTES = 2_000_000
MAX_PACKET_ARTIFACTS = 100
MAX_REPORT_ARTIFACT_BYTES = 1_000_000_000
MAX_REPORT_TOTAL_ARTIFACT_BYTES = 2_000_000_000
_SOURCE_EVIDENCE_SUMMARY_KEYS = (
    "status",
    "validation_ok",
    "review_evidence_complete",
    "artifacts_complete",
    "plots_complete",
    "comparison_complete",
    "failure_hint_count",
    "caveat_count",
    "ready_to_cite",
)
_EVIDENCE_REFERENCE = re.compile(r"\[evidence:([A-Za-z0-9_.:-]{1,120})\]")


def prepare_report_packet(
    *,
    source_output_dir: Path,
    packet_output_dir: Path,
    packet_id: str,
    query_names: list[str],
    max_rows: int,
    handling: dict[str, Any],
    approval_id: str,
    writer: Any | None = None,
    logical_output_dir: str | Path | None = None,
    logical_source_output_dir: str | Path | None = None,
    source_reader: Any | None = None,
    protected_review: bool | None = None,
) -> dict[str, Any]:
    writer = writer or _current_output_writer()
    source_output_dir = Path(source_output_dir)
    logical_source_root = Path(logical_source_output_dir) if logical_source_output_dir is not None else source_output_dir
    # A report packet is a downstream view of a completed MCP execution. Do
    # not turn a source manifest's ``artifacts_complete`` flag into a trust
    # decision without rechecking the content-bound records. The source
    # manifest and mutable index are intentionally excluded from that list.
    source_manifest_path = source_output_dir / "mcp_execution_manifest.json"
    source_manifest_present = _source_is_file(
        source_manifest_path,
        physical_root=source_output_dir,
        logical_root=logical_source_root,
        reader=source_reader,
    )
    if source_manifest_present:
        _validate_source_manifest(
            source_manifest_path,
            source_output_dir,
            logical_root=logical_source_root,
            reader=source_reader,
        )
    inspection = inspect_output(
        source_output_dir,
        query_names=query_names or None,
        max_rows=max_rows,
        write_packet=False,
        max_value_bytes=MAX_REVIEW_VALUE_BYTES,
        max_result_bytes=MAX_REVIEW_RESULT_BYTES,
        protected=protected_review,
    )
    artifacts, artifacts_truncated = _artifact_inventory(
        inspection,
        source_output_dir=source_output_dir,
        logical_source_output_dir=logical_source_root,
        source_reader=source_reader,
    )
    # inspect_output and artifact inventory read mutable source files. Recheck
    # the execution manifest after those reads so a source mutation cannot be
    # turned into a completed, ready-to-cite packet by the initial check.
    if source_manifest_present:
        _validate_source_manifest(
            source_manifest_path,
            source_output_dir,
            logical_root=logical_source_root,
            reader=source_reader,
        )
    required_artifacts_missing = any(row["required"] and not row["exists"] for row in artifacts)
    review = _review_projection(dict(inspection.get("review", {}) or {}))
    query_evidence = _query_evidence(review)
    query_summary = dict(review.get("query_summary", {}) or {})
    queries_complete = bool(query_summary.get("evidence_complete", not query_evidence))
    failure_hints = [dict(item) for item in list(inspection.get("failure_hints", []) or [])]
    if not queries_complete:
        failure_hints.append(
            {
                "code": "review_queries_incomplete",
                "unknown_queries": list(query_summary.get("unknown_queries", []) or []),
                "failed_queries": list(query_summary.get("failed_queries", []) or []),
                "next_step": "Use supported saved-query names and resolve failed or unexpectedly empty evidence queries.",
            }
        )
    if required_artifacts_missing:
        failure_hints.append(
            {
                "code": "required_report_artifact_missing",
                "next_step": "Restore or regenerate the required run summary and review store before authoring a report.",
            }
        )
    if artifacts_truncated:
        failure_hints.append(
            {
                "code": "artifact_inventory_truncated",
                "limit": MAX_PACKET_ARTIFACTS,
                "next_step": "Narrow the source artifact set or prepare multiple explicitly scoped report packets.",
            }
        )
    execution_provenance = _report_execution_provenance(
        source_output_dir,
        logical_root=logical_source_root,
        reader=source_reader,
    )
    if execution_provenance.get("available") and execution_provenance.get("status") != "completed":
        failure_hints.append(
            {
                "code": "source_execution_not_completed",
                "status": execution_provenance.get("status"),
                "next_step": "Use a completed run or regenerate the source evidence before report preparation.",
            }
        )
    evidence_summary = _evidence_summary(
        source_output_dir=source_output_dir,
        logical_source_output_dir=logical_source_root,
        artifacts=artifacts,
        queries_complete=queries_complete,
        failure_hints=failure_hints,
        source_reader=source_reader,
    )
    packet_output_dir = Path(packet_output_dir)
    logical_packet_output_dir = _logical_output_dir(
        packet_output_dir,
        writer=writer,
        logical_output_dir=logical_output_dir,
    )
    if writer is None:
        packet_output_dir.mkdir(parents=True, exist_ok=True)
    packet_path = packet_output_dir / "report_evidence_packet.json"
    brief_path = packet_output_dir / "report_brief.md"
    operation_manifest_path = logical_packet_output_dir / "mcp_execution_manifest.json"
    logical_packet_path = logical_packet_output_dir / packet_path.name
    logical_brief_path = logical_packet_output_dir / brief_path.name
    packet = {
        "schema_id": REPORT_PACKET_SCHEMA,
        "packet_id": packet_id,
        "generated_utc": _utc_now(),
        "source_output_dir": str(logical_source_root),
        "handling": {
            "marking": str(handling.get("marking", "")),
            "release_scope": str(handling.get("release_scope", "")),
        },
        "evidence_summary": evidence_summary,
        "review": review,
        "query_evidence": query_evidence,
        "execution_provenance": execution_provenance,
        "source_execution_manifest_sha256": _optional_file_sha256(
            source_output_dir / "mcp_execution_manifest.json",
            physical_root=source_output_dir,
            logical_root=logical_source_root,
            reader=source_reader,
        ),
        "artifact_summary": {**_artifact_summary(artifacts), "truncated": artifacts_truncated},
        "source_evidence_identity_sha256": _sha256_json(
            [
                {
                    "artifact_id": row["artifact_id"],
                    "relative_path": row["relative_path"],
                    "bytes": row["bytes"],
                    "sha256": row["sha256"],
                }
                for row in artifacts
            ]
        ),
        "artifacts_truncated": artifacts_truncated,
        "required_artifacts_missing": required_artifacts_missing,
        "artifacts": artifacts,
        "failure_hints": failure_hints,
        "caveats": [str(item) for item in list(inspection.get("caveats", []) or [])],
        "report_contract": {
            "required_sections": ["Evidence", "Limitations"],
            "evidence_reference_syntax": "[evidence:<evidence_id>]",
            "provider_call_made": False,
            "authoring_owner": "connected_agent_or_human",
        },
        "non_claims": [
            "The packet does not author a report or call a model provider.",
            "Artifact integrity does not establish that narrative claims are analytically correct.",
            "Deterministic OEL artifacts remain the evidence authority.",
        ],
    }

    packet["packet_sha256"] = _sha256_json({key: value for key, value in packet.items() if key != "packet_sha256"})
    packet_bytes = _serialized_json(packet)
    if len(packet_bytes) > MAX_REPORT_SOURCE_BYTES:
        raise ValueError("The report packet exceeds the MCP report-packet size budget.")
    _write_output_bytes(packet_path, packet_bytes, writer=writer)
    _write_output_bytes(
        brief_path,
        _report_brief(packet).encode("utf-8"),
        writer=writer,
    )
    manifest = manifest_base(tool_id="oel.prepare_report_packet.v1", approval_id=approval_id)
    manifest.update(
        {
            "packet_id": packet_id,
            "source_output_dir": str(logical_source_root),
            "packet_sha256": packet["packet_sha256"],
            "source_evidence_identity_sha256": packet["source_evidence_identity_sha256"],
            "provider_call_made": False,
        }
    )
    packet_status = "completed" if not failure_hints and not required_artifacts_missing else "partial"
    _complete_output_manifest(
        manifest,
        status=packet_status,
        logical_artifacts=[str(logical_packet_path), str(logical_brief_path)],
        physical_artifacts=[str(packet_path), str(brief_path)],
        writer=writer,
    )
    write_execution_manifest(packet_output_dir, manifest, writer=writer)
    return {
        "status": packet_status,
        "packet_id": packet_id,
        "source_output_dir": str(logical_source_root),
        "packet_output_dir": str(logical_packet_output_dir),
        "packet_path": str(logical_packet_path),
        "brief_path": str(logical_brief_path),
        "manifest_path": str(operation_manifest_path),
        "packet_sha256": packet["packet_sha256"],
        "artifact_count": len(artifacts),
        "evidence_summary": packet["evidence_summary"],
        "provider_call_made": False,
        "non_claims": list(packet["non_claims"]),
    }


def _source_manifest_artifacts_valid(
    manifest: dict[str, Any],
    output: Path,
    *,
    logical_root: Path | None = None,
    reader: Any | None = None,
) -> bool:
    """Validate immutable source artifacts before projecting a report packet."""

    if not bool(manifest.get("artifacts_complete", False)):
        return False
    logical_root = logical_root or output
    mutable_paths = {logical_root / "index.md"}
    artifacts = [
        str(path)
        for path in list(manifest.get("artifacts", []) or [])
        if _logical_source_path(Path(str(path)), physical_root=output, logical_root=logical_root) not in mutable_paths
    ]
    records = [
        dict(raw or {})
        for raw in list(manifest.get("artifact_records", []) or [])
        if _logical_source_path(
            Path(str(dict(raw or {}).get("path", "") or "")),
            physical_root=output,
            logical_root=logical_root,
        ) not in mutable_paths
    ]
    if not artifacts or len(records) != len(artifacts):
        return False
    expected_paths = set(artifacts)
    if len(expected_paths) != len(artifacts):
        return False
    seen: set[str] = set()
    for record in records:
        raw_path = str(record.get("path", "") or "")
        logical_path = _logical_source_path(
            Path(raw_path),
            physical_root=output,
            logical_root=logical_root,
        )
        if logical_path is None:
            return False
        path = _physical_source_path(
            logical_path,
            physical_root=output,
            logical_root=logical_root,
        )
        if path is None:
            return False
        if raw_path not in expected_paths or raw_path in seen:
            return False
        if not _source_path_within(path, output):
            return False
        if not _source_is_file(path, physical_root=output, logical_root=logical_root, reader=reader):
            return False
        try:
            expected_bytes = int(record.get("bytes", -1))
        except (TypeError, ValueError):
            return False
        actual_record = _source_file_record(
            path,
            physical_root=output,
            logical_root=logical_root,
            reader=reader,
            maximum=MAX_REPORT_ARTIFACT_BYTES,
        )
        if actual_record is None or expected_bytes != int(actual_record["bytes"]):
            return False
        if str(record.get("sha256", "")) != str(actual_record.get("sha256", "")):
            return False
        seen.add(raw_path)
    return seen == expected_paths


def _validate_source_manifest(
    path: Path,
    output: Path,
    *,
    logical_root: Path | None = None,
    reader: Any | None = None,
) -> None:
    try:
        payload = _source_read_bytes(
            path,
            physical_root=output,
            logical_root=logical_root or output,
            reader=reader,
            maximum=MAX_REPORT_SOURCE_BYTES,
        )
        manifest = json.loads(payload.decode("utf-8"))
    except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("The source execution manifest is invalid.") from exc
    if not isinstance(manifest, dict):
        raise ValueError("The source execution manifest is invalid.")
    if manifest.get("status") == "completed" and not _source_manifest_artifacts_valid(
        manifest,
        output,
        logical_root=logical_root,
        reader=reader,
    ):
        raise ValueError("The source execution manifest artifact records do not match the current output.")


def audit_report(
    *,
    report_path: Path,
    packet_path: Path,
    audit_output_dir: Path,
    author: str,
    model: str,
    approval_id: str,
    preparation_manifest_path: Path | None = None,
    writer: Any | None = None,
    logical_output_dir: str | Path | None = None,
    report_bytes: bytes | None = None,
    packet_bytes: bytes | None = None,
    preparation_manifest_bytes: bytes | None = None,
    source_reader: Any | None = None,
) -> dict[str, Any]:
    writer = writer or _current_output_writer()
    report_payload = report_bytes if report_bytes is not None else report_path.read_bytes()
    packet_payload_bytes = packet_bytes if packet_bytes is not None else packet_path.read_bytes()
    if len(report_payload) > MAX_REPORT_SOURCE_BYTES:
        raise ValueError("The report exceeds the MCP report-audit size budget.")
    if len(packet_payload_bytes) > MAX_REPORT_SOURCE_BYTES:
        raise ValueError("The report packet exceeds the MCP report-audit size budget.")
    packet = _load_packet(packet_path, payload=packet_payload_bytes)
    provenance_binding = _validate_prepare_provenance(
        packet,
        packet_path=packet_path,
        manifest_path=preparation_manifest_path,
        manifest_bytes=preparation_manifest_bytes,
        source_reader=source_reader,
    )
    _validate_audit_packet_budget(packet, packet_path=packet_path, source_reader=source_reader)
    try:
        markdown = report_payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("The report is not valid UTF-8 Markdown.") from exc
    integrity = _verify_packet_artifacts(packet, packet_path=packet_path, source_reader=source_reader)
    references = sorted(set(_EVIDENCE_REFERENCE.findall(markdown)))
    artifact_rows = [dict(item) for item in list(packet.get("artifacts", []) or [])]
    artifact_id_list = [str(item.get("artifact_id", "")) for item in artifact_rows]
    artifact_path_list = [str(item.get("relative_path", "")) for item in artifact_rows]
    query_id_list = [str(item.get("evidence_id", "")) for item in list(packet.get("query_evidence", []) or [])]
    evidence_ids = set(artifact_id_list) | set(query_id_list)
    unknown_references = sorted(set(references) - evidence_ids)
    available_evidence_ids = {
        str(item.get("artifact_id", "")) for item in artifact_rows if bool(item.get("exists", False))
    } | {
        str(item.get("evidence_id", ""))
        for item in list(packet.get("query_evidence", []) or [])
        if item.get("status") == "ok" and not item.get("truncated", False)
    }
    unavailable_references = sorted(set(references) - available_evidence_ids - set(unknown_references))
    headings = _markdown_headings(markdown)
    required_sections = [
        str(item) for item in list(dict(packet.get("report_contract", {}) or {}).get("required_sections", []) or [])
    ]
    missing_sections = [section for section in required_sections if section.casefold() not in headings]
    checks = {
        "packet_schema_supported": True,
        "packet_content_hash_valid": _packet_content_hash_valid(packet),
        "prepare_provenance_bound": provenance_binding["valid"],
        "artifact_ids_unique": len(artifact_id_list) == len(set(artifact_id_list)),
        "artifact_paths_unique": len(artifact_path_list) == len(set(artifact_path_list)),
        "query_evidence_ids_unique": len(query_id_list) == len(set(query_id_list)),
        "required_core_artifacts_present": {"run_summary", "review_store"}.issubset(set(artifact_id_list)),
        "artifact_integrity_valid": integrity["valid"],
        "required_sections_present": not missing_sections,
        "evidence_references_known": not unknown_references,
        "evidence_references_available": not unavailable_references,
        "evidence_reference_present": bool(references) or not evidence_ids,
    }
    passed = all(checks.values())
    audit = {
        "schema_id": REPORT_AUDIT_SCHEMA,
        "generated_utc": _utc_now(),
        "status": "passed" if passed else "needs_review",
        "author": author,
        "model": model,
        "provider_call_made": False,
        "report_path": str(report_path),
        "packet_path": str(packet_path),
        "packet_id": str(packet.get("packet_id", "")),
        "checks": checks,
        "artifact_integrity": integrity,
        "prepare_provenance": provenance_binding,
        "evidence_references": references,
        "unknown_evidence_references": unknown_references,
        "unavailable_evidence_references": unavailable_references,
        "missing_required_sections": missing_sections,
        "semantic_claim_review_performed": False,
        "non_claims": [
            "This audit verifies packet structure, hashes, report structure, and evidence references.",
            "It does not determine whether every narrative interpretation or operational conclusion is correct.",
            "It does not call a model provider or authorize external release.",
        ],
    }
    audit_output_dir = Path(audit_output_dir)
    logical_audit_output_dir = _logical_output_dir(
        audit_output_dir,
        writer=writer,
        logical_output_dir=logical_output_dir,
    )
    if writer is None:
        audit_output_dir.mkdir(parents=True, exist_ok=True)
    audit_json = audit_output_dir / "report_audit.json"
    audit_md = audit_output_dir / "report_audit.md"
    operation_manifest_path = logical_audit_output_dir / "mcp_execution_manifest.json"
    logical_audit_json = logical_audit_output_dir / audit_json.name
    logical_audit_md = logical_audit_output_dir / audit_md.name
    _write_output_bytes(audit_json, _serialized_json(audit), writer=writer)
    _write_output_bytes(
        audit_md,
        _audit_markdown(audit).encode("utf-8"),
        writer=writer,
    )
    manifest = manifest_base(tool_id="oel.audit_report.v1", approval_id=approval_id)
    manifest.update(
        {
            "packet_id": audit["packet_id"],
            "audit_status": audit["status"],
            "provider_call_made": False,
            "semantic_claim_review_performed": False,
        }
    )
    _complete_output_manifest(
        manifest,
        status="completed" if passed else "partial",
        logical_artifacts=[str(logical_audit_json), str(logical_audit_md)],
        physical_artifacts=[str(audit_json), str(audit_md)],
        writer=writer,
    )
    write_execution_manifest(audit_output_dir, manifest, writer=writer)
    return {
        "status": audit["status"],
        "packet_id": audit["packet_id"],
        "report_path": str(report_path),
        "packet_path": str(packet_path),
        "audit_output_dir": str(logical_audit_output_dir),
        "audit_json_path": str(logical_audit_json),
        "audit_markdown_path": str(logical_audit_md),
        "manifest_path": str(operation_manifest_path),
        "checks": checks,
        "unknown_evidence_references": unknown_references,
        "unavailable_evidence_references": unavailable_references,
        "missing_required_sections": missing_sections,
        "provider_call_made": False,
        "semantic_claim_review_performed": False,
        "non_claims": list(audit["non_claims"]),
    }


def _artifact_inventory(
    inspection: dict[str, Any],
    *,
    source_output_dir: Path,
    logical_source_output_dir: Path | None = None,
    source_reader: Any | None = None,
) -> tuple[list[dict[str, Any]], bool]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    artifact_ids: set[str] = set()
    total_bytes = 0
    fixed_candidates = (
        ("run_summary", source_output_dir / "master_run_summary.json", "application/json", True),
        ("review_store", source_output_dir / "review" / "run.sqlite", "application/vnd.sqlite3", True),
        ("review_store_wal", source_output_dir / "review" / "run.sqlite-wal", "application/vnd.sqlite3-wal", False),
        ("review_store_shm", source_output_dir / "review" / "run.sqlite-shm", "application/octet-stream", False),
        ("review_schema", source_output_dir / "review" / "schema.json", "application/json", False),
        ("agent_evidence_packet", source_output_dir / "agent_evidence_packet.json", "application/json", False),
        ("run_index", source_output_dir / "index.md", "text/markdown", False),
        ("execution_manifest", source_output_dir / "mcp_execution_manifest.json", "application/json", False),
    )
    logical_root = logical_source_output_dir or source_output_dir
    candidates: list[dict[str, Any]] = []
    for artifact_id, path, media_type, required in fixed_candidates:
        candidates.append(
            {
                "artifact_id": artifact_id,
                "resolved_path": str(path),
                "media_type": media_type,
                "required": required,
            }
        )
    candidates.extend(dict(item or {}) for item in list(inspection.get("artifacts", []) or []))
    truncated = False
    for index, raw in enumerate(candidates):
        if len(rows) >= MAX_PACKET_ARTIFACTS:
            truncated = True
            break
        item = dict(raw or {})
        raw_path = str(item.get("resolved_path") or item.get("path") or "")
        if not raw_path:
            continue
        logical_path = _logical_source_path(
            Path(raw_path).expanduser(),
            physical_root=source_output_dir,
            logical_root=logical_root,
        )
        if logical_path is None:
            continue
        path = _physical_source_path(
            logical_path,
            physical_root=source_output_dir,
            logical_root=logical_root,
        )
        if path is None:
            continue
        relative = logical_path.relative_to(logical_root).as_posix()
        if relative in seen:
            continue
        required = bool(item.get("required", False))
        exists = _source_is_file(
            path,
            physical_root=source_output_dir,
            logical_root=logical_root,
            reader=source_reader,
        )
        if not exists and not required:
            continue
        seen.add(relative)
        artifact_id = _unique_identifier(
            _safe_identifier(str(item.get("artifact_id") or item.get("artifact_key") or f"artifact_{index + 1}")),
            artifact_ids,
        )
        artifact_ids.add(artifact_id)
        record = (
            _source_file_record(
                path,
                physical_root=source_output_dir,
                logical_root=logical_root,
                reader=source_reader,
                maximum=MAX_REPORT_ARTIFACT_BYTES,
            )
            if exists
            else None
        )
        artifact_bytes = int(record["bytes"]) if record is not None else 0
        if artifact_bytes > MAX_REPORT_ARTIFACT_BYTES:
            raise ValueError("An artifact exceeds the MCP report-packet size budget.")
        total_bytes += artifact_bytes
        if total_bytes > MAX_REPORT_TOTAL_ARTIFACT_BYTES:
            raise ValueError("The artifacts exceed the MCP report-packet aggregate size budget.")
        rows.append(
            {
                "artifact_id": artifact_id,
                "relative_path": relative,
                "media_type": str(item.get("media_type") or _media_type(path)),
                "required": required,
                "exists": exists,
                "bytes": artifact_bytes,
                "sha256": str(record["sha256"]) if record is not None else "",
            }
        )
    return rows, truncated


def _review_projection(review: dict[str, Any]) -> dict[str, Any]:
    return {
        key: review[key]
        for key in ("tables", "query_summary", "queries", "error")
        if key in review
    }


def _query_evidence(review: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in list(review.get("queries", []) or []):
        item = dict(raw or {})
        name = str(item.get("name", "query"))
        rows.append(
            {
                "evidence_id": f"query.{_safe_identifier(name)}",
                "name": name,
                "status": str(item.get("status", "")),
                "known": bool(item.get("known", False)),
                "description": str(item.get("description", "")),
                "columns": [str(value) for value in list(item.get("columns", []) or [])],
                "rows": [dict(value) for value in list(item.get("rows", []) or [])],
                "row_count": int(item.get("row_count", 0) or 0),
                "empty_result": bool(item.get("empty_result", False)),
                "truncated": bool(item.get("truncated", False)),
                "reason": str(item.get("reason", "")),
            }
        )
    return rows


def _artifact_summary(artifacts: list[dict[str, Any]]) -> dict[str, Any]:
    existing = [row for row in artifacts if row["exists"]]
    missing = [row for row in artifacts if not row["exists"]]
    return {
        "total": len(artifacts),
        "existing": len(existing),
        "missing": len(missing),
        "total_bytes": sum(int(row["bytes"]) for row in existing),
        "missing_artifacts": [str(row["artifact_id"]) for row in missing],
        "required_missing": [str(row["artifact_id"]) for row in missing if row["required"]],
        "artifacts_complete": not any(row["required"] and not row["exists"] for row in artifacts),
    }


def _evidence_summary(
    *,
    source_output_dir: Path,
    logical_source_output_dir: Path | None = None,
    artifacts: list[dict[str, Any]],
    queries_complete: bool,
    failure_hints: list[dict[str, Any]],
    source_reader: Any | None = None,
) -> dict[str, Any]:
    source_summary: dict[str, Any] = {}
    source_packet = source_output_dir / "agent_evidence_packet.json"
    logical_root = logical_source_output_dir or source_output_dir
    if _source_is_file(
        source_packet,
        physical_root=source_output_dir,
        logical_root=logical_root,
        reader=source_reader,
    ):
        try:
            payload = json.loads(
                _source_read_bytes(
                    source_packet,
                    physical_root=source_output_dir,
                    logical_root=logical_root,
                    reader=source_reader,
                    maximum=MAX_REPORT_SOURCE_BYTES,
                ).decode("utf-8")
            )
        except (OSError, ValueError, UnicodeDecodeError, json.JSONDecodeError):
            payload = {}
        if isinstance(payload, dict):
            raw_summary = dict(payload.get("evidence_summary", {}) or {})
            source_summary = {
                key: raw_summary[key]
                for key in _SOURCE_EVIDENCE_SUMMARY_KEYS
                if key in raw_summary and isinstance(raw_summary[key], (str, int, float, bool, type(None)))
            }
    required_complete = not any(row["required"] and not row["exists"] for row in artifacts)
    source_summary.update(
        {
            "ready_to_cite": not failure_hints and queries_complete and required_complete,
            "review_queries_complete": queries_complete,
            "required_artifacts_complete": required_complete,
            "artifact_count": len(artifacts),
        }
    )
    return source_summary


def _report_execution_provenance(
    source_output_dir: Path,
    *,
    logical_root: Path | None = None,
    reader: Any | None = None,
) -> dict[str, Any]:
    manifest_path = source_output_dir / "mcp_execution_manifest.json"
    logical_root = logical_root or source_output_dir
    if not _source_is_file(
        manifest_path,
        physical_root=source_output_dir,
        logical_root=logical_root,
        reader=reader,
    ):
        return {"available": False}
    try:
        payload = json.loads(
            _source_read_bytes(
                manifest_path,
                physical_root=source_output_dir,
                logical_root=logical_root,
                reader=reader,
                maximum=MAX_REPORT_SOURCE_BYTES,
            ).decode("utf-8")
        )
    except (OSError, ValueError, UnicodeDecodeError, json.JSONDecodeError):
        return {"available": False}
    if not isinstance(payload, dict):
        return {"available": False}
    allowed = (
        "tool_id",
        "status",
        "recipe_id",
        "validation_id",
        "source_config_sha256",
        "normalized_config_sha256",
        "resource_profile",
        "artifacts_complete",
        "cancelled",
        "started_utc",
        "completed_utc",
    )
    return {"available": True, **{key: payload.get(key) for key in allowed if key in payload}}


def _load_packet(path: Path, *, payload: bytes | None = None) -> dict[str, Any]:
    raw = path.read_bytes() if payload is None else payload
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_id") not in {
        REPORT_PACKET_SCHEMA,
        LEGACY_REPORT_PACKET_SCHEMA,
    }:
        raise ValueError("The report packet does not use the supported OEL MCP report schema.")
    if not _packet_content_hash_valid(payload):
        raise ValueError("The report packet content hash does not match its content.")
    if not isinstance(payload.get("artifacts"), list) or not isinstance(payload.get("report_contract"), dict):
        raise ValueError("The report packet is missing its artifact or report contract.")
    return payload


def _validate_prepare_provenance(
    packet: dict[str, Any],
    *,
    packet_path: Path,
    manifest_path: Path | None = None,
    manifest_bytes: bytes | None = None,
    source_reader: Any | None = None,
) -> dict[str, Any]:
    manifest_path = manifest_path or packet_path.parent / "mcp_execution_manifest.json"
    # MCP callers pass the exact preparation-manifest bytes they already
    # consumed through the path policy.  Do not turn a missing exact read
    # into a second path-only existence check or reopen; the legacy fallback
    # remains for direct non-MCP callers that intentionally use this helper.
    manifest_present = manifest_bytes is not None or (
        source_reader is None and manifest_path.is_file()
    )
    if packet.get("schema_id") == LEGACY_REPORT_PACKET_SCHEMA and not manifest_present:
        return {
            "valid": True,
            "compatibility": "legacy_v1_without_preparation_manifest",
            "manifest_path": "",
            "checks": {"legacy_schema_accepted": True},
        }
    if manifest_bytes is None:
        if source_reader is not None:
            raise ValueError("The report packet is not accompanied by its bounded preparation manifest.")
        if not manifest_path.is_file() or manifest_path.stat().st_size > MAX_REPORT_SOURCE_BYTES:
            raise ValueError("The report packet is not accompanied by its bounded preparation manifest.")
        manifest_bytes = manifest_path.read_bytes()
    if len(manifest_bytes) > MAX_REPORT_SOURCE_BYTES:
        raise ValueError("The report packet is not accompanied by its bounded preparation manifest.")
    manifest = json.loads(manifest_bytes.decode("utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("The report packet preparation manifest is invalid.")
    source_root = Path(str(packet.get("source_output_dir", ""))).expanduser().resolve()
    expected_source_manifest = str(packet.get("source_execution_manifest_sha256", ""))
    actual_source_manifest = _optional_file_sha256(
        source_root / "mcp_execution_manifest.json",
        physical_root=source_root,
        logical_root=source_root,
        reader=source_reader,
    )
    checks = {
        "tool_id_matches": manifest.get("tool_id") == "oel.prepare_report_packet.v1",
        "status_complete": manifest.get("status") in {"completed", "partial"},
        "packet_id_matches": str(manifest.get("packet_id", "")) == str(packet.get("packet_id", "")),
        "packet_sha256_matches": str(manifest.get("packet_sha256", "")) == str(packet.get("packet_sha256", "")),
        "source_output_dir_matches": Path(str(manifest.get("source_output_dir", ""))).expanduser().resolve()
        == source_root,
        "source_execution_manifest_matches": bool(expected_source_manifest)
        and expected_source_manifest == actual_source_manifest
        if expected_source_manifest
        else actual_source_manifest == "",
        "source_evidence_identity_matches": bool(packet.get("source_evidence_identity_sha256"))
        and str(manifest.get("source_evidence_identity_sha256", ""))
        == str(packet.get("source_evidence_identity_sha256", "")),
    }
    if not all(checks.values()):
        raise ValueError("The report packet is not bound to its preparation and source-run provenance.")
    return {"valid": True, "manifest_path": str(manifest_path), "checks": checks}


def _validate_audit_packet_budget(
    packet: dict[str, Any],
    *,
    packet_path: Path,
    source_reader: Any | None = None,
) -> None:
    artifacts = list(packet.get("artifacts", []) or [])
    if len(artifacts) > MAX_PACKET_ARTIFACTS:
        raise ValueError("The report packet exceeds the audit artifact-count budget.")
    source_root = Path(str(packet.get("source_output_dir", ""))).expanduser().resolve()
    seen_paths: set[str] = set()
    seen_ids: set[str] = set()
    total_bytes = 0
    for raw in artifacts:
        if not isinstance(raw, dict):
            raise ValueError("The report packet artifact inventory is invalid.")
        artifact_id = str(raw.get("artifact_id", ""))
        relative = str(raw.get("relative_path", ""))
        if not artifact_id or artifact_id in seen_ids or not relative or relative in seen_paths:
            raise ValueError("The report packet contains duplicate or empty artifact identities.")
        seen_ids.add(artifact_id)
        seen_paths.add(relative)
        target = _physical_source_path(
            source_root / relative,
            physical_root=source_root,
            logical_root=source_root,
        )
        if target is None or not _source_path_within(target, source_root):
            raise ValueError("The report packet artifact path escapes its bound source run.")
        if not _source_is_file(
            target,
            physical_root=source_root,
            logical_root=source_root,
            reader=source_reader,
        ):
            continue
        record = _source_file_record(
            target,
            physical_root=source_root,
            logical_root=source_root,
            reader=source_reader,
            maximum=MAX_REPORT_ARTIFACT_BYTES,
        )
        if record is None:
            continue
        size = int(record["bytes"])
        if size > MAX_REPORT_ARTIFACT_BYTES:
            raise ValueError("An artifact exceeds the MCP report-audit size budget.")
        total_bytes += size
        if total_bytes > MAX_REPORT_TOTAL_ARTIFACT_BYTES:
            raise ValueError("The artifacts exceed the MCP report-audit aggregate size budget.")


def _optional_file_sha256(
    path: Path,
    *,
    physical_root: Path,
    logical_root: Path,
    reader: Any | None = None,
) -> str:
    if not _source_is_file(path, physical_root=physical_root, logical_root=logical_root, reader=reader):
        return ""
    record = _source_file_record(
        path,
        physical_root=physical_root,
        logical_root=logical_root,
        reader=reader,
        maximum=MAX_REPORT_SOURCE_BYTES,
    )
    return "" if record is None else str(record["sha256"])


def _packet_content_hash_valid(packet: dict[str, Any]) -> bool:
    expected = str(packet.get("packet_sha256", ""))
    actual = _sha256_json({key: value for key, value in packet.items() if key != "packet_sha256"})
    return len(expected) == 64 and expected == actual


def _verify_packet_artifacts(
    packet: dict[str, Any],
    *,
    packet_path: Path,
    source_reader: Any | None = None,
) -> dict[str, Any]:
    source_root = Path(str(packet.get("source_output_dir", ""))).expanduser().resolve()
    rows: list[dict[str, Any]] = []
    valid = True
    for raw in list(packet.get("artifacts", []) or []):
        item = dict(raw or {})
        relative = str(item.get("relative_path", ""))
        target = _physical_source_path(
            source_root / relative,
            physical_root=source_root,
            logical_root=source_root,
        )
        within = target is not None and _source_path_within(target, source_root)
        exists = bool(
            within
            and target is not None
            and _source_is_file(
                target,
                physical_root=source_root,
                logical_root=source_root,
                reader=source_reader,
            )
        )
        record = (
            _source_file_record(
                target,
                physical_root=source_root,
                logical_root=source_root,
                reader=source_reader,
                maximum=MAX_REPORT_ARTIFACT_BYTES,
            )
            if exists and target is not None
            else None
        )
        actual_sha = "" if record is None else str(record["sha256"])
        matched = exists and actual_sha == str(item.get("sha256", ""))
        required = bool(item.get("required", False))
        valid = valid and matched
        rows.append(
            {
                "artifact_id": str(item.get("artifact_id", "")),
                "relative_path": relative,
                "required": required,
                "exists": exists,
                "sha256_matches": matched,
            }
        )
    return {"valid": valid, "packet_path": str(packet_path), "artifacts": rows}


def _report_brief(packet: dict[str, Any]) -> str:
    artifact_lines = [
        f"- `{row['artifact_id']}`: `{row['relative_path']}` — "
        + (
            f"cite as `[evidence:{row['artifact_id']}]`"
            if row.get("exists")
            else "required artifact is missing; do not cite as available evidence"
        )
        for row in list(packet.get("artifacts", []) or [])
    ]
    query_lines = [
        f"- `{row['evidence_id']}`: saved query `{row['name']}` ({row['status']}) — "
        + (
            f"cite as `[evidence:{row['evidence_id']}]`"
            if row.get("status") == "ok" and not row.get("truncated")
            else "not available as complete report evidence"
        )
        for row in list(packet.get("query_evidence", []) or [])
    ]
    provenance = dict(packet.get("execution_provenance", {}) or {})
    provenance_lines = [
        f"- {key}: `{value}`"
        for key, value in provenance.items()
        if key != "available" and value not in {None, ""}
    ]
    return "\n".join(
        [
            "# OEL MCP Report Brief",
            "",
            "Author the report from the deterministic packet and cited artifacts. OEL has not called a model provider.",
            "",
            "Required report sections: `Evidence` and `Limitations`.",
            "",
            "## Evidence References",
            "",
            *(artifact_lines or ["- No file artifacts were available; explain the incomplete evidence explicitly."]),
            *(query_lines or ["- No saved-query evidence was requested."]),
            "",
            "## Execution Provenance",
            "",
            *(provenance_lines or ["- No MCP execution manifest was available for projection."]),
            "",
            "## Non-Claims",
            "",
            *(f"- {item}" for item in list(packet.get("non_claims", []) or [])),
            "",
        ]
    )


def _audit_markdown(audit: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# OEL MCP Report Audit",
            "",
            f"- Status: `{audit['status']}`",
            f"- Packet: `{audit['packet_id']}`",
            "- Provider call made by OEL: `false`",
            "- Semantic claim review performed: `false`",
            "",
            "## Checks",
            "",
            *(f"- {key}: `{'passed' if value else 'failed'}`" for key, value in audit["checks"].items()),
            "",
            "## Non-Claims",
            "",
            *(f"- {item}" for item in audit["non_claims"]),
            "",
        ]
    )


def _markdown_headings(markdown: str) -> set[str]:
    return {
        match.group(1).strip().casefold()
        for line in markdown.splitlines()
        if (match := re.match(r"^#{1,6}\s+(.+?)\s*$", line))
    }


def _safe_identifier(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.:-]+", "_", value.strip()).strip("_")
    return (normalized or "artifact")[:120]


def _unique_identifier(candidate: str, existing: set[str]) -> str:
    if candidate not in existing:
        return candidate
    suffix = 2
    while f"{candidate[:115]}_{suffix}" in existing:
        suffix += 1
    return f"{candidate[:115]}_{suffix}"


def _media_type(path: Path) -> str:
    return {
        ".json": "application/json",
        ".md": "text/markdown",
        ".csv": "text/csv",
        ".sqlite": "application/vnd.sqlite3",
        ".png": "image/png",
        ".svg": "image/svg+xml",
        ".pdf": "application/pdf",
    }.get(path.suffix.lower(), "application/octet-stream")


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _source_path_within(path: Path, root: Path) -> bool:
    """Check containment lexically without following a source symlink."""

    try:
        _absolute_lexical(path).relative_to(_absolute_lexical(root))
    except ValueError:
        return False
    return True


def _logical_source_path(
    value: Path,
    *,
    physical_root: Path,
    logical_root: Path,
) -> Path | None:
    path = Path(value).expanduser()
    physical = _absolute_lexical(path if path.is_absolute() else physical_root / path)
    physical_base = _absolute_lexical(physical_root)
    logical_base = _absolute_lexical(logical_root)
    try:
        relative = physical.relative_to(physical_base)
    except ValueError:
        try:
            relative = physical.relative_to(logical_base)
        except ValueError:
            return None
    if not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
        return None
    return logical_base / relative


def _physical_source_path(
    value: Path,
    *,
    physical_root: Path,
    logical_root: Path,
) -> Path | None:
    path = Path(value).expanduser()
    physical_base = _absolute_lexical(physical_root)
    logical_base = _absolute_lexical(logical_root)
    if not path.is_absolute():
        path = physical_base / path
    path = _absolute_lexical(path)
    try:
        relative = path.relative_to(logical_base)
        return physical_base / relative
    except ValueError:
        try:
            path.relative_to(physical_base)
        except ValueError:
            return None
        return path


def _source_is_file(
    path: Path,
    *,
    physical_root: Path,
    logical_root: Path,
    reader: Any | None,
) -> bool:
    if not _source_path_within(path, physical_root):
        return False
    if reader is not None:
        logical = _logical_source_path(path, physical_root=physical_root, logical_root=logical_root)
        return logical is not None and bool(reader.is_file(logical))
    return path.is_file() and not path.is_symlink()


def _source_file_record(
    path: Path,
    *,
    physical_root: Path,
    logical_root: Path,
    reader: Any | None,
    maximum: int,
) -> dict[str, int | str] | None:
    if not _source_path_within(path, physical_root):
        return None
    if reader is not None:
        logical = _logical_source_path(path, physical_root=physical_root, logical_root=logical_root)
        if logical is None:
            return None
        try:
            return dict(reader.file_record(logical, max_bytes=maximum))
        except FileNotFoundError:
            return None
    if not path.is_file() or path.is_symlink():
        return None
    metadata = path.stat()
    if int(metadata.st_size) > int(maximum):
        raise ValueError("An artifact exceeds the MCP report size budget.")
    return {
        "bytes": int(metadata.st_size),
        "sha256": _sha256_file(path),
        "st_dev": int(metadata.st_dev),
        "st_ino": int(metadata.st_ino),
        "mtime_ns": int(metadata.st_mtime_ns),
        "ctime_ns": int(metadata.st_ctime_ns),
        "nlink": int(metadata.st_nlink),
    }


def _source_read_bytes(
    path: Path,
    *,
    physical_root: Path,
    logical_root: Path,
    reader: Any | None,
    maximum: int,
) -> bytes:
    if not _source_is_file(
        path,
        physical_root=physical_root,
        logical_root=logical_root,
        reader=reader,
    ):
        raise FileNotFoundError(path)
    if reader is not None:
        logical = _logical_source_path(path, physical_root=physical_root, logical_root=logical_root)
        if logical is None:
            raise PermissionError("Source path is outside the retained report input root.")
        return reader.read_bytes(logical, max_bytes=maximum)
    payload = path.read_bytes()
    if len(payload) > int(maximum):
        raise ValueError("Authorized source exceeds the MCP report size budget.")
    return payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _logical_output_dir(
    physical_output_dir: Path,
    *,
    writer: Any | None,
    logical_output_dir: str | Path | None,
) -> Path:
    """Map one producer directory from protected staging to final identity."""

    if writer is None:
        return Path(logical_output_dir) if logical_output_dir is not None else physical_output_dir
    relative = _lexical_relative_path(physical_output_dir, Path(writer.path))
    if relative is None:
        raise ValueError("The report output directory must be below the protected writer root.")
    logical_root = Path(logical_output_dir) if logical_output_dir is not None else Path(writer.logical_root)
    return logical_root / relative


def _current_output_writer() -> Any | None:
    try:
        from sim.security.output_writes import current_output_writer
    except ImportError:
        return None
    return current_output_writer()


def _write_output_bytes(path: Path, payload: bytes, *, writer: Any | None) -> Path:
    if writer is None:
        if path.suffix.lower() == ".json":
            _write_serialized_json(path, payload)
        else:
            path.write_bytes(payload)
        return path
    relative = _lexical_relative_path(path, Path(writer.path))
    if relative is None:
        raise ValueError("The report output path must be below the protected writer root.")
    written = writer.write_bytes(relative, payload)
    if _absolute_lexical(written) != _absolute_lexical(path):
        raise ValueError("The protected writer returned an unexpected physical output path.")
    return path


def _complete_output_manifest(
    manifest: dict[str, Any],
    *,
    status: str,
    logical_artifacts: list[str],
    physical_artifacts: list[str],
    writer: Any | None,
) -> dict[str, Any]:
    """Complete a manifest with logical artifact names and physical hash sources."""

    try:
        return complete_manifest(
            manifest,
            status=status,
            artifacts=logical_artifacts,
            artifact_files=physical_artifacts,
        )
    except TypeError as exc:
        # Keep direct CLI compatibility with older execution helpers while
        # refusing to silently degrade protected MCP provenance.
        if "artifact_files" not in str(exc):
            raise
        if writer is not None:
            raise RuntimeError(
                "The execution manifest helper lacks logical/physical artifact binding."
            ) from exc
        return complete_manifest(manifest, status=status, artifacts=physical_artifacts)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    _write_serialized_json(path, _serialized_json(payload))


def _serialized_json(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def _absolute_lexical(path: str | Path) -> Path:
    """Normalize a path without following symlinks."""

    return Path(os.path.abspath(os.fspath(Path(path).expanduser())))


def _lexical_relative_path(path: str | Path, root: str | Path) -> Path | None:
    candidate = _absolute_lexical(path)
    base = _absolute_lexical(root)
    try:
        return candidate.relative_to(base)
    except ValueError:
        return None


def _write_serialized_json(path: Path, payload: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(payload)
    tmp.replace(path)


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


__all__ = [
    "MAX_REPORT_SOURCE_BYTES",
    "MAX_REPORT_ARTIFACT_BYTES",
    "MAX_REPORT_TOTAL_ARTIFACT_BYTES",
    "REPORT_AUDIT_SCHEMA",
    "REPORT_PACKET_SCHEMA",
    "audit_report",
    "prepare_report_packet",
]
