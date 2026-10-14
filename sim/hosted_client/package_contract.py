"""Public-safe Hosted execution package contract and local validator."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from sim.study_planning import (
    bind_study_plan,
    bind_study_request,
    build_capability_manifest,
    canonical_sha256,
    compile_typed_operation_edges,
    discovery_capability_catalog,
    preflight_study_plan,
)
from sim.study_planning.config_validation import validate_config_path
from sim.study_planning.schema_validation import validate_schema
from sim.utils.io import SafeReadError, read_regular_file_nofollow, sha256_regular_file_nofollow

HOSTED_PRO_PACKAGE_SCHEMA_ID = "oel.hosted_pro_package.v1"
HOSTED_EXECUTION_PACKAGE_SCHEMA_ID = "oel.hosted_execution_package.v1"
HOSTED_EXECUTION_PACKAGE_VALIDATION_SCHEMA_ID = "oel.hosted_execution_package_validation.v1"
HOSTED_PRO_PACKAGE_VALIDATION_SCHEMA_ID = HOSTED_EXECUTION_PACKAGE_VALIDATION_SCHEMA_ID
DEFAULT_PACKAGE_MANIFEST = "oel-hosted-package.json"
MAX_CONTROL_DOCUMENT_BYTES = 2_000_000
MAX_INPUT_BYTES = 17_179_869_184

_IDENTIFIER = {"type": "string", "pattern": r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,159}"}
_RELATIVE_PATH = {"type": "string", "minLength": 1, "maxLength": 500}

HOSTED_PRO_PACKAGE_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://orbital-engineering-lab.vercel.app/schemas/oel-hosted-pro-package-v1.schema.json",
    "title": "OEL Hosted Pro package",
    "type": "object",
    "properties": {
        "schema": {"const": HOSTED_PRO_PACKAGE_SCHEMA_ID},
        "package_name": deepcopy(_IDENTIFIER),
        "request_path": deepcopy(_RELATIVE_PATH),
        "plan_path": deepcopy(_RELATIVE_PATH),
        "inputs": {
            "type": "array",
            "minItems": 1,
            "maxItems": 128,
            "items": {
                "type": "object",
                "properties": {
                    "input_id": deepcopy(_IDENTIFIER),
                    "relative_path": deepcopy(_RELATIVE_PATH),
                    "media_type": {
                        "type": "string",
                        "enum": [
                            "application/json",
                            "application/yaml",
                            "application/ccsds-kvn",
                            "text/csv",
                            "text/plain",
                            "application/octet-stream",
                        ],
                    },
                    "config_ref": {"type": ["string", "null"], "maxLength": 160},
                },
                "required": ["input_id", "relative_path", "media_type", "config_ref"],
                "additionalProperties": False,
            },
        },
        "retention": {
            "type": "object",
            "properties": {
                "policy_id": {
                    "type": "string",
                    "enum": [
                        "oel.retention.ephemeral-7d.v1",
                        "oel.retention.standard-30d.v1",
                        "oel.retention.extended-90d.v1",
                    ],
                },
                "delete_incomplete_uploads": {"const": True},
                "local_originals_retained": {"const": True},
            },
            "required": ["policy_id", "delete_incomplete_uploads", "local_originals_retained"],
            "additionalProperties": False,
        },
    },
    "required": ["schema", "package_name", "request_path", "plan_path", "inputs", "retention"],
    "additionalProperties": False,
}

HOSTED_EXECUTION_PACKAGE_SCHEMA: dict[str, Any] = deepcopy(HOSTED_PRO_PACKAGE_SCHEMA)
HOSTED_EXECUTION_PACKAGE_SCHEMA.update(
    {
        "$id": "https://orbital-engineering-lab.vercel.app/schemas/oel-hosted-execution-package-v1.schema.json",
        "title": "OEL Hosted execution package",
    }
)
HOSTED_EXECUTION_PACKAGE_SCHEMA["properties"]["schema"] = {
    "const": HOSTED_EXECUTION_PACKAGE_SCHEMA_ID
}

_KNOWN_INPUT_METADATA_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "content_sha256": {"type": "string", "pattern": r"[0-9a-f]{64}"},
        "input_bytes": {"type": "integer", "minimum": 0, "maximum": MAX_INPUT_BYTES},
        "measurement_model": {"type": "string", "minLength": 1, "maxLength": 160},
        "observation_count": {"type": "integer", "minimum": 1, "maximum": 10_000_000},
        "normal_point_count": {"type": "integer", "minimum": 1, "maximum": 10_000_000},
        "sample_count": {"type": "integer", "minimum": 1, "maximum": 10_000_000},
        "arc_duration_s": {"type": "number", "minimum": 0.001, "maximum": 31_536_000},
        "station_count": {"type": "integer", "minimum": 1, "maximum": 100_000},
        "propagation_steps": {"type": "integer", "minimum": 1, "maximum": 1_000_000_000},
        "estimator_evaluations": {"type": "integer", "minimum": 1, "maximum": 1_000_000_000},
        "batch_evaluations": {"type": "integer", "minimum": 1, "maximum": 100_000_000},
    },
}


@dataclass(frozen=True, slots=True)
class PackageIssue:
    code: str
    path: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code, "path": self.path, "message": self.message}


def hosted_pro_package_schema() -> dict[str, Any]:
    """Return the legacy Pro-specific package manifest schema."""

    return deepcopy(HOSTED_PRO_PACKAGE_SCHEMA)


def hosted_execution_package_schema() -> dict[str, Any]:
    """Return the public package schema for either public or Pro hosted execution."""

    return deepcopy(HOSTED_EXECUTION_PACKAGE_SCHEMA)


def validate_hosted_execution_package(
    package: str | Path,
    *,
    workspace_root: str | Path | None = None,
) -> dict[str, Any]:
    """Validate and content-bind one local package without executing OEL."""

    issues: list[PackageIssue] = []
    package_value = Path(package).expanduser()
    manifest_path = package_value / DEFAULT_PACKAGE_MANIFEST if package_value.is_dir() else package_value
    package_root = manifest_path.parent.absolute()
    authorized_root = Path(workspace_root or package_root).expanduser().absolute()
    try:
        package_root.relative_to(authorized_root)
    except ValueError:
        issues.append(PackageIssue("package.outside_workspace", "$", "Package must remain inside the selected workspace."))
        return _invalid_receipt(manifest_path, issues)

    try:
        manifest_bytes = read_regular_file_nofollow(manifest_path, min_bytes=1, max_bytes=MAX_CONTROL_DOCUMENT_BYTES)
        manifest = _decode_json_object(manifest_bytes, label="package manifest")
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        issues.append(PackageIssue("package.manifest_unreadable", "$", str(exc)))
        return _invalid_receipt(manifest_path, issues)

    manifest_content_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    manifest_schema_id = str(manifest.get("schema", ""))
    manifest_schema = {
        HOSTED_EXECUTION_PACKAGE_SCHEMA_ID: HOSTED_EXECUTION_PACKAGE_SCHEMA,
        HOSTED_PRO_PACKAGE_SCHEMA_ID: HOSTED_PRO_PACKAGE_SCHEMA,
    }.get(manifest_schema_id)
    if manifest_schema is None:
        issues.append(
            PackageIssue(
                "package.manifest_invalid",
                "$.schema",
                f"Unsupported Hosted execution package schema {manifest_schema_id!r}.",
            )
        )
        manifest_schema = HOSTED_EXECUTION_PACKAGE_SCHEMA
    for item in validate_schema(manifest, manifest_schema):
        issues.append(PackageIssue("package.manifest_invalid", item.path, item.message))
    if issues:
        return _invalid_receipt(
            manifest_path,
            issues,
            manifest=manifest,
            manifest_sha256=manifest_content_sha256,
        )

    manifest_sha256 = manifest_content_sha256
    manifest_contract_sha256 = canonical_sha256(manifest)
    input_rows = list(manifest["inputs"])
    input_ids = [str(item["input_id"]) for item in input_rows]
    if len(set(input_ids)) != len(input_ids):
        issues.append(PackageIssue("package.input_id_duplicate", "$.inputs", "Input identifiers must be unique."))
    logical_paths = [str(manifest["request_path"]), str(manifest["plan_path"]), *(str(item["relative_path"]) for item in input_rows)]
    if len(set(logical_paths)) != len(logical_paths):
        issues.append(PackageIssue("package.path_duplicate", "$", "Request, plan, and input paths must be distinct."))

    resolved: dict[str, Path] = {}
    for label, relative in (
        ("request", str(manifest["request_path"])),
        ("plan", str(manifest["plan_path"])),
        *((str(item["input_id"]), str(item["relative_path"])) for item in input_rows),
    ):
        try:
            resolved[label] = _package_file(package_root, relative)
        except ValueError as exc:
            issues.append(PackageIssue("package.path_unsafe", f"$.{label}", str(exc)))
    if issues:
        return _invalid_receipt(
            manifest_path,
            issues,
            manifest=manifest,
            manifest_sha256=manifest_sha256,
        )

    try:
        request_bytes = read_regular_file_nofollow(resolved["request"], min_bytes=1, max_bytes=MAX_CONTROL_DOCUMENT_BYTES)
        plan_bytes = read_regular_file_nofollow(resolved["plan"], min_bytes=1, max_bytes=MAX_CONTROL_DOCUMENT_BYTES)
        request_proposal = _decode_json_object(request_bytes, label="study request")
        plan_proposal = _decode_json_object(plan_bytes, label="study plan")
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        issues.append(PackageIssue("package.control_document_invalid", "$", str(exc)))
        return _invalid_receipt(
            manifest_path,
            issues,
            manifest=manifest,
            manifest_sha256=manifest_sha256,
        )

    request_inputs = {
        str(item.get("input_id", "")): item
        for item in list(request_proposal.get("inputs", []) or [])
        if isinstance(item, dict)
    }
    if set(request_inputs) != set(input_ids):
        issues.append(
            PackageIssue(
                "package.input_inventory_mismatch",
                "$.inputs",
                "Package inputs must exactly match the study request input identifiers.",
            )
        )

    inventory: list[dict[str, Any]] = []
    normalized_configs: dict[str, dict[str, Any]] = {}
    config_receipts: list[dict[str, Any]] = []
    seen_config_refs: set[str] = set()
    for index, package_input in enumerate(input_rows):
        input_id = str(package_input["input_id"])
        request_input = request_inputs.get(input_id)
        if request_input is None:
            continue
        source = resolved[input_id]
        try:
            digest, byte_count = sha256_regular_file_nofollow(source, min_bytes=1, max_bytes=MAX_INPUT_BYTES)
        except (OSError, ValueError) as exc:
            issues.append(PackageIssue("package.input_unreadable", f"$.inputs[{index}]", str(exc)))
            continue
        declared_digest = request_input.get("content_sha256")
        if declared_digest is not None and str(declared_digest) != digest:
            issues.append(
                PackageIssue(
                    "package.input_digest_mismatch",
                    f"$.inputs[{index}]",
                    f"Input {input_id!r} does not match its declared SHA-256.",
                )
            )
        metadata = dict(request_input.get("metadata", {}) or {})
        if metadata.get("content_sha256") not in {None, digest} or metadata.get("input_bytes") not in {None, byte_count}:
            issues.append(
                PackageIssue(
                    "package.input_metadata_mismatch",
                    f"$.inputs[{index}]",
                    f"Input {input_id!r} declares content metadata that does not match the selected file.",
                )
            )
        metadata.update({"content_sha256": digest, "input_bytes": byte_count})
        request_input["content_sha256"] = digest
        request_input["metadata"] = metadata
        for item in validate_schema(metadata, _KNOWN_INPUT_METADATA_SCHEMA, path=f"$.request.inputs[{index}].metadata"):
            issues.append(PackageIssue("package.input_metadata_invalid", item.path, item.message))
        _inspect_payload_identity(
            source,
            request_input=request_input,
            media_type=str(package_input["media_type"]),
            issue_path=f"$.inputs[{index}]",
            issues=issues,
        )
        config_ref = package_input.get("config_ref")
        if str(request_input.get("kind", "")) == "scenario_config":
            if not config_ref:
                issues.append(PackageIssue("package.config_ref_missing", f"$.inputs[{index}].config_ref", "Scenario inputs require a config_ref."))
            elif str(config_ref) in seen_config_refs:
                issues.append(PackageIssue("package.config_ref_duplicate", f"$.inputs[{index}].config_ref", "Scenario config_ref values must be unique."))
            else:
                seen_config_refs.add(str(config_ref))
                receipt, document = validate_config_path(str(config_ref), source, workspace_root=authorized_root)
                config_receipts.append(receipt)
                if document is not None:
                    if str(document.get("simulator", {}).get("central_body", "earth")).lower() != "earth":
                        issues.append(PackageIssue(
                            "package.planetary_unsupported", f"$.inputs[{index}].config_ref",
                            "Planetary ONP is available locally; Hosted execution is not qualified for non-Earth origins.",
                        ))
                    elif document.get("simulator", {}).get("additional_gravity") is not None:
                        issues.append(PackageIssue(
                            "package.additional_gravity_unsupported", f"$.inputs[{index}].config_ref",
                            "Extended third-body gravity is available locally; Hosted execution is not qualified for additional harmonic fields.",
                        ))
                    else:
                        normalized_configs[str(config_ref)] = document
        elif config_ref is not None:
            issues.append(PackageIssue("package.config_ref_not_applicable", f"$.inputs[{index}].config_ref", "Only scenario_config inputs may declare config_ref."))
        inventory.append(
            {
                "input_id": input_id,
                "kind": str(request_input.get("kind", "")),
                "schema_id": str(request_input.get("schema_id", "")),
                "relative_path": str(package_input["relative_path"]),
                "media_type": str(package_input["media_type"]),
                "content_sha256": digest,
                "bytes": byte_count,
                "config_ref": config_ref,
            }
        )

    planning_result: dict[str, Any] | None = None
    edge_receipt: dict[str, Any] | None = None
    bound_request: dict[str, Any] | None = None
    bound_plan: dict[str, Any] | None = None
    if not issues:
        try:
            bound_request = bind_study_request(request_proposal)
            catalog = discovery_capability_catalog()
            operations = list(plan_proposal.get("operations", []) or [])
            compiled = compile_typed_operation_edges(
                operations,
                request_inputs=bound_request["inputs"],
                catalog=catalog,
            )
            edge_receipt = compiled["receipt"]
            plan_proposal["operations"] = compiled["operations"]
            _bind_transfer_digests(plan_proposal, bound_request)
            capability_manifest = build_capability_manifest(catalog)
            bound_plan = bind_study_plan(
                plan_proposal,
                request_id=bound_request["request_id"],
                request_sha256=bound_request["request_sha256"],
                capability_manifest_sha256=capability_manifest["manifest_sha256"],
                planner_version=capability_manifest["planner_version"],
            )
            planning_result = preflight_study_plan(
                bound_request,
                bound_plan,
                catalog=catalog,
                normalized_configs=normalized_configs,
                config_receipts=config_receipts,
                allow_unbound_pro_planning=True,
            )
        except (KeyError, TypeError, ValueError) as exc:
            issues.append(PackageIssue("package.plan_invalid", "$.plan", str(exc)))

    if planning_result is not None:
        pro_ids = list(planning_result["execution_route"]["pro_capability_ids"])
        if manifest_schema_id == HOSTED_PRO_PACKAGE_SCHEMA_ID and not pro_ids:
            issues.append(PackageIssue("package.pro_capability_missing", "$.plan.operations", "A Hosted Pro package must request at least one published Pro capability."))
        if planning_result["status"] != "PLAN_VALID" or planning_result["execution_route"]["route"] not in {
            "LOCAL_FREE_AVAILABLE",
            "HOSTED_PRO_REQUIRED",
        }:
            issues.extend(
                PackageIssue(str(item["code"]), str((item.get("paths") or ["$.plan"])[0]), str(item["message"]))
                for item in planning_result.get("findings", [])
                if item.get("severity") in {"error", "blocker"}
            )
            if not any(not item.code.startswith("package.") for item in issues):
                issues.append(PackageIssue("package.plan_not_hosted_ready", "$.plan", "The plan did not reach a Hosted-eligible planning route."))

    request_source_sha256 = hashlib.sha256(request_bytes).hexdigest()
    plan_source_sha256 = hashlib.sha256(plan_bytes).hexdigest()
    bound_request_sha256 = None if bound_request is None else str(bound_request["request_sha256"])
    bound_plan_sha256 = None if bound_plan is None else str(bound_plan["plan_sha256"])
    package_sha256 = canonical_sha256(
        {
            "manifest_sha256": manifest_sha256,
            "manifest_contract_sha256": manifest_contract_sha256,
            "request_source_sha256": request_source_sha256,
            "plan_source_sha256": plan_source_sha256,
            "bound_request_sha256": bound_request_sha256,
            "bound_plan_sha256": bound_plan_sha256,
            "inputs": inventory,
        }
    )
    route = None if planning_result is None else planning_result["execution_route"]["route"]
    local_available = not issues and route == "LOCAL_FREE_AVAILABLE"
    hosted_available = not issues and route in {"LOCAL_FREE_AVAILABLE", "HOSTED_PRO_REQUIRED"}
    receipt: dict[str, Any] = {
        "schema": HOSTED_EXECUTION_PACKAGE_VALIDATION_SCHEMA_ID,
        "status": "CLIENT_INVALID" if issues else "CLIENT_VALID",
        "package_name": str(manifest["package_name"]),
        "package_sha256": package_sha256,
        "manifest_sha256": manifest_sha256,
        "manifest_contract_sha256": manifest_contract_sha256,
        "control_documents": {
            "request": {
                "relative_path": str(manifest["request_path"]),
                "source_sha256": request_source_sha256,
                "bound_sha256": bound_request_sha256,
            },
            "plan": {
                "relative_path": str(manifest["plan_path"]),
                "source_sha256": plan_source_sha256,
                "bound_sha256": bound_plan_sha256,
            },
        },
        "retention": deepcopy(manifest["retention"]),
        "inventory": inventory,
        "input_count": len(inventory),
        "total_input_bytes": sum(int(item["bytes"]) for item in inventory),
        "issues": [item.to_dict() for item in issues],
        "bound_request": bound_request,
        "bound_plan": bound_plan,
        "edge_compilation_receipt": edge_receipt,
        "planning_result": planning_result,
        "execution_options": {
            "local": {"available": local_available, "oel_execution_amount_usd": 0 if local_available else None},
            "hosted": {"available": False, "quote_required": hosted_available},
        },
        "recommendation": {
            "option": "local" if local_available else None,
            "requirement": not local_available,
        },
        "hosted_access": {
            "status": "closed_alpha",
            "public_registration_available": False,
            "message": "Hosted OEL is a closed alpha; access is not publicly available. Package validation does not establish service access.",
        },
        "local_execution_available": local_available,
        "hosted_preflight_required": hosted_available,
        "execution_authorized": False,
        "payment_authorized": False,
        "pro_code_imported": False,
        "files_uploaded": False,
        "non_claims": [
            "Local validation does not execute OEL or upload package bytes.",
            "Local validation does not establish Hosted availability, entitlement, price, or approval.",
            "The Hosted service must revalidate the exact sealed bytes with its authoritative validators.",
            "Schema and planning validity are not scientific qualification or operational authority.",
        ],
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    return receipt


def validate_hosted_pro_package(
    package: str | Path,
    *,
    workspace_root: str | Path | None = None,
) -> dict[str, Any]:
    """Validate the legacy Pro-specific package contract."""

    return validate_hosted_execution_package(package, workspace_root=workspace_root)


def _invalid_receipt(
    manifest_path: Path,
    issues: list[PackageIssue],
    *,
    manifest: Mapping[str, Any] | None = None,
    manifest_sha256: str | None = None,
) -> dict[str, Any]:
    receipt: dict[str, Any] = {
        "schema": HOSTED_EXECUTION_PACKAGE_VALIDATION_SCHEMA_ID,
        "status": "CLIENT_INVALID",
        "package_name": str((manifest or {}).get("package_name", manifest_path.parent.name or "unknown")),
        "package_sha256": None,
        "manifest_sha256": manifest_sha256,
        "manifest_contract_sha256": None if manifest is None else canonical_sha256(manifest),
        "control_documents": None,
        "retention": deepcopy(dict((manifest or {}).get("retention", {}) or {})),
        "inventory": [],
        "input_count": 0,
        "total_input_bytes": 0,
        "issues": [item.to_dict() for item in issues],
        "bound_request": None,
        "bound_plan": None,
        "edge_compilation_receipt": None,
        "planning_result": None,
        "execution_options": {
            "local": {"available": False, "oel_execution_amount_usd": None},
            "hosted": {"available": False, "quote_required": False},
        },
        "recommendation": {"option": None, "requirement": True},
        "local_execution_available": False,
        "hosted_preflight_required": False,
        "execution_authorized": False,
        "payment_authorized": False,
        "pro_code_imported": False,
        "files_uploaded": False,
        "non_claims": [
            "Local validation does not execute OEL or upload package bytes.",
            "An invalid package cannot receive a Hosted quote or execution approval.",
        ],
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    return receipt


def _package_file(root: Path, relative: str) -> Path:
    if "\\" in relative:
        raise ValueError("Package paths must use portable forward slashes.")
    path = PurePosixPath(relative)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"Unsafe package-relative path: {relative!r}")
    return root.joinpath(*path.parts)


def _decode_json_object(content: bytes, *, label: str) -> dict[str, Any]:
    def pairs(rows: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in rows:
            if key in value:
                raise ValueError(f"Duplicate JSON field {key!r} in {label}.")
            value[key] = item
        return value

    def constant(value: str) -> None:
        raise ValueError(f"Non-finite JSON value {value!r} in {label}.")

    value = json.loads(content.decode("utf-8"), object_pairs_hook=pairs, parse_constant=constant)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {label}.")
    return value


def _inspect_payload_identity(
    path: Path,
    *,
    request_input: Mapping[str, Any],
    media_type: str,
    issue_path: str,
    issues: list[PackageIssue],
) -> None:
    if media_type != "application/json":
        return
    try:
        content = read_regular_file_nofollow(path, min_bytes=1, max_bytes=MAX_CONTROL_DOCUMENT_BYTES)
        payload = _decode_json_object(content, label=f"input {request_input.get('input_id', '')!r}")
    except SafeReadError:
        return
    except (UnicodeDecodeError, ValueError) as exc:
        issues.append(PackageIssue("package.input_json_invalid", issue_path, str(exc)))
        return
    expected = str(request_input.get("schema_id", "") or "")
    declared = str(payload.get("schema") or payload.get("schema_version") or "")
    if expected and not expected.startswith("oel.artifact.") and declared != expected:
        issues.append(
            PackageIssue(
                "package.input_schema_mismatch",
                issue_path,
                f"JSON input declares schema {declared!r}; expected {expected!r}.",
            )
        )


def _bind_transfer_digests(plan: dict[str, Any], request: Mapping[str, Any]) -> None:
    request_by_id = {str(item["input_id"]): item for item in request["inputs"]}
    for item in list(plan.get("transfers", []) or []):
        input_id = str(item.get("input_id", ""))
        source = request_by_id.get(input_id)
        if source is None:
            continue
        expected = str(source["content_sha256"])
        declared = item.get("content_sha256")
        if declared not in {None, expected}:
            raise ValueError(f"Transfer {input_id!r} does not match the selected input SHA-256.")
        item["content_sha256"] = expected


__all__ = [
    "DEFAULT_PACKAGE_MANIFEST",
    "HOSTED_EXECUTION_PACKAGE_SCHEMA",
    "HOSTED_EXECUTION_PACKAGE_SCHEMA_ID",
    "HOSTED_EXECUTION_PACKAGE_VALIDATION_SCHEMA_ID",
    "HOSTED_PRO_PACKAGE_SCHEMA",
    "HOSTED_PRO_PACKAGE_SCHEMA_ID",
    "HOSTED_PRO_PACKAGE_VALIDATION_SCHEMA_ID",
    "hosted_execution_package_schema",
    "hosted_pro_package_schema",
    "validate_hosted_execution_package",
    "validate_hosted_pro_package",
]
