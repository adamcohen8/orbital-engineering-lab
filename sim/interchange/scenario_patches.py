from __future__ import annotations

import hashlib
import math
import os
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from sim.data_io.common import read_text, strict_json
from sim.data_io.contracts import DataIOError
from sim.scenarios import ScenarioArtifact

from .manifests import bind_output_path, write_handoff_manifest, write_json_document
from .materialization import (
    _ensure_distinct_materialization_paths,
    _failed_manifest,
    _result,
    materialize_scenario_document,
)
from .provenance import canonical_json_bytes, compute_product_id, sha256_file
from .validation import load_interchange_document, validate_product

SCENARIO_PATCH_ADAPTER_ID = "oel.scenario_patch_materializer"
SCENARIO_PATCH_ADAPTER_VERSION = "2"
PATCH_OPERATION_KINDS = {
    "mission_burn",
    "duration_extension",
    "controller_pointer",
    "flight_software_stack",
    "scenario_override",
}


class ScenarioPatchError(ValueError):
    """Raised when a typed scenario patch cannot be safely produced or applied."""


def scenario_document(path: str | Path) -> tuple[dict[str, Any], str, str]:
    source = Path(path).expanduser().resolve()
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ScenarioPatchError("Source scenario YAML must contain a mapping.")
    artifact = ScenarioArtifact.from_dict(raw)
    digest = hashlib.sha256(canonical_json_bytes(artifact.to_artifact_dict())).hexdigest()
    return raw, sha256_file(source), digest


def build_scenario_patch_product(
    source_scenario: str | Path,
    *,
    patch_type: str,
    selection: Mapping[str, Any],
    operations: Sequence[Mapping[str, Any]],
    evidence: Mapping[str, Any],
    producer_capability_id: str,
    producer_run_id: str,
    source_artifacts: Sequence[str | Path] = (),
    disposition: str = "accepted",
    producer_status: str = "ready",
    warnings: Sequence[str] = (),
    non_claims: Sequence[str] = (),
    data_markings: Mapping[str, Any] | None = None,
    output_path: str | Path | None = None,
    logical_output_path: str | Path | None = None,
    writer: Any | None = None,
) -> dict[str, Any]:
    source_path = Path(source_scenario).expanduser().resolve()
    target, logical_target = bind_output_path(
        output_path if output_path is not None else source_path.with_suffix(".scenario_patch.json"),
        logical_output_path,
        writer=writer,
    )
    artifacts = [source_path, *(Path(item).expanduser().resolve() for item in source_artifacts)]
    resolved_target = target.expanduser().resolve()
    for artifact in artifacts:
        if artifact.expanduser().resolve() == resolved_target:
            raise ScenarioPatchError(
                f"Patch output must be distinct from source artifact: {artifact}."
            )
    raw, source_hash, source_digest = scenario_document(source_path)
    seen: set[Path] = set()
    provenance = []
    for artifact in artifacts:
        if artifact in seen:
            continue
        seen.add(artifact)
        if not artifact.is_file():
            raise ScenarioPatchError(f"Patch source artifact does not exist: {artifact}")
        provenance.append(
            {
                "artifact_id": artifact.stem,
                "sha256": sha256_file(artifact),
                "path": os.path.relpath(artifact, start=logical_target.parent),
                "media_type": "application/yaml" if artifact.suffix in {".yaml", ".yml"} else "application/json",
                "size_bytes": artifact.stat().st_size,
            }
        )
    created = _now_utc()
    product: dict[str, Any] = {
        "schema_id": "oel-product-envelope-v1",
        "schema_version": 1,
        "product_kind": "oel.scenario_patch",
        "product_id": "oel.scenario_patch:" + "0" * 64,
        "created_utc": created,
        "producer": {
            "capability_id": str(producer_capability_id),
            "oel_version": _oel_version(),
            "run_id": str(producer_run_id),
        },
        "payload": {
            "source_scenario": {
                "scenario_name": str(raw.get("scenario_name", "") or ""),
                "sha256": source_hash,
                "canonical_digest": source_digest,
            },
            "patch": {
                "patch_type": str(patch_type),
                "operations": [deepcopy(dict(item)) for item in operations],
            },
            "selection": deepcopy(dict(selection)),
            "evidence": deepcopy(dict(evidence)),
        },
        "quality": {
            "disposition": str(disposition),
            "producer_status": str(producer_status),
            "gates": {"operation_count": len(operations), "source_scenario_hash_bound": True},
            "warnings": [str(item) for item in warnings],
            "non_claims": [str(item) for item in non_claims],
        },
        "freshness": {
            "integrity_status": "current",
            "age_status": "not_applicable",
            "reference_epoch_jd_utc": None,
            "evaluated_utc": created,
            "policy": {},
        },
        "provenance": {
            "source_artifacts": provenance,
            "source_product_ids": [],
            "transformations": [
                {
                    "transformation_id": "typed_scenario_patch_emission",
                    "version": "1",
                    "details": {"patch_type": str(patch_type), "operation_count": len(operations)},
                }
            ],
        },
        "data_markings": _markings(data_markings),
    }
    product["product_id"] = compute_product_id(product)
    report = validate_product(product, source_path=logical_target)
    if not report.valid:
        messages = "; ".join(f"{issue.path}: {issue.message}" for issue in report.errors)
        raise ScenarioPatchError(f"Generated scenario patch failed validation: {messages}")
    return product


def write_scenario_patch_product(
    product: Mapping[str, Any],
    path: str | Path,
    *,
    logical_output_path: str | Path | None = None,
    writer: Any | None = None,
    overwrite: bool = True,
) -> dict[str, Any]:
    target, logical_target = bind_output_path(
        path,
        logical_output_path,
        writer=writer,
    )
    logical_target = logical_target.expanduser().resolve()
    for item in list(dict(product.get("provenance", {}) or {}).get("source_artifacts", []) or []):
        source_ref = str(dict(item or {}).get("path", "") or "").strip()
        if not source_ref:
            continue
        source_path = Path(source_ref)
        if not source_path.is_absolute():
            source_path = logical_target.parent / source_path
        if source_path.expanduser().resolve() == logical_target:
            raise ScenarioPatchError(
                f"Patch output must be distinct from source artifact: {source_path.resolve()}."
            )
    write_json_document(product, target, writer=writer, overwrite=overwrite)
    report = validate_product(product, source_path=logical_target)
    return {
        "product_path": str(target),
        "product_id": product.get("product_id"),
        "selection_id": dict(dict(product.get("payload", {}) or {}).get("selection", {}) or {}).get("selection_id"),
        "disposition": dict(product.get("quality", {}) or {}).get("disposition"),
        "valid": report.valid,
        "promotable": report.promotable,
    }


def select_patch_product(index_path: str | Path, selection_id: str) -> Path:
    source = Path(index_path).expanduser().resolve()
    try:
        value = strict_json(read_text(source))
    except (FileNotFoundError, DataIOError) as exc:
        raise ScenarioPatchError(f"Patch index is not valid bounded JSON: {source}") from exc
    if not isinstance(value, dict) or value.get("schema_id") != "oel-scenario-patch-index-v1":
        raise ScenarioPatchError("Patch index has an unsupported schema.")
    matches = [dict(item) for item in list(value.get("patches", []) or []) if str(dict(item).get("selection_id", "")) == str(selection_id)]
    if len(matches) != 1:
        raise ScenarioPatchError(f"selection_id {selection_id!r} must match exactly one patch index entry.")
    path = Path(str(matches[0].get("product_path", "")))
    if not path.is_absolute():
        path = source.parent / path
    if not path.is_file():
        raise ScenarioPatchError(f"Selected patch product does not exist: {path}")
    return path.resolve()


def materialize_scenario_patch(
    patch_product: str | Path,
    source_scenario: str | Path,
    *,
    scenario_name: str,
    scenario_path: str | Path,
    output_dir: str | Path,
    manifest_path: str | Path | None = None,
    logical_scenario_path: str | Path | None = None,
    logical_manifest_path: str | Path | None = None,
    logical_output_dir: str | Path | None = None,
    validation_path_policy: Any | None = None,
    trust_plugins: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    patch_path = Path(patch_product).expanduser().resolve()
    source_path = Path(source_scenario).expanduser().resolve()
    destination = Path(scenario_path).expanduser().resolve()
    manifest_target = Path(manifest_path).expanduser().resolve() if manifest_path is not None else destination.with_name(f"{destination.stem}.handoff_manifest.json")
    logical_destination = (
        Path(logical_scenario_path).expanduser().resolve()
        if logical_scenario_path is not None
        else destination
    )
    logical_manifest_target = (
        Path(logical_manifest_path).expanduser().resolve()
        if logical_manifest_path is not None
        else logical_destination.with_name(f"{logical_destination.stem}.handoff_manifest.json")
    )
    logical_output = (
        Path(logical_output_dir).expanduser().resolve()
        if logical_output_dir is not None
        else Path(output_dir).expanduser().resolve()
    )
    _ensure_distinct_materialization_paths(
        (
            ("patch source product", patch_path),
            ("source scenario", source_path),
            ("scenario", destination),
            ("manifest", manifest_target),
        ),
        ScenarioPatchError,
    )
    product = load_interchange_document(patch_path)
    product_report = validate_product(product, source_path=patch_path)
    source_id = str(product.get("product_id", "") or "")
    created = _now_utc()
    markings = deepcopy(dict(product.get("data_markings", {}) or {}))
    source_binding = dict(dict(product.get("payload", {}) or {}).get("source_scenario", {}) or {})
    source_scenario_key = f"source_scenario:{source_binding.get('scenario_name', source_path.stem)}"
    base_manifest: dict[str, Any] = {
        "schema_id": "oel-handoff-manifest-v1",
        "schema_version": 1,
        "manifest_id": "oel.handoff_manifest:" + "0" * 64,
        "created_utc": created,
        "source_product_ids": [source_id],
        "source_hashes": {
            source_id: sha256_file(patch_path),
            source_scenario_key: str(source_binding.get("sha256", "") or ""),
        },
        "adapter": {"adapter_id": SCENARIO_PATCH_ADAPTER_ID, "adapter_version": SCENARIO_PATCH_ADAPTER_VERSION},
        "materialization_options": {
            "scenario_name": str(scenario_name), "scenario_path": str(logical_destination),
            "source_scenario": str(source_path), "output_dir": str(logical_output),
        },
        "defaults_applied": {"review_detail": "standard", "execution_occurred": False},
        "overrides": ([{"field": "scenario_path", "reason": "explicit overwrite authorization"}] if overwrite else []),
        "source_markings": markings,
        "output_markings": markings,
        "output": {"kind": "patched_scenario", "path": str(logical_destination), "digest": "", "status": "pending"},
        "validation": {"safe_validation_result": {"status": "not_run"}, "ordinary_validation_result": {"status": "not_run"}},
        "warnings": [], "failures": [],
        "recommended_next_action": "Review patch and source-scenario compatibility.",
        "execution_occurred": False,
    }
    failures = [issue.to_dict() for issue in product_report.issues if issue.severity in {"error", "blocker"}]
    if product.get("product_kind") != "oel.scenario_patch":
        failures.append({"code": "patch.kind", "path": "$.product_kind", "message": "Expected oel.scenario_patch."})
    try:
        raw, source_hash, source_digest = scenario_document(source_path)
    except (OSError, ValueError) as exc:
        raw, source_hash, source_digest = {}, "", ""
        failures.append({"code": "patch.source_unreadable", "path": "$.payload.source_scenario", "message": str(exc)})
    if source_hash and source_binding.get("sha256") != source_hash:
        failures.append({"code": "patch.source_hash_mismatch", "path": "$.payload.source_scenario.sha256", "message": "Source scenario bytes have changed since patch emission."})
    if source_digest and source_binding.get("canonical_digest") != source_digest:
        failures.append({"code": "patch.source_digest_mismatch", "path": "$.payload.source_scenario.canonical_digest", "message": "Source scenario semantics have changed since patch emission."})
    if failures or not product_report.valid or not product_report.promotable:
        manifest = _failed_manifest(base_manifest, output_status="blocked", failures=failures, next_action="Use the exact source scenario and an accepted, current typed patch product.")
        write_handoff_manifest(manifest, manifest_target)
        return _result("blocked", destination, manifest_target, manifest, product_report.to_dict())

    scenario = deepcopy(raw)
    try:
        for operation in list(dict(product["payload"])["patch"]["operations"]):
            _apply_operation(scenario, dict(operation))
    except ScenarioPatchError as exc:
        failures.append(
            {
                "code": "patch.application_failed",
                "path": "$.payload.patch.operations",
                "message": str(exc),
            }
        )
        manifest = _failed_manifest(
            base_manifest,
            output_status="blocked",
            failures=failures,
            next_action="Regenerate the patch from this exact source scenario.",
        )
        write_handoff_manifest(manifest, manifest_target)
        return _result("blocked", destination, manifest_target, manifest, product_report.to_dict())
    scenario["scenario_name"] = str(scenario_name).strip()
    _align_duration_to_timestep(scenario, base_manifest=base_manifest)
    outputs = scenario.setdefault("outputs", {})
    outputs["output_dir"] = str(logical_output)
    outputs.setdefault("review", {"enabled": True, "detail": "standard"})
    metadata = scenario.setdefault("metadata", {})
    metadata["handoff"] = {
        "source_product_id": source_id,
        "source_scenario_sha256": source_hash,
        "source_scenario_digest": source_digest,
        "patch_type": dict(dict(product["payload"])["patch"])["patch_type"],
        "selection": deepcopy(dict(dict(product["payload"])["selection"])),
        "adapter_id": SCENARIO_PATCH_ADAPTER_ID,
        "adapter_version": SCENARIO_PATCH_ADAPTER_VERSION,
        "manifest_path": str(logical_manifest_target),
        "execution_occurred": False,
    }
    return materialize_scenario_document(
        scenario=scenario, destination=destination, manifest_target=manifest_target,
        protected_paths=(("patch source product", patch_path), ("source scenario", source_path)),
        collision_error_type=ScenarioPatchError,
        logical_destination=logical_destination, logical_manifest_target=logical_manifest_target,
        logical_output_dir=logical_output,
        validation_path_policy=validation_path_policy,
        base_manifest=base_manifest, product_report=product_report.to_dict(), output_kind="patched_scenario",
        trust_plugins=trust_plugins, overwrite=overwrite,
    )


def _align_duration_to_timestep(
    scenario: dict[str, Any], *, base_manifest: dict[str, Any]
) -> None:
    simulator = scenario.get("simulator")
    if not isinstance(simulator, dict):
        return
    try:
        duration_s = float(simulator.get("duration_s"))
        dt_s = float(simulator.get("dt_s"))
    except (TypeError, ValueError):
        return
    if not math.isfinite(duration_s) or not math.isfinite(dt_s) or duration_s <= 0.0 or dt_s <= 0.0:
        return
    steps = duration_s / dt_s
    nearest = round(steps)
    if math.isclose(steps, nearest, rel_tol=0.0, abs_tol=1.0e-9):
        return
    aligned = float(math.ceil(steps - 1.0e-12) * dt_s)
    simulator["duration_s"] = aligned
    base_manifest.setdefault("overrides", []).append(
        {
            "field": "simulator.duration_s",
            "source_value": duration_s,
            "output_value": aligned,
            "reason": "Round patch-derived duration up to the next complete simulator timestep.",
        }
    )


def _apply_operation(root: dict[str, Any], operation: Mapping[str, Any]) -> None:
    op = str(operation.get("op", ""))
    path = str(operation.get("path", ""))
    tokens = [token for token in path.split(".") if token]
    if not tokens:
        raise ScenarioPatchError("Patch operation path must be non-empty.")
    current: Any = root
    for token in tokens[:-1]:
        if not isinstance(current, dict):
            raise ScenarioPatchError(f"Patch path cannot descend through {token!r}.")
        if token not in current:
            if op == "upsert" or (op == "append" and token == tokens[-2]):
                current[token] = {}
            else:
                raise ScenarioPatchError(f"Patch replace path does not exist: {path}")
        current = current[token]
    if not isinstance(current, dict):
        raise ScenarioPatchError(f"Patch path parent is not a mapping: {path}")
    key = tokens[-1]
    if op == "replace":
        if key not in current:
            raise ScenarioPatchError(f"Patch replace path does not exist: {path}")
        current[key] = deepcopy(operation.get("value"))
    elif op == "append":
        target = current.setdefault(key, [])
        if not isinstance(target, list):
            raise ScenarioPatchError(f"Patch append path is not a list: {path}")
        target.append(deepcopy(operation.get("value")))
    elif op == "upsert":
        current[key] = deepcopy(operation.get("value"))
    else:
        raise ScenarioPatchError(f"Unsupported patch operation {op!r}.")


def _markings(value: Mapping[str, Any] | None) -> dict[str, Any]:
    raw = dict(value or {})
    return {
        "scope": str(raw.get("scope", "private_pro") or "private_pro"),
        "handling": str(raw.get("handling", "private") or "private"),
        "approved_for_public_export": bool(raw.get("approved_for_public_export", False)),
        "contains_customer_data": bool(raw.get("contains_customer_data", False)),
        "contains_hidden_truth": bool(raw.get("contains_hidden_truth", False)),
    }


def _now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _oel_version() -> str:
    try:
        from importlib.metadata import version

        return version("orbital-engineering-lab")
    except Exception:
        return "0.25.0"


__all__ = [
    "PATCH_OPERATION_KINDS", "SCENARIO_PATCH_ADAPTER_ID", "SCENARIO_PATCH_ADAPTER_VERSION",
    "ScenarioPatchError", "build_scenario_patch_product", "materialize_scenario_patch",
    "scenario_document", "select_patch_product", "write_scenario_patch_product",
]
