from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import yaml

from sim.interchange.scenario_patches import build_scenario_patch_product, write_scenario_patch_product

CONTROLLER_PATCH_ADAPTER_ID = "oel.controller_optimization.scenario_patch"
CONTROLLER_PATCH_ADAPTER_VERSION = "1"


class ControllerPatchError(ValueError):
    """Raised when controller optimization output cannot produce bound patches."""


def emit_controller_optimization_scenario_patches(
    result: Mapping[str, Any],
    *,
    source_suite: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    suite_path = Path(source_suite).expanduser().resolve()
    if not suite_path.is_file():
        raise ControllerPatchError("Controller optimization source suite is required.")
    optimization = deepcopy(dict(result.get("optimization", {}) or {}))
    if optimization.get("enabled") is not True:
        raise ControllerPatchError("Controller result does not contain enabled optimization evidence.")
    variants = [dict(item or {}) for item in list(result.get("variants", []) or [])]
    if len(variants) != 1:
        raise ControllerPatchError("Optimized controller result must contain exactly one best variant.")
    variant = variants[0]
    variant_name = str(variant.get("name", "") or "").strip()
    target = dict(result.get("controller_target", {}) or {})
    if not variant_name or not target.get("object_id") or not target.get("slot"):
        raise ControllerPatchError("Optimized variant and controller target must be explicit.")

    variant_document = {
        "schema_id": "oel-controller-variant-v1",
        "name": variant_name,
        "description": variant.get("description", ""),
        "controller_target": target,
        "controller": deepcopy(dict(variant.get("pointer", {}) or {})),
        "pointers": deepcopy(list(variant.get("pointers", []) or [])),
        "overrides": deepcopy(list(variant.get("overrides", []) or [])),
    }
    variant_path = output / "best_controller_variant.yaml"
    variant_path.write_text(yaml.safe_dump(variant_document, sort_keys=False), encoding="utf-8")
    evidence_path = output / "controller_optimization_patch_evidence.json"
    bounded_optimization = {
        key: deepcopy(value)
        for key, value in optimization.items()
        if key not in {"all_evaluations", "evaluations", "best_variant_yaml"}
    }
    evidence_payload = {
        "schema_id": "oel-controller-optimization-patch-evidence-v1",
        "suite_name": result.get("suite_name"),
        "controller_target": target,
        "variant": variant,
        "optimization": bounded_optimization,
        "constraint_summary": deepcopy(dict(result.get("constraint_summary", {}) or {})),
        "cases": deepcopy(list(result.get("cases", []) or [])),
        "runs": deepcopy(list(result.get("runs", []) or [])),
    }
    evidence_path.write_text(json.dumps(evidence_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    cases = [dict(item or {}) for item in list(result.get("cases", []) or [])]
    expected_case_names = [str(item.get("name", "") or "").strip() for item in cases]
    all_runs = [dict(item or {}) for item in list(result.get("runs", []) or [])]
    all_passed = (
        bool(expected_case_names)
        and all(expected_case_names)
        and all(any(str(run.get("case_name", "")) == case_name for run in all_runs) for case_name in expected_case_names)
        and all(bool(item.get("passed", False)) for item in all_runs)
    )
    entries = []
    for case_rank, case_row in enumerate(cases, start=1):
        case_name = str(case_row.get("name", "") or "").strip()
        case_path = Path(str(case_row.get("config_path", "") or "")).expanduser().resolve()
        if not case_name or not case_path.is_file():
            raise ControllerPatchError("Every optimized case requires a resolvable config_path.")
        source = yaml.safe_load(case_path.read_text(encoding="utf-8"))
        if not isinstance(source, dict):
            raise ControllerPatchError(f"Controller case {case_name!r} must contain a YAML mapping.")
        operations = _variant_operations(source, target=target, variant=variant)
        selection_id = f"{variant_name}::{case_name}"
        patch_path = output / f"controller_{_slug(variant_name)}__{_slug(case_name)}.scenario_patch.json"
        case_runs = [row for row in all_runs if str(row.get("case_name", "")) == case_name]
        case_passed = bool(case_runs) and all(bool(row.get("passed", False)) for row in case_runs)
        accepted = all_passed and case_passed
        product = build_scenario_patch_product(
            case_path,
            patch_type="controller_optimized_variant",
            selection={
                "selection_id": selection_id,
                "selection_kind": "controller_optimized_variant",
                "rank": case_rank,
                "recommended_modes": [],
                "case_name": case_name,
                "variant_name": variant_name,
            },
            operations=operations,
            evidence={
                "objective": {"case_objectives": deepcopy(list(case_row.get("objectives", []) or [])), "best_cost": optimization.get("best_cost")},
                "constraints": {"all_best_runs_passed": all_passed, "case_passed": case_passed, "constraint_summary": deepcopy(dict(result.get("constraint_summary", {}) or {}))},
                "producer_evidence": {"optimization": bounded_optimization, "case_runs": case_runs},
            },
            producer_capability_id="controller_bench_optimization",
            producer_run_id=str(result.get("suite_name", suite_path.stem)),
            source_artifacts=[suite_path, variant_path, evidence_path],
            disposition="accepted" if accepted else "review_required",
            producer_status="best_variant_rerun_passed" if accepted else "best_variant_requires_review",
            warnings=[] if accepted else ["The best-in-search variant did not pass every configured best-variant rerun."],
            non_claims=[
                "Best-in-search does not establish controller superiority outside the configured suite.",
                "Materialization does not execute or independently validate the optimized controller.",
            ],
            data_markings={
                "scope": "private_pro", "handling": "private", "approved_for_public_export": False,
                "contains_customer_data": False, "contains_hidden_truth": False,
            },
            output_path=patch_path,
        )
        entry = write_scenario_patch_product(product, patch_path)
        entry.update({"case_name": case_name, "variant_name": variant_name, "rank": case_rank, "product_path": patch_path.name})
        entries.append(entry)
    index = {
        "schema_id": "oel-scenario-patch-index-v1",
        "schema_version": 1,
        "patch_type": "controller_optimized_variant",
        "source_suite": str(suite_path),
        "selection_required": True,
        "standalone_variant_path": variant_path.name,
        "patches": entries,
    }
    index_path = output / "controller_optimized_scenario_patches.json"
    index_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "status": "emitted",
        "index_path": str(index_path),
        "standalone_variant_path": str(variant_path),
        "selection_required": True,
        "patches": entries,
    }


def _variant_operations(
    source: Mapping[str, Any], *, target: Mapping[str, Any], variant: Mapping[str, Any]
) -> list[dict[str, Any]]:
    operations = []
    object_id = str(target.get("object_id", ""))
    slot = str(target.get("slot", ""))
    operations.append(_controller_operation(source, object_id, slot, dict(variant.get("pointer", {}) or {})))
    for pointer_raw in list(variant.get("pointers", []) or []):
        pointer = dict(pointer_raw or {})
        operations.append(
            _controller_operation(
                source,
                str(pointer.get("object_id", "")),
                str(pointer.get("slot", "")),
                dict(pointer.get("pointer", pointer.get("controller", {})) or {}),
            )
        )
    for override_raw in list(variant.get("overrides", []) or []):
        override = dict(override_raw or {})
        path = str(override.get("path", "") or "").strip()
        if not _path_exists(source, path):
            raise ControllerPatchError(f"Optimized scenario override path does not exist in source case: {path}")
        operations.append(
            {
                "op": "replace", "kind": "scenario_override", "path": path,
                "value": deepcopy(override.get("value")), "reason": "Apply ordered optimized scenario override.",
            }
        )
    return operations


def _controller_operation(source: Mapping[str, Any], object_id: str, slot: str, pointer: Mapping[str, Any]) -> dict[str, Any]:
    path = f"objects.{object_id}.{slot}"
    if not object_id or slot not in {"flight_software", "base_guidance"} or not _path_exists(source, path):
        raise ControllerPatchError(f"Controller target path does not exist in source case: {path}")
    if not pointer:
        raise ControllerPatchError("Controller pointer must be non-empty.")
    return {
        "op": "replace", "kind": "flight_software_stack" if slot == "flight_software" else "controller_pointer", "path": path,
        "value": deepcopy(dict(pointer)), "reason": "Apply the exact best controller variant pointer.",
    }


def _path_exists(root: Mapping[str, Any], path: str) -> bool:
    current: Any = root
    for token in [item for item in path.split(".") if item]:
        if not isinstance(current, Mapping) or token not in current:
            return False
        current = current[token]
    return True


def _slug(value: str) -> str:
    return "".join(character if character.isalnum() or character in {"-", "_"} else "_" for character in value)


__all__ = [
    "CONTROLLER_PATCH_ADAPTER_ID", "CONTROLLER_PATCH_ADAPTER_VERSION", "ControllerPatchError",
    "emit_controller_optimization_scenario_patches",
]
