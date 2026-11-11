from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from sim.interchange.manifests import write_handoff_manifest
from sim.interchange.materialization import (
    _ensure_distinct_materialization_paths,
    _failed_manifest,
    _now_utc,
    _result,
    build_onp_scenario,
    materialize_scenario_document,
)
from sim.interchange.provenance import compute_product_id, sha256_file
from sim.interchange.validation import load_interchange_document, validate_product

RELATIVE_OD_ADAPTER_ID = "oel.relative_od.relative_state_estimate"
RELATIVE_OD_ADAPTER_VERSION = "1"
RELATIVE_ONP_ADAPTER_ID = "oel.relative_state_estimate_to_onp"
RELATIVE_ONP_ADAPTER_VERSION = "1"
ORDER = ["r_radial", "i_intrack", "c_crosstrack", "vr_radial", "vi_intrack", "vc_crosstrack"]
UNITS = ["km", "km", "km", "km/s", "km/s", "km/s"]


class RelativeODInterchangeError(ValueError):
    """Raised when relative-OD evidence or chief context is incoherent."""


def build_relative_od_state_product(
    report_path: str | Path,
    chief_state_product: str | Path,
    *,
    chief_id: str,
    deputy_id: str,
    output_path: str | Path | None = None,
    deputy_specs: Mapping[str, Any] | None = None,
    data_markings: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    report_source = Path(report_path).expanduser().resolve()
    chief_source = Path(chief_state_product).expanduser().resolve()
    target = (
        Path(output_path).expanduser().resolve()
        if output_path is not None
        else report_source.with_name("relative_state_estimate_product.json")
    )
    report = _load_object(report_source)
    chief = load_interchange_document(chief_source)
    chief_validation = validate_product(chief, source_path=chief_source)
    if not chief_validation.valid or not chief_validation.promotable:
        raise RelativeODInterchangeError("Chief state product must be valid, current, and accepted.")
    if chief.get("product_kind") != "oel.state_estimate":
        raise RelativeODInterchangeError("Chief context must be an oel.state_estimate product.")
    chief_payload = dict(chief.get("payload", {}) or {})
    chief_object = dict(chief_payload.get("object", {}) or {})
    expected_chief = str(chief_id or "").strip()
    deputy = str(deputy_id or "").strip()
    if not expected_chief or str(chief_object.get("object_id", "") or "") != expected_chief:
        raise RelativeODInterchangeError("Explicit chief_id must match the chief state product object_id.")
    if not deputy or deputy == expected_chief:
        raise RelativeODInterchangeError("deputy_id must be non-empty and differ from chief_id.")
    chief_state = dict(chief_payload.get("state", {}) or {})
    epoch = float(dict(chief_state.get("epoch", {}) or {}).get("value"))
    state = _vector6(report.get("estimated_initial_state_ric"), "estimated_initial_state_ric")
    covariance = _relative_covariance(report.get("covariance_native"), epoch=epoch)
    gates = deepcopy(dict(report.get("quality_gates", {}) or {}))
    verdict = deepcopy(dict(report.get("verdict", {}) or {}))
    producer_warnings = [str(item) for item in list(gates.get("warnings", []) or [])]
    accepted = (
        gates.get("evidence_status") == "ready"
        and gates.get("solver_success") is True
        and gates.get("fit_improved_prefit_rms") is True
        and gates.get("holdout_acceptable") is True
        and gates.get("covariance_valid") is True
        and gates.get("data_full_rank") is True
        and verdict.get("usable_for_relative_od_study") is True
        and not producer_warnings
        and covariance.get("present") is True
    )
    source_paths = [report_source, chief_source]
    product: dict[str, Any] = {
        "schema_id": "oel-product-envelope-v1",
        "schema_version": 1,
        "product_kind": "oel.relative_state_estimate",
        "product_id": "oel.relative_state_estimate:" + "0" * 64,
        "created_utc": str(chief.get("created_utc")),
        "producer": {
            "capability_id": "relative_orbit_determination",
            "oel_version": str(dict(chief.get("producer", {}) or {}).get("oel_version", "unknown")),
            "run_id": report_source.parent.name,
        },
        "payload": {
            "chief": {"object_id": expected_chief, "state_product_id": chief["product_id"]},
            "deputy": {"object_id": deputy, "role": deputy, "kind": "satellite"},
            "relative_state": {
                "representation": "cartesian_relative_position_velocity",
                "frame": "RIC",
                "convention": "rectangular",
                "axes": {
                    "radial": "chief_position_outward",
                    "intrack": "chief_motion_direction",
                    "crosstrack": "right_handed_orbit_normal",
                },
                "sign_convention": "deputy_minus_chief_expressed_in_chief_ric",
                "epoch": {"value": epoch, "format": "jd", "time_system": "UTC"},
                "component_order": ORDER,
                "units": UNITS,
                "values": state,
            },
            "covariance": covariance,
            "object_specs": deepcopy(dict(deputy_specs or {})),
            "model_assumptions": {
                "relative_dynamics_model": str(report.get("dynamics_model", "") or ""),
                "dynamics_metadata": deepcopy(dict(report.get("dynamics_metadata", {}) or {})),
            },
            "estimator_evidence": {
                "method": str(report.get("method", "") or ""),
                "source_report_sha256": sha256_file(report_source),
            },
        },
        "quality": {
            "disposition": "accepted" if accepted else "review_required",
            "producer_status": str(gates.get("evidence_status", "unknown") or "unknown"),
            "gates": {
                "quality_gates": gates,
                "verdict": verdict,
                "adapter": {"adapter_id": RELATIVE_OD_ADAPTER_ID, "adapter_version": RELATIVE_OD_ADAPTER_VERSION},
            },
            "warnings": producer_warnings + ([] if accepted else ["Relative OD evidence requires review before promotion."]),
            "non_claims": [
                "Rectangular RIC is chief-referenced at the stated epoch; this product does not imply an absolute deputy OD solution.",
                "Relative covariance is not statistically calibrated unless an explicit calibration scope is supplied.",
            ],
        },
        "freshness": {
            "integrity_status": "current",
            "age_status": "not_evaluated",
            "reference_epoch_jd_utc": epoch,
            "evaluated_utc": str(chief.get("created_utc")),
            "policy": {},
        },
        "provenance": {
            "source_artifacts": [_source_artifact(path, target.parent) for path in source_paths],
            "source_product_ids": [chief["product_id"]],
            "transformations": [{
                "transformation_id": "bind_relative_od_to_chief_state_product",
                "version": "1",
                "details": {
                    "chief_id": expected_chief,
                    "deputy_id": deputy,
                    "frame": "RIC",
                    "convention": "rectangular",
                    "sign_convention": "deputy_minus_chief_expressed_in_chief_ric",
                },
            }],
        },
        "data_markings": _merge_markings(
            chief.get("data_markings"),
            data_markings,
        ),
    }
    product["product_id"] = compute_product_id(product)
    validation = validate_product(product, source_path=target)
    if not validation.valid:
        messages = "; ".join(f"{issue.path}: {issue.message}" for issue in validation.errors)
        raise RelativeODInterchangeError(f"Generated relative state product failed validation: {messages}")
    return product


def emit_relative_od_state_product(
    report_path: str | Path,
    chief_state_product: str | Path,
    *,
    chief_id: str,
    deputy_id: str,
    output_path: str | Path,
    deputy_specs: Mapping[str, Any] | None = None,
    data_markings: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    target = Path(output_path).expanduser().resolve()
    product = build_relative_od_state_product(
        report_path,
        chief_state_product,
        chief_id=chief_id,
        deputy_id=deputy_id,
        output_path=target,
        deputy_specs=deputy_specs,
        data_markings=data_markings,
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(product, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    validation = validate_product(product, source_path=target)
    return {
        "status": "emitted",
        "product_path": str(target),
        "product_id": product["product_id"],
        "disposition": product["quality"]["disposition"],
        "valid": validation.valid,
        "promotable": validation.promotable,
        "issues": [issue.to_dict() for issue in validation.issues],
    }


def materialize_relative_onp(
    relative_state_product: str | Path,
    chief_state_product: str | Path,
    *,
    scenario_name: str,
    scenario_path: str | Path,
    output_dir: str | Path,
    duration_s: float,
    dt_s: float,
    manifest_path: str | Path | None = None,
    trust_plugins: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    relative_path = Path(relative_state_product).expanduser().resolve()
    chief_path = Path(chief_state_product).expanduser().resolve()
    destination = Path(scenario_path).expanduser().resolve()
    manifest_target = (
        Path(manifest_path).expanduser().resolve()
        if manifest_path is not None
        else destination.with_name(f"{destination.stem}.handoff_manifest.json")
    )
    _ensure_distinct_materialization_paths(
        (
            ("relative source product", relative_path),
            ("chief source product", chief_path),
            ("scenario", destination),
            ("manifest", manifest_target),
        ),
        RelativeODInterchangeError,
    )
    if not str(scenario_name or "").strip() or float(duration_s) <= 0.0 or float(dt_s) <= 0.0:
        raise RelativeODInterchangeError("scenario_name must be non-empty and duration_s/dt_s must be positive.")
    relative = load_interchange_document(relative_path)
    chief = load_interchange_document(chief_path)
    relative_validation = validate_product(relative, source_path=relative_path)
    chief_validation = validate_product(chief, source_path=chief_path)
    source_ids = [str(chief.get("product_id", "")), str(relative.get("product_id", ""))]
    source_hashes = {source_ids[0]: sha256_file(chief_path), source_ids[1]: sha256_file(relative_path)}
    created_utc = _now_utc()
    markings = _merge_markings(
        chief.get("data_markings"),
        relative.get("data_markings"),
    )
    base_manifest: dict[str, Any] = {
        "schema_id": "oel-handoff-manifest-v1",
        "schema_version": 1,
        "manifest_id": "oel.handoff_manifest:" + "0" * 64,
        "created_utc": created_utc,
        "source_product_ids": source_ids,
        "source_hashes": source_hashes,
        "adapter": {"adapter_id": RELATIVE_ONP_ADAPTER_ID, "adapter_version": RELATIVE_ONP_ADAPTER_VERSION},
        "materialization_options": {
            "scenario_name": str(scenario_name), "scenario_path": str(destination), "output_dir": str(output_dir),
            "duration_s": float(duration_s), "dt_s": float(dt_s),
        },
        "defaults_applied": {
            "flight_software": "fsw.passive@hardware.passive.v1", "attitude_enabled": False, "review_detail": "standard",
            "relative_frame": "RIC", "relative_convention": "rectangular",
        },
        "overrides": ([{"field": "scenario_path", "reason": "explicit overwrite authorization"}] if overwrite else []),
        "source_markings": markings,
        "output_markings": markings,
        "output": {"kind": "relative_onp_scenario", "path": str(destination), "digest": "", "status": "pending"},
        "validation": {"safe_validation_result": {"status": "not_run"}, "ordinary_validation_result": {"status": "not_run"}},
        "warnings": [],
        "failures": [],
        "recommended_next_action": "Review both products before materialization.",
        "execution_occurred": False,
    }
    failures: list[dict[str, str]] = []
    if not relative_validation.valid or not relative_validation.promotable:
        failures.extend(issue.to_dict() for issue in relative_validation.issues if issue.severity in {"error", "blocker"})
    if not chief_validation.valid or not chief_validation.promotable:
        failures.extend(issue.to_dict() for issue in chief_validation.issues if issue.severity in {"error", "blocker"})
    failures.extend(_relative_compatibility_errors(relative, chief))
    if failures:
        manifest = _failed_manifest(
            base_manifest,
            output_status="blocked",
            failures=failures,
            next_action="Resolve chief identity, epoch, RIC convention, product quality, or provenance blockers.",
        )
        write_handoff_manifest(manifest, manifest_target)
        return _result("blocked", destination, manifest_target, manifest, relative_validation.to_dict())

    scenario = build_onp_scenario(
        chief,
        scenario_name=str(scenario_name).strip(),
        output_dir=output_dir,
        duration_s=float(duration_s),
        dt_s=float(dt_s),
        manifest_path=manifest_target,
        adapter_id=RELATIVE_ONP_ADAPTER_ID,
        adapter_version=RELATIVE_ONP_ADAPTER_VERSION,
        scenario_description=f"Passive chief/deputy continuation materialized from {relative['product_id']}.",
    )
    relative_payload = dict(relative["payload"])
    deputy = dict(relative_payload["deputy"])
    chief_id = str(dict(relative_payload["chief"])["object_id"])
    deputy_id = str(deputy["object_id"])
    values = [float(item) for item in dict(relative_payload["relative_state"])["values"]]
    scenario["objects"][deputy_id] = {
        "enabled": True,
        "role": str(deputy["role"]),
        "kind": str(deputy["kind"]),
        "specs": deepcopy(dict(relative_payload.get("object_specs", {}) or {})),
        "initial_state": {"relative_to": chief_id, "relative_ric_rect": values},
        "flight_software": {
            "stack": "fsw.passive", "hardware_profile": "hardware.passive.v1",
        },
    }
    handoff = scenario["metadata"]["handoff"]
    handoff.update({
        "source_product_ids": source_ids,
        "source_product_sha256": source_hashes,
        "relative_semantics": {
            "chief_id": chief_id,
            "deputy_id": deputy_id,
            "frame": "RIC",
            "convention": "rectangular",
            "sign_convention": "deputy_minus_chief_expressed_in_chief_ric",
        },
    })
    return materialize_scenario_document(
        scenario=scenario,
        destination=destination,
        manifest_target=manifest_target,
        protected_paths=(
            ("relative source product", relative_path),
            ("chief source product", chief_path),
        ),
        collision_error_type=RelativeODInterchangeError,
        base_manifest=base_manifest,
        product_report=relative_validation.to_dict(),
        output_kind="relative_onp_scenario",
        trust_plugins=trust_plugins,
        overwrite=overwrite,
    )


def _relative_compatibility_errors(relative: Mapping[str, Any], chief: Mapping[str, Any]) -> list[dict[str, str]]:
    errors: list[dict[str, str]] = []
    payload = dict(relative.get("payload", {}) or {})
    chief_ref = dict(payload.get("chief", {}) or {})
    state = dict(payload.get("relative_state", {}) or {})
    chief_payload = dict(chief.get("payload", {}) or {})
    chief_object = dict(chief_payload.get("object", {}) or {})
    chief_state = dict(chief_payload.get("state", {}) or {})
    if chief_ref.get("state_product_id") != chief.get("product_id"):
        errors.append({"code": "relative.chief_product_mismatch", "path": "$.payload.chief.state_product_id", "message": "Relative product references a different chief state product."})
    if chief_ref.get("object_id") != chief_object.get("object_id"):
        errors.append({"code": "relative.chief_id_mismatch", "path": "$.payload.chief.object_id", "message": "Chief object IDs do not match."})
    if dict(state.get("epoch", {}) or {}).get("value") != dict(chief_state.get("epoch", {}) or {}).get("value"):
        errors.append({"code": "relative.epoch_mismatch", "path": "$.payload.relative_state.epoch.value", "message": "Relative and chief state epochs must match exactly."})
    if state.get("frame") != "RIC" or state.get("convention") != "rectangular":
        errors.append({"code": "relative.convention_mismatch", "path": "$.payload.relative_state", "message": "Materialization requires rectangular RIC."})
    return errors


def _relative_covariance(value: Any, *, epoch: float) -> dict[str, Any]:
    if value is None:
        return {"present": False, "reason": "Relative OD producer did not emit covariance."}
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (6, 6) or not np.all(np.isfinite(matrix)):
        raise RelativeODInterchangeError("Relative covariance must be a finite 6x6 matrix.")
    if not np.allclose(matrix, matrix.T, rtol=0.0, atol=1.0e-12) or float(np.min(np.linalg.eigvalsh(matrix))) < -1.0e-12:
        raise RelativeODInterchangeError("Relative covariance must be symmetric positive semidefinite.")
    return {
        "present": True, "frame": "RIC", "convention": "rectangular", "epoch_jd_utc": epoch,
        "component_order": ORDER, "units": UNITS, "matrix": matrix.tolist(),
        "mathematically_valid": True, "calibrated": False, "calibration_scope": None,
    }


def _source_artifact(path: Path, base: Path) -> dict[str, Any]:
    return {
        "artifact_id": path.stem,
        "sha256": sha256_file(path),
        "path": os.path.relpath(path, start=base),
        "media_type": "application/json",
        "size_bytes": path.stat().st_size,
    }


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RelativeODInterchangeError(f"{path.name} must contain a JSON object.")
    return value


def _vector6(value: Any, field: str) -> list[float]:
    array = np.asarray(value, dtype=float).reshape(-1)
    if array.size != 6 or not np.all(np.isfinite(array)):
        raise RelativeODInterchangeError(f"{field} must contain six finite values.")
    return array.tolist()


def _markings(value: Mapping[str, Any] | None) -> dict[str, Any]:
    raw = dict(value or {})
    return {
        "scope": str(raw.get("scope", "private_pro") or "private_pro"),
        "handling": str(raw.get("handling", "private") or "private"),
        "approved_for_public_export": bool(raw.get("approved_for_public_export", False)),
        "contains_customer_data": bool(raw.get("contains_customer_data", False)),
        "contains_hidden_truth": bool(raw.get("contains_hidden_truth", False)),
        **(
            {"source_markings": deepcopy(list(raw["source_markings"]))}
            if "source_markings" in raw
            else {}
        ),
    }


def _merge_markings(*values: Mapping[str, Any] | None) -> dict[str, Any]:
    """Combine markings without allowing any source restriction to be downgraded."""

    normalized = [_markings(value) for value in values if value is not None]
    if not normalized:
        normalized = [_markings(None)]
    scope_rank = {
        "public": 0,
        "private_pro": 1,
        "customer_restricted": 2,
        "governed_restricted": 3,
    }
    most_restrictive = max(
        normalized,
        key=lambda row: scope_rank.get(str(row.get("scope")), 3),
    )
    return {
        "scope": str(most_restrictive["scope"]),
        "handling": str(most_restrictive["handling"]),
        "approved_for_public_export": all(
            bool(row["approved_for_public_export"]) for row in normalized
        ),
        "contains_customer_data": any(
            bool(row["contains_customer_data"]) for row in normalized
        ),
        "contains_hidden_truth": any(
            bool(row["contains_hidden_truth"]) for row in normalized
        ),
        "source_markings": deepcopy(normalized),
    }


__all__ = [
    "RELATIVE_OD_ADAPTER_ID",
    "RELATIVE_OD_ADAPTER_VERSION",
    "RELATIVE_ONP_ADAPTER_ID",
    "RELATIVE_ONP_ADAPTER_VERSION",
    "RelativeODInterchangeError",
    "build_relative_od_state_product",
    "emit_relative_od_state_product",
    "materialize_relative_onp",
]
