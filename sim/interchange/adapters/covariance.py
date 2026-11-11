from __future__ import annotations

import hashlib
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from sim.interchange.manifests import write_handoff_manifest
from sim.interchange.materialization import (
    _ensure_distinct_materialization_paths,
    _failed_manifest,
    _now_utc,
    _onp_compatibility_errors,
    _result,
    build_onp_scenario,
    materialize_scenario_document,
)
from sim.interchange.provenance import canonical_json_bytes, sha256_file
from sim.interchange.validation import load_interchange_document, validate_product

COVARIANCE_MATERIALIZATION_ADAPTER_ID = "oel.state_estimate_to_covariance_study"
COVARIANCE_MATERIALIZATION_ADAPTER_VERSION = "1"


class CovarianceMaterializationError(ValueError):
    """Raised when covariance-analysis inputs cannot be materialized safely."""


def materialize_covariance_study(
    state_product: str | Path,
    *,
    scenario_name: str,
    scenario_path: str | Path,
    output_dir: str | Path,
    duration_s: float,
    dt_s: float,
    position_step_km: float = 1.0e-3,
    velocity_step_km_s: float = 1.0e-6,
    process_noise_acceleration_sigma_km_s2: float = 0.0,
    manifest_path: str | Path | None = None,
    trust_plugins: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    product_path = Path(state_product).expanduser().resolve()
    destination = Path(scenario_path).expanduser().resolve()
    manifest_target = (
        Path(manifest_path).expanduser().resolve()
        if manifest_path is not None
        else destination.with_name(f"{destination.stem}.handoff_manifest.json")
    )
    _ensure_distinct_materialization_paths(
        (("source product", product_path), ("scenario", destination), ("manifest", manifest_target)),
        CovarianceMaterializationError,
    )
    if not str(scenario_name or "").strip():
        raise CovarianceMaterializationError("scenario_name must be non-empty.")
    if float(duration_s) <= 0.0 or float(dt_s) <= 0.0:
        raise CovarianceMaterializationError("duration_s and dt_s must be positive.")
    if float(position_step_km) <= 0.0 or float(velocity_step_km_s) <= 0.0:
        raise CovarianceMaterializationError("finite-difference position and velocity steps must be positive.")
    if float(process_noise_acceleration_sigma_km_s2) < 0.0:
        raise CovarianceMaterializationError("process-noise acceleration sigma must be nonnegative.")

    try:
        product = load_interchange_document(product_path)
    except ValueError as exc:
        # Keep bounded-loader failures on the materializer's structured
        # handoff surface.  In particular, strict JSON rejects NaN/Infinity
        # before a product validation report can be built; callers still need
        # a durable blocked result rather than an untyped loader escape.
        source_hash = _source_hash_or_zero(product_path)
        source_id = f"oel.invalid_source:{source_hash}"
        failure = {
            "code": "source.load_failed",
            "path": "$",
            "message": str(exc),
            "severity": "error",
        }
        created_utc = _now_utc()
        base_manifest = _manifest_base(
            created_utc=created_utc,
            source_id=source_id,
            source_hash=source_hash,
            product={},
            scenario_name=str(scenario_name).strip(),
            destination=destination,
            output_dir=output_dir,
            duration_s=float(duration_s),
            dt_s=float(dt_s),
            position_step_km=float(position_step_km),
            velocity_step_km_s=float(velocity_step_km_s),
            process_noise_acceleration_sigma_km_s2=float(process_noise_acceleration_sigma_km_s2),
            overwrite=overwrite,
        )
        manifest = _failed_manifest(
            base_manifest,
            output_status="blocked",
            failures=[failure],
            next_action="Provide a valid bounded interchange JSON state product.",
        )
        write_handoff_manifest(manifest, manifest_target)
        return _result(
            "blocked",
            destination,
            manifest_target,
            manifest,
            {
                "document_type": "unknown",
                "schema_id": "",
                "schema_version": None,
                "identifier": source_id,
                "valid": False,
                "promotable": False,
                "issue_counts": {"errors": 1, "blockers": 0, "warnings": 0},
                "issues": [failure],
            },
        )
    product_report = validate_product(product, source_path=product_path)
    source_id = str(product.get("product_id", "") or "")
    source_hash = sha256_file(product_path)
    created_utc = _now_utc()
    base_manifest = _manifest_base(
        created_utc=created_utc,
        source_id=source_id,
        source_hash=source_hash,
        product=product,
        scenario_name=str(scenario_name).strip(),
        destination=destination,
        output_dir=output_dir,
        duration_s=float(duration_s),
        dt_s=float(dt_s),
        position_step_km=float(position_step_km),
        velocity_step_km_s=float(velocity_step_km_s),
        process_noise_acceleration_sigma_km_s2=float(process_noise_acceleration_sigma_km_s2),
        overwrite=overwrite,
    )
    if not product_report.valid or not product_report.promotable:
        failures = [issue.to_dict() for issue in product_report.issues if issue.severity in {"error", "blocker"}]
        manifest = _failed_manifest(
            base_manifest,
            output_status="blocked",
            failures=failures,
            next_action="Resolve state-product validation, quality, freshness, and provenance blockers.",
        )
        write_handoff_manifest(manifest, manifest_target)
        return _result("blocked", destination, manifest_target, manifest, product_report.to_dict())

    compatibility_errors = [*_onp_compatibility_errors(product), *_covariance_compatibility_errors(product)]
    if compatibility_errors:
        manifest = _failed_manifest(
            base_manifest,
            output_status="blocked",
            failures=compatibility_errors,
            next_action="Provide a state-bound canonical ECI covariance or use a dedicated compatible adapter.",
        )
        write_handoff_manifest(manifest, manifest_target)
        return _result("blocked", destination, manifest_target, manifest, product_report.to_dict())

    payload = dict(product["payload"])
    covariance = deepcopy(dict(payload["covariance"]))
    binding = covariance_binding(product)
    scenario = build_onp_scenario(
        product,
        scenario_name=str(scenario_name).strip(),
        output_dir=output_dir,
        duration_s=float(duration_s),
        dt_s=float(dt_s),
        manifest_path=manifest_target,
        adapter_id=COVARIANCE_MATERIALIZATION_ADAPTER_ID,
        adapter_version=COVARIANCE_MATERIALIZATION_ADAPTER_VERSION,
        scenario_description=f"Covariance continuation materialized from {source_id}.",
    )
    object_id = str(dict(payload["object"])["object_id"])
    scenario["metadata"]["handoff"]["source_product_sha256"] = source_hash
    scenario["metadata"]["handoff"]["covariance_binding"] = binding
    scenario["analysis"] = {
        "enabled": True,
        "study_type": "covariance",
        "covariance": {
            "enabled": True,
            "objects": {
                object_id: {
                    "enabled": True,
                    "frame": "eci",
                    "covariance": covariance["matrix"],
                }
            },
            "pairs": [],
            "finite_difference": {
                "position_step_km": float(position_step_km),
                "velocity_step_km_s": float(velocity_step_km_s),
            },
            "process_noise": {
                "enabled": float(process_noise_acceleration_sigma_km_s2) > 0.0,
                "acceleration_sigma_km_s2": float(process_noise_acceleration_sigma_km_s2),
            },
            "write_review_tables": True,
        },
    }
    return materialize_scenario_document(
        scenario=scenario,
        destination=destination,
        manifest_target=manifest_target,
        protected_paths=(("source product", product_path),),
        collision_error_type=CovarianceMaterializationError,
        base_manifest=base_manifest,
        product_report=product_report.to_dict(),
        output_kind="covariance_study_scenario",
        trust_plugins=trust_plugins,
        overwrite=overwrite,
    )


def covariance_binding(product: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(product.get("payload", {}) or {})
    state = deepcopy(dict(payload.get("state", {}) or {}))
    covariance = deepcopy(dict(payload.get("covariance", {}) or {}))
    estimator = dict(payload.get("estimator_evidence", {}) or {})
    semantic = {"state": state, "covariance": covariance}
    digest = hashlib.sha256(canonical_json_bytes(semantic)).hexdigest()
    return {
        "binding_id": f"oel.state_covariance_binding:{digest}",
        "frame": covariance.get("frame"),
        "epoch_jd_utc": covariance.get("epoch_jd_utc"),
        "component_order": deepcopy(list(covariance.get("component_order", []) or [])),
        "units": deepcopy(list(covariance.get("units", []) or [])),
        "mathematically_valid": covariance.get("mathematically_valid"),
        "calibrated": covariance.get("calibrated"),
        "calibration_scope": covariance.get("calibration_scope") or "none",
        "source_report_sha256": estimator.get("source_report_sha256"),
    }


def _covariance_compatibility_errors(product: Mapping[str, Any]) -> list[dict[str, str]]:
    payload = dict(product.get("payload", {}) or {})
    covariance = dict(payload.get("covariance", {}) or {})
    if covariance.get("present") is not True:
        return [
            {
                "code": "covariance.absent",
                "path": "$.payload.covariance.present",
                "message": "Covariance analysis requires a covariance bound to the promoted state.",
            }
        ]
    return []


def _source_hash_or_zero(path: Path) -> str:
    try:
        return sha256_file(path)
    except (OSError, ValueError):
        return "0" * 64


def _manifest_base(
    *,
    created_utc: str,
    source_id: str,
    source_hash: str,
    product: Mapping[str, Any],
    scenario_name: str,
    destination: Path,
    output_dir: str | Path,
    duration_s: float,
    dt_s: float,
    position_step_km: float,
    velocity_step_km_s: float,
    process_noise_acceleration_sigma_km_s2: float,
    overwrite: bool,
) -> dict[str, Any]:
    markings = deepcopy(dict(product.get("data_markings", {}) or {}))
    covariance = dict(dict(product.get("payload", {}) or {}).get("covariance", {}) or {})
    warnings = []
    if covariance.get("present") is True and covariance.get("calibrated") is not True:
        warnings.append(
            "The source covariance is mathematically valid but not calibrated; downstream uncertainty claims remain bounded."
        )
    return {
        "schema_id": "oel-handoff-manifest-v1",
        "schema_version": 1,
        "manifest_id": "oel.handoff_manifest:" + "0" * 64,
        "created_utc": created_utc,
        "source_product_ids": [source_id],
        "source_hashes": {source_id: source_hash},
        "adapter": {
            "adapter_id": COVARIANCE_MATERIALIZATION_ADAPTER_ID,
            "adapter_version": COVARIANCE_MATERIALIZATION_ADAPTER_VERSION,
        },
        "materialization_options": {
            "scenario_name": scenario_name,
            "scenario_path": str(destination),
            "output_dir": str(output_dir),
            "duration_s": duration_s,
            "dt_s": dt_s,
            "position_step_km": position_step_km,
            "velocity_step_km_s": velocity_step_km_s,
            "process_noise_acceleration_sigma_km_s2": process_noise_acceleration_sigma_km_s2,
        },
        "defaults_applied": {
            "flight_software": "fsw.passive@hardware.passive.v1",
            "attitude_enabled": False,
            "covariance_frame": "eci",
            "covariance_pairs": [],
            "write_review_tables": True,
            "plots_enabled": False,
            "animations_enabled": False,
        },
        "overrides": ([{"field": "scenario_path", "reason": "explicit overwrite authorization"}] if overwrite else []),
        "source_markings": markings,
        "output_markings": markings,
        "output": {
            "kind": "covariance_study_scenario",
            "path": str(destination),
            "digest": "",
            "status": "pending",
        },
        "validation": {
            "safe_validation_result": {"status": "not_run"},
            "ordinary_validation_result": {"status": "not_run"},
        },
        "warnings": warnings,
        "failures": [],
        "recommended_next_action": "Review covariance compatibility before materialization.",
        "execution_occurred": False,
    }


__all__ = [
    "COVARIANCE_MATERIALIZATION_ADAPTER_ID",
    "COVARIANCE_MATERIALIZATION_ADAPTER_VERSION",
    "CovarianceMaterializationError",
    "covariance_binding",
    "materialize_covariance_study",
]
