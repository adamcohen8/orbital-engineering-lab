from __future__ import annotations

import json
import os
from copy import deepcopy
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from sim.data_io.common import read_text, strict_json
from sim.data_io.contracts import DataIOError
from sim.interchange.provenance import compute_product_id, sha256_file
from sim.interchange.validation import validate_product
from sim.security.output_writes import current_output_writer, write_output_text

BATCH_OD_ADAPTER_ID = "oel.batch_od.state_estimate"
BATCH_OD_ADAPTER_VERSION = "1"
STATE_PARAMETER_ORDER = ("dx_m", "dy_m", "dz_m", "dvx_mm_s", "dvy_mm_s", "dvz_mm_s")


class BatchODInterchangeError(ValueError):
    """Raised when a batch-OD artifact set cannot produce a coherent product."""


def build_batch_od_state_product(
    artifact_dir: str | Path,
    *,
    output_path: str | Path | None = None,
    data_markings: Mapping[str, Any] | None = None,
    writer: Any | None = None,
) -> dict[str, Any]:
    logical_root = _lexical_absolute(artifact_dir)
    root = _writer_physical_path(logical_root, writer)
    report_path = root / "od_fit_report.json"
    packet_path = root / "fitted_mission_input_packet.json"
    parameters_path = root / "estimated_parameters.json"
    required = (report_path, packet_path, parameters_path)
    missing = [path.name for path in required if not path.is_file()]
    if missing:
        raise BatchODInterchangeError(f"Batch OD artifact set is missing: {', '.join(missing)}.")

    report = _load_object(report_path)
    packet = _load_object(packet_path)
    estimated = _load_object(parameters_path)
    _verify_report_references(
        report,
        {
            "report_json_path": report_path,
            "fitted_mission_input_packet_path": packet_path,
            "estimated_parameters_path": parameters_path,
        },
        path_mapper=lambda path: _writer_physical_path(path, writer),
    )
    object_id, obj = _coherent_object(report, packet)
    state, epoch = _state_payload(obj)
    force_model = _coherent_force_model(report)
    covariance = _state_covariance(estimated, epoch=epoch)
    disposition, adapter_warnings = _quality_disposition(report)
    attitude = {
        "source": str(report.get("attitude_source", "none") or "none"),
        "mode": str(report.get("attitude_mode", "") or ""),
        "body_axis": deepcopy(list(report.get("attitude_body_axis", []) or [])),
        "sample_count": int(report.get("attitude_history_sample_count", 0) or 0),
    }
    if attitude["source"] != "none" and disposition == "accepted":
        disposition = "review_required"
        adapter_warnings.append(
            "Attitude-dependent OD evidence requires a dedicated replay/model adapter before ONP materialization."
        )

    logical_target = (
        _lexical_absolute(output_path)
        if output_path is not None
        else logical_root / "state_estimate_product.json"
    )
    target = _writer_physical_path(logical_target, writer)
    source_paths = [report_path, packet_path, parameters_path]
    for optional_name in ("od_solution.json", "oel_workflow_evidence.json"):
        optional = root / optional_name
        if optional.is_file():
            source_paths.append(optional)
    source_artifacts = [
        {
            "artifact_id": path.stem,
            "sha256": sha256_file(path),
            "path": os.path.relpath(path, start=target.parent),
            "media_type": "application/json",
            "size_bytes": path.stat().st_size,
        }
        for path in source_paths
    ]
    report_hash = sha256_file(report_path)
    markings = _normalize_data_markings(data_markings or report.get("data_markings"))
    created_utc = _normalize_utc_timestamp(packet.get("created_utc"))
    quality_gates = deepcopy(dict(report.get("quality_gates", {}) or {}))
    producer_warnings = [str(item) for item in list(quality_gates.get("warnings", []) or [])]
    product = {
        "schema_id": "oel-product-envelope-v1",
        "schema_version": 1,
        "product_kind": "oel.state_estimate",
        "product_id": "oel.state_estimate:" + "0" * 64,
        "created_utc": created_utc,
        "producer": {
            "capability_id": "dynamics_orbit_determination",
            "oel_version": _oel_version(),
            "run_id": str(report.get("scenario_name", root.name) or root.name),
        },
        "payload": {
            "object": {
                "object_id": object_id,
                "role": str(obj.get("role", object_id) or object_id),
                "kind": str(obj.get("kind", "") or ""),
            },
            "state": state,
            "covariance": covariance,
            "object_specs": deepcopy(dict(obj.get("specs", {}) or {})),
            "model_assumptions": {
                "orbit_force_model": force_model,
                "attitude": attitude,
            },
            "estimator_evidence": {
                "method": str(report.get("method", "") or ""),
                "selected_parameters": [str(item) for item in list(report.get("selected_parameters", []) or [])],
                "od_solution_id": _optional_od_solution_id(root / "od_solution.json"),
                "source_report_sha256": report_hash,
            },
        },
        "quality": {
            "disposition": disposition,
            "producer_status": str(quality_gates.get("evidence_status", "unknown") or "unknown"),
            "gates": {
                "quality_gates": quality_gates,
                "verdict": deepcopy(dict(report.get("verdict", {}) or {})),
                "adapter": {
                    "adapter_id": BATCH_OD_ADAPTER_ID,
                    "adapter_version": BATCH_OD_ADAPTER_VERSION,
                },
            },
            "warnings": [*producer_warnings, *adapter_warnings],
            "non_claims": [str(item) for item in list(quality_gates.get("non_claims", []) or [])],
        },
        "freshness": {
            "integrity_status": "current",
            "age_status": "not_evaluated",
            "reference_epoch_jd_utc": epoch,
            "evaluated_utc": created_utc,
            "policy": {},
        },
        "provenance": {
            "source_artifacts": source_artifacts,
            "source_product_ids": [],
            "transformations": [
                {
                    "transformation_id": "native_parameter_covariance_to_cartesian_state_covariance",
                    "version": "1",
                    "details": {
                        "source_order": list(STATE_PARAMETER_ORDER),
                        "target_order": ["x", "y", "z", "vx", "vy", "vz"],
                        "diagonal_scale": [1.0e-3, 1.0e-3, 1.0e-3, 1.0e-6, 1.0e-6, 1.0e-6],
                    },
                }
            ],
        },
        "data_markings": markings,
    }
    product["product_id"] = compute_product_id(product)
    report_validation = validate_product(product, source_path=target)
    if not report_validation.valid:
        messages = "; ".join(f"{issue.path}: {issue.message}" for issue in report_validation.errors)
        raise BatchODInterchangeError(f"Generated state product failed validation: {messages}")
    return product


def emit_batch_od_state_product(
    artifact_dir: str | Path,
    *,
    output_path: str | Path | None = None,
    data_markings: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    writer = current_output_writer()
    root = _lexical_absolute(artifact_dir)
    target = (
        _lexical_absolute(output_path)
        if output_path is not None
        else root / "state_estimate_product.json"
    )
    product = build_batch_od_state_product(
        root,
        output_path=target,
        data_markings=data_markings,
        writer=writer,
    )
    serialized = json.dumps(product, indent=2, sort_keys=True) + "\n"
    if writer is not None:
        write_output_text(target, serialized)
        physical_target = _writer_physical_path(target, writer)
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(serialized, encoding="utf-8")
        physical_target = target
    report = validate_product(product, source_path=physical_target)
    return {
        "status": "emitted",
        "product_path": str(target),
        "product_id": product["product_id"],
        "disposition": product["quality"]["disposition"],
        "valid": report.valid,
        "promotable": report.promotable,
        "issues": [issue.to_dict() for issue in report.issues],
    }


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = strict_json(read_text(path))
    except (FileNotFoundError, DataIOError) as exc:
        raise BatchODInterchangeError(f"{path.name} is not valid bounded JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise BatchODInterchangeError(f"{path.name} must contain a JSON object.")
    return value


def _verify_report_references(
    report: Mapping[str, Any],
    expected: Mapping[str, Path],
    *,
    path_mapper: Any | None = None,
) -> None:
    for field, supplied in expected.items():
        raw = report.get(field)
        if not raw:
            continue
        referenced = Path(str(raw)).expanduser()
        if referenced.is_absolute() and path_mapper is not None:
            referenced = Path(path_mapper(referenced))
        elif not referenced.is_absolute():
            referenced = supplied.parent / referenced
        if not referenced.is_file():
            raise BatchODInterchangeError(f"Report reference {field} does not resolve: {referenced}.")
        if sha256_file(referenced) != sha256_file(supplied):
            raise BatchODInterchangeError(f"Report reference {field} does not match supplied {supplied.name}.")


def _writer_physical_path(path: str | Path, writer: Any | None) -> Path:
    candidate = _lexical_absolute(path)
    if writer is None:
        return candidate
    logical_root = _lexical_absolute(writer.logical_root)
    physical_root = _lexical_absolute(writer.root)
    try:
        return physical_root / candidate.relative_to(logical_root)
    except ValueError:
        return candidate


def _lexical_absolute(path: str | Path) -> Path:
    return Path(os.path.abspath(os.fspath(Path(path).expanduser())))


def _coherent_object(report: Mapping[str, Any], packet: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    if int(packet.get("packet_version", 0) or 0) != 1 or packet.get("kind") != "oel.mission_input_packet":
        raise BatchODInterchangeError("Unsupported fitted mission-input packet kind or version.")
    object_id = str(report.get("object_id", "") or "")
    objects = packet.get("objects")
    if not object_id or not isinstance(objects, Mapping) or set(objects) != {object_id}:
        raise BatchODInterchangeError("Report object_id and fitted packet object set must match exactly.")
    obj = dict(objects[object_id] or {})
    if str(obj.get("object_id", object_id) or object_id) != object_id:
        raise BatchODInterchangeError("Fitted packet object_id conflicts with the OD report.")
    if str(obj.get("kind", "") or "") != "satellite":
        raise BatchODInterchangeError("The first batch OD adapter supports satellite state products only.")
    if str(obj.get("frame", "") or "") != "ECI":
        raise BatchODInterchangeError("The first batch OD adapter requires explicit canonical ECI state evidence.")
    return object_id, obj


def _state_payload(obj: Mapping[str, Any]) -> tuple[dict[str, Any], float]:
    initial = dict(obj.get("initial_state", {}) or {})
    position = _finite_vector(initial.get("position_eci_km"), 3, "position_eci_km")
    velocity = _finite_vector(initial.get("velocity_eci_km_s"), 3, "velocity_eci_km_s")
    epoch = _finite_float(initial.get("epoch_jd_utc"), "epoch_jd_utc")
    if epoch <= 0.0:
        raise BatchODInterchangeError("epoch_jd_utc must be positive.")
    return (
        {
            "representation": "cartesian_position_velocity",
            "frame": "ECI",
            "epoch": {"value": epoch, "format": "jd", "time_system": "UTC"},
            "component_order": ["x", "y", "z", "vx", "vy", "vz"],
            "units": ["km", "km", "km", "km/s", "km/s", "km/s"],
            "values": [*position, *velocity],
        },
        epoch,
    )


def _coherent_force_model(report: Mapping[str, Any]) -> dict[str, Any]:
    force_model = deepcopy(dict(report.get("orbit_force_model", {}) or {}))
    if not force_model:
        force_model = {
            "model": str(report.get("dynamics_model", "two_body") or "two_body"),
            "j2": bool(report.get("j2", False)),
            "drag": bool(report.get("drag", False)),
            "srp": bool(report.get("srp", False)),
        }
    top_atmosphere = str(report.get("atmosphere_model", "") or "").strip().lower()
    force_atmosphere = str(force_model.get("atmosphere_model", "") or "").strip().lower()
    if top_atmosphere and force_atmosphere and top_atmosphere != force_atmosphere:
        raise BatchODInterchangeError("OD report atmosphere-model fields conflict.")
    if top_atmosphere:
        force_model["atmosphere_model"] = top_atmosphere
    if str(force_model.get("model", "") or "") != str(report.get("dynamics_model", "") or ""):
        raise BatchODInterchangeError("OD report dynamics model conflicts with orbit_force_model.model.")
    return force_model


def _state_covariance(estimated: Mapping[str, Any], *, epoch: float) -> dict[str, Any]:
    parameters = list(estimated.get("parameters", []) or [])
    names = [str(item.get("name", "") or "") for item in parameters if isinstance(item, Mapping)]
    if names[:6] != list(STATE_PARAMETER_ORDER):
        raise BatchODInterchangeError(
            "Estimated parameter metadata must begin with dx_m, dy_m, dz_m, dvx_mm_s, dvy_mm_s, dvz_mm_s."
        )
    native = estimated.get("covariance")
    if native is None:
        return {"present": False, "reason": "The producer did not emit native parameter covariance."}
    matrix = np.asarray(native, dtype=float)
    if matrix.shape != (len(names), len(names)) or not np.all(np.isfinite(matrix)):
        raise BatchODInterchangeError("Native parameter covariance dimensions or values are invalid.")
    block = matrix[:6, :6]
    scale = np.diag([1.0e-3, 1.0e-3, 1.0e-3, 1.0e-6, 1.0e-6, 1.0e-6])
    converted = scale @ block @ scale.T
    symmetric = np.allclose(converted, converted.T, rtol=0.0, atol=1.0e-12)
    psd = symmetric and float(np.min(np.linalg.eigvalsh(0.5 * (converted + converted.T)))) >= -1.0e-12
    if not psd:
        raise BatchODInterchangeError("Converted Cartesian state covariance is not symmetric positive semidefinite.")
    return {
        "present": True,
        "frame": "ECI",
        "epoch_jd_utc": epoch,
        "component_order": ["x", "y", "z", "vx", "vy", "vz"],
        "units": ["km", "km", "km", "km/s", "km/s", "km/s"],
        "matrix": converted.tolist(),
        "mathematically_valid": True,
        "calibrated": False,
        "calibration_scope": None,
    }


def _quality_disposition(report: Mapping[str, Any]) -> tuple[str, list[str]]:
    gates = dict(report.get("quality_gates", {}) or {})
    verdict = dict(report.get("verdict", {}) or {})
    warnings = [str(item) for item in list(gates.get("warnings", []) or [])]
    accepted = (
        gates.get("evidence_status") == "ready_with_caveats"
        and verdict.get("analyst_action") == "usable_for_propagation_study"
        and gates.get("solver_success") is True
        and gates.get("data_full_rank") is True
        and gates.get("fit_improved_prefit_rms") is True
        and gates.get("holdout_acceptable") is True
        and gates.get("covariance_valid") is True
        and not list(gates.get("parameter_bounds_hit", []) or [])
        and not warnings
    )
    if accepted:
        return "accepted", []
    explicit = str(verdict.get("analyst_action", "") or "").lower()
    if "reject" in explicit or str(gates.get("evidence_status", "") or "").lower() == "rejected":
        return "rejected", []
    if bool(report.get("multiple_solutions_unresolved", False)):
        return "ambiguous", []
    if gates or verdict:
        return "review_required", []
    return "unknown", ["Batch OD quality evidence was missing or unrecognized."]


def _normalize_data_markings(value: Mapping[str, Any] | None) -> dict[str, Any]:
    raw = dict(value or {})
    return {
        "scope": str(raw.get("scope", "private_pro") or "private_pro"),
        "handling": str(raw.get("handling", "private") or "private"),
        "approved_for_public_export": bool(raw.get("approved_for_public_export", False)),
        "contains_customer_data": bool(raw.get("contains_customer_data", False)),
        "contains_hidden_truth": bool(raw.get("contains_hidden_truth", False)),
        **({"source_markings": deepcopy(list(raw["source_markings"]))} if "source_markings" in raw else {}),
    }


def _normalize_utc_timestamp(value: Any) -> str:
    text = str(value or "").strip()
    if text:
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError as exc:
            raise BatchODInterchangeError("Mission-input packet created_utc is invalid.") from exc
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    raise BatchODInterchangeError("Mission-input packet created_utc is required for deterministic product emission.")


def _oel_version() -> str:
    try:
        return version("orbital-engineering-lab")
    except PackageNotFoundError:
        return "0.25.0"


def _optional_od_solution_id(path: Path) -> str | None:
    if not path.is_file():
        return None
    value = _load_object(path)
    raw = value.get("solution_id") or value.get("product_id")
    return None if not raw else str(raw)


def _finite_vector(value: Any, size: int, field: str) -> list[float]:
    try:
        array = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise BatchODInterchangeError(f"{field} must contain {size} finite values.") from exc
    if array.size != size or not np.all(np.isfinite(array)):
        raise BatchODInterchangeError(f"{field} must contain {size} finite values.")
    return array.tolist()


def _finite_float(value: Any, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise BatchODInterchangeError(f"{field} must be finite and explicit.") from exc
    if not np.isfinite(result):
        raise BatchODInterchangeError(f"{field} must be finite and explicit.")
    return result


__all__ = [
    "BATCH_OD_ADAPTER_ID",
    "BATCH_OD_ADAPTER_VERSION",
    "BatchODInterchangeError",
    "build_batch_od_state_product",
    "emit_batch_od_state_product",
]
