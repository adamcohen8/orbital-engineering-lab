from __future__ import annotations

import hashlib
import json
import math
import os
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from sim.interchange.provenance import canonical_json_bytes, compute_product_id, sha256_file
from sim.interchange.validation import validate_product
from sim.review import ReviewWorkspace

COMPLETED_RUN_STATE_ADAPTER_ID = "oel.completed_run.state_export"
COMPLETED_RUN_STATE_ADAPTER_VERSION = "2"
COMPLETED_RUN_SELECTOR_KINDS = frozenset({"final", "sample_index", "time_s", "event"})

_STATE_COLUMNS = (
    "sample_index",
    "time_s",
    "object_id",
    "pos_x_eci_km",
    "pos_y_eci_km",
    "pos_z_eci_km",
    "vel_x_eci_km_s",
    "vel_y_eci_km_s",
    "vel_z_eci_km_s",
    "mass_kg",
)
_KINEMATIC_COLUMNS = _STATE_COLUMNS[3:9]
_STATE_COMPONENT_ORDER = ["x", "y", "z", "vx", "vy", "vz"]
_STATE_UNITS = ["km", "km", "km", "km/s", "km/s", "km/s"]


class CompletedRunStateExportError(ValueError):
    """Raised when completed-run evidence cannot identify one promotable state."""


def build_completed_run_state_product(
    completed_run: str | Path,
    *,
    output_path: str | Path,
    object_id: str | None = None,
    selector: str = "final",
    sample_index: int | None = None,
    time_s: float | None = None,
    event_id: str | None = None,
    epoch_jd_utc: float | None = None,
    data_markings: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one state product from an exact, read-only completed-run selection."""

    target = Path(output_path).expanduser().resolve()
    workspace = ReviewWorkspace.open(completed_run)
    db_path = workspace.db_path
    starting_db_hash = sha256_file(db_path)
    _require_review_contract(workspace)
    metadata = _one_row(
        workspace.query(
            "SELECT run_id, scenario_name, oel_version, review_schema_version, generated_utc, "
            "config_path, config_sha256, config_json FROM run_metadata",
            max_rows=2,
        ),
        "run_metadata",
    )
    config = _verified_config(metadata)
    initial_jd_utc, epoch_source = _initial_jd_utc(config, override=epoch_jd_utc)

    selector_kind = str(selector or "").strip().lower()
    _validate_selector_arguments(
        selector_kind,
        sample_index=sample_index,
        time_s=time_s,
        event_id=event_id,
    )
    event = _selected_event(workspace, event_id) if selector_kind == "event" else None
    resolved_object_id = _resolve_object_id(workspace, requested=object_id, event=event)
    object_row = _selected_object(workspace, resolved_object_id)
    object_config = _object_config(config, resolved_object_id)
    frame = _state_frame(workspace, resolved_object_id)
    if frame != "ECI":
        raise CompletedRunStateExportError(
            f"Completed-run continuation requires canonical ECI object_state evidence; "
            f"{resolved_object_id!r} is recorded as {frame!r}."
        )
    state_row = _select_state_row(
        workspace,
        object_id=resolved_object_id,
        selector=selector_kind,
        sample_index=sample_index,
        time_s=time_s,
        event=event,
    )
    selected_sample_index = int(state_row["sample_index"])
    selected_time_s = _finite_float(state_row["time_s"], "selected time_s")
    if event is not None:
        _verify_event_association(event, state_row)
    epoch_jd_utc = initial_jd_utc + selected_time_s / 86400.0
    state_values = [_finite_float(state_row[name], name) for name in _KINEMATIC_COLUMNS]
    resource_state = _resource_state(state_row, object_config)
    covariance = _matching_covariance(
        workspace,
        object_id=resolved_object_id,
        sample_index=selected_sample_index,
        time_s=selected_time_s,
        epoch_jd_utc=epoch_jd_utc,
    )
    db_hash_after_read = sha256_file(db_path)
    if db_hash_after_read != starting_db_hash:
        raise CompletedRunStateExportError("Review store changed while the selected sample was being exported.")

    source_artifacts = [_source_artifact(db_path, target=target, artifact_id="completed_run_review_store")]
    if workspace.schema_path.is_file():
        source_artifacts.append(
            _source_artifact(workspace.schema_path, target=target, artifact_id="completed_run_review_schema")
        )
    created_utc = _normalize_utc(
        metadata["generated_utc"],
        fallback=_schema_generated_utc(workspace.schema_path),
    )
    selection = {
        "selector_kind": selector_kind,
        "requested": _selector_request(
            selector_kind,
            sample_index=sample_index,
            time_s=time_s,
            event_id=event_id,
        ),
        "sample_index": selected_sample_index,
        "time_s": selected_time_s,
        "state_row_sha256": hashlib.sha256(canonical_json_bytes(state_row)).hexdigest(),
        "associated_event": deepcopy(event),
    }
    orbit_force_model, source_attitude_enabled = _model_assumptions(config)
    warnings = []
    if covariance.get("present") is not True:
        warnings.append("No matching full 6x6 ECI covariance row was available for the selected sample.")
    if source_attitude_enabled:
        warnings.append("Source-run attitude state is not included in this orbital continuation product.")
    product: dict[str, Any] = {
        "schema_id": "oel-product-envelope-v1",
        "schema_version": 1,
        "product_kind": "oel.completed_run_state",
        "product_id": "oel.completed_run_state:" + "0" * 64,
        "created_utc": created_utc,
        "producer": {
            "capability_id": "completed_run_state_export",
            "oel_version": str(metadata["oel_version"] or "unknown"),
            "run_id": str(metadata["run_id"]),
        },
        "payload": {
            "object": {
                "object_id": resolved_object_id,
                "role": str(object_config.get("role", object_row.get("role", resolved_object_id)) or resolved_object_id),
                "kind": "satellite",
            },
            "state": {
                "representation": "cartesian_position_velocity",
                "frame": "ECI",
                "epoch": {"value": epoch_jd_utc, "format": "jd", "time_system": "UTC"},
                "component_order": list(_STATE_COMPONENT_ORDER),
                "units": list(_STATE_UNITS),
                "values": state_values,
            },
            "covariance": covariance,
            "object_specs": deepcopy(dict(object_config.get("specs", {}) or {})),
            "resource_state": resource_state,
            "model_assumptions": {
                "orbit_force_model": orbit_force_model,
                "attitude": {
                    "source": "none",
                    "mode": "orbital_state_only",
                    "sample_count": 0,
                },
            },
            "source_run": {
                "run_id": str(metadata["run_id"]),
                "scenario_name": str(metadata["scenario_name"]),
                "review_schema_version": str(metadata["review_schema_version"]),
                "generated_utc": created_utc,
                "config_sha256": str(metadata["config_sha256"]),
                "review_db_sha256": starting_db_hash,
                "initial_jd_utc": initial_jd_utc,
                "initial_jd_utc_source": epoch_source,
            },
            "selection": selection,
        },
        "quality": {
            "disposition": "accepted",
            "producer_status": "completed_run_sample_selected",
            "gates": {
                "review_store_hash_stable": True,
                "config_hash_verified": True,
                "sample_unambiguous": True,
                "frame_supported": True,
                "absolute_epoch_anchored": True,
                "covariance_status": "matched" if covariance.get("present") is True else "not_available",
                "resource_state_bound": True,
                "adapter": {
                    "adapter_id": COMPLETED_RUN_STATE_ADAPTER_ID,
                    "adapter_version": COMPLETED_RUN_STATE_ADAPTER_VERSION,
                },
            },
            "warnings": warnings,
            "non_claims": [
                "The product contains one selected simulator truth state, not a new orbit determination result.",
                "No propagation or scenario execution occurred during state export.",
                "Attitude, controller memory, estimator memory, and mission-module state are not continued.",
            ],
        },
        "freshness": {
            "integrity_status": "current",
            "age_status": "not_applicable",
            "reference_epoch_jd_utc": epoch_jd_utc,
            "evaluated_utc": created_utc,
            "policy": {"assessment": "content-bound_completed_run_sample"},
        },
        "provenance": {
            "source_artifacts": source_artifacts,
            "source_product_ids": [],
            "transformations": [
                {
                    "transformation_id": "review_store_object_state_to_completed_run_state",
                    "version": "1",
                    "details": {
                        "selector_kind": selector_kind,
                        "sample_index": selected_sample_index,
                        "time_s": selected_time_s,
                        "epoch_derivation": "initial_jd_utc + time_s / 86400",
                        "initial_jd_utc_source": epoch_source,
                        "frame_transform_applied": False,
                        "mass_and_propellant_bound": True,
                    },
                }
            ],
        },
        "data_markings": _data_markings(config, override=data_markings),
    }
    product["product_id"] = compute_product_id(product)
    report = validate_product(product, source_path=target)
    if not report.valid:
        messages = "; ".join(f"{issue.path}: {issue.message}" for issue in report.errors)
        raise CompletedRunStateExportError(f"Generated completed-run state product failed validation: {messages}")
    return product


def _resource_state(state_row: Mapping[str, Any], object_config: Mapping[str, Any]) -> dict[str, Any]:
    mass_kg = _finite_float(state_row.get("mass_kg"), "mass_kg")
    if mass_kg <= 0.0:
        raise CompletedRunStateExportError("Selected object-state mass_kg must be positive.")
    specs = dict(object_config.get("specs", {}) or {})
    dry_raw = specs.get("dry_mass_kg")
    if dry_raw is None:
        return {
            "mass_kg": mass_kg,
            "dry_mass_kg": None,
            "fuel_mass_kg": None,
            "propellant_state": "mass_only",
        }
    dry_mass_kg = _finite_float(dry_raw, "object specs dry_mass_kg")
    if dry_mass_kg < 0.0 or dry_mass_kg > mass_kg + 1.0e-9:
        raise CompletedRunStateExportError(
            "Selected mass is incompatible with the configured dry_mass_kg; propellant continuity cannot be represented."
        )
    return {
        "mass_kg": mass_kg,
        "dry_mass_kg": dry_mass_kg,
        "fuel_mass_kg": max(mass_kg - dry_mass_kg, 0.0),
        "propellant_state": "tracked",
    }


def export_completed_run_state(
    completed_run: str | Path,
    *,
    output_path: str | Path,
    object_id: str | None = None,
    selector: str = "final",
    sample_index: int | None = None,
    time_s: float | None = None,
    event_id: str | None = None,
    epoch_jd_utc: float | None = None,
    data_markings: Mapping[str, Any] | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Write a deterministic state product; never mutate or execute the source run."""

    target = Path(output_path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    product = build_completed_run_state_product(
        completed_run,
        output_path=target,
        object_id=object_id,
        selector=selector,
        sample_index=sample_index,
        time_s=time_s,
        event_id=event_id,
        epoch_jd_utc=epoch_jd_utc,
        data_markings=data_markings,
    )
    text = json.dumps(product, indent=2, sort_keys=True) + "\n"
    if target.exists():
        if target.read_text(encoding="utf-8") != text and not overwrite:
            raise CompletedRunStateExportError(
                "Output product exists with different content; pass overwrite=True explicitly to replace it."
            )
    if not target.exists() or target.read_text(encoding="utf-8") != text:
        target.write_text(text, encoding="utf-8")
    report = validate_product(product, source_path=target)
    return {
        "status": "exported",
        "product_path": str(target),
        "product_id": product["product_id"],
        "object_id": product["payload"]["object"]["object_id"],
        "selection": deepcopy(product["payload"]["selection"]),
        "epoch_jd_utc": product["payload"]["state"]["epoch"]["value"],
        "covariance_present": product["payload"]["covariance"]["present"],
        "valid": report.valid,
        "promotable": report.promotable,
        "execution_occurred": False,
        "issues": [issue.to_dict() for issue in report.issues],
    }


def _require_review_contract(workspace: ReviewWorkspace) -> None:
    columns = workspace.table_columns()
    if "planetary_environment" in columns:
        raise CompletedRunStateExportError("Planetary states cannot be exported through the Earth ECI v1 state-product contract.")
    required = {
        "run_metadata": {
            "run_id",
            "scenario_name",
            "oel_version",
            "review_schema_version",
            "generated_utc",
            "config_path",
            "config_sha256",
            "config_json",
        },
        "objects": {"object_id", "object_type", "enabled", "role"},
        "object_state": set(_STATE_COLUMNS),
        "object_state_frame": {"object_id", "state_frame"},
        "events": {"event_id", "time_s", "sample_index", "object_id", "event_type", "severity", "message", "source"},
    }
    for table, expected in required.items():
        actual = {str(item["name"]) for item in columns.get(table, [])}
        missing = sorted(expected - actual)
        if missing:
            raise CompletedRunStateExportError(
                f"Review store does not satisfy the completed-run export contract: {table} is missing {missing}."
            )


def _verified_config(metadata: Mapping[str, Any]) -> dict[str, Any]:
    text = str(metadata.get("config_json", "") or "")
    declared = str(metadata.get("config_sha256", "") or "")
    actual = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if not text or declared != actual:
        raise CompletedRunStateExportError("run_metadata config_json does not match config_sha256.")
    try:
        config = json.loads(text)
    except json.JSONDecodeError as exc:
        raise CompletedRunStateExportError("run_metadata config_json is invalid JSON.") from exc
    if not isinstance(config, dict):
        raise CompletedRunStateExportError("run_metadata config_json must contain an object.")
    return config


def _initial_jd_utc(
    config: Mapping[str, Any], *, override: float | None
) -> tuple[float, str]:
    value = dict(config.get("simulator", {}) or {}).get("initial_jd_utc")
    configured: float | None = None
    try:
        configured = _finite_float(value, "simulator.initial_jd_utc")
    except CompletedRunStateExportError:
        configured = None
    explicit: float | None = None
    if override is not None:
        explicit = _finite_float(override, "epoch_jd_utc override")
        if explicit <= 0.0:
            raise CompletedRunStateExportError("epoch_jd_utc override must be positive.")
    if configured is not None and configured > 0.0:
        if explicit is not None and not math.isclose(configured, explicit, rel_tol=0.0, abs_tol=1.0e-12):
            raise CompletedRunStateExportError(
                "epoch_jd_utc override conflicts with simulator.initial_jd_utc in the verified source config."
            )
        return configured, "verified_source_config"
    if explicit is not None:
        return explicit, "explicit_export_override"
    raise CompletedRunStateExportError(
        "simulator.initial_jd_utc must be finite and positive, or epoch_jd_utc must be supplied explicitly."
    )


def _validate_selector_arguments(
    selector: str,
    *,
    sample_index: int | None,
    time_s: float | None,
    event_id: str | None,
) -> None:
    if selector not in COMPLETED_RUN_SELECTOR_KINDS:
        raise CompletedRunStateExportError(
            f"Unsupported completed-run selector {selector!r}; expected one of {sorted(COMPLETED_RUN_SELECTOR_KINDS)}."
        )
    supplied = {
        "sample_index": sample_index is not None,
        "time_s": time_s is not None,
        "event": bool(str(event_id or "").strip()),
    }
    expected = {"final": None, "sample_index": "sample_index", "time_s": "time_s", "event": "event"}[selector]
    if expected is None and any(supplied.values()):
        raise CompletedRunStateExportError("The final selector does not accept sample_index, time_s, or event_id.")
    if expected is not None and not supplied[expected]:
        raise CompletedRunStateExportError(f"The {selector} selector requires {expected}.")
    if expected is not None and sum(bool(value) for value in supplied.values()) != 1:
        raise CompletedRunStateExportError("Exactly one selector value may be supplied.")
    if sample_index is not None and (isinstance(sample_index, bool) or int(sample_index) < 0):
        raise CompletedRunStateExportError("sample_index must be a non-negative integer.")
    if time_s is not None and _finite_float(time_s, "time_s") < 0.0:
        raise CompletedRunStateExportError("time_s must be non-negative.")


def _selected_event(workspace: ReviewWorkspace, event_id: str | None) -> dict[str, Any]:
    event_key = str(event_id or "").strip()
    result = workspace.query(
        "SELECT event_id, time_s, sample_index, object_id, event_type, severity, message, source "
        "FROM events WHERE event_id = ?",
        (event_key,),
        max_rows=2,
    )
    row = _one_row(result, f"event_id {event_key!r}")
    if row.get("sample_index") is None:
        raise CompletedRunStateExportError(f"Event {event_key!r} is not associated with a sample_index.")
    row["sample_index"] = int(row["sample_index"])
    row["time_s"] = _finite_float(row.get("time_s"), f"event {event_key!r} time_s")
    row["object_id"] = None if row.get("object_id") in {None, ""} else str(row["object_id"])
    row["event_row_sha256"] = hashlib.sha256(canonical_json_bytes(row)).hexdigest()
    return row


def _resolve_object_id(
    workspace: ReviewWorkspace,
    *,
    requested: str | None,
    event: Mapping[str, Any] | None,
) -> str:
    result = workspace.query(
        "SELECT DISTINCT s.object_id FROM object_state s "
        "JOIN objects o ON o.object_id = s.object_id WHERE o.enabled = 1 ORDER BY s.object_id",
        max_rows=1001,
    )
    if result.truncated:
        raise CompletedRunStateExportError("Object selection exceeded the completed-run export limit.")
    available = [str(row["object_id"]) for row in result.rows]
    event_object = str((event or {}).get("object_id") or "").strip()
    requested_id = str(requested or "").strip()
    if event_object and requested_id and event_object != requested_id:
        raise CompletedRunStateExportError(
            f"Event object {event_object!r} conflicts with requested object {requested_id!r}."
        )
    resolved = requested_id or event_object
    if resolved:
        if resolved not in available:
            raise CompletedRunStateExportError(
                f"Object {resolved!r} has no enabled object_state evidence; available objects: {available}."
            )
        return resolved
    if len(available) != 1:
        raise CompletedRunStateExportError(
            f"Object selection is ambiguous; specify --object-id from: {available}."
        )
    return available[0]


def _selected_object(workspace: ReviewWorkspace, object_id: str) -> dict[str, Any]:
    return _one_row(
        workspace.query(
            "SELECT object_id, object_type, enabled, role FROM objects WHERE object_id = ?",
            (object_id,),
            max_rows=2,
        ),
        f"object {object_id!r}",
    )


def _object_config(config: Mapping[str, Any], object_id: str) -> dict[str, Any]:
    objects = config.get("objects")
    if not isinstance(objects, Mapping) or object_id not in objects or not isinstance(objects[object_id], Mapping):
        raise CompletedRunStateExportError(
            f"Verified run config does not contain selected object {object_id!r} under objects."
        )
    obj = deepcopy(dict(objects[object_id]))
    kind = str(obj.get("kind", "satellite") or "satellite")
    if kind != "satellite":
        raise CompletedRunStateExportError(
            f"Completed-run continuation v1 supports satellite objects; {object_id!r} is {kind!r}."
        )
    return obj


def _state_frame(workspace: ReviewWorkspace, object_id: str) -> str:
    row = _one_row(
        workspace.query(
            "SELECT state_frame FROM object_state_frame WHERE object_id = ?",
            (object_id,),
            max_rows=2,
        ),
        f"state frame for {object_id!r}",
    )
    return str(row["state_frame"] or "").strip().upper()


def _select_state_row(
    workspace: ReviewWorkspace,
    *,
    object_id: str,
    selector: str,
    sample_index: int | None,
    time_s: float | None,
    event: Mapping[str, Any] | None,
) -> dict[str, Any]:
    columns = ", ".join(_STATE_COLUMNS)
    if selector == "final":
        sql = (
            f"SELECT {columns} FROM object_state WHERE object_id = ? AND sample_index = "
            "(SELECT MAX(sample_index) FROM object_state WHERE object_id = ?)"
        )
        params: tuple[Any, ...] = (object_id, object_id)
        label = f"final state for {object_id!r}"
    elif selector == "sample_index":
        sql = f"SELECT {columns} FROM object_state WHERE object_id = ? AND sample_index = ?"
        params = (object_id, int(sample_index))
        label = f"sample_index {sample_index} for {object_id!r}"
    elif selector == "time_s":
        sql = f"SELECT {columns} FROM object_state WHERE object_id = ? AND time_s = ?"
        params = (object_id, float(time_s))
        label = f"time_s {time_s} for {object_id!r}"
    else:
        sql = f"SELECT {columns} FROM object_state WHERE object_id = ? AND sample_index = ?"
        params = (object_id, int(dict(event or {})["sample_index"]))
        label = f"event-associated sample for {object_id!r}"
    return _one_row(workspace.query(sql, params, max_rows=2), label)


def _verify_event_association(event: Mapping[str, Any], state: Mapping[str, Any]) -> None:
    if int(event["sample_index"]) != int(state["sample_index"]):
        raise CompletedRunStateExportError("Selected event and object state do not share a sample_index.")
    if not math.isclose(float(event["time_s"]), float(state["time_s"]), rel_tol=0.0, abs_tol=1.0e-9):
        raise CompletedRunStateExportError("Selected event and object state do not share the same sample time.")


def _matching_covariance(
    workspace: ReviewWorkspace,
    *,
    object_id: str,
    sample_index: int,
    time_s: float,
    epoch_jd_utc: float,
) -> dict[str, Any]:
    columns = workspace.table_columns().get("object_state_covariance", [])
    names = {str(item["name"]) for item in columns}
    required = {
        "sample_index",
        "time_s",
        "object_id",
        "frame",
        "component_order_json",
        "units_json",
        "covariance_json",
        "mathematically_valid",
        "calibrated",
        "calibration_scope",
    }
    if not columns:
        return {"present": False, "reason": "Review store has no supported full state-covariance table."}
    if not required.issubset(names):
        missing = sorted(required - names)
        raise CompletedRunStateExportError(
            f"object_state_covariance exists but is missing required columns: {missing}."
        )
    result = workspace.query(
        "SELECT sample_index, time_s, object_id, frame, component_order_json, units_json, "
        "covariance_json, mathematically_valid, calibrated, calibration_scope "
        "FROM object_state_covariance WHERE object_id = ? AND sample_index = ?",
        (object_id, sample_index),
        max_rows=2,
    )
    if result.row_count == 0:
        return {"present": False, "reason": "No full state covariance matches the selected object and sample."}
    row = _one_row(result, f"covariance for {object_id!r} sample_index {sample_index}")
    if not math.isclose(float(row["time_s"]), time_s, rel_tol=0.0, abs_tol=1.0e-9):
        raise CompletedRunStateExportError("Covariance sample time does not match the selected state sample.")
    if str(row["frame"] or "").strip().upper() != "ECI":
        raise CompletedRunStateExportError("Matching covariance is not in canonical ECI.")
    order = _json_list(row["component_order_json"], "covariance component order")
    units = _json_list(row["units_json"], "covariance units")
    if order != _STATE_COMPONENT_ORDER or units != _STATE_UNITS:
        raise CompletedRunStateExportError("Matching covariance ordering or units are incompatible with state product v1.")
    matrix = np.asarray(_json_list(row["covariance_json"], "covariance matrix"), dtype=float)
    if matrix.shape != (6, 6) or not np.all(np.isfinite(matrix)):
        raise CompletedRunStateExportError("Matching covariance must be a finite 6x6 matrix.")
    if not np.allclose(matrix, matrix.T, rtol=0.0, atol=1.0e-12):
        raise CompletedRunStateExportError("Matching covariance must be symmetric.")
    if float(np.min(np.linalg.eigvalsh(matrix))) < -1.0e-12:
        raise CompletedRunStateExportError("Matching covariance must be positive semidefinite.")
    if int(row["mathematically_valid"] or 0) != 1:
        raise CompletedRunStateExportError("Matching covariance is not marked mathematically valid.")
    calibrated = bool(row["calibrated"])
    scope = row["calibration_scope"]
    if calibrated and not str(scope or "").strip():
        raise CompletedRunStateExportError("Calibrated covariance is missing calibration_scope.")
    return {
        "present": True,
        "frame": "ECI",
        "epoch_jd_utc": epoch_jd_utc,
        "component_order": list(_STATE_COMPONENT_ORDER),
        "units": list(_STATE_UNITS),
        "matrix": matrix.tolist(),
        "mathematically_valid": True,
        "calibrated": calibrated,
        "calibration_scope": None if scope is None else str(scope),
    }


def _model_assumptions(config: Mapping[str, Any]) -> tuple[dict[str, Any], bool]:
    simulator = dict(config.get("simulator", {}) or {})
    dynamics = dict(simulator.get("dynamics", {}) or {})
    orbit = deepcopy(dict(dynamics.get("orbit", {}) or {}))
    orbit.setdefault("model", "two_body")
    environment = deepcopy(dict(simulator.get("environment", {}) or {}))
    if environment:
        orbit["environment"] = environment
    attitude_enabled = bool(dict(dynamics.get("attitude", {}) or {}).get("enabled", False))
    return orbit, attitude_enabled


def _selector_request(
    selector: str,
    *,
    sample_index: int | None,
    time_s: float | None,
    event_id: str | None,
) -> dict[str, Any]:
    if selector == "sample_index":
        return {"sample_index": int(sample_index)}
    if selector == "time_s":
        return {"time_s": float(time_s)}
    if selector == "event":
        return {"event_id": str(event_id)}
    return {"sample": "final"}


def _source_artifact(path: Path, *, target: Path, artifact_id: str) -> dict[str, Any]:
    return {
        "artifact_id": artifact_id,
        "sha256": sha256_file(path),
        "path": os.path.relpath(path, start=target.parent),
        "media_type": "application/vnd.sqlite3" if path.suffix == ".sqlite" else "application/json",
        "size_bytes": path.stat().st_size,
    }


def _data_markings(config: Mapping[str, Any], *, override: Mapping[str, Any] | None) -> dict[str, Any]:
    metadata = dict(config.get("metadata", {}) or {})
    export_review = dict(metadata.get("export_review", {}) or {})
    raw = dict(override or metadata.get("data_markings", {}) or {})
    explicit_public = (
        str(metadata.get("owner", "") or "").strip().lower() == "public"
        and bool(export_review.get("approved_for_public_export", False))
    )
    source_markings = []
    if export_review:
        source_markings.append(deepcopy(export_review))
    source_markings.extend(deepcopy(list(raw.get("source_markings", []) or [])))
    return {
        "scope": str(raw.get("scope", "public" if explicit_public else "private_pro") or "private_pro"),
        "handling": str(raw.get("handling", "completed simulator truth continuation") or "private"),
        "approved_for_public_export": bool(raw.get("approved_for_public_export", explicit_public)),
        "contains_customer_data": bool(raw.get("contains_customer_data", False)),
        "contains_hidden_truth": True,
        "source_markings": source_markings,
    }


def _one_row(result: Any, label: str) -> dict[str, Any]:
    if result.truncated or result.row_count != 1:
        raise CompletedRunStateExportError(
            f"Expected exactly one {label} row; found {'more than one' if result.truncated else result.row_count}."
        )
    return dict(result.rows[0])


def _finite_float(value: Any, label: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise CompletedRunStateExportError(f"{label} must be a finite number.") from exc
    if not math.isfinite(out):
        raise CompletedRunStateExportError(f"{label} must be a finite number.")
    return out


def _json_list(value: Any, label: str) -> list[Any]:
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError as exc:
        raise CompletedRunStateExportError(f"{label} is not valid JSON.") from exc
    if not isinstance(parsed, list):
        raise CompletedRunStateExportError(f"{label} must be a JSON array.")
    return parsed


def _schema_generated_utc(schema_path: Path) -> str:
    if not schema_path.is_file():
        return ""
    try:
        payload = json.loads(schema_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    return str(dict(payload or {}).get("generated_utc", "") or "").strip()


def _normalize_utc(value: Any, *, fallback: Any = None) -> str:
    text = str(value or "").strip() or str(fallback or "").strip()
    if not text:
        raise CompletedRunStateExportError(
            "Completed-run provenance has no generated_utc in run_metadata or review/schema.json; "
            "rerun the scenario before exporting a handoff product."
        )
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise CompletedRunStateExportError("run_metadata generated_utc is invalid.") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


__all__ = [
    "COMPLETED_RUN_SELECTOR_KINDS",
    "COMPLETED_RUN_STATE_ADAPTER_ID",
    "COMPLETED_RUN_STATE_ADAPTER_VERSION",
    "CompletedRunStateExportError",
    "build_completed_run_state_product",
    "export_completed_run_state",
]
