from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
import stat
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from sim.interchange.manifests import bind_output_path, write_json_document
from sim.interchange.provenance import canonical_json_bytes, compute_product_id, sha256_file
from sim.interchange.validation import validate_product
from sim.review import ReviewStoreNotFoundError, ReviewWorkspace
from sim.security.input_reads import (
    DEFAULT_MAX_READ_BYTES,
    SecureReadChanged,
    SecureReadError,
)

MANEUVER_DETECTION_ADAPTER_ID = "oel.maneuver_detection.review_export"
MANEUVER_DETECTION_ADAPTER_VERSION = "1"


class ManeuverDetectionExportError(ValueError):
    """Raised when completed evidence cannot identify one confirmed detection."""


def export_maneuver_detection_product(
    completed_run: str | Path,
    *,
    output_path: str | Path,
    logical_output_path: str | Path | None = None,
    writer: Any | None = None,
    protected: bool | None = None,
    event_id: str | None = None,
    observer_id: str | None = None,
    target_id: str | None = None,
    export_created_utc: str | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    target, logical_target = bind_output_path(
        output_path,
        logical_output_path,
        writer=writer,
    )
    workspace = ReviewWorkspace.open(completed_run, protected=protected)
    try:
        _require_checkpointed_source(workspace)
        product, source_bindings = _build_maneuver_detection_product_from_workspace(
            workspace,
            output_path=target,
            logical_output_path=logical_target,
            writer=writer,
            export_created_utc=export_created_utc,
            event_id=event_id,
            observer_id=observer_id,
            target_id=target_id,
        )
        try:
            write_json_document(product, target, writer=writer, overwrite=overwrite)
        except FileExistsError as exc:
            raise ManeuverDetectionExportError(
                "Output product exists with different content; pass overwrite=True explicitly to replace it."
            ) from exc
        _assert_maneuver_source_bindings(workspace, source_bindings)
        report = validate_product(
            product,
            source_path=logical_target,
            source_digest_reader=_maneuver_source_digest_reader(workspace, source_bindings),
        )
        _assert_maneuver_source_bindings(workspace, source_bindings)
        return {
            "status": "exported",
            "product_path": str(target),
            "product_id": product["product_id"],
            "observer_id": product["payload"]["observer"]["object_id"],
            "target_id": product["payload"]["target"]["object_id"],
            "event_id": product["payload"]["detection"]["event_id"],
            "valid": report.valid,
            "promotable": report.promotable,
            "execution_occurred": False,
            "issues": [item.to_dict() for item in report.issues],
        }
    finally:
        workspace.close()


def build_maneuver_detection_product(
    completed_run: str | Path,
    *,
    output_path: str | Path,
    logical_output_path: str | Path | None = None,
    writer: Any | None = None,
    protected: bool | None = None,
    event_id: str | None = None,
    observer_id: str | None = None,
    target_id: str | None = None,
    export_created_utc: str | None = None,
) -> dict[str, Any]:
    workspace = ReviewWorkspace.open(completed_run, protected=protected)
    try:
        _require_checkpointed_source(workspace)
        product, _source_bindings = _build_maneuver_detection_product_from_workspace(
            workspace,
            output_path=output_path,
            logical_output_path=logical_output_path,
            writer=writer,
            export_created_utc=export_created_utc,
            event_id=event_id,
            observer_id=observer_id,
            target_id=target_id,
        )
        return product
    finally:
        workspace.close()


def _build_maneuver_detection_product_from_workspace(
    workspace: ReviewWorkspace,
    *,
    output_path: str | Path,
    logical_output_path: str | Path | None,
    writer: Any | None,
    export_created_utc: str | None,
    event_id: str | None,
    observer_id: str | None,
    target_id: str | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    target, logical_target = bind_output_path(
        output_path,
        logical_output_path,
        writer=writer,
    )
    try:
        workspace.freeze_source()
    except (ReviewStoreNotFoundError, SecureReadError, sqlite3.DatabaseError) as exc:
        raise ManeuverDetectionExportError(
            "Completed-run review evidence could not be bound to a stable source snapshot."
        ) from exc
    metadata = _one(
        workspace.query(
            "SELECT run_id, scenario_name, oel_version, review_schema_version, generated_utc, "
            "config_json FROM run_metadata",
            max_rows=2,
        ).rows,
        "run metadata",
    )
    config = json.loads(str(metadata["config_json"]))
    if not isinstance(config, dict):
        raise ManeuverDetectionExportError("Verified run config must contain a mapping.")
    summary_path = workspace.output_dir / "master_run_summary.json"
    summary_bytes, summary_record = _read_output_file(
        workspace,
        "master_run_summary.json",
        required=True,
    )
    assert summary_bytes is not None and summary_record is not None
    try:
        summary = json.loads(summary_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ManeuverDetectionExportError(
            "Master run summary must be bounded UTF-8 JSON."
        ) from exc
    if not isinstance(summary, dict):
        raise ManeuverDetectionExportError("Master run summary must contain a mapping.")
    event = _select_event(workspace, event_id=event_id, target_id=target_id)
    observer, subject, detector = _select_detector_evidence(
        summary,
        observer_id=observer_id,
        target_id=str(event["object_id"] or target_id or ""),
        event_time_s=float(event["time_s"]),
    )
    detector_config = _detector_config(config, observer_id=observer, target_id=subject)
    initial_epoch = _optional_epoch(config)
    detection_epoch = (
        None if initial_epoch is None else initial_epoch + float(event["time_s"]) / 86400.0
    )
    db_path = workspace.logical_db_path
    db_record = workspace.source_db_record()
    source_artifacts = [
        _artifact(
            db_path,
            target=logical_target,
            artifact_id="completed_run_review_store",
            record=db_record,
        ),
        _artifact(
            summary_path,
            target=logical_target,
            artifact_id="completed_run_summary",
            record=summary_record,
        ),
    ]
    source_bindings: list[dict[str, Any]] = [
        {"kind": "db", "path": db_path, "record": db_record, "required": True},
        {
            "kind": "output",
            "name": "master_run_summary.json",
            "path": summary_path,
            "record": summary_record,
            "required": True,
        },
    ]
    run_log_path = workspace.output_dir / "master_run_log.json"
    _run_log_bytes, run_log_record = _read_output_file(
        workspace,
        "master_run_log.json",
        required=False,
    )
    source_bindings.append(
        {
            "kind": "output",
            "name": "master_run_log.json",
            "path": run_log_path,
            "record": run_log_record,
            "required": False,
        }
    )
    if run_log_record is not None:
        source_artifacts.append(
            _artifact(
                run_log_path,
                target=logical_target,
                artifact_id="completed_run_log",
                record=run_log_record,
            )
        )
    schema_record = workspace.source_schema_record()
    if schema_record is not None:
        source_artifacts.append(
            _artifact(
                workspace.schema_path,
                target=logical_target,
                artifact_id="completed_run_review_schema",
                record=schema_record,
            )
        )
        source_bindings.append(
            {"kind": "schema", "path": workspace.schema_path, "record": schema_record, "required": True}
        )
    source_generated_utc = _normalize_utc(metadata["generated_utc"])
    created = _export_created_utc(export_created_utc)
    summary_hash = str(summary_record["sha256"])
    event_hash = hashlib.sha256(canonical_json_bytes(event)).hexdigest()
    warnings: list[str] = []
    if source_generated_utc is None:
        warnings.append(
            "Source review provenance has no generated_utc; producer timestamp is unavailable and age freshness was not evaluated."
        )
    if detection_epoch is None:
        warnings.append("Source run uses relative time; supply an explicit epoch for state continuation.")
    product: dict[str, Any] = {
        "schema_id": "oel-product-envelope-v1",
        "schema_version": 1,
        "product_kind": "oel.maneuver_detection",
        "product_id": "oel.maneuver_detection:" + "0" * 64,
        "created_utc": created,
        "producer": {
            "capability_id": "runtime_ekf_maneuver_detection",
            "oel_version": str(metadata["oel_version"] or "unknown"),
            "run_id": str(metadata["run_id"]),
        },
        "payload": {
            "observer": {"object_id": observer},
            "target": {"object_id": subject},
            "detection": {
                "event_id": str(event["event_id"]),
                "status": "confirmed",
                "time_s": float(event["time_s"]),
                "sample_index": int(event["sample_index"]),
                "epoch_jd_utc": detection_epoch,
            },
            "detector": {
                "configuration": detector_config,
                "summary": deepcopy(detector),
            },
            "source_run": {
                "run_id": str(metadata["run_id"]),
                "scenario_name": str(metadata["scenario_name"]),
                "review_schema_version": str(metadata["review_schema_version"]),
                "initial_jd_utc": initial_epoch,
            },
            "evidence": {
                "event_row_sha256": event_hash,
                "summary_sha256": summary_hash,
                "event_query": (
                    "SELECT event_id, time_s, sample_index, object_id, event_type, severity, message, source "
                    "FROM events WHERE event_id = ?"
                ),
            },
        },
        "quality": {
            "disposition": "accepted",
            "producer_status": (
                "confirmed"
                if source_generated_utc is not None
                else "confirmed_timestamp_unavailable"
            ),
            "gates": {
                "event_unambiguous": True,
                "summary_confirmation_count_positive": int(
                    detector.get("maneuver_confirmed_event_count", 0) or 0
                )
                > 0,
                "event_time_matches_summary": math.isclose(
                    float(detector["maneuver_first_confirmed_t_s"]),
                    float(event["time_s"]),
                    rel_tol=0.0,
                    abs_tol=1.0e-9,
                ),
                "absolute_epoch_available": detection_epoch is not None,
                "adapter": {
                    "adapter_id": MANEUVER_DETECTION_ADAPTER_ID,
                    "adapter_version": MANEUVER_DETECTION_ADAPTER_VERSION,
                },
            },
            "warnings": warnings,
            "non_claims": [
                "A confirmed innovation-consistency change does not identify maneuver intent.",
                "The product does not estimate maneuver magnitude, direction, or probability.",
            ],
        },
        "freshness": {
            "integrity_status": "current",
            "age_status": "not_applicable" if source_generated_utc is not None else "not_evaluated",
            "reference_epoch_jd_utc": detection_epoch,
            "evaluated_utc": created if source_generated_utc is not None else None,
            "policy": {
                "assessment": "content_bound_completed_run_detection",
                **(
                    {}
                    if source_generated_utc is not None
                    else {"reason": "source_generated_utc_unavailable"}
                ),
            },
        },
        "provenance": {
            "source_artifacts": source_artifacts,
            "source_product_ids": [],
            "transformations": [
                {
                    "transformation_id": "review_event_and_summary_to_maneuver_detection",
                    "version": "1",
                    "details": {
                        "event_id": str(event["event_id"]),
                        "observer_id": observer,
                        "target_id": subject,
                    },
                }
            ],
        },
        "data_markings": _markings(config),
    }
    try:
        _assert_maneuver_source_bindings(workspace, source_bindings)
    except (ReviewStoreNotFoundError, SecureReadChanged, SecureReadError) as exc:
        raise ManeuverDetectionExportError(
            "Completed-run evidence changed while the maneuver detection product was being built."
        ) from exc
    product["product_id"] = compute_product_id(product)
    report = validate_product(
        product,
        source_path=logical_target,
        source_digest_reader=_maneuver_source_digest_reader(workspace, source_bindings),
    )
    if not report.valid:
        messages = "; ".join(f"{item.path}: {item.message}" for item in report.errors)
        raise ManeuverDetectionExportError(
            f"Generated maneuver detection product failed validation: {messages}"
        )
    _assert_maneuver_source_bindings(workspace, source_bindings)
    return product, source_bindings


def _select_event(
    workspace: ReviewWorkspace, *, event_id: str | None, target_id: str | None
) -> dict[str, Any]:
    if event_id:
        result = workspace.query(
            "SELECT event_id, time_s, sample_index, object_id, event_type, severity, message, source "
            "FROM events WHERE event_id = ?",
            (str(event_id),),
            max_rows=2,
        )
    elif target_id:
        result = workspace.query(
            "SELECT event_id, time_s, sample_index, object_id, event_type, severity, message, source "
            "FROM events WHERE event_type = 'maneuver_detection_confirmed' AND object_id = ? "
            "ORDER BY time_s",
            (str(target_id),),
            max_rows=2,
        )
    else:
        result = workspace.query(
            "SELECT event_id, time_s, sample_index, object_id, event_type, severity, message, source "
            "FROM events WHERE event_type = 'maneuver_detection_confirmed' ORDER BY time_s",
            max_rows=2,
        )
    event = _one(result.rows, "confirmed maneuver detection event")
    if event.get("event_type") != "maneuver_detection_confirmed":
        raise ManeuverDetectionExportError("Selected event is not a confirmed maneuver detection.")
    if event.get("sample_index") is None or event.get("object_id") in {None, ""}:
        raise ManeuverDetectionExportError("Detection event must bind a sample and target object.")
    return event


def _select_detector_evidence(
    summary: Mapping[str, Any], *, observer_id: str | None, target_id: str, event_time_s: float
) -> tuple[str, str, dict[str, Any]]:
    root = dict(summary.get("knowledge_consistency_by_observer", {}) or {})
    matches: list[tuple[str, str, dict[str, Any]]] = []
    for observer, targets_raw in root.items():
        if observer_id and str(observer) != str(observer_id):
            continue
        for subject, evidence_raw in dict(targets_raw or {}).items():
            if target_id and str(subject) != str(target_id):
                continue
            evidence = dict(evidence_raw or {})
            first = evidence.get("maneuver_first_confirmed_t_s")
            if (
                int(evidence.get("maneuver_confirmed_event_count", 0) or 0) > 0
                and first is not None
                and math.isclose(float(first), event_time_s, rel_tol=0.0, abs_tol=1.0e-9)
            ):
                matches.append((str(observer), str(subject), evidence))
    if len(matches) != 1:
        raise ManeuverDetectionExportError(
            f"Detection event must match exactly one observer/target summary; found {len(matches)}."
        )
    return matches[0]


def _detector_config(
    config: Mapping[str, Any], *, observer_id: str, target_id: str
) -> dict[str, Any]:
    observer = dict(dict(config.get("objects", {}) or {}).get(observer_id, {}) or {})
    knowledge = dict(observer.get("knowledge", {}) or {})
    if target_id not in [str(item) for item in list(knowledge.get("targets", []) or [])]:
        raise ManeuverDetectionExportError("Observer config does not bind the detected target.")
    estimation = dict(knowledge.get("estimation", {}) or {})
    detector = dict(estimation.get("maneuver_detection", {}) or {})
    if detector.get("enabled") is not True:
        raise ManeuverDetectionExportError("Observer maneuver detection is not enabled in source config.")
    result = deepcopy(detector)
    result["estimation_type"] = estimation.get("type")
    result["measurement_noise_diag"] = deepcopy(
        dict(estimation.get("ekf", {}) or {}).get("meas_noise_diag")
    )
    result["sensor_error"] = deepcopy(dict(knowledge.get("sensor_error", {}) or {}))
    return result


def export_event_centered_observations(
    detection_product_path: str | Path,
    *,
    output_path: str | Path,
    logical_output_path: str | Path | None = None,
    writer: Any | None = None,
    pre_event_s: float = 30.0,
    post_event_s: float = 30.0,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Export measured ECI states around a confirmed event without hidden truth."""

    source = Path(detection_product_path).expanduser().resolve()
    product = json.loads(source.read_text(encoding="utf-8"))
    report = validate_product(product, source_path=source)
    if not report.promotable or product.get("product_kind") != "oel.maneuver_detection":
        raise ManeuverDetectionExportError(
            "Event-centered observations require a promotable oel.maneuver_detection product."
        )
    if not math.isfinite(float(pre_event_s)) or float(pre_event_s) <= 0.0:
        raise ManeuverDetectionExportError("pre_event_s must be finite and positive.")
    if not math.isfinite(float(post_event_s)) or float(post_event_s) <= 0.0:
        raise ManeuverDetectionExportError("post_event_s must be finite and positive.")
    log_path = _provenance_artifact_path(product, source=source, artifact_id="completed_run_log")
    if log_path is None:
        raise ManeuverDetectionExportError(
            "Detection product has no bound master run log; event-centered measurements are unavailable."
        )
    run_log = json.loads(log_path.read_text(encoding="utf-8"))
    payload = dict(product["payload"])
    observer = str(dict(payload["observer"])["object_id"])
    target_id = str(dict(payload["target"])["object_id"])
    event_time = float(dict(payload["detection"])["time_s"])
    times = list(run_log.get("time_s", []) or [])
    measurements = list(
        dict(dict(run_log.get("knowledge_measurements_by_observer", {}) or {}).get(observer, {}) or {}).get(
            target_id, []
        )
        or []
    )
    if len(times) != len(measurements):
        raise ManeuverDetectionExportError(
            "Bound run log has inconsistent time and observer measurement history lengths."
        )
    lower = event_time - float(pre_event_s)
    upper = event_time + float(post_event_s)
    selected = [
        (float(time_s), list(state))
        for time_s, state in zip(times, measurements)
        if lower <= float(time_s) <= upper
        and len(list(state)) >= 6
        and all(math.isfinite(float(value)) for value in list(state)[:6])
    ]
    if not selected or not any(time_s <= event_time for time_s, _ in selected) or not any(
        time_s > event_time for time_s, _ in selected
    ):
        raise ManeuverDetectionExportError(
            "Event window must contain finite measurements both at/before and after the detection."
        )
    window_start = selected[0][0]
    config = dict(dict(payload["detector"])["configuration"])
    noise_diag = list(config.get("measurement_noise_diag") or [])
    position_sigma = _positive_sigma(noise_diag[:3], fallback=1.0e-6)
    velocity_sigma = _positive_sigma(noise_diag[3:6], fallback=1.0e-9)
    initial_epoch = dict(payload["source_run"]).get("initial_jd_utc")
    rows: list[dict[str, Any]] = []
    for time_s, state in selected:
        row: dict[str, Any] = {
            "time_s": time_s - window_start,
            "position_eci_km": [float(value) for value in state[:3]],
            "velocity_eci_km_s": [float(value) for value in state[3:6]],
            "position_sigma_km": position_sigma,
            "velocity_sigma_km_s": velocity_sigma,
            "partition": "fit" if time_s <= event_time else "holdout",
        }
        if initial_epoch is not None:
            row["jd_utc"] = float(initial_epoch) + time_s / 86400.0
        rows.append(row)
    packet = _build_event_observation_packet(
        object_id=target_id,
        rows=rows,
        metadata={
            "detection_product_id": product["product_id"],
            "observer_id": observer,
            "target_id": target_id,
            "event_id": dict(payload["detection"])["event_id"],
            "source_event_time_s": event_time,
            "event_time_in_packet_s": event_time - window_start,
            "pre_event_s": float(pre_event_s),
            "post_event_s": float(post_event_s),
            "measurement_source": "knowledge_measurements_by_observer",
            "hidden_truth_included": False,
            "position_sigma_policy": "sqrt_median_positive_ekf_measurement_variance",
            "velocity_sigma_policy": "sqrt_median_positive_ekf_measurement_variance",
        },
    )
    target, _logical_target = bind_output_path(
        output_path,
        logical_output_path,
        writer=writer,
    )
    try:
        write_json_document(packet, target, writer=writer, overwrite=overwrite)
    except FileExistsError as exc:
        raise ManeuverDetectionExportError(
            "Observation output exists with different content; pass overwrite=True explicitly to replace it."
        ) from exc
    return {
        "status": "exported",
        "observation_packet_path": str(target),
        "observation_count": len(rows),
        "fit_duration_s": event_time - window_start,
        "holdout_duration_s": selected[-1][0] - event_time,
        "event_time_in_packet_s": event_time - window_start,
        "truth_included": False,
        "execution_occurred": False,
    }


def _build_event_observation_packet(
    *,
    object_id: str,
    rows: list[dict[str, Any]],
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the bounded public event-window packet without generic Pro ingestion."""

    normalized: list[dict[str, Any]] = []
    for index, source_row in enumerate(rows):
        row = deepcopy(source_row)
        position_sigma = float(row["position_sigma_km"])
        velocity_sigma = float(row["velocity_sigma_km_s"])
        components = ["x_km", "y_km", "z_km", "vx_km_s", "vy_km_s", "vz_km_s"]
        variances = [position_sigma**2] * 3 + [velocity_sigma**2] * 3
        covariance = [
            [variance if row_index == column_index else 0.0 for column_index in range(6)]
            for row_index, variance in enumerate(variances)
        ]
        observation_id = f"{object_id}:observation:{index:06d}"
        normalized.append(
            {
                "observation_id": observation_id,
                "time_s": float(row["time_s"]),
                "jd_utc": float(row["jd_utc"]) if row.get("jd_utc") is not None else None,
                "time_system": (
                    "utc_julian_date" if row.get("jd_utc") is not None else "relative_seconds"
                ),
                "frame": "ECI",
                "partition": str(row["partition"]),
                "position_eci_km": [float(value) for value in row["position_eci_km"]],
                "velocity_eci_km_s": [
                    float(value) for value in row["velocity_eci_km_s"]
                ],
                "position_sigma_km": position_sigma,
                "velocity_sigma_km_s": velocity_sigma,
                "measurement_type": "eci_position_velocity",
                "uncertainty": {
                    "representation": "covariance",
                    "components": components,
                    "matrix": covariance,
                    "source": "scalar_sigmas",
                },
                "source_record": {"record_id": observation_id, "values": row},
                "normalization": {
                    "position_scale_to_km": 1.0,
                    "velocity_scale_to_km_s": 1.0,
                    "time_scale_to_s": 1.0,
                    "source_frame": "ECI",
                    "frame_transform": "identity",
                },
            }
        )
    partitions = ["excluded", "fit", "holdout", "unassigned"]
    return {
        "packet_version": 1,
        "observation_schema_version": 1,
        "kind": "oel.observation_packet",
        "created_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "source": {
            "type": "structured_observations",
            "label": "confirmed_maneuver_event_window",
            "metadata": deepcopy(dict(metadata)),
        },
        "object_id": str(object_id),
        "frame": "ECI",
        "measurement_type": "eci_position_velocity",
        "normalized_units": {"position": "km", "velocity": "km/s", "time": "s"},
        "observation_contract": {
            "schema_version": 1,
            "time_system": normalized[0]["time_system"],
            "time_origin": "first_observation",
            "frame": "ECI",
            "source_frame": "ECI",
            "normalized_units": {"position": "km", "velocity": "km/s", "time": "s"},
            "ordering": "nondecreasing_time_s",
            "duplicate_epoch_policy": "allow_with_explicit_unique_observation_id",
            "partition_values": partitions,
            "uncertainty_representation": "component_ordered_covariance",
            "source_value_policy": "preserved_per_observation",
            "normalization_transform_policy": "recorded_per_observation",
        },
        "observations": normalized,
        "summary": {
            "observation_count": len(normalized),
            "first_time_s": normalized[0]["time_s"],
            "last_time_s": normalized[-1]["time_s"],
            "arc_length_s": normalized[-1]["time_s"] - normalized[0]["time_s"],
            "has_velocity": True,
            "has_any_velocity": True,
            "velocity_observation_count": len(normalized),
            "has_attitude": False,
            "has_any_attitude": False,
            "attitude_observation_count": 0,
            "has_position_sigma": True,
            "has_velocity_sigma": True,
            "partition_counts": {
                name: sum(1 for row in normalized if row["partition"] == name)
                for name in partitions
            },
            "uncertainty_complete": True,
        },
        "warnings": [
            "Observation packets are estimation inputs, not simulation evidence.",
            "Preliminary OD and Kalman outputs from this packet use simplified estimator models unless a downstream tool states otherwise.",
        ],
        "validation": {
            "status": "ready_with_warnings",
            "notes": [
                "Use preliminary OD or Kalman filtering to convert observations into a mission input packet.",
                "Validate any generated scenario YAML before simulation execution.",
            ],
        },
    }


def _provenance_artifact_path(
    product: Mapping[str, Any], *, source: Path, artifact_id: str
) -> Path | None:
    for item in list(dict(product.get("provenance", {}) or {}).get("source_artifacts", []) or []):
        row = dict(item or {})
        if row.get("artifact_id") == artifact_id:
            path = (source.parent / str(row.get("path", ""))).resolve()
            if not path.is_file() or sha256_file(path) != row.get("sha256"):
                raise ManeuverDetectionExportError(
                    f"Bound {artifact_id} artifact is missing or has changed."
                )
            return path
    return None


def _positive_sigma(variances: list[Any], *, fallback: float) -> float:
    positive = [
        math.sqrt(float(value))
        for value in variances
        if isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) > 0.0
    ]
    return float(sorted(positive)[len(positive) // 2]) if positive else float(fallback)


def _optional_epoch(config: Mapping[str, Any]) -> float | None:
    value = dict(config.get("simulator", {}) or {}).get("initial_jd_utc")
    try:
        epoch = float(value)
    except (TypeError, ValueError):
        return None
    return epoch if math.isfinite(epoch) and epoch > 0.0 else None


def _require_checkpointed_source(workspace: ReviewWorkspace) -> None:
    try:
        live_wal = workspace.has_live_wal()
    except (ReviewStoreNotFoundError, SecureReadError, OSError) as exc:
        raise ManeuverDetectionExportError(
            "Completed-run review evidence could not be inspected securely."
        ) from exc
    if live_wal:
        raise ManeuverDetectionExportError(
            "Maneuver detection export requires a checkpointed review store; "
            "live SQLite WAL evidence must be checkpointed before handoff."
        )


def _read_output_file(
    workspace: ReviewWorkspace,
    name: str,
    *,
    required: bool,
) -> tuple[bytes | None, dict[str, int | str] | None]:
    """Read one run artifact through the retained source directory handle."""

    path = workspace.output_dir / name
    reader = workspace.source_output_reader
    if reader is not None:
        try:
            before = dict(reader.file_record(name, max_bytes=DEFAULT_MAX_READ_BYTES))
        except (FileNotFoundError, NotADirectoryError):
            before = None
        if before is None:
            _assert_output_missing_stable(workspace, name)
            if required:
                raise ManeuverDetectionExportError(f"Completed run has no {name}.")
            return None, None
        try:
            payload = reader.read_bytes(name, max_bytes=DEFAULT_MAX_READ_BYTES)
            after = dict(reader.file_record(name, max_bytes=DEFAULT_MAX_READ_BYTES))
        except (FileNotFoundError, NotADirectoryError, SecureReadError) as exc:
            raise ManeuverDetectionExportError(f"Completed run artifact {name} could not be read securely.") from exc
    else:
        if getattr(workspace, "_protected", False):
            raise ManeuverDetectionExportError(
                "Protected maneuver detection export requires a retained output directory descriptor."
            )
        before = _legacy_file_record(path, max_bytes=DEFAULT_MAX_READ_BYTES, optional=True)
        if before is None:
            _assert_output_missing_stable(workspace, name)
            if required:
                raise ManeuverDetectionExportError(f"Completed run has no {name}.")
            return None, None
        try:
            payload = path.read_bytes()
        except OSError as exc:
            raise ManeuverDetectionExportError(f"Completed run artifact {name} could not be read.") from exc
        after = _legacy_file_record(path, max_bytes=DEFAULT_MAX_READ_BYTES, optional=False)

    assert after is not None
    digest = hashlib.sha256(payload).hexdigest()
    if (
        before != after
        or int(after.get("bytes", -1)) != len(payload)
        or str(after.get("sha256", "")) != digest
    ):
        raise ManeuverDetectionExportError(
            f"Completed run artifact {name} changed while it was being read."
        )
    return payload, after


def _assert_output_missing_stable(workspace: ReviewWorkspace, name: str) -> None:
    """Ensure an optional output did not appear between two descriptor reads."""

    reader = workspace.source_output_reader
    if reader is not None:
        try:
            reader.file_record(name, max_bytes=DEFAULT_MAX_READ_BYTES)
        except (FileNotFoundError, NotADirectoryError):
            return
        raise SecureReadChanged(f"Completed run artifact appeared during its evidence read: {name}")
    if getattr(workspace, "_protected", False):
        raise ManeuverDetectionExportError(
            "Protected maneuver detection export requires a retained output directory descriptor."
        )
    if _legacy_file_record(workspace.output_dir / name, max_bytes=DEFAULT_MAX_READ_BYTES, optional=True) is not None:
        raise SecureReadChanged(f"Completed run artifact appeared during its evidence read: {name}")


def _legacy_file_record(
    path: Path,
    *,
    max_bytes: int,
    optional: bool,
) -> dict[str, int | str] | None:
    try:
        if path.is_symlink():
            raise SecureReadError(f"Legacy maneuver source must not be a symbolic link: {path}")
        before = path.stat()
    except FileNotFoundError:
        if optional:
            return None
        raise
    if not stat.S_ISREG(before.st_mode):
        raise SecureReadError(f"Legacy maneuver source must be a regular file: {path}")
    if int(before.st_size) > int(max_bytes):
        raise SecureReadError(
            f"Legacy maneuver source exceeds the {int(max_bytes)}-byte read budget: {path}"
        )
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            if size > int(max_bytes):
                raise SecureReadError(
                    f"Legacy maneuver source exceeds the {int(max_bytes)}-byte read budget: {path}"
                )
            digest.update(chunk)
    after = path.stat()
    if (
        (
            int(before.st_dev),
            int(before.st_ino),
            int(before.st_size),
            int(before.st_mtime_ns),
            int(before.st_ctime_ns),
            int(before.st_nlink),
        )
        != (
            int(after.st_dev),
            int(after.st_ino),
            int(after.st_size),
            int(after.st_mtime_ns),
            int(after.st_ctime_ns),
            int(after.st_nlink),
        )
        or size != int(after.st_size)
    ):
        raise SecureReadChanged(f"Legacy maneuver source changed during its evidence read: {path}")
    return {
        "bytes": int(size),
        "sha256": digest.hexdigest(),
        "st_dev": int(after.st_dev),
        "st_ino": int(after.st_ino),
        "mtime_ns": int(after.st_mtime_ns),
        "ctime_ns": int(after.st_ctime_ns),
        "nlink": int(after.st_nlink),
    }


def _current_maneuver_binding_record(
    workspace: ReviewWorkspace,
    binding: Mapping[str, Any],
) -> dict[str, int | str] | None:
    kind = str(binding.get("kind", ""))
    if kind == "db":
        return workspace.source_db_record()
    if kind == "schema":
        return workspace.source_schema_record()
    if kind == "output":
        name = str(binding.get("name", ""))
        _payload, record = _read_output_file(
            workspace,
            name,
            required=bool(binding.get("required", True)),
        )
        return record
    raise ManeuverDetectionExportError(f"Unknown retained maneuver source binding kind: {kind!r}")


def _assert_maneuver_source_bindings(
    workspace: ReviewWorkspace,
    bindings: list[Mapping[str, Any]],
) -> None:
    try:
        workspace.assert_source_unchanged()
        for binding in bindings:
            expected = binding.get("record")
            current = _current_maneuver_binding_record(workspace, binding)
            if expected != current:
                raise SecureReadChanged(
                    f"Maneuver source changed: {binding.get('path', '<unknown>')}"
                )
    except (ReviewStoreNotFoundError, SecureReadChanged, SecureReadError, OSError) as exc:
        raise ManeuverDetectionExportError(
            "Completed-run evidence changed while the maneuver detection product was being built."
        ) from exc


def _maneuver_source_digest_reader(
    workspace: ReviewWorkspace,
    bindings: list[Mapping[str, Any]],
):
    by_path = {
        _absolute_lexical(Path(binding["path"])): binding
        for binding in bindings
    }

    def read(candidate: Path) -> str | None:
        try:
            workspace.assert_source_unchanged()
            binding = by_path.get(_absolute_lexical(candidate))
            if binding is None:
                return None
            current = _current_maneuver_binding_record(workspace, binding)
            expected = binding.get("record")
            if expected != current:
                raise SecureReadChanged(
                    f"Maneuver source changed: {binding.get('path', '<unknown>')}"
                )
            if current is None:
                return None
            return str(current["sha256"])
        except (ReviewStoreNotFoundError, SecureReadChanged, SecureReadError, OSError) as exc:
            raise ManeuverDetectionExportError(
                "Retained maneuver source verification failed."
            ) from exc

    return read


def _absolute_lexical(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _artifact(
    path: Path,
    *,
    target: Path,
    artifact_id: str,
    record: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "artifact_id": artifact_id,
        "sha256": str(record["sha256"]),
        "path": os.path.relpath(path, start=target.parent),
        "media_type": "application/vnd.sqlite3" if path.suffix == ".sqlite" else "application/json",
        "size_bytes": int(record["bytes"]),
    }


def _one(rows: list[dict[str, Any]], label: str) -> dict[str, Any]:
    if len(rows) != 1:
        raise ManeuverDetectionExportError(f"Expected exactly one {label}; found {len(rows)}.")
    return dict(rows[0])


def _export_created_utc(value: str | None) -> str:
    """Return the actual product export time, unless a test override is supplied."""

    if value is None:
        return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    normalized = _normalize_utc(value, label="export_created_utc", allow_missing=False)
    assert normalized is not None
    return normalized


def _normalize_utc(
    value: Any,
    *,
    label: str = "generated_utc",
    allow_missing: bool = True,
) -> str | None:
    text = str(value or "").strip()
    if not text:
        if allow_missing:
            return None
        raise ManeuverDetectionExportError(f"Maneuver detection provenance has no {label}.")
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ManeuverDetectionExportError(f"{label} is invalid.") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _markings(config: Mapping[str, Any]) -> dict[str, Any]:
    metadata = dict(config.get("metadata", {}) or {})
    public = metadata.get("owner") == "public" and bool(metadata.get("public_surface"))
    return {
        "scope": "public" if public else "private_pro",
        "handling": "public_synthetic" if public else "private",
        "approved_for_public_export": bool(public),
        "contains_customer_data": False,
        "contains_hidden_truth": False,
    }


__all__ = [
    "MANEUVER_DETECTION_ADAPTER_ID",
    "MANEUVER_DETECTION_ADAPTER_VERSION",
    "ManeuverDetectionExportError",
    "build_maneuver_detection_product",
    "export_maneuver_detection_product",
    "export_event_centered_observations",
]
