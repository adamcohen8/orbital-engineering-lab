from __future__ import annotations

import json
import os
from copy import deepcopy
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from sim.interchange.provenance import compute_product_id, sha256_file
from sim.interchange.validation import validate_product
from sim.security.output_writes import write_output_text

SEQUENTIAL_OD_ADAPTER_ID = "oel.sequential_od.state_estimate"
SEQUENTIAL_OD_ADAPTER_VERSION = "1"
SELECTIONS = {"filtered", "smoothed"}


class SequentialODInterchangeError(ValueError):
    """Raised when sequential-OD evidence cannot produce a coherent state product."""


def build_sequential_od_state_product(
    artifact_dir: str | Path,
    *,
    selection: str,
    output_path: str | Path | None = None,
    object_specs: Mapping[str, Any] | None = None,
    data_markings: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    selected = str(selection).strip().lower()
    if selected not in SELECTIONS:
        raise SequentialODInterchangeError("selection must be explicitly 'filtered' or 'smoothed'.")
    root = Path(artifact_dir).expanduser().resolve()
    report_path = root / "sequential_od_report.json"
    packet_path = root / "observation_packet.json"
    if not report_path.is_file() or not packet_path.is_file():
        raise SequentialODInterchangeError(
            "Sequential OD promotion requires sequential_od_report.json and observation_packet.json."
        )
    report = _load_object(report_path)
    packet = _load_object(packet_path)
    records = list(report.get("records", []) or [])
    observations = list(packet.get("observations", []) or [])
    if not records:
        raise SequentialODInterchangeError("Sequential OD report contains no state records.")
    final = dict(records[-1] or {})
    sample_index = int(final.get("sample_index", -1))
    if sample_index < 0 or sample_index >= len(observations):
        raise SequentialODInterchangeError("Final sequential state cannot be bound to its observation epoch.")
    observation = dict(observations[sample_index] or {})
    if float(observation.get("time_s", float("nan"))) != float(final.get("time_s", float("nan"))):
        raise SequentialODInterchangeError("Final state and observation time_s values do not match exactly.")
    epoch = _positive_float(observation.get("jd_utc"), "final observation jd_utc")
    state = _vector6(final.get(f"{selected}_state"), f"{selected}_state")
    covariance = _covariance(final.get(f"{selected}_covariance"), epoch=epoch)
    object_id = str(report.get("object_id", "") or "").strip()
    if not object_id or object_id != str(packet.get("object_id", "") or "").strip():
        raise SequentialODInterchangeError("Sequential report and observation packet object IDs must match.")

    target = (
        Path(output_path).expanduser().resolve()
        if output_path is not None
        else root / f"state_estimate_{selected}_product.json"
    )
    source_paths = [report_path, packet_path]
    for name in ("sequential_od_samples.csv", "review/run.sqlite"):
        candidate = root / name
        if candidate.is_file():
            source_paths.append(candidate)
    source_artifacts = [_source_artifact(path, target.parent) for path in source_paths]
    gates = deepcopy(dict(report.get("quality_gates", {}) or {}))
    accepted = (
        gates.get("evidence_status") == "ready_for_statistical_campaign"
        and gates.get("covariance_symmetric_psd") is True
        and gates.get("smoothing_improves_or_preserves") is True
    )
    warnings: list[str] = []
    if not accepted:
        warnings.append("Sequential OD quality gates do not support automatic promotion.")
    created_utc = _required_timestamp(packet.get("created_utc"))
    dynamics = dict(report.get("dynamics", {}) or {})
    force_model = {
        "model": str(dynamics.get("model", "two_body") or "two_body"),
        "integrator": str(dynamics.get("integrator", "rk4") or "rk4"),
        "orbit_substep_s": float(dynamics.get("integration_substep_s", 10.0)),
    }
    product: dict[str, Any] = {
        "schema_id": "oel-product-envelope-v1",
        "schema_version": 1,
        "product_kind": "oel.state_estimate",
        "product_id": "oel.state_estimate:" + "0" * 64,
        "created_utc": created_utc,
        "producer": {
            "capability_id": "sequential_orbit_determination",
            "oel_version": _oel_version(),
            "run_id": str(report.get("packet_digest_sha256", root.name) or root.name),
        },
        "payload": {
            "object": {"object_id": object_id, "role": object_id, "kind": "satellite"},
            "state": {
                "representation": "cartesian_position_velocity",
                "frame": "ECI",
                "epoch": {"value": epoch, "format": "jd", "time_system": "UTC"},
                "component_order": ["x", "y", "z", "vx", "vy", "vz"],
                "units": ["km", "km", "km", "km/s", "km/s", "km/s"],
                "values": state,
            },
            "covariance": covariance,
            "object_specs": deepcopy(dict(object_specs or {})),
            "model_assumptions": {
                "orbit_force_model": force_model,
                "attitude": {"source": "none"},
            },
            "estimator_evidence": {
                "method": f"{report.get('estimator', 'sequential')}_{selected}",
                "selected_parameters": ["x", "y", "z", "vx", "vy", "vz"],
                "od_solution_id": None,
                "source_report_sha256": sha256_file(report_path),
            },
        },
        "quality": {
            "disposition": "accepted" if accepted else "review_required",
            "producer_status": str(gates.get("evidence_status", "unknown") or "unknown"),
            "gates": {
                "quality_gates": gates,
                "selection": selected,
                "final_record": {
                    "sample_index": sample_index,
                    "time_s": float(final["time_s"]),
                    "maneuver_confirmed": bool(final.get("maneuver_confirmed", False)),
                    "reinitialized": bool(final.get("reinitialized", False)),
                    "smoothing_boundary": bool(final.get("smoothing_boundary", False)),
                },
                "maneuver_recovery": deepcopy(dict(report.get("maneuver_recovery", {}) or {})),
                "adapter": {"adapter_id": SEQUENTIAL_OD_ADAPTER_ID, "adapter_version": SEQUENTIAL_OD_ADAPTER_VERSION},
            },
            "warnings": warnings,
            "non_claims": [str(report.get("claim_posture", {}).get("note", "Covariance is not calibrated."))],
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
            "transformations": [{
                "transformation_id": "sequential_od_record_selection",
                "version": "1",
                "details": {"selection": selected, "sample_index": sample_index, "time_s": float(final["time_s"])},
            }],
        },
        "data_markings": _markings(data_markings),
    }
    product["product_id"] = compute_product_id(product)
    validation = validate_product(product, source_path=target)
    if not validation.valid:
        messages = "; ".join(f"{issue.path}: {issue.message}" for issue in validation.errors)
        raise SequentialODInterchangeError(f"Generated sequential state product failed validation: {messages}")
    return product


def emit_sequential_od_state_products(
    artifact_dir: str | Path,
    *,
    object_specs: Mapping[str, Any] | None = None,
    data_markings: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    root = Path(artifact_dir).expanduser().resolve()
    products: dict[str, Any] = {}
    for selection in sorted(SELECTIONS):
        target = root / f"state_estimate_{selection}_product.json"
        product = build_sequential_od_state_product(
            root,
            selection=selection,
            output_path=target,
            object_specs=object_specs,
            data_markings=data_markings,
        )
        write_output_text(str(target), json.dumps(product, indent=2, sort_keys=True) + "\n")
        validation = validate_product(product, source_path=target)
        products[selection] = {
            "product_path": str(target),
            "product_id": product["product_id"],
            "disposition": product["quality"]["disposition"],
            "valid": validation.valid,
            "promotable": validation.promotable,
        }
    return {"status": "emitted", "selection_required_for_materialization": True, "products": products}


def _covariance(value: Any, *, epoch: float) -> dict[str, Any]:
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (6, 6) or not np.all(np.isfinite(matrix)):
        raise SequentialODInterchangeError("Selected sequential covariance must be a finite 6x6 matrix.")
    if not np.allclose(matrix, matrix.T, rtol=0.0, atol=1.0e-12):
        raise SequentialODInterchangeError("Selected sequential covariance must be symmetric.")
    if float(np.min(np.linalg.eigvalsh(matrix))) < -1.0e-12:
        raise SequentialODInterchangeError("Selected sequential covariance must be positive semidefinite.")
    return {
        "present": True,
        "frame": "ECI",
        "epoch_jd_utc": epoch,
        "component_order": ["x", "y", "z", "vx", "vy", "vz"],
        "units": ["km", "km", "km", "km/s", "km/s", "km/s"],
        "matrix": matrix.tolist(),
        "mathematically_valid": True,
        "calibrated": False,
        "calibration_scope": None,
    }


def _source_artifact(path: Path, base: Path) -> dict[str, Any]:
    media = "application/json" if path.suffix == ".json" else "text/csv" if path.suffix == ".csv" else "application/vnd.sqlite3"
    return {
        "artifact_id": path.stem,
        "sha256": sha256_file(path),
        "path": os.path.relpath(path, start=base),
        "media_type": media,
        "size_bytes": path.stat().st_size,
    }


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SequentialODInterchangeError(f"{path.name} must contain a JSON object.")
    return value


def _vector6(value: Any, field: str) -> list[float]:
    array = np.asarray(value, dtype=float).reshape(-1)
    if array.size != 6 or not np.all(np.isfinite(array)):
        raise SequentialODInterchangeError(f"{field} must contain six finite values.")
    return array.tolist()


def _positive_float(value: Any, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise SequentialODInterchangeError(f"{field} must be explicit and finite.") from exc
    if not np.isfinite(number) or number <= 0.0:
        raise SequentialODInterchangeError(f"{field} must be explicit and positive.")
    return number


def _required_timestamp(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        raise SequentialODInterchangeError("Observation packet created_utc is required.")
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise SequentialODInterchangeError("Observation packet created_utc must be ISO-8601.") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _markings(value: Mapping[str, Any] | None) -> dict[str, Any]:
    raw = dict(value or {})
    return {
        "scope": str(raw.get("scope", "private_pro") or "private_pro"),
        "handling": str(raw.get("handling", "private") or "private"),
        "approved_for_public_export": bool(raw.get("approved_for_public_export", False)),
        "contains_customer_data": bool(raw.get("contains_customer_data", False)),
        "contains_hidden_truth": bool(raw.get("contains_hidden_truth", False)),
    }


def _oel_version() -> str:
    try:
        return version("orbital-engineering-lab")
    except PackageNotFoundError:
        return "0.25.0"


__all__ = [
    "SEQUENTIAL_OD_ADAPTER_ID",
    "SEQUENTIAL_OD_ADAPTER_VERSION",
    "SequentialODInterchangeError",
    "build_sequential_od_state_product",
    "emit_sequential_od_state_products",
]
