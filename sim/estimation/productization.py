from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from sim.data_io.common import strict_json
from sim.data_io.contracts import DataIOError
from sim.observations import inspect_observation_packet, observation_packet_from_dict
from sim.utils.io import SafeReadError, read_regular_file_nofollow, write_json

OD_DATASET_MANIFEST_SCHEMA_VERSION = 1
OD_SOLUTION_SCHEMA_VERSION = 1
OD_AGENT_EVIDENCE_PACKET_SCHEMA_VERSION = 1
OD_CLAIM_LEVELS = frozenset({"L0", "L1", "L2", "L3", "L4", "L5", "L6"})
MAX_SAFE_INSPECTION_BYTES = 16 * 1024 * 1024

STANDARD_OD_REVIEW_QUERIES: dict[str, str] = {
    "sequential_summary": "SELECT object_id, estimator, observation_count, accepted_count, rejected_count, evidence_status FROM sequential_od_run",
    "sequential_samples": "SELECT sample_index, time_s, measurement_accepted, nis, filtered_nees, smoothed_nees, maneuver_confirmed FROM sequential_od_samples ORDER BY sample_index",
    "relative_models": "SELECT model, fit_position_rms_m, holdout_position_rms_m, batch_solver_success, sequential_position_rms_m FROM relative_od_models ORDER BY model",
    "ogp_summary": "SELECT * FROM ogp_od_run LIMIT 1",
    "phase_campaign_cases": "SELECT case_id, passed, status FROM od_phase9_pilots ORDER BY case_id",
}


@dataclass(frozen=True)
class ODAdapterRegistration:
    adapter_id: str
    packet_kind: str
    implementation: str
    trust: str
    description: str

    def __post_init__(self) -> None:
        if self.trust not in {"builtin", "reviewed_external", "untrusted"}:
            raise ValueError("adapter trust must be builtin, reviewed_external, or untrusted.")


class ODAdapterRegistry:
    def __init__(self) -> None:
        self._registrations: dict[str, ODAdapterRegistration] = {}

    def register(self, registration: ODAdapterRegistration) -> None:
        if registration.adapter_id in self._registrations:
            raise ValueError(f"OD adapter {registration.adapter_id!r} is already registered.")
        self._registrations[registration.adapter_id] = registration

    def list(self) -> list[dict[str, str]]:
        return [asdict(self._registrations[key]) for key in sorted(self._registrations)]

    def inspect(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        kind = str(payload.get("kind") or payload.get("format") or "")
        matches = [item for item in self._registrations.values() if item.packet_kind == kind]
        if not matches:
            return {
                "status": "unsupported",
                "packet_kind": kind,
                "execution_authorized": False,
                "reason": "no registered adapter matches this packet kind",
            }
        registration = matches[0]
        trusted = registration.trust in {"builtin", "reviewed_external"}
        return {
            "status": "safe_inspected" if trusted else "trust_required",
            "packet_kind": kind,
            "adapter": asdict(registration),
            "execution_authorized": False,
            "ordinary_validation_allowed": trusted,
            "note": "Inspection never authorizes execution; referenced modules and paths remain separate trust decisions.",
        }


def default_od_adapter_registry() -> ODAdapterRegistry:
    registry = ODAdapterRegistry()
    for registration in (
        ODAdapterRegistration(
            "oel_observation_packet_v1",
            "oel.observation_packet",
            "sim.observations.observation_packet_from_dict",
            "builtin",
            "Normalized ECI position/velocity observation packet.",
        ),
        ODAdapterRegistration(
            "oel_ogp_teme_pv_v1",
            "oel.ogp_pv_observation_packet",
            "sim.estimation.ogp_od.run_ogp_pv_packet",
            "builtin",
            "Native TEME OGP position/velocity observation packet.",
        ),
        ODAdapterRegistration(
            "oel_optical_radec_v1",
            "oel.optical_angle_packet",
            "sim.optical_tracking.adapt_optical_ra_dec_observations",
            "builtin",
            "Optical RA/Dec packet with explicit observer metadata.",
        ),
        ODAdapterRegistration(
            "oel_ground_tracking_v1",
            "oel_ground_tracking",
            "sim.tracking_data.adapt_ground_tracking_packet",
            "builtin",
            "Versioned OEL ground tracking packet with explicit station metadata.",
        ),
        ODAdapterRegistration(
            "oel_relative_ric_v1",
            "oel.relative_ric_observation_packet",
            "sim.estimation.relative_od_integrated.run_integrated_relative_od",
            "builtin",
            "Rectangular RIC state observations with explicit chief and covariance inputs.",
        ),
    ):
        registry.register(registration)
    return registry


def safely_inspect_od_input(
    value: str | Path | Mapping[str, Any],
    *,
    registry: ODAdapterRegistry | None = None,
) -> dict[str, Any]:
    source_path = None
    if isinstance(value, (str, Path)):
        source_path = Path(os.path.abspath(os.fspath(Path(value).expanduser())))
        if source_path.suffix.lower() != ".json":
            return {
                "status": "unsupported_safe_inspection",
                "source_path": str(source_path),
                "execution_authorized": False,
                "reason": "Phase 9 safe inspection accepts JSON packets only; source-specific adapters must be trusted separately.",
            }
        try:
            raw = read_regular_file_nofollow(
                source_path,
                min_bytes=1,
                max_bytes=MAX_SAFE_INSPECTION_BYTES,
            )
            payload = strict_json(raw.decode("utf-8"))
        except (SafeReadError, UnicodeDecodeError, DataIOError) as exc:
            return {
                "status": "safe_inspection_failed",
                "source_path": str(source_path),
                "execution_authorized": False,
                "reason": str(exc),
            }
    else:
        payload = dict(value)
    if not isinstance(payload, Mapping):
        return {
            "status": "safe_inspection_failed",
            "source_path": None if source_path is None else str(source_path),
            "execution_authorized": False,
            "reason": "OD input JSON root must be an object.",
        }
    try:
        payload_sha256 = _digest(payload)
    except (TypeError, ValueError) as exc:
        return {
            "status": "safe_inspection_failed",
            "source_path": None if source_path is None else str(source_path),
            "execution_authorized": False,
            "reason": f"OD input is not a finite JSON object: {exc}",
        }
    inspection = (registry or default_od_adapter_registry()).inspect(payload)
    inspection["source_path"] = None if source_path is None else str(source_path)
    if inspection["status"] == "safe_inspected" and payload.get("kind") == "oel.observation_packet":
        try:
            inspection["normalized_packet"] = inspect_observation_packet(observation_packet_from_dict(payload))
        except (KeyError, TypeError, ValueError) as exc:
            inspection.update(
                {
                    "status": "safe_inspection_failed",
                    "ordinary_validation_allowed": False,
                    "reason": str(exc),
                }
            )
    inspection["payload_sha256"] = payload_sha256
    return inspection


def build_od_dataset_manifest(
    *,
    dataset_id: str,
    sources: Sequence[str | Path],
    source_metadata: Mapping[str, Mapping[str, Any]] | None = None,
    handling_classification: str = "private",
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    metadata = dict(source_metadata or {})
    records = []
    for raw_path in sources:
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"OD dataset source does not exist: {path}")
        item_metadata = dict(metadata.get(str(raw_path), metadata.get(str(path), {})) or {})
        records.append(
            {
                "path": str(path),
                "sha256": _file_digest(path),
                "size_bytes": path.stat().st_size,
                "canonical_source": item_metadata.get("canonical_source"),
                "archive_identifier": item_metadata.get("archive_identifier"),
                "license_or_terms": item_metadata.get("license_or_terms", "workspace_governed"),
                "retrieved_utc": item_metadata.get("retrieved_utc"),
                "frame": item_metadata.get("frame"),
                "time_system": item_metadata.get("time_system"),
                "station_metadata": item_metadata.get("station_metadata"),
                "restrictions": item_metadata.get("restrictions", []),
                "transformations": item_metadata.get("transformations", []),
                "reference_uncertainty": item_metadata.get("reference_uncertainty"),
            }
        )
    manifest = {
        "schema_version": OD_DATASET_MANIFEST_SCHEMA_VERSION,
        "kind": "oel.od_dataset_manifest",
        "dataset_id": str(dataset_id),
        "created_utc": _utc_now(),
        "handling_classification": str(handling_classification),
        "sources": records,
    }
    manifest["manifest_fingerprint_sha256"] = _digest(
        {"dataset_id": manifest["dataset_id"], "handling": handling_classification, "sources": records}
    )
    if output_path is not None:
        write_json(str(Path(output_path).expanduser()), manifest)
    return manifest


def normalize_od_solution(
    *,
    capability_id: str,
    report: Mapping[str, Any],
    claim_level: str,
    input_fingerprints: Mapping[str, str],
    estimator: Mapping[str, Any],
    propagator: Mapping[str, Any],
    frame_policy: Mapping[str, Any],
    quality_state: str | None = None,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    level = str(claim_level).upper()
    if level not in OD_CLAIM_LEVELS:
        raise ValueError(f"claim_level must be one of {sorted(OD_CLAIM_LEVELS)}")
    components = {
        "inputs": dict(sorted((str(key), str(value)) for key, value in input_fingerprints.items())),
        "estimator": dict(estimator),
        "propagator": dict(propagator),
        "frame_policy": dict(frame_policy),
        "report_schema_version": report.get("schema_version"),
        "capability_id": capability_id,
    }
    fingerprint = _digest(components)
    gates = dict(report.get("quality_gates", {}) or {})
    resolved_quality = str(
        quality_state
        or gates.get("evidence_status")
        or dict(report.get("verdict", {}) or {}).get("recommended_action")
        or "review_required"
    )
    solution = {
        "schema_version": OD_SOLUTION_SCHEMA_VERSION,
        "kind": "oel.od_solution",
        "solution_id": f"od_solution:{capability_id}:{fingerprint[:20]}",
        "created_utc": _utc_now(),
        "capability_id": str(capability_id),
        "quality_state": resolved_quality,
        "claim_level": level,
        "fingerprint_sha256": fingerprint,
        "fingerprint_components": components,
        "estimator": dict(estimator),
        "propagator": dict(propagator),
        "frame_policy": dict(frame_policy),
        "quality_gates": gates,
        "claim_posture": dict(report.get("claim_posture", {}) or {}),
        "source_report": dict(report),
    }
    if output_path is not None:
        write_json(str(Path(output_path).expanduser()), solution)
    return solution


def build_od_agent_evidence_packet(
    *,
    task_id: str,
    solution: Mapping[str, Any],
    dataset_manifest: Mapping[str, Any],
    reproduction_commands: Sequence[str],
    artifacts: Sequence[str | Path] = (),
    review_queries: Mapping[str, str] | None = None,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    artifact_rows = []
    for raw_path in artifacts:
        path = Path(raw_path).expanduser().resolve()
        artifact_rows.append(
            {
                "path": str(path),
                "exists": path.is_file(),
                "sha256": _file_digest(path) if path.is_file() else None,
                "size_bytes": path.stat().st_size if path.is_file() else None,
            }
        )
    packet = {
        "schema_version": OD_AGENT_EVIDENCE_PACKET_SCHEMA_VERSION,
        "kind": "oel.od_agent_evidence_packet",
        "task_id": str(task_id),
        "created_utc": _utc_now(),
        "solution": dict(solution),
        "dataset_manifest": dict(dataset_manifest),
        "reproduction": {
            "commands": [str(item) for item in reproduction_commands],
            "requires_clean_environment": True,
            "hidden_state_allowed": False,
        },
        "review_queries": dict(review_queries or STANDARD_OD_REVIEW_QUERIES),
        "artifacts": artifact_rows,
        "freshness": {
            "status": "current",
            "solution_fingerprint_sha256": solution["fingerprint_sha256"],
            "dataset_fingerprint_sha256": dataset_manifest["manifest_fingerprint_sha256"],
        },
        "handling_classification": dataset_manifest.get("handling_classification", "private"),
        "association_boundary": "No association, custody, or track-management decision is performed by this packet.",
    }
    packet["packet_fingerprint_sha256"] = _digest(
        {
            "solution": solution["fingerprint_sha256"],
            "dataset": dataset_manifest["manifest_fingerprint_sha256"],
            "commands": packet["reproduction"]["commands"],
            "artifacts": artifact_rows,
        }
    )
    if output_path is not None:
        write_json(str(Path(output_path).expanduser()), packet)
    return packet


def check_od_evidence_freshness(
    packet: Mapping[str, Any],
    *,
    input_fingerprints: Mapping[str, str],
    estimator: Mapping[str, Any],
    propagator: Mapping[str, Any],
    frame_policy: Mapping[str, Any],
) -> dict[str, Any]:
    solution = dict(packet.get("solution", {}) or {})
    prior = dict(solution.get("fingerprint_components", {}) or {})
    current = {
        "inputs": dict(sorted((str(key), str(value)) for key, value in input_fingerprints.items())),
        "estimator": dict(estimator),
        "propagator": dict(propagator),
        "frame_policy": dict(frame_policy),
        "report_schema_version": prior.get("report_schema_version"),
        "capability_id": prior.get("capability_id"),
    }
    changed = [key for key in current if current[key] != prior.get(key)]
    return {
        "status": "current" if not changed else "stale",
        "changed_components": changed,
        "prior_fingerprint_sha256": solution.get("fingerprint_sha256"),
        "current_fingerprint_sha256": _digest(current),
        "usable_without_recompute": not changed,
    }


def compare_od_execution_parity(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    *,
    fields: Mapping[str, float],
) -> dict[str, Any]:
    rows = []
    for field, tolerance in fields.items():
        left_value = _nested_value(left, field)
        right_value = _nested_value(right, field)
        if isinstance(left_value, (list, tuple)) or isinstance(right_value, (list, tuple)):
            difference = float(
                np.max(np.abs(np.asarray(left_value, dtype=float) - np.asarray(right_value, dtype=float)))
            )
        else:
            difference = abs(float(left_value) - float(right_value))
        rows.append(
            {
                "field": field,
                "left": left_value,
                "right": right_value,
                "absolute_max_difference": difference,
                "tolerance": float(tolerance),
                "passed": difference <= float(tolerance),
            }
        )
    return {
        "passed": all(row["passed"] for row in rows),
        "rows": rows,
        "note": "Performance timing is intentionally excluded from physics/output parity.",
    }


def write_standard_od_plots(report: Mapping[str, Any], output_dir: str | Path) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output_dir).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    records = _plot_records(report)
    if not records:
        raise ValueError("report does not contain sequential records usable by standard OD plots.")
    time = np.asarray([float(row["time_s"]) for row in records])
    nis = np.asarray([float(row.get("nis", np.nan)) for row in records])
    covariance_trace = np.asarray(
        [float(np.trace(np.asarray(row.get("filtered_covariance", row.get("covariance")), dtype=float))) for row in records]
    )
    state_error = np.asarray([_record_position_error_m(row) for row in records])
    maneuver = np.asarray([bool(row.get("maneuver_confirmed", False)) for row in records])
    paths = {}

    def save(name: str, ylabel: str, values: np.ndarray, *, markers: bool = False) -> None:
        fig, axis = plt.subplots(figsize=(8.0, 4.5), constrained_layout=True)
        axis.plot(time, values, color="#26c6da", linewidth=1.8)
        if markers and np.any(maneuver):
            axis.scatter(time[maneuver], values[maneuver], color="#ffb300", marker="x", s=65, label="confirmed")
            axis.legend(loc="best")
        axis.set_xlabel("Time (s)")
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.25)
        path = output / f"od_{name}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths[name] = str(path)

    save("innovation", "NIS", nis)
    save("state_error", "Position error (m)", state_error)
    save("covariance", "State covariance trace", covariance_trace)
    save("maneuver", "NIS", nis, markers=True)
    return paths


def _plot_records(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    direct = list(report.get("records", []) or [])
    if direct:
        return [dict(row) for row in direct]
    sequential = dict(report.get("sequential_model_comparison", {}) or {})
    selected = report.get("recommended_model_inside_validated_envelope") or report.get(
        "selected_model_by_withheld_nonlinear_position_rms"
    )
    if selected in sequential:
        return [dict(row) for row in sequential[selected].get("records", [])]
    if sequential:
        return [dict(row) for row in next(iter(sequential.values())).get("records", [])]
    return []


def _record_position_error_m(record: Mapping[str, Any]) -> float:
    truth = record.get("truth_state")
    state = record.get("filtered_state", record.get("state"))
    if truth is None or state is None:
        return float("nan")
    return float(1000.0 * np.linalg.norm(np.asarray(state, dtype=float)[:3] - np.asarray(truth, dtype=float)[:3]))


def _nested_value(payload: Mapping[str, Any], field: str) -> Any:
    value: Any = payload
    for component in str(field).split("."):
        if not isinstance(value, Mapping) or component not in value:
            raise KeyError(f"parity field {field!r} is missing at component {component!r}.")
        value = value[component]
    return value


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
