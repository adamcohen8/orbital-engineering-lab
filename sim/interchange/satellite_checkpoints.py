"""Versioned, restartable GNC v2 satellite continuation products."""

from __future__ import annotations

import base64
import gzip
import hashlib
import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

from sim.flight_software import FlightSoftwareSnapshot, NativeFlightSoftwareSnapshot, from_primitive
from sim.interchange.completed_runs import build_completed_run_state_product
from sim.interchange.manifests import bind_output_path, write_json_document
from sim.interchange.materialization import (
    _continued_object_specs,
    _ensure_distinct_materialization_paths,
    _manifest_base,
    materialize_scenario_document,
)
from sim.interchange.provenance import canonical_json_bytes, compute_product_id, sha256_file
from sim.interchange.validation import load_interchange_document, validate_product
from sim.review import ReviewWorkspace

SATELLITE_CHECKPOINT_ADAPTER_ID = "oel.completed_run.satellite_checkpoint_export"
SATELLITE_CHECKPOINT_ADAPTER_VERSION = "1"
SATELLITE_CHECKPOINT_MATERIALIZER_ID = "oel.satellite_checkpoint_to_scenario"
SATELLITE_CHECKPOINT_MATERIALIZER_VERSION = "1"


class SatelliteCheckpointError(ValueError):
    """Raised when a complete satellite continuation cannot be represented safely."""


def export_satellite_checkpoint(
    completed_run: str | Path,
    *,
    output_path: str | Path,
    logical_output_path: str | Path | None = None,
    writer: Any | None = None,
    export_created_utc: str | None = None,
    object_id: str,
    context_object_ids: Sequence[str] = (),
    overwrite: bool = False,
) -> dict[str, Any]:
    """Export the final truth state plus an opaque, hash-bound GNC v2 checkpoint."""

    target, logical_target = bind_output_path(
        output_path,
        logical_output_path,
        writer=writer,
    )
    object_key = str(object_id or "").strip()
    context_ids = [str(item).strip() for item in context_object_ids]
    if not object_key or any(not item for item in context_ids):
        raise SatelliteCheckpointError("object_id and every context object ID must be non-empty.")
    if object_key in context_ids or len(context_ids) != len(set(context_ids)):
        raise SatelliteCheckpointError("Context object IDs must be unique and exclude the active satellite.")

    state_product = build_completed_run_state_product(
        completed_run,
        output_path=target,
        logical_output_path=logical_target,
        writer=writer,
        export_created_utc=export_created_utc,
        object_id=object_key,
        selector="final",
    )
    workspace = ReviewWorkspace.open(completed_run)
    metadata = _one_row(
        workspace.query(
            "SELECT run_id, config_json, config_sha256 FROM run_metadata",
            max_rows=2,
        ),
        "run_metadata",
    )
    config = _verified_config(metadata)
    object_config = _configured_object(config, object_key)
    flight_software = deepcopy(dict(object_config.get("flight_software", {}) or {}))
    if not flight_software:
        raise SatelliteCheckpointError("The selected satellite has no complete flight_software configuration.")
    flight_software.pop("checkpoint", None)
    _freeze_flight_software_defaults(
        flight_software,
        object_config=object_config,
        simulator=dict(config.get("simulator", {}) or {}),
    )

    state_payload = dict(state_product["payload"])
    selection = deepcopy(dict(state_payload["selection"]))
    selected_time_ns = int(round(float(selection["time_s"]) * 1.0e9))
    truth_row = _selected_truth_row(workspace, object_key, int(selection["sample_index"]))
    checkpoint = _selected_checkpoint(workspace, object_key, selected_time_ns)
    snapshot_raw = checkpoint["fsw_snapshot"]
    if isinstance(snapshot_raw, dict) and snapshot_raw.get("schema") == "oel.flight_software.native_snapshot.v1":
        snapshot = from_primitive(NativeFlightSoftwareSnapshot, snapshot_raw)
    else:
        snapshot = from_primitive(FlightSoftwareSnapshot, snapshot_raw)
    configured_stack = str(flight_software.get("stack", "") or "")
    if configured_stack == "fsw.game_pilot_reference":
        raise SatelliteCheckpointError(
            "Game-pilot continuation requires the surrounding interactive game session and is not supported by "
            "scenario-only satellite checkpoints."
        )
    if configured_stack and configured_stack != snapshot.stack_id:
        raise SatelliteCheckpointError("Recorded checkpoint stack does not match the verified source configuration.")
    runtime_state = _decode_runtime_state(checkpoint)
    runtime_profile_params = runtime_state.get("profile_params")
    if not isinstance(runtime_profile_params, dict):
        raise SatelliteCheckpointError(
            "Runtime checkpoint does not bind its effective flight-software parameters."
        )
    flight_software["params"] = deepcopy(runtime_profile_params)
    if int(runtime_state.get("external_publisher_count", 0)) != 0:
        raise SatelliteCheckpointError(
            "The selected runtime depends on external input publishers and cannot be continued by scenario materialization."
        )
    if int(runtime_state.get("checkpoint_time_ns", -1)) != int(checkpoint["checkpoint_time_ns"]):
        raise SatelliteCheckpointError("Runtime and checkpoint clock identities do not match.")

    context_states = [
        _context_state(
            completed_run,
            target,
            context_id,
            int(selection["sample_index"]),
            logical_output_path=logical_target,
            writer=writer,
        )
        for context_id in context_ids
    ]
    selection["truth_row_sha256"] = hashlib.sha256(canonical_json_bytes(truth_row)).hexdigest()
    selection["checkpoint_row_sha256"] = hashlib.sha256(canonical_json_bytes(checkpoint)).hexdigest()
    model_assumptions = deepcopy(dict(state_payload["model_assumptions"]))
    simulator = dict(config.get("simulator", {}) or {})
    dynamics = dict(simulator.get("dynamics", {}) or {})
    model_assumptions["attitude_dynamics"] = deepcopy(dict(dynamics.get("attitude", {}) or {}))
    force = dict(model_assumptions.get("orbit_force_model", {}) or {})
    model_assumptions["environment"] = deepcopy(dict(force.pop("environment", {}) or {}))
    model_assumptions["orbit_force_model"] = force

    product: dict[str, Any] = {
        key: deepcopy(state_product[key])
        for key in (
            "schema_id",
            "schema_version",
            "created_utc",
            "producer",
            "freshness",
            "provenance",
            "data_markings",
        )
    }
    product.update(
        {
            "product_kind": "oel.satellite_checkpoint",
            "product_id": "oel.satellite_checkpoint:" + "0" * 64,
            "payload": {
                "object": deepcopy(state_payload["object"]),
                "state": deepcopy(state_payload["state"]),
                "covariance": deepcopy(state_payload["covariance"]),
                "attitude": {
                    "quaternion_bn": [float(truth_row[name]) for name in ("quat_w", "quat_x", "quat_y", "quat_z")],
                    "angular_rate_body_rad_s": [
                        float(truth_row[name]) for name in ("omega_x_rad_s", "omega_y_rad_s", "omega_z_rad_s")
                    ],
                },
                "resource_state": deepcopy(state_payload["resource_state"]),
                "object_specs": deepcopy(state_payload["object_specs"]),
                "flight_software": flight_software,
                "knowledge": deepcopy(dict(object_config.get("knowledge", {}) or {})),
                "checkpoint": checkpoint,
                "context_states": context_states,
                "model_assumptions": model_assumptions,
                "source_run": deepcopy(state_payload["source_run"]),
                "selection": selection,
            },
            "quality": {
                "disposition": "accepted",
                "producer_status": "restartable_satellite_checkpoint_selected",
                "gates": {
                    "truth_state_complete": True,
                    "mass_and_propellant_bound": True,
                    "flight_software_state_hash_verified": True,
                    "runtime_state_hash_verified": True,
                    "stack_identity_matches_config": True,
                    "external_publishers_absent": True,
                    "adapter": {
                        "adapter_id": SATELLITE_CHECKPOINT_ADAPTER_ID,
                        "adapter_version": SATELLITE_CHECKPOINT_ADAPTER_VERSION,
                    },
                },
                "warnings": [
                    "Context objects are materialized as passive truth-state references; their onboard software is not restored."
                ] if context_states else [],
                "non_claims": [
                    "Checkpoint compatibility does not establish flight-software correctness or operational suitability.",
                    "Export does not execute or mutate a scenario.",
                ],
            },
        }
    )
    product["producer"]["capability_id"] = "satellite_checkpoint_export"
    product["provenance"]["transformations"] = [
        {
            "transformation_id": "review_store_to_satellite_checkpoint",
            "version": "1",
            "details": {
                "object_id": object_key,
                "context_object_ids": context_ids,
                "sample_index": selection["sample_index"],
                "time_s": selection["time_s"],
                "checkpoint_time_ns": checkpoint["checkpoint_time_ns"],
            },
        }
    ]
    product["product_id"] = compute_product_id(product)
    report = validate_product(product, source_path=logical_target)
    if not report.valid:
        messages = "; ".join(f"{item.path}: {item.message}" for item in report.errors)
        raise SatelliteCheckpointError(f"Generated satellite checkpoint failed validation: {messages}")
    try:
        write_json_document(product, target, writer=writer, overwrite=overwrite)
    except FileExistsError as exc:
        raise SatelliteCheckpointError(
            "Checkpoint output exists with different content; pass overwrite=True explicitly to replace it."
        ) from exc
    return {
        "status": "exported",
        "product_path": str(target),
        "product_id": product["product_id"],
        "object_id": object_key,
        "context_object_ids": context_ids,
        "sample_index": selection["sample_index"],
        "time_s": selection["time_s"],
        "execution_occurred": False,
    }


def materialize_satellite_checkpoint(
    checkpoint_product: str | Path,
    *,
    scenario_name: str,
    scenario_path: str | Path,
    output_dir: str | Path,
    duration_s: float,
    dt_s: float,
    logical_scenario_path: str | Path | None = None,
    logical_manifest_path: str | Path | None = None,
    logical_output_dir: str | Path | None = None,
    trust_plugins: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Materialize a restartable active-satellite scenario without executing it."""

    source = Path(checkpoint_product).expanduser().resolve()
    destination = Path(scenario_path).expanduser().resolve()
    manifest_target = destination.with_name(f"{destination.stem}.handoff_manifest.json")
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
        (("source product", source), ("scenario", destination), ("manifest", manifest_target)),
        SatelliteCheckpointError,
    )
    if not str(scenario_name or "").strip() or float(duration_s) <= 0.0 or float(dt_s) <= 0.0:
        raise SatelliteCheckpointError("scenario_name and positive duration_s/dt_s are required.")
    product = load_interchange_document(source)
    report = validate_product(product, source_path=source)
    if product.get("product_kind") != "oel.satellite_checkpoint" or not report.promotable:
        raise SatelliteCheckpointError("Satellite materialization requires a promotable satellite checkpoint product.")
    payload = dict(product["payload"])
    scenario = _checkpoint_scenario(
        product,
        scenario_name=str(scenario_name).strip(),
        output_dir=logical_output,
        duration_s=float(duration_s),
        dt_s=float(dt_s),
        manifest_path=logical_manifest_target,
    )
    base = _manifest_base(
        created_utc=str(product["created_utc"]),
        source_id=str(product["product_id"]),
        source_hash=sha256_file(source),
        product=product,
        scenario_name=scenario_name,
        destination=logical_destination,
        output_dir=logical_output,
        duration_s=duration_s,
        dt_s=dt_s,
        overwrite=overwrite,
    )
    base["adapter"] = {
        "adapter_id": SATELLITE_CHECKPOINT_MATERIALIZER_ID,
        "adapter_version": SATELLITE_CHECKPOINT_MATERIALIZER_VERSION,
    }
    base["defaults_applied"] = {
        "continuation_posture": "active_satellite_checkpoint",
        "context_posture": "passive" if payload.get("context_states") else "none",
        "review_detail": "standard",
        "plots_enabled": False,
        "animations_enabled": False,
    }
    base["materialization_options"]["source_run"] = deepcopy(payload["source_run"])
    base["materialization_options"]["selection"] = deepcopy(payload["selection"])
    base["source_hashes"]["completed_run_review_store"] = str(payload["source_run"]["review_db_sha256"])
    base["output"]["kind"] = "satellite_checkpoint_scenario"
    return materialize_scenario_document(
        scenario=scenario,
        destination=destination,
        manifest_target=manifest_target,
        protected_paths=(("source product", source),),
        collision_error_type=SatelliteCheckpointError,
        logical_destination=logical_destination,
        logical_manifest_target=logical_manifest_target,
        logical_output_dir=logical_output,
        base_manifest=base,
        product_report=report.to_dict(),
        output_kind="satellite_checkpoint_scenario",
        trust_plugins=trust_plugins,
        overwrite=overwrite,
    )


def _checkpoint_scenario(
    product: Mapping[str, Any],
    *,
    scenario_name: str,
    output_dir: str | Path,
    duration_s: float,
    dt_s: float,
    manifest_path: Path,
) -> dict[str, Any]:
    payload = dict(product["payload"])
    obj = dict(payload["object"])
    state = dict(payload["state"])
    values = [float(value) for value in state["values"]]
    attitude = dict(payload["attitude"])
    assumptions = dict(payload["model_assumptions"])
    force = deepcopy(dict(assumptions.get("orbit_force_model", {}) or {}))
    source_substep = force.get("orbit_substep_s")
    if source_substep is None or float(source_substep) > dt_s:
        force["orbit_substep_s"] = dt_s
    flight_software = deepcopy(dict(payload["flight_software"]))
    flight_software["checkpoint"] = deepcopy(dict(payload["checkpoint"]))
    object_id = str(obj["object_id"])
    objects: dict[str, Any] = {
        object_id: {
            "enabled": True,
            "role": str(obj["role"]),
            "kind": "satellite",
            "specs": _continued_object_specs(payload),
            "initial_state": {
                "position_eci_km": values[:3],
                "velocity_eci_km_s": values[3:],
                "attitude_quat_bn": list(attitude["quaternion_bn"]),
                "angular_rate_body_rad_s": list(attitude["angular_rate_body_rad_s"]),
            },
            "flight_software": flight_software,
            "knowledge": deepcopy(dict(payload.get("knowledge", {}) or {})),
        }
    }
    for context in list(payload.get("context_states", []) or []):
        item = dict(context)
        context_obj = dict(item["object"])
        context_state = dict(item["state"])
        context_values = [float(value) for value in context_state["values"]]
        context_attitude = dict(item["attitude"])
        objects[str(context_obj["object_id"])] = {
            "enabled": True,
            "role": str(context_obj["role"]),
            "kind": "satellite",
            "specs": _continued_object_specs(item),
            "initial_state": {
                "position_eci_km": context_values[:3],
                "velocity_eci_km_s": context_values[3:],
                "attitude_quat_bn": list(context_attitude["quaternion_bn"]),
                "angular_rate_body_rad_s": list(context_attitude["angular_rate_body_rad_s"]),
            },
            "flight_software": {"stack": "fsw.passive", "hardware_profile": "hardware.passive.v1"},
        }
    return {
        "scenario_name": scenario_name,
        "scenario_description": f"Active satellite continuation materialized from {product['product_id']}.",
        "metadata": {
            "owner": "oel-handoff",
            "handoff": {
                "source_product_id": product["product_id"],
                "adapter_id": SATELLITE_CHECKPOINT_MATERIALIZER_ID,
                "adapter_version": SATELLITE_CHECKPOINT_MATERIALIZER_VERSION,
                "manifest_path": str(manifest_path),
                "continuation_posture": "active_satellite_checkpoint",
                "execution_occurred": False,
            },
        },
        "objects": objects,
        "simulator": {
            "initial_jd_utc": float(dict(state["epoch"])["value"]),
            "duration_s": duration_s,
            "dt_s": dt_s,
            "dynamics": {
                "orbit": force,
                "attitude": deepcopy(dict(assumptions.get("attitude_dynamics", {}) or {})),
                "rocket": {"enabled": False},
            },
            "environment": deepcopy(dict(assumptions.get("environment", {}) or {})),
            "plugin_validation": {"strict": True},
            "termination": {"earth_impact_enabled": False},
        },
        "outputs": {
            "output_dir": str(output_dir),
            "mode": "save",
            "stats": {"enabled": True, "print_summary": False, "save_json": True, "save_full_log": False},
            "review": {"enabled": True, "detail": "standard"},
            "plots": {"enabled": False, "figure_ids": []},
            "animations": {"enabled": False, "types": []},
        },
    }


def _context_state(
    completed_run: str | Path,
    target: Path,
    object_id: str,
    sample_index: int,
    *,
    logical_output_path: str | Path | None = None,
    writer: Any | None = None,
) -> dict[str, Any]:
    product = build_completed_run_state_product(
        completed_run,
        output_path=target,
        logical_output_path=logical_output_path,
        writer=writer,
        object_id=object_id,
        selector="sample_index",
        sample_index=sample_index,
    )
    workspace = ReviewWorkspace.open(completed_run)
    truth = _selected_truth_row(workspace, object_id, sample_index)
    payload = dict(product["payload"])
    return {
        "object": deepcopy(payload["object"]),
        "state": deepcopy(payload["state"]),
        "attitude": {
            "quaternion_bn": [float(truth[name]) for name in ("quat_w", "quat_x", "quat_y", "quat_z")],
            "angular_rate_body_rad_s": [
                float(truth[name]) for name in ("omega_x_rad_s", "omega_y_rad_s", "omega_z_rad_s")
            ],
        },
        "resource_state": deepcopy(payload["resource_state"]),
        "object_specs": deepcopy(payload["object_specs"]),
        "state_row_sha256": str(dict(payload["selection"])["state_row_sha256"]),
    }


def _selected_truth_row(workspace: ReviewWorkspace, object_id: str, sample_index: int) -> dict[str, Any]:
    result = workspace.query(
        "SELECT sample_index, time_s, object_id, pos_x_eci_km, pos_y_eci_km, pos_z_eci_km, "
        "vel_x_eci_km_s, vel_y_eci_km_s, vel_z_eci_km_s, quat_w, quat_x, quat_y, quat_z, "
        "omega_x_rad_s, omega_y_rad_s, omega_z_rad_s, mass_kg "
        "FROM object_state WHERE object_id = ? AND sample_index = ?",
        (object_id, sample_index),
        max_rows=2,
    )
    row = _one_row(result, "object_state")
    for name, value in row.items():
        if name == "object_id":
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise SatelliteCheckpointError(f"Selected truth field {name} must be finite.")
    return row


def _selected_checkpoint(
    workspace: ReviewWorkspace,
    object_id: str,
    selected_time_ns: int,
) -> dict[str, Any]:
    snapshot_columns = {
        str(item["name"]) for item in workspace.table_columns().get("fsw_snapshots", [])
    }
    detail_columns = ", detail_gzip" if "detail_gzip" in snapshot_columns else ""
    result = workspace.query(
        "SELECT invocation_id, stack_id, stack_version, state_hash_sha256, detail_json"
        f"{detail_columns} FROM fsw_snapshots WHERE object_id = ? ORDER BY invocation_id DESC",
        (object_id,),
        max_rows=10001,
    )
    if result.truncated:
        raise SatelliteCheckpointError("FSW checkpoint evidence exceeded the bounded selection limit.")
    candidates: list[dict[str, Any]] = []
    for row in result.rows:
        try:
            if row.get("detail_json") is not None:
                raw_detail = str(row["detail_json"])
            elif row.get("detail_gzip") is not None:
                raw_detail = gzip.decompress(bytes(row["detail_gzip"])).decode("utf-8")
            else:
                continue
            detail = json.loads(raw_detail)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if isinstance(detail, dict) and int(detail.get("run_time_ns", -1)) == selected_time_ns:
            candidates.append(detail)
    if not candidates and result.row_count:
        raise SatelliteCheckpointError(
            "The review store contains hash-only legacy FSW evidence, not restartable state."
        )
    if len(candidates) != 1:
        raise SatelliteCheckpointError(
            "Final satellite continuation requires exactly one full runtime checkpoint at the selected sample."
        )
    checkpoint = candidates[0]
    required = {
        "checkpoint_schema",
        "checkpoint_time_ns",
        "run_time_ns",
        "implementation_hash",
        "fsw_snapshot",
        "runtime_state_bytes_base64",
        "runtime_state_hash_sha256",
    }
    if not required.issubset(checkpoint):
        raise SatelliteCheckpointError("The review store contains hash-only legacy FSW evidence, not restartable state.")
    return checkpoint


def _decode_runtime_state(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    encoded = checkpoint.get("runtime_state_bytes_base64")
    if not isinstance(encoded, str):
        raise SatelliteCheckpointError("Runtime checkpoint state is not base64 text.")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (ValueError, TypeError) as exc:
        raise SatelliteCheckpointError("Runtime checkpoint contains invalid base64.") from exc
    if hashlib.sha256(raw).hexdigest() != checkpoint.get("runtime_state_hash_sha256"):
        raise SatelliteCheckpointError("Runtime checkpoint hash does not match its state bytes.")
    try:
        state = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SatelliteCheckpointError("Runtime checkpoint state is invalid JSON.") from exc
    if not isinstance(state, dict) or state.get("schema") != "oel.satellite_runtime_state.v1":
        raise SatelliteCheckpointError("Runtime checkpoint state schema is unsupported.")
    return state


def _verified_config(metadata: Mapping[str, Any]) -> dict[str, Any]:
    text = str(metadata.get("config_json", "") or "")
    if hashlib.sha256(text.encode("utf-8")).hexdigest() != str(metadata.get("config_sha256", "") or ""):
        raise SatelliteCheckpointError("run_metadata config_json does not match config_sha256.")
    try:
        config = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SatelliteCheckpointError("run_metadata config_json is invalid JSON.") from exc
    if not isinstance(config, dict):
        raise SatelliteCheckpointError("run_metadata config_json must contain an object.")
    return config


def _configured_object(config: Mapping[str, Any], object_id: str) -> dict[str, Any]:
    obj = dict(dict(config.get("objects", {}) or {}).get(object_id, {}) or {})
    if not obj:
        raise SatelliteCheckpointError(f"Verified source config does not contain object {object_id!r}.")
    return obj


def _freeze_flight_software_defaults(
    flight_software: dict[str, Any],
    *,
    object_config: Mapping[str, Any],
    simulator: Mapping[str, Any],
) -> None:
    """Make source-run defaults explicit when continuation changes physical mass/cadence."""

    if flight_software.get("task_period_s") is None:
        flight_software["task_period_s"] = float(simulator.get("dt_s", 1.0))
    stack_id = str(flight_software.get("stack", "") or "")
    if stack_id not in {"fsw.orbit_reference", "fsw.rpo_reference", "fsw.low_thrust_reference"}:
        return
    specs = dict(object_config.get("specs", {}) or {})
    if "dry_mass_kg" in specs or "fuel_mass_kg" in specs:
        initial_mass_kg = float(specs.get("dry_mass_kg", 0.0)) + float(specs.get("fuel_mass_kg", 0.0))
    else:
        initial_mass_kg = float(specs.get("mass_kg", 300.0))
    params = dict(flight_software.get("params", {}) or {})
    params.setdefault("assumed_mass_kg", initial_mass_kg)
    max_acceleration = float(params.get("max_acceleration_m_s2", 0.02))
    params.setdefault("max_force_n", max(max_acceleration * initial_mass_kg, 1.0e-9))
    flight_software["params"] = params


def _one_row(result: Any, label: str) -> dict[str, Any]:
    if result.truncated or result.row_count != 1:
        raise SatelliteCheckpointError(f"{label} must contain exactly one bounded row.")
    return dict(result.rows[0])


__all__ = [
    "SATELLITE_CHECKPOINT_ADAPTER_ID",
    "SATELLITE_CHECKPOINT_ADAPTER_VERSION",
    "SATELLITE_CHECKPOINT_MATERIALIZER_ID",
    "SATELLITE_CHECKPOINT_MATERIALIZER_VERSION",
    "SatelliteCheckpointError",
    "export_satellite_checkpoint",
    "materialize_satellite_checkpoint",
]
