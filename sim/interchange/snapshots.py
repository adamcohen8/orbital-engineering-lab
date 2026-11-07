from __future__ import annotations

import hashlib
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from sim.interchange.completed_runs import build_completed_run_state_product
from sim.interchange.manifests import bind_output_path, write_json_document
from sim.interchange.materialization import (
    _ensure_distinct_materialization_paths,
    _manifest_base,
    _onp_cadence_override,
    _onp_compatibility_errors,
    build_onp_scenario,
    materialize_scenario_document,
)
from sim.interchange.provenance import canonical_json_bytes, compute_product_id, sha256_file
from sim.interchange.validation import load_interchange_document, validate_product
from sim.review import ReviewWorkspace


class CompletedRunSnapshotError(ValueError):
    """Raised when an atomic multi-object continuation cannot be represented."""


def export_completed_run_snapshot(
    completed_run: str | Path,
    *,
    output_path: str | Path,
    logical_output_path: str | Path | None = None,
    writer: Any | None = None,
    export_created_utc: str | None = None,
    protected: bool | None = None,
    object_ids: Sequence[str],
    selector: str = "final",
    sample_index: int | None = None,
    time_s: float | None = None,
    event_id: str | None = None,
    epoch_jd_utc: float | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    target, logical_target = bind_output_path(
        output_path,
        logical_output_path,
        writer=writer,
    )
    ids = [str(item).strip() for item in object_ids]
    if len(ids) < 2 or any(not item for item in ids) or len(set(ids)) != len(ids):
        raise CompletedRunSnapshotError("object_ids must contain at least two unique non-empty IDs.")
    workspace = ReviewWorkspace.open(completed_run, protected=protected)
    workspace.freeze_source()
    event: dict[str, Any] | None = None
    effective_selector = selector
    effective_sample_index = sample_index
    if selector == "event":
        result = workspace.query(
            "SELECT event_id, time_s, sample_index, object_id, event_type, severity, message, source "
            "FROM events WHERE event_id = ?",
            (str(event_id or ""),),
            max_rows=2,
        )
        if result.truncated or result.row_count != 1 or result.rows[0].get("sample_index") is None:
            raise CompletedRunSnapshotError("Event selector must identify one sample-associated event.")
        event = dict(result.rows[0])
        event["sample_index"] = int(event["sample_index"])
        event["time_s"] = float(event["time_s"])
        event["object_id"] = None if event.get("object_id") in {None, ""} else str(event["object_id"])
        event["event_row_sha256"] = hashlib.sha256(canonical_json_bytes(event)).hexdigest()
        effective_selector = "sample_index"
        effective_sample_index = int(event["sample_index"])
    products = [
        build_completed_run_state_product(
            completed_run,
            output_path=target,
            logical_output_path=logical_target,
            writer=writer,
            export_created_utc=export_created_utc,
            protected=protected,
            object_id=object_id,
            selector=effective_selector,
            sample_index=effective_sample_index,
            time_s=time_s,
            event_id=None,
            epoch_jd_utc=epoch_jd_utc,
        )
        for object_id in ids
    ]
    selections = [dict(dict(item["payload"])["selection"]) for item in products]
    selected_indices = {int(item["sample_index"]) for item in selections}
    selected_times = {float(item["time_s"]) for item in selections}
    epochs = {
        float(dict(dict(dict(item["payload"])["state"])["epoch"])["value"])
        for item in products
    }
    if len(selected_indices) != 1 or len(selected_times) != 1 or len(epochs) != 1:
        raise CompletedRunSnapshotError(
            "Selected object states do not share one exact sample, time, and epoch; choose an exact common selector."
        )
    selected_index = next(iter(selected_indices))
    selected_time = next(iter(selected_times))
    relative_rows = workspace.query(
        "SELECT deputy_id, chief_id, r_radial_km, i_intrack_km, c_crosstrack_km, "
        "v_radial_km_s, v_intrack_km_s, v_crosstrack_km_s, range_km, range_rate_km_s "
        "FROM relative_state WHERE sample_index = ? ORDER BY chief_id, deputy_id",
        (selected_index,),
        max_rows=10001,
    )
    if relative_rows.truncated:
        raise CompletedRunSnapshotError("Relative-pair snapshot exceeded the bounded export limit.")
    selected_set = set(ids)
    pairs = [
        dict(row)
        for row in relative_rows.rows
        if str(row["chief_id"]) in selected_set and str(row["deputy_id"]) in selected_set
    ]
    first = products[0]
    first_payload = dict(first["payload"])
    common_selection = deepcopy(selections[0])
    if event is not None:
        common_selection.update(
            {
                "selector_kind": "event",
                "requested": {"event_id": str(event_id)},
                "associated_event": event,
            }
        )
    common_selection["state_rows_sha256"] = {
        object_id: str(selection.pop("state_row_sha256"))
        for object_id, selection in zip(ids, selections)
    }
    common_selection.pop("state_row_sha256", None)
    states = []
    for item in products:
        payload = dict(item["payload"])
        states.append(
            {
                key: deepcopy(payload[key])
                for key in (
                    "object",
                    "state",
                    "covariance",
                    "object_specs",
                    "resource_state",
                    "model_assumptions",
                )
            }
        )
    product: dict[str, Any] = {
        key: deepcopy(first[key])
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
            "product_kind": "oel.completed_run_snapshot",
            "product_id": "oel.completed_run_snapshot:" + "0" * 64,
            "payload": {
                "states": states,
                "relative_pairs": pairs,
                "source_run": deepcopy(first_payload["source_run"]),
                "selection": common_selection,
            },
            "quality": {
                "disposition": "accepted",
                "producer_status": "atomic_completed_run_sample_selected",
                "gates": {
                    "object_count": len(states),
                    "sample_unambiguous": True,
                    "common_sample_index": selected_index,
                    "common_time_s": selected_time,
                    "common_epoch": True,
                    "relative_pair_count": len(pairs),
                },
                "warnings": [
                    "Controller, estimator, attitude, and mission-module memory are not continued."
                ],
                "non_claims": [
                    "This snapshot contains selected simulator truth states, not a new orbit determination result.",
                    "Snapshot export does not execute or mutate a scenario.",
                ],
            },
        }
    )
    product["producer"]["capability_id"] = "completed_run_snapshot_export"
    product["provenance"]["transformations"] = [
        {
            "transformation_id": "review_store_atomic_object_state_snapshot",
            "version": "1",
            "details": {
                "object_ids": ids,
                "sample_index": selected_index,
                "time_s": selected_time,
                "relative_pair_count": len(pairs),
            },
        }
    ]
    product["product_id"] = compute_product_id(product)
    report = validate_product(product, source_path=logical_target)
    if not report.valid:
        messages = "; ".join(f"{item.path}: {item.message}" for item in report.errors)
        raise CompletedRunSnapshotError(f"Generated snapshot failed validation: {messages}")
    try:
        write_json_document(product, target, writer=writer, overwrite=overwrite)
    except FileExistsError as exc:
        raise CompletedRunSnapshotError(
            "Snapshot output exists with different content; pass overwrite=True explicitly to replace it."
        ) from exc
    return {
        "status": "exported",
        "product_path": str(target),
        "product_id": product["product_id"],
        "object_ids": ids,
        "sample_index": selected_index,
        "time_s": selected_time,
        "relative_pair_count": len(pairs),
        "execution_occurred": False,
    }


def materialize_snapshot_onp(
    snapshot_product: str | Path,
    *,
    scenario_name: str,
    scenario_path: str | Path,
    output_dir: str | Path,
    duration_s: float,
    dt_s: float,
    logical_scenario_path: str | Path | None = None,
    logical_manifest_path: str | Path | None = None,
    logical_output_dir: str | Path | None = None,
    validation_path_policy: Any | None = None,
    trust_plugins: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    source = Path(snapshot_product).expanduser().resolve()
    destination = Path(scenario_path).expanduser().resolve()
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
    manifest_target = destination.with_name(f"{destination.stem}.handoff_manifest.json")
    _ensure_distinct_materialization_paths(
        (("source product", source), ("scenario", destination), ("manifest", manifest_target)),
        CompletedRunSnapshotError,
    )
    product = load_interchange_document(source)
    report = validate_product(product, source_path=source)
    if product.get("product_kind") != "oel.completed_run_snapshot" or not report.promotable:
        raise CompletedRunSnapshotError("ONP snapshot materialization requires a promotable snapshot product.")
    if not str(scenario_name).strip() or float(duration_s) <= 0.0 or float(dt_s) <= 0.0:
        raise CompletedRunSnapshotError("scenario_name and positive duration_s/dt_s are required.")
    states = list(dict(product["payload"])["states"])
    first_payload = dict(states[0])
    first_product = deepcopy(product)
    first_product["product_kind"] = "oel.completed_run_state"
    first_product["payload"] = {
        **first_payload,
        "source_run": deepcopy(dict(product["payload"])["source_run"]),
        "selection": _single_selection(dict(product["payload"])["selection"], first_payload),
    }
    compatibility_errors = _onp_compatibility_errors(first_product)
    if compatibility_errors:
        messages = "; ".join(str(item.get("message", item)) for item in compatibility_errors)
        raise CompletedRunSnapshotError(
            "Snapshot cannot be relabeled as an ECI ONP continuation: " + messages
        )
    scenario = build_onp_scenario(
        first_product,
        scenario_name=str(scenario_name),
        output_dir=logical_output,
        duration_s=float(duration_s),
        dt_s=float(dt_s),
        manifest_path=logical_manifest_target,
        adapter_id="oel.completed_run_snapshot_to_onp",
        adapter_version="1",
        scenario_description=f"Passive atomic continuation from {product['product_id']}.",
    )
    scenario["metadata"]["handoff"] = {
        "source_product_id": product["product_id"],
        "adapter_id": "oel.completed_run_snapshot_to_onp",
        "adapter_version": "1",
        "selection": deepcopy(dict(product["payload"])["selection"]),
        "relative_pairs": deepcopy(dict(product["payload"])["relative_pairs"]),
        "continuation_posture": "passive_zero_controller",
        "execution_occurred": False,
    }
    scenario["objects"] = {}
    for item_raw in states:
        item = dict(item_raw)
        obj = dict(item["object"])
        values = [float(value) for value in dict(item["state"])["values"]]
        scenario["objects"][str(obj["object_id"])] = {
            "enabled": True,
            "role": str(obj["role"]),
            "kind": "satellite",
            "specs": _snapshot_object_specs(item),
            "initial_state": {
                "position_eci_km": values[:3],
                "velocity_eci_km_s": values[3:],
            },
            "flight_software": {
                "stack": "fsw.passive",
                "hardware_profile": "hardware.passive.v1",
            },
        }
    base_manifest = _manifest_base(
        created_utc=datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
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
    base_manifest["adapter"] = {
        "adapter_id": "oel.completed_run_snapshot_to_onp",
        "adapter_version": "1",
    }
    base_manifest["materialization_options"]["object_count"] = len(states)
    cadence_override = _onp_cadence_override(first_product, dt_s=float(dt_s))
    if cadence_override:
        base_manifest["overrides"].append(cadence_override)
    base_manifest["output"]["kind"] = "snapshot_onp_scenario"
    result = materialize_scenario_document(
        scenario=scenario,
        destination=destination,
        manifest_target=manifest_target,
        protected_paths=(("source product", source),),
        collision_error_type=CompletedRunSnapshotError,
        logical_destination=logical_destination,
        logical_manifest_target=logical_manifest_target,
        logical_output_dir=logical_output,
        validation_path_policy=validation_path_policy,
        base_manifest=base_manifest,
        product_report=report.to_dict(),
        output_kind="snapshot_onp_scenario",
        trust_plugins=trust_plugins,
        overwrite=overwrite,
    )
    result["source_product_id"] = product["product_id"]
    result["object_count"] = len(states)
    return result


def _single_selection(selection: Mapping[str, Any], state_payload: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(selection))
    hashes = dict(result.pop("state_rows_sha256"))
    object_id = str(dict(state_payload["object"])["object_id"])
    result["state_row_sha256"] = str(hashes[object_id])
    return result


def _snapshot_object_specs(item: Mapping[str, Any]) -> dict[str, Any]:
    specs = deepcopy(dict(item.get("object_specs", {}) or {}))
    resource = dict(item.get("resource_state", {}) or {})
    if not resource:
        return specs
    specs["mass_kg"] = float(resource["mass_kg"])
    if resource.get("propellant_state") == "tracked":
        specs["dry_mass_kg"] = float(resource["dry_mass_kg"])
        specs["fuel_mass_kg"] = float(resource["fuel_mass_kg"])
    else:
        specs.pop("dry_mass_kg", None)
        specs.pop("fuel_mass_kg", None)
    return specs


__all__ = [
    "CompletedRunSnapshotError",
    "export_completed_run_snapshot",
    "materialize_snapshot_onp",
]
