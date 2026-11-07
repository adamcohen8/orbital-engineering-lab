"""Content binding for acquired inputs, using ordinary scenario metadata.

These are reproducibility receipts, not signatures or disclosure authorization.
Verification never acquires data or imports a configured plugin.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from .contracts import MAX_BYTES, AcquisitionError, canonical, digest, epoch, strict_json
from .files import read_snapshot

FILE_FIELDS = {
    "simulator.dynamics.orbit.spherical_harmonics.coeff_path": "earth-gravity",
    "simulator.frames.eop_path": "runtime-frames",
    "simulator.environment.nrlmsise00_sw_path": "nrlmsise00",
    "simulator.environment.msis86_sw_path": "msis86",
    "simulator.environment.atmosphere_env.nrlmsise00_sw_path": "nrlmsise00",
    "simulator.environment.atmosphere_env.msis86_sw_path": "msis86",
}


def field_value(root, field):
    value = root
    try:
        for part in field.split("."):
            value = value[part]
        return value
    except (KeyError, TypeError) as exc:
        raise AcquisitionError("BINDING_CHANGED", "A bound scenario field is missing.") from exc


def _scenario_requirements(root, requirements):
    simulator = root["simulator"]
    consumer = requirements["consumer"]
    if consumer == "earth-gravity":
        block = simulator["dynamics"]["orbit"]["spherical_harmonics"]
        if block.get("source") != "icgem" or block.get("degree") != requirements["degree"] or block.get("order", block["degree"]) != requirements["order"]:
            raise AcquisitionError("BINDING_CHANGED", "Gravity source/degree/order differs from its acquired-input check.")
    elif consumer != "ogp":
        from datetime import timedelta

        from sim.dynamics.orbit.epoch import julian_date_to_datetime
        if consumer in {"nrlmsise00", "msis86"}:
            environment = simulator.get("environment", {})
            if environment.get("atmosphere_model") != consumer:
                raise AcquisitionError("BINDING_CHANGED", "Select the checked atmosphere model in simulator.environment before binding weather.")
        elif consumer == "runtime-frames":
            from sim.dynamics.orbit.frames import FRAME_MODEL_IAU76_80_EOP, normalize_frame_model

            if normalize_frame_model(simulator.get("frames", {}).get("model", "simple")) != FRAME_MODEL_IAU76_80_EOP:
                raise AcquisitionError("BINDING_CHANGED", "Select the EOP-backed runtime frame model before binding EOP.")
        start = julian_date_to_datetime(float(simulator["initial_jd_utc"]))
        stop = start + timedelta(seconds=float(simulator["duration_s"]))
        tolerance = timedelta(microseconds=100)
        if start < epoch(requirements["start"]) - tolerance or stop > epoch(requirements["stop"]) + tolerance:
            raise AcquisitionError("BINDING_CHANGED", "Scenario time interval exceeds the checked acquired-input interval.")


def verify_bindings(config, *, path_policy=None):
    metadata = config.get("metadata", {})
    if not isinstance(metadata, dict) or "acquisition_inputs" not in metadata:
        return []
    entries = metadata["acquisition_inputs"]
    if not isinstance(entries, list) or len(entries) > 64:
        raise AcquisitionError("INVALID_BINDING", "Acquisition metadata must contain at most 64 input bindings.")
    lineage = metadata.get("acquisition_source_receipts", {})
    if not isinstance(lineage, dict) or len(lineage) > 256:
        raise AcquisitionError("INVALID_BINDING", "Too many or invalid source receipt records.")
    decoded_lineage = {}
    for identity, record in lineage.items():
        if not isinstance(record, dict) or set(record) != {"receipt_json", "sha256"} or not isinstance(record["receipt_json"], str) or digest(record["receipt_json"].encode()) != record["sha256"]:
            raise AcquisitionError("BINDING_CHANGED", "A retained source receipt changed.")
        original = strict_json(record["receipt_json"])
        if not isinstance(original, dict) or original.get("artifact_id") != identity:
            raise AcquisitionError("BINDING_CHANGED", "Source receipt identity changed.")
        decoded_lineage[identity] = original
    verified = []
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"field", "path", "sha256", "receipt", "value_sha256"}:
            raise AcquisitionError("INVALID_BINDING", "Invalid acquired-input binding contract.")
        field = entry["field"]
        expected = field_value(config, field)
        if field in FILE_FIELDS:
            if str(expected) != entry["path"]:
                raise AcquisitionError("BINDING_CHANGED", "Bound file path changed.")
        elif isinstance(field, str) and field.startswith("objects.") and field.endswith(".initial_state.ogp_mean_elements"):
            if len(field.split(".")) != 4 or digest(canonical(expected)) != entry["value_sha256"]:
                raise AcquisitionError("BINDING_CHANGED", "Selected catalog mean elements changed.")
            object_id = field.split(".")[1]
            if field_value(config, f"objects.{object_id}.propagation_method") != "general":
                raise AcquisitionError("BINDING_CHANGED", "Acquired mean elements require the selected OGP propagation path.")
        else:
            raise AcquisitionError("INVALID_BINDING", "This acquired-input field is not supported.")
        if path_policy is not None:
            _, raw = path_policy.read_input_bytes(entry["path"], purpose="bound acquired input", max_bytes=MAX_BYTES)
        else:
            raw = read_snapshot(Path(entry["path"]), MAX_BYTES)
        receipt = entry["receipt"]
        if not isinstance(receipt, dict) or receipt.get("schema") != "oel.acquisition-materialization.v1" or receipt.get("sha256") != entry["sha256"] or digest(raw) != entry["sha256"]:
            raise AcquisitionError("BINDING_CHANGED", "Bound acquired file or receipt identity changed.")
        if field in FILE_FIELDS:
            requirements = receipt.get("requirements")
            if not isinstance(requirements, dict) or requirements.get("consumer") != FILE_FIELDS[field]:
                raise AcquisitionError("INVALID_BINDING", "Materialize this input with the matching consumer requirements first.")
            _scenario_requirements(config, requirements)
        source = receipt["source_receipt"]
        if field not in FILE_FIELDS:
            from sim.dynamics.orbit.epoch import julian_date_to_datetime

            selection = source.get("selection", {})
            initial = julian_date_to_datetime(float(config["simulator"]["initial_jd_utc"]))
            if not selection.get("at") or abs((initial - epoch(selection["at"])).total_seconds()) > 0.0001:
                raise AcquisitionError("BINDING_CHANGED", "Scenario initial epoch differs from the catalog record-selection epoch.")
        for parent in source.get("parents", []):
            if parent["artifact_id"] in decoded_lineage and decoded_lineage[parent["artifact_id"]]["sha256"] != parent["sha256"]:
                raise AcquisitionError("BINDING_CHANGED", "A selected source identity differs from its original receipt.")
        verified.append({"field": field, "sha256": entry["sha256"], "source_sha256": receipt["source_sha256"],
                         "artifact_id": source["artifact_id"], "provider": source["provider"],
                         "terms": source.get("terms", {}), "source_receipt": source})
    return verified


def verify_config(config):
    if not getattr(config, "metadata", {}).get("acquisition_inputs"):
        return []
    return verify_bindings(config.to_dict())


def bind_file(config: dict, *, path, field, path_policy=None):
    if field not in FILE_FIELDS:
        raise AcquisitionError("INVALID_BINDING", "Select a documented gravity, EOP or MSIS file field.")
    path = Path(path).expanduser().absolute()
    receipt_path = path.with_name(path.name + ".acquisition.json")
    receipt = strict_json(read_snapshot(receipt_path, 2 * 1024 * 1024))
    result = deepcopy(config)
    section = result
    parts = field.split(".")
    for part in parts[:-1]:
        section = section.setdefault(part, {})
    section[parts[-1]] = str(path)
    metadata = result.setdefault("metadata", {})
    entries = metadata.setdefault("acquisition_inputs", [])
    if any(v.get("field") == field for v in entries):
        raise AcquisitionError("BINDING_EXISTS", "This field is already bound; create a new explicit study configuration.")
    entries.append({"field": field, "path": str(path), "sha256": receipt["sha256"], "receipt": receipt, "value_sha256": None})
    verify_bindings(result, path_policy=path_policy)
    return result


def bind_catalog_sources(scenario, sources):
    """Called by the existing IO materializer only for adjacent acquisition receipts."""
    for source in sources:
        path = Path(source).expanduser().absolute()
        adjacent = path.with_name(path.name + ".acquisition.json")
        if not adjacent.exists():
            continue
        receipt = strict_json(read_snapshot(adjacent, 2 * 1024 * 1024))
        origin = receipt.get("source_receipt", {})
        if origin.get("kind") != "catalog":
            continue
        if not origin.get("selection"):
            raise AcquisitionError("SELECTION_REQUIRED", "Run data select before materializing an acquired catalog study.")
        if digest(read_snapshot(path, MAX_BYTES)) != receipt.get("sha256"):
            raise AcquisitionError("BINDING_CHANGED", "Selected catalog file changed before scenario composition.")
        # Retain original queries and retrieval receipts once, rather than
        # duplicating potentially large history lineage for every object.
        metadata = scenario.setdefault("metadata", {})
        sources = metadata.setdefault("acquisition_source_receipts", {})
        for identity, original in origin.get("source_receipts", {}).items():
            # Canonical JSON survives the scenario serializer's pruning of
            # empty/None mapping values without altering receipt identity.
            serialized = canonical(original)
            record = {"receipt_json": serialized.decode(), "sha256": digest(serialized)}
            if identity in sources and sources[identity] != record:
                raise AcquisitionError("BINDING_CHANGED", "Conflicting source receipt identities.")
            sources[identity] = record
        receipt = deepcopy(receipt)
        receipt["source_receipt"].pop("source_receipts", None)
        entries = scenario.setdefault("metadata", {}).setdefault("acquisition_inputs", [])
        for oid, obj in scenario["objects"].items():
            entries.append({"field": f"objects.{oid}.initial_state.ogp_mean_elements", "path": str(path),
                            "sha256": receipt["sha256"], "receipt": receipt,
                            "value_sha256": digest(canonical(obj["initial_state"]["ogp_mean_elements"]))})
    verify_bindings(scenario)
    return scenario
