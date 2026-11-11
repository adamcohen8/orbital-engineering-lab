"""Versioned scientific evidence over recorded OEL samples."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass

from sim.review.workspace import ReviewQueryError, ReviewStoreNotFoundError, ReviewWorkspace
from sim.security.input_reads import SecureReadChanged

CONTRACT = "oel.review.scientific.v1"


@dataclass(frozen=True)
class Quantity:
    table: str
    column: str
    units: str
    definition: str
    frame_dependent: bool = True


QUANTITIES = {
    quantity.column: quantity
    for quantity in (
        *(Quantity("object_state", f"pos_{axis}_eci_km", "km", f"Cartesian position {axis.upper()} component")
          for axis in "xyz"),
        *(Quantity("object_state", f"vel_{axis}_eci_km_s", "km/s", f"Cartesian velocity {axis.upper()} component")
          for axis in "xyz"),
        Quantity("object_orbital_elements", "radius_km", "km", "Recorded position magnitude", False),
        Quantity("object_orbital_elements", "speed_km_s", "km/s", "Recorded velocity magnitude", False),
    )
}
_UNITS = {"km": ("length", 1000), "m": ("length", 1), "km/s": ("speed", 1000), "m/s": ("speed", 1)}


def _object(properties, required):
    return {"type": "object", "properties": properties, "required": list(required),
            "additionalProperties": False}


def operation_schema():
    """The same input contract serves Python, CLI and protocol adapters."""
    label = {"type": "string", "minLength": 1, "maxLength": 160}
    number = {"type": "number", "minimum": -1e300, "maximum": 1e300}
    quantity = {"type": "string", "enum": sorted(QUANTITIES)}
    quantities = {"type": "array", "items": quantity, "minItems": 1, "maxItems": len(QUANTITIES),
                  "uniqueItems": True}
    units = {"type": "string", "enum": sorted(_UNITS)}
    reference = _object({
        "source": {**label, "description": "Attribution for a supplied value; e.g. user question or dataset identifier."},
        "value": number, "units": units, "quantity_id": quantity, "object_id": label,
        "frame": label, "frame_model": label, "epoch_jd_utc": number,
    }, ("source", "value", "units", "quantity_id", "object_id", "epoch_jd_utc"))
    variants = {
        "describe": ({}, ()),
        "exact": ({"time_s": number, "sample_index": {"type": "integer", "minimum": 0, "maximum": 2**63 - 1}}, ()),
        "nearest": ({"time_s": number, "tie": {"enum": ["earlier", "later"]},
                     "max_separation_s": {"type": "number", "minimum": 0}}, ("time_s", "tie")),
        "bracket": ({"time_s": number}, ("time_s",)),
        "extremum": ({"start_s": number, "stop_s": number, "direction": {"enum": ["min", "max"]},
                      "tie": {"enum": ["earlier", "later"]}}, ("direction", "tie")),
        "difference": ({"time_s": number, "reference": reference, "output_units": units,
                        "order": {"enum": ["reference_minus_saved", "saved_minus_reference"]}},
                       ("time_s", "output_units", "order")),
    }
    options = []
    for name, (fields, required) in variants.items():
        props = {"operation": {"const": name}, "object_id": label, "quantity_ids": quantities, **fields}
        if name in {"extremum", "difference"}:
            props["quantity_ids"] = {**quantities, "maxItems": 1}
        required = ["operation", *required, *([] if name == "describe" else ["quantity_ids"])]
        option = _object(props, required)
        if name == "exact":
            option["oneOf"] = [{"required": ["time_s"]}, {"required": ["sample_index"]}]
        options.append(option)
    return {"oneOf": options}


class _Unavailable(Exception):
    def __init__(self, code, action, **facts):
        self.detail = {"code": code, "action": action, "facts": facts}


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _rows(workspace, sql, parameters=(), *, limit=2):
    result = workspace.query(sql, parameters, max_rows=limit)
    if result.truncated:
        raise _Unavailable("retrieval_incomplete", "Narrow the requested object, quantity or interval.")
    return result.rows


def _source(workspace, columns):
    fields = ("run_id", "oel_version", "review_schema_version", "config_sha256", "config_json")
    if not set(fields) <= columns.get("run_metadata", set()):
        raise _Unavailable("run_provenance_unavailable", "Select an identity-bound OEL run review store.")
    rows = _rows(workspace, "SELECT " + ", ".join(fields) + " FROM run_metadata LIMIT 2")
    if len(rows) != 1:
        raise _Unavailable("run_provenance_unavailable", "Select a review store for one OEL run.")
    record = rows[0]
    raw = record.pop("config_json")
    if not isinstance(raw, str) or hashlib.sha256(raw.encode()).hexdigest() != record["config_sha256"]:
        raise _Unavailable("configuration_identity_mismatch", "Restore the verified source review store.")
    try:
        initial = json.loads(raw)["simulator"].get("initial_jd_utc")
    except (KeyError, TypeError, AttributeError, ValueError):
        initial = None
    return {**record, "database": workspace.evidence_identity(),
            "schema_file": workspace.source_schema_record(),
            "initial_epoch_jd_utc": initial if _finite(initial) and initial > 0 else None}


def _frame(workspace, columns, object_id):
    def lookup(table, wanted, where="", parameters=()):
        fields = [name for name in wanted if name in columns.get(table, ())]
        if not fields:
            return {}
        rows = _rows(workspace, f"SELECT {', '.join(fields)} FROM {table} {where} LIMIT 2", parameters)
        return rows[0] if len(rows) == 1 else {}

    label = lookup("object_state_frame", ("state_frame",), "WHERE object_id = ?", (object_id,))
    frame = lookup("frame_provenance", ("model", "legacy_frame_model", "time_scale_model",
                                         "polar_motion_applied", "nutation_corrections_applied"))
    propagation = lookup("object_propagation", ("native_frame", "output_frame", "state_history_frame",
                                                "frame_transform"), "WHERE object_id = ?", (object_id,))
    value = label.get("state_frame")
    alternate = propagation.get("state_history_frame")
    if value and alternate and value.casefold() != alternate.casefold():
        raise _Unavailable("conflicting_frame_metadata", "Inspect the recorded frame and propagation provenance.",
                           state_frame=value, propagation_frame=alternate)
    return {"frame": value or alternate, "frame_model": frame.get("model") or frame.get("legacy_frame_model"),
            "time_scale_model": frame.get("time_scale_model"), "provenance": {**frame, **propagation}}


def _catalog(workspace, columns, ids):
    stored = workspace.schema().get("tables", {})
    result = {}
    for identifier in ids:
        q = QUANTITIES[identifier]
        metadata = next((c for c in stored.get(q.table, {}).get("columns", []) if c["name"] == q.column), {})
        recorded_units = metadata.get("unit")
        available = {"sample_index", "time_s", "object_id", q.column} <= columns.get(q.table, set())
        result[identifier] = {
            "definition": q.definition, "units": q.units, "value_type": "finite_number",
            "table": q.table, "column": q.column, "frame_dependent": q.frame_dependent,
            "recorded_unit": recorded_units,
            "availability": ("metadata_conflict" if recorded_units and recorded_units != q.units
                             else "column_present" if available else "column_unavailable"),
            "value_origin": "recorded_simulation_review",
        }
    return result


def _selection(workspace, source, request, catalog, frame, object_id):
    operation = request["operation"]
    quantities = [QUANTITIES[key] for key in request["quantity_ids"]]
    table = quantities[0].table
    if any(q.table != table for q in quantities):
        raise _Unavailable("mixed_quantity_sources", "Select each source table separately at the same recorded sample.")
    if any(c["availability"] != "column_present" for c in catalog.values()):
        raise _Unavailable("quantity_unavailable", "Inspect the registered quantity's recorded columns and metadata.",
                           availability={key: c["availability"] for key, c in catalog.items()})
    fields = ", ".join(["object_id", "sample_index", "time_s", *(q.column for q in quantities)])
    base = f"SELECT {fields} FROM {table} WHERE object_id = ?"

    def sample(where, value, order="time_s ASC, sample_index ASC"):
        rows = _rows(workspace, f"{base} AND {where} ORDER BY {order} LIMIT 2", (object_id, value))
        if len(rows) == 2 and rows[0]["time_s"] == rows[1]["time_s"]:
            raise _Unavailable("ambiguous_sample", "Choose an explicit recorded sample index.",
                               time_s=rows[0]["time_s"], object_id=object_id)
        return rows[0] if rows else None

    def neighbors(time):
        before = sample("time_s <= ?", time, "time_s DESC, sample_index ASC")
        after = sample("time_s >= ?", time)
        return [r for r in (before, after if after != before else None) if r is not None]

    scope = {"kind": "saved_samples", "object_id": object_id}
    if operation == "extremum":
        start, stop = request.get("start_s"), request.get("stop_s")
        if start is not None and stop is not None and start > stop:
            raise ValueError("start_s must be at or before stop_s")
        where, args = " WHERE object_id = ?", [object_id]
        for value, condition in ((start, "time_s >= ?"), (stop, "time_s <= ?")):
            if value is not None:
                where += " AND " + condition
                args.append(value)
        column = quantities[0].column
        summary = _rows(workspace, f"SELECT COUNT(*) AS samples, COUNT({column}) AS values_present, "
                        f"MIN(time_s) AS first_time_s, MAX(time_s) AS last_time_s, "
                        f"MIN({column}) AS minimum, MAX({column}) AS maximum FROM {table}{where}", args)[0]
        if summary["samples"] and (summary["values_present"] != summary["samples"] or
                                   not all(_finite(summary[k]) for k in ("minimum", "maximum"))):
            raise _Unavailable("incomplete_quantity", "Select an interval with finite recorded values for every sample.",
                               samples=summary["samples"], values_present=summary["values_present"])
        direction = "ASC" if request["direction"] == "min" else "DESC"
        tie = "ASC" if request["tie"] == "earlier" else "DESC"
        rows = _rows(workspace, f"SELECT {fields} FROM {table}{where} "
                     f"ORDER BY {column} {direction}, time_s {tie}, sample_index ASC LIMIT 1", args)
        scope.update(start_s=start, stop_s=stop, inclusive=True, tie=request["tie"], **summary)
    elif operation in {"exact", "difference"}:
        if "sample_index" in request:
            rows = _rows(workspace, f"{base} AND sample_index = ? LIMIT 2", (object_id, request["sample_index"]))
            if len(rows) > 1:
                raise _Unavailable("ambiguous_sample", "Inspect the duplicate recorded sample identity.")
            if not rows and operation == "exact":
                raise _Unavailable("sample_not_recorded", "Choose a saved sample index.",
                                   object_id=object_id, sample_index=request["sample_index"])
        else:
            time = request["time_s"]
            row = sample("time_s = ?", time)
            if row is None:
                adjacent = neighbors(time)
                raise _Unavailable("epoch_not_recorded", "Choose a saved sample or request a separately validated run.",
                                   object_id=object_id, requested_time_s=time,
                                   neighbor_times_s=[r["time_s"] for r in adjacent])
            rows = [row]
    else:
        time = request["time_s"]
        rows = neighbors(time)
        if operation == "nearest" and rows:
            sign = 1 if request["tie"] == "earlier" else -1
            row = min(rows, key=lambda r: (abs(r["time_s"] - time), sign * r["time_s"]))
            distance = abs(row["time_s"] - time)
            if distance > request.get("max_separation_s", math.inf):
                raise _Unavailable("nearest_outside_tolerance", "Choose a different epoch or explicitly revise the tolerance.",
                                   nearest_time_s=row["time_s"], separation_s=distance)
            rows = [row]
            scope.update(separation_s=distance, sample_minus_requested_s=row["time_s"] - time,
                         tie=request["tie"])
        elif operation == "bracket" and rows and not (rows[0]["time_s"] <= time <= rows[-1]["time_s"]):
            raise _Unavailable("bracket_unavailable", "Choose a time within the saved sample interval.",
                               requested_time_s=time, neighbor_times_s=[r["time_s"] for r in rows])
    if not rows:
        raise _Unavailable("no_matching_samples", "Inspect the run's recorded object, quantities and time coverage.")
    initial = source["initial_epoch_jd_utc"]
    for row in rows:
        if not _finite(row["time_s"]) or not all(_finite(row[q.column]) for q in quantities):
            raise _Unavailable("incomplete_quantity", "Inspect the missing or nonfinite recorded sample values.")
        row["epoch_jd_utc"] = initial + row["time_s"] / 86400 if initial is not None else None
        row["sample_sha256"] = _sha(row)
    if operation == "difference":
        return _difference(request, rows[0], quantities[0], frame), scope
    return rows, scope


def _difference(request, row, quantity, frame):
    reference = request.get("reference")
    if reference is None:
        raise _Unavailable("reference_missing", "Supply the attributed reference value, units, object, frame and epoch.")
    if reference["quantity_id"] != quantity.column or reference["object_id"] != row["object_id"]:
        raise _Unavailable("reference_quantity_mismatch", "Supply a reference for the same physical quantity and object.")
    if row["epoch_jd_utc"] is None:
        raise _Unavailable("epoch_metadata_missing", "Select a run with a recorded absolute UTC epoch.")
    if reference["epoch_jd_utc"] != row["epoch_jd_utc"]:
        raise _Unavailable("reference_epoch_mismatch", "Supply a reference at the selected saved UTC epoch.",
                           saved_epoch_jd_utc=row["epoch_jd_utc"])
    if quantity.frame_dependent:
        if not all((frame["frame"], frame["frame_model"], reference.get("frame"), reference.get("frame_model"))):
            raise _Unavailable("frame_metadata_missing", "Supply the recorded and reference frame and frame model.")
        if (reference["frame"].casefold(), reference["frame_model"].casefold()) != (
            frame["frame"].casefold(), frame["frame_model"].casefold()
        ):
            raise _Unavailable("reference_frame_mismatch", "Supply reference evidence in the saved frame and frame model.")
    dimension, saved_scale = _UNITS[quantity.units]
    ref_dimension, ref_scale = _UNITS[reference["units"]]
    output_dimension, output_scale = _UNITS[request["output_units"]]
    if dimension != ref_dimension or dimension != output_dimension:
        raise _Unavailable("incompatible_units", "Choose units of the same dimension as the recorded quantity.")
    saved = row[quantity.column] * (saved_scale / output_scale)
    supplied = reference["value"] * (ref_scale / output_scale)
    difference = supplied - saved if request["order"] == "reference_minus_saved" else saved - supplied
    if not all(map(_finite, (saved, supplied, difference))):
        raise _Unavailable("nonfinite_derivation", "Check operand magnitudes and output units.")
    return [{**row, "difference": difference, "difference_units": request["output_units"],
             "operand_order": request["order"], "saved_in_output_units": saved,
             "reference_in_output_units": supplied, "reference": {**reference, "verification": "caller_supplied"}}]


def scientific_evidence(workspace, request):
    """Read one operation from a bound snapshot and return a portable receipt."""
    from jsonschema import Draft202012Validator

    if not Draft202012Validator(operation_schema()).is_valid(request):
        raise ValueError("Invalid scientific evidence request. Inspect the operation schema.")
    # JSON strictness rejects NaN and Infinity even on direct Python calls.
    request = json.loads(json.dumps(request, allow_nan=False))
    result = {"schema": CONTRACT, "request": request, "status": "available", "rows": [],
              "limitations": ["Saved simulation evidence", "No interpolation or continuous-time inference"]}
    try:
        workspace.freeze_source()
        if workspace.has_live_wal():
            raise _Unavailable("review_store_unsettled", "Wait for the source writer to checkpoint and complete.")
        columns = {table: {c["name"] for c in fields} for table, fields in workspace.table_columns().items()}
        result["source"] = _source(workspace, columns)
        ids = request.get("quantity_ids", sorted(QUANTITIES))
        result["quantities"] = _catalog(workspace, columns, ids)
        objects = _rows(workspace, "SELECT object_id FROM objects ORDER BY object_id", limit=1024)
        object_ids = [r["object_id"] for r in objects]
        result["objects"] = object_ids
        object_id = request.get("object_id")
        if object_id is None and len(object_ids) == 1:
            object_id = object_ids[0]
        if object_id is None and request["operation"] != "describe":
            raise _Unavailable("object_ambiguous", "Choose the exact object from the recorded inventory.", objects=object_ids)
        if object_id is not None and object_id not in object_ids:
            raise _Unavailable("object_not_recorded", "Choose an object from the recorded inventory.", objects=object_ids)
        if object_id is not None:
            result["coordinate_context"] = _frame(workspace, columns, object_id)
        if request["operation"] != "describe":
            result["rows"], result["sampling"] = _selection(
                workspace, result["source"], request, result["quantities"], result["coordinate_context"], object_id
            )
    except _Unavailable as exc:
        result.update(status="unavailable", blocker=exc.detail)
    except ReviewQueryError as exc:
        code = getattr(exc, "code", "query_failed")
        result.update(status="unavailable", blocker={"code": code, "facts": {},
                      "action": "Inspect the recorded schema or narrow the evidence request."})
    except SecureReadChanged:
        result.update(status="unavailable", blocker={"code": "evidence_changed", "facts": {},
                      "action": "Reopen and verify the accepted evidence identity."})
    except ReviewStoreNotFoundError:
        result = {"schema": CONTRACT, "request": request, "status": "unavailable", "rows": [],
                  "blocker": {"code": "review_source_unavailable", "facts": {},
                              "action": "Open an intact, settled review store."}}
    else:
        if result["rows"]:
            result["row_count"] = len(result["rows"])
    try:
        if "source" in result:
            workspace.assert_source_unchanged()
    except SecureReadChanged:
        result = {"schema": CONTRACT, "request": request, "status": "unavailable", "rows": [],
                  "blocker": {"code": "evidence_changed", "facts": {},
                              "action": "Reopen and verify the accepted evidence identity."}}
    return {**result, "receipt_sha256": _sha(result)}


def read_scientific_evidence(path, request, *, protected=None):
    """Open the normal review-store entry point and close its source handles."""
    try:
        with ReviewWorkspace.open(path, protected=protected) as workspace:
            return scientific_evidence(workspace, request)
    except ReviewStoreNotFoundError:
        result = {"schema": CONTRACT, "request": request, "status": "unavailable", "rows": [],
                  "blocker": {"code": "review_store_missing", "facts": {},
                              "action": "Select a completed run with recorded review evidence."}}
        return {**result, "receipt_sha256": _sha(result)}
