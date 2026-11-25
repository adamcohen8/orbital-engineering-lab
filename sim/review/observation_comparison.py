"""Compare independently documented observation selections without fitting an orbit."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

PARTITIONS = ("fit", "holdout", "excluded")
FIELDS = ("uncertainty", "weight", "catalog_correction")
SCHEMA = "oel.observation_treatment.v1"


def _object(value, label):
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object.")
    return value


def _label(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string.")
    return value


def _check_value(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Treatment values must be finite or null.")
    if isinstance(value, (dict, list, bool)):
        raise ValueError("Use named scalar treatment fields with units and component labels.")


def _csv_rows(path, files):
    """OEL Cartesian residual files have one row per observation and unit-bearing sigma fields."""
    if not isinstance(files, list) or not files:
        raise ValueError("residual_csv must list one or more workspace-relative files.")
    rows = []
    for name in files:
        relative = Path(_label(name, "residual_csv entry"))
        target = (path.parent / relative).resolve()
        if relative.is_absolute() or not target.is_relative_to(path.parent.resolve()):
            raise ValueError("A residual CSV must be relative to its manifest directory.")
        with target.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if not {"observation_id", "partition"} <= set(reader.fieldnames or []):
                raise ValueError("Expected a Cartesian residual CSV with observation_id and partition.")
            for row in reader:
                uncertainty = {key: float(row[key]) for key in ("position_sigma_km", "velocity_sigma_km_s")
                               if row.get(key)}
                rows.append({"observation_id": row["observation_id"], "partition": row["partition"],
                             "uncertainty": uncertainty})
    return rows


def load_treatment(path):
    path = Path(path)
    value = _object(json.loads(path.read_text(encoding="utf-8")), "manifest")
    if "residual_csv" in value:
        if "observations" in value:
            raise ValueError("Select observations or residual_csv once.")
        value["observations"] = _csv_rows(path, value.pop("residual_csv"))
        value.setdefault("complete_partitions", sorted({r["partition"] for r in value["observations"]}))
    return value


def _normalize(manifest):
    _object(manifest, "manifest")
    if manifest.get("schema") != SCHEMA:
        raise ValueError(f"Expected schema {SCHEMA}.")
    if set(manifest) - {"schema", "source", "id_namespace", "counts", "observations", "complete_partitions"}:
        raise ValueError("Unknown observation treatment manifest field.")
    source = _label(manifest.get("source"), "source")
    namespace = manifest.get("id_namespace")
    if namespace is not None:
        _label(namespace, "id_namespace")
    counts = dict(_object(manifest.get("counts", {}), "counts"))
    if set(counts) - {"available", *PARTITIONS} or any(type(c) is not int or c < 0 for c in counts.values()):
        raise ValueError("Counts must be nonnegative integers for available, fit, holdout or excluded.")
    groups = None
    indexed = None
    if "observations" in manifest:
        if not isinstance(manifest["observations"], list) or namespace is None:
            raise ValueError("Named observations require an array and a stable id_namespace.")
        complete = manifest.get("complete_partitions", list(PARTITIONS))
        if (not isinstance(complete, list) or any(p not in PARTITIONS for p in complete)
                or len(complete) != len(set(complete))):
            raise ValueError("complete_partitions must list distinct partition names.")
        groups = {part: [] if part in complete else None for part in PARTITIONS}
        indexed = {}
        for row in manifest["observations"]:
            _object(row, "observation")
            if set(row) - {"observation_id", "partition", *FIELDS}:
                raise ValueError("Unknown treatment row field. Map source fields into the documented treatments.")
            identity = _label(row.get("observation_id"), "observation_id")
            partition = row.get("partition")
            if identity in indexed or partition not in complete:
                raise ValueError("Duplicate observation identity or unknown partition.")
            for field in FIELDS:
                if row.get(field) is not None:
                    for key, value in _object(row[field], field).items():
                        _label(key, "treatment field")
                        _check_value(value)
            indexed[identity] = row
            groups[partition].append(identity)
        for part, ids in groups.items():
            if ids is None:
                continue
            ids.sort()
            if part in counts and counts[part] != len(ids):
                raise ValueError(f"Listed observations disagree with the {part} count.")
            counts[part] = len(ids)
    elif "complete_partitions" in manifest:
        raise ValueError("complete_partitions requires row identities.")
    elif not counts:
        raise ValueError("Supply either aggregate counts or complete observation identities.")
    if "available" in counts and sum(counts.get(p, 0) for p in PARTITIONS) > counts["available"]:
        raise ValueError("Partition counts exceed available observations.")
    identity = hashlib.sha256(json.dumps(manifest, allow_nan=False, sort_keys=True,
                                         separators=(",", ":")).encode()).hexdigest()
    return {"source": source, "id_namespace": namespace, "counts": counts, "ids_by_partition": groups,
            "manifest_sha256": identity, "reported_treatments": (
                {i: {k: row[k] for k in FIELDS if row.get(k)} for i, row in sorted(indexed.items())}
                if indexed is not None else None)}, indexed


def compare_observations(left, right):
    """Compare stable identities and verbatim reported treatments in two manifests."""
    a, rows_a = _normalize(left)
    b, rows_b = _normalize(right)
    result = {"schema": "oel.observation_comparison.v1", "left": a, "right": b,
              "count_delta_right_minus_left": {p: b["counts"][p] - a["counts"][p]
                    for p in ("available", *PARTITIONS) if p in a["counts"] and p in b["counts"]},
              "fit_overlap": None, "partition_changes": None, "only_left": None, "only_right": None,
              "treatment_comparison": None}
    if rows_a is None or rows_b is None:
        return {**result, "identity_status": "unknown_aggregate_only"}
    if a["id_namespace"] != b["id_namespace"]:
        return {**result, "identity_status": "unknown_different_namespaces"}

    common = sorted(rows_a.keys() & rows_b.keys())
    treatments = {field: {"changed": [], "unknown": [], "matching_observation_count": 0} for field in FIELDS}
    for identity in common:
        for field in FIELDS:
            values_a = rows_a[identity].get(field) or {}
            values_b = rows_b[identity].get(field) or {}
            changed, unknown = {}, []
            for key in sorted(values_a.keys() | values_b.keys()):
                x, y = values_a.get(key), values_b.get(key)
                if x is None or y is None:
                    unknown.append(key)
                elif x != y:
                    changed[key] = {"left": x, "right": y}
            report = treatments[field]
            if changed:
                report["changed"].append({"observation_id": identity, "fields": changed})
            if unknown or not (values_a or values_b):
                report["unknown"].append({"observation_id": identity, "fields": unknown or None})
            elif not changed:
                report["matching_observation_count"] += 1
    fit_a, fit_b = a["ids_by_partition"]["fit"], b["ids_by_partition"]["fit"]
    return {**result, "identity_status": "matched_by_stable_id",
            "fit_overlap": sorted(set(fit_a) & set(fit_b)) if fit_a is not None and fit_b is not None else None,
            "only_left": sorted(rows_a.keys() - rows_b.keys()), "only_right": sorted(rows_b.keys() - rows_a.keys()),
            "partition_changes": [{"observation_id": i, "left": rows_a[i]["partition"], "right": rows_b[i]["partition"]}
                                  for i in common if rows_a[i]["partition"] != rows_b[i]["partition"]],
            "treatment_comparison": treatments}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    args = parser.parse_args(argv)
    try:
        report = compare_observations(load_treatment(args.left), load_treatment(args.right))
    except (ValueError, OSError, csv.Error) as error:
        print(str(error), file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0
