from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from sim.dynamics.orbit.tle import parse_tle_lines
from sim.interchange.provenance import sha256_file
from sim.interchange.validation import load_interchange_document, validate_product
from sim.scale.store import (
    OdSolutionRecord,
    TLEIngestRecord,
    TLEIngestResult,
    ingest_tle_records,
    write_od_solution_records,
)


def ingest_ogp_mean_element_product(
    product_path: str | Path,
    *,
    store_path: str | Path,
    object_id: str | None = None,
) -> dict[str, Any]:
    """Promote an accepted native OGP product into Scale without TLE reconstruction."""

    source = Path(product_path).expanduser().resolve()
    product = load_interchange_document(source)
    validation = validate_product(product, source_path=source)
    if product.get("product_kind") != "oel.ogp_mean_element_product":
        raise ValueError("Scale OGP ingest requires product_kind oel.ogp_mean_element_product.")
    if not validation.valid or not validation.promotable:
        blockers = [f"{item.path}: {item.message}" for item in validation.issues if item.severity in {"error", "blocker"}]
        raise ValueError("Scale OGP ingest blocked: " + "; ".join(blockers))
    payload = dict(product["payload"])
    identity = dict(payload["object"])
    declared_id = str(identity["object_id"])
    resolved_id = str(object_id or declared_id).strip()
    if object_id is not None and resolved_id != declared_id:
        raise ValueError("object_id override must exactly match the product object identity.")
    state = dict(payload["mean_elements"])
    elements = dict(state["values"])
    product_id = str(product["product_id"])
    digest = product_id.rsplit(":", maxsplit=1)[-1]
    solution = OdSolutionRecord(
        od_solution_id=f"od:ogp_interchange:{digest}",
        object_id=resolved_id,
        method="ogp_interchange",
        source_input_ids=(product_id,),
        input_digest=digest,
        epoch_jd_utc=float(dict(state["epoch"])["value"]),
        native_frame="teme",
        state={},
        mean_elements=elements,
        residual_summary=dict(dict(payload.get("estimator_evidence", {}) or {}).get("fit", {}) or {}),
        quality_flags={
            "source": "oel_product_envelope",
            "source_product_id": product_id,
            "source_product_sha256": sha256_file(source),
            "source_product_path": str(source),
            "source_freshness": dict(product.get("freshness", {}) or {}),
            "source_provenance": dict(product.get("provenance", {}) or {}),
            "object_identity": identity,
            "warnings": list(dict(product.get("quality", {}) or {}).get("warnings", []) or []),
        },
        status="current_candidate",
    )
    result = write_od_solution_records(store_path, solutions=[solution], update_current=True)
    return {
        "schema_version": 1,
        "status": "completed" if result["od_solutions_inserted"] in {0, 1} else "failed",
        "store_path": result["store_path"],
        "object_id": resolved_id,
        "source_product_id": product_id,
        "source_product_sha256": sha256_file(source),
        "reference_epoch_jd_utc": solution.epoch_jd_utc,
        "source_freshness": dict(product.get("freshness", {}) or {}),
        "od_solution_id": solution.od_solution_id,
        "od_solutions_inserted": result["od_solutions_inserted"],
        "current_states_updated": result["current_states_updated"],
        "invalidations_inserted": result["invalidations_inserted"],
        "tle_text_generated": False,
        "execution_occurred": False,
    }


def ingest_tle_file(
    tle_path: str | Path,
    *,
    store_path: str | Path,
    source_label: str = "",
    object_id_prefix: str = "norad",
    require_checksum: bool = False,
) -> dict[str, Any]:
    source = Path(tle_path).expanduser().resolve()
    records = parse_tle_file(source, object_id_prefix=object_id_prefix, require_checksum=require_checksum)
    digest = _file_digest(source)
    result = ingest_tle_records(
        store_path,
        records=records,
        source_label=source_label or source.stem,
        source_digest=digest,
        normalized_by="sim.scale.ingest_tle_file",
    )
    return _result_payload(result, records=records)


def parse_tle_file(
    path: str | Path,
    *,
    object_id_prefix: str = "norad",
    require_checksum: bool = False,
) -> list[TLEIngestRecord]:
    source = Path(path).expanduser().resolve()
    lines = [line.rstrip("\n") for line in source.read_text(encoding="utf-8").splitlines()]
    records: list[TLEIngestRecord] = []
    idx = 0
    while idx < len(lines):
        line = lines[idx].strip()
        if not line:
            idx += 1
            continue
        name = ""
        if line.startswith("1 "):
            line1 = lines[idx].rstrip()
            if idx + 1 >= len(lines):
                raise ValueError(f"TLE line1 at {source}:{idx + 1} is missing line2.")
            line2 = lines[idx + 1].rstrip()
            idx += 2
        else:
            name = line
            if idx + 2 >= len(lines):
                raise ValueError(f"TLE name at {source}:{idx + 1} is missing line1/line2.")
            line1 = lines[idx + 1].rstrip()
            line2 = lines[idx + 2].rstrip()
            idx += 3
        elements = parse_tle_lines(line1, line2, require_checksum=require_checksum)
        digest = _tle_digest(line1, line2)
        object_id = f"{object_id_prefix}:{elements.norad_number}" if object_id_prefix else elements.norad_number
        warnings: list[str] = []
        records.append(
            TLEIngestRecord(
                object_id=object_id,
                primary_name=name or f"NORAD {elements.norad_number}",
                line1=line1,
                line2=line2,
                norad_id=elements.norad_number,
                cospar_id=elements.international_designator,
                tle_epoch_jd_utc=float(elements.epoch_jd_utc),
                parsed_elements=asdict(elements),
                digest=digest,
                warnings=tuple(warnings),
            )
        )
    if not records:
        raise ValueError(f"No TLE records found in {source}.")
    return records


def _result_payload(result: TLEIngestResult, *, records: list[TLEIngestRecord]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "completed",
        "store_path": str(result.store_path),
        "batch_id": result.batch_id,
        "source_label": result.source_label,
        "source_digest": result.source_digest,
        "records_seen": result.records_seen,
        "objects_created": result.objects_created,
        "tle_records_inserted": result.tle_records_inserted,
        "od_solutions_inserted": result.od_solutions_inserted,
        "current_states_updated": result.current_states_updated,
        "invalidations_inserted": result.invalidations_inserted,
        "objects": [
            {
                "object_id": record.object_id,
                "primary_name": record.primary_name,
                "norad_id": record.norad_id,
                "tle_epoch_jd_utc": record.tle_epoch_jd_utc,
                "digest": record.digest,
                "warnings": list(record.warnings),
            }
            for record in records
        ],
        "non_claims": [
            "TLE ingest normalized catalog records only.",
            "No propagation was executed.",
            "No conjunction candidates were generated.",
        ],
    }


def _file_digest(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _tle_digest(line1: str, line2: str) -> str:
    payload = json.dumps({"line1": line1.rstrip(), "line2": line2.rstrip()}, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
