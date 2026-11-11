from __future__ import annotations

import os
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from sim.data_io.common import read_text, strict_json
from sim.data_io.contracts import DataIOError
from sim.dynamics.orbit.tle import ogp_mean_elements_from_mapping, ogp_mean_elements_to_mapping
from sim.interchange.provenance import compute_product_id, sha256_file
from sim.interchange.validation import validate_product

OGP_OD_ADAPTER_ID = "oel.ogp_od.mean_element_product"
OGP_OD_ADAPTER_VERSION = "1"


class OGPODInterchangeError(ValueError):
    pass


def build_ogp_mean_element_product(
    report_path: str | Path,
    *,
    output_path: str | Path | None = None,
    object_id: str | None = None,
) -> dict[str, Any]:
    source = Path(report_path).expanduser().resolve()
    try:
        report = strict_json(read_text(source))
    except (FileNotFoundError, DataIOError) as exc:
        raise OGPODInterchangeError(f"OGP OD report is not valid bounded JSON: {source}") from exc
    if not isinstance(report, dict) or report.get("kind") != "oel.ogp_mean_element_od_result":
        raise OGPODInterchangeError("Expected an OGP mean-element OD report.")
    target = Path(output_path).expanduser().resolve() if output_path else source.with_name(
        "fitted_ogp_mean_element_product.json"
    )
    elements = ogp_mean_elements_from_mapping(dict(report.get("estimated_elements", {}) or {}))
    values = ogp_mean_elements_to_mapping(elements)
    gates = dict(report.get("quality_gates", {}) or {})
    warnings = [str(item) for item in list(gates.get("warnings", []) or [])]
    accepted = (
        gates.get("solver_success") is True
        and gates.get("data_full_rank") is True
        and gates.get("fit_improved") is True
        and gates.get("regime_stable") is True
        and gates.get("holdout_evidence_status") == "evaluated"
        and not warnings
    )
    norad = str(values.get("norad_number", "") or "")
    resolved_object_id = str(object_id or (f"norad:{norad}" if norad else "") or "").strip()
    if not resolved_object_id:
        raise OGPODInterchangeError("OGP product requires an explicit object_id when no NORAD number is present.")
    created = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    report_hash = sha256_file(source)
    source_artifact = {
        "artifact_id": "ogp_od_report",
        "sha256": report_hash,
        "path": os.path.relpath(source, start=target.parent),
        "media_type": "application/json",
        "size_bytes": source.stat().st_size,
    }
    product: dict[str, Any] = {
        "schema_id": "oel-product-envelope-v1",
        "schema_version": 1,
        "product_kind": "oel.ogp_mean_element_product",
        "product_id": "oel.ogp_mean_element_product:" + "0" * 64,
        "created_utc": created,
        "producer": {
            "capability_id": "ogp_mean_element_orbit_determination",
            "oel_version": _oel_version(),
            "run_id": source.parent.name,
        },
        "payload": {
            "object": {
                "object_id": resolved_object_id,
                "kind": "satellite",
                "norad_number": norad,
                "international_designator": str(values.get("international_designator", "") or ""),
            },
            "mean_elements": {
                "representation": "ogp_mean_elements",
                "frame": "TEME",
                "epoch": {"value": float(elements.epoch_jd_utc), "format": "jd", "time_system": "UTC"},
                "values": values,
            },
            "propagation": {
                "family": "OGP",
                "regime": str(report.get("ogp_regime", "") or ""),
                "propagator_name": str(report.get("propagator_name", "") or ""),
                "not_tle_text": True,
            },
            "estimator_evidence": {
                "method": str(report.get("method", "") or ""),
                "parameterization": str(dict(report.get("parameterization", {}) or {}).get("name", "") or ""),
                "selected_parameters": [
                    str(item) for item in list(dict(report.get("parameterization", {}) or {}).get("estimated_parameter_names", []) or [])
                ],
                "fit": {
                    "sample_count": int(report.get("fit_sample_count", 0) or 0),
                    "whitened_rms": dict(report.get("state_space_metrics", {}) or {}).get("fit_whitened_rms"),
                },
                "holdout": {
                    "status": str(gates.get("holdout_evidence_status", "not_evaluated")),
                    "sample_count": int(report.get("holdout_sample_count", 0) or 0),
                    "whitened_rms": dict(report.get("state_space_metrics", {}) or {}).get("holdout_whitened_rms"),
                },
                "source_report_sha256": report_hash,
            },
        },
        "quality": {
            "disposition": "accepted" if accepted else "review_required",
            "producer_status": str(gates.get("evidence_status", "unknown") or "unknown"),
            "gates": {"quality_gates": gates, "adapter": {"adapter_id": OGP_OD_ADAPTER_ID, "adapter_version": OGP_OD_ADAPTER_VERSION}},
            "warnings": warnings + ([] if gates.get("holdout_evidence_status") == "evaluated" else ["holdout_not_evaluated"]),
            "non_claims": [str(item) for item in list(report.get("non_claims", []) or [])],
        },
        "freshness": {
            "integrity_status": "current",
            "age_status": "not_applicable",
            "reference_epoch_jd_utc": float(elements.epoch_jd_utc),
            "evaluated_utc": created,
            "policy": {"assessment": "consumer_must_evaluate_product_age_for_analysis_epoch"},
        },
        "provenance": {
            "source_artifacts": [source_artifact],
            "source_product_ids": [],
            "transformations": [{
                "transformation_id": "ogp_od_report_to_native_mean_elements",
                "version": "1",
                "details": {"tle_text_generated": False, "frame_transform_applied": False},
            }],
        },
        "data_markings": {
            "scope": "private_pro",
            "handling": "OEL Pro fitted-orbit evidence",
            "approved_for_public_export": False,
            "contains_customer_data": False,
            "contains_hidden_truth": False,
            "source_markings": [],
        },
    }
    product["product_id"] = compute_product_id(product)
    validation = validate_product(product, source_path=target)
    if not validation.valid:
        messages = "; ".join(f"{item.path}: {item.message}" for item in validation.errors)
        raise OGPODInterchangeError(f"Generated OGP product failed validation: {messages}")
    return product


def _oel_version() -> str:
    try:
        return version("orbital-engineering-lab")
    except PackageNotFoundError:
        return "source-tree"
