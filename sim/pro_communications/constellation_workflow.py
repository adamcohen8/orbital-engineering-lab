"""Trusted synthetic multi-spacecraft Pro RF-service flagship workflow."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from sim.pro_communications.artifacts import register_bundle_artifacts, write_pro_study_bundle
from sim.pro_communications.constellation import evaluate_pro_constellation_service
from sim.pro_communications.coverage import ProCommunicationsCoverageConfig
from sim.pro_communications.equipment_library import load_governed_equipment_library
from sim.pro_communications.network_trades import run_constellation_service_trade
from sim.pro_communications.plotting import write_constellation_coverage_plot
from sim.pro_communications.scenario_adapter import (
    PRO_SCENARIO_ADAPTER_VERSION,
    evaluate_scenario_rf_coverage,
    load_completed_scenario_object,
)

WORKFLOW_SCHEMA_VERSION = "oel.pro-communications-constellation-workflow.v0.1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path, root: Path) -> dict[str, Any]:
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError("Pro constellation workflow config must resolve inside the workspace.") from exc
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "study_id",
        "scenario_path",
        "equipment_library_path",
        "profile_id",
        "object_ids",
        "coverage_order",
        "coverage_stride",
        "alternatives",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError("Constellation workflow fields are incomplete or unknown.")
    if payload["schema_version"] != WORKFLOW_SCHEMA_VERSION:
        raise ValueError("Unsupported constellation workflow schema_version.")
    for field_name in ("scenario_path", "equipment_library_path"):
        source = (root / str(payload[field_name])).resolve()
        try:
            source.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"{field_name} must stay inside the workspace.") from exc
        if not source.is_file():
            raise FileNotFoundError(source)
        payload[f"_{field_name}"] = source
    object_ids = tuple(str(value) for value in payload["object_ids"])
    if len(object_ids) < 3 or len(set(object_ids)) != len(object_ids):
        raise ValueError("The constellation flagship requires at least three unique objects.")
    alternatives = payload["alternatives"]
    if not isinstance(alternatives, dict) or not alternatives:
        raise ValueError("alternatives must be a non-empty object.")
    for alternative_id, members in alternatives.items():
        if not str(alternative_id).strip() or len(members) < 2 or not set(members) <= set(object_ids):
            raise ValueError("Every alternative requires at least two known members.")
    payload["_object_ids"] = object_ids
    return payload


def run_constellation_workflow(config_path: str | Path, output_dir: str | Path):
    root = Path(__file__).resolve().parents[2]
    config_source = Path(config_path).expanduser().resolve()
    config = _load(config_source, root)
    library = load_governed_equipment_library(config["_equipment_library_path"])
    profile_id = str(config["profile_id"])
    if profile_id not in library.entries:
        raise ValueError(f"Unknown governed profile {profile_id!r}.")
    profile = library.entries[profile_id].profile
    stride = int(config["coverage_stride"])
    order = int(config["coverage_order"])
    if stride <= 0 or order < 0:
        raise ValueError("coverage_stride must be positive and coverage_order non-negative.")

    coverages = {}
    source_provenance = {}
    for object_id in config["_object_ids"]:
        source = load_completed_scenario_object(
            config["_scenario_path"],
            object_id=object_id,
            attitude_source_kind="not_required",
            workspace_root=root,
        )
        coverage = evaluate_scenario_rf_coverage(
            config["_scenario_path"],
            ProCommunicationsCoverageConfig(
                analysis_id=f"{object_id}-s-band-rf-service",
                service_id="synthetic-s-band-declared-rate-service",
                source_asset_id=object_id,
                state_provider_id=f"scenario-adapter:{source.review_sha256}:{object_id}",
                attitude_source_kind="not_required",
                attitude_provider_id=None,
                quat_body_from_source_terminal=(1.0, 0.0, 0.0, 0.0),
                direction="spacecraft_to_earth",
                order=order,
                link_profile=profile,
                minimum_elevation_rad=np.deg2rad(10.0),
                maximum_range_km=2500.0,
            ),
            workspace_root=root,
            sample_stride=stride,
        )
        coverages[object_id] = coverage
        source_provenance[object_id] = {
            "review_path": str(source.review_path.relative_to(root)),
            "review_sha256": source.review_sha256,
            "normalized_config_sha256": source.normalized_config_sha256,
            "samples_before_stride": int(source.times_s.size),
            "coverage_semantic_sha256": coverage.semantic_sha256,
        }

    alternative_members = {
        str(alternative_id): tuple(coverages[member] for member in members)
        for alternative_id, members in sorted(config["alternatives"].items())
    }
    constellations = {
        alternative_id: evaluate_pro_constellation_service(
            analysis_id=f"{config['study_id']}:{alternative_id}",
            members=members,
        )
        for alternative_id, members in alternative_members.items()
    }
    design_trade = run_constellation_service_trade(
        f"{config['study_id']}:declared-alternatives",
        alternative_members,
    )
    provenance = {
        "workflow_config": {
            "path": str(config_source.relative_to(root)),
            "sha256": _sha256(config_source),
        },
        "scenario": {
            "path": str(config["_scenario_path"].relative_to(root)),
            "sha256": _sha256(config["_scenario_path"]),
        },
        "scenario_adapter_version": PRO_SCENARIO_ADAPTER_VERSION,
        "equipment_library": {
            "path": str(library.source_path.relative_to(root)),
            "sha256": library.source_sha256,
            "semantic_sha256": library.semantic_sha256,
            "profile_disposition": library.entries[profile_id].use_disposition,
        },
        "coverage_stride": stride,
        "object_sources": source_provenance,
    }
    bundle = write_pro_study_bundle(
        output_dir,
        study_id=str(config["study_id"]),
        coverages=coverages,
        constellations=constellations,
        design_trades={"constellation-service": design_trade},
        study_provenance=provenance,
        study_claims=(
            "Deterministic synchronized RF-qualified service comparison for the declared synthetic alternatives.",
        ),
        study_non_claims=(
            "No calibrated terminals, orbital optimization, contention, routing, cost, staffing, or operational availability.",
        ),
    )
    plots_dir = bundle.output_dir / "plots"
    plots_dir.mkdir()
    plot = write_constellation_coverage_plot(
        plots_dir / "constellation_rf_service.png",
        study_id=str(config["study_id"]),
        constellations=constellations,
    )
    register_bundle_artifacts(bundle, {"constellation-rf-service": plot})
    return bundle


def main() -> int:
    from sim.pro_features import FEATURE_PRO_COMMUNICATIONS, require_pro_feature

    require_pro_feature(FEATURE_PRO_COMMUNICATIONS)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    bundle = run_constellation_workflow(args.config, args.output)
    print(bundle.manifest_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
