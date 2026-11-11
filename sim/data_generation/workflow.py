from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from sim.dynamics.orbit.tle import parse_tle_lines, with_tle_checksum
from sim.scale.ingest import parse_tle_file
from sim.scale.store import TLEIngestRecord, ingest_tle_records, load_current_tle_states, summarize_store
from sim.security import ConfigPathPolicy

_GENERATOR_VERSION = 2


def plan_synthetic_data(config_path: str | Path) -> dict[str, Any]:
    cfg = _load_config(config_path)
    payload = {
        "schema_version": 1,
        "status": "planned",
        "generator": _generator_metadata(cfg),
        "truth": {
            "object_count": cfg["object_count"],
            "template_tle_file": str(cfg["template_tle_file"]),
            "object_id_prefix": cfg["object_id_prefix"],
            "close_approach_count": len(cfg["close_approaches"]),
        },
        "outputs": _artifact_paths(cfg),
        "non_claims": _non_claims(),
    }
    _write_json(Path(payload["outputs"]["plan"]), payload)
    return payload


def generate_synthetic_data(config_path: str | Path) -> dict[str, Any]:
    cfg = _load_config(config_path)
    template = parse_tle_file(cfg["template_tle_file"])[0]
    records: list[TLEIngestRecord] = []
    truth_objects: list[dict[str, Any]] = []
    rng = np.random.default_rng(int(cfg["seed"]))
    phases = [
        (360.0 * index) / cfg["object_count"] + float(rng.uniform(-0.01, 0.01))
        for index in range(cfg["object_count"])
    ]
    for event in cfg["close_approaches"]:
        left_index, right_index = event["object_indices"]
        phases[right_index] = phases[left_index] + event["phase_separation_deg"]
    for index in range(cfg["object_count"]):
        object_id = f"{cfg['object_id_prefix']}:{index:04d}"
        line1, line2 = _phased_tle(
            template.line1,
            template.line2,
            satellite_number=80000 + index,
            phase_offset_deg=phases[index],
        )
        elements = parse_tle_lines(line1, line2, require_checksum=True)
        digest = hashlib.sha256(f"{line1}\n{line2}".encode()).hexdigest()
        records.append(
            TLEIngestRecord(
                object_id=object_id,
                primary_name=f"Synthetic Object {index:04d}",
                line1=line1,
                line2=line2,
                norad_id=elements.norad_number,
                cospar_id=elements.international_designator,
                tle_epoch_jd_utc=elements.epoch_jd_utc,
                parsed_elements={
                    "inclination_deg": elements.inclination_deg,
                    "raan_deg": elements.raan_deg,
                    "eccentricity": elements.eccentricity,
                    "argp_deg": elements.argp_deg,
                    "mean_anomaly_deg": elements.mean_anomaly_deg,
                    "mean_motion_rev_per_day": elements.mean_motion_rev_per_day,
                },
                digest=digest,
            )
        )
        truth_objects.append(
            {
                "object_id": object_id,
                "truth_kind": "ogp_sgp4_template_phase_offset",
                "tle_digest": digest,
                "mean_anomaly_deg": elements.mean_anomaly_deg,
            }
        )

    source_digest = hashlib.sha256(
        json.dumps([record.digest for record in records], sort_keys=True).encode("utf-8")
    ).hexdigest()
    ingest = ingest_tle_records(
        cfg["store_path"],
        records=records,
        source_label=f"synthetic:{cfg['name']}",
        source_digest=source_digest,
        normalized_by="sim.data_generation.generate",
    )
    paths = _artifact_paths(cfg)
    truth_events = [
        {
            "event_id": event["event_id"],
            "event_type": "sampled_close_approach",
            "object_ids": [
                f"{cfg['object_id_prefix']}:{event['object_indices'][0]:04d}",
                f"{cfg['object_id_prefix']}:{event['object_indices'][1]:04d}",
            ],
            "expected_tca_jd_utc": cfg["propagation_start_jd_utc"],
            "expected_miss_distance_km": 0.0 if event["phase_separation_deg"] == 0.0 else None,
            "phase_separation_deg": event["phase_separation_deg"],
        }
        for event in cfg["close_approaches"]
    ]
    truth_manifest = {
        "schema_version": 1,
        "generator": _generator_metadata(cfg),
        "objects": truth_objects,
        "events": truth_events,
        "hidden_from_operational_processing": True,
        "non_claims": _non_claims(),
    }
    input_manifest = {
        "schema_version": 1,
        "batch_id": ingest.batch_id,
        "source_digest": source_digest,
        "store_path": str(cfg["store_path"]),
        "records_seen": ingest.records_seen,
        "tle_records_inserted": ingest.tle_records_inserted,
    }
    expected = {
        "schema_version": 1,
        "expected_object_ids": [row["object_id"] for row in truth_objects],
        "expected_current_state_count": len(truth_objects),
        "expected_candidate_pairs": [event["object_ids"] for event in truth_events],
        "expected_candidates": [
            {
                "event_id": event["event_id"],
                "object_ids": event["object_ids"],
                "expected_tca_jd_utc": event["expected_tca_jd_utc"],
                "expected_miss_distance_km": event["expected_miss_distance_km"],
                "tca_tolerance_s": cfg["expected_candidate_tolerance_s"],
                "miss_distance_tolerance_km": cfg["expected_miss_distance_tolerance_km"],
            }
            for event in truth_events
        ],
        "negative_control_policy": "all unlisted pairs should remain outside the sampled threshold",
    }
    manifest = {
        "schema_version": 1,
        "status": "generated",
        "generator": _generator_metadata(cfg),
        "artifacts": paths,
        "store_path": str(cfg["store_path"]),
        "object_count": len(truth_objects),
        "non_claims": _non_claims(),
    }
    _write_json(Path(paths["truth_manifest"]), truth_manifest)
    _write_json(Path(paths["input_batches"]), input_manifest)
    _write_json(Path(paths["expected_results"]), expected)
    _write_json(Path(paths["generation_manifest"]), manifest)
    Path(paths["dataset_card"]).write_text(_dataset_card(cfg), encoding="utf-8")
    return manifest


def score_synthetic_data(config_path: str | Path) -> dict[str, Any]:
    cfg = _load_config(config_path)
    paths = _artifact_paths(cfg)
    generation = json.loads(Path(paths["generation_manifest"]).read_text(encoding="utf-8"))
    generator = _generator_metadata(cfg)
    original_generator = generation.get("generator", {})
    if any(original_generator.get(key) != generator[key]
           for key in ("config_digest", "template_sha256", "version", "seed")):
        raise ValueError("Synthetic data configuration no longer matches generation; regenerate before scoring.")
    expected = json.loads(Path(paths["expected_results"]).read_text(encoding="utf-8"))
    summary = summarize_store(cfg["store_path"])
    expected_ids = set(expected["expected_object_ids"])
    actual_ids = set(summary.current_object_ids)
    missing = sorted(expected_ids - actual_ids)
    unexpected = sorted(actual_ids - expected_ids)
    truth = json.loads(Path(paths["truth_manifest"]).read_text(encoding="utf-8"))
    expected_digests = {
        str(row["object_id"]): str(row["tle_digest"])
        for row in list(truth.get("objects", []) or [])
        if isinstance(row, dict) and row.get("object_id") and row.get("tle_digest")
    }
    actual_states = {
        state.object_id: hashlib.sha256(f"{state.line1}\n{state.line2}".encode()).hexdigest()
        for state in load_current_tle_states(cfg["store_path"])
        if state.line1 and state.line2
    }
    digest_mismatches = sorted(
        object_id
        for object_id in expected_ids
        if actual_states.get(object_id) != expected_digests.get(object_id)
    )
    payload = {
        "schema_version": 1,
        "status": "passed" if not missing and not unexpected and not digest_mismatches else "failed",
        "generator": generator,
        "checks": {
            "expected_current_states_present": not missing,
            "no_unexpected_current_states": not unexpected,
            "expected_tle_digests_match": not digest_mismatches,
            "expected_count": len(expected_ids),
            "actual_count": len(actual_ids),
            "missing_object_ids": missing,
            "unexpected_object_ids": unexpected,
            "digest_mismatch_object_ids": digest_mismatches,
        },
        "non_claims": _non_claims(),
    }
    _write_json(Path(paths["score_report"]), payload)
    Path(paths["score_report_markdown"]).write_text(_score_markdown(payload), encoding="utf-8")
    return payload


def _load_config(config_path: str | Path) -> dict[str, Any]:
    source = Path(config_path).expanduser().resolve()
    raw = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    section = dict(raw.get("synthetic_data", {}) or {})
    if not section:
        raise ValueError("Synthetic data config requires a synthetic_data mapping.")
    truth = dict(section.get("truth", {}) or {})
    events = dict(section.get("events", {}) or {})
    scoring = dict(section.get("scoring", {}) or {})
    store = dict(section.get("store", {}) or {})
    policy = ConfigPathPolicy.default(config_path=source)
    count = int(truth.get("object_count", 0) or 0)
    if count < 2 or count > 1000:
        raise ValueError("synthetic_data.truth.object_count must be between 2 and 1000.")
    name = str(raw.get("name") or source.stem).strip()
    close_approaches: list[dict[str, Any]] = []
    event_ids: set[str] = set()
    for index, item in enumerate(events.get("close_approaches", []) or []):
        event = dict(item or {})
        indices = tuple(int(value) for value in event.get("object_indices", ()))
        if len(indices) != 2 or indices[0] == indices[1]:
            raise ValueError("Each close approach requires two distinct object_indices.")
        if min(indices) < 0 or max(indices) >= count:
            raise ValueError("Close-approach object_indices must reference generated objects.")
        event_id = str(event.get("event_id") or f"close_approach_{index:03d}").strip()
        if not event_id or event_id in event_ids:
            raise ValueError("Close-approach event_id values must be non-empty and unique.")
        event_ids.add(event_id)
        close_approaches.append(
            {
                "event_id": event_id,
                "object_indices": indices,
                "phase_separation_deg": _finite_config_float(
                    event.get("phase_separation_deg", 0.0) or 0.0,
                    name="phase_separation_deg",
                ),
            }
        )
    scale_catalog = dict(raw.get("scale_catalog", {}) or {})
    propagation = dict(scale_catalog.get("propagation", {}) or {})
    output_dir = policy.resolve_output_dir(
        section.get("output_dir", f"outputs/data_generation/{name}"),
        purpose="synthetic_data.output_dir",
    )
    return {
        "name": name,
        "seed": int(section.get("seed", 0) or 0),
        "config_path": source,
        "output_dir": output_dir,
        "store_path": policy.resolve_output_dir(
            store.get("path", f"outputs/data_generation/{name}/catalog_store.sqlite"),
            purpose="synthetic_data.store.path",
        ),
        "object_count": count,
        "object_id_prefix": _required_text(
            truth.get("object_id_prefix", f"synthetic:{name}"),
            name="synthetic_data.truth.object_id_prefix",
        ),
        "template_tle_file": policy.resolve_input_file(
            truth.get("template_tle_file", ""),
            purpose="synthetic_data.truth.template_tle_file",
        ),
        "close_approaches": close_approaches,
        "propagation_start_jd_utc": _finite_config_float(
            propagation.get("start_jd_utc", 0.0) or 0.0,
            name="scale_catalog.propagation.start_jd_utc",
        ),
        "expected_candidate_tolerance_s": _finite_config_float(
            scoring.get("expected_candidate_tolerance_s", 1.0e-6) or 1.0e-6,
            name="synthetic_data.scoring.expected_candidate_tolerance_s",
            minimum=0.0,
        ),
        "expected_miss_distance_tolerance_km": _finite_config_float(
            scoring.get("expected_miss_distance_tolerance_km", 1.0e-9) or 1.0e-9,
            name="synthetic_data.scoring.expected_miss_distance_tolerance_km",
            minimum=0.0,
        ),
    }


def _required_text(value: Any, *, name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{name} must be a non-empty string.")
    return text


def _finite_config_float(
    value: Any,
    *,
    name: str,
    minimum: float | None = None,
) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number.") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be a finite number.")
    if minimum is not None and parsed < float(minimum):
        raise ValueError(f"{name} must be >= {float(minimum):g}.")
    return parsed


def _phased_tle(
    line1: str,
    line2: str,
    *,
    satellite_number: int,
    phase_offset_deg: float,
) -> tuple[str, str]:
    number = f"{satellite_number:05d}"
    base = parse_tle_lines(line1, line2)
    updated_line1 = with_tle_checksum(f"{line1[:2]}{number}{line1[7:]}")
    mean_anomaly = (base.mean_anomaly_deg + float(phase_offset_deg)) % 360.0
    updated_line2 = f"{line2[:2]}{number}{line2[7:43]}{mean_anomaly:8.4f}{line2[51:]}"
    return updated_line1, with_tle_checksum(updated_line2)


def _generator_metadata(cfg: dict[str, Any]) -> dict[str, Any]:
    config_digest = hashlib.sha256(Path(cfg["config_path"]).read_bytes()).hexdigest()
    return {
        "name": cfg["name"],
        "version": _GENERATOR_VERSION,
        "seed": cfg["seed"],
        "phase_jitter_half_width_deg": 0.01,
        "config_path": str(cfg["config_path"]),
        "config_digest": config_digest,
        "template_sha256": hashlib.sha256(Path(cfg["template_tle_file"]).read_bytes()).hexdigest(),
    }


def _artifact_paths(cfg: dict[str, Any]) -> dict[str, str]:
    root = Path(cfg["output_dir"])
    return {
        "plan": str(root / "synthetic_generation_plan.json"),
        "generation_manifest": str(root / "synthetic_generation_manifest.json"),
        "truth_manifest": str(root / "synthetic_truth_manifest.json"),
        "input_batches": str(root / "synthetic_input_batches.json"),
        "expected_results": str(root / "synthetic_expected_results.json"),
        "score_report": str(root / "synthetic_score_report.json"),
        "score_report_markdown": str(root / "synthetic_score_report.md"),
        "dataset_card": str(root / "synthetic_dataset_card.md"),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _non_claims() -> list[str]:
    return [
        "Generated objects are deterministic synthetic fixtures, not operational catalog data.",
        "Template-phased TLEs do not establish sensor, OD, conjunction, or collision-probability validity.",
        "Truth manifests are validation-only inputs and must not be exposed to ordinary Scale processing.",
    ]


def _dataset_card(cfg: dict[str, Any]) -> str:
    return (
        f"# Synthetic Dataset Card: {cfg['name']}\n\n"
        f"- Generator version: `{_GENERATOR_VERSION}`\n"
        f"- Seed: `{cfg['seed']}`\n"
        f"- Physics source: phased OGP-SGP4-compatible template TLE\n"
        f"- Object count: `{cfg['object_count']}`\n"
        f"- Injected sampled close approaches: `{len(cfg['close_approaches'])}`\n"
        "- Intended use: deterministic OEL Scale ingest and workflow fixtures\n"
        "- Prohibited use: operational catalog, intent, collision probability, or flight-safety claims\n"
        "- Split policy: no AI train/validation/test split is emitted by this Phase 1 fixture\n"
    )


def _score_markdown(payload: dict[str, Any]) -> str:
    checks = payload["checks"]
    return (
        f"# Synthetic Data Score: {payload['generator']['name']}\n\n"
        f"- Status: `{payload['status']}`\n"
        f"- Expected current states: `{checks['expected_count']}`\n"
        f"- Actual current states: `{checks['actual_count']}`\n"
        f"- Missing objects: `{len(checks['missing_object_ids'])}`\n"
        f"- Unexpected objects: `{len(checks['unexpected_object_ids'])}`\n"
    )
