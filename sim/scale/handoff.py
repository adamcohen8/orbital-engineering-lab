from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from sim.scale._output import bind_output_writer, staged_sqlite_store
from sim.scale.planner import (
    ScalePlannerConfig,
    enforce_scale_planner_limits,
    load_scale_planner_config,
    resolve_selected_object_ids,
)
from sim.scale.store import (
    OnpHandoffCase,
    OnpHandoffPacketRecord,
    load_onp_handoff_packets,
    load_pending_onp_handoff_cases,
    write_onp_handoff_packets,
)
from sim.security.output_writes import OutputWriter


@enforce_scale_planner_limits("ONP handoff")
def run_onp_handoff(
    config_path: str | Path,
    *,
    writer: OutputWriter | None = None,
) -> dict[str, Any]:
    """Write manifest-only ONP handoff packets using the legacy-compatible store shape."""

    return _run_handoff(config_path, legacy_hpop_names=False, writer=writer)


def _run_handoff(
    config_path: str | Path,
    *,
    legacy_hpop_names: bool,
    writer: OutputWriter | None = None,
) -> dict[str, Any]:
    cfg = load_scale_planner_config(config_path)
    if cfg.mode != "run":
        raise ValueError("ONP handoff requires scale_catalog.mode: run.")
    if not cfg.onp_handoff_enabled:
        raise ValueError("ONP handoff requires scale_catalog.onp_handoff.enabled: true (legacy hpop_handoff is accepted).")

    with staged_sqlite_store(cfg.store_path, writer=writer) as store_path:
        store_cfg = replace(cfg, store_path=store_path)
        cases = load_pending_onp_handoff_cases(
            store_path,
            object_ids=resolve_selected_object_ids(cfg),
            limit=cfg.onp_handoff_limit,
        )
        packets = _build_handoff_packets(cfg, cases=cases, legacy_hpop_names=legacy_hpop_names)
        binding = bind_output_writer(writer, cfg.onp_handoff_output_dir)
        _write_packet_files(packets, writer=binding)
        write_summary = write_onp_handoff_packets(store_cfg.store_path, packets=packets)
        manifest_packets = load_onp_handoff_packets(
            store_cfg.store_path,
            object_ids=resolve_selected_object_ids(cfg),
        )
        manifest_path = _write_manifest(
            cfg,
            packets=manifest_packets,
            legacy_hpop_names=legacy_hpop_names,
            writer=binding,
        )
    return {
        "schema_version": 1,
        "status": "completed",
        "workflow": {
            "name": cfg.name,
            "config_path": str(cfg.config_path),
            "mode": cfg.mode,
        },
        "store_path": str(cfg.store_path),
        "output_dir": str(cfg.onp_handoff_output_dir),
        "manifest_path": str(manifest_path),
        "cases_seen": len(cases),
        "packets_seen": write_summary["packets_seen"],
        "packets_inserted": write_summary["packets_inserted"],
        "propagator_name": "ONP",
        "propagator_full_name": "OEL Numerical Propagator",
        "force_model_profile": cfg.force_model_profile,
        "non_claims": [
            "ONP handoff generated manifest and job packets only.",
            "No ONP scenario YAML was generated or validated.",
            "No ONP propagation was executed.",
        ],
    }


@enforce_scale_planner_limits("legacy HPOP-named ONP handoff")
def run_hpop_handoff(
    config_path: str | Path,
    *,
    writer: OutputWriter | None = None,
) -> dict[str, Any]:
    """Compatibility alias for the legacy HPOP-named ONP handoff artifact."""

    return _run_handoff(config_path, legacy_hpop_names=True, writer=writer)


def build_onp_handoff_packets(
    cfg: ScalePlannerConfig,
    *,
    cases: list[OnpHandoffCase],
) -> list[OnpHandoffPacketRecord]:
    return _build_handoff_packets(cfg, cases=cases, legacy_hpop_names=False)


def _build_handoff_packets(
    cfg: ScalePlannerConfig,
    *,
    cases: list[OnpHandoffCase],
    legacy_hpop_names: bool,
) -> list[OnpHandoffPacketRecord]:
    packets: list[OnpHandoffPacketRecord] = []
    for case in cases:
        case_start_jd = case.refined_tca_jd_utc - cfg.onp_handoff_half_window_s / 86400.0
        case_end_jd = case.refined_tca_jd_utc + cfg.onp_handoff_half_window_s / 86400.0
        digest = _digest(
            {
                "candidate_id": case.candidate_id,
                "refinement_id": case.refinement_id,
                "force_model_profile": cfg.force_model_profile,
                "case_start_jd_utc": case_start_jd,
                "case_end_jd_utc": case_end_jd,
                "handoff_kind": "manifest_only",
            }
        )
        packet_id = f"hpop_handoff:{digest}"
        packet_path = cfg.onp_handoff_output_dir / "packets" / f"{_safe_stem(packet_id)}.json"
        packet_json = _packet_json(
            cfg,
            case=case,
            packet_id=packet_id,
            case_start_jd=case_start_jd,
            case_end_jd=case_end_jd,
            legacy_hpop_names=legacy_hpop_names,
        )
        packets.append(
            OnpHandoffPacketRecord(
                packet_id=packet_id,
                candidate_id=case.candidate_id,
                refinement_id=case.refinement_id,
                screening_window_id=case.screening_window_id,
                deputy_id=case.deputy_id,
                chief_id=case.chief_id,
                force_model_profile=cfg.force_model_profile,
                case_start_jd_utc=case_start_jd,
                case_end_jd_utc=case_end_jd,
                tca_jd_utc=case.refined_tca_jd_utc,
                miss_distance_km=case.refined_miss_distance_km,
                relative_speed_km_s=case.refined_relative_speed_km_s,
                ranking_score=case.ranking_score,
                output_dir=str(cfg.onp_handoff_output_dir),
                packet_path=str(packet_path),
                packet_json=packet_json,
                input_digest=digest,
                status="planned",
            )
        )
    return packets


def build_hpop_handoff_packets(cfg: ScalePlannerConfig, *, cases: list[OnpHandoffCase]) -> list[OnpHandoffPacketRecord]:
    """Deprecated compatibility alias for :func:`build_onp_handoff_packets`."""

    return _build_handoff_packets(cfg, cases=cases, legacy_hpop_names=True)


def _packet_json(
    cfg: ScalePlannerConfig,
    *,
    case: OnpHandoffCase,
    packet_id: str,
    case_start_jd: float,
    case_end_jd: float,
    legacy_hpop_names: bool,
) -> dict[str, Any]:
    return {
        "schema_id": "oel-onp-handoff-packet-v1",
        "schema_version": 1,
        "packet_id": packet_id,
        "status": "planned",
        "workflow_name": cfg.name,
        "candidate_id": case.candidate_id,
        "refinement_id": case.refinement_id,
        "screening_window_id": case.screening_window_id,
        "objects": {
            "deputy_id": case.deputy_id,
            "chief_id": case.chief_id,
        },
        "source_screening": {
            "source_product_ids": list(case.source_product_ids),
            "ranking_features": dict(case.ranking_features),
        },
        "handoff_case": {
            "propagator_family": "oel_hpop_style" if legacy_hpop_names else "ONP",
            "canonical_propagator_family": "ONP",
            "compatibility_aliases": ["HPOP", "oel_hpop_style"],
            "propagator_name": "ONP",
            "propagator_full_name": "OEL Numerical Propagator",
            "force_model_profile": cfg.force_model_profile,
            "case_start_jd_utc": case_start_jd,
            "case_end_jd_utc": case_end_jd,
            "focus_tca_jd_utc": case.refined_tca_jd_utc,
            "screening_miss_distance_km": case.refined_miss_distance_km,
            "screening_relative_speed_km_s": case.refined_relative_speed_km_s,
            "ranking_score": case.ranking_score,
        },
        "execution": {
            "generated_scenario_path": "",
            "review_output_dir": "",
            "run_status": "not_started",
        },
        "non_claims": [
            "This is a handoff packet, not a validated OEL scenario.",
            "The ONP case has not been executed.",
        ],
    }


def _write_packet_files(
    packets: list[OnpHandoffPacketRecord],
    *,
    writer: Any | None = None,
) -> None:
    for packet in packets:
        path = Path(packet.packet_path)
        serialized = json.dumps(packet.packet_json, indent=2, sort_keys=True) + "\n"
        if writer is None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(serialized, encoding="utf-8")
        else:
            writer.write_text(path, serialized)


def _write_manifest(
    cfg: ScalePlannerConfig,
    *,
    packets: list[OnpHandoffPacketRecord],
    legacy_hpop_names: bool,
    writer: Any | None = None,
) -> Path:
    manifest_path = cfg.onp_handoff_output_dir / "onp_handoff_manifest.json"
    manifest = {
        "schema_id": "oel-onp-handoff-manifest-v1",
        "schema_version": 1,
        "workflow_name": cfg.name,
        "config_path": str(cfg.config_path),
        "store_path": str(cfg.store_path),
        "propagator_name": "ONP",
        "propagator_full_name": "OEL Numerical Propagator",
        "force_model_profile": cfg.force_model_profile,
        "packet_count": len(packets),
        "packets": [
            {
                "packet_id": packet.packet_id,
                "candidate_id": packet.candidate_id,
                "refinement_id": packet.refinement_id,
                "deputy_id": packet.deputy_id,
                "chief_id": packet.chief_id,
                "tca_jd_utc": packet.tca_jd_utc,
                "miss_distance_km": packet.miss_distance_km,
                "relative_speed_km_s": packet.relative_speed_km_s,
                "ranking_score": packet.ranking_score,
                "packet_path": packet.packet_path,
                "status": packet.status,
            }
            for packet in packets
        ],
        "non_claims": [
            "Manifest-only ONP handoff.",
            "No generated scenario configs were validated.",
            "No ONP propagation was executed.",
        ],
    }
    serialized_manifest = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    if writer is None:
        cfg.onp_handoff_output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(serialized_manifest, encoding="utf-8")
    else:
        writer.write_text(manifest_path, serialized_manifest)
    legacy_path = cfg.onp_handoff_output_dir / "hpop_handoff_manifest.json"
    legacy = dict(manifest)
    legacy["deprecated_alias"] = {"name": "HPOP", "replacement": "ONP", "canonical_manifest": str(manifest_path)}
    serialized_legacy = json.dumps(legacy, indent=2, sort_keys=True) + "\n"
    if writer is None:
        legacy_path.write_text(serialized_legacy, encoding="utf-8")
    else:
        writer.write_text(legacy_path, serialized_legacy)
    return legacy_path if legacy_hpop_names else manifest_path


def _safe_stem(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)


def _digest(payload: dict[str, Any]) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
