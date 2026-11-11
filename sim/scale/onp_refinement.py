from __future__ import annotations

import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from sim.api import SimulationWorkspace
from sim.dynamics.orbit.ogp import ogp_propagate_teme_batch_accelerated, ogp_regime_for_elements
from sim.dynamics.orbit.sgp4 import transform_teme_to_output_frame
from sim.scale._output import bind_output_writer, staged_sqlite_store
from sim.scale.onp_identity import (
    ONP_INITIAL_STATE_FRAME,
    ONP_INITIAL_STATE_FRAME_TRANSFORM,
    ONP_REFINEMENT_SCHEMA_VERSION,
    onp_materialization_signature,
)
from sim.scale.planner import (
    ScalePlannerConfig,
    enforce_scale_planner_limits,
    load_scale_planner_config,
    resolve_selected_object_ids,
)
from sim.scale.store import (
    CurrentTLEState,
    HpopHandoffPacketRecord,
    OnpRefinementCaseRecord,
    current_ogp_elements,
    initialize_store,
    load_current_tle_states,
    load_onp_refinement_cases,
    load_pending_onp_refinement_packets,
    write_onp_refinement_cases,
)
from sim.security.output_writes import OutputWriter


@enforce_scale_planner_limits("ONP refinement materialization")
def run_onp_refinement_materialization(
    config_path: str | Path,
    *,
    writer: OutputWriter | None = None,
) -> dict[str, Any]:
    """Materialize and validate ordinary ONP scenarios from current Scale handoff packets."""

    cfg = load_scale_planner_config(config_path)
    if cfg.mode != "run":
        raise ValueError("ONP refinement materialization requires scale_catalog.mode: run.")
    if not cfg.hpop_handoff_enabled:
        raise ValueError("ONP refinement materialization requires ONP handoff to be enabled.")

    with staged_sqlite_store(cfg.store_path, writer=writer) as store_path:
        store_cfg = replace(cfg, store_path=store_path)
        initialize_store(store_path, description=f"OEL Scale store for {cfg.name}")
        packets = load_pending_onp_refinement_packets(
            store_path,
            object_ids=resolve_selected_object_ids(cfg),
            limit=cfg.hpop_handoff_limit,
            materialization_signature=onp_materialization_signature(
                refinement_step_s=cfg.refinement_step_s,
            ),
        )
        object_ids = tuple(sorted({packet.deputy_id for packet in packets} | {packet.chief_id for packet in packets}))
        states = {state.object_id: state for state in load_current_tle_states(store_path, object_ids=object_ids)}
        binding = bind_output_writer(writer, cfg.hpop_handoff_output_dir)
        cases = [
            materialize_onp_refinement_case(cfg, packet=packet, states=states, writer=binding)
            for packet in packets
        ]
        write_summary = write_onp_refinement_cases(store_cfg.store_path, cases=cases)
        failed = [case for case in cases if case.validation_status != "ok"]
        manifest_cases = load_onp_refinement_cases(
            store_cfg.store_path,
            object_ids=resolve_selected_object_ids(cfg),
        )
        manifest_path = _write_manifest(cfg, cases=manifest_cases, writer=binding)
    return {
        "schema_version": ONP_REFINEMENT_SCHEMA_VERSION,
        "status": "completed" if not failed else "failed",
        "workflow": {
            "name": cfg.name,
            "config_path": str(cfg.config_path),
            "mode": cfg.mode,
        },
        "store_path": str(cfg.store_path),
        "output_dir": str(_materialization_root(cfg)),
        "manifest_path": str(manifest_path),
        "packets_seen": len(packets),
        "cases_materialized": len(cases),
        "cases_inserted": write_summary["cases_inserted"],
        "cases_updated": write_summary["cases_updated"],
        "validated_cases": len(cases) - len(failed),
        "failed_validation_cases": len(failed),
        "initial_state_frame": ONP_INITIAL_STATE_FRAME,
        "frame_transform": ONP_INITIAL_STATE_FRAME_TRANSFORM,
        "non_claims": [
            "Generated ONP scenarios were validated but not executed.",
            "Materialization does not produce ONP propagation or OGP-vs-ONP comparison evidence.",
        ],
    }


def materialize_onp_refinement_case(
    cfg: ScalePlannerConfig,
    *,
    packet: HpopHandoffPacketRecord,
    states: dict[str, CurrentTLEState],
    writer: Any | None = None,
) -> OnpRefinementCaseRecord:
    state_rows = []
    for object_id in (packet.deputy_id, packet.chief_id):
        state = states.get(object_id)
        if state is None:
            raise ValueError(f"ONP refinement packet {packet.packet_id!r} has no current TLE state for {object_id!r}.")
        state_rows.append(state)

    source_product_ids = tuple(
        str(item) for item in dict(packet.packet_json.get("source_screening", {}) or {}).get("source_product_ids", [])
    )
    state_digest_map = {state.object_id: state.input_digest for state in state_rows}
    materialization_signature = onp_materialization_signature(
        refinement_step_s=cfg.refinement_step_s,
    )
    input_digest = _digest(
        {
            "schema_version": ONP_REFINEMENT_SCHEMA_VERSION,
            "packet_id": packet.packet_id,
            "packet_input_digest": packet.input_digest,
            "force_model_profile": packet.force_model_profile,
            "analysis_start_jd_utc": packet.case_start_jd_utc,
            "analysis_end_jd_utc": packet.case_end_jd_utc,
            "initial_state_frame": ONP_INITIAL_STATE_FRAME,
            "frame_transform": ONP_INITIAL_STATE_FRAME_TRANSFORM,
            "materialization_signature": materialization_signature,
            "source_product_ids": list(source_product_ids),
            "source_state_digests": state_digest_map,
            "ogp_backend": cfg.propagation_backend,
        }
    )
    case_id = f"onp_refinement:{input_digest}"
    case_stem = _safe_stem(case_id)
    root = _materialization_root(cfg)
    case_dir = root / "cases" / case_stem
    config_path = case_dir / "scenario.yaml"
    scenario_output_dir = case_dir / "run"

    initial_states, state_provenance = _derive_initial_states(
        state_rows,
        target_jd_utc=packet.case_start_jd_utc,
        backend=cfg.propagation_backend,
    )
    scenario = _build_scenario(
        cfg,
        packet=packet,
        case_id=case_id,
        initial_states=initial_states,
        state_provenance=state_provenance,
        source_product_ids=source_product_ids,
        scenario_output_dir=scenario_output_dir,
    )
    config_text = yaml.safe_dump(scenario, sort_keys=False)
    if writer is None:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(config_text, encoding="utf-8")
        config_digest = hashlib.sha256(config_path.read_bytes()).hexdigest()
    else:
        writer.write_text(config_path, config_text)
        config_digest = writer.sha256(config_path)

    validation_path = config_path if writer is None else writer.physical(config_path)
    validation = SimulationWorkspace().validate_candidate_config(validation_path, trust_plugins=True)
    validation = _json_safe(validation)
    validation_status = "ok" if bool(validation.get("ok", False)) else "failed"
    status = "validated" if validation_status == "ok" else "validation_failed"
    warnings = tuple(str(item) for item in packet.warnings)
    return OnpRefinementCaseRecord(
        case_id=case_id,
        packet_id=packet.packet_id,
        packet_input_digest=packet.input_digest,
        candidate_id=packet.candidate_id,
        refinement_id=packet.refinement_id,
        screening_window_id=packet.screening_window_id,
        deputy_id=packet.deputy_id,
        chief_id=packet.chief_id,
        force_model_profile=packet.force_model_profile,
        analysis_start_jd_utc=packet.case_start_jd_utc,
        analysis_end_jd_utc=packet.case_end_jd_utc,
        initial_state_frame=ONP_INITIAL_STATE_FRAME,
        frame_transform=ONP_INITIAL_STATE_FRAME_TRANSFORM,
        source_product_ids=source_product_ids,
        source_state_provenance=state_provenance,
        generated_config_path=str(config_path),
        config_digest=config_digest,
        scenario_output_dir=str(scenario_output_dir),
        validation_status=validation_status,
        validation_report=validation,
        materialization_signature=materialization_signature,
        input_digest=input_digest,
        status=status,
        warnings=warnings,
    )


def _derive_initial_states(
    states: list[CurrentTLEState],
    *,
    target_jd_utc: float,
    backend: str = "scalar",
) -> tuple[dict[str, list[float]], dict[str, Any]]:
    elements = [current_ogp_elements(state) for state in states]
    time_grid = np.array(
        [[(float(target_jd_utc) - float(element.epoch_jd_utc)) * 1440.0] for element in elements],
        dtype=float,
    )
    batch = ogp_propagate_teme_batch_accelerated(
        elements, time_grid, backend="rust" if backend == "rust" else "auto",
    )
    initial_states: dict[str, list[float]] = {}
    provenance: dict[str, Any] = {}
    for index, (state, element) in enumerate(zip(states, elements, strict=True)):
        error = str(batch.errors[index, 0] or "")
        if error:
            raise ValueError(f"OGP could not derive the ONP initial state for {state.object_id!r}: {error}")
        position_eci, velocity_eci = transform_teme_to_output_frame(
            batch.position_teme_km[index, 0, :],
            batch.velocity_teme_km_s[index, 0, :],
            jd_utc=float(target_jd_utc),
            output_frame=ONP_INITIAL_STATE_FRAME,
            frame_transform=ONP_INITIAL_STATE_FRAME_TRANSFORM,
        )
        state_vector = [
            *(float(value) for value in position_eci),
            *(float(value) for value in velocity_eci),
        ]
        initial_states[state.object_id] = state_vector
        provenance[state.object_id] = {
            "source_kind": "scale_current_tle_state",
            "source_od_solution_id": state.od_solution_id,
            "source_state_input_digest": state.input_digest,
            "source_tle_epoch_jd_utc": state.tle_epoch_jd_utc,
            "target_epoch_jd_utc": float(target_jd_utc),
            "propagator_family": "OGP",
            "ogp_regime": ogp_regime_for_elements(element),
            "runtime_backend": str(batch.backend),
            "native_frame": "teme",
            "output_frame": ONP_INITIAL_STATE_FRAME,
            "frame_transform": ONP_INITIAL_STATE_FRAME_TRANSFORM,
            "state_eci_km_km_s": state_vector,
        }
    return initial_states, provenance


def _build_scenario(
    cfg: ScalePlannerConfig,
    *,
    packet: HpopHandoffPacketRecord,
    case_id: str,
    initial_states: dict[str, list[float]],
    state_provenance: dict[str, Any],
    source_product_ids: tuple[str, ...],
    scenario_output_dir: Path,
) -> dict[str, Any]:
    duration_s = max(
        0.0,
        (float(packet.case_end_jd_utc) - float(packet.case_start_jd_utc)) * 86400.0,
    )
    if duration_s <= 0.0:
        raise ValueError(f"ONP refinement packet {packet.packet_id!r} has an empty analysis window.")
    requested_dt_s = min(float(cfg.refinement_step_s), duration_s)
    interval_count = max(1, int(math.ceil(duration_s / requested_dt_s)))
    dt_s = duration_s / interval_count
    dynamics = _force_model_dynamics(packet.force_model_profile, orbit_substep_s=dt_s)
    deputy_key = _object_key("deputy", packet.deputy_id)
    chief_key = _object_key("chief", packet.chief_id)

    def object_section(
        *,
        key: str,
        source_id: str,
        role: str,
        relative_to: str = "",
    ) -> dict[str, Any]:
        state = initial_states[source_id]
        initial_state = {
            "position_eci_km": state[:3],
            "velocity_eci_km_s": state[3:],
        }
        if relative_to:
            initial_state["relative_to"] = relative_to
        return {
            "enabled": True,
            "object_id": key,
            "kind": "satellite",
            "role": role,
            "initial_state": initial_state,
            "flight_software": {
                "stack": "fsw.passive",
                "hardware_profile": "hardware.passive.v1",
            },
        }

    return {
        "scenario_name": _safe_stem(case_id),
        "scenario_description": "Scale-generated focused ONP refinement scenario.",
        "metadata": {
            "owner": "private_scale",
            "support_level": "generated_candidate",
            "scale_provenance": {
                "schema_version": ONP_REFINEMENT_SCHEMA_VERSION,
                "case_id": case_id,
                "packet_id": packet.packet_id,
                "candidate_id": packet.candidate_id,
                "refinement_id": packet.refinement_id,
                "screening_window_id": packet.screening_window_id,
                "source_store_path": str(cfg.store_path),
                "source_product_ids": list(source_product_ids),
                "object_map": {
                    deputy_key: packet.deputy_id,
                    chief_key: packet.chief_id,
                },
                "force_model_profile": packet.force_model_profile,
                "analysis_start_jd_utc": packet.case_start_jd_utc,
                "analysis_end_jd_utc": packet.case_end_jd_utc,
                "focus_tca_jd_utc": packet.tca_jd_utc,
                "screening_miss_distance_km": packet.miss_distance_km,
                "screening_relative_speed_km_s": packet.relative_speed_km_s,
                "initial_state_provenance": state_provenance,
            },
        },
        "objects": {
            deputy_key: object_section(
                key=deputy_key,
                source_id=packet.deputy_id,
                role="scale_deputy",
                relative_to=chief_key,
            ),
            chief_key: object_section(key=chief_key, source_id=packet.chief_id, role="scale_chief"),
        },
        "simulator": {
            "duration_s": duration_s,
            "dt_s": dt_s,
            "initial_jd_utc": float(packet.case_start_jd_utc),
            "dynamics": dynamics,
            "environment": {
                "ephemeris_mode": "analytic_simple",
                "atmosphere_env": {},
            },
            "plugin_validation": {"strict": True},
            "termination": {
                "earth_impact_enabled": True,
                "earth_radius_km": 6378.137,
            },
        },
        "outputs": {
            "output_dir": str(scenario_output_dir),
            "mode": "save",
            "stats": {
                "enabled": True,
                "print_summary": False,
                "save_json": True,
                "save_csv": False,
                "save_full_log": False,
            },
            "plots": {"enabled": False, "figure_ids": []},
            "animations": {"enabled": False, "types": []},
            "review": {"enabled": True, "detail": "standard"},
        },
    }


def _force_model_dynamics(profile: str, *, orbit_substep_s: float) -> dict[str, Any]:
    key = str(profile or "quick_numerical").strip().lower()
    flags_by_profile = {
        "quick_numerical": {},
        "j2": {"j2": True},
        "j2_j3_j4": {"j2": True, "j3": True, "j4": True},
    }
    if key not in flags_by_profile:
        supported = ", ".join(sorted(flags_by_profile))
        raise ValueError(f"Unsupported Scale ONP force_model_profile {profile!r}; supported profiles: {supported}.")
    orbit = {
        "model": "two_body",
        "orbit_substep_s": float(orbit_substep_s),
        "j2": False,
        "j3": False,
        "j4": False,
        "drag": False,
        "srp": False,
        "third_body_sun": False,
        "third_body_moon": False,
    }
    orbit.update(flags_by_profile[key])
    return {
        "orbit": orbit,
        "attitude": {"enabled": False},
        "rocket": {"enabled": False},
    }


def _materialization_root(cfg: ScalePlannerConfig) -> Path:
    return cfg.hpop_handoff_output_dir / "onp_refinement"


def _write_manifest(
    cfg: ScalePlannerConfig,
    *,
    cases: list[OnpRefinementCaseRecord],
    writer: Any | None = None,
) -> Path:
    root = _materialization_root(cfg)
    path = root / "onp_refinement_materialization_manifest.json"
    payload = {
        "schema_version": ONP_REFINEMENT_SCHEMA_VERSION,
        "workflow_name": cfg.name,
        "source_store_path": str(cfg.store_path),
        "initial_state_frame": ONP_INITIAL_STATE_FRAME,
        "frame_transform": ONP_INITIAL_STATE_FRAME_TRANSFORM,
        "case_count": len(cases),
        "cases": [
            {
                "case_id": case.case_id,
                "packet_id": case.packet_id,
                "candidate_id": case.candidate_id,
                "generated_config_path": case.generated_config_path,
                "config_digest": case.config_digest,
                "scenario_output_dir": case.scenario_output_dir,
                "validation_status": case.validation_status,
                "status": case.status,
                "source_product_ids": list(case.source_product_ids),
            }
            for case in cases
        ],
        "non_claims": [
            "Generated ONP scenarios were validated but not executed.",
            "No ONP propagation or OGP-vs-ONP comparison evidence was produced.",
        ],
    }
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if writer is None:
        root.mkdir(parents=True, exist_ok=True)
        path.write_text(serialized, encoding="utf-8")
    else:
        writer.write_text(path, serialized)
    return path


def _object_key(role: str, object_id: str) -> str:
    readable = "".join(char.lower() if char.isalnum() else "_" for char in str(object_id)).strip("_")
    readable = readable[:32] or "object"
    suffix = hashlib.sha256(str(object_id).encode("utf-8")).hexdigest()[:8]
    return f"{role}_{readable}_{suffix}"


def _safe_stem(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value)


def _digest(payload: dict[str, Any]) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _json_safe(payload: dict[str, Any]) -> dict[str, Any]:
    return dict(json.loads(json.dumps(payload, default=str)))
