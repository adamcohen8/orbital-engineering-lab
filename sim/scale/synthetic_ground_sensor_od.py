from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from sim import SimulationConfig, SimulationSession
from sim.data_generation import (
    ground_station_measurement_count,
    offset_cartesian_state,
    state_comparison_payload,
    truth_arc_tracking_network_from_eci_history,
)
from sim.scale.measurements import ingest_ground_station_measurement_payload, run_ground_sensor_od_from_store

_DEFAULT_INITIAL_STATE = (7000.0, 0.0, 0.0, 0.0, 7.546053290107541, 0.0)
_DEFAULT_EPOCH_JD = 2460000.5


def run_synthetic_ground_sensor_od_workflow(
    *,
    output_dir: str | Path,
    object_id: str = "synthetic_native_target",
    store_path: str | Path | None = None,
    initial_state_eci_km_s: Sequence[float] = _DEFAULT_INITIAL_STATE,
    prior_position_offset_m: Sequence[float] = (1000.0, 0.0, 0.0),
    prior_velocity_offset_mm_s: Sequence[float] = (100.0, 0.0, 0.0),
    duration_s: float = 3600.0,
    step_s: float = 120.0,
    station_spacing_s: float | None = None,
    station_longitude_offset_deg: float = 2.0,
    range_sigma_km: float = 0.01,
    range_rate_sigma_km_s: float = 1.0e-5,
    angle_sigma_deg: float = 0.02,
    fit_duration_s: float | None = None,
    holdout_duration_s: float | None = None,
    dynamics_model: str = "two_body",
    j2: bool = False,
    j3: bool = False,
    j4: bool = False,
    drag: bool = False,
    srp: bool = False,
    third_body_sun: bool = False,
    third_body_moon: bool = False,
    atmosphere_model: str | None = None,
    max_nfev: int = 48,
) -> dict[str, Any]:
    """Run a compact synthetic native ground-station sensor OD workflow."""

    truth_state = np.array(initial_state_eci_km_s, dtype=float).reshape(6)
    prior_state = offset_cartesian_state(truth_state, prior_position_offset_m, prior_velocity_offset_mm_s)
    duration = float(duration_s)
    step = float(step_s)
    if duration <= 0.0:
        raise ValueError("duration_s must be positive.")
    if step <= 0.0:
        raise ValueError("step_s must be positive.")
    fit_duration = float(fit_duration_s) if fit_duration_s is not None else max(step, min(duration, duration * 2.0 / 3.0))
    holdout_duration = float(holdout_duration_s) if holdout_duration_s is not None else max(0.0, duration - fit_duration)
    if fit_duration <= 0.0 or fit_duration > duration + 1.0e-9:
        raise ValueError("fit_duration_s must be positive and no greater than duration_s.")
    if holdout_duration < 0.0:
        raise ValueError("holdout_duration_s must be non-negative.")

    output_root = Path(output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    store = Path(store_path).expanduser().resolve() if store_path is not None else output_root / "catalog_store.sqlite"
    force_model = {
        "dynamics_model": str(dynamics_model or "two_body"),
        "j2": bool(j2),
        "j3": bool(j3),
        "j4": bool(j4),
        "drag": bool(drag),
        "srp": bool(srp),
        "third_body_sun": bool(third_body_sun),
        "third_body_moon": bool(third_body_moon),
        "atmosphere_model": str(atmosphere_model or ""),
    }

    truth_probe = _scenario(
        output_root,
        object_id=object_id,
        initial_state=truth_state,
        duration_s=duration,
        step_s=step,
        force_model=force_model,
        ground_stations=[],
        output_name="truth_probe",
    )
    truth_probe_result = SimulationSession.from_config(SimulationConfig.from_dict(truth_probe)).run()
    station_network = truth_arc_tracking_network_from_eci_history(
        truth_probe_result.truth[object_id],
        duration_s=duration,
        step_s=step,
        station_spacing_s=station_spacing_s if station_spacing_s is not None else step,
        station_longitude_offset_deg=station_longitude_offset_deg,
        range_sigma_km=range_sigma_km,
        range_rate_sigma_km_s=range_rate_sigma_km_s,
        angle_sigma_deg=angle_sigma_deg,
        epoch_jd_utc=_DEFAULT_EPOCH_JD,
    )
    measurement_scenario = _scenario(
        output_root,
        object_id=object_id,
        initial_state=truth_state,
        duration_s=duration,
        step_s=step,
        force_model=force_model,
        ground_stations=station_network,
        output_name="simulation",
    )
    scenario_path = output_root / "synthetic_ground_sensor_od_scenario.json"
    scenario_path.write_text(json.dumps(measurement_scenario, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    sim_result = SimulationSession.from_config(SimulationConfig.from_dict(measurement_scenario)).run()
    measurement_payload = {"ground_station_measurements": sim_result.ground_station_measurements}
    measurement_payload_path = output_root / "synthetic_ground_station_measurements.json"
    measurement_payload_path.write_text(json.dumps(measurement_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    measurement_count = ground_station_measurement_count(measurement_payload)
    if measurement_count < 2:
        raise ValueError("synthetic scenario produced fewer than two ground-station measurements.")

    ingest = ingest_ground_station_measurement_payload(
        measurement_payload,
        store_path=store,
        source_label="synthetic_ground_sensor_od_workflow",
    )
    od_dir = output_root / "ground_sensor_od"
    od_manifest = run_ground_sensor_od_from_store(
        store,
        object_id=object_id,
        output_dir=od_dir,
        initial_state_eci_km_s=tuple(float(item) for item in prior_state),
        initial_state_source="synthetic_truth_offset_prior",
        fit_duration_s=fit_duration,
        holdout_duration_s=holdout_duration,
        dt_s=step,
        dynamics_model=force_model["dynamics_model"],
        j2=force_model["j2"],
        j3=force_model["j3"],
        j4=force_model["j4"],
        drag=force_model["drag"],
        srp=force_model["srp"],
        third_body_sun=force_model["third_body_sun"],
        third_body_moon=force_model["third_body_moon"],
        atmosphere_model=force_model["atmosphere_model"] or None,
        max_nfev=int(max_nfev),
    )
    fitted_state = np.array(od_manifest.get("fitted_state_eci_km_s", []) or [], dtype=float)
    if fitted_state.size != 6:
        report_path = Path(str(od_manifest.get("native_sensor_od_report_json_path", ""))).expanduser()
        if report_path.is_file():
            fitted_state = np.array(json.loads(report_path.read_text(encoding="utf-8"))["fitted_state_eci_km_s"], dtype=float)
    comparison = state_comparison_payload(truth_state=truth_state, prior_state=prior_state, fitted_state=fitted_state)
    payload = {
        "schema_version": 1,
        "status": "completed",
        "workflow": "synthetic_ground_sensor_od",
        "object_id": object_id,
        "store_path": str(store),
        "simulation": {
            "duration_s": duration,
            "step_s": step,
            "station_count": len(station_network),
            "measurement_count": measurement_count,
            "scenario_path": str(scenario_path),
            "measurement_payload_path": str(measurement_payload_path),
        },
        "truth": {"initial_state_eci_km_s": truth_state.tolist()},
        "prior": {
            "initial_state_eci_km_s": prior_state.tolist(),
            "position_offset_m": [float(item) for item in prior_position_offset_m],
            "velocity_offset_mm_s": [float(item) for item in prior_velocity_offset_mm_s],
        },
        "noise": {
            "angle_sigma_deg": float(angle_sigma_deg),
            "range_sigma_km": float(range_sigma_km),
            "range_rate_sigma_km_s": float(range_rate_sigma_km_s),
        },
        "force_model": force_model,
        "od": {
            "fit_duration_s": fit_duration,
            "holdout_duration_s": holdout_duration,
            "manifest_path": od_manifest.get("manifest_path", ""),
            "report_json_path": od_manifest.get("native_sensor_od_report_json_path", ""),
            "report_md_path": od_manifest.get("native_sensor_od_report_md_path", ""),
            "fitted_mission_input_packet_path": od_manifest.get("fitted_mission_input_packet_path", ""),
            "residual_csv_path": od_manifest.get("residual_csv_path", ""),
            "residual_plot_path": od_manifest.get("residual_plot_path", ""),
            "quality_gates": dict(od_manifest.get("quality_gates", {}) or {}),
            "verdict": dict(od_manifest.get("verdict", {}) or {}),
            "fit_metrics": dict(od_manifest.get("fit_metrics", {}) or {}),
            "holdout_metrics": dict(od_manifest.get("holdout_metrics", {}) or {}),
        },
        "comparison": comparison,
        "ingest": ingest,
        "artifacts": {},
        "non_claims": [
            "This is a synthetic workflow for exercising native OEL ground-station sensor OD.",
            "The synthetic tracking network is generated for deterministic measurement coverage, not as an operational sensor model.",
            "This convenience wrapper uses the default state-only solve; configure the native ground-sensor API or Scale fit command explicitly for Phase 3 bias, clock, weighting, or refraction models.",
        ],
    }
    report_json = output_root / "synthetic_ground_sensor_od_workflow.json"
    report_md = output_root / "synthetic_ground_sensor_od_workflow.md"
    payload["artifacts"] = {
        "workflow_json": str(report_json),
        "workflow_markdown": str(report_md),
        "scenario_json": str(scenario_path),
        "measurement_payload_json": str(measurement_payload_path),
    }
    report_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_md.write_text(_render_report(payload), encoding="utf-8")
    return payload


def _scenario(
    output_root: Path,
    *,
    object_id: str,
    initial_state: np.ndarray,
    duration_s: float,
    step_s: float,
    force_model: dict[str, Any],
    ground_stations: list[dict[str, Any]],
    output_name: str,
) -> dict[str, Any]:
    return {
        "scenario_name": f"synthetic_ground_sensor_od_{output_name}",
        "objects": {
            object_id: {
                "kind": "satellite",
                "enabled": True,
                "role": "tracked_satellite",
                "specs": {"mass_kg": 120.0},
                "initial_state": {
                    "position_eci_km": [float(item) for item in initial_state[:3]],
                    "velocity_eci_km_s": [float(item) for item in initial_state[3:]],
                },
            }
        },
        "ground_stations": ground_stations,
        "simulator": {
            "duration_s": float(duration_s),
            "dt_s": float(step_s),
            "initial_jd_utc": _DEFAULT_EPOCH_JD,
            "dynamics": {
                "orbit": {
                    "model": str(force_model["dynamics_model"]),
                    "orbit_substep_s": float(step_s),
                    "j2": bool(force_model["j2"]),
                    "j3": bool(force_model["j3"]),
                    "j4": bool(force_model["j4"]),
                    "drag": bool(force_model["drag"]),
                    "srp": bool(force_model["srp"]),
                    "third_body_sun": bool(force_model["third_body_sun"]),
                    "third_body_moon": bool(force_model["third_body_moon"]),
                    "atmosphere_model": str(force_model.get("atmosphere_model", "") or ""),
                },
                "attitude": {"enabled": False},
                "rocket": {"enabled": False},
            },
            "environment": {"ephemeris_mode": "analytic_simple", "atmosphere_env": {}},
            "termination": {"earth_impact_enabled": False},
        },
        "outputs": {
            "output_dir": str(output_root / output_name),
            "mode": "save",
            "stats": {"print_summary": False, "save_json": True, "save_full_log": False},
            "plots": {"enabled": False, "figure_ids": []},
            "animations": {"enabled": False, "types": []},
        },
    }


def _render_report(payload: dict[str, Any]) -> str:
    simulation = dict(payload.get("simulation", {}) or {})
    noise = dict(payload.get("noise", {}) or {})
    od = dict(payload.get("od", {}) or {})
    fit = dict(od.get("fit_metrics", {}) or {})
    holdout = dict(od.get("holdout_metrics", {}) or {})
    quality = dict(od.get("quality_gates", {}) or {})
    comparison = dict(payload.get("comparison", {}) or {})
    lines = [
        "# Synthetic Ground-Sensor OD Workflow",
        "",
        f"- Status: `{payload.get('status')}`",
        f"- Object ID: `{payload.get('object_id')}`",
        f"- Measurements: `{simulation.get('measurement_count')}` from `{simulation.get('station_count')}` synthetic sites",
        f"- Force model: `{payload.get('force_model')}`",
        f"- Noise: `{noise}`",
        f"- Fit weighted RMS: `{fit.get('weighted_rms')}`",
        f"- Holdout weighted RMS: `{holdout.get('weighted_rms')}`",
        f"- Solver success: `{quality.get('solver_success')}`",
        f"- Prior position error m: `{comparison.get('prior_position_error_m')}`",
        f"- Fitted position error m: `{comparison.get('fitted_position_error_m')}`",
        f"- Prior velocity error mm/s: `{comparison.get('prior_velocity_error_mm_s')}`",
        f"- Fitted velocity error mm/s: `{comparison.get('fitted_velocity_error_mm_s')}`",
        "",
        "## Artifacts",
        "",
        f"- Workflow JSON: `{dict(payload.get('artifacts', {}) or {}).get('workflow_json')}`",
        f"- Scenario JSON: `{simulation.get('scenario_path')}`",
        f"- Measurement payload: `{simulation.get('measurement_payload_path')}`",
        f"- OD manifest: `{od.get('manifest_path')}`",
        f"- Residual CSV: `{od.get('residual_csv_path')}`",
        "",
        "## Non-Claims",
        "",
    ]
    lines.extend(f"- {item}" for item in list(payload.get("non_claims", []) or []))
    return "\n".join(lines) + "\n"
