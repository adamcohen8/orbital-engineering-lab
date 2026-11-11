from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sim import SimulationConfig, SimulationSession
from sim.config import validate_scenario_plugins
from sim.data_generation import (
    ground_station_measurement_count,
    tle_element_comparison_payload,
    tle_line2_with_mean_fields,
    truth_arc_tracking_network_from_sgp4,
)
from sim.dynamics.orbit.ogp import ogp_propagator_name_for_elements, ogp_regime_for_elements
from sim.dynamics.orbit.tle import parse_tle_lines
from sim.estimation.sgp4_od import DEFAULT_SGP4_SENSOR_FRAME_TRANSFORM
from sim.scale.measurements import (
    ingest_ground_station_measurement_payload,
    run_ogp_ground_sensor_od_from_store,
    run_sgp4_ground_sensor_od_from_store,
)


def run_synthetic_sgp4_sensor_od_workflow(
    *,
    prior_line1: str,
    prior_line2: str,
    output_dir: str | Path,
    object_id: str = "synthetic_sgp4_target",
    store_path: str | Path | None = None,
    mean_anomaly_offset_deg: float = 0.2,
    mean_motion_scale: float = 1.00002,
    duration_s: float = 720.0,
    step_s: float = 120.0,
    station_spacing_s: float | None = None,
    station_longitude_offset_deg: float = 2.0,
    range_sigma_km: float = 0.01,
    range_rate_sigma_km_s: float = 1.0e-5,
    angle_sigma_deg: float = 0.02,
    fit_duration_s: float | None = None,
    holdout_duration_s: float | None = None,
    frame_transform: str = DEFAULT_SGP4_SENSOR_FRAME_TRANSFORM,
    max_nfev: int = 160,
) -> dict[str, Any]:
    """Fit near-Earth SGP4 observations through the shared owner."""
    return _run_synthetic_mean_element_sensor_od_workflow(
        prior_line1=prior_line1,
        prior_line2=prior_line2,
        output_dir=output_dir,
        object_id=object_id,
        store_path=store_path,
        mean_anomaly_offset_deg=mean_anomaly_offset_deg,
        mean_motion_scale=mean_motion_scale,
        duration_s=duration_s,
        step_s=step_s,
        station_spacing_s=station_spacing_s,
        station_longitude_offset_deg=station_longitude_offset_deg,
        range_sigma_km=range_sigma_km,
        range_rate_sigma_km_s=range_rate_sigma_km_s,
        angle_sigma_deg=angle_sigma_deg,
        fit_duration_s=fit_duration_s,
        holdout_duration_s=holdout_duration_s,
        frame_transform=frame_transform,
        max_nfev=max_nfev,
        ogp_dispatch=False,
    )


def run_synthetic_ogp_sensor_od_workflow(
    *,
    prior_line1: str,
    prior_line2: str,
    output_dir: str | Path,
    object_id: str = "synthetic_sgp4_target",
    store_path: str | Path | None = None,
    mean_anomaly_offset_deg: float = 0.2,
    mean_motion_scale: float = 1.00002,
    duration_s: float = 720.0,
    step_s: float = 120.0,
    station_spacing_s: float | None = None,
    station_longitude_offset_deg: float = 2.0,
    range_sigma_km: float = 0.01,
    range_rate_sigma_km_s: float = 1.0e-5,
    angle_sigma_deg: float = 0.02,
    fit_duration_s: float | None = None,
    holdout_duration_s: float | None = None,
    frame_transform: str = DEFAULT_SGP4_SENSOR_FRAME_TRANSFORM,
    max_nfev: int = 160,
) -> dict[str, Any]:
    """Fit regime-dispatched OGP observations through the shared owner."""
    return _run_synthetic_mean_element_sensor_od_workflow(
        prior_line1=prior_line1,
        prior_line2=prior_line2,
        output_dir=output_dir,
        object_id=object_id,
        store_path=store_path,
        mean_anomaly_offset_deg=mean_anomaly_offset_deg,
        mean_motion_scale=mean_motion_scale,
        duration_s=duration_s,
        step_s=step_s,
        station_spacing_s=station_spacing_s,
        station_longitude_offset_deg=station_longitude_offset_deg,
        range_sigma_km=range_sigma_km,
        range_rate_sigma_km_s=range_rate_sigma_km_s,
        angle_sigma_deg=angle_sigma_deg,
        fit_duration_s=fit_duration_s,
        holdout_duration_s=holdout_duration_s,
        frame_transform=frame_transform,
        max_nfev=max_nfev,
        ogp_dispatch=True,
    )


def _run_synthetic_mean_element_sensor_od_workflow(
    *,
    prior_line1: str,
    prior_line2: str,
    output_dir: str | Path,
    object_id: str = "synthetic_sgp4_target",
    store_path: str | Path | None = None,
    mean_anomaly_offset_deg: float = 0.2,
    mean_motion_scale: float = 1.00002,
    duration_s: float = 720.0,
    step_s: float = 120.0,
    station_spacing_s: float | None = None,
    station_longitude_offset_deg: float = 2.0,
    range_sigma_km: float = 0.01,
    range_rate_sigma_km_s: float = 1.0e-5,
    angle_sigma_deg: float = 0.02,
    fit_duration_s: float | None = None,
    holdout_duration_s: float | None = None,
    frame_transform: str = DEFAULT_SGP4_SENSOR_FRAME_TRANSFORM,
    max_nfev: int = 160,
    ogp_dispatch: bool,
) -> dict[str, Any]:
    """Run a compact synthetic SGP4 sensor-OD workflow.

    The workflow uses the prior TLE as the OD template, creates a synthetic
    truth TLE by applying bounded mean-element offsets, simulates geometric
    ground-station measurements from that truth, ingests them into a Scale
    store, and fits an SGP4 mean-element product back from the prior.
    """

    prefix = "ogp" if ogp_dispatch else "sgp4"
    prior = parse_tle_lines(prior_line1, prior_line2)
    duration = float(duration_s)
    step = float(step_s)
    if duration <= 0.0:
        raise ValueError("duration_s must be positive.")
    if step <= 0.0:
        raise ValueError("step_s must be positive.")
    fit_duration = (
        float(fit_duration_s) if fit_duration_s is not None else max(step, min(duration, duration * 2.0 / 3.0))
    )
    holdout_duration = (
        float(holdout_duration_s) if holdout_duration_s is not None else max(0.0, duration - fit_duration)
    )
    if fit_duration <= 0.0 or fit_duration > duration + 1.0e-9:
        raise ValueError("fit_duration_s must be positive and no greater than duration_s.")
    if holdout_duration < 0.0:
        raise ValueError("holdout_duration_s must be non-negative.")

    output_root = Path(output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    store = Path(store_path).expanduser().resolve() if store_path is not None else output_root / "catalog_store.sqlite"
    truth_line2 = tle_line2_with_mean_fields(
        str(prior.line2),
        mean_anomaly_deg=float(prior.mean_anomaly_deg) + float(mean_anomaly_offset_deg),
        mean_motion_rev_per_day=float(prior.mean_motion_rev_per_day) * float(mean_motion_scale),
    )
    truth = parse_tle_lines(str(prior.line1), truth_line2)
    if ogp_dispatch and ogp_regime_for_elements(truth) != ogp_regime_for_elements(prior):
        raise ValueError("Synthetic OGP truth must stay in the prior TLE propagation regime.")
    scenario = _synthetic_sgp4_sensor_scenario(
        output_root,
        object_id=object_id,
        truth_line1=str(prior.line1),
        truth_line2=truth_line2,
        duration_s=duration,
        step_s=step,
        station_spacing_s=station_spacing_s,
        station_longitude_offset_deg=station_longitude_offset_deg,
        range_sigma_km=range_sigma_km,
        range_rate_sigma_km_s=range_rate_sigma_km_s,
        angle_sigma_deg=angle_sigma_deg,
        frame_transform=frame_transform,
    )
    if ogp_dispatch:
        scenario["scenario_name"] = "synthetic_ogp_sensor_od_workflow"
        scenario["outputs"]["review"] = {"enabled": True, "detail": "standard"}
    scenario_path = output_root / f"synthetic_{prefix}_sensor_od_scenario.json"
    scenario_path.write_text(json.dumps(scenario, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # Validate the trusted generated scenario before starting the engine.
    config = SimulationConfig.from_dict(scenario)
    validation_errors = validate_scenario_plugins(config.to_scenario_config())
    if validation_errors:
        raise ValueError("Synthetic sensor scenario validation failed: " + "; ".join(validation_errors))
    session = SimulationSession.from_config(config)
    sim_result = session.run()
    measurement_payload = {"ground_station_measurements": sim_result.ground_station_measurements}
    measurement_payload_path = output_root / "synthetic_ground_station_measurements.json"
    measurement_payload_path.write_text(
        json.dumps(measurement_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    measurement_count = ground_station_measurement_count(measurement_payload)
    if measurement_count < 2:
        raise ValueError("synthetic scenario produced fewer than two ground-station measurements.")

    ingest = ingest_ground_station_measurement_payload(
        measurement_payload,
        store_path=store,
        source_label=f"synthetic_{prefix}_sensor_od_workflow",
    )
    od_dir = output_root / f"{prefix}_sensor_od"
    fitter = run_ogp_ground_sensor_od_from_store if ogp_dispatch else run_sgp4_ground_sensor_od_from_store
    od_manifest = fitter(
        store,
        object_id=object_id,
        prior_line1=str(prior.line1),
        prior_line2=str(prior.line2),
        output_dir=od_dir,
        fit_duration_s=fit_duration,
        holdout_duration_s=holdout_duration,
        frame_transform=frame_transform,
        max_nfev=int(max_nfev),
    )
    comparison = tle_element_comparison_payload(
        prior=prior,
        truth=truth,
        estimated=dict(od_manifest.get("estimated_elements", {}) or {}),
    )
    payload = {
        "schema_version": 1,
        "ogp_regime": ogp_regime_for_elements(truth),
        "propagator_name": ogp_propagator_name_for_elements(truth),
        "status": "completed",
        "workflow": f"synthetic_{prefix}_sensor_od",
        "object_id": object_id,
        "store_path": str(store),
        "simulation": {
            "duration_s": duration,
            "step_s": step,
            "station_count": len(list(scenario.get("ground_stations", []) or [])),
            "measurement_count": measurement_count,
            "scenario_path": str(scenario_path),
            "measurement_payload_path": str(measurement_payload_path),
        },
        "truth_generation": {
            "mean_anomaly_offset_deg": float(mean_anomaly_offset_deg),
            "mean_motion_scale": float(mean_motion_scale),
            "prior_tle": {"line1": str(prior.line1), "line2": str(prior.line2)},
            "truth_tle": {"line1": str(prior.line1), "line2": truth_line2},
        },
        "noise": {
            "angle_sigma_deg": float(angle_sigma_deg),
            "range_sigma_km": float(range_sigma_km),
            "range_rate_sigma_km_s": float(range_rate_sigma_km_s),
        },
        "od": {
            "fit_duration_s": fit_duration,
            "holdout_duration_s": holdout_duration,
            "frame_transform": str(frame_transform or DEFAULT_SGP4_SENSOR_FRAME_TRANSFORM),
            "manifest_path": od_manifest.get("manifest_path", ""),
            "report_json_path": od_manifest.get(f"{prefix}_sensor_od_report_json_path", ""),
            "report_md_path": od_manifest.get(f"{prefix}_sensor_od_report_md_path", ""),
            "fitted_product_path": od_manifest.get(f"fitted_{prefix}_od_product_path", ""),
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
            "This is a synthetic workflow for exercising the OEL Scale OGP sensor-OD path.",
            "The truth TLE is generated by editing mean anomaly and mean motion fields from the prior TLE.",
            "The synthetic tracking network is generated for deterministic measurement coverage, not as an operational sensor model.",
            "The fitted product is a JSON mean-element OD artifact, not a checksum-correct operational TLE publication.",
        ],
    }
    report_json = output_root / f"synthetic_{prefix}_sensor_od_workflow.json"
    report_md = output_root / f"synthetic_{prefix}_sensor_od_workflow.md"
    payload["artifacts"] = {
        "workflow_json": str(report_json),
        "workflow_markdown": str(report_md),
        "scenario_json": str(scenario_path),
        "measurement_payload_json": str(measurement_payload_path),
    }
    report_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_md.write_text(_render_report(payload), encoding="utf-8")
    return payload


def _synthetic_sgp4_sensor_scenario(
    output_root: Path,
    *,
    object_id: str,
    truth_line1: str,
    truth_line2: str,
    duration_s: float,
    step_s: float,
    station_spacing_s: float | None,
    station_longitude_offset_deg: float,
    range_sigma_km: float,
    range_rate_sigma_km_s: float,
    angle_sigma_deg: float,
    frame_transform: str,
) -> dict[str, Any]:
    truth = parse_tle_lines(truth_line1, truth_line2)
    return {
        "scenario_name": "synthetic_sgp4_sensor_od_workflow",
        "objects": {
            object_id: {
                "kind": "satellite",
                "enabled": True,
                "role": "tracked_satellite",
                "propagation_method": "general",
                "general": {
                    "model": "sgp4",
                    "output_frame": "eci",
                    "frame_transform": str(frame_transform or DEFAULT_SGP4_SENSOR_FRAME_TRANSFORM),
                },
                "specs": {"mass_kg": 420000.0},
                "initial_state": {"tle": {"line1": truth_line1, "line2": truth_line2}},
            }
        },
        "ground_stations": truth_arc_tracking_network_from_sgp4(
            truth,
            duration_s=duration_s,
            station_spacing_s=station_spacing_s if station_spacing_s is not None else step_s,
            station_longitude_offset_deg=station_longitude_offset_deg,
            range_sigma_km=range_sigma_km,
            range_rate_sigma_km_s=range_rate_sigma_km_s,
            angle_sigma_deg=angle_sigma_deg,
            frame_transform=frame_transform,
        ),
        "simulator": {
            "duration_s": float(duration_s),
            "dt_s": float(step_s),
            "initial_jd_utc": float(truth.epoch_jd_utc),
            "dynamics": {
                "orbit": {"propagation_method": "special", "model": "two_body"},
                "attitude": {"enabled": False},
                "rocket": {"enabled": False},
            },
            "termination": {"earth_impact_enabled": False},
        },
        "outputs": {
            "output_dir": str(output_root / "simulation"),
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
    comparison = dict(dict(payload.get("comparison", {}) or {}).get("elements", {}) or {})
    lines = [
        f"# Synthetic {payload.get('propagator_name', 'SGP4')} Sensor OD Workflow",
        "",
        f"- Status: `{payload.get('status')}`",
        f"- Object ID: `{payload.get('object_id')}`",
        f"- Measurements: `{simulation.get('measurement_count')}` from `{simulation.get('station_count')}` synthetic sites",
        f"- Noise: `{noise}`",
        f"- Fit weighted RMS: `{fit.get('weighted_rms')}`",
        f"- Holdout weighted RMS: `{holdout.get('weighted_rms')}`",
        f"- Solver success: `{quality.get('solver_success')}`",
        "",
        "## Element Comparison",
        "",
        "| Element | Prior | Truth | Fitted | Fitted - Truth |",
        "|---|---:|---:|---:|---:|",
    ]
    for field, row in comparison.items():
        values = dict(row or {})
        lines.append(
            f"| `{field}` | {_fmt(values.get('prior'))} | {_fmt(values.get('truth'))} | "
            f"{_fmt(values.get('estimated'))} | {_fmt(values.get('estimated_minus_truth'))} |"
        )
    lines.extend(
        [
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
    )
    lines.extend(f"- {item}" for item in list(payload.get("non_claims", []) or []))
    return "\n".join(lines) + "\n"


def _fmt(value: Any) -> str:
    try:
        return f"{float(value):.12g}"
    except (TypeError, ValueError):
        return ""
