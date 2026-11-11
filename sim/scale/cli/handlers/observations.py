from __future__ import annotations

import argparse
import json
from pathlib import Path

from sim.scale.cli.common import (
    _first_tle_lines_from_file,
    _initial_state_from_mission_input_packet,
    _json_mapping_from_path,
    _spherical_harmonics_from_args,
)
from sim.scale.frozen_od import assess_frozen_ogp_od_solution, freeze_ogp_od_solution
from sim.scale.maneuver import run_maneuver_detection
from sim.scale.measurements import (
    assess_ground_station_measurement_quality,
    export_ground_station_observation_packet_from_store,
    ingest_ground_station_measurement_file,
    run_ground_sensor_od_from_store,
    run_ground_station_measurement_od,
    run_ogp_ground_sensor_od_from_store,
    run_sgp4_ground_sensor_od_from_store,
)
from sim.scale.od import run_ogp_od_from_track_observations, run_sgp4_od_from_track_observations
from sim.scale.synthetic_ground_sensor_od import run_synthetic_ground_sensor_od_workflow
from sim.scale.synthetic_sgp4_sensor_od import (
    run_synthetic_ogp_sensor_od_workflow,
    run_synthetic_sgp4_sensor_od_workflow,
)


def handle_ingest_ground_measurements(args: argparse.Namespace) -> int:
    payload = ingest_ground_station_measurement_file(
        args.payload_json,
        store_path=args.store,
        source_label=args.source_label,
        jd_utc_start=args.jd_utc_start,
    )
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"status: {payload.get('status')}")
        print(f"store_path: {payload.get('store_path')}")
        print(f"records_seen: {payload.get('records_seen')}")
        print(f"stations_recorded: {payload.get('stations_recorded')}")
        print(f"measurements_inserted: {payload.get('measurements_inserted')}")
        print(f"invalidations_inserted: {payload.get('invalidations_inserted')}")
    return 0 if payload.get("status") == "completed" else 2

def handle_export_ground_measurement_packet(args: argparse.Namespace) -> int:
    payload = export_ground_station_observation_packet_from_store(
        args.store,
        station_id=args.station_id,
        object_id=args.object_id,
        output_path=args.out,
        source_label=args.source_label or "scale_ground_station_measurements",
    )
    if args.json:
        printable = dict(payload)
        printable.pop("packet", None)
        print(json.dumps(printable, indent=2, sort_keys=True))
    else:
        print(f"status: {payload.get('status')}")
        print(f"store_path: {payload.get('store_path')}")
        print(f"station_id: {payload.get('station_id')}")
        print(f"object_id: {payload.get('object_id')}")
        print(f"observation_count: {payload.get('observation_count')}")
        print(f"packet_path: {payload.get('packet_path')}")
    return 0 if payload.get("status") == "completed" else 2

def handle_fit_ground_measurement_od(args: argparse.Namespace) -> int:
    payload = run_ground_station_measurement_od(
        args.store,
        station_id=args.station_id,
        object_id=args.object_id,
        output_dir=args.output_dir,
        fit_duration_s=args.fit_duration_s,
        holdout_duration_s=args.holdout_duration_s,
        dt_s=args.dt_s,
        estimate=args.estimate,
        dynamics_model=args.dynamics_model,
        j2=bool(args.j2),
        drag=bool(args.drag),
        srp=bool(args.srp),
        atmosphere_model=args.atmosphere_model,
        max_nfev=int(args.max_nfev),
        detect_maneuvers=bool(args.detect_maneuvers),
    )
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"status: {payload.get('status')}")
        print(f"store_path: {payload.get('store_path')}")
        print(f"station_id: {payload.get('station_id')}")
        print(f"object_id: {payload.get('object_id')}")
        print(f"measurement_count: {payload.get('measurement_count')}")
        print(f"manifest_path: {payload.get('manifest_path')}")
        print(f"od_report_json_path: {payload.get('od_report_json_path')}")
    return 0 if payload.get("status") == "completed" else 2

def handle_fit_ground_sensor_od(args: argparse.Namespace) -> int:
    initial_state = None
    initial_state_source = None
    if args.initial_state_eci_km_s is not None:
        initial_state = tuple(float(item) for item in args.initial_state_eci_km_s)
        initial_state_source = "cli_initial_state_eci_km_s"
    elif args.initial_state_packet:
        packet_path = Path(args.initial_state_packet)
        initial_state = _initial_state_from_mission_input_packet(packet_path, object_id=args.object_id)
        initial_state_source = f"mission_input_packet:{packet_path.expanduser().resolve()}"
    spherical_harmonics = _spherical_harmonics_from_args(args)
    if spherical_harmonics and any(bool(flag) for flag in (args.j2, args.j3, args.j4)):
        raise ValueError("spherical harmonics cannot be combined with --j2, --j3, or --j4.")
    payload = run_ground_sensor_od_from_store(
        args.store,
        object_id=args.object_id,
        station_ids=tuple(str(item) for item in args.station_id if str(item).strip()),
        output_dir=args.output_dir,
        initial_state_eci_km_s=initial_state,
        initial_state_source=initial_state_source,
        fit_duration_s=args.fit_duration_s,
        holdout_duration_s=args.holdout_duration_s,
        dt_s=args.dt_s,
        dynamics_model=args.dynamics_model,
        j2=bool(args.j2),
        j3=bool(args.j3),
        j4=bool(args.j4),
        drag=bool(args.drag),
        srp=bool(args.srp),
        third_body_sun=bool(args.third_body_sun),
        third_body_moon=bool(args.third_body_moon),
        atmosphere_model=args.atmosphere_model or None,
        spherical_harmonics=spherical_harmonics,
        max_nfev=int(args.max_nfev),
        systematic_error_model=_json_mapping_from_path(args.systematic_error_model_json),
        exclude_station_ids=tuple(str(item) for item in args.exclude_station_id if str(item).strip()),
        holdout_station_ids=tuple(str(item) for item in args.holdout_station_id if str(item).strip()),
        robust_loss=str(args.robust_loss),
        robust_f_scale=float(args.robust_f_scale),
        sigma_clip_threshold=args.sigma_clip_threshold,
    )
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"status: {payload.get('status')}")
        print(f"store_path: {payload.get('store_path')}")
        print(f"object_id: {payload.get('object_id')}")
        print(f"measurement_count: {payload.get('measurement_count')}")
        print(f"initial_state_source: {payload.get('initial_state_source')}")
        print(f"force_model: {payload.get('force_model')}")
        print(f"manifest_path: {payload.get('manifest_path')}")
        print(f"native_sensor_od_report_json_path: {payload.get('native_sensor_od_report_json_path')}")
    return 0 if payload.get("status") == "completed" else 2

def handle_assess_ground_measurement_quality(args: argparse.Namespace) -> int:
    payload = assess_ground_station_measurement_quality(
        store_path=args.store,
        manifest_path=args.manifest,
        warning_sigma=float(args.warning_sigma),
        reject_sigma=float(args.reject_sigma),
        warning_error_m=args.warning_error_m,
        reject_error_m=args.reject_error_m,
    )
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"status: {payload.get('status')}")
        print(f"assessment_id: {payload.get('assessment_id')}")
        print(f"store_path: {payload.get('store_path')}")
        print(f"manifest_path: {payload.get('manifest_path')}")
        print(f"measurement_count: {payload.get('measurement_count')}")
        print(f"status_counts: {payload.get('status_counts')}")
        print(f"quality_records_inserted: {payload.get('write_summary', {}).get('quality_records_inserted')}")
    return 0 if payload.get("status") == "completed" else 2

def handle_detect_maneuvers(args: argparse.Namespace) -> int:
    thresholds = {
        key: value
        for key, value in {
            "mean_motion_delta_rev_per_day": args.mean_motion_threshold,
            "eccentricity_delta": args.eccentricity_threshold,
            "inclination_delta_deg": args.inclination_threshold_deg,
            "raan_delta_deg": args.raan_threshold_deg,
            "argp_delta_deg": args.argp_threshold_deg,
        }.items()
        if value is not None
    }
    payload = run_maneuver_detection(
        args.store,
        object_ids=tuple(str(item) for item in args.object_id if str(item).strip()),
        thresholds=thresholds,
    )
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"status: {payload.get('status')}")
        print(f"store_path: {payload.get('store_path')}")
        print(f"objects_with_comparable_od: {payload.get('objects_with_comparable_od')}")
        print(f"flags_inserted: {payload.get('flags_inserted')}")
        print(f"invalidations_inserted: {payload.get('invalidations_inserted')}")
    return 0 if payload.get("status") == "completed" else 2

def handle_freeze_ogp_od(args):
    payload = freeze_ogp_od_solution(args.store, od_solution_id=args.od_solution_id, output_path=args.out,
        position_rms_m_max=args.position_rms_m_max, minimum_observations=args.minimum_observations)
    print(json.dumps(payload, indent=2) if args.json else "Frozen exact OGP solution: " + payload["od_solution_id"])
    return 0


def handle_assess_frozen_ogp_od(args):
    payload = assess_frozen_ogp_od_solution(args.store, frozen_solution_path=args.frozen_solution,
        observation_ids=tuple(args.observation_id), output_dir=args.output_dir)
    print(json.dumps(payload, indent=2) if args.json else "Frozen OGP assessment passed: " + str(payload["passed"]))
    return 0 if payload["passed"] else 2


def handle_fit_sgp4_od(args: argparse.Namespace) -> int:
    runner = run_ogp_od_from_track_observations if "ogp" in args.command else run_sgp4_od_from_track_observations
    options = {}
    if "ogp" in args.command:
        options = {"estimate_bstar": args.estimate_bstar, "bstar_prior_sigma": args.bstar_prior_sigma}
    elif args.estimate_bstar or args.bstar_prior_sigma is not None:
        raise ValueError("B* estimation requires fit-ogp-od.")
    payload = runner(
        args.store,
        object_ids=tuple(str(item) for item in args.object_id if str(item).strip()),
        min_observations=int(args.min_observations),
        position_sigma_km=float(args.position_sigma_km),
        velocity_sigma_km_s=float(args.velocity_sigma_km_s),
        fit_until_jd_utc=args.fit_until_jd_utc,
        max_nfev=int(args.max_nfev),
        **options,
    )
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"status: {payload.get('status')}")
        print(f"store_path: {payload.get('store_path')}")
        print(f"objects_with_observations: {payload.get('objects_with_observations')}")
        print(f"objects_fit: {payload.get('objects_fit')}")
        print(f"od_solutions_inserted: {payload.get('write_summary', {}).get('od_solutions_inserted')}")
        print(f"invalidations_inserted: {payload.get('write_summary', {}).get('invalidations_inserted')}")
    return 0 if payload.get("status") == "completed" else 2

def handle_fit_ground_sgp4_sensor_od(args: argparse.Namespace) -> int:
    line1, line2 = _first_tle_lines_from_file(args.prior_tle)
    runner = run_ogp_ground_sensor_od_from_store if "ogp" in args.command else run_sgp4_ground_sensor_od_from_store
    payload = runner(
        args.store,
        object_id=args.object_id,
        station_ids=tuple(str(item) for item in args.station_id if str(item).strip()),
        prior_line1=line1,
        prior_line2=line2,
        output_dir=args.output_dir,
        fit_duration_s=args.fit_duration_s,
        holdout_duration_s=args.holdout_duration_s,
        frame_transform=args.frame_transform,
        max_nfev=int(args.max_nfev),
    )
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"status: {payload.get('status')}")
        print(f"store_path: {payload.get('store_path')}")
        print(f"object_id: {payload.get('object_id')}")
        print(f"measurement_count: {payload.get('measurement_count')}")
        print(f"manifest_path: {payload.get('manifest_path')}")
        report_key = "ogp_sensor_od_report_json_path" if "ogp" in args.command else "sgp4_sensor_od_report_json_path"
        print(f"{report_key}: {payload.get(report_key)}")
    return 0 if payload.get("status") == "completed" else 2

def handle_run_synthetic_sgp4_sensor_od(args: argparse.Namespace) -> int:
    line1, line2 = _first_tle_lines_from_file(args.prior_tle)
    runner = run_synthetic_ogp_sensor_od_workflow if "ogp" in args.command else run_synthetic_sgp4_sensor_od_workflow
    payload = runner(
        prior_line1=line1,
        prior_line2=line2,
        output_dir=args.output_dir,
        object_id=args.object_id,
        store_path=args.store,
        mean_anomaly_offset_deg=float(args.mean_anomaly_offset_deg),
        mean_motion_scale=float(args.mean_motion_scale),
        duration_s=float(args.duration_s),
        step_s=float(args.step_s),
        station_spacing_s=args.station_spacing_s,
        station_longitude_offset_deg=float(args.station_longitude_offset_deg),
        range_sigma_km=float(args.range_sigma_km),
        range_rate_sigma_km_s=float(args.range_rate_sigma_km_s),
        angle_sigma_deg=float(args.angle_sigma_deg),
        fit_duration_s=args.fit_duration_s,
        holdout_duration_s=args.holdout_duration_s,
        frame_transform=args.frame_transform,
        max_nfev=int(args.max_nfev),
    )
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"status: {payload.get('status')}")
        print(f"store_path: {payload.get('store_path')}")
        print(f"measurement_count: {payload.get('simulation', {}).get('measurement_count')}")
        print(f"fit_weighted_rms: {payload.get('od', {}).get('fit_metrics', {}).get('weighted_rms')}")
        print(f"holdout_weighted_rms: {payload.get('od', {}).get('holdout_metrics', {}).get('weighted_rms')}")
        print(f"workflow_json: {payload.get('artifacts', {}).get('workflow_json')}")
        print(f"workflow_markdown: {payload.get('artifacts', {}).get('workflow_markdown')}")
    return 0 if payload.get("status") == "completed" else 2

def handle_run_synthetic_ground_sensor_od(args: argparse.Namespace) -> int:
    payload = run_synthetic_ground_sensor_od_workflow(
        output_dir=args.output_dir,
        object_id=args.object_id,
        store_path=args.store,
        initial_state_eci_km_s=tuple(float(item) for item in args.initial_state_eci_km_s),
        prior_position_offset_m=tuple(float(item) for item in args.prior_position_offset_m),
        prior_velocity_offset_mm_s=tuple(float(item) for item in args.prior_velocity_offset_mm_s),
        duration_s=float(args.duration_s),
        step_s=float(args.step_s),
        station_spacing_s=args.station_spacing_s,
        station_longitude_offset_deg=float(args.station_longitude_offset_deg),
        range_sigma_km=float(args.range_sigma_km),
        range_rate_sigma_km_s=float(args.range_rate_sigma_km_s),
        angle_sigma_deg=float(args.angle_sigma_deg),
        fit_duration_s=args.fit_duration_s,
        holdout_duration_s=args.holdout_duration_s,
        dynamics_model=args.dynamics_model,
        j2=bool(args.j2),
        j3=bool(args.j3),
        j4=bool(args.j4),
        drag=bool(args.drag),
        srp=bool(args.srp),
        third_body_sun=bool(args.third_body_sun),
        third_body_moon=bool(args.third_body_moon),
        atmosphere_model=args.atmosphere_model or None,
        max_nfev=int(args.max_nfev),
    )
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"status: {payload.get('status')}")
        print(f"store_path: {payload.get('store_path')}")
        print(f"measurement_count: {payload.get('simulation', {}).get('measurement_count')}")
        print(f"fit_weighted_rms: {payload.get('od', {}).get('fit_metrics', {}).get('weighted_rms')}")
        print(f"holdout_weighted_rms: {payload.get('od', {}).get('holdout_metrics', {}).get('weighted_rms')}")
        print(f"fitted_position_error_m: {payload.get('comparison', {}).get('fitted_position_error_m')}")
        print(f"workflow_json: {payload.get('artifacts', {}).get('workflow_json')}")
        print(f"workflow_markdown: {payload.get('artifacts', {}).get('workflow_markdown')}")
    return 0 if payload.get("status") == "completed" else 2

COMMAND_HANDLERS = {
    'ingest-ground-measurements': handle_ingest_ground_measurements,
    'export-ground-measurement-packet': handle_export_ground_measurement_packet,
    'fit-ground-measurement-od': handle_fit_ground_measurement_od,
    'fit-ground-sensor-od': handle_fit_ground_sensor_od,
    'assess-ground-measurement-quality': handle_assess_ground_measurement_quality,
    'detect-maneuvers': handle_detect_maneuvers,
    'fit-sgp4-od': handle_fit_sgp4_od,
    'fit-ogp-od': handle_fit_sgp4_od,
    'freeze-ogp-od': handle_freeze_ogp_od,
    'assess-frozen-ogp-od': handle_assess_frozen_ogp_od,
    'fit-ground-sgp4-sensor-od': handle_fit_ground_sgp4_sensor_od,
    'fit-ground-ogp-sensor-od': handle_fit_ground_sgp4_sensor_od,
    'run-synthetic-sgp4-sensor-od': handle_run_synthetic_sgp4_sensor_od,
    'run-synthetic-ogp-sensor-od': handle_run_synthetic_sgp4_sensor_od,
    'run-synthetic-ground-sensor-od': handle_run_synthetic_ground_sensor_od,
}
