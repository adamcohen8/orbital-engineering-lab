from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from sim.dynamics.orbit.frames import FrameContext, frame_context_from_mapping, transform_position
from sim.dynamics.orbit.tle import parse_tle_lines
from sim.estimation.ground_station_od import solve_ground_station_measurement_od
from sim.estimation.orbit_determination import solve_dynamics_orbit_determination
from sim.estimation.sgp4_od import (
    DEFAULT_SGP4_SENSOR_FRAME_TRANSFORM,
    fit_ogp_mean_elements_to_ground_station_measurements,
    fit_sgp4_mean_elements_to_ground_station_measurements,
)
from sim.ground_stations import _ground_measurement_physical_validity
from sim.observations import ObservationPacket, ingest_observations
from sim.scale.store import (
    GroundStationMeasurementQualityRecord,
    GroundStationMeasurementRecord,
    ingest_ground_station_measurement_records,
    load_ground_station_measurement_rows,
    write_ground_station_measurement_quality_records,
)
from sim.utils.geodesy import enu_to_ecef_rotation, geodetic_to_ecef_km

_DAY_S = 86400.0


def ingest_ground_station_measurement_file(
    path: str | Path,
    *,
    store_path: str | Path,
    source_label: str = "",
    jd_utc_start: float | None = None,
) -> dict[str, Any]:
    source_path = Path(path).expanduser().resolve()
    raw_text = source_path.read_text(encoding="utf-8")
    payload = json.loads(raw_text)
    source_digest = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()
    label = source_label or source_path.name
    return ingest_ground_station_measurement_payload(
        payload,
        store_path=store_path,
        source_label=label,
        source_digest=source_digest,
        jd_utc_start=jd_utc_start,
    )


def ingest_ground_station_measurement_payload(
    payload: dict[str, Any],
    *,
    store_path: str | Path,
    source_label: str,
    source_digest: str = "",
    jd_utc_start: float | None = None,
) -> dict[str, Any]:
    measurements = dict(payload.get("ground_station_measurements", payload) or {})
    records = _records_from_ground_station_payload(measurements, jd_utc_start=jd_utc_start)
    result = ingest_ground_station_measurement_records(
        store_path,
        records=records,
        source_label=source_label,
        source_digest=source_digest or _payload_digest(measurements),
    )
    return {
        "status": "completed",
        "store_path": str(result.store_path),
        "batch_id": result.batch_id,
        "source_label": result.source_label,
        "source_digest": result.source_digest,
        "records_seen": result.records_seen,
        "objects_created": result.objects_created,
        "stations_recorded": result.stations_recorded,
        "measurements_inserted": result.measurements_inserted,
        "invalidations_inserted": result.invalidations_inserted,
    }


def export_ground_station_observation_packet_from_store(
    store_path: str | Path,
    *,
    station_id: str,
    object_id: str,
    output_path: str | Path | None = None,
    source_label: str = "scale_ground_station_measurements",
    frame_context: FrameContext | None = None,
) -> dict[str, Any]:
    rows_by_object = load_ground_station_measurement_rows(
        store_path,
        object_ids=(object_id,),
        station_ids=(station_id,),
        min_measurements=2,
    )
    rows = [
        row
        for row in rows_by_object.get(object_id, [])
        if row.get("station_id") == station_id and _has_az_el_range(row)
    ]
    for row_index, row in enumerate(rows):
        physically_valid, validity_reason = _stored_ground_measurement_validity(row)
        if not physically_valid:
            raise ValueError(
                f"stored ground-station measurement row {row_index} is physically invalid "
                f"({validity_reason}); invalid raw measurements cannot be exported as observations."
            )
    if len(rows) < 2:
        raise ValueError("at least two azimuth/elevation/range measurements are required to export an OD packet.")
    packet = _observation_packet_from_store_rows(
        rows,
        object_id=object_id,
        station_id=station_id,
        source_label=source_label,
        frame_context=frame_context,
    )
    written_path = ""
    if output_path is not None:
        written_path = str(packet.write_json(output_path))
    data = packet.to_dict()
    return {
        "status": "completed",
        "store_path": str(Path(store_path).expanduser().resolve()),
        "station_id": station_id,
        "object_id": object_id,
        "observation_count": int(data.get("summary", {}).get("observation_count", 0) or 0),
        "measurement_type": data.get("measurement_type", ""),
        "packet": data,
        "packet_path": written_path,
        "non_claims": [
            "Exported positions are reconstructed from measured azimuth, elevation, and range.",
            "Range-rate is not converted into velocity observations in this v0 bridge.",
            "Scale SGP4 OD still requires native TEME position/velocity rows.",
        ],
    }


def run_ground_station_measurement_od(
    store_path: str | Path,
    *,
    station_id: str,
    object_id: str,
    output_dir: str | Path,
    fit_duration_s: float | None = None,
    holdout_duration_s: float | None = None,
    dt_s: float | None = None,
    estimate: str = "state",
    dynamics_model: str = "two_body",
    j2: bool = False,
    drag: bool = False,
    srp: bool = False,
    atmosphere_model: str = "",
    max_nfev: int = 24,
    detect_maneuvers: bool = False,
) -> dict[str, Any]:
    output_root = Path(output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    packet_path = output_root / "observation_packet.json"
    export = export_ground_station_observation_packet_from_store(
        store_path,
        station_id=station_id,
        object_id=object_id,
        output_path=packet_path,
        source_label="scale_ground_station_measurement_od",
    )
    packet = export["packet"]
    od_result = solve_dynamics_orbit_determination(
        packet,
        output_dir=output_root,
        object_id=object_id,
        scenario_name=f"scale_ground_measurement_od_{_safe_name(object_id)}",
        fit_duration_s=fit_duration_s,
        holdout_duration_s=holdout_duration_s,
        dt_s=dt_s,
        estimate=estimate,
        dynamics_model=dynamics_model,
        j2=bool(j2),
        drag=bool(drag),
        srp=bool(srp),
        atmosphere_model=atmosphere_model.strip() or None,
        max_nfev=int(max_nfev),
        detect_maneuvers=bool(detect_maneuvers),
    )
    measurement_ids = list(dict(packet.get("source", {}) or {}).get("metadata", {}).get("measurement_ids", []) or [])
    manifest = {
        "status": "completed",
        "workflow": "scale_ground_station_measurement_od",
        "store_path": str(Path(store_path).expanduser().resolve()),
        "station_id": station_id,
        "object_id": object_id,
        "measurement_ids": measurement_ids,
        "measurement_count": len(measurement_ids),
        "observation_packet_path": str(packet_path),
        "od_report_json_path": od_result.get("report_json_path", ""),
        "od_report_md_path": od_result.get("report_md_path", ""),
        "fitted_mission_input_packet_path": od_result.get("fitted_mission_input_packet_path", ""),
        "materialized_fit_config_path": od_result.get("materialized_fit_config_path", ""),
        "materialized_prediction_config_path": od_result.get("materialized_prediction_config_path", ""),
        "fit_metrics": dict(od_result.get("fit_metrics", {}) or {}),
        "holdout_metrics": dict(od_result.get("holdout_metrics", {}) or {}),
        "quality_gates": dict(od_result.get("quality_gates", {}) or {}),
        "verdict": od_result.get("verdict", ""),
        "non_claims": [
            "This workflow preserves raw ground-station measurements and runs the existing structured-observation dynamics OD path.",
            "It does not perform native sensor-frame OD, bias estimation, association, or covariance-rich tracking.",
            "Range-rate measurements are not converted into velocity observations in this v0 bridge.",
            "The result is an OD artifact manifest, not a stored SGP4 mean-element od_solutions product.",
        ],
    }
    manifest_path = output_root / "ground_measurement_od_manifest.json"
    manifest["manifest_path"] = str(manifest_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def run_ground_sensor_od_from_store(
    store_path: str | Path,
    *,
    object_id: str,
    output_dir: str | Path,
    station_ids: tuple[str, ...] = (),
    initial_state_eci_km_s: tuple[float, float, float, float, float, float] | None = None,
    initial_state_source: str | None = None,
    fit_duration_s: float | None = None,
    holdout_duration_s: float | None = None,
    dt_s: float | None = None,
    dynamics_model: str = "two_body",
    j2: bool = False,
    j3: bool = False,
    j4: bool = False,
    drag: bool = False,
    srp: bool = False,
    third_body_sun: bool = False,
    third_body_moon: bool = False,
    atmosphere_model: str | None = None,
    spherical_harmonics: dict[str, Any] | None = None,
    max_nfev: int = 24,
    systematic_error_model: dict[str, Any] | None = None,
    exclude_station_ids: tuple[str, ...] = (),
    holdout_station_ids: tuple[str, ...] = (),
    robust_loss: str = "linear",
    robust_f_scale: float = 1.0,
    sigma_clip_threshold: float | None = None,
) -> dict[str, Any]:
    rows_by_object = load_ground_station_measurement_rows(
        store_path,
        object_ids=(object_id,),
        station_ids=station_ids,
        min_measurements=2,
    )
    rows = list(rows_by_object.get(object_id, []) or [])
    if not rows:
        raise ValueError(f"no ground-station measurements found for object_id {object_id!r}.")
    report = solve_ground_station_measurement_od(
        rows,
        object_id=object_id,
        output_dir=output_dir,
        initial_state_eci_km_s=initial_state_eci_km_s,
        initial_state_source=initial_state_source,
        fit_duration_s=fit_duration_s,
        holdout_duration_s=holdout_duration_s,
        dt_s=dt_s,
        dynamics_model=dynamics_model,
        j2=bool(j2),
        j3=bool(j3),
        j4=bool(j4),
        drag=bool(drag),
        srp=bool(srp),
        third_body_sun=bool(third_body_sun),
        third_body_moon=bool(third_body_moon),
        atmosphere_model=atmosphere_model,
        spherical_harmonics=spherical_harmonics,
        max_nfev=int(max_nfev),
        systematic_error_model=systematic_error_model,
        exclude_station_ids=exclude_station_ids,
        holdout_station_ids=holdout_station_ids,
        robust_loss=robust_loss,
        robust_f_scale=float(robust_f_scale),
        sigma_clip_threshold=sigma_clip_threshold,
        scenario_name=f"scale_ground_sensor_od_{_safe_name(object_id)}",
    )
    manifest = {
        "status": "completed",
        "workflow": "scale_ground_sensor_od",
        "store_path": str(Path(store_path).expanduser().resolve()),
        "object_id": object_id,
        "station_ids": sorted({str(row.get("station_id", "")) for row in rows}),
        "measurement_ids": [str(row.get("measurement_id", "")) for row in rows],
        "measurement_count": len(rows),
        "initial_state_source": report.get("initial_state_source", ""),
        "initial_state_eci_km_s": list(report.get("initial_state_eci_km_s", []) or []),
        "force_model": {
            "dynamics_model": report.get("dynamics_model", ""),
            "j2": bool(report.get("j2", False)),
            "j3": bool(report.get("j3", False)),
            "j4": bool(report.get("j4", False)),
            "drag": bool(report.get("drag", False)),
            "srp": bool(report.get("srp", False)),
            "third_body_sun": bool(report.get("third_body_sun", False)),
            "third_body_moon": bool(report.get("third_body_moon", False)),
            "atmosphere_model": str(report.get("atmosphere_model", "") or ""),
            "spherical_harmonics": dict(report.get("spherical_harmonics", {}) or {}),
        },
        "native_sensor_od_report_json_path": report.get("report_json_path", ""),
        "native_sensor_od_report_md_path": report.get("report_md_path", ""),
        "residual_csv_path": report.get("residual_csv_path", ""),
        "residual_plot_path": report.get("residual_plot_path", ""),
        "fitted_mission_input_packet_path": report.get("fitted_mission_input_packet_path", ""),
        "fit_metrics": dict(report.get("fit_metrics", {}) or {}),
        "holdout_metrics": dict(report.get("holdout_metrics", {}) or {}),
        "systematic_error_model": dict(report.get("systematic_error_model", {}) or {}),
        "systematic_parameter_evidence": list(report.get("systematic_parameter_evidence", []) or []),
        "station_comparison": dict(report.get("station_comparison", {}) or {}),
        "review": dict(report.get("review", {}) or {}),
        "quality_gates": dict(report.get("quality_gates", {}) or {}),
        "verdict": dict(report.get("verdict", {}) or {}),
        "non_claims": [
            "This workflow fits directly against azimuth/elevation/range/range-rate measurements.",
            "It does not use the ECI observation-packet bridge in the optimizer residual.",
            "Biases, clocks, elevation weighting, and refraction are applied only when explicitly configured.",
            "It does not perform association, light-time correction, media range-delay calibration, or infer operational covariance.",
        ],
    }
    output_root = Path(output_dir).expanduser().resolve()
    manifest_path = output_root / "ground_sensor_od_manifest.json"
    manifest["manifest_path"] = str(manifest_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def run_sgp4_ground_sensor_od_from_store(
    store_path: str | Path,
    *,
    object_id: str,
    prior_line1: str,
    prior_line2: str,
    output_dir: str | Path,
    station_ids: tuple[str, ...] = (),
    fit_duration_s: float | None = None,
    holdout_duration_s: float | None = None,
    frame_transform: str = DEFAULT_SGP4_SENSOR_FRAME_TRANSFORM,
    max_nfev: int = 100,
) -> dict[str, Any]:
    """Fit near-Earth SGP4 observations through the shared owner."""
    return _run_mean_element_ground_sensor_od_from_store(
        store_path,
        object_id=object_id,
        prior_line1=prior_line1,
        prior_line2=prior_line2,
        output_dir=output_dir,
        station_ids=station_ids,
        fit_duration_s=fit_duration_s,
        holdout_duration_s=holdout_duration_s,
        frame_transform=frame_transform,
        max_nfev=max_nfev,
        ogp_dispatch=False,
    )


def run_ogp_ground_sensor_od_from_store(
    store_path: str | Path,
    *,
    object_id: str,
    prior_line1: str,
    prior_line2: str,
    output_dir: str | Path,
    station_ids: tuple[str, ...] = (),
    fit_duration_s: float | None = None,
    holdout_duration_s: float | None = None,
    frame_transform: str = DEFAULT_SGP4_SENSOR_FRAME_TRANSFORM,
    max_nfev: int = 100,
) -> dict[str, Any]:
    """Fit regime-dispatched OGP observations through the shared owner."""
    return _run_mean_element_ground_sensor_od_from_store(
        store_path,
        object_id=object_id,
        prior_line1=prior_line1,
        prior_line2=prior_line2,
        output_dir=output_dir,
        station_ids=station_ids,
        fit_duration_s=fit_duration_s,
        holdout_duration_s=holdout_duration_s,
        frame_transform=frame_transform,
        max_nfev=max_nfev,
        ogp_dispatch=True,
    )


def _run_mean_element_ground_sensor_od_from_store(
    store_path: str | Path,
    *,
    object_id: str,
    prior_line1: str,
    prior_line2: str,
    output_dir: str | Path,
    station_ids: tuple[str, ...] = (),
    fit_duration_s: float | None = None,
    holdout_duration_s: float | None = None,
    frame_transform: str = DEFAULT_SGP4_SENSOR_FRAME_TRANSFORM,
    max_nfev: int = 100,
    ogp_dispatch: bool,
) -> dict[str, Any]:
    rows_by_object = load_ground_station_measurement_rows(
        store_path,
        object_ids=(object_id,),
        station_ids=station_ids,
        min_measurements=2,
    )
    rows = list(rows_by_object.get(object_id, []) or [])
    if not rows:
        raise ValueError(f"no ground-station measurements found for object_id {object_id!r}.")
    template = parse_tle_lines(prior_line1, prior_line2)
    prefix = "ogp" if ogp_dispatch else "sgp4"
    fitter = fit_ogp_mean_elements_to_ground_station_measurements if ogp_dispatch else fit_sgp4_mean_elements_to_ground_station_measurements
    report = fitter(
        rows,
        template_elements=template,
        object_id=object_id,
        output_dir=output_dir,
        fit_duration_s=fit_duration_s,
        holdout_duration_s=holdout_duration_s,
        frame_transform=frame_transform,
        max_nfev=int(max_nfev),
    )
    output_root = Path(output_dir).expanduser().resolve()
    manifest = {
        "status": "completed",
        "workflow": f"scale_{prefix}_ground_sensor_od",
        "store_path": str(Path(store_path).expanduser().resolve()),
        "object_id": object_id,
        "ogp_regime": report.get("ogp_regime"),
        "propagator_name": report.get("propagator_name"),
        "station_ids": sorted({str(row.get("station_id", "")) for row in rows}),
        "measurement_ids": [str(row.get("measurement_id", "")) for row in rows],
        "measurement_count": len(rows),
        "prior_tle": {"line1": str(prior_line1), "line2": str(prior_line2)},
        "frame_transform": str(frame_transform or DEFAULT_SGP4_SENSOR_FRAME_TRANSFORM),
        f"{prefix}_sensor_od_report_json_path": report.get("report_json_path", ""),
        f"{prefix}_sensor_od_report_md_path": report.get("report_md_path", ""),
        f"fitted_{prefix}_od_product_path": report.get(f"fitted_{prefix}_od_product_path", ""),
        "residual_csv_path": report.get("residual_csv_path", ""),
        "residual_plot_path": report.get("residual_plot_path", ""),
        "estimated_elements": dict(report.get("estimated_elements", {}) or {}),
        "fit_metrics": dict(report.get("fit_metrics", {}) or {}),
        "holdout_metrics": dict(report.get("holdout_metrics", {}) or {}),
        "quality_gates": dict(report.get("quality_gates", {}) or {}),
        "verdict": dict(report.get("verdict", {}) or {}),
        "non_claims": [
            "This workflow fits OGP mean elements directly against azimuth/elevation/range/range-rate measurements.",
            "It requires a prior TLE and does not perform standalone initial orbit determination from raw station rows.",
            "It estimates fixed-epoch regime-bounded OGP mean elements; epoch, B*, and mean-motion derivative terms are fixed.",
        ],
    }
    manifest_path = output_root / f"{prefix}_ground_sensor_od_manifest.json"
    manifest["manifest_path"] = str(manifest_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def assess_ground_station_measurement_quality(
    *,
    store_path: str | Path,
    manifest_path: str | Path,
    warning_sigma: float = 3.0,
    reject_sigma: float = 6.0,
    warning_error_m: float | None = None,
    reject_error_m: float | None = None,
) -> dict[str, Any]:
    manifest_file = Path(manifest_path).expanduser().resolve()
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    packet_path = Path(str(manifest["observation_packet_path"])).expanduser().resolve()
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    od_report_path = Path(str(manifest["od_report_json_path"])).expanduser().resolve()
    od_report = json.loads(od_report_path.read_text(encoding="utf-8"))
    measurement_ids = [str(item) for item in manifest.get("measurement_ids", []) or []]
    observations = list(packet.get("observations", []) or [])
    if len(measurement_ids) != len(observations):
        raise ValueError("OD manifest measurement_ids count does not match observation packet count.")
    fit_count = int(od_report.get("fit_observation_count", 0) or 0)
    thresholds = {
        "warning_sigma": float(warning_sigma),
        "reject_sigma": float(reject_sigma),
        "warning_error_m": None if warning_error_m is None else float(warning_error_m),
        "reject_error_m": None if reject_error_m is None else float(reject_error_m),
    }
    residual_rows = _read_residual_rows(
        fit_csv=Path(str(od_report.get("fit_residuals_csv", ""))).expanduser(),
        holdout_csv=Path(str(od_report.get("holdout_errors_csv", ""))).expanduser(),
        fit_count=fit_count,
    )
    if len(residual_rows) != len(measurement_ids):
        raise ValueError("OD residual row count does not match source measurement count.")
    assessment_id = (
        f"ground_measurement_quality:{_payload_digest({'manifest': str(manifest_file), 'thresholds': thresholds})[:16]}"
    )
    records: list[GroundStationMeasurementQualityRecord] = []
    status_counts: dict[str, int] = {}
    rows_out: list[dict[str, Any]] = []
    for idx, residual in enumerate(residual_rows):
        obs = dict(observations[idx] or {})
        sigma_m = float(obs.get("position_sigma_km") or 0.0) * 1000.0
        position_error_m = float(residual["position_error_m"])
        normalized = position_error_m / sigma_m if sigma_m > 0.0 else None
        status, flags = _measurement_quality_status(
            position_error_m=position_error_m,
            normalized_position_error=normalized,
            thresholds=thresholds,
        )
        status_counts[status] = status_counts.get(status, 0) + 1
        metrics = {
            "position_error_m": position_error_m,
            "normalized_position_error": normalized,
            "dx_m": float(residual.get("dx_m", 0.0) or 0.0),
            "dy_m": float(residual.get("dy_m", 0.0) or 0.0),
            "dz_m": float(residual.get("dz_m", 0.0) or 0.0),
            "position_sigma_m": sigma_m,
        }
        quality_id = f"ground_measurement_quality:{_payload_digest({'assessment': assessment_id, 'measurement': measurement_ids[idx], 'kind': residual['residual_kind']})}"
        record = GroundStationMeasurementQualityRecord(
            quality_id=quality_id,
            assessment_id=assessment_id,
            measurement_id=measurement_ids[idx],
            object_id=str(manifest.get("object_id", "")),
            station_id=str(manifest.get("station_id", "")),
            od_manifest_path=str(manifest_file),
            od_report_json_path=str(od_report_path),
            residual_kind=str(residual["residual_kind"]),
            time_s=float(residual["time_s"]),
            position_error_m=position_error_m,
            normalized_position_error=normalized,
            status=status,
            flags=tuple(flags),
            metrics=metrics,
            thresholds=thresholds,
        )
        records.append(record)
        rows_out.append(
            {
                "measurement_id": record.measurement_id,
                "residual_kind": record.residual_kind,
                "time_s": record.time_s,
                "position_error_m": record.position_error_m,
                "normalized_position_error": record.normalized_position_error,
                "status": record.status,
                "flags": list(record.flags),
            }
        )
    write_summary = write_ground_station_measurement_quality_records(store_path, records=records)
    return {
        "status": "completed",
        "assessment_id": assessment_id,
        "store_path": str(Path(store_path).expanduser().resolve()),
        "manifest_path": str(manifest_file),
        "od_report_json_path": str(od_report_path),
        "object_id": str(manifest.get("object_id", "")),
        "station_id": str(manifest.get("station_id", "")),
        "measurement_count": len(records),
        "status_counts": status_counts,
        "thresholds": thresholds,
        "write_summary": write_summary,
        "rows": rows_out,
        "non_claims": [
            "Quality rows are advisory residual checks against the fitted OD artifacts.",
            "They are not sensor association, calibration, or covariance-rich tracking validation.",
            "Rejected rows are not deleted from the append-only measurement tables.",
        ],
    }


def _records_from_ground_station_payload(
    measurements: dict[str, Any],
    *,
    jd_utc_start: float | None,
) -> list[GroundStationMeasurementRecord]:
    records: list[GroundStationMeasurementRecord] = []
    for station_id_raw, station_payload_raw in sorted(measurements.items()):
        station_id = str(station_id_raw)
        station_payload = dict(station_payload_raw or {})
        station_metadata = dict(station_payload.get("station", {}) or {})
        station_metadata.setdefault("id", station_id)
        station_metadata["source_measurement_type"] = str(station_payload.get("measurement_type", "") or "")
        targets = dict(station_payload.get("targets", {}) or {})
        for object_id_raw, target_payload_raw in sorted(targets.items()):
            object_id = str(object_id_raw)
            target_payload = dict(target_payload_raw or {})
            for row_raw in list(target_payload.get("measurements", []) or []):
                row = dict(row_raw or {})
                time_jd_utc = _row_jd_utc(row, jd_utc_start=jd_utc_start)
                components = tuple(str(item) for item in row.get("components", []) or [])
                vector = tuple(float(item) for item in row.get("vector", []) or [])
                if not components or not vector:
                    components, vector = _components_from_measurement_row(row)
                sigma = tuple(float(item) for item in row.get("sigma", []) or [])
                if not sigma:
                    sigma = _sigma_from_measurement_row(row, components)
                values = {
                    name: value for name, value in zip(components, vector, strict=False)
                }
                if "valid" in row and type(row["valid"]) is not bool:
                    physical_valid, validity_reason = False, "validity_flag_not_boolean"
                elif row.get("valid", True) is False:
                    physical_valid = False
                    validity_reason = str(row.get("validity_reason", "flagged_invalid") or "flagged_invalid")
                else:
                    physical_valid, validity_reason = _ground_measurement_physical_validity(values)
                quality_flags = dict(row.get("quality_flags", {}) or {})
                quality_flags["physical_valid"] = bool(physical_valid)
                quality_flags["physical_validity_reason"] = validity_reason
                truth_geometry = {
                    key: row[key]
                    for key in (
                        "truth_azimuth_deg",
                        "truth_elevation_deg",
                        "truth_range_km",
                        "truth_range_rate_km_s",
                    )
                    if key in row
                }
                records.append(
                    GroundStationMeasurementRecord(
                        object_id=object_id,
                        station_id=station_id,
                        time_jd_utc=time_jd_utc,
                        measurement_type=str(
                            row.get("measurement_type", station_payload.get("measurement_type", "")) or ""
                        ),
                        components=components,
                        vector=vector,
                        sigma=sigma,
                        station_metadata=station_metadata,
                        quality_flags=quality_flags,
                        truth_geometry=truth_geometry,
                    )
                )
    return records


def _observation_packet_from_store_rows(
    rows: list[dict[str, Any]],
    *,
    object_id: str,
    station_id: str,
    source_label: str,
    frame_context: FrameContext | None = None,
) -> ObservationPacket:
    sorted_rows = sorted(rows, key=lambda item: (float(item["time_jd_utc"]), str(item["measurement_id"])))
    first_jd = float(sorted_rows[0]["time_jd_utc"])
    frame_ctx = frame_context or frame_context_from_mapping(
        {},
        jd_utc_start=first_jd,
        source="scale_ground_station_measurements",
    )
    obs_rows: list[dict[str, Any]] = []
    for row in sorted_rows:
        physically_valid, validity_reason = _stored_ground_measurement_validity(row)
        if not physically_valid:
            raise ValueError(
                f"stored ground-station measurement row is physically invalid ({validity_reason})"
            )
        station = dict(row.get("station_metadata", {}) or {})
        lat_deg = float(station["lat_deg"])
        lon_deg = float(station["lon_deg"])
        alt_km = float(station.get("alt_km", 0.0) or 0.0)
        station_ecef = geodetic_to_ecef_km(lat_deg, lon_deg, alt_km)
        enu_to_ecef = enu_to_ecef_rotation(lat_deg, lon_deg)
        values = {name: float(value) for name, value in zip(row["components"], row["vector"], strict=False)}
        az = np.deg2rad(values["azimuth_deg"])
        el = np.deg2rad(values["elevation_deg"])
        slant_range_km = values["range_km"]
        enu = slant_range_km * np.array(
            [np.cos(el) * np.sin(az), np.cos(el) * np.cos(az), np.sin(el)],
            dtype=float,
        )
        target_ecef = station_ecef + enu_to_ecef @ enu
        jd = float(row["time_jd_utc"])
        target_eci = transform_position(
            target_ecef,
            "ecef",
            "eci",
            t_s=_relative_time_s(jd, first_jd),
            context=frame_ctx,
        )
        obs_rows.append(
            {
                "time_s": _relative_time_s(jd, first_jd),
                "jd_utc": jd,
                "position_eci_km": [float(item) for item in target_eci],
                "position_sigma_km": _position_sigma_from_store_row(row),
            }
        )
    return ingest_observations(
        object_id=object_id,
        observations=obs_rows,
        source_label=source_label,
        frame="eci",
        source_metadata={
            "type": "scale_ground_station_measurements",
            "station_id": station_id,
            "measurement_ids": [str(row["measurement_id"]) for row in sorted_rows],
        },
    )


def _stored_ground_measurement_validity(row: dict[str, Any]) -> tuple[bool, str]:
    """Validate a store row without discarding its raw evidence."""

    quality_flags = dict(row.get("quality_flags", {}) or {})
    if "physical_valid" in quality_flags and type(quality_flags["physical_valid"]) is not bool:
        return False, "physical_valid_flag_not_boolean"
    if quality_flags.get("physical_valid") is False:
        return False, str(
            quality_flags.get("physical_validity_reason", "flagged_invalid") or "flagged_invalid"
        )
    values = {
        name: value
        for name, value in zip(row.get("components", []) or [], row.get("vector", []) or [], strict=False)
    }
    return _ground_measurement_physical_validity(values)


def _row_jd_utc(row: dict[str, Any], *, jd_utc_start: float | None) -> float:
    jd = row.get("jd_utc")
    if jd is not None:
        return float(jd)
    if jd_utc_start is not None and row.get("time_s") is not None:
        return float(jd_utc_start) + float(row["time_s"]) / _DAY_S
    raise ValueError("ground-station measurement rows require jd_utc or a caller-provided jd_utc_start.")


def _relative_time_s(jd_utc: float, first_jd_utc: float) -> float:
    raw = float((jd_utc - first_jd_utc) * _DAY_S)
    nearest_integer = round(raw)
    if abs(raw - nearest_integer) < 1.0e-4:
        return float(nearest_integer)
    return float(round(raw, 9))


def _read_residual_rows(*, fit_csv: Path, holdout_csv: Path, fit_count: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for residual_kind, path in (("fit", fit_csv), ("holdout", holdout_csv)):
        if not path.is_file():
            continue
        with path.open("r", encoding="utf-8", newline="") as f:
            for raw in csv.DictReader(f):
                row = dict(raw)
                row["residual_kind"] = residual_kind
                row["time_s"] = float(row.get("time_s", 0.0) or 0.0)
                row["position_error_m"] = float(row.get("position_error_m", 0.0) or 0.0)
                for key in ("dx_m", "dy_m", "dz_m"):
                    row[key] = float(row.get(key, 0.0) or 0.0)
                rows.append(row)
    fit_rows = [row for row in rows if row["residual_kind"] == "fit"]
    holdout_rows = [row for row in rows if row["residual_kind"] == "holdout"]
    return fit_rows[: max(int(fit_count), 0)] + holdout_rows


def _measurement_quality_status(
    *,
    position_error_m: float,
    normalized_position_error: float | None,
    thresholds: dict[str, Any],
) -> tuple[str, list[str]]:
    flags: list[str] = []
    reject_error_m = thresholds.get("reject_error_m")
    warning_error_m = thresholds.get("warning_error_m")
    if normalized_position_error is not None:
        if normalized_position_error > float(thresholds["reject_sigma"]):
            flags.append("position_residual_exceeds_reject_sigma")
        elif normalized_position_error > float(thresholds["warning_sigma"]):
            flags.append("position_residual_exceeds_warning_sigma")
    if reject_error_m is not None and float(position_error_m) > float(reject_error_m):
        flags.append("position_residual_exceeds_reject_error_m")
    elif warning_error_m is not None and float(position_error_m) > float(warning_error_m):
        flags.append("position_residual_exceeds_warning_error_m")
    if any("reject" in flag for flag in flags):
        return "rejected", flags
    if flags:
        return "warning", flags
    return "accepted", flags


def _components_from_measurement_row(row: dict[str, Any]) -> tuple[tuple[str, ...], tuple[float, ...]]:
    components = ["azimuth_deg", "elevation_deg", "range_km"]
    values = [float(row["azimuth_deg"]), float(row["elevation_deg"]), float(row["range_km"])]
    if row.get("range_rate_km_s") is not None:
        components.append("range_rate_km_s")
        values.append(float(row["range_rate_km_s"]))
    return tuple(components), tuple(values)


def _sigma_from_measurement_row(row: dict[str, Any], components: tuple[str, ...]) -> tuple[float, ...]:
    noise = dict(row.get("noise", {}) or {})
    values = []
    for component in components:
        if component in {"azimuth_deg", "elevation_deg"}:
            values.append(float(row.get("angle_sigma_deg", noise.get("angle_sigma_deg", 0.0)) or 0.0))
        elif component == "range_km":
            values.append(float(row.get("range_sigma_km", noise.get("range_sigma_km", 0.0)) or 0.0))
        elif component == "range_rate_km_s":
            values.append(float(row.get("range_rate_sigma_km_s", noise.get("range_rate_sigma_km_s", 0.0)) or 0.0))
        else:
            values.append(0.0)
    return tuple(values)


def _has_az_el_range(row: dict[str, Any]) -> bool:
    components = {str(item) for item in row.get("components", []) or []}
    return {"azimuth_deg", "elevation_deg", "range_km"}.issubset(components)


def _position_sigma_from_store_row(row: dict[str, Any]) -> float:
    components = [str(item) for item in row.get("components", []) or []]
    sigma_by_component = {
        component: float(sigma) for component, sigma in zip(components, row.get("sigma", []) or [], strict=False)
    }
    range_sigma = sigma_by_component.get("range_km", 0.0)
    angle_sigma_rad = np.deg2rad(
        max(sigma_by_component.get("azimuth_deg", 0.0), sigma_by_component.get("elevation_deg", 0.0))
    )
    values = {name: float(value) for name, value in zip(components, row.get("vector", []) or [], strict=False)}
    angular_sigma_km = abs(values.get("range_km", 0.0)) * float(angle_sigma_rad)
    return max(float(range_sigma), angular_sigma_km, 1.0e-9)


def _payload_digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _safe_name(value: str) -> str:
    out = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(value))
    return out.strip("_") or "object"
