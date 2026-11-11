from __future__ import annotations

import csv
import json
import sqlite3
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from sim.dynamics.orbit.ogp import ogp_propagate_teme, ogp_propagator_name_for_elements, ogp_regime_for_elements
from sim.dynamics.orbit.sdp4 import sdp4_initialize, sdp4_propagate_teme_from_context
from sim.dynamics.orbit.sgp4 import transform_teme_to_output_frame
from sim.dynamics.orbit.tle import TLEElements
from sim.estimation.batch_least_squares import solve_batch_least_squares
from sim.estimation.parameters import EstimatedParameter, ParameterSet
from sim.estimation.partitioning import partition_time_arc
from sim.estimation.sgp4_od import (
    _element_vector,
    _elements_from_vector,
    _elements_payload,
)
from sim.numeric_backend import normalize_numeric_backend
from sim.slr_tracking import C_LIGHT_KM_S, SLR_TWO_WAY_CONTRACT, predict_slr_two_way_time_of_flight_s


def fit_ogp_mean_elements_to_slr_normal_points(
    measurements: Sequence[Mapping[str, Any]],
    *,
    template_elements: TLEElements,
    object_id: str,
    output_dir: str | Path,
    fit_duration_s: float | None = None,
    holdout_duration_s: float | None = None,
    frame_transform: str = "teme_to_eci_iau80",
    frame_model: str = "simple",
    eop_path: str | Path | None = None,
    estimate_station_biases: bool = True,
    station_bias_prior_sigma_km: float = 0.25,
    max_nfev: int = 120,
    robust_loss: str = "huber",
    robust_f_scale: float = 2.0,
    sigma_clip_threshold: float | None = 8.0,
    numeric_backend: str = "rust",
) -> dict[str, Any]:
    """Fit OGP mean elements to actual CRD two-way laser normal points."""
    numeric_backend = normalize_numeric_backend(numeric_backend, error_message="numeric_backend must be 'python' or 'rust'.")
    rows = [_normalize_slr_row(item) for item in measurements]
    if len(rows) < 8:
        raise ValueError("SLR mean-element OD requires at least eight governed normal points.")
    rows.sort(key=lambda item: (item["time_jd_utc"], item["station_id"], item["measurement_id"]))
    epoch0 = float(rows[0]["time_jd_utc"])
    times_s = np.array([(float(row["time_jd_utc"]) - epoch0) * 86400.0 for row in rows], dtype=float)
    for row, time_s in zip(rows, times_s, strict=True):
        row["time_s"] = float(time_s)
    partition = partition_time_arc(
        times_s,
        fit_duration_s=fit_duration_s,
        holdout_duration_s=holdout_duration_s,
        allow_repeated_epochs=True,
    )
    fit_rows = [row for idx, row in enumerate(rows) if bool(partition.fit_mask[idx])]
    holdout_rows = [row for idx, row in enumerate(rows) if bool(partition.holdout_mask[idx])]
    if len(fit_rows) < 8:
        raise ValueError("SLR mean-element OD requires at least eight fit normal points.")

    output_root = Path(output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    station_ids = sorted({str(row["station_id"]) for row in rows})
    fit_station_ids = sorted({str(row["station_id"]) for row in fit_rows})
    holdout_station_ids = sorted({str(row["station_id"]) for row in holdout_rows})
    base = _ogp_mean_element_parameters(_element_vector(template_elements))
    extra = []
    if estimate_station_biases:
        for station_id in fit_station_ids:
            extra.append(
                EstimatedParameter(
                    f"slr_range_bias_km__{station_id}",
                    0.0,
                    scale=0.01,
                    lower=-5.0,
                    upper=5.0,
                    unit="km_one_way_equivalent",
                    description="Aggregate unmodeled one-way station/path/target range correction.",
                )
            )
    parameters = ParameterSet([*base.parameters, *extra])
    state_cache: OrderedDict[tuple[TLEElements, float, str, str], np.ndarray] = OrderedDict()

    def residual(native_values: np.ndarray) -> np.ndarray:
        elements = _elements_from_vector(template_elements, np.asarray(native_values, dtype=float)[:6])
        values = parameters.mapping(native_values)
        try:
            return _slr_residual_vector(
                elements,
                fit_rows,
                parameter_values=values,
                frame_transform=frame_transform,
                frame_model=frame_model,
                eop_path=eop_path,
                state_cache=state_cache,
                numeric_backend=numeric_backend,
            )
        except ValueError:
            return np.full(len(fit_rows), 1.0e12, dtype=float)

    initial_values = parameters.initial_native()
    prefit_records = _slr_residual_records(
        template_elements,
        rows,
        parameter_values=parameters.mapping(initial_values),
        frame_transform=frame_transform,
        frame_model=frame_model,
        eop_path=eop_path,
        partition="prefit",
        state_cache=state_cache,
        numeric_backend=numeric_backend,
    )
    prior_names = [item.name for item in extra]
    prior_mean = np.zeros(len(extra), dtype=float) if extra else None
    prior_cov = np.eye(len(extra), dtype=float) * float(station_bias_prior_sigma_km) ** 2 if extra else None
    solve = solve_batch_least_squares(
        parameters,
        residual,
        max_nfev=int(max_nfev),
        xtol=1.0e-11,
        ftol=1.0e-11,
        gtol=1.0e-11,
        robust_loss=robust_loss,
        robust_f_scale=float(robust_f_scale),
        sigma_clip_threshold=sigma_clip_threshold,
        prior_parameter_names=prior_names or None,
        prior_mean_native=prior_mean,
        prior_covariance_native=prior_cov,
    )
    estimated_values = np.asarray(solve.x_native, dtype=float)
    estimated_elements = _elements_from_vector(template_elements, estimated_values[:6])
    state_epoch_jd_utc = float(rows[0]["time_jd_utc"])
    estimated_state_at_first_observation = _ogp_state_at_jd_factory(
        estimated_elements,
        frame_transform=frame_transform,
        state_cache=state_cache,
    )(state_epoch_jd_utc)
    value_map = parameters.mapping(estimated_values)
    fit_records = _slr_residual_records(
        estimated_elements,
        fit_rows,
        parameter_values=value_map,
        frame_transform=frame_transform,
        frame_model=frame_model,
        eop_path=eop_path,
        partition="fit",
        state_cache=state_cache,
        numeric_backend=numeric_backend,
    )
    holdout_records = _slr_residual_records(
        estimated_elements,
        holdout_rows,
        parameter_values=value_map,
        frame_transform=frame_transform,
        frame_model=frame_model,
        eop_path=eop_path,
        partition="holdout",
        state_cache=state_cache,
        numeric_backend=numeric_backend,
    )
    prefit_fit_records = [record for idx, record in enumerate(prefit_records) if bool(partition.fit_mask[idx])]
    prefit_holdout_records = [record for idx, record in enumerate(prefit_records) if bool(partition.holdout_mask[idx])]
    fit_metrics = _range_metrics(fit_records)
    holdout_metrics = _range_metrics(holdout_records)
    prefit_fit_metrics = _range_metrics(prefit_fit_records)
    prefit_holdout_metrics = _range_metrics(prefit_holdout_records)

    residual_path = output_root / "slr_normal_point_residuals.csv"
    report_path = output_root / "slr_od_report.json"
    product_path = output_root / "fitted_slr_ogp_product.json"
    review_path = output_root / "review" / "run.sqlite"
    _write_residual_csv(residual_path, [*prefit_records, *fit_records, *holdout_records])
    quality = {
        "schema_version": 1,
        "solver_success": bool(solve.success),
        "fit_improved_over_tle_seed": float(fit_metrics["range_rms_m"]) < float(prefit_fit_metrics["range_rms_m"]),
        "holdout_evaluated": bool(holdout_records),
        "actual_tracking_source": all(str(row["source_format"]) == "ilrs_crd_v2" for row in rows),
        "two_way_light_time_contract": all(str(row["measurement_contract"]) == SLR_TWO_WAY_CONTRACT for row in rows),
        "data_full_rank": bool(solve.diagnostics["observability"]["data_full_rank"]),
        "station_bias_prior_enabled": bool(extra),
        "unmodeled_correction_flags_present": any(
            not bool(row["correction_flags"].get("tropospheric_refraction_applied", False))
            or not bool(row["correction_flags"].get("center_of_mass_applied", False))
            for row in rows
        ),
    }
    passing = bool(
        quality["solver_success"]
        and quality["fit_improved_over_tle_seed"]
        and quality["holdout_evaluated"]
        and quality["actual_tracking_source"]
        and quality["two_way_light_time_contract"]
    )
    quality["evidence_status"] = "ready_with_caveats" if passing else "review_required"
    quality["warnings"] = [
        message
        for condition, message in (
            (not quality["data_full_rank"], "The data-only Jacobian is rank deficient or poorly conditioned."),
            (
                quality["unmodeled_correction_flags_present"],
                "Some CRD sessions do not have troposphere and/or target center-of-mass corrections applied; aggregate station biases absorb unresolved one-way terms.",
            ),
        )
        if condition
    ]

    product = {
        "kind": "oel.slr_ogp_mean_element_product",
        "object_id": object_id,
        "measurement_contract": SLR_TWO_WAY_CONTRACT,
        "template_elements": _elements_payload(template_elements),
        "estimated_elements": _elements_payload(estimated_elements),
        "estimated_parameters": parameters.metadata(estimated_values),
        "state_epoch_jd_utc": state_epoch_jd_utc,
        "estimated_state_at_first_observation_eci_km_km_s": estimated_state_at_first_observation.tolist(),
        "claim_boundary": "experimental actual-SLR tracking fit; not a calibrated geodetic orbit product",
    }
    product_path.write_text(json.dumps(product, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    solver_summary = {
        "success": bool(solve.success),
        "message": solve.message,
        "nfev": int(solve.nfev),
        "initial_cost": float(solve.initial_cost),
        "final_cost": float(solve.cost),
        "rms_weighted_residual": float(solve.rms_residual),
        "diagnostics": solve.diagnostics,
        "robust_loss": robust_loss,
        "robust_f_scale": float(robust_f_scale),
        "sigma_clip_threshold": sigma_clip_threshold,
    }
    if numeric_backend == "rust":
        solver_summary["numeric_backend"] = numeric_backend
    report = {
        "schema_version": 1,
        "method": "ogp_mean_elements_to_ilrs_crd_v2_two_way_normal_points",
        "object_id": object_id,
        "measurement_contract": SLR_TWO_WAY_CONTRACT,
        "observation_count": len(rows),
        "fit_observation_count": len(fit_rows),
        "holdout_observation_count": len(holdout_rows),
        "station_ids": station_ids,
        "fit_station_ids": fit_station_ids,
        "holdout_station_ids": holdout_station_ids,
        "observation_partition": partition.summary,
        "frame_contract": {
            "satellite_source_frame": "TEME",
            "ogp_regime": ogp_propagator_name_for_elements(template_elements),
            "satellite_measurement_frame": "ECI",
            "frame_transform": frame_transform,
            "station_coordinate_frame": "ITRF approximate positions from governed ILRS site pages",
            "earth_rotation_model": frame_model,
            "eop_path": None if eop_path is None else str(Path(eop_path).expanduser().resolve()),
        },
        "correction_contract": {
            "h4_flags_preserved": True,
            "station_system_delay": "must be represented by each CRD H4 flag; not silently re-applied",
            "troposphere": "not modeled in this vertical slice",
            "target_center_of_mass": "not modeled in this vertical slice",
            "aggregate_station_range_bias": "estimated with a declared zero-mean prior when enabled",
        },
        "template_elements": _elements_payload(template_elements),
        "estimated_elements": _elements_payload(estimated_elements),
        "estimated_parameters": parameters.metadata(estimated_values),
        "state_epoch_jd_utc": state_epoch_jd_utc,
        "estimated_state_at_first_observation_eci_km_km_s": estimated_state_at_first_observation.tolist(),
        "solver": {
            **({"numeric_backend": "rust"} if numeric_backend == "rust" else {}),
            "success": bool(solve.success),
            "message": solve.message,
            "nfev": int(solve.nfev),
            "initial_cost": float(solve.initial_cost),
            "final_cost": float(solve.cost),
            "rms_weighted_residual": float(solve.rms_residual),
            "diagnostics": solve.diagnostics,
            "robust_loss": robust_loss,
            "robust_f_scale": float(robust_f_scale),
            "sigma_clip_threshold": sigma_clip_threshold,
            "parameter_order": list(solve.parameter_names),
            "parameter_covariance_native": (
                None if solve.covariance_native is None else np.asarray(solve.covariance_native, dtype=float).tolist()
            ),
            "parameter_covariance_claim": "local batch least-squares approximation; not transformed to Cartesian state covariance and not calibrated",
        },
        "prefit_fit_metrics": prefit_fit_metrics,
        "fit_metrics": fit_metrics,
        "prefit_holdout_metrics": prefit_holdout_metrics,
        "holdout_metrics": holdout_metrics,
        "quality_gates": quality,
        "verdict": {
            "passed": passing,
            "maturity": "L4 actual tracking measurement-path evidence with calibration caveats"
            if passing
            else "review_required",
            "scope": "ILRS CRD v2 passive two-way SLR normal points, OGP mean-element refinement, recorded stations and arc only",
        },
        "artifacts": {
            "residual_csv": str(residual_path),
            "fitted_product": str(product_path),
            "review_sqlite": str(review_path),
            "report_json": str(report_path),
        },
        "non_claims": [
            "This is not an ILRS analysis-center-quality orbit solution.",
            "The public TLE is a seed and prediction product, not truth.",
            "Approximate site-page station coordinates are not a substitute for epoch-propagated SINEX coordinates and eccentricities.",
            "Aggregate station biases do not validate individual tropospheric, center-of-mass, calibration, or target-signature corrections.",
        ],
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_review_store(
        review_path, rows=rows, residuals=[*prefit_records, *fit_records, *holdout_records], report=report
    )
    return report


def fit_sgp4_mean_elements_to_slr_normal_points(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Compatibility wrapper for callers using the original Phase 4 function name."""

    return fit_ogp_mean_elements_to_slr_normal_points(*args, **kwargs)


def _ogp_mean_element_parameters(seed_vector: np.ndarray) -> ParameterSet:
    """Build bounds valid for both the near-Earth and deep-space OGP regimes."""

    return ParameterSet(
        [
            EstimatedParameter("inclination_deg", seed_vector[0], scale=1.0, lower=0.0, upper=180.0),
            EstimatedParameter("raan_deg", seed_vector[1], scale=1.0, lower=0.0, upper=360.0),
            EstimatedParameter("eccentricity", seed_vector[2], scale=0.01, lower=1.0e-7, upper=0.999),
            EstimatedParameter("argp_deg", seed_vector[3], scale=1.0, lower=0.0, upper=360.0),
            EstimatedParameter("mean_anomaly_deg", seed_vector[4], scale=1.0, lower=0.0, upper=360.0),
            EstimatedParameter(
                "mean_motion_rev_per_day",
                seed_vector[5],
                scale=0.01,
                lower=0.05,
                upper=18.0,
            ),
        ]
    )


def _normalize_slr_row(raw: Mapping[str, Any]) -> dict[str, Any]:
    row = dict(raw)
    required = {
        "measurement_id",
        "source_format",
        "measurement_contract",
        "station_id",
        "station_metadata",
        "time_jd_utc",
        "epoch_event",
        "observed_time_of_flight_s",
        "sigma_range_km",
        "correction_flags",
    }
    missing = sorted(required - set(row))
    if missing:
        raise ValueError(f"SLR normal-point row is missing required fields: {missing}.")
    if str(row["measurement_contract"]) != SLR_TWO_WAY_CONTRACT:
        raise ValueError(f"SLR row requires measurement_contract={SLR_TWO_WAY_CONTRACT!r}.")
    station = dict(row["station_metadata"])
    xyz = np.asarray(station.get("ecef_km"), dtype=float).reshape(3)
    values = np.array(
        [row["time_jd_utc"], row["observed_time_of_flight_s"], row["sigma_range_km"], *xyz],
        dtype=float,
    )
    if (
        not np.all(np.isfinite(values))
        or float(row["sigma_range_km"]) <= 0.0
        or float(row["observed_time_of_flight_s"]) <= 0.0
    ):
        raise ValueError("SLR normal-point epochs, TOF, station coordinates, and sigma must be finite and valid.")
    event_value = float(row["epoch_event"])
    if not np.isfinite(event_value) or event_value not in {0.0, 1.0, 2.0}:
        raise ValueError("SLR normal-point epoch_event must be 0, 1, or 2.")
    row["epoch_event"] = int(event_value)
    row["station_metadata"] = station
    row["station_id"] = str(row["station_id"]).upper()
    return row


def _slr_residual_records(
    elements: TLEElements,
    rows: Sequence[Mapping[str, Any]],
    *,
    parameter_values: Mapping[str, float],
    frame_transform: str,
    frame_model: str,
    eop_path: str | Path | None,
    partition: str,
    state_cache: OrderedDict[tuple[TLEElements, float, str, str], np.ndarray] | None = None,
    numeric_backend: str = "rust",
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    state_at_jd = _ogp_state_at_jd_factory(
        elements,
        frame_transform=frame_transform,
        state_cache=state_cache,
        numeric_backend=numeric_backend,
    )

    for row in rows:
        station_id = str(row["station_id"])
        bias = float(parameter_values.get(f"slr_range_bias_km__{station_id}", 0.0))
        prediction = predict_slr_two_way_time_of_flight_s(
            epoch_jd_utc=float(row["time_jd_utc"]),
            epoch_event=int(row["epoch_event"]),
            station_ecef_km=dict(row["station_metadata"])["ecef_km"],
            state_at_jd_utc=state_at_jd,
            frame_model=frame_model,
            eop_path=eop_path,
            aggregate_one_way_range_correction_km=bias,
        )
        observed_range = 0.5 * C_LIGHT_KM_S * float(row["observed_time_of_flight_s"])
        predicted_range = float(prediction["predicted_two_way_range_km"])
        residual_km = predicted_range - observed_range
        sigma_km = float(row["sigma_range_km"])
        records.append(
            {
                "partition": partition,
                "measurement_id": str(row["measurement_id"]),
                "station_id": station_id,
                "time_jd_utc": float(row["time_jd_utc"]),
                "epoch_event": int(row["epoch_event"]),
                "observed_time_of_flight_s": float(row["observed_time_of_flight_s"]),
                "predicted_time_of_flight_s": float(prediction["predicted_time_of_flight_s"]),
                "observed_range_km": observed_range,
                "predicted_range_km": predicted_range,
                "residual_km": residual_km,
                "residual_m": residual_km * 1000.0,
                "sigma_range_km": sigma_km,
                "normalized_residual": residual_km / sigma_km,
                "whitened_residual": residual_km / sigma_km,
                "aggregate_station_bias_km": bias,
                "outbound_light_time_s": float(prediction["outbound_light_time_s"]),
                "inbound_light_time_s": float(prediction["inbound_light_time_s"]),
                "correction_flags": json.dumps(dict(row["correction_flags"]), sort_keys=True),
            }
        )
    return records


def _ogp_state_at_jd_factory(
    elements: TLEElements,
    *,
    frame_transform: str,
    state_cache: OrderedDict[tuple[TLEElements, float, str, str], np.ndarray] | None = None,
    numeric_backend: str = "rust",
) -> Any:
    if numeric_backend == "rust":
        from sim.rust_ogp_backend import RustOGPContext

        rust_context = RustOGPContext(elements)
        context = None
    elif numeric_backend == "python":
        rust_context = None
        context = sdp4_initialize(elements) if ogp_regime_for_elements(elements) == "sdp4" else None
    else:
        raise ValueError("numeric_backend must be 'python' or 'rust'.")

    def state_at_jd(jd_utc: float) -> np.ndarray:
        cache_key = (elements, float(jd_utc), str(frame_transform), numeric_backend)
        cached = None if state_cache is None else state_cache.get(cache_key)
        if cached is not None:
            state_cache.move_to_end(cache_key)
            return cached
        tsince_min = (float(jd_utc) - float(elements.epoch_jd_utc)) * 1440.0
        if rust_context is None:
            state = (
                ogp_propagate_teme(elements, tsince_min, backend="python")
                if context is None
                else sdp4_propagate_teme_from_context(context, tsince_min)
            )
            if state.error:
                raise ValueError(state.error)
            position_teme_km, velocity_teme_km_s = state.position_teme_km, state.velocity_teme_km_s
        else:
            position_teme_km, velocity_teme_km_s = rust_context.propagate(tsince_min)
        position, velocity = transform_teme_to_output_frame(
            position_teme_km,
            velocity_teme_km_s,
            jd_utc=float(jd_utc),
            output_frame="eci",
            frame_transform=frame_transform,
        )
        result = np.hstack((position, velocity))
        if state_cache is not None:
            state_cache[cache_key] = result
            state_cache.move_to_end(cache_key)
            if len(state_cache) > 4096:
                state_cache.popitem(last=False)
        return result

    return state_at_jd


def _slr_residual_vector(
    elements: TLEElements,
    rows: Sequence[Mapping[str, Any]],
    *,
    parameter_values: Mapping[str, float],
    frame_transform: str,
    frame_model: str,
    eop_path: str | Path | None,
    state_cache: OrderedDict[tuple[TLEElements, float, str, str], np.ndarray] | None = None,
    numeric_backend: str = "rust",
) -> np.ndarray:
    state_at_jd = _ogp_state_at_jd_factory(
        elements, frame_transform=frame_transform, state_cache=state_cache, numeric_backend=numeric_backend
    )
    raw_residuals = np.empty(len(rows), dtype=float)
    sigmas = np.empty(len(rows), dtype=float)

    for index, row in enumerate(rows):
        station_id = str(row["station_id"])
        bias = float(parameter_values.get(f"slr_range_bias_km__{station_id}", 0.0))
        prediction = predict_slr_two_way_time_of_flight_s(
            epoch_jd_utc=float(row["time_jd_utc"]),
            epoch_event=int(row["epoch_event"]),
            station_ecef_km=dict(row["station_metadata"])["ecef_km"],
            state_at_jd_utc=state_at_jd,
            frame_model=frame_model,
            eop_path=eop_path,
            aggregate_one_way_range_correction_km=bias,
        )
        observed_range = 0.5 * C_LIGHT_KM_S * float(row["observed_time_of_flight_s"])
        raw_residuals[index] = float(prediction["predicted_two_way_range_km"]) - observed_range
        sigmas[index] = float(row["sigma_range_km"])
    if normalize_numeric_backend(numeric_backend) == "rust":
        from sim.rust_estimation_backend import measurement_whiten_rows

        factors = np.zeros((len(rows), 1, 1), dtype=float)
        factors[:, 0, 0] = sigmas
        return measurement_whiten_rows(raw_residuals.reshape(-1, 1), factors).reshape(-1)
    return raw_residuals / sigmas


def _range_metrics(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"status": "not_evaluated", "count": 0}
    residual_m = np.array([float(item["residual_m"]) for item in records], dtype=float)
    normalized = np.array([float(item["normalized_residual"]) for item in records], dtype=float)
    return {
        "status": "evaluated",
        "count": len(records),
        "range_rms_m": float(np.sqrt(np.mean(residual_m * residual_m))),
        "range_mean_m": float(np.mean(residual_m)),
        "range_median_m": float(np.median(residual_m)),
        "range_p95_abs_m": float(np.percentile(np.abs(residual_m), 95.0)),
        "range_max_abs_m": float(np.max(np.abs(residual_m))),
        "normalized_rms": float(np.sqrt(np.mean(normalized * normalized))),
        "by_station": {
            station_id: _range_metrics([item for item in records if str(item["station_id"]) == station_id])
            for station_id in sorted({str(item["station_id"]) for item in records})
        }
        if len({str(item["station_id"]) for item in records}) > 1
        else {},
    }


def _write_residual_csv(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    if not records:
        return
    fields = [key for key in records[0] if key != "correction_flags"] + ["correction_flags"]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def _write_review_store(
    path: Path,
    *,
    rows: Sequence[Mapping[str, Any]],
    residuals: Sequence[Mapping[str, Any]],
    report: Mapping[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            DROP TABLE IF EXISTS slr_normal_points;
            DROP TABLE IF EXISTS slr_residuals;
            DROP TABLE IF EXISTS external_case_summary;
            CREATE TABLE slr_normal_points (
                measurement_id TEXT PRIMARY KEY, station_id TEXT, time_jd_utc REAL,
                epoch_event INTEGER, observed_time_of_flight_s REAL, observed_range_km REAL,
                sigma_range_km REAL, raw_range_count INTEGER, source_sha256 TEXT,
                correction_flags_json TEXT
            );
            CREATE TABLE slr_residuals (
                partition TEXT, measurement_id TEXT, station_id TEXT, time_jd_utc REAL,
                residual_m REAL, normalized_residual REAL, aggregate_station_bias_km REAL
            );
            CREATE TABLE external_case_summary (
                method TEXT, evidence_status TEXT, fit_range_rms_m REAL,
                holdout_range_rms_m REAL, observation_count INTEGER, station_count INTEGER
            );
            """
        )
        connection.executemany(
            "INSERT INTO slr_normal_points VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    str(row["measurement_id"]),
                    str(row["station_id"]),
                    float(row["time_jd_utc"]),
                    int(row["epoch_event"]),
                    float(row["observed_time_of_flight_s"]),
                    0.5 * C_LIGHT_KM_S * float(row["observed_time_of_flight_s"]),
                    float(row["sigma_range_km"]),
                    int(row.get("raw_range_count", 0)),
                    str(row.get("source_sha256", "")),
                    json.dumps(dict(row["correction_flags"]), sort_keys=True),
                )
                for row in rows
            ],
        )
        connection.executemany(
            "INSERT INTO slr_residuals VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    str(item["partition"]),
                    str(item["measurement_id"]),
                    str(item["station_id"]),
                    float(item["time_jd_utc"]),
                    float(item["residual_m"]),
                    float(item["normalized_residual"]),
                    float(item["aggregate_station_bias_km"]),
                )
                for item in residuals
            ],
        )
        connection.execute(
            "INSERT INTO external_case_summary VALUES (?, ?, ?, ?, ?, ?)",
            (
                str(report["method"]),
                str(report["quality_gates"]["evidence_status"]),
                float(report["fit_metrics"]["range_rms_m"]),
                float(report["holdout_metrics"].get("range_rms_m", float("nan"))),
                int(report["observation_count"]),
                len(report["station_ids"]),
            ),
        )
