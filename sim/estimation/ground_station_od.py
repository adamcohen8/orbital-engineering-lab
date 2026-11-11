from __future__ import annotations

import csv
import json
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from sim.dynamics.orbit.frames import FrameContext, frame_context_from_mapping, transform_position, transform_state
from sim.estimation.batch_least_squares import solve_batch_least_squares
from sim.estimation.epoch_evaluation import evaluate_artifact_at_epochs, exact_epoch_provenance
from sim.estimation.ground_systematics import (
    elevation_weighted_covariance,
    extend_parameter_set_for_ground_systematics,
    normalize_ground_systematics,
    systematic_prediction,
)
from sim.estimation.parameters import EstimatedParameter, ParameterSet
from sim.estimation.partitioning import JULIAN_DATE_BOUNDARY_TOLERANCE_S, partition_time_arc
from sim.estimation.residual_audit import build_residual_audit
from sim.estimation.weighting import (
    covariance_from_sigmas,
    prepare_covariance_whitener,
    validate_covariance_block,
    whiten_residual_block,
    whiten_residual_with_factor,
)
from sim.ground_stations import _ground_measurement_physical_validity
from sim.numeric_backend import normalize_numeric_backend
from sim.review import write_workflow_review
from sim.scenarios import ScenarioArtifact
from sim.security.output_writes import mkdir_output, staged_output_file, write_output_text
from sim.utils.geodesy import ecef_to_enu_rotation, enu_to_ecef_rotation, geodetic_to_ecef_km
from sim.utils.vector_math import dot3, norm3

_DAY_S = 86400.0
_DEFAULT_ANGLE_SIGMA_DEG = 0.02
_DEFAULT_RANGE_SIGMA_KM = 0.01
_DEFAULT_RANGE_RATE_SIGMA_KM_S = 1.0e-5


def predict_ground_station_measurement(
    *,
    target_state_eci: Sequence[float],
    station: Mapping[str, Any],
    t_s: float,
    jd_utc_start: float | None,
    frame_context: FrameContext | None = None,
) -> dict[str, float]:
    return _predict_ground_station_measurement(
        target_state_eci=target_state_eci,
        station=station,
        t_s=t_s,
        jd_utc_start=jd_utc_start,
        frame_context=frame_context,
        geometry=None,
    )


def normalize_ground_station_measurements(
    measurements: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return the canonical bounded ground-sensor measurement rows."""

    return _normalize_measurements(measurements)


def _predict_ground_station_measurement(
    *,
    target_state_eci: Sequence[float],
    station: Mapping[str, Any],
    t_s: float,
    jd_utc_start: float | None,
    frame_context: FrameContext | None,
    geometry: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None,
) -> dict[str, float]:
    state = np.array(target_state_eci, dtype=float).reshape(6)
    lat_deg = float(station["lat_deg"])
    lon_deg = float(station["lon_deg"])
    alt_km = float(station.get("alt_km", 0.0) or 0.0)
    frame_ctx = frame_context or frame_context_from_mapping(
        {},
        jd_utc_start=jd_utc_start,
        source="ground_station_od",
    )
    if geometry is None:
        station_ecef = geodetic_to_ecef_km(lat_deg, lon_deg, alt_km)
        station_eci, station_vel_eci = transform_state(
            station_ecef,
            np.zeros(3),
            "ecef",
            "eci",
            t_s=float(t_s),
            context=frame_ctx,
        )
        enu_rotation = ecef_to_enu_rotation(lat_deg, lon_deg)
    else:
        station_ecef, station_eci, station_vel_eci, enu_rotation = geometry
    target_ecef = transform_position(state[:3], "eci", "ecef", t_s=float(t_s), context=frame_ctx)
    rho_ecef = target_ecef - station_ecef
    enu = np.array([dot3(row, rho_ecef) for row in enu_rotation])
    slant_range_km = norm3(rho_ecef)
    if slant_range_km <= 0.0:
        azimuth_deg = 0.0
        elevation_deg = 90.0
        range_rate_km_s = 0.0
    else:
        azimuth_deg = float(np.rad2deg(np.arctan2(enu[0], enu[1])) % 360.0)
        elevation_deg = float(np.rad2deg(np.arcsin(np.clip(enu[2] / slant_range_km, -1.0, 1.0))))
        los_eci = (state[:3] - station_eci) / max(norm3(state[:3] - station_eci), 1.0e-12)
        range_rate_km_s = dot3(state[3:6] - station_vel_eci, los_eci)
    return {
        "azimuth_deg": azimuth_deg,
        "elevation_deg": elevation_deg,
        "range_km": slant_range_km,
        "range_rate_km_s": range_rate_km_s,
    }


def solve_ground_station_measurement_od(
    measurements: Sequence[Mapping[str, Any]],
    *,
    object_id: str,
    output_dir: str | Path,
    initial_state_eci_km_s: Sequence[float] | None = None,
    initial_state_source: str | None = None,
    fit_duration_s: float | None = None,
    holdout_duration_s: float | None = None,
    partition_boundary_tolerance_s: float = JULIAN_DATE_BOUNDARY_TOLERANCE_S,
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
    spherical_harmonics: Mapping[str, Any] | None = None,
    object_specs: Mapping[str, Any] | None = None,
    orbit_force_model: Mapping[str, Any] | None = None,
    environment: Mapping[str, Any] | None = None,
    frame_context: FrameContext | None = None,
    max_nfev: int = 24,
    robust_loss: str = "linear",
    robust_f_scale: float = 1.0,
    sigma_clip_threshold: float | None = None,
    prior_mean_native: Sequence[float] | np.ndarray | None = None,
    prior_covariance_native: Sequence[Sequence[float]] | np.ndarray | None = None,
    systematic_error_model: Mapping[str, Any] | None = None,
    exclude_station_ids: Sequence[str] = (),
    holdout_station_ids: Sequence[str] = (),
    scenario_name: str = "ground_station_sensor_od",
    numeric_backend: str = "rust",
) -> dict[str, Any]:
    numeric_backend = normalize_numeric_backend(numeric_backend, error_message="numeric_backend must be 'python' or 'rust'.")
    if numeric_backend not in {"python", "rust"}:
        raise ValueError("numeric_backend must be 'python' or 'rust'.")
    all_rows = _normalize_measurements(measurements)
    known_station_ids = sorted({str(row["station_id"]) for row in all_rows})
    station_catalog = [
        {
            "station_id": station_id,
            "lat_deg": float(
                next(row for row in all_rows if str(row["station_id"]) == station_id)["station"]["lat_deg"]
            ),
            "lon_deg": float(
                next(row for row in all_rows if str(row["station_id"]) == station_id)["station"]["lon_deg"]
            ),
            "alt_km": float(
                next(row for row in all_rows if str(row["station_id"]) == station_id)["station"].get("alt_km", 0.0)
            ),
            "location_frame": "wgs84_geodetic",
        }
        for station_id in known_station_ids
    ]
    excluded_station_set = {str(item) for item in exclude_station_ids if str(item)}
    holdout_station_set = {str(item) for item in holdout_station_ids if str(item)}
    unknown_station_ids = sorted((excluded_station_set | holdout_station_set) - set(known_station_ids))
    if unknown_station_ids:
        raise ValueError(f"Unknown ground-station selection IDs: {unknown_station_ids}.")
    if excluded_station_set & holdout_station_set:
        raise ValueError("A station cannot be both excluded and held out.")
    excluded_rows = [row for row in all_rows if str(row["station_id"]) in excluded_station_set]
    rows = [row for row in all_rows if str(row["station_id"]) not in excluded_station_set]
    if len(rows) < 2:
        raise ValueError("native ground-station OD requires at least two measurement rows.")
    epoch_jd_utc = min(float(row["time_jd_utc"]) for row in rows)
    exact_time_available = all(row.get("time_tai_seconds") is not None for row in rows)
    epoch_tai_seconds = (
        min(float(row["time_tai_seconds"]) for row in rows) if exact_time_available else None
    )
    rows = [
        {
            **row,
            "time_s": (
                float(row["time_tai_seconds"]) - float(epoch_tai_seconds)
                if epoch_tai_seconds is not None
                else (float(row["time_jd_utc"]) - epoch_jd_utc) * _DAY_S
            ),
        }
        for row in rows
    ]
    output_root = Path(output_dir).expanduser().resolve()
    mkdir_output(output_root)
    frame_ctx = frame_context or frame_context_from_mapping(
        {},
        jd_utc_start=epoch_jd_utc,
        source="ground_station_od",
    )
    if frame_ctx.jd_utc_start is None or not np.isclose(
        float(frame_ctx.jd_utc_start), epoch_jd_utc, rtol=0.0, atol=1.0e-12
    ):
        raise ValueError(
            "ground-station OD frame_context.jd_utc_start must match the retained measurement epoch."
        )
    times = np.array([float(row["time_s"]) for row in rows], dtype=float)
    time_partition = partition_time_arc(
        times,
        fit_duration_s=fit_duration_s,
        holdout_duration_s=holdout_duration_s,
        allow_repeated_epochs=True,
        boundary_tolerance_s=partition_boundary_tolerance_s,
    )
    fit_duration = time_partition.fit_duration_s
    holdout_duration = time_partition.holdout_duration_s
    station_holdout_mask = np.array(
        [str(row["station_id"]) in holdout_station_set for row in rows],
        dtype=bool,
    )
    fit_mask = time_partition.fit_mask & ~station_holdout_mask
    holdout_mask = time_partition.holdout_mask | (time_partition.fit_mask & station_holdout_mask)
    if int(np.count_nonzero(fit_mask)) < 2:
        raise ValueError("native ground-station OD requires at least two fit measurements after station selection.")
    partition_summary = dict(time_partition.summary)
    partition_summary.update(
        {
            "strategy": "time_window_with_station_exclusion_and_holdout",
            "fit_observation_count": int(np.count_nonzero(fit_mask)),
            "holdout_observation_count": int(np.count_nonzero(holdout_mask)),
            "excluded_observation_count": int(len(excluded_rows) + np.count_nonzero(time_partition.excluded_mask)),
            "excluded_station_ids": sorted(excluded_station_set),
            "holdout_station_ids": sorted(holdout_station_set),
            "holdout_status": "evaluated" if bool(np.any(holdout_mask)) else "not_evaluated",
        }
    )
    dt = float(dt_s) if dt_s is not None else _default_dt_from_times(times)
    if dt <= 0.0:
        raise ValueError("dt_s must be positive.")
    sh = _normalize_spherical_harmonics(spherical_harmonics)
    if sh and any(bool(flag) for flag in (j2, j3, j4)):
        raise ValueError("spherical_harmonics cannot be combined with j2, j3, or j4.")
    total_duration = max(fit_duration + holdout_duration, fit_duration, float(np.max(times)), dt)
    total_duration = _integer_multiple_duration(total_duration, dt)
    if initial_state_eci_km_s is not None:
        x0 = np.array(initial_state_eci_km_s, dtype=float).reshape(6)
        prior_source = str(initial_state_source or "provided_initial_state")
    else:
        x0 = _initial_guess_from_measurements(
            rows,
            object_id=object_id,
            epoch_jd_utc=epoch_jd_utc,
            frame_context=frame_ctx,
        )
        prior_source = str(initial_state_source or "measurement_bootstrap")
    base_parameters = _cartesian_state_parameter_set()
    systematic_model = normalize_ground_systematics(
        systematic_error_model,
        station_ids=known_station_ids,
    )
    fit_rows = [row for idx, row in enumerate(rows) if bool(fit_mask[idx])]
    parameters, systematic_parameter_records, systematic_prior_names, systematic_prior_mean, systematic_prior_cov = (
        extend_parameter_set_for_ground_systematics(
            base_parameters,
            rows=fit_rows,
            model=systematic_model,
        )
    )
    prior_names, combined_prior_mean, combined_prior_cov = _combine_ground_priors(
        parameters,
        base_parameters=base_parameters,
        user_mean=prior_mean_native,
        user_covariance=prior_covariance_native,
        systematic_names=systematic_prior_names,
        systematic_mean=systematic_prior_mean,
        systematic_covariance=systematic_prior_cov,
    )
    eval_counter = {"count": 0}
    trajectory_cache: OrderedDict[tuple[bytes, float, bytes], tuple[np.ndarray, np.ndarray]] = OrderedDict()
    geometry_cache: dict[tuple[str, float, float, float, float], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}

    def evaluate(
        native_values: np.ndarray,
        *,
        duration_s: float,
        scratch_name: str,
        requested_epochs: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        state = _state_from_parameters(x0, parameters.mapping(native_values))
        epochs = np.unique(np.asarray(requested_epochs, dtype=float))
        evaluated_duration = _integer_multiple_duration(
            max(float(duration_s), float(np.max(epochs)) if epochs.size else 0.0),
            dt,
        )
        cache_key = (state.tobytes(), float(evaluated_duration), epochs.tobytes())
        cached = trajectory_cache.get(cache_key)
        if cached is not None:
            trajectory_cache.move_to_end(cache_key)
            return cached
        eval_counter["count"] += 1
        artifact = _build_ground_od_propagation_artifact(
            object_id=object_id,
            state_eci_km_s=state,
            epoch_jd_utc=epoch_jd_utc,
            scenario_name=f"{scenario_name}_{scratch_name}",
            output_dir=output_root / "_sensor_od_eval_scratch" / f"{scratch_name}_{eval_counter['count']:04d}",
            duration_s=evaluated_duration,
            dt_s=dt,
            dynamics_model=dynamics_model,
            j2=bool(j2),
            j3=bool(j3),
            j4=bool(j4),
            drag=bool(drag),
            srp=bool(srp),
            third_body_sun=bool(third_body_sun),
            third_body_moon=bool(third_body_moon),
            atmosphere_model=atmosphere_model,
            spherical_harmonics=sh,
        )
        if object_specs or orbit_force_model or environment:
            raw_artifact = artifact.to_dict()
            if object_specs:
                raw_specs = raw_artifact["objects"][object_id].setdefault("specs", {})
                raw_specs.update(_jsonable(dict(object_specs)))
            if orbit_force_model:
                raw_orbit = raw_artifact.setdefault("simulator", {}).setdefault("dynamics", {}).setdefault("orbit", {})
                raw_orbit.update(_jsonable(dict(orbit_force_model)))
            if environment:
                raw_environment = raw_artifact.setdefault("simulator", {}).setdefault("environment", {})
                raw_environment.update(_jsonable(dict(environment)))
            artifact = ScenarioArtifact.from_dict(raw_artifact)
        result = evaluate_artifact_at_epochs(
            artifact,
            object_id=object_id,
            epochs_s=epochs,
        )
        trajectory_cache[cache_key] = result
        trajectory_cache.move_to_end(cache_key)
        if len(trajectory_cache) > 8:
            trajectory_cache.popitem(last=False)
        return result

    clock_model = dict(systematic_model.get("clock_linearization", {}) or {})
    clock_enabled = bool(clock_model.get("enabled", False))
    clock_fd_step_s = float(clock_model.get("finite_difference_step_s", 0.25))
    fit_covariance_factors = _measurement_covariance_factors(fit_rows, systematic_model=systematic_model)

    def evaluate_rows(
        native_values: np.ndarray,
        selected_rows: Sequence[Mapping[str, Any]],
        *,
        duration_s: float,
        scratch_name: str,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        if not selected_rows:
            return np.empty((0, 6), dtype=float), None
        row_times = np.array([float(row["time_s"]) for row in selected_rows], dtype=float)
        requested = row_times
        if clock_enabled:
            requested = np.concatenate((row_times, row_times + clock_fd_step_s))
        sim_t, sim_x = evaluate(
            native_values,
            duration_s=duration_s,
            scratch_name=scratch_name,
            requested_epochs=requested,
        )
        nominal_states = _states_at_epochs(sim_t, sim_x, row_times)
        plus_states = _states_at_epochs(sim_t, sim_x, row_times + clock_fd_step_s) if clock_enabled else None
        return nominal_states, plus_states

    def residual(native_values: np.ndarray) -> np.ndarray:
        fit_states, fit_states_plus = evaluate_rows(
            native_values,
            fit_rows,
            duration_s=fit_duration,
            scratch_name="fit",
        )
        return _measurement_residual_vector(
            fit_rows,
            fit_states,
            epoch_jd_utc=epoch_jd_utc,
            frame_context=frame_ctx,
            systematic_model=systematic_model,
            parameter_values=parameters.mapping(native_values),
            states_plus=fit_states_plus,
            clock_fd_step_s=clock_fd_step_s,
            covariance_factors=fit_covariance_factors,
            geometry_cache=geometry_cache,
            numeric_backend=numeric_backend,
        )

    prefit_rows = [row for idx, row in enumerate(rows) if bool(fit_mask[idx])]
    prefit_states, prefit_states_plus = evaluate_rows(
        parameters.initial_native(),
        prefit_rows,
        duration_s=fit_duration,
        scratch_name="prefit",
    )
    prefit_residuals = _measurement_residual_rows(
        prefit_rows,
        prefit_states,
        epoch_jd_utc=epoch_jd_utc,
        residual_kind="prefit",
        frame_context=frame_ctx,
        systematic_model=systematic_model,
        parameter_values=parameters.mapping(parameters.initial_native()),
        states_plus=prefit_states_plus,
        clock_fd_step_s=clock_fd_step_s,
        geometry_cache=geometry_cache,
    )
    solve = solve_batch_least_squares(
        parameters,
        residual,
        max_nfev=max_nfev,
        robust_loss=robust_loss,
        robust_f_scale=robust_f_scale,
        sigma_clip_threshold=sigma_clip_threshold,
        prior_mean_native=combined_prior_mean,
        prior_covariance_native=combined_prior_cov,
        prior_parameter_names=prior_names,
    )
    fit_states, fit_states_plus = evaluate_rows(
        solve.x_native,
        fit_rows,
        duration_s=fit_duration,
        scratch_name="postfit",
    )
    fit_residuals = _measurement_residual_rows(
        fit_rows,
        fit_states,
        epoch_jd_utc=epoch_jd_utc,
        residual_kind="fit",
        frame_context=frame_ctx,
        systematic_model=systematic_model,
        parameter_values=parameters.mapping(solve.x_native),
        states_plus=fit_states_plus,
        clock_fd_step_s=clock_fd_step_s,
        geometry_cache=geometry_cache,
    )
    holdout_rows = [row for idx, row in enumerate(rows) if bool(holdout_mask[idx])]
    holdout_states, holdout_states_plus = evaluate_rows(
        solve.x_native,
        holdout_rows,
        duration_s=total_duration,
        scratch_name="prediction",
    )
    holdout_residuals = _measurement_residual_rows(
        holdout_rows,
        holdout_states,
        epoch_jd_utc=epoch_jd_utc,
        residual_kind="holdout",
        frame_context=frame_ctx,
        systematic_model=systematic_model,
        parameter_values=parameters.mapping(solve.x_native),
        states_plus=holdout_states_plus,
        clock_fd_step_s=clock_fd_step_s,
        geometry_cache=geometry_cache,
    )
    fitted_state = _state_from_parameters(x0, parameters.mapping(solve.x_native))
    residual_csv_path = output_root / "ground_sensor_od_residuals.csv"
    residual_plot_path = output_root / "ground_sensor_od_normalized_residuals.png"
    report_json_path = output_root / "ground_sensor_od_report.json"
    report_md_path = output_root / "ground_sensor_od_report.md"
    fitted_packet_path = output_root / "fitted_mission_input_packet.json"
    _write_residual_csv(residual_csv_path, [*prefit_residuals, *fit_residuals, *holdout_residuals])
    _write_residual_plot(residual_plot_path, [*prefit_residuals, *fit_residuals, *holdout_residuals])
    write_output_text(
        str(fitted_packet_path),
        json.dumps(
            _fitted_state_packet(
                object_id=object_id,
                state_eci_km_s=fitted_state,
                epoch_jd_utc=epoch_jd_utc,
                frame_provenance=frame_ctx.metadata(),
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
    decision_records = _label_sensor_residual_decisions(
        solve.decision_records,
        fit_rows,
    )
    all_residuals = [*prefit_residuals, *fit_residuals, *holdout_residuals]
    systematic_evidence = _systematic_parameter_evidence(
        systematic_parameter_records,
        parameters=parameters,
        values=solve.x_native,
        diagnostics=solve.diagnostics,
        covariance_native=solve.covariance_native,
    )
    station_comparison = _station_comparison(
        prefit_residuals,
        fit_residuals,
        holdout_residuals,
        decision_records=decision_records,
        included_station_ids=sorted({str(row["station_id"]) for row in rows}),
        excluded_station_ids=sorted(excluded_station_set),
        holdout_station_ids=sorted(holdout_station_set),
        station_catalog=station_catalog,
    )
    exclusion_records = [
        {
            "measurement_id": str(row["measurement_id"]),
            "station_id": str(row["station_id"]),
            "time_jd_utc": float(row["time_jd_utc"]),
            "time_s": float(row["time_s"]),
            "measurement_type": str(row["measurement_type"]),
            "components": list(row["components"]),
            "reason": "station_excluded_by_request",
        }
        for row in excluded_rows
    ]
    estimation_policy = {
        "robust_loss": robust_loss,
        "robust_f_scale": float(robust_f_scale),
        "sigma_clip_threshold": sigma_clip_threshold,
        "prior_enabled": combined_prior_mean is not None,
        "prior_parameter_names": list(prior_names or []),
    }
    if numeric_backend == "rust":
        estimation_policy["numeric_backend"] = numeric_backend
    report = {
        "method": "ground_station_sensor_dynamics_least_squares",
        "object_id": object_id,
        "scenario_name": scenario_name,
        "dynamics_model": dynamics_model,
        "j2": bool(j2),
        "j3": bool(j3),
        "j4": bool(j4),
        "drag": bool(drag),
        "srp": bool(srp),
        "third_body_sun": bool(third_body_sun),
        "third_body_moon": bool(third_body_moon),
        "atmosphere_model": str(atmosphere_model or ""),
        "spherical_harmonics": dict(sh or {}),
        "object_specs": dict(object_specs or {}),
        "orbit_force_model": dict(orbit_force_model or {}),
        "environment": dict(environment or {}),
        "frame_provenance": frame_ctx.metadata(),
        "epoch_jd_utc": epoch_jd_utc,
        "station_count": len(known_station_ids),
        "station_catalog": station_catalog,
        "included_station_ids": sorted({str(row["station_id"]) for row in rows}),
        "excluded_station_ids": sorted(excluded_station_set),
        "holdout_station_ids": sorted(holdout_station_set),
        "observation_count": len(rows),
        "fit_observation_count": int(np.count_nonzero(fit_mask)),
        "holdout_observation_count": int(np.count_nonzero(holdout_mask)),
        "fit_duration_s": fit_duration,
        "holdout_duration_s": holdout_duration,
        "dt_s": dt,
        "observation_partition": partition_summary,
        "epoch_evaluation": exact_epoch_provenance(times[fit_mask | holdout_mask]),
        "initial_state_source": prior_source,
        "estimated_parameters": parameters.metadata(solve.x_native),
        "systematic_error_model": systematic_model,
        "systematic_parameter_evidence": systematic_evidence,
        "station_comparison": station_comparison,
        "exclusions": exclusion_records,
        "solver": {
            "success": bool(solve.success),
            "message": solve.message,
            "nfev": int(solve.nfev),
            "initial_cost": float(solve.initial_cost),
            "final_cost": float(solve.cost),
            "rms_weighted_residual": float(solve.rms_residual),
            "diagnostics": solve.diagnostics,
            "decision_records": decision_records,
        },
        "estimation_policy": estimation_policy,
        "prefit_metrics": _sensor_residual_metrics(prefit_residuals),
        "fit_metrics": _sensor_residual_metrics(fit_residuals),
        "holdout_metrics": _sensor_residual_metrics(holdout_residuals),
        "residual_audit": build_residual_audit(
            all_residuals,
            decision_records=decision_records,
        ),
        "initial_state_eci_km_s": x0.tolist(),
        "fitted_state_eci_km_s": fitted_state.tolist(),
        "state_covariance_eci_km_s": _state_covariance_eci_km_s(solve.covariance_native),
        "residual_csv_path": str(residual_csv_path),
        "residual_plot_path": str(residual_plot_path),
        "fitted_mission_input_packet_path": str(fitted_packet_path),
        "report_json_path": str(report_json_path),
        "report_md_path": str(report_md_path),
        "non_claims": [
            "This is native synthetic/geometric ground-station measurement OD, not calibrated operational sensor processing.",
            "Bias and clock estimates are valid only when their data-only identifiability and prior-influence diagnostics pass.",
            "The Bennett refraction option corrects apparent elevation only; it is not a calibrated tropospheric range-delay model.",
            "Light-time, association, media calibration, and operational tracking covariance are not included.",
        ],
    }
    report["quality_gates"] = _ground_sensor_quality_gates(report)
    report["verdict"] = _ground_sensor_verdict(report["quality_gates"])
    review_root = output_root / "review"
    anticipated_review_outputs = {
        "workflow_manifest_json": str(review_root / "workflow_manifest.json"),
        "sqlite": str(review_root / "run.sqlite"),
        "schema_json": str(review_root / "schema.json"),
        "saved_views_json": str(review_root / "saved_views.json"),
    }
    report["review"] = anticipated_review_outputs
    write_output_text(str(report_json_path), json.dumps(report, indent=2, sort_keys=True) + "\n")
    write_output_text(str(report_md_path), _render_report_md(report))
    review_outputs = write_workflow_review(
        output_dir=output_root,
        workflow_type="ground_station_orbit_determination",
        title=scenario_name,
        scenario_name=scenario_name,
        status="complete" if report["verdict"]["evidence_status"] == "ready_with_caveats" else "review_required",
        summary={
            "object_id": object_id,
            "station_count": len(known_station_ids),
            "fit_observation_count": int(np.count_nonzero(fit_mask)),
            "holdout_observation_count": int(np.count_nonzero(holdout_mask)),
            "excluded_observation_count": len(exclusion_records),
            "fit_weighted_rms": report["fit_metrics"]["weighted_rms"],
            "holdout_weighted_rms": report["holdout_metrics"]["weighted_rms"],
            "evidence_status": report["verdict"]["evidence_status"],
        },
        artifacts={
            "report_json": str(report_json_path),
            "report_markdown": str(report_md_path),
            "residual_csv": str(residual_csv_path),
            "residual_plot": str(residual_plot_path),
            "fitted_state_packet": str(fitted_packet_path),
        },
        recommended_queries=[
            {
                "name": "ground_od_stations",
                "description": "Station geodetic metadata and fit/holdout/exclusion disposition.",
                "sql": "SELECT * FROM ground_od_stations ORDER BY station_id",
            },
            {
                "name": "ground_od_station_comparison",
                "description": "Station-level fit, holdout, rejection, and disposition evidence.",
                "sql": "SELECT * FROM ground_od_station_comparison ORDER BY station_id",
            },
            {
                "name": "ground_od_residuals",
                "description": "Component residuals with correction and weighting provenance.",
                "sql": "SELECT measurement_id, station_id, partition, time_jd_utc, time_s, component, residual, whitened_residual, clock_shift_s, refraction_correction_deg, elevation_sigma_scale FROM ground_od_residuals ORDER BY time_s, station_id, measurement_id, component",
            },
            {
                "name": "ground_od_systematics",
                "description": "Estimated systematic parameters and observability diagnostics.",
                "sql": "SELECT * FROM ground_od_systematic_parameters ORDER BY parameter",
            },
            {
                "name": "ground_od_exclusions",
                "description": "Excluded measurements and immutable reasons.",
                "sql": "SELECT * FROM ground_od_exclusions ORDER BY time_s, station_id, measurement_id",
            },
        ],
        recommended_review_order=[
            "ground_od_stations",
            "ground_od_station_comparison",
            "ground_od_systematics",
            "ground_od_residuals",
            "ground_od_exclusions",
        ],
        provenance={
            "frame": report["frame_provenance"],
            "epoch_evaluation": report["epoch_evaluation"],
            "systematic_error_model": systematic_model,
        },
        tables={
            "ground_od_residuals": all_residuals,
            "ground_od_stations": [
                {
                    **station,
                    "disposition": next(
                        item["disposition"]
                        for item in station_comparison["stations"]
                        if item["station_id"] == station["station_id"]
                    ),
                }
                for station in station_catalog
            ],
            "ground_od_station_comparison": list(station_comparison["stations"]),
            "ground_od_systematic_parameters": systematic_evidence,
            "ground_od_exclusions": exclusion_records,
            "ground_od_decisions": decision_records,
        },
    )
    if review_outputs != anticipated_review_outputs:
        raise RuntimeError("ground-station OD review output contract changed unexpectedly")
    return report


def _normalize_measurements(measurements: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    raw_rows = [dict(row or {}) for row in measurements]
    if not raw_rows:
        return []
    exact_epoch_flags = [row.get("time_tai_seconds") is not None for row in raw_rows]
    if any(exact_epoch_flags) and not all(exact_epoch_flags):
        raise ValueError(
            "measurement rows must provide time_tai_seconds consistently for exact epoch identity."
        )
    rows: list[dict[str, Any]] = []
    raw_epochs: list[float] = []
    raw_epoch_identities: list[float] = []
    for idx, raw in enumerate(raw_rows):
        raw_epoch = raw.get("time_jd_utc", raw.get("jd_utc"))
        if raw_epoch is None:
            raise ValueError(f"measurement row {idx} requires time_jd_utc or jd_utc.")
        epoch = float(raw_epoch)
        if not np.isfinite(epoch):
            raise ValueError(f"measurement row {idx} epoch must be finite.")
        raw_epochs.append(epoch)
        if exact_epoch_flags[idx]:
            exact_identity = float(raw["time_tai_seconds"])
            if not np.isfinite(exact_identity):
                raise ValueError(f"measurement row {idx} exact epoch identity must be finite.")
            raw_epoch_identities.append(exact_identity)
    first_jd = min(raw_epochs)
    first_epoch_identity = min(raw_epoch_identities) if raw_epoch_identities else None
    seen_measurement_ids: set[str] = set()
    station_metadata_by_id: dict[str, tuple[float, float, float]] = {}
    supported_components = {"azimuth_deg", "elevation_deg", "range_km", "range_rate_km_s"}
    for idx, raw in enumerate(raw_rows):
        jd = raw.get("time_jd_utc", raw.get("jd_utc"))
        if jd is None:
            raise ValueError(f"measurement row {idx} requires time_jd_utc or jd_utc.")
        jd_f = float(jd)
        if not np.isfinite(jd_f):
            raise ValueError(f"measurement row {idx} epoch must be finite.")
        station = dict(raw.get("station_metadata", raw.get("station", {})) or {})
        if not {"lat_deg", "lon_deg"}.issubset(station):
            raise ValueError(f"measurement row {idx} requires station lat_deg and lon_deg metadata.")
        station_values = np.array(
            [station["lat_deg"], station["lon_deg"], station.get("alt_km", 0.0)],
            dtype=float,
        )
        if not np.all(np.isfinite(station_values)):
            raise ValueError(f"measurement row {idx} station coordinates must be finite.")
        lat_deg, lon_deg, alt_km = (float(item) for item in station_values)
        if not -90.0 <= lat_deg <= 90.0:
            raise ValueError(f"measurement row {idx} station lat_deg must be in [-90, 90].")
        if not -180.0 <= lon_deg <= 360.0:
            raise ValueError(f"measurement row {idx} station lon_deg must be in [-180, 360].")
        station_id = str(raw.get("station_id", station.get("id", "")) or "").strip()
        if not station_id:
            raise ValueError(f"measurement row {idx} requires a non-empty station_id.")
        coordinates = (lat_deg, lon_deg, alt_km)
        previous_coordinates = station_metadata_by_id.get(station_id)
        if previous_coordinates is not None and not np.allclose(
            coordinates,
            previous_coordinates,
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise ValueError(f"station {station_id!r} has inconsistent geodetic metadata across measurements.")
        station_metadata_by_id[station_id] = coordinates
        station.update({"id": station_id, "lat_deg": lat_deg, "lon_deg": lon_deg, "alt_km": alt_km})
        components = [str(item) for item in raw.get("components", []) or []]
        vector = [float(item) for item in raw.get("vector", []) or []]
        if not components or not vector:
            components, vector = _components_from_row(raw)
        if len(components) != len(vector):
            raise ValueError(f"measurement row {idx} components and vector must have the same length.")
        if len(components) != len(set(components)) or not set(components).issubset(supported_components):
            raise ValueError(f"measurement row {idx} contains duplicate or unsupported components.")
        if not np.all(np.isfinite(np.asarray(vector, dtype=float))):
            raise ValueError(f"measurement row {idx} vector must be finite.")
        values = {name: value for name, value in zip(components, vector, strict=False)}
        if "valid" in raw and type(raw["valid"]) is not bool:
            raise ValueError(f"measurement row {idx} validity flag must be boolean.")
        if raw.get("valid", True) is False:
            reason = str(raw.get("validity_reason", "flagged_invalid") or "flagged_invalid")
            raise ValueError(
                f"measurement row {idx} is physically invalid ({reason}); "
                "invalid raw measurements cannot enter orbit determination."
            )
        physically_valid, validity_reason = _ground_measurement_physical_validity(values)
        if not physically_valid:
            raise ValueError(
                f"measurement row {idx} is physically invalid ({validity_reason}); "
                "impossible angular/range domains cannot enter orbit determination."
            )
        sigma_values = [float(item) for item in raw.get("sigma", []) or []]
        if sigma_values and len(sigma_values) != len(components):
            raise ValueError(f"measurement row {idx} sigma must match the measurement dimension.")
        sigmas = _sigma_by_component(components, sigma_values, raw)
        raw_uncertainty = dict(raw.get("uncertainty", {}) or {})
        raw_covariance = raw.get(
            "covariance",
            raw.get("covariance_matrix", raw_uncertainty.get("matrix")),
        )
        if raw_covariance is None:
            covariance = covariance_from_sigmas([sigmas[component] for component in components])
            covariance_source = "diagonal_sigmas"
        else:
            covariance = validate_covariance_block(
                raw_covariance,
                dimension=len(components),
                field_name=f"measurement row {idx} covariance",
            )
            covariance_source = str(raw_uncertainty.get("source", "provided_covariance"))
        measurement_id = str(raw.get("measurement_id", f"measurement:{idx}") or "").strip()
        if not measurement_id:
            raise ValueError(f"measurement row {idx} measurement_id must be non-empty.")
        if measurement_id in seen_measurement_ids:
            raise ValueError(f"duplicate measurement_id: {measurement_id!r}.")
        seen_measurement_ids.add(measurement_id)
        rows.append(
            {
                "source_index": idx,
                "measurement_id": measurement_id,
                "station_id": station_id,
                "station": station,
                "time_jd_utc": jd_f,
                "time_tai_seconds": (
                    None if raw.get("time_tai_seconds") is None else float(raw["time_tai_seconds"])
                ),
                "time_s": float(
                    float(raw["time_tai_seconds"]) - first_epoch_identity
                    if first_epoch_identity is not None
                    else (jd_f - first_jd) * _DAY_S
                ),
                "time_system": "utc_julian_date",
                "frame": "sensor_topocentric",
                "partition": "unassigned",
                "measurement_type": str(raw.get("measurement_type", "") or "+".join(components)),
                "arc_id": str(raw.get("arc_id", "ground_station_od") or "ground_station_od"),
                "components": components,
                "values": values,
                "sigmas": sigmas,
                "uncertainty": {
                    "representation": "covariance",
                    "components": components,
                    "matrix": covariance.tolist(),
                    "source": covariance_source,
                },
                "source_record": _jsonable(raw),
                "normalization": {
                    "epoch_transform": "jd_utc_to_seconds_from_earliest_measurement",
                    "value_transform": "identity",
                    "sigma_transform": "zero_as_unspecified_default_by_component"
                    if any(value == 0.0 for value in sigma_values)
                    else "identity_or_default_when_absent",
                    "station_frame": "ecef_from_geodetic_wgs84",
                },
            }
        )
    rows.sort(key=lambda item: (float(item["time_s"]), str(item["measurement_id"])))
    previous = -np.inf
    for row in rows:
        t = float(row["time_s"])
        if t < previous - 1.0e-9:
            raise ValueError("ground-station measurement times must be nondecreasing after normalization.")
        previous = t
    return rows


def _components_from_row(row: Mapping[str, Any]) -> tuple[list[str], list[float]]:
    components = ["azimuth_deg", "elevation_deg", "range_km"]
    values = [float(row["azimuth_deg"]), float(row["elevation_deg"]), float(row["range_km"])]
    if row.get("range_rate_km_s") is not None:
        components.append("range_rate_km_s")
        values.append(float(row["range_rate_km_s"]))
    return components, values


def _sigma_by_component(
    components: Sequence[str], sigma_values: Sequence[float], row: Mapping[str, Any]
) -> dict[str, float]:
    out: dict[str, float] = {}
    for idx, component in enumerate(components):
        if idx < len(sigma_values):
            sigma = float(sigma_values[idx])
            if not np.isfinite(sigma) or sigma < 0.0:
                raise ValueError("measurement sigma entries must be finite and non-negative.")
            if sigma > 0.0:
                out[str(component)] = sigma
                continue
        if component in {"azimuth_deg", "elevation_deg"}:
            out[str(component)] = float(
                row.get("angle_sigma_deg", _DEFAULT_ANGLE_SIGMA_DEG) or _DEFAULT_ANGLE_SIGMA_DEG
            )
        elif component == "range_km":
            out[str(component)] = float(row.get("range_sigma_km", _DEFAULT_RANGE_SIGMA_KM) or _DEFAULT_RANGE_SIGMA_KM)
        elif component == "range_rate_km_s":
            out[str(component)] = float(
                row.get("range_rate_sigma_km_s", _DEFAULT_RANGE_RATE_SIGMA_KM_S) or _DEFAULT_RANGE_RATE_SIGMA_KM_S
            )
        else:
            out[str(component)] = 1.0
        if not np.isfinite(out[str(component)]) or out[str(component)] <= 0.0:
            raise ValueError(f"measurement sigma for {component!r} must be finite and positive.")
    return out


def _normalize_spherical_harmonics(value: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not value:
        return None
    raw = dict(value)
    if not bool(raw.get("enabled", True)):
        return None
    degree = int(raw.get("degree", 0) or 0)
    if degree < 2:
        raise ValueError("spherical_harmonics requires degree >= 2.")
    order = int(raw.get("order", degree) if raw.get("order", None) is not None else degree)
    if order < 0 or order > degree:
        raise ValueError("spherical_harmonics order must satisfy 0 <= order <= degree.")
    out: dict[str, Any] = {"enabled": True, "degree": degree, "order": order}
    for key in (
        "source",
        "model",
        "coeff_path",
        "source_path",
        "frame_model",
        "eop_path",
        "reference_radius_km",
        "fd_step_km",
        "normalized",
        "terms",
    ):
        item = raw.get(key)
        if item not in (None, ""):
            out[key] = _jsonable(item)
    if out.get("coeff_path") and not out.get("source"):
        out["source"] = "hpop_ggm03"
    return out


def _build_ground_od_propagation_artifact(
    *,
    object_id: str,
    state_eci_km_s: Sequence[float],
    epoch_jd_utc: float,
    scenario_name: str,
    output_dir: str | Path,
    duration_s: float,
    dt_s: float,
    dynamics_model: str,
    j2: bool,
    j3: bool,
    j4: bool,
    drag: bool,
    srp: bool,
    third_body_sun: bool,
    third_body_moon: bool,
    atmosphere_model: str | None,
    spherical_harmonics: Mapping[str, Any] | None,
) -> ScenarioArtifact:
    """Build the internal scratch trajectory through the public scenario contract."""

    state = np.asarray(state_eci_km_s, dtype=float).reshape(6)
    orbit: dict[str, Any] = {
        "model": str(dynamics_model),
        "orbit_substep_s": float(dt_s),
        "j2": bool(j2),
        "j3": bool(j3),
        "j4": bool(j4),
        "drag": bool(drag),
        "srp": bool(srp),
        "third_body_sun": bool(third_body_sun),
        "third_body_moon": bool(third_body_moon),
    }
    environment: dict[str, Any] = {"ephemeris_mode": "analytic_simple", "atmosphere_env": {}}
    if atmosphere_model:
        model = str(atmosphere_model).strip().lower()
        orbit["atmosphere_model"] = model
        environment["atmosphere_model"] = model
    if spherical_harmonics:
        orbit["spherical_harmonics"] = _jsonable(dict(spherical_harmonics))
    return ScenarioArtifact.from_dict(
        {
            "scenario_name": str(scenario_name),
            "scenario_description": "Ground-station OD deterministic trajectory evaluation.",
            "metadata": {
                "owner": "oel-ground-station-od",
                "assumptions": [
                    "The estimator evaluates only the declared OEL dynamics and does not query external services."
                ],
            },
            "objects": {
                str(object_id): {
                    "object_id": str(object_id),
                    "kind": "satellite",
                    "enabled": True,
                    "role": str(object_id),
                    "runtime_profile": "trajectory_only",
                    "specs": {"mass_kg": 100.0},
                    "initial_state": {
                        "position_eci_km": state[:3].tolist(),
                        "velocity_eci_km_s": state[3:].tolist(),
                        "epoch_jd_utc": float(epoch_jd_utc),
                    },
                }
            },
            "simulator": {
                "duration_s": float(duration_s),
                "dt_s": float(dt_s),
                "initial_jd_utc": float(epoch_jd_utc),
                "dynamics": {
                    "orbit": orbit,
                    "attitude": {"enabled": False},
                    "rocket": {"enabled": False},
                },
                "environment": environment,
                "plugin_validation": {"strict": True},
                "termination": {"earth_impact_enabled": False},
            },
            "outputs": {
                "output_dir": str(output_dir),
                "mode": "save",
                "stats": {"print_summary": False, "save_json": True, "save_csv": False, "save_full_log": False},
                "review": {"enabled": True, "detail": "standard"},
                "plots": {"enabled": False, "figure_ids": []},
                "animations": {"enabled": False, "types": []},
            },
        }
    )


def _fitted_state_packet(
    *,
    object_id: str,
    state_eci_km_s: Sequence[float],
    epoch_jd_utc: float,
    frame_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    state = np.asarray(state_eci_km_s, dtype=float).reshape(6)
    return {
        "packet_version": 1,
        "kind": "oel.mission_input_packet",
        "source": {"type": "state_vector", "label": "ground_station_sensor_od"},
        "frame_provenance": _jsonable(dict(frame_provenance)),
        "objects": {
            str(object_id): {
                "object_id": str(object_id),
                "kind": "satellite",
                "role": str(object_id),
                "state_type": "state_vector",
                "frame": "ECI",
                "initial_state": {
                    "position_eci_km": state[:3].tolist(),
                    "velocity_eci_km_s": state[3:].tolist(),
                    "epoch_jd_utc": float(epoch_jd_utc),
                },
                "normalized_units": {"position": "km", "velocity": "km/s"},
                "provenance": {
                    "source_label": "ground_station_sensor_od",
                    "source_type": "state_vector",
                    "frame_model": str(frame_provenance.get("model", "")),
                },
            }
        },
        "warnings": [],
        "validation": {
            "status": "ready",
            "notes": ["Generated from the fitted OEL ground-station OD state."],
        },
    }


def _state_covariance_eci_km_s(covariance_native: np.ndarray | None) -> list[list[float]] | None:
    if covariance_native is None:
        return None
    native = np.asarray(covariance_native, dtype=float)
    if native.ndim != 2 or native.shape[0] < 6 or native.shape[1] < 6:
        raise ValueError("ground-station OD covariance must contain the six Cartesian correction parameters.")
    native_to_state = np.array([1.0e-3, 1.0e-3, 1.0e-3, 1.0e-6, 1.0e-6, 1.0e-6], dtype=float)
    converted = native[:6, :6] * np.outer(native_to_state, native_to_state)
    return converted.tolist()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _initial_guess_from_measurements(
    rows: Sequence[Mapping[str, Any]],
    *,
    object_id: str,
    epoch_jd_utc: float,
    frame_context: FrameContext,
) -> np.ndarray:
    from sim.observations import fit_state_from_position_observations, ingest_observations

    obs_rows = []
    seen_times: set[float] = set()
    for row in rows:
        time_s = float(row["time_s"])
        if any(abs(time_s - seen) <= 1.0e-9 for seen in seen_times):
            continue
        values = dict(row["values"])
        if not {"azimuth_deg", "elevation_deg", "range_km"}.issubset(values):
            continue
        seen_times.add(time_s)
        station = dict(row["station"])
        lat_deg = float(station["lat_deg"])
        lon_deg = float(station["lon_deg"])
        station_ecef = geodetic_to_ecef_km(lat_deg, lon_deg, float(station.get("alt_km", 0.0) or 0.0))
        enu_to_ecef = enu_to_ecef_rotation(lat_deg, lon_deg)
        az = np.deg2rad(float(values["azimuth_deg"]))
        el = np.deg2rad(float(values["elevation_deg"]))
        rng = float(values["range_km"])
        enu = rng * np.array([np.cos(el) * np.sin(az), np.cos(el) * np.cos(az), np.sin(el)], dtype=float)
        eci = transform_position(
            station_ecef + enu_to_ecef @ enu,
            "ecef",
            "eci",
            t_s=float(row["time_s"]),
            context=frame_context,
        )
        obs_rows.append(
            {
                "time_s": float(row["time_s"]),
                "jd_utc": float(row["time_jd_utc"]),
                "position_eci_km": [float(item) for item in eci],
                "position_sigma_km": max(
                    float(dict(row["sigmas"]).get("range_km", _DEFAULT_RANGE_SIGMA_KM)),
                    1.0e-9,
                ),
            }
        )
    if len(obs_rows) < 2:
        raise ValueError(
            "automatic initial-state bootstrap requires at least two distinct measurements containing "
            "azimuth, elevation, and range; provide initial_state_eci_km_s for component-only data."
        )
    packet = ingest_observations(
        object_id=object_id,
        observations=obs_rows,
        source_label="ground_sensor_od_initial_guess",
    )
    fit = fit_state_from_position_observations(packet, object_id=object_id)
    return np.array(fit["state_eci_km_s"], dtype=float).reshape(6)


def _cartesian_state_parameter_set() -> ParameterSet:
    return ParameterSet(
        [
            EstimatedParameter("dx_m", 0.0, scale=100.0, lower=-100000.0, upper=100000.0, unit="m"),
            EstimatedParameter("dy_m", 0.0, scale=100.0, lower=-100000.0, upper=100000.0, unit="m"),
            EstimatedParameter("dz_m", 0.0, scale=100.0, lower=-100000.0, upper=100000.0, unit="m"),
            EstimatedParameter("dvx_mm_s", 0.0, scale=10.0, lower=-100000.0, upper=100000.0, unit="mm/s"),
            EstimatedParameter("dvy_mm_s", 0.0, scale=10.0, lower=-100000.0, upper=100000.0, unit="mm/s"),
            EstimatedParameter("dvz_mm_s", 0.0, scale=10.0, lower=-100000.0, upper=100000.0, unit="mm/s"),
        ]
    )


def _combine_ground_priors(
    parameters: ParameterSet,
    *,
    base_parameters: ParameterSet,
    user_mean: Sequence[float] | np.ndarray | None,
    user_covariance: Sequence[Sequence[float]] | np.ndarray | None,
    systematic_names: Sequence[str],
    systematic_mean: np.ndarray,
    systematic_covariance: np.ndarray,
) -> tuple[list[str] | None, np.ndarray | None, np.ndarray | None]:
    if (user_mean is None) != (user_covariance is None):
        raise ValueError("prior_mean_native and prior_covariance_native must be provided together.")
    systematic_name_list = [str(name) for name in systematic_names]
    if user_mean is None or user_covariance is None:
        if not systematic_name_list:
            return None, None, None
        return (
            systematic_name_list,
            np.asarray(systematic_mean, dtype=float),
            np.asarray(
                systematic_covariance,
                dtype=float,
            ),
        )
    mean = np.asarray(user_mean, dtype=float).reshape(-1)
    covariance = np.asarray(user_covariance, dtype=float)
    if mean.size == len(parameters.parameters):
        return list(parameters.names), mean, covariance
    if mean.size != len(base_parameters.parameters):
        raise ValueError(
            "ground OD prior dimension must match either the six Cartesian state parameters or the full extended "
            "parameter set."
        )
    if not systematic_name_list:
        return list(base_parameters.names), mean, covariance
    combined_covariance = np.zeros(
        (mean.size + systematic_mean.size, mean.size + systematic_mean.size),
        dtype=float,
    )
    combined_covariance[: mean.size, : mean.size] = covariance
    combined_covariance[mean.size :, mean.size :] = systematic_covariance
    return (
        [*base_parameters.names, *systematic_name_list],
        np.concatenate((mean, systematic_mean)),
        combined_covariance,
    )


def _state_from_parameters(initial_state: np.ndarray, values: Mapping[str, float]) -> np.ndarray:
    state = np.array(initial_state, dtype=float).reshape(6).copy()
    state[:3] += np.array([values.get("dx_m", 0.0), values.get("dy_m", 0.0), values.get("dz_m", 0.0)]) / 1000.0
    state[3:] += (
        np.array([values.get("dvx_mm_s", 0.0), values.get("dvy_mm_s", 0.0), values.get("dvz_mm_s", 0.0)]) / 1.0e6
    )
    return state


def _measurement_residual_vector(
    rows: Sequence[Mapping[str, Any]],
    states: np.ndarray,
    *,
    epoch_jd_utc: float,
    frame_context: FrameContext,
    systematic_model: Mapping[str, Any] | None = None,
    parameter_values: Mapping[str, float] | None = None,
    states_plus: np.ndarray | None = None,
    clock_fd_step_s: float = 0.25,
    covariance_factors: Sequence[np.ndarray] | None = None,
    geometry_cache: dict[tuple[str, float, float, float, float], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] | None = None,
    numeric_backend: str = "rust",
) -> np.ndarray:
    residual_blocks: list[np.ndarray] = []
    factor_blocks: list[np.ndarray] = []
    native_predictions = None
    native_plus_predictions = None
    if normalize_numeric_backend(numeric_backend) == "rust":
        native_predictions = _native_ground_station_predictions(
            rows,
            states,
            epoch_jd_utc=epoch_jd_utc,
            frame_context=frame_context,
            geometry_cache=geometry_cache,
        )
        if states_plus is not None:
            native_plus_predictions = _native_ground_station_predictions(
                rows,
                states_plus,
                epoch_jd_utc=epoch_jd_utc,
                frame_context=frame_context,
                geometry_cache=geometry_cache,
                time_offset_s=float(clock_fd_step_s),
            )
    for index, (row, state) in enumerate(zip(rows, states, strict=True)):
        pred, _corrections = _predicted_measurement_with_systematics(
            row,
            state,
            state_plus=None if states_plus is None else states_plus[index],
            epoch_jd_utc=epoch_jd_utc,
            frame_context=frame_context,
            systematic_model=systematic_model,
            parameter_values=parameter_values,
            clock_fd_step_s=clock_fd_step_s,
            geometry_cache=geometry_cache,
            geometric_override=None if native_predictions is None else native_predictions[index],
            geometric_plus_override=(
                None if native_plus_predictions is None else native_plus_predictions[index]
            ),
        )
        residual = np.array(
            [
                _component_residual(component, float(pred[component]), float(dict(row["values"])[component]))
                for component in row["components"]
            ],
            dtype=float,
        )
        if covariance_factors is None:
            covariance, _weighting = elevation_weighted_covariance(
                _measurement_covariance(row),
                row=row,
                model=dict(systematic_model or {}),
            )
            factor = prepare_covariance_whitener(
                covariance,
                dimension=residual.size,
                field_name=f"measurement {row['measurement_id']!r} covariance",
            )
        else:
            factor = np.asarray(covariance_factors[index], dtype=float)
        residual_blocks.append(residual)
        factor_blocks.append(factor)
    if not residual_blocks:
        return np.zeros(0, dtype=float)
    if normalize_numeric_backend(numeric_backend) == "rust":
        from sim.rust_estimation_backend import measurement_whiten_variable_rows

        return measurement_whiten_variable_rows(residual_blocks, factor_blocks)
    return np.concatenate(
        [
            whiten_residual_with_factor(
                residual,
                factor,
                field_name=f"measurement {rows[index]['measurement_id']!r} covariance",
            )
            for index, (residual, factor) in enumerate(zip(residual_blocks, factor_blocks, strict=True))
        ]
    )


def _native_ground_station_predictions(
    rows: Sequence[Mapping[str, Any]],
    states: np.ndarray,
    *,
    epoch_jd_utc: float,
    frame_context: FrameContext,
    geometry_cache: dict[tuple[str, float, float, float, float], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] | None,
    time_offset_s: float = 0.0,
) -> list[dict[str, float]] | None:
    """Prepare frame-owned geometry and evaluate the native station kernel."""

    from sim.rust_estimation_backend import ground_station_predictions

    states_array = np.asarray(states, dtype=float)
    if states_array.shape != (len(rows), 6):
        raise ValueError("ground-station states must have shape (row_count, 6)")
    target_ecef: list[np.ndarray] = []
    station_ecef: list[np.ndarray] = []
    station_eci: list[np.ndarray] = []
    station_velocity_eci: list[np.ndarray] = []
    enu_rotations: list[np.ndarray] = []
    for row, state in zip(rows, states_array, strict=True):
        time_s = float(row["time_s"]) + float(time_offset_s)
        geometry = _cached_ground_station_geometry(
            geometry_cache,
            station_id=str(row["station_id"]),
            station=dict(row["station"]),
            t_s=time_s,
            frame_context=frame_context,
        )
        if geometry is None:
            return None
        station_ecef_row, station_eci_row, station_velocity_row, enu_rotation = geometry
        target_ecef.append(
            transform_position(
                state[:3],
                "eci",
                "ecef",
                t_s=time_s,
                context=frame_context,
            )
        )
        station_ecef.append(station_ecef_row)
        station_eci.append(station_eci_row)
        station_velocity_eci.append(station_velocity_row)
        enu_rotations.append(enu_rotation)
    predicted = ground_station_predictions(
        states_array,
        np.asarray(target_ecef, dtype=float),
        np.asarray(station_ecef, dtype=float),
        np.asarray(station_eci, dtype=float),
        np.asarray(station_velocity_eci, dtype=float),
        np.asarray(enu_rotations, dtype=float),
    )
    names = ("azimuth_deg", "elevation_deg", "range_km", "range_rate_km_s")
    return [{name: float(predicted[index, column]) for column, name in enumerate(names)} for index in range(len(rows))]


def _measurement_covariance_factors(
    rows: Sequence[Mapping[str, Any]],
    *,
    systematic_model: Mapping[str, Any] | None = None,
) -> tuple[np.ndarray, ...]:
    factors: list[np.ndarray] = []
    for row in rows:
        covariance, _weighting = elevation_weighted_covariance(
            _measurement_covariance(row),
            row=row,
            model=dict(systematic_model or {}),
        )
        factors.append(
            prepare_covariance_whitener(
                covariance,
                dimension=len(row["components"]),
                field_name=f"measurement {row['measurement_id']!r} covariance",
            )
        )
    return tuple(factors)


def _measurement_residual_rows(
    rows: Sequence[Mapping[str, Any]],
    states: np.ndarray,
    *,
    epoch_jd_utc: float,
    residual_kind: str,
    frame_context: FrameContext,
    systematic_model: Mapping[str, Any] | None = None,
    parameter_values: Mapping[str, float] | None = None,
    states_plus: np.ndarray | None = None,
    clock_fd_step_s: float = 0.25,
    geometry_cache: dict[tuple[str, float, float, float, float], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] | None = None,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for index, (row, state) in enumerate(zip(rows, states, strict=True)):
        pred, corrections = _predicted_measurement_with_systematics(
            row,
            state,
            state_plus=None if states_plus is None else states_plus[index],
            epoch_jd_utc=epoch_jd_utc,
            frame_context=frame_context,
            systematic_model=systematic_model,
            parameter_values=parameter_values,
            clock_fd_step_s=clock_fd_step_s,
            geometry_cache=geometry_cache,
        )
        residual_vector = np.array(
            [
                _component_residual(
                    component,
                    float(pred[component]),
                    float(dict(row["values"])[component]),
                )
                for component in row["components"]
            ],
            dtype=float,
        )
        covariance, weighting = elevation_weighted_covariance(
            _measurement_covariance(row),
            row=row,
            model=dict(systematic_model or {}),
        )
        whitened = whiten_residual_block(
            residual_vector,
            covariance,
            field_name=f"measurement {row['measurement_id']!r} covariance",
        )
        for component_index, component in enumerate(row["components"]):
            observed = float(dict(row["values"])[component])
            predicted = float(pred[component])
            residual = _component_residual(component, predicted, observed)
            sigma = max(float(dict(row["sigmas"])[component]), 1.0e-12)
            out.append(
                {
                    "residual_kind": residual_kind,
                    "partition": "holdout" if residual_kind == "holdout" else "fit",
                    "measurement_id": str(row["measurement_id"]),
                    "measurement_type": str(row.get("measurement_type") or "ground_station_combined"),
                    "station_id": str(row["station_id"]),
                    "arc_id": str(row.get("arc_id") or "ground_station_od"),
                    "frame": "sensor_topocentric_from_eci",
                    "epoch_evaluation_method": "simulation_session_variable_step_exact",
                    "time_jd_utc": float(row["time_jd_utc"]),
                    "time_s": float(row["time_s"]),
                    "component": str(component),
                    "observed": observed,
                    "predicted": predicted,
                    "residual": residual,
                    "sigma": sigma,
                    "normalized_residual": residual / sigma,
                    "whitened_residual": float(whitened[component_index]),
                    "covariance_source": str(dict(row.get("uncertainty", {}) or {}).get("source", "")),
                    "systematic_bias": float(
                        dict(corrections.get("biases", {}) or {}).get(_bias_for_component(component), 0.0)
                    ),
                    "clock_shift_s": float(corrections.get("clock_shift_s", 0.0) or 0.0),
                    "refraction_correction_deg": float(corrections.get("refraction_correction_deg", 0.0) or 0.0),
                    "elevation_sigma_scale": float(weighting.get("sigma_scale", 1.0) or 1.0),
                    "correction_provenance": corrections,
                    "weighting_provenance": weighting,
                }
            )
    return out


def _predicted_measurement_with_systematics(
    row: Mapping[str, Any],
    state: np.ndarray,
    *,
    state_plus: np.ndarray | None,
    epoch_jd_utc: float,
    frame_context: FrameContext,
    systematic_model: Mapping[str, Any] | None,
    parameter_values: Mapping[str, float] | None,
    clock_fd_step_s: float,
    geometry_cache: dict[tuple[str, float, float, float, float], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] | None = None,
    geometric_override: Mapping[str, float] | None = None,
    geometric_plus_override: Mapping[str, float] | None = None,
) -> tuple[dict[str, float], dict[str, Any]]:
    station = dict(row["station"])
    nominal_time_s = float(row["time_s"])
    geometric = (
        dict(geometric_override)
        if geometric_override is not None
        else _predict_ground_station_measurement(
            target_state_eci=state,
            station=station,
            t_s=nominal_time_s,
            jd_utc_start=epoch_jd_utc,
            frame_context=frame_context,
            geometry=_cached_ground_station_geometry(
                geometry_cache,
                station_id=str(row["station_id"]),
                station=station,
                t_s=nominal_time_s,
                frame_context=frame_context,
            ),
        )
    )
    derivative: dict[str, float] | None = None
    if state_plus is not None:
        plus_time_s = nominal_time_s + float(clock_fd_step_s)
        plus = (
            dict(geometric_plus_override)
            if geometric_plus_override is not None
            else _predict_ground_station_measurement(
                target_state_eci=state_plus,
                station=station,
                t_s=plus_time_s,
                jd_utc_start=epoch_jd_utc,
                frame_context=frame_context,
                geometry=_cached_ground_station_geometry(
                    geometry_cache,
                    station_id=str(row["station_id"]),
                    station=station,
                    t_s=plus_time_s,
                    frame_context=frame_context,
                ),
            )
        )
        derivative = {
            component: _component_residual(component, float(plus[component]), float(geometric[component]))
            / float(clock_fd_step_s)
            for component in geometric
        }
    return systematic_prediction(
        geometric,
        row=row,
        parameter_values=dict(parameter_values or {}),
        model=dict(systematic_model or {}),
        time_derivative=derivative,
    )


def _cached_ground_station_geometry(
    cache: dict[tuple[str, float, float, float, float], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] | None,
    *,
    station_id: str,
    station: Mapping[str, Any],
    t_s: float,
    frame_context: FrameContext,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    if cache is None:
        return None
    lat_deg = float(station["lat_deg"])
    lon_deg = float(station["lon_deg"])
    alt_km = float(station.get("alt_km", 0.0) or 0.0)
    key = (str(station_id), float(t_s), lat_deg, lon_deg, alt_km)
    cached = cache.get(key)
    if cached is not None:
        return cached
    station_ecef = geodetic_to_ecef_km(lat_deg, lon_deg, alt_km)
    station_eci, station_vel_eci = transform_state(
        station_ecef,
        np.zeros(3),
        "ecef",
        "eci",
        t_s=float(t_s),
        context=frame_context,
    )
    value = (station_ecef, station_eci, station_vel_eci, ecef_to_enu_rotation(lat_deg, lon_deg))
    cache[key] = value
    return value


def _bias_for_component(component: str) -> str:
    return {
        "range_km": "range_bias_km",
        "range_rate_km_s": "range_rate_bias_km_s",
        "azimuth_deg": "azimuth_bias_deg",
        "elevation_deg": "elevation_bias_deg",
    }.get(str(component), "")


def _component_residual(component: str, predicted: float, observed: float) -> float:
    if component == "azimuth_deg":
        return _wrap_degrees(predicted - observed)
    return float(predicted - observed)


def _measurement_covariance(row: Mapping[str, Any]) -> np.ndarray:
    components = list(row.get("components", []) or [])
    uncertainty = dict(row.get("uncertainty", {}) or {})
    matrix = uncertainty.get("matrix")
    if matrix is not None:
        return validate_covariance_block(
            matrix,
            dimension=len(components),
            field_name=f"measurement {row.get('measurement_id', '')!r} covariance",
        )
    return covariance_from_sigmas([float(dict(row["sigmas"])[component]) for component in components])


def _wrap_degrees(value: float) -> float:
    return float((float(value) + 180.0) % 360.0 - 180.0)


def _sensor_residual_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_component: dict[str, list[float]] = {}
    norm_values = []
    for row in rows:
        component = str(row["component"])
        by_component.setdefault(component, []).append(float(row["residual"]))
        norm_values.append(float(row.get("whitened_residual", row["normalized_residual"])))
    component_metrics = {}
    for component, values in sorted(by_component.items()):
        arr = np.array(values, dtype=float)
        component_metrics[component] = {
            "count": int(arr.size),
            "rms": float(np.sqrt(np.mean(arr**2))) if arr.size else float("nan"),
            "max_abs": float(np.max(np.abs(arr))) if arr.size else float("nan"),
        }
    norm_arr = np.array(norm_values, dtype=float)
    return {
        "residual_count": int(norm_arr.size),
        "weighted_rms": float(np.sqrt(np.mean(norm_arr**2))) if norm_arr.size else float("nan"),
        "weighted_max_abs": float(np.max(np.abs(norm_arr))) if norm_arr.size else float("nan"),
        "components": component_metrics,
    }


def _systematic_parameter_evidence(
    records: Sequence[Mapping[str, Any]],
    *,
    parameters: ParameterSet,
    values: np.ndarray,
    diagnostics: Mapping[str, Any],
    covariance_native: np.ndarray | None,
) -> list[dict[str, Any]]:
    value_by_name = parameters.mapping(values)
    observability = dict(diagnostics.get("observability", {}) or {})
    diagnostic_by_name = {
        str(item.get("parameter", "")): dict(item) for item in list(observability.get("parameters", []) or [])
    }
    covariance = None if covariance_native is None else np.asarray(covariance_native, dtype=float)
    evidence: list[dict[str, Any]] = []
    for raw_record in records:
        record = dict(raw_record)
        name = str(record["parameter"])
        estimate = float(value_by_name[name])
        prior_mean = float(record["prior_mean"])
        prior_sigma = float(record["prior_sigma"])
        diagnostic = diagnostic_by_name.get(name, {})
        index = parameters.names.index(name)
        posterior_sigma = None
        if covariance is not None and covariance.shape == (len(parameters.names), len(parameters.names)):
            variance = float(covariance[index, index])
            if np.isfinite(variance) and variance >= 0.0:
                posterior_sigma = float(np.sqrt(variance))
        prior_pull_sigma = (estimate - prior_mean) / prior_sigma
        identifiable = bool(diagnostic.get("identifiable", False))
        prior_dominated = bool(diagnostic.get("prior_dominated", False))
        evidence.append(
            {
                **record,
                "estimate": estimate,
                "posterior_sigma": posterior_sigma,
                "prior_pull_sigma": float(prior_pull_sigma),
                "data_jacobian_column_norm": diagnostic.get("data_jacobian_column_norm"),
                "max_abs_correlation": diagnostic.get("max_abs_correlation"),
                "data_identifiable": identifiable,
                "prior_dominated": prior_dominated,
                "prior_influence_assessment": (
                    "prior_dominated_not_data_identifiable"
                    if prior_dominated
                    else "data_identifiable_with_declared_prior"
                    if identifiable
                    else "not_data_identifiable"
                ),
            }
        )
    return evidence


def _station_comparison(
    prefit_rows: Sequence[Mapping[str, Any]],
    fit_rows: Sequence[Mapping[str, Any]],
    holdout_rows: Sequence[Mapping[str, Any]],
    *,
    decision_records: Sequence[Mapping[str, Any]],
    included_station_ids: Sequence[str],
    excluded_station_ids: Sequence[str],
    holdout_station_ids: Sequence[str],
    station_catalog: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    decisions_by_station: dict[str, list[dict[str, Any]]] = {}
    for decision in decision_records:
        station_id = str(decision.get("station_id", ""))
        decisions_by_station.setdefault(station_id, []).append(dict(decision))
    station_ids = sorted(set(included_station_ids) | set(excluded_station_ids))
    excluded = set(excluded_station_ids)
    held_out = set(holdout_station_ids)
    metadata_by_station = {str(item.get("station_id", "")): dict(item) for item in station_catalog}
    stations: list[dict[str, Any]] = []
    for station_id in station_ids:
        station_prefit = [row for row in prefit_rows if str(row.get("station_id")) == station_id]
        station_fit = [row for row in fit_rows if str(row.get("station_id")) == station_id]
        station_holdout = [row for row in holdout_rows if str(row.get("station_id")) == station_id]
        prefit_metrics = _sensor_residual_metrics(station_prefit)
        fit_metrics = _sensor_residual_metrics(station_fit)
        holdout_metrics = _sensor_residual_metrics(station_holdout)
        station_decisions = decisions_by_station.get(station_id, [])
        rejected_count = sum(not bool(item.get("accepted", True)) for item in station_decisions)
        decision_count = len(station_decisions)
        rejection_fraction = float(rejected_count / decision_count) if decision_count else 0.0
        fit_rms = float(fit_metrics["weighted_rms"])
        fit_max = float(fit_metrics["weighted_max_abs"])
        if station_id in excluded:
            disposition = "excluded_by_request"
            reason = "station_excluded_by_request"
        elif station_id in held_out:
            disposition = "holdout_only"
            reason = "cross_station_holdout"
        elif (
            (np.isfinite(fit_rms) and fit_rms > 5.0)
            or (np.isfinite(fit_max) and fit_max > 10.0)
            or rejection_fraction >= 0.2
        ):
            disposition = "review_required"
            reason = "large_station_residuals_or_rejection_fraction"
        else:
            disposition = "included"
            reason = "station_evidence_within_screening_thresholds"
        stations.append(
            {
                **metadata_by_station.get(station_id, {}),
                "station_id": station_id,
                "disposition": disposition,
                "reason": reason,
                "prefit_residual_count": int(prefit_metrics["residual_count"]),
                "prefit_weighted_rms": _finite_or_none(prefit_metrics["weighted_rms"]),
                "fit_residual_count": int(fit_metrics["residual_count"]),
                "fit_weighted_rms": _finite_or_none(fit_metrics["weighted_rms"]),
                "fit_weighted_max_abs": _finite_or_none(fit_metrics["weighted_max_abs"]),
                "holdout_residual_count": int(holdout_metrics["residual_count"]),
                "holdout_weighted_rms": _finite_or_none(holdout_metrics["weighted_rms"]),
                "decision_count": decision_count,
                "rejected_residual_count": rejected_count,
                "rejected_residual_fraction": rejection_fraction,
            }
        )
    review_required = [item["station_id"] for item in stations if item["disposition"] == "review_required"]
    return {
        "schema_version": 1,
        "stations": stations,
        "review_required_station_ids": review_required,
        "all_included_stations_screened": not review_required,
        "screening_thresholds": {
            "fit_weighted_rms": 5.0,
            "fit_weighted_max_abs": 10.0,
            "rejected_residual_fraction": 0.2,
        },
    }


def _finite_or_none(value: Any) -> float | None:
    parsed = float(value)
    return parsed if np.isfinite(parsed) else None


def _ground_sensor_quality_gates(report: Mapping[str, Any]) -> dict[str, Any]:
    solver = dict(report.get("solver", {}) or {})
    prefit = dict(report.get("prefit_metrics", {}) or {})
    fit = dict(report.get("fit_metrics", {}) or {})
    holdout = dict(report.get("holdout_metrics", {}) or {})
    prefit_rms = float(prefit.get("weighted_rms", np.nan))
    fit_rms = float(fit.get("weighted_rms", np.nan))
    holdout_rms = float(holdout.get("weighted_rms", np.nan))
    fit_improved = bool(np.isfinite(prefit_rms) and np.isfinite(fit_rms) and fit_rms < prefit_rms)
    solver_reported_success = bool(solver.get("success", False))
    solver_success = solver_reported_success
    warnings = []
    diagnostics = dict(solver.get("diagnostics", {}) or {})
    observability = dict(diagnostics.get("observability", {}) or {})
    data_full_rank = bool(observability.get("data_full_rank", False))
    holdout_evaluated = int(holdout.get("residual_count", 0) or 0) > 0
    station_comparison = dict(report.get("station_comparison", {}) or {})
    review_required_stations = list(station_comparison.get("review_required_station_ids", []) or [])
    systematic_evidence = list(report.get("systematic_parameter_evidence", []) or [])
    unidentifiable_systematics = [
        str(item.get("parameter")) for item in systematic_evidence if not bool(item.get("data_identifiable", False))
    ]
    prior_dominated_systematics = [
        str(item.get("parameter")) for item in systematic_evidence if bool(item.get("prior_dominated", False))
    ]
    if not solver_success:
        warnings.append("solver_not_successful")
    if not data_full_rank:
        warnings.append("data_jacobian_rank_deficient")
    if not fit_improved:
        warnings.append("fit_did_not_improve_prefit_residual")
    if not holdout_evaluated:
        warnings.append("holdout_not_evaluated")
    elif np.isfinite(holdout_rms) and holdout_rms > max(10.0, 5.0 * max(fit_rms, 1.0e-12)):
        warnings.append("holdout_weighted_residual_large")
    if review_required_stations:
        warnings.append("station_residual_screen_review_required")
    if unidentifiable_systematics:
        warnings.append("systematic_parameter_not_data_identifiable")
    if prior_dominated_systematics:
        warnings.append("systematic_parameter_prior_dominated")
    return {
        "schema_version": 1,
        "solver_success": solver_success,
        "solver_reported_success": solver_reported_success,
        "data_full_rank": data_full_rank,
        "failure_classification": diagnostics.get("failure_classification"),
        "fit_improved_prefit_rms": fit_improved,
        "prefit_weighted_rms": prefit_rms,
        "fit_weighted_rms": fit_rms,
        "holdout_weighted_rms": holdout_rms,
        "holdout_evidence_status": "evaluated" if holdout_evaluated else "not_evaluated",
        "review_required_station_ids": review_required_stations,
        "unidentifiable_systematic_parameters": unidentifiable_systematics,
        "prior_dominated_systematic_parameters": prior_dominated_systematics,
        "warnings": warnings,
    }


def _label_sensor_residual_decisions(
    decisions: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    index_map = [
        {
            "measurement_id": str(row["measurement_id"]),
            "station_id": str(row["station_id"]),
            "time_s": float(row["time_s"]),
            "component": str(component),
        }
        for row in rows
        for component in row["components"]
    ]
    labeled: list[dict[str, Any]] = []
    for decision in decisions:
        record = dict(decision)
        index = int(record["residual_index"])
        if index < len(index_map):
            record.update(index_map[index])
        labeled.append(record)
    return labeled


def _ground_sensor_verdict(quality_gates: Mapping[str, Any]) -> dict[str, Any]:
    warnings = list(quality_gates.get("warnings", []) or [])
    usable = (
        bool(quality_gates.get("solver_success"))
        and bool(quality_gates.get("fit_improved_prefit_rms"))
        and not warnings
    )
    return {
        "evidence_status": "ready_with_caveats" if usable else "review_required",
        "analyst_action": "review_native_sensor_od_report" if not usable else "usable_for_propagation_study",
        "warning_count": len(warnings),
    }


def _write_residual_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fieldnames = [
        "residual_kind",
        "partition",
        "measurement_id",
        "station_id",
        "frame",
        "epoch_evaluation_method",
        "time_jd_utc",
        "time_s",
        "component",
        "observed",
        "predicted",
        "residual",
        "sigma",
        "normalized_residual",
        "whitened_residual",
        "covariance_source",
        "systematic_bias",
        "clock_shift_s",
        "refraction_correction_deg",
        "elevation_sigma_scale",
    ]
    mkdir_output(path.parent)
    with staged_output_file(path) as target:
        with target.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows([{key: row.get(key, "") for key in fieldnames} for row in rows])


def _write_residual_plot(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcdefaults()
    fig, ax = plt.subplots(figsize=(11.0, 5.2), facecolor="white")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.tick_params(colors="black")
    for spine in ax.spines.values():
        spine.set_color("black")
    plotted = False
    marker_by_kind = {"prefit": "x", "fit": "o", "holdout": "^"}
    components = sorted({str(row.get("component", "")) for row in rows})
    palette = plt.get_cmap("tab10")
    color_by_component = {component: palette(index % 10) for index, component in enumerate(components)}
    for kind in ("prefit", "fit", "holdout"):
        kind_rows = [dict(row) for row in rows if str(row.get("residual_kind", "")) == kind]
        for component in components:
            component_rows = [row for row in kind_rows if str(row.get("component", "")) == component]
            if not component_rows:
                continue
            ax.scatter(
                [float(row["time_s"]) for row in component_rows],
                [float(row.get("whitened_residual", row["normalized_residual"])) for row in component_rows],
                color=color_by_component[component],
                marker=marker_by_kind[kind],
                s=28,
                alpha=0.85,
            )
            plotted = True
    ax.axhline(0.0, color="black", linewidth=0.8)
    plotted_values = np.array(
        [float(row.get("whitened_residual", row.get("normalized_residual", 0.0))) for row in rows],
        dtype=float,
    )
    use_symlog = bool(plotted_values.size and float(np.max(np.abs(plotted_values))) > 25.0)
    if use_symlog:
        ax.set_yscale("symlog", linthresh=3.0)
    ax.set_xlabel("Time (s)", color="black")
    ax.set_ylabel("Whitened residual", color="black")
    title = "Ground-Station Sensor OD Residuals"
    if use_symlog:
        title += " (symmetric log scale)"
    ax.set_title(title, color="black")
    if plotted:
        from matplotlib.lines import Line2D

        component_handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                color=color_by_component[component],
                label=component,
            )
            for component in components
        ]
        present_kinds = [kind for kind in marker_by_kind if any(row.get("residual_kind") == kind for row in rows)]
        kind_handles = [
            Line2D(
                [0],
                [0],
                marker=marker_by_kind[kind],
                linestyle="none",
                color="black",
                label=kind,
            )
            for kind in present_kinds
        ]
        ax.legend(
            handles=[*component_handles, *kind_handles],
            fontsize=7,
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            borderaxespad=0.0,
            title="Component / residual kind",
            title_fontsize=7,
            facecolor="white",
            labelcolor="black",
        )
    fig.tight_layout(rect=(0.0, 0.0, 0.82, 1.0))
    mkdir_output(path.parent)
    with staged_output_file(path) as target:
        fig.savefig(target, dpi=140, facecolor="white", transparent=False)
    plt.close(fig)


def _render_report_md(report: Mapping[str, Any]) -> str:
    solver = dict(report.get("solver", {}) or {})
    fit = dict(report.get("fit_metrics", {}) or {})
    holdout = dict(report.get("holdout_metrics", {}) or {})
    station_comparison = dict(report.get("station_comparison", {}) or {})
    systematic_evidence = list(report.get("systematic_parameter_evidence", []) or [])
    review = dict(report.get("review", {}) or {})
    lines = [
        "# Ground-Station Sensor OD Report",
        "",
        f"- Object: `{report.get('object_id')}`",
        f"- Method: `{report.get('method')}`",
        f"- Solver success: `{solver.get('success')}`",
        f"- Fit weighted RMS: `{fit.get('weighted_rms')}`",
        f"- Holdout weighted RMS: `{holdout.get('weighted_rms')}`",
        f"- Included stations: `{', '.join(report.get('included_station_ids', []) or [])}`",
        f"- Held-out stations: `{', '.join(report.get('holdout_station_ids', []) or [])}`",
        f"- Excluded stations: `{', '.join(report.get('excluded_station_ids', []) or [])}`",
        f"- Residual CSV: `{report.get('residual_csv_path')}`",
        f"- Residual plot: `{report.get('residual_plot_path')}`",
        f"- Review SQLite: `{review.get('sqlite', '')}`",
        "",
        "## Station Comparison",
        "",
        "| Station | Disposition | Fit weighted RMS | Holdout weighted RMS | Rejected fraction |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for station in list(station_comparison.get("stations", []) or []):
        lines.append(
            "| {station_id} | {disposition} | {fit_value} | {holdout_value} | {rejected} |".format(
                station_id=station.get("station_id", ""),
                disposition=station.get("disposition", ""),
                fit_value=station.get("fit_weighted_rms"),
                holdout_value=station.get("holdout_weighted_rms"),
                rejected=station.get("rejected_residual_fraction"),
            )
        )
    lines.extend(
        [
            "",
            "## Systematic Parameters",
            "",
            "| Parameter | Estimate | Unit | Data identifiable | Prior dominated | Prior pull (sigma) |",
            "| --- | ---: | --- | --- | --- | ---: |",
        ]
    )
    for item in systematic_evidence:
        lines.append(
            "| {parameter} | {estimate} | {unit} | {identifiable} | {prior_dominated} | {pull} |".format(
                parameter=item.get("parameter", ""),
                estimate=item.get("estimate"),
                unit=item.get("unit", ""),
                identifiable=item.get("data_identifiable"),
                prior_dominated=item.get("prior_dominated"),
                pull=item.get("prior_pull_sigma"),
            )
        )
    if not systematic_evidence:
        lines.append("| _none estimated_ |  |  |  |  |  |")
    lines.extend(
        [
            "",
            "## Non-Claims",
            "",
        ]
    )
    for item in list(report.get("non_claims", []) or []):
        lines.append(f"- {item}")
    return "\n".join(lines) + "\n"


def _state_history_from_payload(payload: Mapping[str, Any], *, object_id: str) -> tuple[np.ndarray, np.ndarray]:
    sim_t = np.array(payload.get("time_s", []), dtype=float).reshape(-1)
    truth = dict(payload.get("truth_by_object", {}) or {})
    if object_id not in truth:
        raise ValueError(f"OEL payload did not include object_id '{object_id}'.")
    sim_x = np.array(truth[object_id], dtype=float)[:, :6]
    if sim_t.size == 0 or sim_x.shape[0] != sim_t.size:
        raise ValueError("OEL payload time/state history has an unexpected shape.")
    return sim_t, sim_x


def _states_at_epochs(t_src_s: np.ndarray, x_src: np.ndarray, t_query_s: np.ndarray) -> np.ndarray:
    t_src = np.array(t_src_s, dtype=float).reshape(-1)
    x = np.array(x_src, dtype=float)
    t_query = np.array(t_query_s, dtype=float).reshape(-1)
    if t_query.size == 0:
        return np.empty((0, x.shape[1]), dtype=float)
    indices = np.searchsorted(t_src, t_query)
    if np.any(indices >= t_src.size) or not np.allclose(
        t_src[indices],
        t_query,
        rtol=1.0e-12,
        atol=1.0e-10,
    ):
        raise ValueError("ground-station OD propagation did not produce every observation epoch exactly.")
    return x[indices]


def _default_dt_from_times(times: np.ndarray) -> float:
    diffs = np.diff(np.array(times, dtype=float).reshape(-1))
    positive = diffs[diffs > 1.0e-9]
    if positive.size == 0:
        return 10.0
    return float(max(min(float(np.min(positive)), 60.0), 1.0))


def _integer_multiple_duration(duration_s: float, dt_s: float) -> float:
    steps = max(int(np.ceil(float(duration_s) / float(dt_s) - 1.0e-12)), 1)
    return float(steps * float(dt_s))
