from __future__ import annotations

import json
import math
from collections import OrderedDict
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from sim.dynamics.orbit.ogp import (
    ogp_propagate_teme,
    ogp_propagator_name_for_elements,
    ogp_regime_for_elements,
)
from sim.dynamics.orbit.sgp4 import (
    sgp4_propagate_teme,
    sgp4_propagate_teme_batch_numba,
    sgp4_propagate_teme_batch_reference,
    sgp4_unsupported_reason,
    transform_teme_to_output_frame,
)
from sim.dynamics.orbit.tle import TLEElements
from sim.estimation.batch_least_squares import solve_batch_least_squares
from sim.estimation.ground_station_od import (
    _component_residual,
    _label_sensor_residual_decisions,
    _measurement_covariance,
    _normalize_measurements,
    _sensor_residual_metrics,
    _write_residual_csv,
    _write_residual_plot,
    predict_ground_station_measurement,
)
from sim.estimation.ogp_od import _parameter_set
from sim.estimation.parameters import EstimatedParameter, ParameterSet
from sim.estimation.partitioning import partition_time_arc
from sim.estimation.residual_audit import build_residual_audit
from sim.estimation.weighting import (
    covariance_from_sigmas,
    prepare_covariance_whitener,
    validate_covariance_block,
    whiten_residual_block,
    whiten_residual_with_factor,
)

DEFAULT_SGP4_SENSOR_FRAME_TRANSFORM = "teme_to_eci_iau80"
SUPPORTED_SGP4_SENSOR_FRAME_TRANSFORMS = ("teme_as_eci", DEFAULT_SGP4_SENSOR_FRAME_TRANSFORM)
_OD_PROPAGATION_CACHE_MAX_ENTRIES = 64
_OD_PROPAGATION_CACHE_MAX_BYTES = 32 * 1024 * 1024


def fit_sgp4_mean_elements_to_observations(
    *,
    times_jd_utc: np.ndarray,
    positions_teme_km: np.ndarray,
    velocities_teme_km_s: np.ndarray,
    template_elements: TLEElements,
    position_sigma_km: float = 0.001,
    velocity_sigma_km_s: float = 1.0e-6,
    covariance_blocks: Sequence[Sequence[Sequence[float]]] | np.ndarray | None = None,
    fit_until_jd_utc: float | None = None,
    max_nfev: int = 100,
    robust_loss: str = "linear",
    robust_f_scale: float = 1.0,
    sigma_clip_threshold: float | None = None,
    prior_mean_native: Sequence[float] | np.ndarray | None = None,
    prior_covariance_native: Sequence[Sequence[float]] | np.ndarray | None = None,
    propagation_backend: str = "scalar_reference",
) -> dict[str, Any]:
    """Fit six fixed-epoch near-Earth SGP4 mean elements to TEME PV observations."""

    backend = _normalize_sgp4_od_backend(propagation_backend)
    observed_backends: set[str] = set()
    fallback_reasons: list[dict[str, str]] = []
    propagation_cache: OrderedDict[tuple[TLEElements, bytes], tuple[np.ndarray, np.ndarray]] = OrderedDict()
    times_jd = np.array(times_jd_utc, dtype=float).reshape(-1)
    positions = np.array(positions_teme_km, dtype=float)
    velocities = np.array(velocities_teme_km_s, dtype=float)
    if times_jd.size < 2:
        raise ValueError("SGP4 OD requires at least two observations.")
    if positions.shape != (times_jd.size, 3):
        raise ValueError("positions_teme_km must have shape (observation_count, 3).")
    if velocities.shape != (times_jd.size, 3):
        raise ValueError("velocities_teme_km_s must have shape (observation_count, 3).")
    if not np.all(np.isfinite(times_jd)) or not np.all(np.isfinite(positions)) or not np.all(np.isfinite(velocities)):
        raise ValueError("SGP4 OD observations must be finite.")
    if np.any(np.diff(times_jd) < 0.0):
        raise ValueError("SGP4 OD observation epochs must be ordered.")
    if np.any(np.diff(times_jd) == 0.0):
        raise ValueError("SGP4 OD observation epochs must be unique.")
    if (
        not np.isfinite(float(position_sigma_km))
        or not np.isfinite(float(velocity_sigma_km_s))
        or float(position_sigma_km) <= 0.0
        or float(velocity_sigma_km_s) <= 0.0
    ):
        raise ValueError("SGP4 OD sigma values must be finite and positive.")
    covariance_array: np.ndarray | None = None
    if covariance_blocks is not None:
        covariance_array = np.asarray(covariance_blocks, dtype=float)
        if covariance_array.shape == (6, 6):
            covariance_array = np.broadcast_to(covariance_array, (times_jd.size, 6, 6)).copy()
        if covariance_array.shape != (times_jd.size, 6, 6):
            raise ValueError("covariance_blocks must have shape (6, 6) or (N, 6, 6).")
        for index in range(times_jd.size):
            covariance_array[index] = validate_covariance_block(
                covariance_array[index],
                dimension=6,
                field_name=f"SGP4 observation {index} covariance",
            )
    default_covariance = covariance_from_sigmas(
        [float(position_sigma_km)] * 3 + [float(velocity_sigma_km_s)] * 3
    )
    covariance_source = (
        np.broadcast_to(default_covariance, (times_jd.size, 6, 6))
        if covariance_array is None
        else covariance_array
    )
    covariance_factors = np.stack(
        [
            prepare_covariance_whitener(
                covariance_source[index],
                dimension=6,
                field_name=f"SGP4 observation {index} covariance",
            )
            for index in range(times_jd.size)
        ],
        axis=0,
    )

    unsupported = sgp4_unsupported_reason(template_elements)
    if unsupported:
        raise ValueError(unsupported)

    times_min = (times_jd - float(template_elements.epoch_jd_utc)) * 1440.0
    fit_mask = _make_fit_mask(times_jd, fit_until_jd_utc)
    fit_times_min = times_min[fit_mask]
    seed_vector = _element_vector(template_elements)
    parameters = _sgp4_mean_element_parameters(seed_vector)

    def residual(vector: np.ndarray) -> np.ndarray:
        elements = _elements_from_vector(template_elements, vector)
        try:
            pos_err_km, vel_err_km_s = _state_errors(
                elements,
                fit_times_min,
                positions[fit_mask],
                velocities[fit_mask],
                propagation_backend=backend,
                observed_backends=observed_backends,
                propagation_cache=propagation_cache,
                fallback_reasons=fallback_reasons,
            )
        except ValueError:
            return np.full(int(np.sum(fit_mask)) * 6, 1.0e9, dtype=float)
        return _whiten_sgp4_pv_blocks(
            pos_err_km,
            vel_err_km_s,
            covariance_blocks=None if covariance_array is None else covariance_array[fit_mask],
            covariance_factors=covariance_factors[fit_mask],
            position_sigma_km=float(position_sigma_km),
            velocity_sigma_km_s=float(velocity_sigma_km_s),
        )

    seed_pos_err, seed_vel_err = _state_errors(
        template_elements, times_min, positions, velocities,
        propagation_backend=backend, observed_backends=observed_backends,
        propagation_cache=propagation_cache,
        fallback_reasons=fallback_reasons,
    )
    initial_residual = residual(seed_vector)
    solve = solve_batch_least_squares(
        parameters,
        residual,
        max_nfev=int(max_nfev),
        xtol=1.0e-12,
        ftol=1.0e-12,
        gtol=1.0e-12,
        robust_loss=robust_loss,
        robust_f_scale=robust_f_scale,
        sigma_clip_threshold=sigma_clip_threshold,
        prior_mean_native=None if prior_mean_native is None else np.asarray(prior_mean_native, dtype=float),
        prior_covariance_native=None
        if prior_covariance_native is None
        else np.asarray(prior_covariance_native, dtype=float),
    )
    estimated_vector = np.array(solve.x_native, dtype=float)
    estimated_elements = _elements_from_vector(template_elements, estimated_vector)
    final_pos_err, final_vel_err = _state_errors(
        estimated_elements, times_min, positions, velocities,
        propagation_backend=backend, observed_backends=observed_backends,
        propagation_cache=propagation_cache,
        fallback_reasons=fallback_reasons,
    )
    initial_metrics = _state_error_metrics(seed_pos_err, seed_vel_err)
    final_metrics = _state_error_metrics(final_pos_err, final_vel_err)
    result: dict[str, Any] = {
        "method": "sgp4_mean_elements_least_squares",
        "reference_frame": "teme",
        "fit_parameters": parameters.names,
        "fixed_parameters": [
            "epoch_jd_utc",
            "mean_motion_derivative_rev_per_day2",
            "mean_motion_second_derivative_rev_per_day3",
            "bstar",
        ],
        "observation_count": int(times_jd.size),
        "fit_sample_count": int(np.sum(fit_mask)),
        "holdout_sample_count": int(np.sum(~fit_mask)),
        "observation_partition": {
            "schema_version": 1,
            "strategy": "absolute_epoch_boundary",
            "fit_boundary": "jd_utc <= fit_until_jd_utc",
            "holdout_boundary": "jd_utc > fit_until_jd_utc",
            "fit_observation_count": int(np.sum(fit_mask)),
            "holdout_observation_count": int(np.sum(~fit_mask)),
            "excluded_observation_count": 0,
            "holdout_status": "evaluated" if bool(np.any(~fit_mask)) else "not_evaluated",
        },
        "epoch_evaluation": {
            "schema_version": 1,
            "method": "sgp4_tsince_at_observation_epoch",
            "interpolation_used": False,
            "epoch_count": int(times_jd.size),
        },
        "time_span_min": float(np.max(times_min) - np.min(times_min)),
        "fit_time_span_min": float(np.max(fit_times_min) - np.min(fit_times_min)),
        "position_sigma_km": float(position_sigma_km),
        "velocity_sigma_km_s": float(velocity_sigma_km_s),
        "epoch_jd_utc": float(template_elements.epoch_jd_utc),
        "template_elements": _elements_payload(template_elements),
        "estimated_elements": _elements_payload(estimated_elements),
        "initial_observation_residual": initial_metrics,
        "final_observation_residual": final_metrics,
        "initial_normalized_residual_rms": float(np.sqrt(np.mean(initial_residual * initial_residual))),
        "final_normalized_residual_rms": float(solve.rms_residual),
        "solver": {
            "success": bool(solve.success),
            "message": solve.message,
            "nfev": int(solve.nfev),
            "initial_cost": float(solve.initial_cost),
            "final_cost": float(solve.cost),
            "diagnostics": solve.diagnostics,
            "decision_records": _label_sgp4_pv_decisions(
                solve.decision_records,
                times_jd_utc=times_jd[fit_mask],
            ),
        },
        "estimation_policy": {
            "robust_loss": robust_loss,
            "robust_f_scale": float(robust_f_scale),
            "sigma_clip_threshold": sigma_clip_threshold,
            "prior_enabled": prior_mean_native is not None,
            "covariance_blocks": covariance_array is not None,
        },
        "quality_gates": {
            "schema_version": 1,
            "evidence_status": "ready_with_caveats"
            if bool(solve.success) and bool(solve.diagnostics["observability"]["data_full_rank"])
            else "review_required",
            "solver_success": bool(solve.success),
            "data_full_rank": bool(solve.diagnostics["observability"]["data_full_rank"]),
            "failure_classification": solve.diagnostics["failure_classification"],
            "fit_improved_initial_rms": float(final_metrics["position_rms_m"])
            <= float(initial_metrics["position_rms_m"]),
            "holdout_evidence_status": "evaluated" if bool(np.any(~fit_mask)) else "not_evaluated",
            "warnings": (
                ([] if bool(solve.success) else ["SGP4 mean-element fit solver did not report success"])
                + (
                    []
                    if bool(solve.diagnostics["observability"]["data_full_rank"])
                    else ["SGP4 mean-element fit data Jacobian is rank deficient"]
                )
            ),
            "non_claims": [
                "This is fixed-epoch near-Earth SGP4 mean-element OD against native TEME position/velocity observations.",
                "Epoch, B*, and mean-motion derivative terms are inherited from the template and are not estimated.",
                "This is not covariance-derived OD, sensor-frame OD, catalog association, or operational tracking validation.",
            ],
        },
    }
    if backend == "accelerated":
        result["propagation_backend"] = {
            "requested": "accelerated",
            "observed": sorted(observed_backends),
            "reference_backend": "scalar_reference",
            "fallback_used": bool(fallback_reasons),
            "fallback_reasons": list(fallback_reasons),
            "equivalence": (
                "scalar_reference_exact"
                if "numba_cpu" not in observed_backends
                else "approved_rounding_level"
            ),
            "floating_point_parity": (
                "bitwise_reference"
                if "numba_cpu" not in observed_backends
                else "rounding_level_different"
            ),
        }
    final_fit_whitened = _whiten_sgp4_pv_blocks(
        final_pos_err[fit_mask],
        final_vel_err[fit_mask],
        covariance_blocks=None if covariance_array is None else covariance_array[fit_mask],
        covariance_factors=covariance_factors[fit_mask],
        position_sigma_km=float(position_sigma_km),
        velocity_sigma_km_s=float(velocity_sigma_km_s),
    )
    fit_whitened_rms = float(np.sqrt(np.mean(final_fit_whitened * final_fit_whitened)))
    result["quality_gates"]["fit_whitened_rms"] = fit_whitened_rms
    if bool(np.any(~fit_mask)):
        holdout_whitened = _whiten_sgp4_pv_blocks(
            final_pos_err[~fit_mask],
            final_vel_err[~fit_mask],
            covariance_blocks=None if covariance_array is None else covariance_array[~fit_mask],
            covariance_factors=covariance_factors[~fit_mask],
            position_sigma_km=float(position_sigma_km),
            velocity_sigma_km_s=float(velocity_sigma_km_s),
        )
        result["final_holdout_normalized_observation_residual_rms"] = float(
            np.sqrt(np.mean(holdout_whitened * holdout_whitened))
        )
        holdout_whitened_rms = float(result["final_holdout_normalized_observation_residual_rms"])
        result["quality_gates"]["holdout_whitened_rms"] = holdout_whitened_rms
        result["quality_gates"]["holdout_acceptable"] = bool(
            np.isfinite(holdout_whitened_rms)
            and holdout_whitened_rms <= max(10.0, 5.0 * max(fit_whitened_rms, 1.0e-12))
        )
        if not result["quality_gates"]["holdout_acceptable"]:
            result["quality_gates"]["warnings"].append("SGP4 mean-element holdout residual is too large")
        result["final_holdout_observation_residual"] = _state_error_metrics(
            final_pos_err[~fit_mask],
            final_vel_err[~fit_mask],
        )
    else:
        result["final_holdout_normalized_observation_residual_rms"] = None
        result["final_holdout_observation_residual"] = None
        result["quality_gates"]["holdout_whitened_rms"] = None
        result["quality_gates"]["holdout_acceptable"] = False
        result["quality_gates"]["warnings"].append("SGP4 mean-element holdout was not evaluated")
    if result["quality_gates"]["warnings"]:
        result["quality_gates"]["evidence_status"] = "review_required"
    result["residual_records"] = _sgp4_pv_residual_records(
        times_jd,
        positions,
        velocities,
        final_pos_err,
        final_vel_err,
        fit_mask,
        position_sigma_km=float(position_sigma_km),
        velocity_sigma_km_s=float(velocity_sigma_km_s),
        covariance_blocks=covariance_array,
    )
    result["residual_audit"] = build_residual_audit(
        result["residual_records"],
        decision_records=result["solver"]["decision_records"],
    )
    return result


def fit_sgp4_mean_elements_to_ground_station_measurements(
    measurements: Sequence[Mapping[str, Any]],
    *,
    template_elements: TLEElements,
    object_id: str,
    output_dir: str | Path,
    fit_duration_s: float | None = None,
    holdout_duration_s: float | None = None,
    frame_transform: str = DEFAULT_SGP4_SENSOR_FRAME_TRANSFORM,
    max_nfev: int = 100,
    robust_loss: str = "linear",
    robust_f_scale: float = 1.0,
    sigma_clip_threshold: float | None = None,
    prior_mean_native: Sequence[float] | np.ndarray | None = None,
    prior_covariance_native: Sequence[Sequence[float]] | np.ndarray | None = None,
) -> dict[str, Any]:
    """Fit near-Earth SGP4 observations through the shared owner."""
    return _fit_mean_elements_to_ground_station_measurements(
        measurements,
        template_elements=template_elements,
        object_id=object_id,
        output_dir=output_dir,
        fit_duration_s=fit_duration_s,
        holdout_duration_s=holdout_duration_s,
        frame_transform=frame_transform,
        max_nfev=max_nfev,
        robust_loss=robust_loss,
        robust_f_scale=robust_f_scale,
        sigma_clip_threshold=sigma_clip_threshold,
        prior_mean_native=prior_mean_native,
        prior_covariance_native=prior_covariance_native,
        ogp_dispatch=False,
    )


def fit_ogp_mean_elements_to_ground_station_measurements(
    measurements: Sequence[Mapping[str, Any]],
    *,
    template_elements: TLEElements,
    object_id: str,
    output_dir: str | Path,
    fit_duration_s: float | None = None,
    holdout_duration_s: float | None = None,
    frame_transform: str = DEFAULT_SGP4_SENSOR_FRAME_TRANSFORM,
    max_nfev: int = 100,
    robust_loss: str = "linear",
    robust_f_scale: float = 1.0,
    sigma_clip_threshold: float | None = None,
    prior_mean_native: Sequence[float] | np.ndarray | None = None,
    prior_covariance_native: Sequence[Sequence[float]] | np.ndarray | None = None,
) -> dict[str, Any]:
    """Fit regime-dispatched OGP observations through the shared owner."""
    return _fit_mean_elements_to_ground_station_measurements(
        measurements,
        template_elements=template_elements,
        object_id=object_id,
        output_dir=output_dir,
        fit_duration_s=fit_duration_s,
        holdout_duration_s=holdout_duration_s,
        frame_transform=frame_transform,
        max_nfev=max_nfev,
        robust_loss=robust_loss,
        robust_f_scale=robust_f_scale,
        sigma_clip_threshold=sigma_clip_threshold,
        prior_mean_native=prior_mean_native,
        prior_covariance_native=prior_covariance_native,
        ogp_dispatch=True,
    )


def _fit_mean_elements_to_ground_station_measurements(
    measurements: Sequence[Mapping[str, Any]],
    *,
    template_elements: TLEElements,
    object_id: str,
    output_dir: str | Path,
    fit_duration_s: float | None = None,
    holdout_duration_s: float | None = None,
    frame_transform: str = DEFAULT_SGP4_SENSOR_FRAME_TRANSFORM,
    max_nfev: int = 100,
    robust_loss: str = "linear",
    robust_f_scale: float = 1.0,
    sigma_clip_threshold: float | None = None,
    prior_mean_native: Sequence[float] | np.ndarray | None = None,
    prior_covariance_native: Sequence[Sequence[float]] | np.ndarray | None = None,
    ogp_dispatch: bool,
) -> dict[str, Any]:
    """Fit fixed-epoch SGP4 mean elements directly to ground-station sensor rows."""

    frame_transform = str(frame_transform or DEFAULT_SGP4_SENSOR_FRAME_TRANSFORM).strip().lower()
    if frame_transform not in SUPPORTED_SGP4_SENSOR_FRAME_TRANSFORMS:
        raise ValueError(
            "SGP4 ground-sensor OD frame_transform must be one of: "
            + ", ".join(SUPPORTED_SGP4_SENSOR_FRAME_TRANSFORMS)
            + "."
        )

    regime = ogp_regime_for_elements(template_elements)
    prefix = "ogp" if ogp_dispatch else "sgp4"
    if not ogp_dispatch:
        unsupported = sgp4_unsupported_reason(template_elements)
        if unsupported:
            raise ValueError(unsupported)
    # Validate the seed through the same propagator that evaluates measurements.
    seed_state = ogp_propagate_teme(template_elements, 0.0)
    if seed_state.error:
        raise ValueError(seed_state.error)
    rows = _normalize_measurements(measurements)
    if len(rows) < 2:
        raise ValueError("SGP4 ground-station sensor OD requires at least two measurement rows.")
    output_root = Path(output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    times_s = np.array([float(row["time_s"]) for row in rows], dtype=float)
    partition = partition_time_arc(
        times_s,
        fit_duration_s=fit_duration_s,
        holdout_duration_s=holdout_duration_s,
        allow_repeated_epochs=True,
    )
    fit_duration = partition.fit_duration_s
    holdout_duration = partition.holdout_duration_s
    fit_mask = partition.fit_mask
    holdout_mask = partition.holdout_mask

    seed_vector = _element_vector(template_elements)
    if ogp_dispatch:
        # Resolve the regime-specific mean-motion bounds before constructing
        # the absolute-element compatibility parameters.  SDP4 seeds may
        # validly fall below the near-Earth 225-minute boundary used by the
        # legacy SGP4 helper.
        bounds = _parameter_set(template_elements, parameterization="classical_delta", active_bstar=False, active_epoch=False)
        motion = next(item for item in bounds.parameters if item.name == "mean_motion_rev_per_day")
        parameters = _sgp4_mean_element_parameters(
            seed_vector,
            mean_motion_bounds=(float(motion.lower), float(motion.upper)),
        )
        parameters = ParameterSet([*parameters.parameters[:5], motion])
    else:
        parameters = _sgp4_mean_element_parameters(seed_vector)
    fit_rows = [row for idx, row in enumerate(rows) if bool(fit_mask[idx])]
    holdout_rows = [row for idx, row in enumerate(rows) if bool(holdout_mask[idx])]
    fit_covariance_factors = tuple(
        prepare_covariance_whitener(
            _measurement_covariance(row),
            dimension=len(row["components"]),
            field_name=f"measurement {row['measurement_id']!r} covariance",
        )
        for row in fit_rows
    )

    def residual(vector: np.ndarray) -> np.ndarray:
        elements = _elements_from_vector(template_elements, vector)
        try:
            if ogp_regime_for_elements(elements) != regime:
                raise ValueError("OGP sensor OD candidate crossed its seed regime.")
            return _sgp4_sensor_residual_vector(
                elements,
                fit_rows,
                frame_transform=frame_transform,
                covariance_factors=fit_covariance_factors,
            )
        except ValueError:
            return np.full(_component_count(fit_rows), 1.0e9, dtype=float)

    prefit_residuals = _sgp4_sensor_residual_rows(
        template_elements,
        fit_rows,
        frame_transform=frame_transform,
        residual_kind="prefit",
    )
    solve = solve_batch_least_squares(
        parameters,
        residual,
        max_nfev=int(max_nfev),
        xtol=1.0e-12,
        ftol=1.0e-12,
        gtol=1.0e-12,
        robust_loss=robust_loss,
        robust_f_scale=robust_f_scale,
        sigma_clip_threshold=sigma_clip_threshold,
        prior_mean_native=None if prior_mean_native is None else np.asarray(prior_mean_native, dtype=float),
        prior_covariance_native=None
        if prior_covariance_native is None
        else np.asarray(prior_covariance_native, dtype=float),
    )
    estimated_elements = _elements_from_vector(template_elements, np.array(solve.x_native, dtype=float))
    fit_residuals = _sgp4_sensor_residual_rows(
        estimated_elements,
        fit_rows,
        frame_transform=frame_transform,
        residual_kind="fit",
    )
    holdout_residuals = _sgp4_sensor_residual_rows(
        estimated_elements,
        holdout_rows,
        frame_transform=frame_transform,
        residual_kind="holdout",
    )

    residual_csv_path = output_root / f"{prefix}_ground_sensor_od_residuals.csv"
    residual_plot_path = output_root / f"{prefix}_ground_sensor_od_normalized_residuals.png"
    product_path = output_root / f"fitted_{prefix}_od_product.json"
    report_json_path = output_root / f"{prefix}_ground_sensor_od_report.json"
    report_md_path = output_root / f"{prefix}_ground_sensor_od_report.md"
    all_residuals = [*prefit_residuals, *fit_residuals, *holdout_residuals]
    _write_residual_csv(residual_csv_path, all_residuals)
    _write_residual_plot(residual_plot_path, all_residuals)

    product = {
        "kind": f"oel.{prefix}_ground_sensor_od_product",
        "object_id": object_id,
        "method": f"{prefix}_ground_station_sensor_least_squares",
        "ogp_regime": regime,
        "propagator_name": ogp_propagator_name_for_elements(template_elements),
        "reference_frame": "teme",
        "measurement_frame_contract": {
            "output_frame": "eci",
            "frame_transform": frame_transform,
        },
        "template_elements": _elements_payload(template_elements),
        "estimated_elements": _elements_payload(estimated_elements),
    }
    product_path.write_text(json.dumps(product, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    decision_records = _label_sensor_residual_decisions(solve.decision_records, fit_rows)
    report = {
        "method": f"{prefix}_ground_station_sensor_least_squares",
        "object_id": object_id,
        "ogp_regime": regime,
        "propagator_name": ogp_propagator_name_for_elements(template_elements),
        "reference_frame": "teme",
        "measurement_frame_contract": product["measurement_frame_contract"],
        "fit_parameters": parameters.names,
        "fixed_parameters": [
            "epoch_jd_utc",
            "mean_motion_derivative_rev_per_day2",
            "mean_motion_second_derivative_rev_per_day3",
            "bstar",
        ],
        "observation_count": len(rows),
        "fit_observation_count": int(np.count_nonzero(fit_mask)),
        "holdout_observation_count": int(np.count_nonzero(holdout_mask)),
        "fit_duration_s": fit_duration,
        "holdout_duration_s": holdout_duration,
        "observation_partition": partition.summary,
        "epoch_evaluation": {
            "schema_version": 1,
            "method": f"{regime}_tsince_at_measurement_epoch",
            "interpolation_used": False,
            "epoch_count": int(np.count_nonzero(fit_mask | holdout_mask)),
        },
        "epoch_jd_utc": float(template_elements.epoch_jd_utc),
        "template_elements": _elements_payload(template_elements),
        "estimated_elements": _elements_payload(estimated_elements),
        "estimated_parameters": parameters.metadata(solve.x_native),
        "parameter_names": list(solve.parameter_names),
        "parameter_units": list(parameters.units),
        "covariance_native": None if solve.covariance_native is None else solve.covariance_native.tolist(),
        "data_covariance_native": None if solve.data_covariance_native is None else solve.data_covariance_native.tolist(),
        "covariance_evidence": solve.covariance_evidence,
        "empirically_calibrated": False,
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
        "estimation_policy": {
            "robust_loss": robust_loss,
            "robust_f_scale": float(robust_f_scale),
            "sigma_clip_threshold": sigma_clip_threshold,
            "prior_enabled": prior_mean_native is not None,
        },
        "prefit_metrics": _sensor_residual_metrics(prefit_residuals),
        "fit_metrics": _sensor_residual_metrics(fit_residuals),
        "holdout_metrics": _sensor_residual_metrics(holdout_residuals),
        "residual_audit": build_residual_audit(
            all_residuals,
            decision_records=decision_records,
        ),
        "residual_csv_path": str(residual_csv_path),
        "residual_plot_path": str(residual_plot_path),
        f"fitted_{prefix}_od_product_path": str(product_path),
        "report_json_path": str(report_json_path),
        "report_md_path": str(report_md_path),
        "non_claims": [
            f"This is fixed-epoch {ogp_propagator_name_for_elements(template_elements)} mean-element OD against geometric ground-station measurements.",
            "Epoch, B*, and mean-motion derivative terms are inherited from the template and are not estimated.",
            "This is not covariance-derived OD, catalog association, calibrated sensor processing, or operational tracking validation.",
        ],
    }
    report["quality_gates"] = _sgp4_sensor_quality_gates(report)
    report["verdict"] = _sgp4_sensor_verdict(report["quality_gates"])
    if ogp_dispatch:
        report["verdict"]["analyst_action"] = (
            "usable_for_ogp_catalog_study"
            if report["verdict"]["evidence_status"] == "ready_with_caveats"
            else "review_ogp_sensor_od_report"
        )
    report_json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_md_path.write_text(_render_sgp4_sensor_report_md(report), encoding="utf-8")
    return report


def _sgp4_mean_element_parameters(
    seed_vector: np.ndarray,
    *,
    mean_motion_bounds: tuple[float, float] | None = None,
) -> ParameterSet:
    motion_lower, motion_upper = (
        (1440.0 / 225.0, 18.0) if mean_motion_bounds is None else mean_motion_bounds
    )
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
                lower=motion_lower,
                upper=motion_upper,
            ),
        ]
    )


def _elements_from_vector(template: TLEElements, vector: np.ndarray) -> TLEElements:
    x = np.array(vector, dtype=float).reshape(6)
    return replace(
        template,
        inclination_deg=float(x[0]),
        raan_deg=float(x[1] % 360.0),
        eccentricity=float(np.clip(x[2], 1.0e-7, 0.999)),
        argp_deg=float(x[3] % 360.0),
        mean_anomaly_deg=float(x[4] % 360.0),
        mean_motion_rev_per_day=float(x[5]),
    )


def _element_vector(elements: TLEElements) -> np.ndarray:
    return np.array(
        [
            float(elements.inclination_deg),
            float(elements.raan_deg),
            float(elements.eccentricity),
            float(elements.argp_deg),
            float(elements.mean_anomaly_deg),
            float(elements.mean_motion_rev_per_day),
        ],
        dtype=float,
    )


def _state_errors(
    elements: TLEElements,
    times_min: np.ndarray,
    obs_pos_km: np.ndarray,
    obs_vel_km_s: np.ndarray,
    *,
    propagation_backend: str = "scalar_reference",
    observed_backends: set[str] | None = None,
    propagation_cache: OrderedDict[tuple[TLEElements, bytes], tuple[np.ndarray, np.ndarray]] | None = None,
    fallback_reasons: list[dict[str, str]] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    times = np.asarray(times_min, dtype=float)
    cache_key = (elements, times.tobytes())
    cached = None if propagation_cache is None else propagation_cache.get(cache_key)
    if cached is not None:
        propagation_cache.move_to_end(cache_key)
        propagated_position, propagated_velocity = cached
        return propagated_position - obs_pos_km, propagated_velocity - obs_vel_km_s
    if propagation_backend == "accelerated":
        try:
            batch = sgp4_propagate_teme_batch_numba([elements], times)
        except Exception as exc:
            batch = sgp4_propagate_teme_batch_reference([elements], times)
            if fallback_reasons is not None:
                _record_backend_fallback(
                    fallback_reasons,
                    exception_type=type(exc).__name__,
                    reason=str(exc),
                )
        if observed_backends is not None:
            observed_backends.add(str(batch.backend))
        if not np.all(batch.success):
            error = next(str(item) for item in batch.errors.reshape(-1) if str(item))
            raise ValueError(error)
        propagated_position = batch.position_teme_km[0]
        propagated_velocity = batch.velocity_teme_km_s[0]
        if propagation_cache is not None:
            _store_bounded_propagation_cache(
                propagation_cache, cache_key, (propagated_position, propagated_velocity)
            )
        return propagated_position - obs_pos_km, propagated_velocity - obs_vel_km_s
    if observed_backends is not None:
        observed_backends.add("scalar_python")
    pos_err = []
    vel_err = []
    propagated_positions = []
    propagated_velocities = []
    for t_min, obs_pos, obs_vel in zip(times_min, obs_pos_km, obs_vel_km_s, strict=True):
        state = sgp4_propagate_teme(elements, float(t_min))
        if state.error:
            raise ValueError(state.error)
        propagated_position = np.array(state.position_teme_km, dtype=float)
        propagated_velocity = np.array(state.velocity_teme_km_s, dtype=float)
        propagated_positions.append(propagated_position)
        propagated_velocities.append(propagated_velocity)
        pos_err.append(propagated_position - obs_pos)
        vel_err.append(propagated_velocity - obs_vel)
    position_error = np.vstack(pos_err)
    velocity_error = np.vstack(vel_err)
    if propagation_cache is not None:
        _store_bounded_propagation_cache(
            propagation_cache,
            cache_key,
            (np.vstack(propagated_positions), np.vstack(propagated_velocities)),
        )
    return position_error, velocity_error


def _normalize_sgp4_od_backend(value: str) -> str:
    backend = str(value or "scalar_reference").strip().lower()
    if backend not in {"scalar_reference", "accelerated"}:
        raise ValueError("propagation_backend must be scalar_reference or accelerated.")
    return backend


def _record_backend_fallback(
    records: list[dict[str, str]], *, exception_type: str, reason: str
) -> None:
    record = {"exception_type": str(exception_type), "reason": str(reason)}
    if record not in records:
        records.append(record)


def _store_bounded_propagation_cache(
    cache: OrderedDict[tuple[TLEElements, bytes], tuple[np.ndarray, np.ndarray]],
    key: tuple[TLEElements, bytes],
    value: tuple[np.ndarray, np.ndarray],
    *,
    max_entries: int = _OD_PROPAGATION_CACHE_MAX_ENTRIES,
    max_bytes: int = _OD_PROPAGATION_CACHE_MAX_BYTES,
) -> None:
    cache[key] = value
    cache.move_to_end(key)
    total_bytes = sum(int(position.nbytes + velocity.nbytes) for position, velocity in cache.values())
    while cache and (len(cache) > int(max_entries) or total_bytes > int(max_bytes)):
        _old_key, (old_position, old_velocity) = cache.popitem(last=False)
        total_bytes -= int(old_position.nbytes + old_velocity.nbytes)


def _whiten_sgp4_pv_blocks(
    pos_err_km: np.ndarray,
    vel_err_km_s: np.ndarray,
    *,
    covariance_blocks: np.ndarray | None,
    covariance_factors: np.ndarray | None = None,
    position_sigma_km: float,
    velocity_sigma_km_s: float,
) -> np.ndarray:
    position_errors = np.asarray(pos_err_km, dtype=float)
    velocity_errors = np.asarray(vel_err_km_s, dtype=float)
    if position_errors.shape != velocity_errors.shape or position_errors.ndim != 2 or position_errors.shape[1] != 3:
        raise ValueError("SGP4 position and velocity errors must both have shape (N, 3).")
    default_covariance = covariance_from_sigmas([position_sigma_km] * 3 + [velocity_sigma_km_s] * 3)
    blocks = []
    for index in range(position_errors.shape[0]):
        if covariance_factors is not None:
            blocks.append(
                whiten_residual_with_factor(
                    np.hstack((position_errors[index], velocity_errors[index])),
                    covariance_factors[index],
                    field_name=f"SGP4 observation {index} covariance",
                )
            )
        else:
            covariance = default_covariance if covariance_blocks is None else covariance_blocks[index]
            blocks.append(
                whiten_residual_block(
                np.hstack((position_errors[index], velocity_errors[index])),
                covariance,
                field_name=f"SGP4 observation {index} covariance",
                )
            )
    return np.concatenate(blocks) if blocks else np.array([], dtype=float)


def _label_sgp4_pv_decisions(
    decisions: Sequence[Mapping[str, Any]],
    *,
    times_jd_utc: np.ndarray,
) -> list[dict[str, Any]]:
    component_names = ("x_km", "y_km", "z_km", "vx_km_s", "vy_km_s", "vz_km_s")
    labeled = []
    for decision in decisions:
        record = dict(decision)
        residual_index = int(record["residual_index"])
        observation_index, component_index = divmod(residual_index, 6)
        record.update(
            {
                "observation_id": f"sgp4_pv:observation:{observation_index:06d}",
                "observation_index": observation_index,
                "jd_utc": float(times_jd_utc[observation_index]),
                "component": component_names[component_index],
                "partition": "fit",
                "residual_space": "cholesky_whitened_component",
            }
        )
        labeled.append(record)
    return labeled


def _state_error_metrics(pos_err_km: np.ndarray, vel_err_km_s: np.ndarray) -> dict[str, float]:
    pos_norm_m = np.linalg.norm(np.array(pos_err_km, dtype=float), axis=1) * 1000.0
    vel_norm_mm_s = np.linalg.norm(np.array(vel_err_km_s, dtype=float), axis=1) * 1.0e6
    return {
        "position_rms_m": float(np.sqrt(np.mean(pos_norm_m * pos_norm_m))),
        "position_max_m": float(np.max(pos_norm_m)),
        "velocity_rms_mm_s": float(np.sqrt(np.mean(vel_norm_mm_s * vel_norm_mm_s))),
        "velocity_max_mm_s": float(np.max(vel_norm_mm_s)),
    }


def _sgp4_pv_residual_records(
    times_jd_utc: np.ndarray,
    positions_teme_km: np.ndarray,
    velocities_teme_km_s: np.ndarray,
    position_errors_km: np.ndarray,
    velocity_errors_km_s: np.ndarray,
    fit_mask: np.ndarray,
    *,
    position_sigma_km: float,
    velocity_sigma_km_s: float,
    covariance_blocks: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    components = (
        ("x_km", positions_teme_km, position_errors_km, position_sigma_km, 0),
        ("y_km", positions_teme_km, position_errors_km, position_sigma_km, 1),
        ("z_km", positions_teme_km, position_errors_km, position_sigma_km, 2),
        ("vx_km_s", velocities_teme_km_s, velocity_errors_km_s, velocity_sigma_km_s, 0),
        ("vy_km_s", velocities_teme_km_s, velocity_errors_km_s, velocity_sigma_km_s, 1),
        ("vz_km_s", velocities_teme_km_s, velocity_errors_km_s, velocity_sigma_km_s, 2),
    )
    for observation_index, jd_utc in enumerate(times_jd_utc):
        partition = "fit" if bool(fit_mask[observation_index]) else "holdout"
        covariance = (
            covariance_from_sigmas([position_sigma_km] * 3 + [velocity_sigma_km_s] * 3)
            if covariance_blocks is None
            else covariance_blocks[observation_index]
        )
        whitened = whiten_residual_block(
            np.hstack((position_errors_km[observation_index], velocity_errors_km_s[observation_index])),
            covariance,
            field_name=f"SGP4 observation {observation_index} covariance",
        )
        for component_index, (component, observed_values, errors, sigma, axis) in enumerate(components):
            observed = float(observed_values[observation_index, axis])
            residual = float(errors[observation_index, axis])
            records.append(
                {
                    "observation_id": f"sgp4_pv:observation:{observation_index:06d}",
                    "measurement_type": "teme_position_velocity",
                    "arc_id": "sgp4_pv_od",
                    "partition": partition,
                    "residual_kind": "postfit" if partition == "fit" else "prediction",
                    "frame": "TEME",
                    "epoch_evaluation_method": "sgp4_tsince_at_observation_epoch",
                    "jd_utc": float(jd_utc),
                    "component": component,
                    "observed": observed,
                    "predicted": observed + residual,
                    "residual": residual,
                    "sigma": float(sigma),
                    "normalized_residual": residual / float(sigma),
                    "whitened_residual": float(whitened[component_index]),
                    "covariance_source": "full_block" if covariance_blocks is not None else "diagonal_sigma",
                }
            )
    return records


def _make_fit_mask(times_jd_utc: np.ndarray, fit_until_jd_utc: float | None) -> np.ndarray:
    if fit_until_jd_utc is None:
        return np.ones(times_jd_utc.shape, dtype=bool)
    fit_until = float(fit_until_jd_utc)
    if not math.isfinite(fit_until):
        raise ValueError("fit_until_jd_utc must be finite when provided.")
    mask = np.array(times_jd_utc, dtype=float) <= fit_until + 1.0e-12
    if int(np.sum(mask)) < 2:
        raise ValueError("SGP4 OD requires at least two fit samples.")
    return mask


def _elements_payload(elements: TLEElements) -> dict[str, Any]:
    return asdict(elements)


def _component_count(rows: Sequence[Mapping[str, Any]]) -> int:
    return sum(len(list(row.get("components", []) or [])) for row in rows)


def _sgp4_sensor_residual_vector(
    elements: TLEElements,
    rows: Sequence[Mapping[str, Any]],
    *,
    frame_transform: str,
    covariance_factors: Sequence[np.ndarray] | None = None,
) -> np.ndarray:
    if not rows:
        return np.zeros(0, dtype=float)
    epoch_jd_utc = float(rows[0]["time_jd_utc"]) - float(rows[0]["time_s"]) / 86400.0
    chunks: list[np.ndarray] = []
    for index, row in enumerate(rows):
        state = _sgp4_state_for_measurement(elements, row, frame_transform=frame_transform)
        pred = predict_ground_station_measurement(
            target_state_eci=state,
            station=dict(row["station"]),
            t_s=float(row["time_s"]),
            jd_utc_start=epoch_jd_utc,
        )
        residual = np.array(
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
        if covariance_factors is None:
            chunks.append(
                whiten_residual_block(
                    residual,
                    _measurement_covariance(row),
                    field_name=f"measurement {row['measurement_id']!r} covariance",
                )
            )
        else:
            chunks.append(
                whiten_residual_with_factor(
                    residual,
                    covariance_factors[index],
                    field_name=f"measurement {row['measurement_id']!r} covariance",
                )
            )
    return np.concatenate(chunks)


def _sgp4_sensor_residual_rows(
    elements: TLEElements,
    rows: Sequence[Mapping[str, Any]],
    *,
    frame_transform: str,
    residual_kind: str,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not rows:
        return out
    epoch_jd_utc = float(rows[0]["time_jd_utc"]) - float(rows[0]["time_s"]) / 86400.0
    for row in rows:
        state = _sgp4_state_for_measurement(elements, row, frame_transform=frame_transform)
        pred = predict_ground_station_measurement(
            target_state_eci=state,
            station=dict(row["station"]),
            t_s=float(row["time_s"]),
            jd_utc_start=epoch_jd_utc,
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
        whitened = whiten_residual_block(
            residual_vector,
            _measurement_covariance(row),
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
                    "station_id": str(row["station_id"]),
                    "frame": "sensor_topocentric_from_eci",
                    "epoch_evaluation_method": f"{ogp_regime_for_elements(elements)}_tsince_at_measurement_epoch",
                    "time_s": float(row["time_s"]),
                    "component": str(component),
                    "observed": observed,
                    "predicted": predicted,
                    "residual": residual,
                    "sigma": sigma,
                    "normalized_residual": residual / sigma,
                    "whitened_residual": float(whitened[component_index]),
                    "covariance_source": str(dict(row.get("uncertainty", {}) or {}).get("source", "")),
                }
            )
    return out


def _sgp4_state_for_measurement(
    elements: TLEElements,
    row: Mapping[str, Any],
    *,
    frame_transform: str,
) -> np.ndarray:
    jd = float(row["time_jd_utc"])
    tsince_min = (jd - float(elements.epoch_jd_utc)) * 1440.0
    state = ogp_propagate_teme(elements, tsince_min)
    if state.error:
        raise ValueError(state.error)
    pos_eci, vel_eci = transform_teme_to_output_frame(
        state.position_teme_km,
        state.velocity_teme_km_s,
        jd_utc=jd,
        output_frame="eci",
        frame_transform=frame_transform,
    )
    return np.hstack((pos_eci, vel_eci))


def _sgp4_sensor_quality_gates(report: Mapping[str, Any]) -> dict[str, Any]:
    solver = dict(report.get("solver", {}) or {})
    prefit = dict(report.get("prefit_metrics", {}) or {})
    fit = dict(report.get("fit_metrics", {}) or {})
    holdout = dict(report.get("holdout_metrics", {}) or {})
    prefit_rms = float(prefit.get("weighted_rms", np.nan))
    fit_rms = float(fit.get("weighted_rms", np.nan))
    holdout_rms = float(holdout.get("weighted_rms", np.nan))
    warnings = []
    diagnostics = dict(solver.get("diagnostics", {}) or {})
    observability = dict(diagnostics.get("observability", {}) or {})
    data_full_rank = bool(observability.get("data_full_rank", False))
    holdout_evaluated = int(holdout.get("residual_count", 0) or 0) > 0
    if not bool(solver.get("success", False)):
        warnings.append("solver_not_successful")
    if not data_full_rank:
        warnings.append("data_jacobian_rank_deficient")
    if not (np.isfinite(prefit_rms) and np.isfinite(fit_rms) and fit_rms < prefit_rms):
        if not (np.isfinite(fit_rms) and fit_rms < 1.0e-2):
            warnings.append("fit_did_not_improve_prefit_residual")
    if not holdout_evaluated:
        warnings.append("holdout_not_evaluated")
    elif np.isfinite(holdout_rms) and holdout_rms > max(10.0, 5.0 * max(fit_rms, 1.0e-12)):
        warnings.append("holdout_weighted_residual_large")
    return {
        "schema_version": 1,
        "solver_success": bool(solver.get("success", False)),
        "data_full_rank": data_full_rank,
        "failure_classification": diagnostics.get("failure_classification"),
        "fit_improved_prefit_rms": bool(np.isfinite(prefit_rms) and np.isfinite(fit_rms) and fit_rms < prefit_rms),
        "prefit_weighted_rms": prefit_rms,
        "fit_weighted_rms": fit_rms,
        "holdout_weighted_rms": holdout_rms,
        "holdout_evidence_status": "evaluated" if holdout_evaluated else "not_evaluated",
        "warnings": warnings,
    }


def _sgp4_sensor_verdict(quality_gates: Mapping[str, Any]) -> dict[str, Any]:
    warnings = list(quality_gates.get("warnings", []) or [])
    usable = bool(quality_gates.get("solver_success")) and not warnings
    return {
        "evidence_status": "ready_with_caveats" if usable else "review_required",
        "analyst_action": "review_sgp4_sensor_od_report" if not usable else "usable_for_sgp4_catalog_study",
        "warning_count": len(warnings),
    }


def _render_sgp4_sensor_report_md(report: Mapping[str, Any]) -> str:
    fit = dict(report.get("fit_metrics", {}) or {})
    holdout = dict(report.get("holdout_metrics", {}) or {})
    solver = dict(report.get("solver", {}) or {})
    lines = [
        f"# {report.get('propagator_name', 'SGP4')} Ground-Station Sensor OD Report",
        "",
        f"- Object ID: `{report.get('object_id')}`",
        f"- Method: `{report.get('method')}`",
        f"- Frame transform: `{dict(report.get('measurement_frame_contract', {}) or {}).get('frame_transform')}`",
        f"- Solver success: `{solver.get('success')}`",
        f"- Solver evaluations: `{solver.get('nfev')}`",
        f"- Fit weighted RMS: `{fit.get('weighted_rms')}`",
        f"- Holdout weighted RMS: `{holdout.get('weighted_rms')}`",
        "",
        "## Non-Claims",
        "",
    ]
    lines.extend(f"- {item}" for item in list(report.get("non_claims", []) or []))
    return "\n".join(lines) + "\n"
