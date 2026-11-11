from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from sim.dynamics.orbit.elements import rv_to_coe_eci
from sim.dynamics.orbit.environment import EARTH_J2, EARTH_MU_KM3_S2, EARTH_RADIUS_KM
from sim.dynamics.orbit.relative_linear import RelativeLinearDynamics
from sim.estimation.batch_least_squares import BatchLeastSquaresResult, solve_batch_least_squares
from sim.estimation.parameters import EstimatedParameter, ParameterSet
from sim.estimation.partitioning import partition_time_arc
from sim.estimation.relative_hcw_ekf import (
    _measurement_innovation,
    _normalize_measurement_origin,
    hcw_measurement_dimension,
    hcw_measurement_vector,
    normalize_hcw_measurement_model,
)
from sim.estimation.relative_th_ekf import (
    _propagate_chief_state,
    th_propagate_relative_state,
    th_variational_transition_matrix,
    ya_closed_form_propagate_relative_state_and_stm,
    ya_closed_form_transition_matrix,
)
from sim.estimation.residual_audit import build_residual_audit, residual_records_from_vectors
from sim.estimation.weighting import (
    covariance_from_sigmas,
    prepare_covariance_whitener,
    validate_covariance_block,
    whiten_residual_block,
    whiten_residual_with_factor,
)
from sim.security.output_writes import mkdir_output, staged_output_file, write_output_text
from sim.utils.io import write_json

RELATIVE_OD_DYNAMICS = {"hcw", "ss_j2", "th", "ya"}
RELATIVE_OD_STATE_LABELS = ("R_km", "I_km", "C_km", "Rdot_km_s", "Idot_km_s", "Cdot_km_s")
_SEQUENTIAL_VALIDATED_SUBSTEP_S = 5.0
_SEQUENTIAL_VALIDATED_OBSERVATION_SPACING_S = 120.0
_SEQUENTIAL_VALIDATED_MAX_HORIZON_S = 1440.0
_SEQUENTIAL_VALIDATED_MIN_SEMIMAJOR_AXIS_KM = 7000.0
_SEQUENTIAL_VALIDATED_MAX_SEMIMAJOR_AXIS_KM = 50000.0
_SEQUENTIAL_VALIDATED_MAX_ECCENTRICITY = 0.25
_SEQUENTIAL_VALIDATED_MIN_PERIGEE_KM = 6500.0
_SEQUENTIAL_VALIDATED_MAX_STM_ABS_DELTA = 1.0e-7


@dataclass(frozen=True)
class RelativeODObservation:
    time_s: float
    vector: np.ndarray
    noise_diag: np.ndarray
    covariance: np.ndarray | None = None


@dataclass(frozen=True)
class RelativeODArc:
    observations: tuple[RelativeODObservation, ...]
    measurement_model: str
    measurement_origin: str = "chief"


def build_relative_od_parameter_set(initial_state_ric: np.ndarray) -> ParameterSet:
    state = np.asarray(initial_state_ric, dtype=float).reshape(6)
    return ParameterSet(
        [
            EstimatedParameter("R_km", float(state[0]), scale=0.1, lower=-1000.0, upper=1000.0, unit="km"),
            EstimatedParameter("I_km", float(state[1]), scale=0.1, lower=-1000.0, upper=1000.0, unit="km"),
            EstimatedParameter("C_km", float(state[2]), scale=0.1, lower=-1000.0, upper=1000.0, unit="km"),
            EstimatedParameter("Rdot_km_s", float(state[3]), scale=1.0e-4, lower=-1.0, upper=1.0, unit="km/s"),
            EstimatedParameter("Idot_km_s", float(state[4]), scale=1.0e-4, lower=-1.0, upper=1.0, unit="km/s"),
            EstimatedParameter("Cdot_km_s", float(state[5]), scale=1.0e-4, lower=-1.0, upper=1.0, unit="km/s"),
        ]
    )


def make_relative_od_arc(
    *,
    times_s: Sequence[float],
    measurements: Sequence[Sequence[float] | np.ndarray],
    measurement_model: str,
    noise_diag: Sequence[float] | np.ndarray,
    covariance_blocks: Sequence[Sequence[Sequence[float]]] | np.ndarray | None = None,
    measurement_origin: str = "chief",
) -> RelativeODArc:
    model = normalize_hcw_measurement_model(measurement_model)
    dim = hcw_measurement_dimension(model)
    noise = np.asarray(noise_diag, dtype=float).reshape(-1)
    if noise.size != dim:
        raise ValueError(f"noise_diag must be length-{dim} for measurement_model={model!r}.")
    if not np.all(np.isfinite(noise)) or np.any(noise <= 0.0):
        raise ValueError("noise_diag entries must be finite and positive.")
    times = np.asarray(times_s, dtype=float).reshape(-1)
    if times.size != len(measurements):
        raise ValueError("times_s and measurements must have the same length.")
    if times.size < 2:
        raise ValueError("relative OD requires at least two observations.")
    covariance_array: np.ndarray | None = None
    if covariance_blocks is not None:
        covariance_array = np.asarray(covariance_blocks, dtype=float)
        if covariance_array.shape == (dim, dim):
            covariance_array = np.broadcast_to(covariance_array, (times.size, dim, dim)).copy()
        if covariance_array.shape != (times.size, dim, dim):
            raise ValueError("covariance_blocks must have shape (M, M) or (N, M, M).")
    observations: list[RelativeODObservation] = []
    for time_s, vector_raw in zip(times, measurements):
        vector = np.asarray(vector_raw, dtype=float).reshape(-1)
        if vector.size != dim:
            raise ValueError(f"measurement vector must be length-{dim} for measurement_model={model!r}.")
        if not np.all(np.isfinite(vector)):
            raise ValueError("relative OD measurement vectors must be finite.")
        covariance = None
        if covariance_array is not None:
            covariance = validate_covariance_block(
                covariance_array[len(observations)],
                dimension=dim,
                field_name=f"relative observation {len(observations)} covariance",
            )
        observations.append(
            RelativeODObservation(
                float(time_s),
                vector.astype(float),
                noise.astype(float),
                covariance,
            )
        )
    ordered_times = np.array([item.time_s for item in observations], dtype=float)
    if not np.all(np.isfinite(ordered_times)):
        raise ValueError("relative OD observation times must be finite.")
    if np.any(np.diff(ordered_times) < 0.0):
        raise ValueError("relative OD observation times must be ordered.")
    if np.any(np.diff(ordered_times) == 0.0):
        raise ValueError("relative OD observation times must be unique.")
    t0 = observations[0].time_s
    shifted = tuple(
        RelativeODObservation(
            time_s=float(obs.time_s - t0),
            vector=obs.vector,
            noise_diag=obs.noise_diag,
            covariance=obs.covariance,
        )
        for obs in observations
    )
    return RelativeODArc(
        observations=shifted, measurement_model=model, measurement_origin=str(measurement_origin or "chief")
    )


def sample_relative_od_observations(
    *,
    dynamics_model: str,
    initial_relative_state_ric: np.ndarray,
    times_s: Sequence[float],
    measurement_model: str,
    noise_diag: Sequence[float] | np.ndarray,
    mean_motion_rad_s: float | None = None,
    chief_initial_state_eci_km_s: np.ndarray | None = None,
    measurement_origin: str = "chief",
    mu_km3_s2: float = EARTH_MU_KM3_S2,
    j2: float = EARTH_J2,
    earth_radius_km: float = EARTH_RADIUS_KM,
    maximum_ss_eccentricity: float = 0.01,
    integration_substep_s: float = 10.0,
    transition_evaluation_method: str = "direct",
    numeric_backend: str = "rust",
    rng_seed: int | None = None,
) -> RelativeODArc:
    model = normalize_hcw_measurement_model(measurement_model)
    noise = np.asarray(noise_diag, dtype=float).reshape(-1)
    rng = np.random.default_rng(rng_seed)
    dynamics = normalize_relative_od_dynamics(dynamics_model)
    requested_times = np.asarray(times_s, dtype=float).reshape(-1)
    relative_times = requested_times - float(requested_times[0]) if requested_times.size else requested_times
    transitions = _relative_transition_cache(
        dynamics,
        relative_times,
        mean_motion_rad_s=mean_motion_rad_s,
        chief_initial_state_eci_km_s=chief_initial_state_eci_km_s,
        mu_km3_s2=mu_km3_s2,
        j2=j2,
        earth_radius_km=earth_radius_km,
        maximum_ss_eccentricity=maximum_ss_eccentricity,
        integration_substep_s=integration_substep_s,
        transition_evaluation_method=transition_evaluation_method,
        numeric_backend=numeric_backend,
    )
    initial_state = np.asarray(initial_relative_state_ric, dtype=float).reshape(6)
    measurements: list[np.ndarray] = []
    for time_s in relative_times:
        state = transitions[float(time_s)] @ initial_state
        z = hcw_measurement_vector(model, state, measurement_origin=measurement_origin)
        if rng_seed is not None:
            z = z + rng.normal(0.0, np.sqrt(noise), size=z.size)
        measurements.append(np.asarray(z, dtype=float))
    return make_relative_od_arc(
        times_s=relative_times,
        measurements=measurements,
        measurement_model=model,
        noise_diag=noise,
        measurement_origin=measurement_origin,
    )


def propagate_relative_od_state(
    initial_relative_state_ric: np.ndarray,
    dt_s: float,
    *,
    dynamics_model: str,
    mean_motion_rad_s: float | None = None,
    chief_initial_state_eci_km_s: np.ndarray | None = None,
    mu_km3_s2: float = EARTH_MU_KM3_S2,
    j2: float = EARTH_J2,
    earth_radius_km: float = EARTH_RADIUS_KM,
    maximum_ss_eccentricity: float = 0.01,
    integration_substep_s: float = 10.0,
    numeric_backend: str = "rust",
) -> np.ndarray:
    model = normalize_relative_od_dynamics(dynamics_model)
    state0 = np.asarray(initial_relative_state_ric, dtype=float).reshape(6)
    if model in {"hcw", "ss_j2"}:
        if mean_motion_rad_s is None:
            raise ValueError("mean_motion_rad_s is required for HCW and SS-J2 relative OD.")
        if model == "hcw":
            dynamics = RelativeLinearDynamics(
                model="hcw",
                mean_motion_rad_s=float(mean_motion_rad_s),
                numeric_backend=numeric_backend,
            )
        else:
            if chief_initial_state_eci_km_s is None:
                raise ValueError("chief_initial_state_eci_km_s is required for SS-J2 relative OD.")
            dynamics = RelativeLinearDynamics.ss_j2_from_chief_state(
                chief_initial_state_eci_km_s,
                mu_km3_s2=float(mu_km3_s2),
                j2=float(j2),
                earth_radius_km=float(earth_radius_km),
                maximum_supported_eccentricity=float(maximum_ss_eccentricity),
                numeric_backend=numeric_backend,
            )
        return dynamics.propagate(state0, float(dt_s))
    if chief_initial_state_eci_km_s is None:
        raise ValueError("chief_initial_state_eci_km_s is required for TH and YA relative OD.")
    chief0 = np.asarray(chief_initial_state_eci_km_s, dtype=float).reshape(6)
    if model == "th":
        return th_propagate_relative_state(
            state0,
            float(dt_s),
            chief0,
            mu_km3_s2=float(mu_km3_s2),
            max_step_s=float(integration_substep_s),
            numeric_backend=numeric_backend,
        )
    propagated, _phi = ya_closed_form_propagate_relative_state_and_stm(
        state0,
        float(dt_s),
        chief0,
        mu_km3_s2=float(mu_km3_s2),
        max_step_s=float(integration_substep_s),
        numeric_backend=numeric_backend,
    )
    return propagated


def solve_batch_relative_od(
    arc: RelativeODArc | Mapping[str, Any],
    *,
    dynamics_model: str,
    initial_guess_ric: np.ndarray,
    output_dir: str | Path | None = None,
    fit_duration_s: float | None = None,
    holdout_duration_s: float | None = None,
    mean_motion_rad_s: float | None = None,
    chief_initial_state_eci_km_s: np.ndarray | None = None,
    truth_initial_relative_state_ric: np.ndarray | None = None,
    max_nfev: int = 40,
    robust_loss: str = "linear",
    robust_f_scale: float = 1.0,
    sigma_clip_threshold: float | None = None,
    prior_mean_native: np.ndarray | None = None,
    prior_covariance_native: np.ndarray | None = None,
    mu_km3_s2: float = EARTH_MU_KM3_S2,
    j2: float = EARTH_J2,
    earth_radius_km: float = EARTH_RADIUS_KM,
    maximum_ss_eccentricity: float = 0.01,
    integration_substep_s: float = 10.0,
    transition_evaluation_method: str = "direct",
    numeric_backend: str = "rust",
) -> dict[str, Any]:
    arc_obj = _coerce_relative_od_arc(arc)
    model = normalize_relative_od_dynamics(dynamics_model)
    times = np.array([obs.time_s for obs in arc_obj.observations], dtype=float)
    partition = partition_time_arc(
        times,
        fit_duration_s=fit_duration_s,
        holdout_duration_s=holdout_duration_s,
    )
    fit_duration = partition.fit_duration_s
    holdout_duration = partition.holdout_duration_s
    fit_mask = partition.fit_mask
    holdout_mask = partition.holdout_mask

    parameters = build_relative_od_parameter_set(np.asarray(initial_guess_ric, dtype=float).reshape(6))
    transition_provenance: dict[str, Any] = {}
    transition_cache = _relative_transition_cache(
        model,
        times,
        mean_motion_rad_s=mean_motion_rad_s,
        chief_initial_state_eci_km_s=chief_initial_state_eci_km_s,
        mu_km3_s2=mu_km3_s2,
        j2=j2,
        earth_radius_km=earth_radius_km,
        maximum_ss_eccentricity=maximum_ss_eccentricity,
        integration_substep_s=integration_substep_s,
        transition_evaluation_method=transition_evaluation_method,
        numeric_backend=numeric_backend,
        provenance=transition_provenance,
    )
    observation_factors = tuple(
        prepare_covariance_whitener(
            covariance_from_sigmas(np.sqrt(np.maximum(obs.noise_diag, 1.0e-18)))
            if obs.covariance is None
            else obs.covariance,
            dimension=obs.vector.size,
            field_name=f"relative observation at {obs.time_s:g} s covariance",
        )
        for obs in arc_obj.observations
    )

    def predict_state(x0: np.ndarray, t: float) -> np.ndarray:
        epoch = float(t)
        phi = transition_cache.get(epoch)
        if phi is None:
            phi = _relative_transition_matrix_for_od(
                model,
                epoch,
                mean_motion_rad_s=mean_motion_rad_s,
                chief_initial_state_eci_km_s=chief_initial_state_eci_km_s,
                mu_km3_s2=mu_km3_s2,
                j2=j2,
                earth_radius_km=earth_radius_km,
                maximum_ss_eccentricity=maximum_ss_eccentricity,
                integration_substep_s=integration_substep_s,
                numeric_backend=numeric_backend,
            )
            transition_cache[epoch] = phi
        return phi @ np.asarray(x0, dtype=float)

    # Linear state measurements share one shape. Keep the exact observation
    # ordering and LAPACK solve for each factor while batching their dispatch.
    state_residual_arrays = None
    if arc_obj.measurement_model == "relative_state":
        state_residual_arrays = (
            np.stack([transition_cache[float(obs.time_s)] for obs in arc_obj.observations]),
            np.stack([obs.vector for obs in arc_obj.observations]),
            np.stack(observation_factors),
            1.0 if _normalize_measurement_origin(arc_obj.measurement_origin) == "chief" else -1.0,
        )

    def residual_for_mask(x0: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if state_residual_arrays is not None:
            transitions, observations, factors, sign = state_residual_arrays
            if not np.any(mask):
                return np.zeros(0, dtype=float)
            predicted = sign * (transitions[mask] @ np.asarray(x0, dtype=float))
            innovation = observations[mask] - predicted
            if not np.isfinite(innovation).all():
                raise ValueError("residual block must be finite.")
            return np.linalg.solve(factors[mask], innovation[..., None])[..., 0].reshape(-1)
        chunks: list[np.ndarray] = []
        for obs, factor, include in zip(arc_obj.observations, observation_factors, mask):
            if not include:
                continue
            predicted = hcw_measurement_vector(
                arc_obj.measurement_model,
                predict_state(x0, obs.time_s),
                measurement_origin=arc_obj.measurement_origin,
            )
            innovation = _measurement_innovation(arc_obj.measurement_model, obs.vector, predicted)
            chunks.append(
                whiten_residual_with_factor(
                    innovation,
                    factor,
                    field_name=f"relative observation at {obs.time_s:g} s covariance",
                )
            )
        return np.concatenate(chunks) if chunks else np.zeros(0, dtype=float)

    def residual(x0: np.ndarray) -> np.ndarray:
        return residual_for_mask(x0, fit_mask)

    initial_state = parameters.initial_native()
    prefit_fit_residual = residual(initial_state)
    solve = solve_batch_least_squares(
        parameters,
        residual,
        max_nfev=max_nfev,
        robust_loss=robust_loss,
        robust_f_scale=robust_f_scale,
        sigma_clip_threshold=sigma_clip_threshold,
        prior_mean_native=prior_mean_native,
        prior_covariance_native=prior_covariance_native,
    )
    fit_residual = residual_for_mask(solve.x_native, fit_mask)
    holdout_residual = residual_for_mask(solve.x_native, holdout_mask)

    predicted_fit_states = _states_for_mask(predict_state, solve.x_native, times, fit_mask)
    predicted_holdout_states = _states_for_mask(predict_state, solve.x_native, times, holdout_mask)
    truth_fit_states = _truth_states_for_mask(
        truth_initial_relative_state_ric,
        predict_state,
        times,
        fit_mask,
    )
    truth_holdout_states = _truth_states_for_mask(
        truth_initial_relative_state_ric,
        predict_state,
        times,
        holdout_mask,
    )
    prefit_fit_states = _states_for_mask(predict_state, initial_state, times, fit_mask)
    truth_prefit_states = _truth_states_for_mask(truth_initial_relative_state_ric, predict_state, times, fit_mask)
    prefit_state_metrics = _state_error_metrics(prefit_fit_states, truth_prefit_states)
    fit_state_metrics = _state_error_metrics(predicted_fit_states, truth_fit_states)
    holdout_state_metrics = _state_error_metrics(predicted_holdout_states, truth_holdout_states)
    state_metric_basis = {
        "kind": (
            "model_consistent_initial_state_reference"
            if truth_initial_relative_state_ric is not None
            else "unavailable"
        ),
        "independent_truth_trajectory": False,
        "supports_model_mismatch_assessment": False,
        "note": (
            "The supplied initial reference is propagated with the fitted dynamics model. "
            "These state-error metrics assess model-consistent initial-state recovery, not model mismatch."
            if truth_initial_relative_state_ric is not None
            else "No initial state reference was supplied."
        ),
    }

    quality_gates = _relative_od_quality_gates(
        solve,
        prefit_fit_residual,
        fit_residual,
        holdout_residual,
        fit_state_metrics,
        holdout_state_metrics,
    )
    quality_gates["state_metric_basis"] = state_metric_basis["kind"]
    quality_gates["model_mismatch_evaluated"] = False
    solver_summary = _solver_summary(solve)
    solver_summary["decision_records"] = _label_relative_residual_decisions(
        solve.decision_records,
        observations=[obs for obs, include in zip(arc_obj.observations, fit_mask) if include],
        measurement_dimension=hcw_measurement_dimension(arc_obj.measurement_model),
    )
    component_count = hcw_measurement_dimension(arc_obj.measurement_model)
    audit_records = [
        *residual_records_from_vectors(
            fit_residual,
            partition="fit",
            measurement_type=arc_obj.measurement_model,
            component_count=component_count,
            arc_id="relative_od",
        ),
        *residual_records_from_vectors(
            holdout_residual,
            partition="holdout",
            measurement_type=arc_obj.measurement_model,
            component_count=component_count,
            arc_id="relative_od",
        ),
    ]
    result: dict[str, Any] = {
        "method": "batch_relative_orbit_determination",
        "dynamics_model": model,
        "dynamics_metadata": _relative_od_dynamics_metadata(
            model,
            mean_motion_rad_s=mean_motion_rad_s,
            chief_initial_state_eci_km_s=chief_initial_state_eci_km_s,
            mu_km3_s2=mu_km3_s2,
            j2=j2,
            earth_radius_km=earth_radius_km,
            maximum_ss_eccentricity=maximum_ss_eccentricity,
            integration_substep_s=integration_substep_s,
            numeric_backend=numeric_backend,
            transition_provenance=transition_provenance,
        ),
        "measurement_model": arc_obj.measurement_model,
        "measurement_origin": arc_obj.measurement_origin,
        "fit_duration_s": fit_duration,
        "holdout_duration_s": holdout_duration,
        "observation_count": len(arc_obj.observations),
        "fit_observation_count": int(np.count_nonzero(fit_mask)),
        "holdout_observation_count": int(np.count_nonzero(holdout_mask)),
        "observation_partition": partition.summary,
        "epoch_evaluation": {
            "schema_version": 1,
            "method": str(transition_provenance["epoch_evaluation_method"]),
            "transition_provenance": dict(transition_provenance),
            "interpolation_used": False,
            "epoch_count": int(np.count_nonzero(fit_mask | holdout_mask)),
        },
        "initial_guess_ric": _float_list(initial_state),
        "estimated_initial_state_ric": _float_list(solve.x_native),
        "estimated_parameters": parameters.metadata(solve.x_native),
        "state_metric_basis": state_metric_basis,
        "solver": solver_summary,
        "estimation_policy": {
            "robust_loss": robust_loss,
            "robust_f_scale": float(robust_f_scale),
            "sigma_clip_threshold": sigma_clip_threshold,
            "prior_enabled": prior_mean_native is not None,
        },
        "prefit_metrics": _residual_metrics(prefit_fit_residual) | {"state": prefit_state_metrics},
        "fit_metrics": _residual_metrics(fit_residual) | {"state": fit_state_metrics},
        "holdout_metrics": _residual_metrics(holdout_residual) | {"state": holdout_state_metrics},
        "residual_audit": build_residual_audit(
            audit_records,
            decision_records=solver_summary["decision_records"],
        ),
        "quality_gates": quality_gates,
        "verdict": _relative_od_verdict(quality_gates),
    }
    if solve.covariance_native is not None:
        result["covariance_native"] = np.asarray(solve.covariance_native, dtype=float).tolist()

    if output_dir is not None:
        output_root = Path(output_dir)
        mkdir_output(output_root)
        fit_csv = output_root / "relative_od_fit_residuals.csv"
        holdout_csv = output_root / "relative_od_holdout_residuals.csv"
        report_json = output_root / "relative_od_fit_report.json"
        report_md = output_root / "relative_od_fit_report.md"
        _write_residual_csv(
            fit_csv,
            arc_obj,
            fit_mask,
            solve.x_native,
            predict_state,
            partition="fit",
            epoch_evaluation_method=str(transition_provenance["epoch_evaluation_method"]),
        )
        _write_residual_csv(
            holdout_csv,
            arc_obj,
            holdout_mask,
            solve.x_native,
            predict_state,
            partition="holdout",
            epoch_evaluation_method=str(transition_provenance["epoch_evaluation_method"]),
        )
        result["fit_residuals_csv"] = str(fit_csv)
        result["holdout_residuals_csv"] = str(holdout_csv)
        result["report_json_path"] = str(report_json)
        result["report_md_path"] = str(report_md)
        write_json(str(report_json), result)
        write_output_text(str(report_md), _render_relative_od_report_md(result))
    return result


def compare_relative_od_models(
    arc: RelativeODArc | Mapping[str, Any],
    *,
    models: Sequence[str] = ("hcw", "ss_j2", "th", "ya"),
    initial_guess_ric: np.ndarray,
    output_dir: str | Path | None = None,
    fit_duration_s: float | None = None,
    holdout_duration_s: float | None = None,
    mean_motion_rad_s: float | None = None,
    chief_initial_state_eci_km_s: np.ndarray | None = None,
    truth_initial_relative_state_ric: np.ndarray | None = None,
    eci_baseline: Mapping[str, Any] | None = None,
    max_nfev: int = 40,
    robust_loss: str = "linear",
    robust_f_scale: float = 1.0,
    sigma_clip_threshold: float | None = None,
    prior_mean_native: np.ndarray | None = None,
    prior_covariance_native: np.ndarray | None = None,
    mu_km3_s2: float = EARTH_MU_KM3_S2,
    j2: float = EARTH_J2,
    earth_radius_km: float = EARTH_RADIUS_KM,
    maximum_ss_eccentricity: float = 0.01,
    integration_substep_s: float = 10.0,
    transition_evaluation_method: str = "direct",
    numeric_backend: str = "rust",
) -> dict[str, Any]:
    output_root = None if output_dir is None else Path(output_dir)
    if output_root is not None:
        mkdir_output(output_root)
    rows: list[dict[str, Any]] = []
    reports: dict[str, dict[str, Any]] = {}
    for raw_model in models:
        model = normalize_relative_od_dynamics(raw_model)
        model_dir = None if output_root is None else output_root / model
        report = solve_batch_relative_od(
            arc,
            dynamics_model=model,
            initial_guess_ric=initial_guess_ric,
            output_dir=model_dir,
            fit_duration_s=fit_duration_s,
            holdout_duration_s=holdout_duration_s,
            mean_motion_rad_s=mean_motion_rad_s,
            chief_initial_state_eci_km_s=chief_initial_state_eci_km_s,
            truth_initial_relative_state_ric=truth_initial_relative_state_ric,
            max_nfev=max_nfev,
            robust_loss=robust_loss,
            robust_f_scale=robust_f_scale,
            sigma_clip_threshold=sigma_clip_threshold,
            prior_mean_native=prior_mean_native,
            prior_covariance_native=prior_covariance_native,
            mu_km3_s2=mu_km3_s2,
            j2=j2,
            earth_radius_km=earth_radius_km,
            maximum_ss_eccentricity=maximum_ss_eccentricity,
            integration_substep_s=integration_substep_s,
            transition_evaluation_method=transition_evaluation_method,
            numeric_backend=numeric_backend,
        )
        reports[model] = report
        rows.append(_comparison_row(model, report))
    if eci_baseline is not None:
        rows.append({"model": "eci_baseline", **dict(eci_baseline)})
    comparison = {
        "method": "relative_od_model_comparison",
        "models": [row["model"] for row in rows],
        "rows": rows,
        "reports": reports,
    }
    if output_root is not None:
        report_json = output_root / "relative_od_comparison_report.json"
        report_md = output_root / "relative_od_comparison_report.md"
        comparison["report_json_path"] = str(report_json)
        comparison["report_md_path"] = str(report_md)
        write_json(str(report_json), comparison)
        write_output_text(str(report_md), _render_relative_od_comparison_md(comparison))
    return comparison


def normalize_relative_od_dynamics(value: str) -> str:
    raw = str(value or "hcw").strip().lower().replace("-", "_")
    aliases = {
        "cw": "hcw",
        "clohessy_wiltshire": "hcw",
        "hill_clohessy_wiltshire": "hcw",
        "relative_hcw": "hcw",
        "ss": "ss_j2",
        "ssj2": "ss_j2",
        "schweighart_sedwick": "ss_j2",
        "schweighart_sedwick_j2": "ss_j2",
        "relative_ss_j2": "ss_j2",
        "tschauner_hempel": "th",
        "relative_th": "th",
        "th_integrated": "th",
        "yamanaka_ankersen": "ya",
        "ya_stm": "ya",
        "closed_form_ya": "ya",
        "relative_ya": "ya",
    }
    model = aliases.get(raw, raw)
    if model not in RELATIVE_OD_DYNAMICS:
        valid = ", ".join(sorted(RELATIVE_OD_DYNAMICS))
        raise ValueError(f"Unsupported relative OD dynamics_model {value!r}. Valid options: {valid}")
    return model


def _relative_od_dynamics_metadata(
    model: str,
    *,
    mean_motion_rad_s: float | None,
    chief_initial_state_eci_km_s: np.ndarray | None,
    mu_km3_s2: float,
    j2: float,
    earth_radius_km: float,
    maximum_ss_eccentricity: float,
    integration_substep_s: float,
    transition_provenance: Mapping[str, Any],
    numeric_backend: str = "rust",
) -> dict[str, Any]:
    if model in {"th", "ya"}:
        metadata = {
            "model": model,
            "constant_coefficient": False,
            "linear_time_varying": True,
            "transition_method": "variational_stm" if model == "th" else "closed_form_ya",
            "observation_epoch_transition_cache": True,
            "chief_reference": "two_body",
            "integration_substep_s": float(integration_substep_s),
        }
        if str(numeric_backend).strip().lower() == "rust":
            metadata["numeric_backend"] = "rust"
        metadata.update(
            {
                "transition_cache_method": str(transition_provenance["observed_method"]),
                "transition_evaluation_provenance": dict(transition_provenance),
            }
        )
        return metadata
    if mean_motion_rad_s is None:
        return {"model": model, "configuration_complete": False}
    if model == "hcw":
        return RelativeLinearDynamics(
            model="hcw", mean_motion_rad_s=mean_motion_rad_s, numeric_backend=numeric_backend
        ).metadata() | {
            "observation_epoch_transition_cache": True
        }
    if chief_initial_state_eci_km_s is None:
        return {"model": model, "configuration_complete": False}
    return RelativeLinearDynamics.ss_j2_from_chief_state(
        chief_initial_state_eci_km_s,
        mu_km3_s2=mu_km3_s2,
        j2=j2,
        earth_radius_km=earth_radius_km,
        maximum_supported_eccentricity=maximum_ss_eccentricity,
        numeric_backend=numeric_backend,
    ).metadata() | {"observation_epoch_transition_cache": True}


def _relative_transition_cache(
    model: str,
    times_s: Sequence[float] | np.ndarray,
    *,
    mean_motion_rad_s: float | None,
    chief_initial_state_eci_km_s: np.ndarray | None,
    mu_km3_s2: float,
    j2: float,
    earth_radius_km: float,
    maximum_ss_eccentricity: float,
    integration_substep_s: float,
    transition_evaluation_method: str = "direct",
    numeric_backend: str = "rust",
    provenance: dict[str, Any] | None = None,
) -> dict[float, np.ndarray]:
    method = _normalize_transition_evaluation_method(transition_evaluation_method)
    unique_times = np.unique(np.asarray(times_s, dtype=float))
    observed_method = method
    fallback_reason: str | None = None
    if method == "sequential_composition":
        if model not in {"th", "ya"}:
            observed_method = "direct"
            fallback_reason = "sequential_composition_only_supported_for_th_or_ya"
        elif unique_times.size and float(unique_times[0]) < 0.0:
            observed_method = "direct"
            fallback_reason = "negative_epoch_requires_direct_evaluation"
        elif not _chief_matches_validated_sequential_envelope(chief_initial_state_eci_km_s):
            observed_method = "direct"
            fallback_reason = "outside_validated_sequential_chief_envelope"
        elif not _epochs_match_validated_sequential_envelope(unique_times, integration_substep_s):
            observed_method = "direct"
            fallback_reason = "outside_validated_sequential_envelope"
    epoch_method = (
        "relative_dynamics_sequential_transition_composition"
        if observed_method == "sequential_composition"
        else "relative_dynamics_direct_at_observation_epoch"
    )
    if provenance is not None:
        provenance.update(
            {
                "requested_method": method,
                "observed_method": observed_method,
                "fallback_used": fallback_reason is not None,
                "fallback_reason": fallback_reason,
                "epoch_evaluation_method": epoch_method,
                "floating_point_parity": (
                    "rounding_level_different_from_direct_epoch_integration"
                    if observed_method == "sequential_composition"
                    else "direct_reference"
                ),
                "validated_sequential_envelope": {
                    "start_time_s": 0.0,
                    "uniform_observation_spacing_s": _SEQUENTIAL_VALIDATED_OBSERVATION_SPACING_S,
                    "integration_substep_s": _SEQUENTIAL_VALIDATED_SUBSTEP_S,
                    "maximum_horizon_s": _SEQUENTIAL_VALIDATED_MAX_HORIZON_S,
                    "minimum_semimajor_axis_km": _SEQUENTIAL_VALIDATED_MIN_SEMIMAJOR_AXIS_KM,
                    "maximum_semimajor_axis_km": _SEQUENTIAL_VALIDATED_MAX_SEMIMAJOR_AXIS_KM,
                    "maximum_eccentricity": _SEQUENTIAL_VALIDATED_MAX_ECCENTRICITY,
                    "minimum_perigee_km": _SEQUENTIAL_VALIDATED_MIN_PERIGEE_KM,
                    "conservative_observed_stm_abs_delta_bound": _SEQUENTIAL_VALIDATED_MAX_STM_ABS_DELTA,
                },
            }
        )
    if observed_method == "sequential_composition":
        if chief_initial_state_eci_km_s is None:
            raise ValueError("chief_initial_state_eci_km_s is required for TH and YA relative OD.")
        chief = np.asarray(chief_initial_state_eci_km_s, dtype=float).reshape(6)
        cumulative = np.eye(6, dtype=float)
        previous_time = 0.0
        cache: dict[float, np.ndarray] = {}
        for raw_time in unique_times:
            time_s = float(raw_time)
            step_s = time_s - previous_time
            chief_next = _propagate_chief_state(
                chief,
                step_s,
                mu_km3_s2=float(mu_km3_s2),
                max_step_s=float(integration_substep_s),
            )
            if model == "th":
                interval = th_variational_transition_matrix(
                    np.zeros(6),
                    step_s,
                    chief,
                    mu_km3_s2=mu_km3_s2,
                    max_step_s=integration_substep_s,
                    numeric_backend=numeric_backend,
                )
            else:
                interval = ya_closed_form_transition_matrix(
                    step_s,
                    chief,
                    chief_next,
                    mu_km3_s2=mu_km3_s2,
                    numeric_backend=numeric_backend,
                )
            cumulative = interval @ cumulative
            cache[time_s] = cumulative.copy()
            chief = chief_next
            previous_time = time_s
        return cache
    return {
        float(time_s): _relative_transition_matrix_for_od(
            model,
            float(time_s),
            mean_motion_rad_s=mean_motion_rad_s,
            chief_initial_state_eci_km_s=chief_initial_state_eci_km_s,
            mu_km3_s2=mu_km3_s2,
            j2=j2,
            earth_radius_km=earth_radius_km,
            maximum_ss_eccentricity=maximum_ss_eccentricity,
            integration_substep_s=integration_substep_s,
            numeric_backend=numeric_backend,
        )
        for time_s in unique_times
    }


def _normalize_transition_evaluation_method(value: str) -> str:
    method = str(value or "direct").strip().lower()
    if method not in {"direct", "sequential_composition"}:
        raise ValueError("transition_evaluation_method must be direct or sequential_composition.")
    return method


def _epochs_match_validated_sequential_envelope(
    times_s: np.ndarray, integration_substep_s: float
) -> bool:
    if float(integration_substep_s) != _SEQUENTIAL_VALIDATED_SUBSTEP_S:
        return False
    epochs = np.asarray(times_s, dtype=float).reshape(-1)
    if epochs.size < 2 or float(epochs[0]) != 0.0:
        return False
    if float(epochs[-1]) > _SEQUENTIAL_VALIDATED_MAX_HORIZON_S:
        return False
    return bool(
        np.array_equal(
            np.diff(epochs),
            np.full(epochs.size - 1, _SEQUENTIAL_VALIDATED_OBSERVATION_SPACING_S, dtype=float),
        )
    )


def _chief_matches_validated_sequential_envelope(
    chief_initial_state_eci_km_s: np.ndarray | None,
) -> bool:
    if chief_initial_state_eci_km_s is None:
        return False
    try:
        elements = rv_to_coe_eci(
            np.asarray(chief_initial_state_eci_km_s, dtype=float).reshape(6)[:3],
            np.asarray(chief_initial_state_eci_km_s, dtype=float).reshape(6)[3:],
            mu_km3_s2=EARTH_MU_KM3_S2,
        )
    except (TypeError, ValueError, FloatingPointError):
        return False
    semimajor_axis = float(elements.a_km)
    eccentricity = float(elements.ecc)
    perigee = semimajor_axis * (1.0 - eccentricity)
    return bool(
        _SEQUENTIAL_VALIDATED_MIN_SEMIMAJOR_AXIS_KM - 1.0e-9
        <= semimajor_axis
        <= _SEQUENTIAL_VALIDATED_MAX_SEMIMAJOR_AXIS_KM + 1.0e-9
        and -1.0e-12 <= eccentricity <= _SEQUENTIAL_VALIDATED_MAX_ECCENTRICITY + 1.0e-12
        and perigee >= _SEQUENTIAL_VALIDATED_MIN_PERIGEE_KM - 1.0e-9
    )


def _relative_transition_matrix_for_od(
    model: str,
    dt_s: float,
    *,
    mean_motion_rad_s: float | None,
    chief_initial_state_eci_km_s: np.ndarray | None,
    mu_km3_s2: float,
    j2: float,
    earth_radius_km: float,
    maximum_ss_eccentricity: float,
    integration_substep_s: float,
    numeric_backend: str = "rust",
) -> np.ndarray:
    if model in {"th", "ya"} and chief_initial_state_eci_km_s is None:
        raise ValueError("chief_initial_state_eci_km_s is required for TH and YA relative OD.")
    if model == "th":
        return th_variational_transition_matrix(
            np.zeros(6),
            dt_s,
            np.asarray(chief_initial_state_eci_km_s, dtype=float),
            mu_km3_s2=mu_km3_s2,
            max_step_s=integration_substep_s,
            numeric_backend=numeric_backend,
        )
    if model == "ya":
        _state, phi = ya_closed_form_propagate_relative_state_and_stm(
            np.zeros(6),
            dt_s,
            np.asarray(chief_initial_state_eci_km_s, dtype=float),
            mu_km3_s2=mu_km3_s2,
            max_step_s=integration_substep_s,
            numeric_backend=numeric_backend,
        )
        return phi
    if mean_motion_rad_s is None:
        raise ValueError("mean_motion_rad_s is required for HCW and SS-J2 relative OD.")
    if model == "hcw":
        return RelativeLinearDynamics(
            model="hcw", mean_motion_rad_s=mean_motion_rad_s, numeric_backend=numeric_backend
        ).state_transition_matrix(dt_s)
    if chief_initial_state_eci_km_s is None:
        raise ValueError("chief_initial_state_eci_km_s is required for SS-J2 relative OD.")
    return RelativeLinearDynamics.ss_j2_from_chief_state(
        chief_initial_state_eci_km_s,
        mu_km3_s2=mu_km3_s2,
        j2=j2,
        earth_radius_km=earth_radius_km,
        maximum_supported_eccentricity=maximum_ss_eccentricity,
        numeric_backend=numeric_backend,
    ).state_transition_matrix(dt_s)


def _coerce_relative_od_arc(value: RelativeODArc | Mapping[str, Any]) -> RelativeODArc:
    if isinstance(value, RelativeODArc):
        return value
    data = dict(value)
    return make_relative_od_arc(
        times_s=data["times_s"],
        measurements=data["measurements"],
        measurement_model=str(data.get("measurement_model", "relative_state")),
        noise_diag=data["noise_diag"],
        covariance_blocks=data.get("covariance_blocks"),
        measurement_origin=str(data.get("measurement_origin", "chief")),
    )


def _states_for_mask(
    predict_state: Any,
    initial_state: np.ndarray,
    times: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray | None:
    selected = times[mask]
    if selected.size == 0:
        return None
    return np.vstack([predict_state(initial_state, float(t)) for t in selected])


def _truth_states_for_mask(
    truth_initial_state: np.ndarray | None,
    predict_state: Any,
    times: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray | None:
    if truth_initial_state is None:
        return None
    return _states_for_mask(predict_state, np.asarray(truth_initial_state, dtype=float).reshape(6), times, mask)


def _state_error_metrics(predicted: np.ndarray | None, truth: np.ndarray | None) -> dict[str, float | None]:
    if predicted is None or truth is None or predicted.size == 0 or truth.size == 0:
        return {
            "position_rms_m": None,
            "velocity_rms_mm_s": None,
            "position_max_m": None,
            "velocity_max_mm_s": None,
        }
    delta = np.asarray(predicted, dtype=float) - np.asarray(truth, dtype=float)
    pos_m = np.linalg.norm(delta[:, :3], axis=1) * 1000.0
    vel_mm_s = np.linalg.norm(delta[:, 3:], axis=1) * 1.0e6
    return {
        "position_rms_m": float(np.sqrt(np.mean(pos_m**2))),
        "velocity_rms_mm_s": float(np.sqrt(np.mean(vel_mm_s**2))),
        "position_max_m": float(np.max(pos_m)),
        "velocity_max_mm_s": float(np.max(vel_mm_s)),
    }


def _residual_metrics(residual: np.ndarray) -> dict[str, float | int | None]:
    r = np.asarray(residual, dtype=float).reshape(-1)
    if r.size == 0:
        return {"weighted_rms": None, "weighted_abs_p95": None, "sample_count": 0}
    return {
        "weighted_rms": float(np.sqrt(np.mean(r**2))),
        "weighted_abs_p95": float(np.percentile(np.abs(r), 95.0)),
        "sample_count": int(r.size),
    }


def _solver_summary(solve: BatchLeastSquaresResult) -> dict[str, Any]:
    return {
        "success": bool(solve.success),
        "message": solve.message,
        "nfev": int(solve.nfev),
        "initial_cost": float(solve.initial_cost),
        "cost": float(solve.cost),
        "rms_residual": float(solve.rms_residual),
        "parameter_names": list(solve.parameter_names),
        "diagnostics": solve.diagnostics,
        "decision_records": [dict(row) for row in solve.decision_records],
    }


def _label_relative_residual_decisions(
    decisions: Sequence[Mapping[str, Any]],
    *,
    observations: Sequence[RelativeODObservation],
    measurement_dimension: int,
) -> list[dict[str, Any]]:
    labeled: list[dict[str, Any]] = []
    for decision in decisions:
        row = dict(decision)
        observation_index, component_index = divmod(int(row["residual_index"]), int(measurement_dimension))
        if observation_index < len(observations):
            row.update(
                {
                    "observation_id": f"relative:observation:{observation_index:06d}",
                    "time_s": float(observations[observation_index].time_s),
                    "component": int(component_index),
                    "measurement_type": "relative_observation",
                    "arc_id": "relative_od",
                    "partition": "fit",
                    "residual_space": "cholesky_whitened_component",
                }
            )
        labeled.append(row)
    return labeled


def _relative_od_quality_gates(
    solve: BatchLeastSquaresResult,
    prefit_residual: np.ndarray,
    fit_residual: np.ndarray,
    holdout_residual: np.ndarray,
    fit_state_metrics: Mapping[str, Any],
    holdout_state_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    prefit = _residual_metrics(prefit_residual)
    fit = _residual_metrics(fit_residual)
    holdout = _residual_metrics(holdout_residual)
    prefit_rms = _finite_or_none(prefit.get("weighted_rms"))
    fit_rms = _finite_or_none(fit.get("weighted_rms"))
    holdout_rms = _finite_or_none(holdout.get("weighted_rms"))
    warnings: list[str] = []
    fit_improved = prefit_rms is not None and fit_rms is not None and fit_rms < prefit_rms
    if not fit_improved:
        warnings.append("fit did not improve prefit weighted RMS")
    holdout_evaluated = holdout_rms is not None
    holdout_acceptable = bool(holdout_evaluated and fit_rms is not None and holdout_rms <= max(10.0 * fit_rms, 3.0))
    if not holdout_evaluated:
        warnings.append("holdout not evaluated")
    elif not holdout_acceptable:
        warnings.append("holdout weighted RMS is high relative to fit RMS")
    covariance_valid = solve.covariance_native is not None and np.all(np.isfinite(solve.covariance_native))
    if not covariance_valid:
        warnings.append("covariance unavailable or non-finite")
    observability = dict(solve.diagnostics.get("observability", {}) or {})
    data_full_rank = bool(observability.get("data_full_rank", False))
    if not data_full_rank:
        warnings.append("relative OD data Jacobian is rank deficient")
    return {
        "solver_success": bool(solve.success),
        "fit_improved_prefit_rms": bool(fit_improved),
        "covariance_valid": bool(covariance_valid),
        "data_full_rank": data_full_rank,
        "failure_classification": solve.diagnostics.get("failure_classification"),
        "holdout_acceptable": bool(holdout_acceptable),
        "holdout_evidence_status": "evaluated" if holdout_evaluated else "not_evaluated",
        "prefit_weighted_rms": prefit_rms,
        "fit_weighted_rms": fit_rms,
        "holdout_weighted_rms": holdout_rms,
        "fit_position_rms_m": _finite_or_none(fit_state_metrics.get("position_rms_m")),
        "holdout_position_rms_m": _finite_or_none(holdout_state_metrics.get("position_rms_m")),
        "warnings": warnings,
        "evidence_status": "review" if warnings else "ready",
    }


def _relative_od_verdict(quality_gates: Mapping[str, Any]) -> dict[str, Any]:
    warnings = list(quality_gates.get("warnings", []) or [])
    usable = (
        bool(quality_gates.get("solver_success"))
        and bool(quality_gates.get("fit_improved_prefit_rms"))
        and bool(quality_gates.get("holdout_acceptable"))
        and not warnings
    )
    return {
        "usable_for_relative_od_study": usable,
        "recommended_action": "use_with_caveats" if usable else "review_relative_od_fit",
        "warnings": warnings,
    }


def _comparison_row(model: str, report: Mapping[str, Any]) -> dict[str, Any]:
    fit = dict(report.get("fit_metrics", {}) or {})
    holdout = dict(report.get("holdout_metrics", {}) or {})
    fit_state = dict(fit.get("state", {}) or {})
    holdout_state = dict(holdout.get("state", {}) or {})
    return {
        "model": model,
        "fit_weighted_rms": fit.get("weighted_rms"),
        "holdout_weighted_rms": holdout.get("weighted_rms"),
        "fit_position_rms_m": fit_state.get("position_rms_m"),
        "holdout_position_rms_m": holdout_state.get("position_rms_m"),
        "position_rms_basis": dict(report.get("state_metric_basis", {}) or {}).get("kind"),
        "model_mismatch_evaluated": bool(
            dict(report.get("state_metric_basis", {}) or {}).get("supports_model_mismatch_assessment", False)
        ),
        "solver_success": dict(report.get("solver", {}) or {}).get("success"),
        "evidence_status": dict(report.get("quality_gates", {}) or {}).get("evidence_status"),
    }


def _write_residual_csv(
    path: Path,
    arc: RelativeODArc,
    mask: np.ndarray,
    initial_state: np.ndarray,
    predict_state: Any,
    *,
    partition: str,
    epoch_evaluation_method: str,
) -> None:
    fieldnames = [
        "observation_id",
        "partition",
        "residual_kind",
        "frame",
        "epoch_evaluation_method",
        "time_s",
        "component",
        "measurement",
        "predicted",
        "residual",
        "sigma",
        "weighted_residual",
    ]
    mkdir_output(path.parent)
    with staged_output_file(path) as target:
        with target.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for observation_index, (obs, include) in enumerate(zip(arc.observations, mask)):
                if not include:
                    continue
                predicted = hcw_measurement_vector(
                    arc.measurement_model,
                    predict_state(initial_state, obs.time_s),
                    measurement_origin=arc.measurement_origin,
                )
                residual = _measurement_innovation(arc.measurement_model, obs.vector, predicted)
                covariance = (
                    covariance_from_sigmas(np.sqrt(np.maximum(obs.noise_diag, 1.0e-18)))
                    if obs.covariance is None
                    else obs.covariance
                )
                weighted = whiten_residual_block(
                    residual,
                    covariance,
                    field_name=f"relative observation at {obs.time_s:g} s covariance",
                )
                sigmas = np.sqrt(np.diag(covariance))
                for idx, (meas, pred, res, wr) in enumerate(zip(obs.vector, predicted, residual, weighted)):
                    writer.writerow(
                        {
                            "observation_id": f"relative:observation:{observation_index:06d}",
                            "partition": partition,
                            "residual_kind": "postfit" if partition == "fit" else "prediction",
                            "frame": "RIC",
                            "epoch_evaluation_method": str(epoch_evaluation_method),
                            "time_s": float(obs.time_s),
                            "component": int(idx),
                            "measurement": float(meas),
                            "predicted": float(pred),
                            "residual": float(res),
                            "sigma": float(sigmas[idx]),
                            "weighted_residual": float(wr),
                        }
                    )


def _render_relative_od_report_md(result: Mapping[str, Any]) -> str:
    fit = dict(result.get("fit_metrics", {}) or {})
    holdout = dict(result.get("holdout_metrics", {}) or {})
    gates = dict(result.get("quality_gates", {}) or {})
    state_basis = dict(result.get("state_metric_basis", {}) or {})
    state_basis_line = (
        "model-consistent initial-state reference; no independent truth trajectory"
        if state_basis.get("kind") == "model_consistent_initial_state_reference"
        else "unavailable; no initial-state reference supplied"
    )
    lines = [
        "# Relative Orbit Determination Report",
        "",
        f"- Dynamics model: `{result.get('dynamics_model')}`",
        f"- Measurement model: `{result.get('measurement_model')}`",
        f"- Solver success: `{dict(result.get('solver', {}) or {}).get('success')}`",
        f"- Evidence status: `{gates.get('evidence_status')}`",
        f"- State-error basis: {state_basis_line}",
        "",
        "| Window | Weighted RMS | Model-consistent position RMS m |",
        "| --- | ---: | ---: |",
        f"| Fit | {_fmt(fit.get('weighted_rms'))} | {_fmt(dict(fit.get('state', {}) or {}).get('position_rms_m'))} |",
        f"| Holdout | {_fmt(holdout.get('weighted_rms'))} | {_fmt(dict(holdout.get('state', {}) or {}).get('position_rms_m'))} |",
        "",
    ]
    warnings = list(gates.get("warnings", []) or [])
    if warnings:
        lines.extend(["## Warnings", "", *[f"- {item}" for item in warnings], ""])
    return "\n".join(lines)


def _render_relative_od_comparison_md(comparison: Mapping[str, Any]) -> str:
    lines = [
        "# Relative OD Model Comparison",
        "",
        "When present, position RMS columns use each model's own propagation of the supplied initial-state reference; "
        "they do not evaluate model mismatch. Use an independent nonlinear or external baseline for that claim.",
        "",
        "| Model | Fit Weighted RMS | Holdout Weighted RMS | Fit Model-consistent Position RMS m | Holdout Model-consistent Position RMS m |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in list(comparison.get("rows", []) or []):
        lines.append(
            f"| `{row.get('model')}` | {_fmt(row.get('fit_weighted_rms'))} | {_fmt(row.get('holdout_weighted_rms'))} | "
            f"{_fmt(row.get('fit_position_rms_m'))} | {_fmt(row.get('holdout_position_rms_m'))} |"
        )
    lines.append("")
    return "\n".join(lines)


def _float_list(value: np.ndarray) -> list[float]:
    return [float(x) for x in np.asarray(value, dtype=float).reshape(-1)]


def _finite_or_none(value: Any) -> float | None:
    if value is None:
        return None
    out = float(value)
    return out if np.isfinite(out) else None


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.6g}"
    except (TypeError, ValueError):
        return str(value)
