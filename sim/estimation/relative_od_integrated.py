from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from sim.core.models import Measurement, StateBelief
from sim.dynamics.orbit.accelerations import OrbitContext
from sim.dynamics.orbit.elements import rv_to_coe_eci
from sim.dynamics.orbit.environment import EARTH_MU_KM3_S2
from sim.dynamics.orbit.propagator import OrbitPropagator
from sim.estimation.relative_hcw_ekf import HCWRelativeEKFEstimator, SSJ2RelativeEKFEstimator
from sim.estimation.relative_od import (
    compare_relative_od_models,
    make_relative_od_arc,
    normalize_relative_od_dynamics,
    propagate_relative_od_state,
    solve_batch_relative_od,
)
from sim.estimation.relative_onp_variational import onp_relative_variational_propagate
from sim.estimation.relative_th_ekf import THRelativeEKFEstimator, YARelativeEKFEstimator
from sim.utils.frames import eci_relative_to_ric_rect, ric_rect_state_to_eci
from sim.utils.io import write_json

INTEGRATED_RELATIVE_OD_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class NonlinearRelativeTruth:
    times_s: np.ndarray
    chief_states_eci_km_s: np.ndarray
    deputy_states_eci_km_s: np.ndarray
    relative_states_ric: np.ndarray
    maneuver: Mapping[str, Any] | None = None
    integration_substep_s: float | None = None


def generate_nonlinear_relative_truth(
    *,
    times_s: Sequence[float] | np.ndarray,
    chief_initial_state_eci_km_s: Sequence[float] | np.ndarray,
    initial_relative_state_ric: Sequence[float] | np.ndarray,
    propagator: OrbitPropagator | None = None,
    chief_context: OrbitContext | None = None,
    deputy_context: OrbitContext | None = None,
    environment: Mapping[str, Any] | None = None,
    integration_substep_s: float = 10.0,
    maneuver_index: int | None = None,
    maneuver_delta_v_ric_m_s: Sequence[float] | np.ndarray | None = None,
) -> NonlinearRelativeTruth:
    times = np.asarray(times_s, dtype=float).reshape(-1)
    if times.size < 3 or np.any(~np.isfinite(times)) or np.any(np.diff(times) <= 0.0):
        raise ValueError("times_s must contain at least three finite, strictly increasing epochs.")
    integration_substep = _positive_finite(
        integration_substep_s, "integration_substep_s"
    )
    chief0 = _state6(chief_initial_state_eci_km_s, "chief_initial_state_eci_km_s")
    relative0 = _state6(initial_relative_state_ric, "initial_relative_state_ric")
    prop = propagator or OrbitPropagator(integrator="rk4")
    chief_ctx = chief_context or OrbitContext(mu_km3_s2=EARTH_MU_KM3_S2, mass_kg=100.0)
    deputy_ctx = deputy_context or chief_ctx
    env = dict(environment or {})
    chief_accel0 = _chief_acceleration_or_none(prop, chief0, float(times[0]), chief_ctx, env)
    deputy0 = ric_rect_state_to_eci(
        relative0,
        chief0[:3],
        chief0[3:],
        chief_accel_eci_km_s2=chief_accel0,
    )
    chief_states = [chief0]
    deputy_states = [deputy0]
    maneuver_record = None
    if maneuver_index is not None:
        if maneuver_index <= 0 or maneuver_index >= times.size:
            raise ValueError("maneuver_index must select an interior observation epoch.")
        if maneuver_delta_v_ric_m_s is None:
            raise ValueError("maneuver_delta_v_ric_m_s is required with maneuver_index.")
        delta_v_ric = np.asarray(maneuver_delta_v_ric_m_s, dtype=float).reshape(3)
        maneuver_record = {
            "observation_index": int(maneuver_index),
            "time_s": float(times[maneuver_index]),
            "delta_v_ric_m_s": delta_v_ric.tolist(),
        }
    for index in range(1, times.size):
        chief_previous = chief_states[-1]
        deputy_previous = deputy_states[-1].copy()
        dt = float(times[index] - times[index - 1])
        chief_current = _propagate_interval(
            prop, chief_previous, times[index - 1], dt, chief_ctx, env, integration_substep
        )
        deputy_current = _propagate_interval(
            prop, deputy_previous, times[index - 1], dt, deputy_ctx, env, integration_substep
        )
        if maneuver_index is not None and index == maneuver_index:
            # The event contract names times[maneuver_index].  Apply the
            # instantaneous delta-v on arrival at that epoch so the recorded
            # state is post-maneuver; the previous implementation applied it
            # one entire observation interval early.
            chief_accel = _chief_acceleration_or_none(
                prop,
                chief_current,
                float(times[index]),
                chief_ctx,
                env,
            )
            relative_at_event = eci_relative_to_ric_rect(
                deputy_current,
                chief_current,
                chief_accel_eci_km_s2=chief_accel,
            )
            relative_at_event[3:] += np.asarray(maneuver_delta_v_ric_m_s, dtype=float) / 1000.0
            deputy_current = ric_rect_state_to_eci(
                relative_at_event,
                chief_current[:3],
                chief_current[3:],
                chief_accel_eci_km_s2=chief_accel,
            )
        chief_states.append(chief_current)
        deputy_states.append(deputy_current)
    chief_array = np.asarray(chief_states)
    deputy_array = np.asarray(deputy_states)
    chief_accelerations = tuple(
        _chief_acceleration_or_none(prop, chief, float(times[index]), chief_ctx, env)
        for index, chief in enumerate(chief_array)
    )
    relative = np.asarray(
        [
            eci_relative_to_ric_rect(
                deputy,
                chief,
                chief_accel_eci_km_s2=chief_accelerations[index],
            )
            for index, (deputy, chief) in enumerate(zip(deputy_array, chief_array, strict=True))
        ]
    )
    return NonlinearRelativeTruth(
        times_s=times,
        chief_states_eci_km_s=chief_array,
        deputy_states_eci_km_s=deputy_array,
        relative_states_ric=relative,
        maneuver=maneuver_record,
        integration_substep_s=integration_substep,
    )


def run_integrated_relative_od(
    *,
    times_s: Sequence[float] | np.ndarray,
    measurements_ric: Sequence[Sequence[float]] | np.ndarray,
    sensor_covariance: Sequence[Sequence[float]] | np.ndarray,
    chief_initial_state_eci_km_s: Sequence[float] | np.ndarray,
    chief_initial_covariance: Sequence[Sequence[float]] | np.ndarray,
    initial_guess_ric: Sequence[float] | np.ndarray,
    truth_initial_relative_state_ric: Sequence[float] | np.ndarray,
    fit_duration_s: float,
    propagator: OrbitPropagator | None = None,
    orbit_context: OrbitContext | None = None,
    environment: Mapping[str, Any] | None = None,
    integration_substep_s: float = 10.0,
    chief_relative_cross_covariance: Sequence[Sequence[float]] | np.ndarray | None = None,
    nonlinear_truth: NonlinearRelativeTruth | None = None,
    models: Sequence[str] = ("hcw", "ss_j2", "th", "ya"),
    sigma_clip_threshold: float | None = None,
    arc_break_indices: Sequence[int] | None = None,
    maximum_recommended_holdout_position_rms_m: float | None = None,
    output_dir: str | Path | None = None,
    numeric_backend: str = "rust",
) -> dict[str, Any]:
    backend = str(numeric_backend).strip().lower()
    if backend not in {"python", "rust"}:
        raise ValueError("numeric_backend must be 'python' or 'rust'.")
    if backend == "rust":
        from sim.rust_orbit_backend import _extension

        _extension()
    times = np.asarray(times_s, dtype=float).reshape(-1)
    if times.size < 3 or np.any(~np.isfinite(times)) or np.any(np.diff(times) <= 0.0):
        raise ValueError("times_s must contain at least three finite, strictly increasing epochs.")
    measurements = np.asarray(measurements_ric, dtype=float)
    if measurements.shape != (times.size, 6):
        raise ValueError("measurements_ric must have shape (observation_count, 6).")
    chief0 = _state6(chief_initial_state_eci_km_s, "chief_initial_state_eci_km_s")
    chief_covariance0 = _covariance6(chief_initial_covariance, "chief_initial_covariance")
    guess = _state6(initial_guess_ric, "initial_guess_ric")
    truth_initial = _state6(truth_initial_relative_state_ric, "truth_initial_relative_state_ric")
    integration_substep = _positive_finite(
        integration_substep_s, "integration_substep_s"
    )
    prop = propagator or OrbitPropagator(integrator="rk4", numeric_backend=backend)
    context = orbit_context or OrbitContext(mu_km3_s2=EARTH_MU_KM3_S2, mass_kg=100.0)
    env = dict(environment or {})
    truth = nonlinear_truth or generate_nonlinear_relative_truth(
        times_s=times,
        chief_initial_state_eci_km_s=chief0,
        initial_relative_state_ric=truth_initial,
        propagator=prop,
        chief_context=context,
        deputy_context=context,
        environment=env,
        integration_substep_s=integration_substep,
    )
    if not np.allclose(truth.times_s, times, rtol=0.0, atol=1.0e-9):
        raise ValueError("nonlinear_truth epochs must match times_s.")
    sensor_blocks = _covariance_blocks(sensor_covariance, count=times.size)
    cross_blocks = (
        np.zeros_like(sensor_blocks)
        if chief_relative_cross_covariance is None
        else _covariance_blocks(chief_relative_cross_covariance, count=times.size, allow_indefinite=True)
    )
    chief_states, chief_covariances = _propagate_state_and_covariance(
        initial_state=chief0,
        initial_covariance=chief_covariance0,
        times=times,
        propagator=prop,
        context=context,
        environment=env,
        integration_substep_s=integration_substep,
    )
    chief_relative_covariances = []
    effective_covariances = []
    for index in range(times.size):
        chief_acceleration = _chief_acceleration_or_none(
            prop,
            chief_states[index],
            float(times[index]),
            context,
            env,
        )
        jacobian = _relative_jacobian_wrt_chief(
            truth.deputy_states_eci_km_s[index],
            chief_states[index],
            chief_accel_eci_km_s2=chief_acceleration,
        )
        chief_relative = jacobian @ chief_covariances[index] @ jacobian.T
        effective = sensor_blocks[index] + chief_relative - cross_blocks[index] - cross_blocks[index].T
        chief_relative_covariances.append(_project_covariance(chief_relative))
        effective_covariances.append(_project_covariance(effective))
    effective_array = np.asarray(effective_covariances)

    arc = make_relative_od_arc(
        times_s=times,
        measurements=measurements,
        measurement_model="relative_state",
        noise_diag=np.diag(sensor_blocks[0]),
        covariance_blocks=effective_array,
    )
    elements = rv_to_coe_eci(chief0[:3], chief0[3:], mu_km3_s2=context.mu_km3_s2)
    mean_motion = float(np.sqrt(context.mu_km3_s2 / float(elements.a_km) ** 3))
    requested_models = tuple(normalize_relative_od_dynamics(model) for model in models)
    excluded_models: dict[str, dict[str, Any]] = {}
    applicable_models = []
    for model in requested_models:
        if model == "ss_j2" and float(elements.ecc) > 0.01:
            excluded_models[model] = {
                "status": "inapplicable",
                "reason": "chief_eccentricity_exceeds_ss_j2_limit",
                "chief_eccentricity": float(elements.ecc),
                "maximum_supported_eccentricity": 0.01,
            }
        else:
            applicable_models.append(model)
    comparison = compare_relative_od_models(
        arc,
        models=tuple(applicable_models),
        initial_guess_ric=guess,
        output_dir=None,
        fit_duration_s=float(fit_duration_s),
        holdout_duration_s=_remaining_arc_duration_s(times, fit_duration_s),
        mean_motion_rad_s=mean_motion,
        chief_initial_state_eci_km_s=chief0,
        truth_initial_relative_state_ric=truth_initial,
        sigma_clip_threshold=sigma_clip_threshold,
        numeric_backend=backend,
    )
    nonlinear_rows = []
    fit_mask = times - times[0] <= float(fit_duration_s) + 1.0e-9
    holdout_mask = ~fit_mask
    for model, model_report in comparison["reports"].items():
        estimated = np.asarray(model_report["estimated_initial_state_ric"], dtype=float)
        prediction = np.asarray(
            [
                propagate_relative_od_state(
                    estimated,
                    float(time_s - times[0]),
                    dynamics_model=model,
                    mean_motion_rad_s=mean_motion,
                    chief_initial_state_eci_km_s=chief0,
                    numeric_backend=backend,
                )
                for time_s in times
            ]
        )
        fit_metrics = _relative_error_metrics(prediction[fit_mask], truth.relative_states_ric[fit_mask])
        holdout_metrics = _relative_error_metrics(prediction[holdout_mask], truth.relative_states_ric[holdout_mask])
        nonlinear = {
            "model": model,
            "fit": fit_metrics,
            "holdout": holdout_metrics,
            "nonlinear_baseline_required": True,
        }
        model_report["nonlinear_baseline"] = nonlinear
        nonlinear_rows.append(nonlinear)
    eligible = [
        row
        for row in nonlinear_rows
        if row["holdout"]["available"]
        and bool(comparison["reports"][row["model"]]["solver"]["success"])
    ]
    selected_model = min(eligible, key=lambda row: row["holdout"]["position_rms_m"])["model"] if eligible else None
    selected_row = next((row for row in nonlinear_rows if row["model"] == selected_model), None)
    recommendation_threshold = maximum_recommended_holdout_position_rms_m
    recommended_model = (
        selected_model
        if selected_row is not None
        and recommendation_threshold is not None
        and selected_row["holdout"]["position_rms_m"] <= float(recommendation_threshold)
        else None
    )
    sequential = {
        model: _run_relative_filter(
            model=model,
            times=times,
            measurements=measurements,
            covariance_blocks=effective_array,
            initial_state=guess,
            initial_covariance=_project_covariance(effective_array[0] * 4.0),
            mean_motion_rad_s=mean_motion,
            chief_initial_state=chief0,
            truth=truth.relative_states_ric,
            numeric_backend=backend,
        )
        for model in applicable_models
    }
    multi_arc = _run_multi_arc(
        arc_break_indices=arc_break_indices,
        times=times,
        measurements=measurements,
        covariance_blocks=effective_array,
        model=selected_model,
        mean_motion_rad_s=mean_motion,
        chief_states=chief_states,
        truth=truth.relative_states_ric,
        numeric_backend=backend,
    )
    onp_variational = onp_relative_variational_propagate(
        truth_initial,
        float(times[-1] - times[0]),
        chief0,
        propagator=prop,
        context=context,
        environment=env,
        epoch_t_s=float(times[0]),
        integration_substep_s=integration_substep,
        numeric_backend=backend,
    )
    variational_terminal_error = onp_variational.relative_state_ric - truth.relative_states_ric[-1]
    report = {
        "schema_version": INTEGRATED_RELATIVE_OD_SCHEMA_VERSION,
        "kind": "oel.integrated_relative_orbit_determination_report",
        "state_distinctions": {
            "truth_state": "nonlinear deputy and chief ECI propagation",
            "chief_knowledge": "chief state with propagated covariance",
            "relative_knowledge": "RIC observations with sensor plus chief-frame covariance",
            "estimated_relative_state": "model-specific batch or sequential RIC state",
        },
        "chief_orbit": {
            "semi_major_axis_km": float(elements.a_km),
            "eccentricity": float(elements.ecc),
            "mean_motion_rad_s": mean_motion,
        },
        "chief_uncertainty": {
            "initial_covariance": chief_covariance0.tolist(),
            "cross_covariance_supplied": chief_relative_cross_covariance is not None,
            "effective_measurement_covariance_policy": "sensor + projected_chief - cross - cross_transpose",
            "chief_not_treated_as_perfect": bool(np.max(np.abs(chief_covariance0)) > 0.0),
            "initial_sensor_covariance_trace": float(np.trace(sensor_blocks[0])),
            "initial_projected_chief_covariance_trace": float(np.trace(chief_relative_covariances[0])),
            "initial_cross_covariance_trace": float(np.trace(cross_blocks[0])),
            "initial_effective_covariance_trace": float(np.trace(effective_array[0])),
        },
        "observation_count": int(times.size),
        "fit_duration_s": float(fit_duration_s),
        "holdout_count": int(np.count_nonzero(holdout_mask)),
        "nonlinear_baseline": {
            "implementation": "two separately propagated OEL ECI objects transformed to rectangular RIC",
            "integrator": prop.integrator,
            "integration_substep_s": truth.integration_substep_s,
            "truth_source": "supplied" if nonlinear_truth is not None else "generated_in_run",
            "observation_cadence_decoupled_from_integrator_step": truth.integration_substep_s
            is not None,
            "rows": nonlinear_rows,
        },
        "model_applicability": {
            "requested": list(requested_models),
            "evaluated": list(applicable_models),
            "excluded": excluded_models,
        },
        "onp_relative_variational_baseline": {
            "metadata": dict(onp_variational.metadata),
            "terminal_state_ric": onp_variational.relative_state_ric.tolist(),
            "terminal_transition_matrix": onp_variational.transition_matrix.tolist(),
            "terminal_truth_consistency_position_m": float(
                1000.0 * np.linalg.norm(variational_terminal_error[:3])
            ),
            "terminal_truth_consistency_velocity_mm_s": float(
                1.0e6 * np.linalg.norm(variational_terminal_error[3:])
            ),
            "role": "nonlinear ONP local-STM reference; not an averaged relative model",
        },
        "batch_model_comparison": comparison,
        "selected_model_by_withheld_nonlinear_position_rms": selected_model,
        "recommended_model_inside_validated_envelope": recommended_model,
        "recommendation_threshold_m": recommendation_threshold,
        "selection_policy": "rank by withheld nonlinear-baseline error; recommend only when a supplied validated-envelope threshold passes",
        "sequential_model_comparison": sequential,
        "multi_arc": multi_arc,
        "quality_gates": {
            "nonlinear_baseline_reported_for_every_applicable_model": len(nonlinear_rows)
            == len(applicable_models),
            "chief_uncertainty_explicit": True,
            "withheld_selection_used": selected_model is not None,
            "evidence_status": "ready_with_caveats" if selected_model is not None else "review_required",
        },
        "claim_posture": {
            "batch_models": "L2 synthetic nonlinear-baseline characterization",
            "relative_covariance_calibrated": False,
            "note": "A relative covariance claim requires the separate Phase 8 ensemble gate.",
        },
    }
    if backend == "rust":
        report["numeric_backend"] = backend
        report["chief_propagator_numeric_backend"] = prop.numeric_backend
    if output_dir is not None:
        output = Path(output_dir).expanduser()
        artifacts = _write_artifacts(output, report, effective_array, truth.relative_states_ric)
        report["artifacts"] = artifacts
        write_json(str(output / "integrated_relative_od_report.json"), report)
    return report


def _remaining_arc_duration_s(times_s: np.ndarray, fit_duration_s: float) -> float:
    """Return post-fit duration for an arc whose epochs may have a nonzero origin."""

    times = np.asarray(times_s, dtype=float).reshape(-1)
    return float(max((times[-1] - times[0]) - float(fit_duration_s), 0.0))


def _run_relative_filter(
    *,
    model: str,
    times: np.ndarray,
    measurements: np.ndarray,
    covariance_blocks: np.ndarray,
    initial_state: np.ndarray,
    initial_covariance: np.ndarray,
    mean_motion_rad_s: float,
    chief_initial_state: np.ndarray,
    truth: np.ndarray,
    numeric_backend: str = "rust",
) -> dict[str, Any]:
    dt = float(np.median(np.diff(times)))
    noise_diag = np.mean(np.asarray([np.diag(item) for item in covariance_blocks]), axis=0)
    process_noise = np.array([1.0e-10] * 3 + [1.0e-14] * 3)
    if model == "hcw":
        estimator = HCWRelativeEKFEstimator(
            mean_motion_rad_s=mean_motion_rad_s,
            dt_s=dt,
            process_noise_diag=process_noise,
            meas_noise_diag=noise_diag,
            numeric_backend=numeric_backend,
        )
    elif model == "ss_j2":
        estimator = SSJ2RelativeEKFEstimator.from_chief_state(
            chief_initial_state,
            dt_s=dt,
            process_noise_diag=process_noise,
            meas_noise_diag=noise_diag,
            numeric_backend=numeric_backend,
        )
    elif model == "th":
        estimator = THRelativeEKFEstimator(
            chief_state_eci_km_s=chief_initial_state,
            chief_epoch_t_s=float(times[0]),
            dt_s=dt,
            process_noise_diag=process_noise,
            meas_noise_diag=noise_diag,
            numeric_backend=numeric_backend,
        )
    elif model == "ya":
        estimator = YARelativeEKFEstimator(
            chief_state_eci_km_s=chief_initial_state,
            chief_epoch_t_s=float(times[0]),
            dt_s=dt,
            process_noise_diag=process_noise,
            meas_noise_diag=noise_diag,
            numeric_backend=numeric_backend,
        )
    else:
        raise ValueError(f"Unsupported sequential relative model {model!r}.")
    belief = StateBelief(
        state=np.asarray(initial_state, dtype=float),
        covariance=np.asarray(initial_covariance, dtype=float),
        last_update_t_s=float(times[0]),
    )
    records = []
    for index, time_s in enumerate(times):
        estimator.set_measurement_covariance(covariance_blocks[index])
        belief = estimator.update(
            belief,
            Measurement(vector=measurements[index], t_s=float(time_s)),
            float(time_s),
        )
        diagnostics = estimator.last_update_diagnostics
        error = belief.state - truth[index]
        nees = _quadratic(error, belief.covariance)
        records.append(
            {
                "sample_index": index,
                "time_s": float(time_s),
                "state": belief.state.tolist(),
                "truth_state": truth[index].tolist(),
                "covariance": belief.covariance.tolist(),
                "nis": None if diagnostics is None else float(diagnostics.nis),
                "nees": nees,
            }
        )
    covariance_psd = all(
        np.min(np.linalg.eigvalsh(np.asarray(row["covariance"]))) >= -1.0e-14 for row in records
    )
    return {
        "model": model,
        "records": records,
        "position_rms_m": _relative_error_metrics(
            np.asarray([row["state"] for row in records]), truth
        )["position_rms_m"],
        "nis_mean": float(np.nanmean([row["nis"] for row in records])),
        "nees_mean": float(np.mean([row["nees"] for row in records])),
        "covariance_symmetric_psd": covariance_psd,
        "calibrated_covariance": False,
    }


def _run_multi_arc(
    *,
    arc_break_indices: Sequence[int] | None,
    times: np.ndarray,
    measurements: np.ndarray,
    covariance_blocks: np.ndarray,
    model: str | None,
    mean_motion_rad_s: float,
    chief_states: np.ndarray,
    truth: np.ndarray,
    numeric_backend: str = "rust",
) -> dict[str, Any]:
    breaks = sorted({int(item) for item in list(arc_break_indices or []) if 0 < int(item) < len(times)})
    if model is None:
        return {
            "enabled": bool(breaks),
            "break_indices": breaks,
            "arc_count": 0,
            "arcs": [],
            "model": None,
            "status": "skipped_no_selected_model",
            "post_maneuver_policy": "not_applied_no_selected_model",
        }
    bounds = [0, *breaks, len(times)]
    arcs = []
    for arc_index, (start, stop) in enumerate(zip(bounds[:-1], bounds[1:], strict=True)):
        if stop - start < 3:
            arcs.append({"arc_index": arc_index, "status": "insufficient", "start_index": start, "stop_index": stop})
            continue
        arc = make_relative_od_arc(
            times_s=times[start:stop],
            measurements=measurements[start:stop],
            measurement_model="relative_state",
            noise_diag=np.diag(covariance_blocks[start]),
            covariance_blocks=covariance_blocks[start:stop],
        )
        duration = float(times[stop - 1] - times[start])
        result = solve_batch_relative_od(
            arc,
            dynamics_model=model,
            initial_guess_ric=measurements[start],
            fit_duration_s=0.7 * duration,
            holdout_duration_s=0.3 * duration,
            mean_motion_rad_s=mean_motion_rad_s,
            chief_initial_state_eci_km_s=chief_states[start],
            truth_initial_relative_state_ric=truth[start],
            numeric_backend=numeric_backend,
        )
        arcs.append(
            {
                "arc_index": arc_index,
                "model": model,
                "status": "reinitialized_and_fit",
                "start_index": start,
                "stop_index": stop,
                "reinitialization_source": "first measurement of arc",
                "estimated_initial_state_ric": result["estimated_initial_state_ric"],
                "quality_gates": result["quality_gates"],
            }
        )
    return {
        "enabled": bool(breaks),
        "break_indices": breaks,
        "arc_count": len(arcs),
        "arcs": arcs,
        "model": model,
        "post_maneuver_policy": "explicit new arc and measurement reinitialization",
    }


def _propagate_state_and_covariance(
    *,
    initial_state: np.ndarray,
    initial_covariance: np.ndarray,
    times: np.ndarray,
    propagator: OrbitPropagator,
    context: OrbitContext,
    environment: dict[str, Any],
    integration_substep_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    states = [initial_state]
    covariances = [initial_covariance]
    for index in range(1, len(times)):
        dt = float(times[index] - times[index - 1])
        previous = states[-1]
        predicted = _propagate_interval(
            propagator,
            previous,
            times[index - 1],
            dt,
            context,
            environment,
            integration_substep_s,
        )
        transition = _numerical_transition(
            propagator,
            previous,
            predicted,
            times[index - 1],
            dt,
            context,
            environment,
            integration_substep_s,
        )
        states.append(predicted)
        covariances.append(_project_covariance(transition @ covariances[-1] @ transition.T))
    return np.asarray(states), np.asarray(covariances)


def _numerical_transition(
    propagator: OrbitPropagator,
    state: np.ndarray,
    base: np.ndarray,
    t_s: float,
    dt_s: float,
    context: OrbitContext,
    environment: dict[str, Any],
    integration_substep_s: float,
) -> np.ndarray:
    steps = np.array([1.0e-3] * 3 + [1.0e-6] * 3)
    if propagator.numeric_backend == "rust":
        from sim.rust_relative_backend import try_orbit_trial_batch

        widths = []
        elapsed = 0.0
        while elapsed < float(dt_s) - 1.0e-12:
            width = min(float(integration_substep_s), float(dt_s) - elapsed)
            widths.append(width)
            elapsed += width
        trials = try_orbit_trial_batch(propagator, state + np.diag(steps), widths, context)
        if trials is not None:
            return ((trials - base) / steps[:, None]).T
    transition = np.zeros((6, 6))
    for axis, step in enumerate(steps):
        perturbed = state.copy()
        perturbed[axis] += step
        transition[:, axis] = (
            _propagate_interval(
                propagator,
                perturbed,
                t_s,
                dt_s,
                context,
                environment,
                integration_substep_s,
            )
            - base
        ) / step
    return transition


def _relative_jacobian_wrt_chief(
    deputy: np.ndarray,
    chief: np.ndarray,
    *,
    chief_accel_eci_km_s2: np.ndarray | None = None,
) -> np.ndarray:
    """Finite-difference the RIC transform at a fixed nominal force context.

    When the ONP force model supplies a chief acceleration, the same nominal
    acceleration is held for each state perturbation.  This differentiates the
    acceleration-aware RIC map used by integrated OD's chief-uncertainty
    projection without claiming a force-model state Jacobian for arbitrary
    plugins.  Omitting it preserves the historical r/v-only fixed-plane map.
    """

    base = eci_relative_to_ric_rect(
        deputy,
        chief,
        chief_accel_eci_km_s2=chief_accel_eci_km_s2,
    )
    steps = np.array([1.0e-3] * 3 + [1.0e-6] * 3)
    jacobian = np.zeros((6, 6))
    for axis, step in enumerate(steps):
        perturbed = chief.copy()
        perturbed[axis] += step
        jacobian[:, axis] = (
            eci_relative_to_ric_rect(
                deputy,
                perturbed,
                chief_accel_eci_km_s2=chief_accel_eci_km_s2,
            )
            - base
        ) / step
    return jacobian


def _relative_error_metrics(prediction: np.ndarray, truth: np.ndarray) -> dict[str, Any]:
    if prediction.size == 0:
        return {"available": False, "position_rms_m": None, "velocity_rms_mm_s": None}
    error = np.asarray(prediction) - np.asarray(truth)
    return {
        "available": True,
        "position_rms_m": float(1000.0 * np.sqrt(np.mean(np.sum(error[:, :3] ** 2, axis=1)))),
        "velocity_rms_mm_s": float(1.0e6 * np.sqrt(np.mean(np.sum(error[:, 3:] ** 2, axis=1)))),
        "position_max_m": float(1000.0 * np.max(np.linalg.norm(error[:, :3], axis=1))),
    }


def _write_artifacts(
    output: Path,
    report: Mapping[str, Any],
    covariance_blocks: np.ndarray,
    nonlinear_truth: np.ndarray,
) -> dict[str, str]:
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "integrated_relative_od_report.json"
    review_path = output / "review/run.sqlite"
    review_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(review_path) as connection:
        connection.executescript(
            """
            DROP TABLE IF EXISTS integrated_relative_od_run;
            DROP TABLE IF EXISTS relative_od_models;
            DROP TABLE IF EXISTS relative_od_samples;
            DROP TABLE IF EXISTS relative_od_arcs;
            CREATE TABLE integrated_relative_od_run (
                schema_version INTEGER NOT NULL, selected_model TEXT,
                observation_count INTEGER NOT NULL, holdout_count INTEGER NOT NULL,
                evidence_status TEXT NOT NULL, report_json TEXT NOT NULL
            );
            CREATE TABLE relative_od_models (
                model TEXT PRIMARY KEY, fit_position_rms_m REAL,
                holdout_position_rms_m REAL, batch_solver_success INTEGER NOT NULL,
                sequential_position_rms_m REAL, evidence_json TEXT NOT NULL
            );
            CREATE TABLE relative_od_samples (
                sample_index INTEGER PRIMARY KEY, truth_state_json TEXT NOT NULL,
                effective_covariance_json TEXT NOT NULL
            );
            CREATE TABLE relative_od_arcs (
                arc_index INTEGER PRIMARY KEY, status TEXT NOT NULL,
                start_index INTEGER, stop_index INTEGER, evidence_json TEXT NOT NULL
            );
            """
        )
        connection.execute(
            "INSERT INTO integrated_relative_od_run VALUES (?, ?, ?, ?, ?, ?)",
            (
                int(report["schema_version"]),
                report["selected_model_by_withheld_nonlinear_position_rms"],
                int(report["observation_count"]),
                int(report["holdout_count"]),
                str(report["quality_gates"]["evidence_status"]),
                str(report_path),
            ),
        )
        nonlinear_by_model = {
            row["model"]: row for row in report["nonlinear_baseline"]["rows"]
        }
        connection.executemany(
            "INSERT INTO relative_od_models VALUES (?, ?, ?, ?, ?, ?)",
            [
                (
                    model,
                    nonlinear_by_model[model]["fit"]["position_rms_m"],
                    nonlinear_by_model[model]["holdout"]["position_rms_m"],
                    int(bool(model_report["solver"]["success"])),
                    report["sequential_model_comparison"][model]["position_rms_m"],
                    json.dumps(model_report, sort_keys=True),
                )
                for model, model_report in report["batch_model_comparison"]["reports"].items()
            ],
        )
        connection.executemany(
            "INSERT INTO relative_od_samples VALUES (?, ?, ?)",
            [
                (index, json.dumps(state.tolist()), json.dumps(covariance.tolist()))
                for index, (state, covariance) in enumerate(
                    zip(nonlinear_truth, covariance_blocks, strict=True)
                )
            ],
        )
        connection.executemany(
            "INSERT INTO relative_od_arcs VALUES (?, ?, ?, ?, ?)",
            [
                (
                    int(row["arc_index"]),
                    str(row["status"]),
                    row.get("start_index"),
                    row.get("stop_index"),
                    json.dumps(row, sort_keys=True),
                )
                for row in report["multi_arc"]["arcs"]
            ],
        )
    write_json(str(report_path), dict(report))
    return {"report_json": str(report_path), "review_sqlite": str(review_path)}


def _propagate(
    propagator: OrbitPropagator,
    state: np.ndarray,
    t_s: float,
    dt_s: float,
    context: OrbitContext,
    environment: dict[str, Any],
) -> np.ndarray:
    return propagator.propagate(
        np.asarray(state, dtype=float),
        float(dt_s),
        float(t_s),
        np.zeros(3),
        dict(environment),
        context,
    )


def _propagate_interval(
    propagator: OrbitPropagator,
    state: np.ndarray,
    t_s: float,
    dt_s: float,
    context: OrbitContext,
    environment: dict[str, Any],
    integration_substep_s: float,
) -> np.ndarray:
    current = np.asarray(state, dtype=float).copy()
    elapsed = 0.0
    while elapsed < float(dt_s) - 1.0e-12:
        step = min(float(integration_substep_s), float(dt_s) - elapsed)
        current = _propagate(
            propagator, current, float(t_s) + elapsed, step, context, environment
        )
        elapsed += step
    return current


def _chief_acceleration_or_none(
    propagator: OrbitPropagator,
    chief_state: np.ndarray,
    t_s: float,
    context: OrbitContext,
    environment: Mapping[str, Any],
) -> np.ndarray | None:
    """Resolve the configured ONP force model for an ECI chief state."""

    if propagator.state_frame != "eci":
        return None
    return np.asarray(
        propagator.acceleration_at(
            t_s=float(t_s),
            x_eci=np.asarray(chief_state, dtype=float).reshape(6),
            command_accel_eci_km_s2=np.zeros(3, dtype=float),
            env=dict(environment),
            ctx=context,
        ),
        dtype=float,
    ).reshape(3)


def _covariance_blocks(
    value: Sequence[Sequence[float]] | np.ndarray,
    *,
    count: int,
    allow_indefinite: bool = False,
) -> np.ndarray:
    covariance = np.asarray(value, dtype=float)
    if covariance.shape == (6, 6):
        covariance = np.broadcast_to(covariance, (count, 6, 6)).copy()
    if covariance.shape != (count, 6, 6) or np.any(~np.isfinite(covariance)):
        raise ValueError("covariance must have shape (6, 6) or (observation_count, 6, 6).")
    if not allow_indefinite:
        for item in covariance:
            if not np.allclose(item, item.T, rtol=1.0e-10, atol=1.0e-14):
                raise ValueError("covariance blocks must be symmetric.")
            _require_positive_semidefinite(item, "covariance blocks must be positive semidefinite.")
    return covariance


def _state6(value: Sequence[float] | np.ndarray, name: str) -> np.ndarray:
    state = np.asarray(value, dtype=float).reshape(-1)
    if state.size != 6 or np.any(~np.isfinite(state)):
        raise ValueError(f"{name} must contain six finite values.")
    return state


def _positive_finite(value: float, name: str) -> float:
    normalized = float(value)
    if not np.isfinite(normalized) or normalized <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return normalized


def _covariance6(value: Sequence[Sequence[float]] | np.ndarray, name: str) -> np.ndarray:
    covariance = np.asarray(value, dtype=float)
    if covariance.shape != (6, 6) or np.any(~np.isfinite(covariance)):
        raise ValueError(f"{name} must be a finite 6x6 matrix.")
    if not np.allclose(covariance, covariance.T, rtol=1.0e-10, atol=1.0e-14):
        raise ValueError(f"{name} must be symmetric.")
    _require_positive_semidefinite(covariance, f"{name} must be positive semidefinite.")
    return covariance


def _require_positive_semidefinite(covariance: np.ndarray, message: str) -> None:
    symmetric = 0.5 * (covariance + covariance.T)
    eigenvalues = np.linalg.eigvalsh(symmetric)
    scale = float(np.max(np.abs(eigenvalues)))
    if eigenvalues[0] < -1.0e-12 * scale:
        raise ValueError(message)


def _project_covariance(value: np.ndarray, floor: float = 1.0e-18) -> np.ndarray:
    symmetric = 0.5 * (np.asarray(value) + np.asarray(value).T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    return (eigenvectors * np.maximum(eigenvalues, floor)) @ eigenvectors.T


def _quadratic(vector: np.ndarray, covariance: np.ndarray) -> float:
    try:
        solved = np.linalg.solve(covariance, vector)
    except np.linalg.LinAlgError:
        solved = np.linalg.pinv(covariance) @ vector
    return float(vector.T @ solved)
