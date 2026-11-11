# ruff: noqa: F401,F403,F405,I001
from .orbit_od_common import *
from .orbit_od_artifacts import *

def build_dynamics_od_quality_gates(
    *,
    solver: Mapping[str, Any],
    parameter_metadata: list[dict[str, Any]],
    covariance: np.ndarray | None,
    prefit_position_rms_m: float,
    fit_metrics: Mapping[str, Any],
    holdout_metrics: Mapping[str, Any],
    selected_parameters: Sequence[str] | None = None,
    attitude_source: str = "none",
    attitude_history_rows: Sequence[Mapping[str, Any]] | None = None,
    base_specs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    fit_rms = float(fit_metrics.get("position_rms_m", np.nan))
    fit_max = float(fit_metrics.get("position_max_m", np.nan))
    holdout_rms = float(holdout_metrics.get("position_rms_m", np.nan))
    holdout_max = float(holdout_metrics.get("position_max_m", np.nan))
    holdout_final = float(holdout_metrics.get("final_position_error_m", np.nan))
    fit_improvement_ratio = _ratio(float(prefit_position_rms_m), fit_rms)
    holdout_degradation_ratio = _ratio(holdout_rms, fit_rms)
    bounds = _parameter_bound_hits(parameter_metadata)
    covariance_valid = _is_covariance_valid(covariance)
    solver_success = bool(solver.get("success", False))
    solver_diagnostics = dict(solver.get("diagnostics", {}) or {})
    observability = dict(solver_diagnostics.get("observability", {}) or {})
    data_full_rank = bool(observability.get("data_full_rank", False))
    parameter_observability = {
        str(row.get("parameter")): bool(row.get("identifiable", False))
        for row in list(observability.get("parameters", []) or [])
    }
    fit_improved = bool(
        np.isfinite(fit_rms) and np.isfinite(float(prefit_position_rms_m)) and fit_rms <= float(prefit_position_rms_m)
    )
    selected = list(selected_parameters or [])
    attitude_rows = list(attitude_history_rows or [])
    specs = dict(base_specs or {})
    warnings: list[str] = []
    if not solver_success:
        warnings.append("solver did not report success")
    if not data_full_rank:
        warnings.append("data Jacobian is rank deficient; solution is not fully observable")
    if not fit_improved:
        warnings.append("dynamics fit did not improve relative to the initial-guess RMS")
    if not covariance_valid:
        warnings.append("parameter covariance is missing or invalid")
    if bounds:
        warnings.append("one or more estimated parameters are at optimizer bounds")
    holdout_evaluated = int(holdout_metrics.get("sample_count", 0) or 0) > 0
    if holdout_degradation_ratio is not None and holdout_degradation_ratio > 10.0:
        warnings.append("holdout RMS is more than 10x the fit RMS")
    if not holdout_evaluated:
        warnings.append("holdout not evaluated")
    elif not np.isfinite(holdout_rms):
        warnings.append("holdout RMS is not finite")
    if "cd_scale" in selected:
        if str(attitude_source or "none") == "none":
            warnings.append(
                "cd_scale was estimated without attitude-aware area; result is equivalent to a ballistic drag scale"
            )
        if "drag_scale" in selected:
            warnings.append("cd_scale and drag_scale are both selected; Cd and area scale are strongly coupled")
        if not _spec_has_geometry_profile(specs):
            warnings.append(
                "cd_scale was estimated without a geometry area profile; projected area is not attitude-dependent"
            )
        if str(attitude_source or "none") in {"observed_history", "modeled_replay"} and len(attitude_rows) < 2:
            warnings.append("attitude replay source has fewer than two samples")
        if attitude_rows and _projected_attitude_variation_warning(attitude_rows):
            warnings.append("attitude history has limited quaternion variation; Cd may be weakly observable")
    optional_parameter_names = {
        "drag_scale": "drag_scale",
        "cd_scale": "cd_scale",
        "srp_scale": "srp_scale",
    }
    unidentifiable_optional_parameters = [
        name
        for selection, name in optional_parameter_names.items()
        if selection in selected and not parameter_observability.get(name, False)
    ]
    if unidentifiable_optional_parameters:
        warnings.append(
            "optional parameters are not identifiable from the fit arc: "
            + ", ".join(unidentifiable_optional_parameters)
        )
    holdout_acceptable = bool(
        holdout_evaluated and np.isfinite(holdout_rms) and not any("holdout" in warning.lower() for warning in warnings)
    )
    return {
        "schema_version": 1,
        "evidence_status": "ready_with_caveats" if not warnings else "review_required",
        "solver_success": solver_success,
        "fit_improved_prefit_rms": fit_improved,
        "covariance_valid": covariance_valid,
        "data_full_rank": data_full_rank,
        "unidentifiable_optional_parameters": unidentifiable_optional_parameters,
        "failure_classification": solver_diagnostics.get("failure_classification"),
        "holdout_acceptable": holdout_acceptable,
        "holdout_evidence_status": "evaluated" if holdout_evaluated else "not_evaluated",
        "parameter_bounds_hit": bounds,
        "prefit_position_rms_m": float(prefit_position_rms_m),
        "fit_position_rms_m": fit_rms,
        "fit_position_max_m": fit_max,
        "holdout_position_rms_m": holdout_rms,
        "holdout_position_max_m": holdout_max,
        "holdout_final_position_error_m": holdout_final,
        "fit_improvement_ratio": fit_improvement_ratio,
        "holdout_degradation_ratio": holdout_degradation_ratio,
        "warnings": warnings,
        "non_claims": [
            "This is OEL-dynamics least-squares OD evidence, not mission-assurance certification.",
            "The result is bounded by the observation source, frame assumptions, force model, estimated parameters, and fit/holdout windows.",
            "A low fit residual without acceptable holdout behavior should be treated as overfitting or model mismatch.",
        ],
    }


def _dynamics_od_verdict(quality_gates: Mapping[str, Any]) -> dict[str, Any]:
    warnings = [str(item) for item in list(quality_gates.get("warnings", []) or [])]
    bounds = list(quality_gates.get("parameter_bounds_hit", []) or [])
    solver_success = bool(quality_gates.get("solver_success", False))
    fit_improved = bool(quality_gates.get("fit_improved_prefit_rms", False))
    covariance_valid = bool(quality_gates.get("covariance_valid", False))
    holdout_acceptable = bool(quality_gates.get("holdout_acceptable", False))
    if solver_success and fit_improved and holdout_acceptable and covariance_valid and not bounds and not warnings:
        action = "usable_for_propagation_study"
        summary = "Dynamics OD fit improved the initial guess and holdout behavior is reviewable."
    elif not solver_success or not fit_improved:
        action = "review_solver_initial_guess_and_observation_arc"
        summary = "The fit did not converge cleanly or did not improve the initial guess."
    elif not holdout_acceptable:
        action = "review_force_model_or_extend_observation_arc"
        summary = "Fit residuals are not enough; holdout behavior needs analyst review."
    elif bounds:
        action = "review_estimated_parameter_bounds"
        summary = "One or more estimated parameters ended at optimizer bounds."
    elif not covariance_valid:
        action = "review_covariance_before_uncertainty_claims"
        summary = "State fit may be usable, but covariance is not strong enough for uncertainty claims."
    else:
        action = "review_warnings_before_use"
        summary = "The run produced OD artifacts, but warnings should be resolved before relying on it."
    return {
        "evidence_status": str(quality_gates.get("evidence_status", "unknown")),
        "summary": summary,
        "analyst_action": action,
        "solver_success": solver_success,
        "fit_improved_prefit_rms": fit_improved,
        "holdout_acceptable": holdout_acceptable,
        "covariance_valid": covariance_valid,
        "parameter_bounds_hit_count": len(bounds),
        "warning_count": len(warnings),
    }

__all__ = [name for name in globals() if not name.startswith("__")]
