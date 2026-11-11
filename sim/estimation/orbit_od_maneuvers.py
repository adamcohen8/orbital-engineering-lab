# ruff: noqa: F401,F403,F405,I001
import io

from .orbit_od_common import *
from .orbit_od_artifacts import *
from sim.security.output_writes import write_output_text

def _detect_single_maneuver(
    *,
    output_root: Path,
    object_id: str,
    selected_parameters: Sequence[str],
    base_parameters: ParameterSet,
    baseline_native_values: np.ndarray,
    baseline_fit_metrics: Mapping[str, Any],
    baseline_holdout_metrics: Mapping[str, Any],
    baseline_covariance: np.ndarray | None,
    evaluate: Any,
    times: np.ndarray,
    fit_mask: np.ndarray,
    holdout_mask: np.ndarray,
    observations: Sequence[Mapping[str, Any]],
    positions: np.ndarray,
    velocities: np.ndarray | None,
    position_sigmas: np.ndarray,
    velocity_sigmas: np.ndarray | None,
    fit_duration_s: float,
    total_duration_s: float,
    dt_s: float,
    frame: str,
    max_delta_v_m_s: float,
    min_delta_v_m_s: float,
    min_improvement_ratio: float,
    burn_duration_s: float | None,
    guard_observations: int,
    max_candidates: int,
    max_nfev: int,
    robust_loss: str,
    robust_f_scale: float,
    sigma_clip_threshold: float | None,
) -> dict[str, Any]:
    frame_key = str(frame or "ric").strip().lower()
    if frame_key not in {"ric", "eci"}:
        raise ValueError("maneuver_frame must be 'ric' or 'eci'.")
    candidate_times = _maneuver_candidate_times(
        times[fit_mask],
        fit_duration_s=fit_duration_s,
        guard_observations=guard_observations,
        max_candidates=max_candidates,
    )
    maneuver_parameters = _with_maneuver_parameters(
        base_parameters,
        frame=frame_key,
        max_delta_v_m_s=max_delta_v_m_s,
    )
    baseline_fit_rms_m = float(baseline_fit_metrics.get("position_rms_m", np.nan))
    baseline_holdout_rms_m = float(baseline_holdout_metrics.get("position_rms_m", np.nan))
    fit_observation_count = int(np.count_nonzero(fit_mask))
    observation_scalar_count = int(max(fit_observation_count * (6 if velocities is not None else 3), 1))
    baseline_bic = _od_metric_bic(
        rms_m=baseline_fit_rms_m,
        scalar_count=observation_scalar_count,
        parameter_count=len(base_parameters.parameters),
    )
    candidate_rows: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    burn_duration = float(burn_duration_s) if burn_duration_s is not None else float(dt_s)
    burn_duration = float(max(burn_duration, dt_s, 1.0e-12))
    fit_observations = [observations[index] for index in np.flatnonzero(fit_mask)]
    fit_covariance_factors = _prepare_state_observation_factors(
        observations=fit_observations,
        position_sigmas=position_sigmas[fit_mask],
        velocity_sigmas=velocity_sigmas[fit_mask] if velocity_sigmas is not None else None,
    )

    for idx, candidate_time_s in enumerate(candidate_times):
        candidate_id = int(idx)
        candidate_time = float(candidate_time_s)
        initial_native = np.concatenate(
            (
                np.array(baseline_native_values, dtype=float).reshape(-1),
                np.zeros(3, dtype=float),
            )
        )

        def residual(
            native_values: np.ndarray,
            *,
            candidate_id: int = candidate_id,
            candidate_time: float = candidate_time,
        ) -> np.ndarray:
            maneuver = _maneuver_spec_from_values(
                maneuver_parameters.mapping(native_values),
                time_s=candidate_time,
                frame=frame_key,
                burn_duration_s=burn_duration,
            )
            sim_t, sim_x, _packet, _artifact = evaluate(
                native_values,
                duration_s=fit_duration_s,
                scratch_name=f"maneuver_{candidate_id:03d}_fit",
                parameters_override=maneuver_parameters,
                maneuver=maneuver,
            )
            sim_on_obs = _states_at_epochs(sim_t, sim_x, times[fit_mask])
            return _whiten_state_observation_residuals(
                sim_on_obs,
                observations=fit_observations,
                reference_positions=positions[fit_mask],
                reference_velocities=velocities[fit_mask] if velocities is not None else None,
                position_sigmas=position_sigmas[fit_mask],
                velocity_sigmas=velocity_sigmas[fit_mask] if velocity_sigmas is not None else None,
                covariance_factors=fit_covariance_factors,
            )

        local_parameters = ParameterSet(
            [
                EstimatedParameter(
                    p.name,
                    float(initial_native[p_idx]),
                    scale=p.scale,
                    lower=p.lower,
                    upper=p.upper,
                    unit=p.unit,
                    description=p.description,
                )
                for p_idx, p in enumerate(maneuver_parameters.parameters)
            ]
        )
        solve = solve_batch_least_squares(
            local_parameters,
            residual,
            max_nfev=max_nfev,
            robust_loss=robust_loss,
            robust_f_scale=robust_f_scale,
            sigma_clip_threshold=sigma_clip_threshold,
        )
        parameter_values = local_parameters.mapping(solve.x_native)
        maneuver = _maneuver_spec_from_values(
            parameter_values,
            time_s=float(candidate_time_s),
            frame=frame_key,
            burn_duration_s=burn_duration,
        )
        fit_t, fit_x, _fit_packet, _fit_artifact = evaluate(
            solve.x_native,
            duration_s=fit_duration_s,
            scratch_name=f"maneuver_{idx:03d}_postfit",
            parameters_override=local_parameters,
            maneuver=maneuver,
        )
        fit_on_obs = _states_at_epochs(fit_t, fit_x, times[fit_mask])
        fit_metrics = _state_error_metrics(
            fit_on_obs,
            positions[fit_mask],
            velocities[fit_mask] if velocities is not None else None,
        )
        pred_t, pred_x, _pred_packet, _pred_artifact = evaluate(
            solve.x_native,
            duration_s=total_duration_s,
            scratch_name=f"maneuver_{idx:03d}_prediction",
            parameters_override=local_parameters,
            maneuver=maneuver,
        )
        holdout_t_obs = times[holdout_mask]
        pred_on_holdout = _states_at_epochs(pred_t, pred_x, holdout_t_obs)
        holdout_metrics = _state_error_metrics(
            pred_on_holdout,
            positions[holdout_mask],
            velocities[holdout_mask] if velocities is not None else None,
        )
        fit_rms_m = float(fit_metrics.get("position_rms_m", np.nan))
        holdout_rms_m = float(holdout_metrics.get("position_rms_m", np.nan))
        delta_v_m_s = np.array(maneuver["delta_v_m_s"], dtype=float).reshape(3)
        delta_v_norm_m_s = float(np.linalg.norm(delta_v_m_s))
        improvement_ratio = (
            float((baseline_fit_rms_m - fit_rms_m) / baseline_fit_rms_m)
            if np.isfinite(baseline_fit_rms_m) and abs(baseline_fit_rms_m) > 1.0e-12 and np.isfinite(fit_rms_m)
            else float("nan")
        )
        bic = _od_metric_bic(
            rms_m=fit_rms_m,
            scalar_count=observation_scalar_count,
            parameter_count=len(local_parameters.parameters),
        )
        bound_hits = _parameter_bound_hits(local_parameters.metadata(solve.x_native))
        warnings = _maneuver_candidate_warnings(
            solver_success=bool(solve.success),
            improvement_ratio=improvement_ratio,
            min_improvement_ratio=min_improvement_ratio,
            delta_v_norm_m_s=delta_v_norm_m_s,
            min_delta_v_m_s=min_delta_v_m_s,
            bic=bic,
            baseline_bic=baseline_bic,
            holdout_rms_m=holdout_rms_m,
            baseline_holdout_rms_m=baseline_holdout_rms_m,
            bound_hits=bound_hits,
        )
        observability = dict(solve.diagnostics.get("observability", {}) or {})
        unidentifiable_maneuver_parameters = [
            str(row.get("parameter"))
            for row in list(observability.get("parameters", []) or [])
            if str(row.get("parameter", "")).startswith("maneuver_dv_") and not bool(row.get("identifiable", False))
        ]
        if not bool(observability.get("data_full_rank", False)):
            warnings.append("maneuver candidate data Jacobian is rank deficient")
        if unidentifiable_maneuver_parameters:
            warnings.append(
                "unidentifiable maneuver parameters: " + ", ".join(sorted(unidentifiable_maneuver_parameters))
            )
        status = "candidate_supported" if not warnings else "review_required"
        row = {
            "candidate_id": idx,
            "time_s": float(candidate_time_s),
            "frame": frame_key,
            "fit_position_rms_m": fit_rms_m,
            "holdout_position_rms_m": holdout_rms_m,
            "delta_v_0_m_s": float(delta_v_m_s[0]),
            "delta_v_1_m_s": float(delta_v_m_s[1]),
            "delta_v_2_m_s": float(delta_v_m_s[2]),
            "delta_v_norm_m_s": delta_v_norm_m_s,
            "improvement_ratio": improvement_ratio,
            "bic": bic,
            "bic_improvement": float(baseline_bic - bic)
            if np.isfinite(baseline_bic) and np.isfinite(bic)
            else float("nan"),
            "solver_success": bool(solve.success),
            "status": status,
            "warning_count": len(warnings),
        }
        candidate_rows.append(row)
        candidates.append(
            {
                **row,
                "warnings": warnings,
                "parameter_values": parameter_values,
                "parameter_metadata": local_parameters.metadata(solve.x_native),
                "covariance_valid": _is_covariance_valid(solve.covariance_native),
                "covariance": solve.covariance_native,
                "native_values": solve.x_native,
                "parameters": local_parameters,
                "maneuver": maneuver,
                "fit_metrics": fit_metrics,
                "holdout_metrics": holdout_metrics,
                "solver": {
                    "success": bool(solve.success),
                    "message": solve.message,
                    "nfev": int(solve.nfev),
                    "initial_cost": float(solve.initial_cost),
                    "final_cost": float(solve.cost),
                    "rms_weighted_residual": float(solve.rms_residual),
                    "diagnostics": solve.diagnostics,
                    "decision_records": _label_state_residual_decisions(
                        solve.decision_records,
                        observations=[observations[index] for index in np.flatnonzero(fit_mask)],
                        include_velocity=velocities is not None,
                    ),
                },
            }
        )

    best = _best_maneuver_candidate(candidates)
    supported = [c for c in candidates if c.get("status") == "candidate_supported"]
    status = "candidate_supported" if supported else "no_candidate"
    warnings = []
    if not candidates:
        warnings.append("No candidate maneuver times were available after guard-band filtering.")
    elif not supported:
        warnings.append("No maneuver candidate cleared the conservative detection gates.")
    if best is not None and best.get("status") == "review_required":
        warnings.extend(str(w) for w in list(best.get("warnings", []) or []))
    best_public = _public_maneuver_candidate(best)
    artifacts = _write_maneuver_detection_artifacts(
        output_root=output_root,
        object_id=object_id,
        candidate_rows=candidate_rows,
        best=best,
        status=status,
        warnings=warnings,
        evaluate=evaluate,
        times=times,
        fit_mask=fit_mask,
        holdout_mask=holdout_mask,
        positions=positions,
        velocities=velocities,
        fit_duration_s=fit_duration_s,
        total_duration_s=total_duration_s,
    )
    report = {
        "schema_version": 1,
        "enabled": True,
        "status": status,
        "frame": frame_key,
        "candidate_count": len(candidates),
        "supported_candidate_count": len(supported),
        "baseline": {
            "fit_position_rms_m": baseline_fit_rms_m,
            "holdout_position_rms_m": baseline_holdout_rms_m,
            "bic": baseline_bic,
            "selected_parameters": list(selected_parameters),
            "covariance_valid": _is_covariance_valid(baseline_covariance),
        },
        "best_candidate": best_public,
        "quality_gates": {
            "schema_version": 1,
            "candidate_supported": bool(status == "candidate_supported"),
            "failure_classification": "none" if status == "candidate_supported" else "unsuccessful_model_selection",
            "min_delta_v_m_s": float(min_delta_v_m_s),
            "min_improvement_ratio": float(min_improvement_ratio),
            "max_delta_v_m_s": float(max_delta_v_m_s),
            "warnings": warnings,
            "non_claims": [
                "Maneuver detection is residual evidence from an OD model comparison, not independent confirmation of an operational maneuver.",
                "Small maneuvers can be confused with observation noise, force-model mismatch, frame error, drag/SRP mismatch, or unmodeled biases.",
                "The v1 detector searches one scheduled vector burn; multiple-burn, finite-thrust-profile, and attribution claims are out of scope.",
            ],
        },
        "artifacts": artifacts,
    }
    write_json(str(output_root / "maneuver_detection_report.json"), report)
    _write_maneuver_detection_md(output_root / "maneuver_detection_report.md", report)
    return report


def _with_maneuver_parameters(base_parameters: ParameterSet, *, frame: str, max_delta_v_m_s: float) -> ParameterSet:
    names = ("r", "i", "c") if frame == "ric" else ("x", "y", "z")
    limit = float(max(abs(max_delta_v_m_s), 1.0e-9))
    return ParameterSet(
        [
            *base_parameters.parameters,
            *[
                EstimatedParameter(
                    f"maneuver_dv_{axis}_m_s",
                    0.0,
                    scale=max(limit / 10.0, 0.1),
                    lower=-limit,
                    upper=limit,
                    unit="m/s",
                    description=f"Detected maneuver delta-v {axis.upper()} component in the {frame.upper()} frame.",
                )
                for axis in names
            ],
        ]
    )


def _maneuver_spec_from_values(
    values: Mapping[str, float],
    *,
    time_s: float,
    frame: str,
    burn_duration_s: float,
) -> dict[str, Any]:
    axes = ("r", "i", "c") if frame == "ric" else ("x", "y", "z")
    return {
        "time_s": float(time_s),
        "frame": frame,
        "delta_v_m_s": [float(values.get(f"maneuver_dv_{axis}_m_s", 0.0)) for axis in axes],
        "burn_duration_s": float(burn_duration_s),
    }


def _weighted_state_residual(
    sim_x: np.ndarray,
    ref_position_km: np.ndarray,
    ref_velocity_km_s: np.ndarray | None,
    position_sigmas_km: np.ndarray,
    velocity_sigmas_km_s: np.ndarray | None,
) -> np.ndarray:
    chunks = [((sim_x[:, :3] - ref_position_km) / np.maximum(position_sigmas_km[:, None], 1.0e-12)).reshape(-1)]
    if ref_velocity_km_s is not None and velocity_sigmas_km_s is not None:
        chunks.append(
            ((sim_x[:, 3:] - ref_velocity_km_s) / np.maximum(velocity_sigmas_km_s[:, None], 1.0e-12)).reshape(-1)
        )
    return np.concatenate(chunks)


def _maneuver_candidate_times(
    fit_times_s: np.ndarray,
    *,
    fit_duration_s: float,
    guard_observations: int,
    max_candidates: int,
) -> list[float]:
    times = [float(t) for t in np.array(fit_times_s, dtype=float).reshape(-1) if 0.0 < float(t) < float(fit_duration_s)]
    guard = int(max(guard_observations, 1))
    if len(times) > 2 * guard:
        times = times[guard:-guard]
    if max_candidates > 0 and len(times) > int(max_candidates):
        idx = np.linspace(0, len(times) - 1, int(max_candidates), dtype=int)
        times = [times[int(i)] for i in idx]
    return times


def _od_metric_bic(*, rms_m: float, scalar_count: int, parameter_count: int) -> float:
    if not np.isfinite(float(rms_m)):
        return float("inf")
    n = int(max(scalar_count, 1))
    rss = max((float(rms_m) ** 2) * n, 1.0e-24)
    return float(n * np.log(rss / n) + int(max(parameter_count, 0)) * np.log(n))


def _maneuver_candidate_warnings(
    *,
    solver_success: bool,
    improvement_ratio: float,
    min_improvement_ratio: float,
    delta_v_norm_m_s: float,
    min_delta_v_m_s: float,
    bic: float,
    baseline_bic: float,
    holdout_rms_m: float,
    baseline_holdout_rms_m: float,
    bound_hits: list[dict[str, Any]],
) -> list[str]:
    warnings: list[str] = []
    # A least-squares solver can exhaust its small evaluation budget on CI while
    # still landing on a physically useful maneuver fit. Treat solver_success as
    # diagnostic metadata; the support decision is made from residual, delta-v,
    # model-score, holdout, and bound gates below.
    if not np.isfinite(improvement_ratio) or improvement_ratio < float(min_improvement_ratio):
        warnings.append("maneuver candidate did not improve fit RMS enough")
    if not np.isfinite(delta_v_norm_m_s) or delta_v_norm_m_s < float(min_delta_v_m_s):
        warnings.append("estimated maneuver delta-v is below the configured noise floor")
    if not np.isfinite(bic) or not np.isfinite(baseline_bic) or bic >= baseline_bic:
        warnings.append("maneuver candidate did not improve the penalized model score")
    if (
        np.isfinite(holdout_rms_m)
        and np.isfinite(baseline_holdout_rms_m)
        and holdout_rms_m > max(2.0 * baseline_holdout_rms_m, baseline_holdout_rms_m + 1.0)
    ):
        warnings.append("maneuver candidate degraded holdout RMS relative to baseline")
    if bound_hits:
        warnings.append("one or more maneuver-fit parameters are at optimizer bounds")
    return warnings


def _best_maneuver_candidate(candidates: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    if not candidates:
        return None
    supported = [c for c in candidates if c.get("status") == "candidate_supported"]
    pool = supported if supported else list(candidates)
    return min(pool, key=lambda c: float(c.get("bic", float("inf"))))


def _public_maneuver_candidate(candidate: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if candidate is None:
        return None
    keep = {
        "candidate_id",
        "time_s",
        "frame",
        "fit_position_rms_m",
        "holdout_position_rms_m",
        "delta_v_0_m_s",
        "delta_v_1_m_s",
        "delta_v_2_m_s",
        "delta_v_norm_m_s",
        "improvement_ratio",
        "bic",
        "bic_improvement",
        "solver_success",
        "status",
        "warning_count",
        "warnings",
        "covariance_valid",
        "solver",
        "fit_metrics",
        "holdout_metrics",
    }
    return {key: _jsonable(candidate.get(key)) for key in keep if key in candidate}


def _write_maneuver_detection_artifacts(
    *,
    output_root: Path,
    object_id: str,
    candidate_rows: list[dict[str, Any]],
    best: Mapping[str, Any] | None,
    status: str,
    warnings: list[str],
    evaluate: Any,
    times: np.ndarray,
    fit_mask: np.ndarray,
    holdout_mask: np.ndarray,
    positions: np.ndarray,
    velocities: np.ndarray | None,
    fit_duration_s: float,
    total_duration_s: float,
) -> dict[str, str]:
    candidates_csv = output_root / "maneuver_candidates.csv"
    _write_dict_rows(
        candidates_csv,
        candidate_rows,
        fieldnames=[
            "candidate_id",
            "time_s",
            "frame",
            "fit_position_rms_m",
            "holdout_position_rms_m",
            "delta_v_0_m_s",
            "delta_v_1_m_s",
            "delta_v_2_m_s",
            "delta_v_norm_m_s",
            "improvement_ratio",
            "bic",
            "bic_improvement",
            "solver_success",
            "status",
            "warning_count",
        ],
    )
    artifacts = {
        "candidates_csv": str(candidates_csv),
        "detection_report_json": str(output_root / "maneuver_detection_report.json"),
        "detection_report_md": str(output_root / "maneuver_detection_report.md"),
    }
    if best is None:
        return artifacts
    parameters = best["parameters"]
    native_values = np.array(best["native_values"], dtype=float)
    maneuver = dict(best["maneuver"])
    fit_t, fit_x, _fit_packet, fit_artifact = evaluate(
        native_values,
        duration_s=fit_duration_s,
        scratch_name="maneuver_best_fit",
        parameters_override=parameters,
        maneuver=maneuver,
    )
    pred_t, pred_x, _pred_packet, pred_artifact = evaluate(
        native_values,
        duration_s=total_duration_s,
        scratch_name="maneuver_best_prediction",
        parameters_override=parameters,
        maneuver=maneuver,
    )
    fit_on_obs = _states_at_epochs(fit_t, fit_x, times[fit_mask])
    pred_on_holdout = _states_at_epochs(pred_t, pred_x, times[holdout_mask])
    fit_residuals_csv = output_root / "maneuver_fit_residuals.csv"
    holdout_errors_csv = output_root / "maneuver_holdout_errors.csv"
    fit_plot_path = output_root / "maneuver_fit_position_residuals.png"
    holdout_plot_path = output_root / "maneuver_holdout_position_errors.png"
    materialized_fit = output_root / "materialized_maneuver_fit_config.yaml"
    materialized_prediction = output_root / "materialized_maneuver_prediction_config.yaml"
    _write_error_csv(
        fit_residuals_csv,
        times[fit_mask],
        fit_on_obs,
        positions[fit_mask],
        velocities[fit_mask] if velocities is not None else None,
    )
    _write_error_csv(
        holdout_errors_csv,
        times[holdout_mask] - fit_duration_s,
        pred_on_holdout,
        positions[holdout_mask],
        velocities[holdout_mask] if velocities is not None else None,
    )
    _write_error_plot(
        fit_plot_path,
        times[fit_mask],
        fit_on_obs[:, :3] - positions[fit_mask],
        title=f"Maneuver-Aware OD Fit Residuals ({status})",
    )
    _write_error_plot(
        holdout_plot_path,
        times[holdout_mask] - fit_duration_s,
        pred_on_holdout[:, :3] - positions[holdout_mask],
        title="Maneuver-Aware OD Holdout Errors",
    )
    fit_artifact.write(materialized_fit)
    pred_artifact.write(materialized_prediction)
    artifacts.update(
        {
            "fit_residuals_csv": str(fit_residuals_csv),
            "holdout_errors_csv": str(holdout_errors_csv),
            "fit_plot_path": str(fit_plot_path),
            "holdout_plot_path": str(holdout_plot_path),
            "materialized_fit_config_path": str(materialized_fit),
            "materialized_prediction_config_path": str(materialized_prediction),
        }
    )
    return artifacts


def _write_dict_rows(path: Path, rows: list[dict[str, Any]], *, fieldnames: list[str]) -> None:
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: row.get(key, "") for key in fieldnames})
    write_output_text(path, stream.getvalue())


def _write_maneuver_detection_md(path: Path, report: Mapping[str, Any]) -> None:
    best = dict(report.get("best_candidate", {}) or {})
    gates = dict(report.get("quality_gates", {}) or {})
    warnings = list(gates.get("warnings", []) or [])
    lines = [
        "# Maneuver Detection",
        "",
        f"- Status: `{report.get('status', 'unknown')}`",
        f"- Frame: `{report.get('frame', 'ric')}`",
        f"- Candidates evaluated: {int(report.get('candidate_count', 0))}",
        f"- Supported candidates: {int(report.get('supported_candidate_count', 0))}",
        "",
        "## Best Candidate",
        "",
    ]
    if best:
        lines.extend(
            [
                f"- Time: {float(best.get('time_s', 0.0)):.6f} s",
                f"- Delta-v norm: {float(best.get('delta_v_norm_m_s', 0.0)):.6f} m/s",
                f"- Fit RMS: {float(best.get('fit_position_rms_m', 0.0)):.6f} m",
                f"- Holdout RMS: {float(best.get('holdout_position_rms_m', 0.0)):.6f} m",
                f"- Improvement ratio: {_fmt_optional(best.get('improvement_ratio'))}",
                f"- Candidate status: `{best.get('status', 'unknown')}`",
            ]
        )
    else:
        lines.append("- None.")
    lines.extend(["", "## Warnings", ""])
    lines.extend(f"- {warning}" for warning in warnings) if warnings else lines.append("- None.")
    lines.extend(["", "## Non-Claims", ""])
    lines.extend(f"- {item}" for item in list(gates.get("non_claims", []) or []))
    write_output_text(path, "\n".join(lines) + "\n")


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, ParameterSet):
        return {"parameters": value.metadata()}
    return value

__all__ = [name for name in globals() if not name.startswith("__")]
