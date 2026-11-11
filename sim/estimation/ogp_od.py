from __future__ import annotations

import argparse
import csv
import json
import math
import sqlite3
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np

from sim.dynamics.orbit.ogp import (
    OGP_DEEP_SPACE_PERIOD_THRESHOLD_MIN,
    ogp_propagate_teme_batch_accelerated,
    ogp_propagate_teme_batch_reference,
    ogp_propagator_name_for_elements,
    ogp_regime_for_elements,
)
from sim.dynamics.orbit.sgp4 import sgp4_orbital_period_min
from sim.dynamics.orbit.tle import TLEElements, parse_tle_lines
from sim.estimation.batch_least_squares import BatchLeastSquaresResult, solve_batch_least_squares
from sim.estimation.parameters import EstimatedParameter, ParameterSet
from sim.estimation.residual_audit import build_residual_audit
from sim.estimation.weighting import (
    covariance_from_sigmas,
    prepare_covariance_whitener,
    validate_covariance_block,
    whiten_residual_block,
    whiten_residual_with_factor,
)
from sim.security.output_writes import mkdir_output, staged_output_file, write_output_text

OGP_OD_SCHEMA_VERSION = 1
OGP_PV_PACKET_CONTRACT = "oel_ogp_teme_pv_observations_v1"
_MINUTES_PER_DAY = 1440.0
_SECONDS_PER_DAY = 86400.0
_OD_PROPAGATION_CACHE_MAX_ENTRIES = 64
_OD_PROPAGATION_CACHE_MAX_BYTES = 32 * 1024 * 1024


def fit_ogp_mean_elements_to_observations(
    *,
    times_jd_utc: Sequence[float] | np.ndarray,
    positions_teme_km: Sequence[Sequence[float]] | np.ndarray,
    velocities_teme_km_s: Sequence[Sequence[float]] | np.ndarray,
    template_elements: TLEElements,
    output_dir: str | Path | None = None,
    position_sigma_km: float = 0.001,
    velocity_sigma_km_s: float = 1.0e-6,
    covariance_blocks: Sequence[Sequence[Sequence[float]]] | np.ndarray | None = None,
    fit_until_jd_utc: float | None = None,
    parameterization: str = "auto",
    estimate_bstar: bool = False,
    bstar_prior_sigma: float | None = None,
    estimate_epoch_offset: bool = False,
    epoch_offset_prior_sigma_s: float | None = None,
    max_nfev: int = 120,
    robust_loss: str = "linear",
    robust_f_scale: float = 1.0,
    sigma_clip_threshold: float | None = None,
    validation_reference: Mapping[str, Any] | None = None,
    propagation_backend: str = "rust",
) -> dict[str, Any]:
    """Fit a regime-specific OGP mean-element product to native TEME PV rows.

    The estimator dispatches propagation through the checked-in OGP-SGP4 or
    OGP-SDP4 implementation. Optional B* and epoch adjustment are admitted only
    through explicit preflight and postfit identifiability gates.
    """

    backend = _normalize_od_propagation_backend(propagation_backend)
    observed_backends: set[str] = set()
    fallback_reasons: list[dict[str, str]] = []
    propagation_cache: OrderedDict[tuple[TLEElements, bytes], tuple[np.ndarray, np.ndarray]] = OrderedDict()
    times, positions, velocities, covariance = _normalize_observations(
        times_jd_utc,
        positions_teme_km,
        velocities_teme_km_s,
        covariance_blocks=covariance_blocks,
        position_sigma_km=position_sigma_km,
        velocity_sigma_km_s=velocity_sigma_km_s,
    )
    fit_mask = _fit_mask(times, fit_until_jd_utc)
    holdout_mask = ~fit_mask
    covariance_factors = _covariance_factors(covariance)
    fit_times = times[fit_mask]
    regime = ogp_regime_for_elements(template_elements)
    propagator_name = ogp_propagator_name_for_elements(template_elements)
    selected_parameterization = _select_parameterization(template_elements, requested=parameterization)
    fit_span_days = float(fit_times[-1] - fit_times[0])
    period_min = _MINUTES_PER_DAY / float(template_elements.mean_motion_rev_per_day)

    optional_decisions = _optional_parameter_preflight(
        regime=regime,
        estimate_bstar=estimate_bstar,
        bstar_prior_sigma=bstar_prior_sigma,
        estimate_epoch_offset=estimate_epoch_offset,
        epoch_offset_prior_sigma_s=epoch_offset_prior_sigma_s,
        fit_sample_count=int(np.count_nonzero(fit_mask)),
        fit_span_days=fit_span_days,
        period_min=period_min,
    )
    active_bstar = bool(optional_decisions["bstar"]["preflight_accepted"])
    active_epoch = bool(optional_decisions["epoch_offset"]["preflight_accepted"])

    solve_bundle = _solve_ogp_fit(
        times=times,
        positions=positions,
        velocities=velocities,
        covariance_factors=covariance_factors,
        fit_mask=fit_mask,
        template=template_elements,
        parameterization=selected_parameterization,
        active_bstar=active_bstar,
        active_epoch=active_epoch,
        bstar_prior_sigma=bstar_prior_sigma,
        epoch_offset_prior_sigma_s=epoch_offset_prior_sigma_s,
        position_sigma_km=float(position_sigma_km),
        velocity_sigma_km_s=float(velocity_sigma_km_s),
        max_nfev=max_nfev,
        robust_loss=robust_loss,
        robust_f_scale=robust_f_scale,
        sigma_clip_threshold=sigma_clip_threshold,
        propagation_backend=backend,
        observed_backends=observed_backends,
        propagation_cache=propagation_cache,
        fallback_reasons=fallback_reasons,
    )
    solve = solve_bundle["solve"]
    optional_names = [name for name in ("bstar", "epoch_offset_s") if name in solve_bundle["parameters"].names]
    rejected_postfit = _unidentifiable_optional_parameters(solve, optional_names)
    if rejected_postfit:
        for name in rejected_postfit:
            decision_key = "epoch_offset" if name == "epoch_offset_s" else name
            optional_decisions[decision_key].update(
                {
                    "estimated": False,
                    "postfit_identifiable": False,
                    "reason": "postfit_data_not_identifiable",
                }
            )
        solve_bundle = _solve_ogp_fit(
            times=times,
            positions=positions,
            velocities=velocities,
            covariance_factors=covariance_factors,
            fit_mask=fit_mask,
            template=template_elements,
            parameterization=selected_parameterization,
            active_bstar=False,
            active_epoch=False,
            bstar_prior_sigma=None,
            epoch_offset_prior_sigma_s=None,
            position_sigma_km=float(position_sigma_km),
            velocity_sigma_km_s=float(velocity_sigma_km_s),
            max_nfev=max_nfev,
            robust_loss=robust_loss,
            robust_f_scale=robust_f_scale,
            sigma_clip_threshold=sigma_clip_threshold,
            propagation_backend=backend,
            observed_backends=observed_backends,
            propagation_cache=propagation_cache,
            fallback_reasons=fallback_reasons,
        )
        solve = solve_bundle["solve"]
    else:
        for name in optional_names:
            decision_key = "epoch_offset" if name == "epoch_offset_s" else name
            optional_decisions[decision_key].update(
                {
                    "estimated": True,
                    "postfit_identifiable": True,
                    "reason": "accepted_after_preflight_and_postfit_observability",
                }
            )

    estimated = solve_bundle["estimated_elements"]
    initial_pos_error, initial_vel_error = _ogp_state_errors(
        template_elements, times, positions, velocities,
        propagation_backend=backend, observed_backends=observed_backends,
        propagation_cache=propagation_cache,
        fallback_reasons=fallback_reasons,
    )
    final_pos_error, final_vel_error = _ogp_state_errors(
        estimated, times, positions, velocities,
        propagation_backend=backend, observed_backends=observed_backends,
        propagation_cache=propagation_cache,
        fallback_reasons=fallback_reasons,
    )
    fit_whitened = _whiten_pv(
        final_pos_error[fit_mask],
        final_vel_error[fit_mask],
        covariance_factors[fit_mask],
        factors_prepared=True,
    )
    holdout_whitened = (
        _whiten_pv(
            final_pos_error[holdout_mask],
            final_vel_error[holdout_mask],
            covariance_factors[holdout_mask],
            factors_prepared=True,
        )
        if np.any(holdout_mask)
        else np.array([], dtype=float)
    )
    residual_records = _residual_records(
        times=times,
        positions=positions,
        velocities=velocities,
        position_errors=final_pos_error,
        velocity_errors=final_vel_error,
        covariance=covariance,
        fit_mask=fit_mask,
        propagator_name=propagator_name,
        covariance_factors=covariance_factors,
    )
    element_comparison = _mean_element_comparison(template_elements, estimated)
    fit_rms = float(np.sqrt(np.mean(fit_whitened * fit_whitened)))
    holdout_rms = float(np.sqrt(np.mean(holdout_whitened * holdout_whitened))) if holdout_whitened.size else None
    warnings = _quality_warnings(
        solve=solve,
        fit_rms=fit_rms,
        holdout_rms=holdout_rms,
        optional_decisions=optional_decisions,
        expected_regime=regime,
        estimated_elements=estimated,
    )
    report: dict[str, Any] = {
        "schema_version": OGP_OD_SCHEMA_VERSION,
        "kind": "oel.ogp_mean_element_od_result",
        "method": "ogp_mean_elements_least_squares",
        "ogp_regime": regime,
        "propagator_name": propagator_name,
        "regime_classification": _regime_classification(template_elements),
        "reference_frame": "TEME",
        "frame_contract": {
            "observation_frame": "TEME",
            "propagator_native_frame": "TEME",
            "transform_applied_inside_fit": False,
            "transformed_sensor_paths_require": "explicit named TEME-to-output-frame transform",
        },
        "parameterization": {
            "name": selected_parameterization,
            "nonsingular": selected_parameterization == "equinoctial_nonsingular",
            "phase_policy": "fixed_mean_phase_while_estimating_epoch"
            if "epoch_offset_s" in solve_bundle["parameters"].names
            else "mean_phase_estimated",
            "estimated_parameter_names": solve_bundle["parameters"].names,
        },
        "optional_parameter_decisions": optional_decisions,
        "observation_contract": OGP_PV_PACKET_CONTRACT,
        "observation_count": int(times.size),
        "fit_sample_count": int(np.count_nonzero(fit_mask)),
        "holdout_sample_count": int(np.count_nonzero(holdout_mask)),
        "fit_span_days": fit_span_days,
        "total_span_days": float(times[-1] - times[0]),
        "orbital_period_min": period_min,
        "observation_partition": {
            "strategy": "absolute_epoch_boundary",
            "fit_boundary": "jd_utc <= fit_until_jd_utc",
            "holdout_boundary": "jd_utc > fit_until_jd_utc",
            "holdout_status": "evaluated" if np.any(holdout_mask) else "not_evaluated",
        },
        "template_elements": asdict(template_elements),
        "estimated_elements": asdict(estimated),
        "estimated_parameters": solve_bundle["parameters"].metadata(solve.x_native),
        "parameter_names": list(solve.parameter_names),
        "parameter_units": list(solve_bundle["parameters"].units),
        "covariance_native": None if solve.covariance_native is None else solve.covariance_native.tolist(),
        "data_covariance_native": None if solve.data_covariance_native is None else solve.data_covariance_native.tolist(),
        "covariance_evidence": solve.covariance_evidence,
        "empirically_calibrated": False,
        "mean_element_comparison": element_comparison,
        "state_space_metrics": {
            "initial_all": _state_metrics(initial_pos_error, initial_vel_error),
            "final_fit": _state_metrics(final_pos_error[fit_mask], final_vel_error[fit_mask]),
            "final_holdout": _state_metrics(final_pos_error[holdout_mask], final_vel_error[holdout_mask])
            if np.any(holdout_mask)
            else {"status": "not_evaluated"},
            "fit_whitened_rms": fit_rms,
            "holdout_whitened_rms": holdout_rms,
        },
        "metric_separation": {
            "state_space": "TEME position/velocity prediction residuals with declared covariance",
            "mean_element_space": "parameter deltas with element-specific units and wrapped angular differences",
            "comparability": "not_interchangeable",
        },
        "solver": {
            "success": bool(solve.success),
            "message": solve.message,
            "nfev": int(solve.nfev),
            "initial_cost": float(solve.initial_cost),
            "final_cost": float(solve.cost),
            "diagnostics": solve.diagnostics,
            "decision_records": list(solve.decision_records),
        },
        "deep_space_diagnostics": _deep_space_diagnostics(
            regime=regime,
            template=template_elements,
            estimated=estimated,
            solve=solve,
        ),
        "residual_records": residual_records,
        "residual_audit": build_residual_audit(
            residual_records,
            decision_records=solve.decision_records,
        ),
        "validation_reference": dict(validation_reference or {}),
        "quality_gates": {
            "solver_success": bool(solve.success),
            "data_full_rank": bool(solve.diagnostics["observability"]["data_full_rank"]),
            "fit_improved": bool(
                _state_metrics(final_pos_error[fit_mask], final_vel_error[fit_mask])["position_rms_m"]
                <= _state_metrics(initial_pos_error[fit_mask], initial_vel_error[fit_mask])["position_rms_m"]
            ),
            "holdout_evidence_status": "evaluated" if np.any(holdout_mask) else "not_evaluated",
            "regime_stable": ogp_regime_for_elements(estimated) == regime,
            "warnings": warnings,
            "evidence_status": "ready_with_caveats" if not warnings else "review_required",
        },
        "claim_boundary": (
            f"Regime-bounded OGP-{regime.upper()} mean-element fit to native TEME PV observations; "
            "not a numerical precise-orbit, calibrated covariance, association, or operational TLE claim."
        ),
        "non_claims": [
            "Mean-element agreement and state-space prediction residuals are different metrics.",
            "A fitted element set is not precise truth without an independent reference product.",
            "This workflow does not format or publish an operational TLE.",
        ],
    }
    if backend in {"accelerated", "rust"}:
        report["propagation_backend"] = {
            "requested": backend,
            "observed": sorted(observed_backends),
            "reference_backend": "ogp_scalar_reference",
            "fallback_used": bool(fallback_reasons),
            "fallback_reasons": list(fallback_reasons),
            "equivalence": (
                "cross_implementation" if backend == "rust" else
                "scalar_reference_exact" if not any("numba_cpu" in item for item in observed_backends)
                else "approved_rounding_level"
            ),
            "floating_point_parity": (
                "bounded_different" if backend == "rust" else
                "bitwise_reference" if not any("numba_cpu" in item for item in observed_backends)
                else "rounding_level_different"
            ),
        }
        if backend == "rust":
            report["propagation_backend"]["residual_evaluation"] = solve_bundle["residual_backend"]
    if output_dir is not None:
        _write_artifacts(Path(output_dir).expanduser().resolve(), report)
    return report


def run_ogp_pv_packet(packet: Mapping[str, Any], *, output_dir: str | Path) -> dict[str, Any]:
    """Validate and fit a version-1 OGP native-TEME PV packet."""

    data = dict(packet)
    if int(data.get("schema_version", 0)) != 1 or str(data.get("kind")) != "oel.ogp_pv_observation_packet":
        raise ValueError("OGP PV packet must use schema_version=1 and kind='oel.ogp_pv_observation_packet'.")
    if str(data.get("measurement_contract")) != OGP_PV_PACKET_CONTRACT:
        raise ValueError(f"OGP PV packet requires measurement_contract={OGP_PV_PACKET_CONTRACT!r}.")
    tle = dict(data.get("template_tle", {}) or {})
    template = parse_tle_lines(str(tle.get("line1", "")), str(tle.get("line2", "")))
    observations = list(data.get("observations", []) or [])
    if len(observations) < 2:
        raise ValueError("OGP PV packet requires at least two observations.")
    options = dict(data.get("fit_options", {}) or {})
    return fit_ogp_mean_elements_to_observations(
        times_jd_utc=[float(row["jd_utc"]) for row in observations],
        positions_teme_km=[row["position_teme_km"] for row in observations],
        velocities_teme_km_s=[row["velocity_teme_km_s"] for row in observations],
        template_elements=template,
        output_dir=output_dir,
        position_sigma_km=float(options.pop("position_sigma_km", 0.001)),
        velocity_sigma_km_s=float(options.pop("velocity_sigma_km_s", 1.0e-6)),
        **options,
    )


def _solve_ogp_fit(
    *,
    times: np.ndarray,
    positions: np.ndarray,
    velocities: np.ndarray,
    covariance_factors: np.ndarray,
    fit_mask: np.ndarray,
    template: TLEElements,
    parameterization: str,
    active_bstar: bool,
    active_epoch: bool,
    bstar_prior_sigma: float | None,
    epoch_offset_prior_sigma_s: float | None,
    position_sigma_km: float,
    velocity_sigma_km_s: float,
    max_nfev: int,
    robust_loss: str,
    robust_f_scale: float,
    sigma_clip_threshold: float | None,
    propagation_backend: str,
    observed_backends: set[str],
    propagation_cache: OrderedDict[tuple[TLEElements, bytes], tuple[np.ndarray, np.ndarray]],
    fallback_reasons: list[dict[str, str]],
) -> dict[str, Any]:
    regime = ogp_regime_for_elements(template)
    parameters = _parameter_set(
        template,
        parameterization=parameterization,
        active_bstar=active_bstar,
        active_epoch=active_epoch,
    )
    native_residual = None
    if propagation_backend == "rust":
        from sim.rust_ogp_backend import RustOGPObservationBatch
        from sim.rust_orbit_backend import _extension

        if hasattr(_extension(), "OGPObservationBatch"):
            native_residual = RustOGPObservationBatch(
                times[fit_mask], positions[fit_mask], velocities[fit_mask], covariance_factors[fit_mask],
            )

    def residual(vector: np.ndarray) -> np.ndarray:
        try:
            elements = _elements_from_parameters(
                template,
                parameters,
                vector,
                parameterization=parameterization,
                active_epoch=active_epoch,
            )
            if ogp_regime_for_elements(elements) != regime:
                raise ValueError("OGP OD candidate crossed the fixed propagator regime boundary")
            if native_residual is not None:
                observed_backends.add("ogp_rust")
                return native_residual.residual(elements)
            pos_error, vel_error = _ogp_state_errors(
                elements,
                times[fit_mask],
                positions[fit_mask],
                velocities[fit_mask],
                propagation_backend=propagation_backend,
                observed_backends=observed_backends,
                propagation_cache=propagation_cache,
                fallback_reasons=fallback_reasons,
            )
            return _whiten_pv(
                pos_error,
                vel_error,
                covariance_factors[fit_mask],
                factors_prepared=True,
            )
        except (ValueError, FloatingPointError):
            return np.full(int(np.count_nonzero(fit_mask)) * 6, 1.0e12, dtype=float)

    prior_names: list[str] = []
    prior_means: list[float] = []
    prior_sigmas: list[float] = []
    if active_bstar:
        prior_names.append("bstar")
        prior_means.append(float(template.bstar))
        prior_sigmas.append(float(bstar_prior_sigma))
    if active_epoch:
        prior_names.append("epoch_offset_s")
        prior_means.append(0.0)
        prior_sigmas.append(float(epoch_offset_prior_sigma_s))
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
        prior_mean_native=np.asarray(prior_means, dtype=float) if prior_names else None,
        prior_covariance_native=np.diag(np.square(prior_sigmas)) if prior_names else None,
        prior_parameter_names=prior_names or None,
    )
    estimated = _elements_from_parameters(
        template,
        parameters,
        solve.x_native,
        parameterization=parameterization,
        active_epoch=active_epoch,
    )
    if ogp_regime_for_elements(estimated) != regime:
        raise ValueError("OGP OD solution crossed the fixed propagator regime boundary")
    return {"solve": solve, "parameters": parameters, "estimated_elements": estimated,
            "residual_backend": "rust_fused" if native_residual is not None else "python_whitening"}


def _parameter_set(
    template: TLEElements,
    *,
    parameterization: str,
    active_bstar: bool,
    active_epoch: bool,
) -> ParameterSet:
    regime = ogp_regime_for_elements(template)
    mean_motion = float(template.mean_motion_rev_per_day)
    threshold_motion = _mean_motion_at_regime_boundary(template)
    motion_lower = max(0.05, 0.5 * mean_motion)
    motion_upper = min(18.0, 1.5 * mean_motion)
    threshold_motion = _mean_motion_at_regime_boundary(template)
    if regime == "sgp4":
        motion_lower = min(mean_motion, max(motion_lower, np.nextafter(threshold_motion, np.inf)))
    else:
        motion_upper = max(mean_motion, min(motion_upper, np.nextafter(threshold_motion, 0.0)))
    parameters: list[EstimatedParameter]
    if parameterization == "equinoctial_nonsingular":
        ex, ey, hx, hy, mean_longitude = _nonsingular_vector(template)
        parameters = [
            EstimatedParameter("eccentricity_x", ex, scale=0.01, lower=-0.99, upper=0.99),
            EstimatedParameter("eccentricity_y", ey, scale=0.01, lower=-0.99, upper=0.99),
            EstimatedParameter("inclination_x", hx, scale=0.01, lower=-10.0, upper=10.0),
            EstimatedParameter("inclination_y", hy, scale=0.01, lower=-10.0, upper=10.0),
        ]
        if not active_epoch:
            parameters.append(
                EstimatedParameter(
                    "mean_longitude_delta_deg",
                    0.0,
                    scale=1.0,
                    lower=-720.0,
                    upper=720.0,
                    unit="deg",
                    description=f"Delta from template mean longitude {mean_longitude:.12g} deg",
                )
            )
    else:
        parameters = [
            EstimatedParameter("inclination_deg", template.inclination_deg, scale=1.0, lower=0.0, upper=180.0),
            EstimatedParameter("raan_delta_deg", 0.0, scale=1.0, lower=-720.0, upper=720.0),
            EstimatedParameter("eccentricity", template.eccentricity, scale=0.01, lower=1.0e-8, upper=0.999),
            EstimatedParameter("argp_delta_deg", 0.0, scale=1.0, lower=-720.0, upper=720.0),
        ]
        if not active_epoch:
            parameters.append(
                EstimatedParameter("mean_anomaly_delta_deg", 0.0, scale=1.0, lower=-720.0, upper=720.0)
            )
    parameters.append(
        EstimatedParameter(
            "mean_motion_rev_per_day",
            mean_motion,
            scale=max(0.001, 0.01 * mean_motion),
            lower=motion_lower,
            upper=motion_upper,
        )
    )
    if active_bstar:
        parameters.append(
            EstimatedParameter(
                "bstar",
                template.bstar,
                scale=max(abs(float(template.bstar)), 1.0e-5),
                lower=-0.05,
                upper=0.05,
            )
        )
    if active_epoch:
        parameters.append(
            EstimatedParameter("epoch_offset_s", 0.0, scale=10.0, lower=-43200.0, upper=43200.0, unit="s")
        )
    return ParameterSet(parameters)


def _mean_motion_at_regime_boundary(template: TLEElements) -> float:
    """Solve the raw mean-motion value corresponding to corrected 225-minute period."""

    lower = 0.05
    upper = 18.0
    target = float(OGP_DEEP_SPACE_PERIOD_THRESHOLD_MIN)
    for _ in range(80):
        midpoint = 0.5 * (lower + upper)
        candidate = replace(template, mean_motion_rev_per_day=midpoint)
        if sgp4_orbital_period_min(candidate) >= target:
            lower = midpoint
        else:
            upper = midpoint
    return 0.5 * (lower + upper)


def _elements_from_parameters(
    template: TLEElements,
    parameters: ParameterSet,
    vector: Sequence[float] | np.ndarray,
    *,
    parameterization: str,
    active_epoch: bool,
) -> TLEElements:
    values = parameters.mapping(np.asarray(vector, dtype=float))
    if parameterization == "equinoctial_nonsingular":
        ex = values["eccentricity_x"]
        ey = values["eccentricity_y"]
        hx = values["inclination_x"]
        hy = values["inclination_y"]
        eccentricity = math.hypot(ex, ey)
        if eccentricity >= 0.999:
            raise ValueError("Nonsingular OGP parameterization produced eccentricity >= 0.999.")
        longitude_periapsis = math.degrees(math.atan2(ey, ex)) if eccentricity > 1.0e-14 else (
            float(template.raan_deg) + float(template.argp_deg)
        )
        half_inclination_coordinate = math.hypot(hx, hy)
        if float(template.inclination_deg) > 90.0:
            inclination = 180.0 - math.degrees(2.0 * math.atan(half_inclination_coordinate))
        else:
            inclination = math.degrees(2.0 * math.atan(half_inclination_coordinate))
        raan = (
            math.degrees(math.atan2(hy, hx))
            if half_inclination_coordinate > 1.0e-14
            else float(template.raan_deg)
        )
        mean_longitude = (
            float(template.raan_deg) + float(template.argp_deg) + float(template.mean_anomaly_deg)
        )
        mean_longitude += float(values.get("mean_longitude_delta_deg", 0.0))
        argp = longitude_periapsis - raan
        mean_anomaly = mean_longitude - longitude_periapsis
    else:
        inclination = values["inclination_deg"]
        raan = float(template.raan_deg) + values["raan_delta_deg"]
        eccentricity = values["eccentricity"]
        argp = float(template.argp_deg) + values["argp_delta_deg"]
        mean_anomaly = float(template.mean_anomaly_deg) + float(values.get("mean_anomaly_delta_deg", 0.0))
    epoch = float(template.epoch_jd_utc) + float(values.get("epoch_offset_s", 0.0)) / _SECONDS_PER_DAY
    return replace(
        template,
        epoch_jd_utc=epoch,
        inclination_deg=float(inclination),
        raan_deg=float(raan % 360.0),
        eccentricity=float(eccentricity),
        argp_deg=float(argp % 360.0),
        mean_anomaly_deg=float(mean_anomaly % 360.0),
        mean_motion_rev_per_day=float(values["mean_motion_rev_per_day"]),
        bstar=float(values.get("bstar", template.bstar)),
    )


def _nonsingular_vector(elements: TLEElements) -> tuple[float, float, float, float, float]:
    longitude_periapsis = math.radians(float(elements.raan_deg) + float(elements.argp_deg))
    eccentricity = float(elements.eccentricity)
    inclination_rad = math.radians(float(elements.inclination_deg))
    half_tangent = math.tan(0.5 * inclination_rad)
    if float(elements.inclination_deg) > 90.0:
        half_tangent = 1.0 / max(abs(half_tangent), 1.0e-15)
    raan = math.radians(float(elements.raan_deg))
    return (
        eccentricity * math.cos(longitude_periapsis),
        eccentricity * math.sin(longitude_periapsis),
        half_tangent * math.cos(raan),
        half_tangent * math.sin(raan),
        float(elements.raan_deg) + float(elements.argp_deg) + float(elements.mean_anomaly_deg),
    )


def _select_parameterization(elements: TLEElements, *, requested: str) -> str:
    key = str(requested or "auto").strip().lower()
    aliases = {
        "auto": "auto",
        "classical": "classical_delta",
        "classical_delta": "classical_delta",
        "nonsingular": "equinoctial_nonsingular",
        "equinoctial": "equinoctial_nonsingular",
        "equinoctial_nonsingular": "equinoctial_nonsingular",
    }
    if key not in aliases:
        raise ValueError("OGP OD parameterization must be auto, classical_delta, or equinoctial_nonsingular.")
    resolved = aliases[key]
    if resolved != "auto":
        return resolved
    inclination = float(elements.inclination_deg)
    if float(elements.eccentricity) < 0.05 or inclination < 5.0 or inclination > 175.0:
        return "equinoctial_nonsingular"
    return "classical_delta"


def _optional_parameter_preflight(
    *,
    regime: str,
    estimate_bstar: bool,
    bstar_prior_sigma: float | None,
    estimate_epoch_offset: bool,
    epoch_offset_prior_sigma_s: float | None,
    fit_sample_count: int,
    fit_span_days: float,
    period_min: float,
) -> dict[str, dict[str, Any]]:
    if estimate_bstar and estimate_epoch_offset:
        raise ValueError("Phase 6 OGP OD does not estimate B* and epoch offset in the same solve.")
    bstar_reasons = []
    if estimate_bstar:
        if regime == "sdp4" and fit_span_days * _MINUTES_PER_DAY < period_min:
            bstar_reasons.append("requires_at_least_one_orbital_period_of_fit_data")
        if fit_sample_count < 8:
            bstar_reasons.append("requires_at_least_8_fit_samples")
        if fit_span_days < 2.0:
            bstar_reasons.append("requires_at_least_2_days_of_fit_data")
        if bstar_prior_sigma is None or not np.isfinite(float(bstar_prior_sigma)) or float(bstar_prior_sigma) <= 0.0:
            bstar_reasons.append("requires_positive_bstar_prior_sigma")
    epoch_reasons = []
    if estimate_epoch_offset:
        if fit_sample_count < 8:
            epoch_reasons.append("requires_at_least_8_fit_samples")
        if fit_span_days * _MINUTES_PER_DAY < float(period_min):
            epoch_reasons.append("requires_at_least_one_orbital_period_of_fit_data")
        if (
            epoch_offset_prior_sigma_s is None
            or not np.isfinite(float(epoch_offset_prior_sigma_s))
            or float(epoch_offset_prior_sigma_s) <= 0.0
        ):
            epoch_reasons.append("requires_positive_epoch_offset_prior_sigma_s")
    return {
        "bstar": {
            "requested": bool(estimate_bstar),
            "preflight_accepted": bool(estimate_bstar and not bstar_reasons),
            "estimated": False,
            "reasons": bstar_reasons,
        },
        "epoch_offset": {
            "requested": bool(estimate_epoch_offset),
            "preflight_accepted": bool(estimate_epoch_offset and not epoch_reasons),
            "estimated": False,
            "reasons": epoch_reasons,
            "identifiability_policy": "mean phase fixed when epoch offset is active",
        },
    }


def _unidentifiable_optional_parameters(solve: BatchLeastSquaresResult, names: Sequence[str]) -> list[str]:
    rows = list(dict(solve.diagnostics["observability"]).get("parameters", []))
    by_name = {str(row["parameter"]): bool(row["identifiable"]) for row in rows}
    return [name for name in names if not by_name.get(name, False)]


def _normalize_observations(
    times: Sequence[float] | np.ndarray,
    positions: Sequence[Sequence[float]] | np.ndarray,
    velocities: Sequence[Sequence[float]] | np.ndarray,
    *,
    covariance_blocks: Sequence[Sequence[Sequence[float]]] | np.ndarray | None,
    position_sigma_km: float,
    velocity_sigma_km_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    epochs = np.asarray(times, dtype=float).reshape(-1)
    pos = np.asarray(positions, dtype=float)
    vel = np.asarray(velocities, dtype=float)
    if epochs.size < 2 or pos.shape != (epochs.size, 3) or vel.shape != (epochs.size, 3):
        raise ValueError("OGP OD requires at least two epochs and matching TEME position/velocity arrays.")
    if not np.all(np.isfinite(epochs)) or not np.all(np.isfinite(pos)) or not np.all(np.isfinite(vel)):
        raise ValueError("OGP OD observations must be finite.")
    if np.any(np.diff(epochs) <= 0.0):
        raise ValueError("OGP OD epochs must be strictly increasing.")
    sigmas = np.array([position_sigma_km] * 3 + [velocity_sigma_km_s] * 3, dtype=float)
    if not np.all(np.isfinite(sigmas)) or np.any(sigmas <= 0.0):
        raise ValueError("OGP OD position and velocity sigmas must be finite and positive.")
    if covariance_blocks is None:
        covariance = np.broadcast_to(covariance_from_sigmas(sigmas), (epochs.size, 6, 6)).copy()
    else:
        covariance = np.asarray(covariance_blocks, dtype=float)
        if covariance.shape == (6, 6):
            covariance = np.broadcast_to(covariance, (epochs.size, 6, 6)).copy()
        if covariance.shape != (epochs.size, 6, 6):
            raise ValueError("OGP OD covariance_blocks must have shape (6, 6) or (N, 6, 6).")
        for index in range(epochs.size):
            covariance[index] = validate_covariance_block(
                covariance[index], dimension=6, field_name=f"OGP observation {index} covariance"
            )
    return epochs, pos, vel, covariance


def _fit_mask(times: np.ndarray, fit_until_jd_utc: float | None) -> np.ndarray:
    if fit_until_jd_utc is None:
        return np.ones(times.size, dtype=bool)
    boundary = float(fit_until_jd_utc)
    if not np.isfinite(boundary):
        raise ValueError("fit_until_jd_utc must be finite.")
    mask = times <= boundary + 1.0e-12
    if int(np.count_nonzero(mask)) < 2:
        raise ValueError("OGP OD requires at least two fit samples.")
    return mask


def _ogp_state_errors(
    elements: TLEElements,
    times_jd_utc: np.ndarray,
    positions: np.ndarray,
    velocities: np.ndarray,
    *,
    propagation_backend: str = "rust",
    observed_backends: set[str] | None = None,
    propagation_cache: OrderedDict[tuple[TLEElements, bytes], tuple[np.ndarray, np.ndarray]] | None = None,
    fallback_reasons: list[dict[str, str]] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    offsets_min = (np.asarray(times_jd_utc, dtype=float) - float(elements.epoch_jd_utc)) * _MINUTES_PER_DAY
    cache_key = (elements, offsets_min.tobytes())
    cached = None if propagation_cache is None else propagation_cache.get(cache_key)
    if cached is not None:
        propagation_cache.move_to_end(cache_key)
        propagated_position, propagated_velocity = cached
        return propagated_position - positions, propagated_velocity - velocities
    if propagation_backend == "rust":
        batch = ogp_propagate_teme_batch_accelerated([elements], offsets_min, backend="rust")
    elif propagation_backend == "accelerated":
        dispatcher_exception = False
        try:
            batch = ogp_propagate_teme_batch_accelerated([elements], offsets_min)
        except Exception as exc:
            dispatcher_exception = True
            batch = ogp_propagate_teme_batch_reference([elements], offsets_min)
            if fallback_reasons is not None:
                _record_backend_fallback(
                    fallback_reasons,
                    exception_type=type(exc).__name__,
                    reason=str(exc),
                )
        if (
            not dispatcher_exception
            and "scalar_reference" in str(batch.backend)
            and fallback_reasons is not None
        ):
            _record_backend_fallback(
                fallback_reasons,
                exception_type="RuntimeDispatchFallback",
                reason="accelerated OGP dispatcher selected the scalar reference backend",
            )
    else:
        batch = ogp_propagate_teme_batch_reference([elements], offsets_min)
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
    return propagated_position - positions, propagated_velocity - velocities


def _normalize_od_propagation_backend(value: str) -> str:
    if value is None:
        value = "rust"
    if not isinstance(value, str):
        raise ValueError("propagation_backend must be scalar_reference, accelerated, or rust.")
    backend = value.strip().lower()
    if backend not in {"scalar_reference", "accelerated", "rust"}:
        raise ValueError("propagation_backend must be scalar_reference, accelerated, or rust.")
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


def _covariance_factors(covariance: np.ndarray) -> np.ndarray:
    # Keep per-observation diagnostics for malformed input, and use NumPy's
    # batched LAPACK dispatch for a valid homogeneous block collection.
    transposed = covariance.swapaxes(-1, -2)
    if covariance.shape[0] and np.isfinite(covariance).all() and np.allclose(covariance, transposed, rtol=1.0e-10, atol=1.0e-14):
        symmetric = 0.5 * (covariance + transposed)
        if np.all(np.linalg.eigvalsh(symmetric) > 0.0):
            return np.linalg.cholesky(symmetric)
    return np.stack(
        [
            prepare_covariance_whitener(
                covariance[index],
                dimension=6,
                field_name=f"OGP observation {index} covariance",
            )
            for index in range(covariance.shape[0])
        ],
        axis=0,
    )


def _whiten_pv(
    position_error: np.ndarray,
    velocity_error: np.ndarray,
    covariance_or_factors: np.ndarray,
    *,
    factors_prepared: bool = False,
) -> np.ndarray:
    if factors_prepared:
        if position_error.shape[0] == 0:
            return np.array([], dtype=float)
        residuals = np.column_stack((position_error, velocity_error))
        if not np.isfinite(residuals).all():
            raise ValueError("residual block must be finite.")
        return np.linalg.solve(covariance_or_factors, residuals[..., None])[..., 0].reshape(-1)
    blocks = [
        (
            whiten_residual_with_factor(
                np.hstack((position_error[index], velocity_error[index])),
                covariance_or_factors[index],
                field_name=f"OGP observation {index} covariance",
            )
            if factors_prepared
            else whiten_residual_block(
                np.hstack((position_error[index], velocity_error[index])),
                covariance_or_factors[index],
                field_name=f"OGP observation {index} covariance",
            )
        )
        for index in range(position_error.shape[0])
    ]
    return np.concatenate(blocks) if blocks else np.array([], dtype=float)


def _state_metrics(position_error: np.ndarray, velocity_error: np.ndarray) -> dict[str, Any]:
    if position_error.size == 0:
        return {"status": "not_evaluated"}
    position_norm_m = np.linalg.norm(position_error, axis=1) * 1000.0
    velocity_norm_mm_s = np.linalg.norm(velocity_error, axis=1) * 1.0e6
    return {
        "status": "evaluated",
        "sample_count": int(position_error.shape[0]),
        "position_rms_m": float(np.sqrt(np.mean(position_norm_m**2))),
        "position_max_m": float(np.max(position_norm_m)),
        "velocity_rms_mm_s": float(np.sqrt(np.mean(velocity_norm_mm_s**2))),
        "velocity_max_mm_s": float(np.max(velocity_norm_mm_s)),
    }


def _mean_element_comparison(template: TLEElements, estimated: TLEElements) -> dict[str, Any]:
    return {
        "units": {
            "angles": "deg",
            "eccentricity": "dimensionless",
            "mean_motion": "rev/day",
            "bstar": "1/Earth-radii",
            "epoch": "s",
        },
        "deltas": {
            "inclination_deg": float(estimated.inclination_deg - template.inclination_deg),
            "raan_deg_wrapped": _wrap_deg(estimated.raan_deg - template.raan_deg),
            "eccentricity": float(estimated.eccentricity - template.eccentricity),
            "argp_deg_wrapped": _wrap_deg(estimated.argp_deg - template.argp_deg),
            "mean_anomaly_deg_wrapped": _wrap_deg(estimated.mean_anomaly_deg - template.mean_anomaly_deg),
            "mean_motion_rev_per_day": float(
                estimated.mean_motion_rev_per_day - template.mean_motion_rev_per_day
            ),
            "bstar": float(estimated.bstar - template.bstar),
            "epoch_offset_s": float((estimated.epoch_jd_utc - template.epoch_jd_utc) * _SECONDS_PER_DAY),
        },
    }


def _residual_records(
    *,
    times: np.ndarray,
    positions: np.ndarray,
    velocities: np.ndarray,
    position_errors: np.ndarray,
    velocity_errors: np.ndarray,
    covariance: np.ndarray,
    fit_mask: np.ndarray,
    propagator_name: str,
    covariance_factors: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    component_names = ("x_km", "y_km", "z_km", "vx_km_s", "vy_km_s", "vz_km_s")
    records = []
    for index, epoch in enumerate(times):
        residual = np.hstack((position_errors[index], velocity_errors[index]))
        observed = np.hstack((positions[index], velocities[index]))
        whitened = (
            whiten_residual_block(residual, covariance[index], field_name=f"OGP observation {index}")
            if covariance_factors is None
            else whiten_residual_with_factor(residual, covariance_factors[index], field_name=f"OGP observation {index}")
        )
        partition = "fit" if bool(fit_mask[index]) else "holdout"
        for component_index, component in enumerate(component_names):
            records.append(
                {
                    "observation_id": f"ogp_pv:{index:06d}",
                    "measurement_type": "teme_position_velocity",
                    "arc_id": "ogp_mean_element_od",
                    "partition": partition,
                    "residual_kind": "postfit" if partition == "fit" else "prediction",
                    "frame": "TEME",
                    "epoch_evaluation_method": f"{propagator_name.lower().replace('-', '_')}_at_observation_epoch",
                    "jd_utc": float(epoch),
                    "component": component,
                    "observed": float(observed[component_index]),
                    "predicted": float(observed[component_index] + residual[component_index]),
                    "residual": float(residual[component_index]),
                    "sigma": float(math.sqrt(covariance[index, component_index, component_index])),
                    "normalized_residual": float(whitened[component_index]),
                    "whitened_residual": float(whitened[component_index]),
                    "covariance_source": "validated_block",
                }
            )
    return records


def _quality_warnings(
    *,
    solve: BatchLeastSquaresResult,
    fit_rms: float,
    holdout_rms: float | None,
    optional_decisions: Mapping[str, Mapping[str, Any]],
    expected_regime: str,
    estimated_elements: TLEElements,
) -> list[str]:
    warnings = []
    if not solve.success:
        warnings.append("solver_not_successful")
    if not bool(solve.diagnostics["observability"]["data_full_rank"]):
        warnings.append("data_jacobian_rank_deficient")
    if holdout_rms is None:
        warnings.append("holdout_not_evaluated")
    elif holdout_rms > max(10.0, 5.0 * max(fit_rms, 1.0e-12)):
        warnings.append("holdout_residual_large")
    if ogp_regime_for_elements(estimated_elements) != expected_regime:
        warnings.append("estimated_elements_crossed_ogp_regime_boundary")
    for name, decision in optional_decisions.items():
        if bool(decision.get("requested")) and not bool(decision.get("estimated")):
            warnings.append(f"requested_{name}_not_estimated")
    return warnings


def _deep_space_diagnostics(
    *,
    regime: str,
    template: TLEElements,
    estimated: TLEElements,
    solve: BatchLeastSquaresResult,
) -> dict[str, Any]:
    return {
        "applicable": regime == "sdp4",
        "classification": _regime_classification(template),
        "template_period_min": _MINUTES_PER_DAY / float(template.mean_motion_rev_per_day),
        "estimated_period_min": _MINUTES_PER_DAY / float(estimated.mean_motion_rev_per_day),
        "resonance_sensitive": _regime_classification(template)
        in {"synchronous_resonance", "half_day_resonance"},
        "solver_nfev": int(solve.nfev),
        "data_condition_number": solve.diagnostics["observability"]["data_condition_number"],
        "failure_classification": solve.diagnostics["failure_classification"],
    }


def _regime_classification(elements: TLEElements) -> str:
    if ogp_regime_for_elements(elements) == "sgp4":
        if float(elements.eccentricity) < 0.01:
            return "near_earth_near_circular"
        if float(elements.eccentricity) >= 0.05:
            return "near_earth_eccentric"
        if float(elements.bstar) != 0.0:
            return "near_earth_drag_sensitive"
        return "near_earth_general"
    mean_motion = float(elements.mean_motion_rev_per_day)
    eccentricity = float(elements.eccentricity)
    if abs(mean_motion - 1.0) < 0.15:
        return "synchronous_resonance"
    if abs(mean_motion - 2.0) < 0.2 and eccentricity > 0.4:
        return "half_day_resonance"
    if eccentricity > 0.4:
        return "high_eccentricity_deep_space"
    return "deep_space_nonresonant"


def _wrap_deg(value: float) -> float:
    return (float(value) + 180.0) % 360.0 - 180.0


def _write_artifacts(output: Path, report: dict[str, Any]) -> None:
    mkdir_output(output)
    report_path = output / "ogp_od_report.json"
    product_path = output / "fitted_ogp_mean_element_product.json"
    residual_path = output / "ogp_od_residuals.csv"
    review_path = output / "review/run.sqlite"
    report["artifacts"] = {
        "report_json": str(report_path),
        "product_json": str(product_path),
        "residual_csv": str(residual_path),
        "review_sqlite": str(review_path),
    }
    residual_rows = list(report["residual_records"])
    mkdir_output(residual_path.parent)
    with staged_output_file(residual_path) as residual_target:
        with residual_target.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(residual_rows[0]))
            writer.writeheader()
            writer.writerows(residual_rows)
    mkdir_output(review_path.parent)
    with staged_output_file(review_path) as review_target:
        with sqlite3.connect(review_target) as connection:
            connection.executescript(
            """
            DROP TABLE IF EXISTS ogp_od_run;
            DROP TABLE IF EXISTS ogp_od_parameters;
            DROP TABLE IF EXISTS ogp_od_residuals;
            CREATE TABLE ogp_od_run (
                ogp_regime TEXT, propagator_name TEXT, regime_classification TEXT,
                parameterization TEXT, solver_success INTEGER, evidence_status TEXT,
                fit_sample_count INTEGER, holdout_sample_count INTEGER,
                fit_whitened_rms REAL, holdout_whitened_rms REAL
            );
            CREATE TABLE ogp_od_parameters (
                name TEXT, value REAL, scale REAL, lower_bound REAL, upper_bound REAL
            );
            CREATE TABLE ogp_od_residuals (
                observation_id TEXT, partition TEXT, jd_utc REAL, component TEXT,
                residual REAL, whitened_residual REAL
            );
            """
            )
            state_metrics = dict(report["state_space_metrics"])
            connection.execute(
            "INSERT INTO ogp_od_run VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                report["ogp_regime"],
                report["propagator_name"],
                report["regime_classification"],
                report["parameterization"]["name"],
                int(bool(report["solver"]["success"])),
                report["quality_gates"]["evidence_status"],
                report["fit_sample_count"],
                report["holdout_sample_count"],
                state_metrics["fit_whitened_rms"],
                state_metrics["holdout_whitened_rms"],
            ),
            )
            connection.executemany(
            "INSERT INTO ogp_od_parameters VALUES (?, ?, ?, ?, ?)",
            [
                (row["name"], row["value"], row["scale"], row["lower"], row["upper"])
                for row in report["estimated_parameters"]
            ],
            )
            connection.executemany(
            "INSERT INTO ogp_od_residuals VALUES (?, ?, ?, ?, ?, ?)",
            [
                (
                    row["observation_id"],
                    row["partition"],
                    row["jd_utc"],
                    row["component"],
                    row["residual"],
                    row["whitened_residual"],
                )
                for row in residual_rows
            ],
            )
    write_output_text(str(report_path), json.dumps(report, indent=2, sort_keys=True) + "\n")
    from sim.interchange.adapters.ogp_od import build_ogp_mean_element_product

    product = build_ogp_mean_element_product(report_path, output_path=product_path)
    write_output_text(str(product_path), json.dumps(product, indent=2, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Fit OGP-SGP4/SDP4 mean elements to a native-TEME PV packet.")
    parser.add_argument("packet", help="Version-1 OGP PV observation packet JSON.")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    packet = json.loads(Path(args.packet).expanduser().read_text(encoding="utf-8"))
    report = run_ogp_pv_packet(packet, output_dir=args.output_dir)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
