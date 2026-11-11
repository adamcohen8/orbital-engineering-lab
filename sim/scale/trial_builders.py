from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from sim.dynamics.orbit.sgp4 import sgp4_propagate_teme
from sim.dynamics.orbit.tle import parse_tle_lines
from sim.scale.planner import ScalePlannerConfig
from sim.scale.sgp4_fit import fit_state_to_sgp4_mean_elements
from sim.scale.store import SensitivityTrialRecord, load_current_tle_states
from sim.utils.frames import ric_rect_state_to_eci


@dataclass(frozen=True)
class TrialBuilderProduct:
    product_id: str
    builder_kind: str
    status: str
    product: dict[str, Any]
    metrics: dict[str, Any]
    product_refs: tuple[str, ...]
    warnings: tuple[str, ...] = ()
    error: str = ""


def build_deployment_ric_state_product(
    cfg: ScalePlannerConfig,
    *,
    trial: SensitivityTrialRecord,
    propagate_sgp4: bool = False,
    propagation_duration_s: float | None = None,
    propagation_step_s: float | None = None,
) -> TrialBuilderProduct:
    if len(trial.target_object_ids) != 1:
        raise ValueError("deployment_ric_state builder requires exactly one deployer target object.")
    states = load_current_tle_states(cfg.store_path, object_ids=tuple(trial.target_object_ids))
    if len(states) != 1:
        raise ValueError("deployment_ric_state builder requires one current deployer TLE state.")
    state = states[0]
    elements = parse_tle_lines(state.line1, state.line2)
    start_jd = float(cfg.start_jd_utc)
    tsince_min = (start_jd - float(elements.epoch_jd_utc)) * 1440.0
    deployer = sgp4_propagate_teme(elements, tsince_min)
    if deployer.error:
        raise ValueError(deployer.error)

    speed_m_s = _required_float(trial.parameters, "deployment_speed_m_s")
    az_deg = _required_float(trial.parameters, "effective_az_deg")
    el_deg = _required_float(trial.parameters, "effective_el_deg")
    rel_velocity_ric_km_s = _ric_velocity_from_az_el(
        speed_m_s=speed_m_s,
        az_deg=az_deg,
        el_deg=el_deg,
    )
    rel_state_ric = np.hstack((np.zeros(3, dtype=float), rel_velocity_ric_km_s))
    deployed_state_teme = ric_rect_state_to_eci(
        rel_state_ric,
        np.array(deployer.position_teme_km, dtype=float),
        np.array(deployer.velocity_teme_km_s, dtype=float),
    )
    fit_cfg = dict(cfg.monte_carlo_trial_builder.get("sgp4_fit", {}) or {})
    fit = fit_state_to_sgp4_mean_elements(
        position_teme_km=deployed_state_teme[:3],
        velocity_teme_km_s=deployed_state_teme[3:],
        epoch_jd_utc=start_jd,
        template_elements=elements,
        arc_duration_s=float(fit_cfg.get("arc_duration_s", 120.0)),
        step_s=float(fit_cfg.get("step_s", 30.0)),
        position_sigma_km=float(fit_cfg.get("position_sigma_km", 0.001)),
        velocity_sigma_km_s=float(fit_cfg.get("velocity_sigma_km_s", 1.0e-4)),
        max_nfev=int(fit_cfg.get("max_nfev", 80)),
        acceptance_position_rms_m=float(fit_cfg.get("acceptance_position_rms_m", 1000.0)),
        acceptance_velocity_rms_mm_s=float(fit_cfg.get("acceptance_velocity_rms_mm_s", 3000.0)),
    )
    sgp4_status = "completed" if fit.get("status") == "completed" else "blocked_sgp4_fit_not_converged"
    product = {
        "schema_version": 1,
        "product_type": "deployment_ric_initial_state",
        "builder_kind": "deployment_ric_state",
        "frame": {
            "native_frame": "teme",
            "local_frame": "ric",
            "angle_convention": "azimuth_elevation",
            "az_reference_axis": "radial",
            "el_reference_plane": "radial_intrack",
        },
        "deployer": {
            "object_id": state.object_id,
            "od_solution_id": state.od_solution_id,
            "tle_epoch_jd_utc": state.tle_epoch_jd_utc,
            "state_jd_utc": start_jd,
            "position_teme_km": _vector(deployer.position_teme_km),
            "velocity_teme_km_s": _vector(deployer.velocity_teme_km_s),
        },
        "deployed": {
            "role": str(cfg.monte_carlo_trial_builder.get("deployed_object_role", "deployed_object")),
            "initial_state_jd_utc": start_jd,
            "position_teme_km": _vector(deployed_state_teme[:3]),
            "velocity_teme_km_s": _vector(deployed_state_teme[3:]),
            "relative_position_ric_km": [0.0, 0.0, 0.0],
            "relative_velocity_ric_km_s": _vector(rel_velocity_ric_km_s),
        },
        "parameters": dict(trial.parameters),
        "sgp4_product_status": sgp4_status,
        "sgp4_fit": fit,
    }
    propagation_summary: dict[str, Any] = {}
    if propagate_sgp4 and sgp4_status == "completed":
        propagation_summary = _propagate_fit_summary(
            deployer_elements=elements,
            fit_elements_payload=dict(fit["estimated_elements"]),
            epoch_jd_utc=start_jd,
            duration_s=float(cfg.duration_s if propagation_duration_s is None else propagation_duration_s),
            step_s=float(cfg.step_s if propagation_step_s is None else propagation_step_s),
        )
        product["sgp4_trial_propagation"] = propagation_summary
    digest = _digest(product)
    product_id = f"trial_product:{digest}"
    sgp4_product_id = f"sgp4_fit:{_digest(fit)}"
    product_refs = [product_id, sgp4_product_id]
    if propagation_summary:
        product_refs.append(f"sgp4_trial_propagation:{_digest(propagation_summary)}")
    metrics = {
        "deployment_speed_m_s": float(speed_m_s),
        "effective_az_deg": float(az_deg),
        "effective_el_deg": float(el_deg),
        "relative_speed_m_s": float(np.linalg.norm(rel_velocity_ric_km_s) * 1000.0),
        "sgp4_fit_status": str(fit.get("status")),
        "sgp4_fit_position_rms_m": float(
            dict(fit.get("final_state_error", {}) or {}).get("position_rms_m", float("nan"))
        ),
        "sgp4_fit_velocity_rms_mm_s": float(
            dict(fit.get("final_state_error", {}) or {}).get("velocity_rms_mm_s", float("nan"))
        ),
        "builder_product": product,
    }
    if propagation_summary:
        metrics.update(
            {
                "propagation_duration_s": float(propagation_summary["duration_s"]),
                "propagation_step_s": float(propagation_summary["step_s"]),
                "propagation_sample_count": int(propagation_summary["sample_count"]),
                "final_separation_km": float(propagation_summary["final_separation_km"]),
                "max_separation_km": float(propagation_summary["max_separation_km"]),
                "min_separation_km": float(propagation_summary["min_separation_km"]),
                "time_of_min_separation_s": float(propagation_summary["time_of_min_separation_s"]),
                "time_of_max_separation_s": float(propagation_summary["time_of_max_separation_s"]),
                "propagation_chunk_count": int(propagation_summary["chunk_count"]),
            }
        )
    return TrialBuilderProduct(
        product_id=product_id,
        builder_kind="deployment_ric_state",
        status="completed" if sgp4_status == "completed" else "blocked",
        product=product,
        metrics=metrics,
        product_refs=tuple(product_refs),
        warnings=(
            "Built RIC deployment initial state and synthetic SGP4 mean-element fit; this is not a catalog TLE.",
        ),
        error="" if sgp4_status == "completed" else sgp4_status,
    )


def _propagate_fit_summary(
    *,
    deployer_elements: Any,
    fit_elements_payload: dict[str, Any],
    epoch_jd_utc: float,
    duration_s: float,
    step_s: float,
) -> dict[str, Any]:
    deployed_elements = _fit_payload_to_elements(deployer_elements, fit_elements_payload)
    duration = max(float(duration_s), 0.0)
    step = max(float(step_s), 1.0)
    sample_count = int(np.floor(duration / step)) + 1
    times_s = np.arange(sample_count, dtype=float) * step
    if times_s[-1] < duration - 1.0e-9:
        times_s = np.append(times_s, duration)
    separations_km: list[float] = []
    final_relative_teme_km: list[float] = []
    for t_s in times_s:
        jd = float(epoch_jd_utc) + float(t_s) / 86400.0
        deployer = sgp4_propagate_teme(
            deployer_elements,
            (jd - float(deployer_elements.epoch_jd_utc)) * 1440.0,
        )
        deployed = sgp4_propagate_teme(
            deployed_elements,
            (jd - float(deployed_elements.epoch_jd_utc)) * 1440.0,
        )
        if deployer.error:
            raise ValueError(deployer.error)
        if deployed.error:
            raise ValueError(deployed.error)
        rel = np.array(deployed.position_teme_km, dtype=float) - np.array(deployer.position_teme_km, dtype=float)
        separations_km.append(float(np.linalg.norm(rel)))
        final_relative_teme_km = _vector(rel)
    separation_array = np.asarray(separations_km, dtype=float)
    min_index = int(np.argmin(separation_array))
    max_index = int(np.argmax(separation_array))
    return {
        "schema_version": 1,
        "product_type": "sgp4_trial_propagation_summary",
        "model": "sgp4",
        "backend": "serial",
        "frame": "teme",
        "epoch_jd_utc": float(epoch_jd_utc),
        "duration_s": duration,
        "step_s": step,
        "sample_count": int(len(times_s)),
        "initial_separation_km": float(separations_km[0]),
        "final_separation_km": float(separations_km[-1]),
        "min_separation_km": float(separation_array[min_index]),
        "max_separation_km": float(separation_array[max_index]),
        "time_of_min_separation_s": float(times_s[min_index]),
        "time_of_max_separation_s": float(times_s[max_index]),
        "chunk_count": 1,
        "max_chunk_sample_count": int(len(times_s)),
        "final_relative_position_teme_km": final_relative_teme_km,
        "non_claims": [
            "Propagation uses synthetic SGP4 mean elements generated from analyst-defined deployment assumptions.",
            "This is not an observed catalog TLE propagation.",
        ],
    }


def fit_payload_to_elements(template: Any, payload: dict[str, Any]) -> Any:
    return replace(
        template,
        epoch_jd_utc=float(payload["epoch_jd_utc"]),
        inclination_deg=float(payload["inclination_deg"]),
        raan_deg=float(payload["raan_deg"]),
        eccentricity=float(payload["eccentricity"]),
        argp_deg=float(payload["argp_deg"]),
        mean_anomaly_deg=float(payload["mean_anomaly_deg"]),
        mean_motion_rev_per_day=float(payload["mean_motion_rev_per_day"]),
        mean_motion_derivative_rev_per_day2=float(payload.get("mean_motion_derivative_rev_per_day2", 0.0)),
        mean_motion_second_derivative_rev_per_day3=float(
            payload.get("mean_motion_second_derivative_rev_per_day3", 0.0)
        ),
        bstar=float(payload.get("bstar", 0.0)),
    )


def _fit_payload_to_elements(template: Any, payload: dict[str, Any]) -> Any:
    return fit_payload_to_elements(template, payload)


def _ric_velocity_from_az_el(*, speed_m_s: float, az_deg: float, el_deg: float) -> np.ndarray:
    speed_km_s = float(speed_m_s) / 1000.0
    az = np.deg2rad(float(az_deg))
    el = np.deg2rad(float(el_deg))
    return speed_km_s * np.array(
        [
            np.cos(el) * np.cos(az),
            np.cos(el) * np.sin(az),
            np.sin(el),
        ],
        dtype=float,
    )


def _required_float(parameters: dict[str, Any], key: str) -> float:
    if key not in parameters:
        raise ValueError(f"deployment_ric_state builder requires trial parameter {key!r}.")
    value = float(parameters[key])
    if not np.isfinite(value):
        raise ValueError(f"deployment_ric_state builder parameter {key!r} must be finite.")
    return value


def _vector(value: Any) -> list[float]:
    return [float(item) for item in np.array(value, dtype=float).reshape(3)]


def _digest(payload: dict[str, Any]) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
