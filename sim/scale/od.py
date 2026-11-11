from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from sim.dynamics.orbit.tle import parse_tle_lines
from sim.estimation.ogp_od import fit_ogp_mean_elements_to_observations
from sim.estimation.sgp4_od import fit_sgp4_mean_elements_to_observations
from sim.scale.store import (
    OdSolutionRecord,
    load_latest_tle_states,
    load_track_observation_rows,
    write_od_solution_records,
)


def run_sgp4_od_from_track_observations(
    store_path: str | Path,
    *,
    object_ids: tuple[str, ...] = (),
    min_observations: int = 2,
    position_sigma_km: float = 0.001,
    velocity_sigma_km_s: float = 1.0e-6,
    fit_until_jd_utc: float | None = None,
    max_nfev: int = 100,
) -> dict[str, Any]:
    """Fit near-Earth SGP4 observations through the shared owner."""
    return _run_mean_element_od_from_track_observations(
        store_path,
        object_ids=object_ids,
        min_observations=min_observations,
        position_sigma_km=position_sigma_km,
        velocity_sigma_km_s=velocity_sigma_km_s,
        fit_until_jd_utc=fit_until_jd_utc,
        max_nfev=max_nfev,
        ogp_dispatch=False,
    )


def run_ogp_od_from_track_observations(
    store_path: str | Path,
    *,
    object_ids: tuple[str, ...] = (),
    min_observations: int = 2,
    position_sigma_km: float = 0.001,
    velocity_sigma_km_s: float = 1.0e-6,
    fit_until_jd_utc: float | None = None,
    max_nfev: int = 100,
    estimate_bstar: bool = False,
    bstar_prior_sigma: float | None = None,
) -> dict[str, Any]:
    """Fit regime-dispatched OGP observations through the shared owner."""
    return _run_mean_element_od_from_track_observations(
        store_path,
        object_ids=object_ids,
        min_observations=min_observations,
        position_sigma_km=position_sigma_km,
        velocity_sigma_km_s=velocity_sigma_km_s,
        fit_until_jd_utc=fit_until_jd_utc,
        max_nfev=max_nfev,
        ogp_dispatch=True,
        estimate_bstar=estimate_bstar,
        bstar_prior_sigma=bstar_prior_sigma,
    )


def _run_mean_element_od_from_track_observations(
    store_path: str | Path,
    *,
    object_ids: tuple[str, ...] = (),
    min_observations: int = 2,
    position_sigma_km: float = 0.001,
    velocity_sigma_km_s: float = 1.0e-6,
    fit_until_jd_utc: float | None = None,
    max_nfev: int = 100,
    estimate_bstar: bool = False,
    bstar_prior_sigma: float | None = None,
    ogp_dispatch: bool,
) -> dict[str, Any]:
    """Fit current SGP4 mean-element OD products from stored TEME track observations."""

    prefix = "ogp" if ogp_dispatch else "sgp4"
    store = Path(store_path).expanduser().resolve()
    observations_by_object = load_track_observation_rows(
        store,
        object_ids=object_ids,
        frame="teme",
        min_observations=max(int(min_observations), 2),
    )
    templates = load_latest_tle_states(store, object_ids=object_ids)
    solutions: list[OdSolutionRecord] = []
    object_results: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for object_id, rows in sorted(observations_by_object.items()):
        template = templates.get(object_id)
        if template is None:
            skipped.append({"object_id": object_id, "reason": "no_tle_template"})
            continue
        try:
            elements = parse_tle_lines(template.line1, template.line2)
            times = np.array([float(row["time_jd_utc"]) for row in rows], dtype=float)
            positions = np.array([row["position_teme_km"] for row in rows], dtype=float)
            velocities = np.array([row["velocity_teme_km_s"] for row in rows], dtype=float)
            fitter = fit_ogp_mean_elements_to_observations if ogp_dispatch else fit_sgp4_mean_elements_to_observations
            options = {"estimate_bstar": estimate_bstar, "bstar_prior_sigma": bstar_prior_sigma} if ogp_dispatch else {}
            fit = fitter(
                times_jd_utc=times,
                positions_teme_km=positions,
                velocities_teme_km_s=velocities,
                template_elements=elements,
                position_sigma_km=float(position_sigma_km),
                velocity_sigma_km_s=float(velocity_sigma_km_s),
                fit_until_jd_utc=fit_until_jd_utc,
                max_nfev=int(max_nfev),
                **options,
            )
        except Exception as exc:
            skipped.append({"object_id": object_id, "reason": str(exc)})
            continue
        if ogp_dispatch:
            # Normalize the shared fitter report at the persistence boundary.
            fit["epoch_jd_utc"] = fit["estimated_elements"]["epoch_jd_utc"]
            fit["initial_observation_residual"] = fit["state_space_metrics"]["initial_all"]
            fit["final_observation_residual"] = fit["state_space_metrics"]["final_fit"]
            fit["final_holdout_observation_residual"] = fit["state_space_metrics"]["final_holdout"]
            fit["initial_normalized_residual_rms"] = None
            fit["final_normalized_residual_rms"] = fit["state_space_metrics"]["fit_whitened_rms"]
            fit["final_holdout_normalized_observation_residual_rms"] = fit["state_space_metrics"]["holdout_whitened_rms"]
        source_input_ids = tuple(str(row["observation_id"]) for row in rows)
        input_digest = _od_input_digest(
            object_id=object_id,
            template_digest=template.input_digest,
            observation_digests=tuple(str(row.get("digest") or row["observation_id"]) for row in rows),
            position_sigma_km=float(position_sigma_km),
            velocity_sigma_km_s=float(velocity_sigma_km_s),
            fit_until_jd_utc=fit_until_jd_utc,
        )
        if ogp_dispatch:
            input_digest = hashlib.sha256(("ogp_mean_elements_fit_v1:" + input_digest + json.dumps({"estimate_bstar": estimate_bstar, "bstar_prior_sigma": bstar_prior_sigma, "max_nfev": max_nfev}, sort_keys=True)).encode()).hexdigest()
        od_solution_id = f"od:{prefix}_mean_elements_fit:{input_digest}"
        quality = dict(fit.get("quality_gates", {}) or {})
        warnings = list(quality.get("warnings", []) or [])
        quality.update(
            {
                "source": "track_observations",
                "optional_parameter_decisions": fit.get("optional_parameter_decisions"),
                "ogp_regime": fit.get("ogp_regime", "sgp4"),
                "propagator_name": fit.get("propagator_name", "OGP-SGP4"),
                "template_tle_digest": template.input_digest,
                "warnings": warnings,
            }
        )
        residual_summary = {
            "method": fit.get("method"),
            "observation_count": fit.get("observation_count"),
            "fit_sample_count": fit.get("fit_sample_count"),
            "holdout_sample_count": fit.get("holdout_sample_count"),
            "initial_observation_residual": fit.get("initial_observation_residual"),
            "final_observation_residual": fit.get("final_observation_residual"),
            "initial_normalized_residual_rms": fit.get("initial_normalized_residual_rms"),
            "final_normalized_residual_rms": fit.get("final_normalized_residual_rms"),
            "final_holdout_normalized_observation_residual_rms": fit.get(
                "final_holdout_normalized_observation_residual_rms"
            ),
            "ogp_regime": fit.get("ogp_regime", "sgp4"),
            "state_space_metrics": fit.get("state_space_metrics"),
            "mean_element_comparison": fit.get("mean_element_comparison"),
            "solver": fit.get("solver"),
        }
        if ogp_dispatch:
            # Keep native parameter covariance with its order/units and limits;
            # this is not an empirical six-state covariance claim.
            residual_summary["parameter_uncertainty"] = {
                key: fit.get(key) for key in (
                    "parameter_names", "parameter_units", "covariance_native",
                    "data_covariance_native", "covariance_evidence", "empirically_calibrated",
                )
            }
        usable = bool(dict(fit.get("solver", {}) or {}).get("success"))
        if ogp_dispatch:
            usable = usable and quality.get("evidence_status") == "ready_with_caveats" and all(
                quality.get(key) is True
                for key in ("solver_success", "data_full_rank", "fit_improved", "regime_stable")
            )
        status = "current_candidate" if usable else "review_required"
        solutions.append(
            OdSolutionRecord(
                od_solution_id=od_solution_id,
                object_id=object_id,
                method=f"{prefix}_mean_elements_fit",
                source_input_ids=source_input_ids,
                input_digest=input_digest,
                epoch_jd_utc=float(fit["epoch_jd_utc"]),
                native_frame="teme",
                state={},
                mean_elements=dict(fit.get("estimated_elements", {}) or {}),
                residual_summary=residual_summary,
                quality_flags=quality,
                status=status,
            )
        )
        object_results.append(
            {
                "object_id": object_id,
                "od_solution_id": od_solution_id,
                "optional_parameter_decisions": fit.get("optional_parameter_decisions"),
                "ogp_regime": fit.get("ogp_regime", "sgp4"),
                "propagator_name": fit.get("propagator_name", "OGP-SGP4"),
                "status": status,
                "observation_count": len(rows),
                "template_tle_digest": template.input_digest,
                "initial_position_rms_m": dict(fit.get("initial_observation_residual", {}) or {}).get("position_rms_m"),
                "final_position_rms_m": dict(fit.get("final_observation_residual", {}) or {}).get("position_rms_m"),
                "solver_success": bool(dict(fit.get("solver", {}) or {}).get("success")),
            }
        )
    write_summary = write_od_solution_records(store, solutions=solutions, update_current=True)
    return {
        "schema_version": 1,
        "status": "completed",
        "store_path": str(store),
        "objects_with_observations": len(observations_by_object),
        "objects_fit": len(solutions),
        "objects_skipped": len(skipped),
        "write_summary": write_summary,
        "objects": object_results,
        "skipped": skipped,
        "non_claims": [
            "Scale mean-element OD fits native TEME position/velocity observations within the selected propagation regime.",
            "Epoch and mean-motion derivative terms are inherited; OGP B* estimation requires explicit opt-in and observability gates.",
            "This is not sensor-frame OD, covariance-derived OD, catalog association, or operational tracking validation.",
        ],
    }


def _od_input_digest(
    *,
    object_id: str,
    template_digest: str,
    observation_digests: tuple[str, ...],
    position_sigma_km: float,
    velocity_sigma_km_s: float,
    fit_until_jd_utc: float | None,
) -> str:
    payload = {
        "object_id": object_id,
        "template_digest": template_digest,
        "observation_digests": list(observation_digests),
        "position_sigma_km": float(position_sigma_km),
        "velocity_sigma_km_s": float(velocity_sigma_km_s),
        "fit_until_jd_utc": fit_until_jd_utc,
        "method": "sgp4_mean_elements_fit_v0",
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
