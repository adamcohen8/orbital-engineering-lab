from __future__ import annotations

import copy
import csv
import json
import math
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from sim.config import SimulationScenarioConfig
from sim.dynamics.orbit.accelerations import OrbitContext
from sim.numeric_backend import normalize_numeric_backend
from sim.plotting.style import OELArtifactMetadata, get_oel_version, role_color, show_save_close_oel
from sim.single_run import _SingleRunEngine
from sim.utils.frames import eci_relative_to_ric_rect
from sim.utils.quaternion import quaternion_to_dcm_bn


@dataclass(frozen=True)
class CovarianceAnalysisResult:
    summary: dict[str, Any]
    object_rows: list[dict[str, Any]]
    object_covariance_rows: list[dict[str, Any]]
    pair_rows: list[dict[str, Any]]
    conjunction_rows: list[dict[str, Any]]
    artifacts: dict[str, str]


def run_covariance_analysis(
    *,
    config_path: str | Path | None,
    cfg: SimulationScenarioConfig,
    step_callback=None,
    numeric_backend: str | None = None,
) -> dict[str, Any]:
    """Run the nominal scenario and propagate configured orbit-state covariance.

    This is a Pro analysis workflow. It uses finite-difference state transition
    matrices around the nominal trajectory and the existing orbit propagator
    plugins. It does not model estimator updates or controller feedback
    sensitivity in this first version.
    """

    engine = _SingleRunEngine(cfg, step_callback=step_callback)
    while not engine.done:
        engine.step()

    parts = engine._build_payload_parts()
    payload = engine._payload_from_parts(parts)
    result = compute_covariance_analysis(
        cfg=cfg,
        engine=engine,
        parts=parts,
        numeric_backend=numeric_backend,
    )
    payload["summary"]["covariance_analysis"] = result.summary
    payload["summary"]["covariance_outputs"] = result.artifacts
    payload["extra_artifacts"] = {"covariance": result.artifacts}

    artifacts = engine._write_artifacts(payload, parts)
    _write_covariance_summary_patch(engine.outdir, payload["summary"])
    if bool(cfg.analysis.covariance.write_review_tables):
        review_db = engine.outdir / "review" / "run.sqlite"
        _write_covariance_review_tables(review_db, result)
        from sim.reporting.review_store import refresh_review_schema

        refresh_review_schema(review_db)

    wrapped = {
        "config_path": "" if config_path is None else str(Path(config_path).expanduser().resolve()),
        "scenario_name": cfg.scenario_name,
        "scenario_description": cfg.scenario_description,
        "monte_carlo": {"enabled": False},
        "analysis": {
            "enabled": True,
            "study_type": "covariance",
            "covariance": result.summary,
        },
        "run": dict(payload.get("summary", {}) or {}),
        "artifacts": {**artifacts, "covariance": result.artifacts},
    }
    return wrapped


def compute_covariance_analysis(
    *,
    cfg: SimulationScenarioConfig,
    engine: _SingleRunEngine,
    parts: Any,
    numeric_backend: str | None = None,
) -> CovarianceAnalysisResult:
    cov_cfg = cfg.analysis.covariance
    backend = str(
        numeric_backend
        if numeric_backend is not None
        else getattr(cov_cfg, "numeric_backend", "rust")
    ).strip().lower()
    if backend not in {"python", "rust"}:
        raise ValueError("covariance numeric_backend must be 'python' or 'rust'.")
    t_s = np.asarray(parts.t_s, dtype=float).reshape(-1)
    object_covariances: dict[str, np.ndarray] = {}
    object_rows: list[dict[str, Any]] = []
    object_covariance_rows: list[dict[str, Any]] = []
    final_by_object: dict[str, dict[str, Any]] = {}

    for object_id, object_cfg in cov_cfg.objects.items():
        if not bool(object_cfg.enabled):
            continue
        if object_id not in parts.truth_hist:
            raise ValueError(f"analysis.covariance.objects.{object_id} does not match an active object.")
        agent = engine.agents.get(object_id)
        if agent is None or getattr(agent, "kind", "") != "satellite":
            raise ValueError(f"analysis.covariance.objects.{object_id} must refer to an active satellite object.")
        hist = np.asarray(parts.truth_hist[object_id], dtype=float)
        thrust = np.asarray(parts.thrust_hist.get(object_id, np.zeros((hist.shape[0], 3))), dtype=float)
        cov_hist = _propagate_object_covariance(
            cfg=cfg,
            engine=engine,
            object_id=object_id,
            truth_hist=hist,
            thrust_hist=thrust,
            initial_covariance=_initial_covariance_matrix(object_cfg),
            numeric_backend=backend,
        )
        object_covariances[object_id] = cov_hist
        rows = _object_covariance_rows(object_id=object_id, t_s=t_s, cov_hist=cov_hist)
        object_rows.extend(rows)
        object_covariance_rows.extend(
            _object_state_covariance_rows(object_id=object_id, t_s=t_s, cov_hist=cov_hist)
        )
        if rows:
            final_by_object[object_id] = dict(rows[-1])

    pair_rows: list[dict[str, Any]] = []
    conjunction_rows: list[dict[str, Any]] = []
    final_by_pair: dict[str, dict[str, Any]] = {}
    final_by_conjunction_screen: dict[str, dict[str, Any]] = {}
    provenance = _handoff_provenance(cfg)
    for pair in cov_cfg.pairs:
        pair_key = f"{pair.deputy_id}:{pair.chief_id}"
        if pair.deputy_id not in object_covariances or pair.chief_id not in object_covariances:
            raise ValueError(
                f"analysis.covariance.pairs {pair_key} requires covariance objects for both participants."
            )
        deputy_hist = np.asarray(parts.truth_hist[pair.deputy_id], dtype=float)
        chief_hist = np.asarray(parts.truth_hist[pair.chief_id], dtype=float)
        rows = _pair_covariance_rows(
            deputy_id=pair.deputy_id,
            chief_id=pair.chief_id,
            t_s=t_s,
            deputy_hist=deputy_hist,
            chief_hist=chief_hist,
            deputy_cov=object_covariances[pair.deputy_id],
            chief_cov=object_covariances[pair.chief_id],
        )
        pair_rows.extend(rows)
        if rows:
            final_by_pair[pair_key] = dict(rows[-1])
        if bool(pair.collision_screening.enabled):
            screen_row = _conjunction_screening_row(
                deputy_id=pair.deputy_id,
                chief_id=pair.chief_id,
                t_s=t_s,
                deputy_hist=deputy_hist,
                chief_hist=chief_hist,
                deputy_cov=object_covariances[pair.deputy_id],
                chief_cov=object_covariances[pair.chief_id],
                hard_body_radius_km=float(pair.collision_screening.hard_body_radius_km),
                method=str(pair.collision_screening.method),
                numeric_backend=backend,
            )
            conjunction_rows.append(screen_row)
            final_by_conjunction_screen[pair_key] = dict(screen_row)

    outdir = Path(cfg.outputs.output_dir)
    artifacts = _write_covariance_artifacts(
        outdir=outdir,
        scenario_name=cfg.scenario_name,
        object_rows=object_rows,
        pair_rows=pair_rows,
        conjunction_rows=conjunction_rows,
        assumptions=_assumptions(cfg),
        provenance=provenance,
        final_by_object=final_by_object,
        final_by_pair=final_by_pair,
        final_by_conjunction_screen=final_by_conjunction_screen,
        style_name=str(dict(cfg.outputs.plots or {}).get("style", "oel_dark") or "oel_dark"),
        dpi=int(dict(cfg.outputs.plots or {}).get("dpi", 150) or 150),
    )
    summary = {
        "enabled": True,
        "method": "finite_difference_stm",
        "state": "eci_cartesian_6",
        "objects": sorted(object_covariances.keys()),
        "pairs": [f"{pair.deputy_id}:{pair.chief_id}" for pair in cov_cfg.pairs],
        "collision_screening_pairs": [
            f"{pair.deputy_id}:{pair.chief_id}" for pair in cov_cfg.pairs if bool(pair.collision_screening.enabled)
        ],
        "object_count": len(object_covariances),
        "pair_count": len(cov_cfg.pairs),
        "collision_screening_count": len(conjunction_rows),
        "samples": int(t_s.size),
        "assumptions": _assumptions(cfg),
        "provenance": provenance,
        "final_by_object": final_by_object,
        "final_by_pair": final_by_pair,
        "final_by_conjunction_screen": final_by_conjunction_screen,
    }
    if backend == "rust":
        summary["numeric_backend"] = "rust"
    return CovarianceAnalysisResult(
        summary=summary,
        object_rows=object_rows,
        object_covariance_rows=object_covariance_rows,
        pair_rows=pair_rows,
        conjunction_rows=conjunction_rows,
        artifacts=artifacts,
    )


def _propagate_object_covariance(
    *,
    cfg: SimulationScenarioConfig,
    engine: _SingleRunEngine,
    object_id: str,
    truth_hist: np.ndarray,
    thrust_hist: np.ndarray,
    initial_covariance: np.ndarray,
    numeric_backend: str = "rust",
) -> np.ndarray:
    n = int(truth_hist.shape[0])
    cov_hist = np.full((n, 6, 6), np.nan)
    cov_hist[0] = _sym_psd(initial_covariance)
    if n <= 1:
        return cov_hist
    fd_cfg = cfg.analysis.covariance.finite_difference
    q_cfg = cfg.analysis.covariance.process_noise
    agent = engine.agents[object_id]
    dynamics = agent.dynamics
    backend = normalize_numeric_backend(numeric_backend, error_message="covariance numeric_backend must be 'python' or 'rust'.")
    transition_history: list[np.ndarray] = []
    process_noise_history: list[np.ndarray] = []
    finite_difference_samples: list[np.ndarray] = []
    finite_difference_steps: list[np.ndarray] = []
    failed = False
    for i in range(n - 1):
        x_nom = np.asarray(truth_hist[i, :6], dtype=float).reshape(6)
        if not np.all(np.isfinite(x_nom)):
            failed = True
            break
        dt_s = float(engine.t_s[i + 1] - engine.t_s[i])
        # Applied commands are sampled at the interval end: row i+1 records
        # the command used to propagate truth from i to i+1.
        u = (
            np.zeros(3)
            if thrust_hist.shape[1] < 3
            else np.asarray(thrust_hist[i + 1, :3], dtype=float).reshape(3)
        )
        u = np.nan_to_num(u, nan=0.0, posinf=0.0, neginf=0.0)
        if backend == "rust":
            samples, steps = _finite_difference_samples(
                dynamics=dynamics,
                state_row=truth_hist[i],
                x_nom=x_nom,
                command_accel_eci_km_s2=u,
                env=engine.base_environment,
                t_s=float(engine.t_s[i]),
                dt_s=dt_s,
                position_step_km=float(fd_cfg.position_step_km),
                velocity_step_km_s=float(fd_cfg.velocity_step_km_s),
            )
            finite_difference_samples.append(samples)
            finite_difference_steps.append(steps)
        else:
            phi = _finite_difference_stm(
                dynamics=dynamics,
                state_row=truth_hist[i],
                x_nom=x_nom,
                command_accel_eci_km_s2=u,
                env=engine.base_environment,
                t_s=float(engine.t_s[i]),
                dt_s=dt_s,
                position_step_km=float(fd_cfg.position_step_km),
                velocity_step_km_s=float(fd_cfg.velocity_step_km_s),
                numeric_backend="python",
            )
        q = _process_noise_matrix(dt_s, float(q_cfg.acceleration_sigma_km_s2)) if bool(q_cfg.enabled) else np.zeros((6, 6))
        if backend == "rust":
            process_noise_history.append(q)
        else:
            cov_hist[i + 1] = _sym_psd(phi @ cov_hist[i] @ phi.T + q)
    if backend == "rust" and not failed and finite_difference_samples:
        from sim.rust_relative_backend import (
            finite_difference_stm_batch,
            propagate_covariance_history,
        )

        transition_history = list(
            finite_difference_stm_batch(
                np.asarray(finite_difference_samples, dtype=float),
                np.asarray(finite_difference_steps, dtype=float),
            )
        )
        propagated = propagate_covariance_history(
            cov_hist[0],
            np.asarray(transition_history, dtype=float),
            np.asarray(process_noise_history, dtype=float),
        )
        for index in range(min(n, propagated.shape[0])):
            cov_hist[index] = _sym_psd(propagated[index])
    return cov_hist


def _finite_difference_samples(
    *,
    dynamics: Any,
    state_row: np.ndarray,
    x_nom: np.ndarray,
    command_accel_eci_km_s2: np.ndarray,
    env: dict[str, Any],
    t_s: float,
    dt_s: float,
    position_step_km: float,
    velocity_step_km_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect authoritative plus/minus flow outputs for one interval."""

    steps = np.array([position_step_km] * 3 + [velocity_step_km_s] * 3, dtype=float)
    from sim.dynamics.orbit.propagator import OrbitPropagator, j2_plugin

    propagator = getattr(dynamics, "orbit_propagator", None)
    if (
        type(propagator) is OrbitPropagator
        and propagator.model == "two_body"
        and propagator.integrator == "rk4"
        and tuple(propagator.plugins) in ((), (j2_plugin,))
    ):
        from sim.rust_relative_backend import try_covariance_orbit_trials

        orbit_dt = dynamics._effective_substep(getattr(dynamics, "orbit_substep_s", None), dt_s)
        native = try_covariance_orbit_trials(
            x_nom,
            steps,
            np.asarray(tuple(dynamics._substep_sequence(dt_s, orbit_dt)), dtype=float),
            mu_km3_s2=float(dynamics.mu_km3_s2),
            include_j2=bool(propagator.plugins),
            command_accel_eci_km_s2=command_accel_eci_km_s2,
        )
        if native is not None:
            return native, steps
    plus_minus = np.zeros((6, 12), dtype=float)
    for j, h in enumerate(steps):
        dx = np.zeros(6, dtype=float)
        dx[j] = float(h)
        f_plus = _propagate_orbit_once(
            dynamics=dynamics,
            state_row=state_row,
            x_eci=x_nom + dx,
            command_accel_eci_km_s2=command_accel_eci_km_s2,
            env=env,
            t_s=t_s,
            dt_s=dt_s,
        )
        f_minus = _propagate_orbit_once(
            dynamics=dynamics,
            state_row=state_row,
            x_eci=x_nom - dx,
            command_accel_eci_km_s2=command_accel_eci_km_s2,
            env=env,
            t_s=t_s,
            dt_s=dt_s,
        )
        plus_minus[j, :6] = f_plus
        plus_minus[j, 6:] = f_minus
    return plus_minus, steps


def _finite_difference_stm(
    *,
    dynamics: Any,
    state_row: np.ndarray,
    x_nom: np.ndarray,
    command_accel_eci_km_s2: np.ndarray,
    env: dict[str, Any],
    t_s: float,
    dt_s: float,
    position_step_km: float,
    velocity_step_km_s: float,
    numeric_backend: str = "rust",
) -> np.ndarray:
    phi = np.zeros((6, 6), dtype=float)
    steps = np.array([position_step_km] * 3 + [velocity_step_km_s] * 3, dtype=float)
    plus_minus = np.zeros((6, 12), dtype=float)
    for j, h in enumerate(steps):
        dx = np.zeros(6, dtype=float)
        dx[j] = float(h)
        f_plus = _propagate_orbit_once(
            dynamics=dynamics,
            state_row=state_row,
            x_eci=x_nom + dx,
            command_accel_eci_km_s2=command_accel_eci_km_s2,
            env=env,
            t_s=t_s,
            dt_s=dt_s,
        )
        f_minus = _propagate_orbit_once(
            dynamics=dynamics,
            state_row=state_row,
            x_eci=x_nom - dx,
            command_accel_eci_km_s2=command_accel_eci_km_s2,
            env=env,
            t_s=t_s,
            dt_s=dt_s,
        )
        plus_minus[j, :6] = f_plus
        plus_minus[j, 6:] = f_minus
        if str(numeric_backend).strip().lower() == "python":
            phi[:, j] = (f_plus - f_minus) / (2.0 * float(h))
    if str(numeric_backend).strip().lower() == "rust":
        from sim.rust_relative_backend import finite_difference_stm

        return finite_difference_stm(plus_minus, steps)
    return phi


def _propagate_orbit_once(
    *,
    dynamics: Any,
    state_row: np.ndarray,
    x_eci: np.ndarray,
    command_accel_eci_km_s2: np.ndarray,
    env: dict[str, Any],
    t_s: float,
    dt_s: float,
) -> np.ndarray:
    env_local = dict(env)
    if getattr(dynamics, "drag_area_m2", None) is not None:
        env_local["drag_area_m2"] = float(dynamics.drag_area_m2)
    if getattr(dynamics, "lift_area_m2", None) is not None:
        env_local["lift_area_m2"] = float(dynamics.lift_area_m2)
    if getattr(dynamics, "srp_area_m2", None) is not None:
        env_local["srp_area_m2"] = float(dynamics.srp_area_m2)
    lift_axis = getattr(dynamics, "lift_axis_body", None)
    if lift_axis is not None and float(getattr(dynamics, "lift_coefficient", 0.0)) != 0.0 and state_row.size >= 10:
        c_bn = quaternion_to_dcm_bn(np.asarray(state_row[6:10], dtype=float))
        axis = np.asarray(lift_axis, dtype=float).reshape(3)
        norm = float(np.linalg.norm(axis))
        if norm > 0.0:
            env_local["lift_coefficient"] = float(dynamics.lift_coefficient)
            env_local["lift_direction_eci"] = c_bn.T @ (axis / norm)
    mass_kg = float(state_row[13]) if state_row.size > 13 and np.isfinite(state_row[13]) else 1.0
    orbit_ctx = OrbitContext(
        mu_km3_s2=float(dynamics.mu_km3_s2),
        mass_kg=max(mass_kg, 1e-12),
        area_m2=float(dynamics.area_m2),
        cd=float(dynamics.cd),
        cr=float(dynamics.cr),
    )
    propagator = copy.deepcopy(dynamics.orbit_propagator)
    orbit_dt = dynamics._effective_substep(getattr(dynamics, "orbit_substep_s", None), dt_s)
    out = np.asarray(x_eci, dtype=float).reshape(6).copy()
    t_local = float(t_s)
    for h in dynamics._substep_sequence(dt_s, orbit_dt):
        out = propagator.propagate(
            x_eci=out,
            dt_s=float(h),
            t_s=t_local,
            command_accel_eci_km_s2=np.asarray(command_accel_eci_km_s2, dtype=float).reshape(3),
            env=env_local,
            ctx=orbit_ctx,
        )
        t_local += float(h)
    return np.asarray(out, dtype=float).reshape(6)


def _initial_covariance_matrix(object_cfg: Any) -> np.ndarray:
    if object_cfg.covariance:
        matrix = np.asarray(object_cfg.covariance, dtype=float).reshape(6, 6)
    elif object_cfg.diagonal:
        matrix = np.diag(np.asarray(object_cfg.diagonal, dtype=float).reshape(6))
    else:
        pos_sigma = 0.0 if object_cfg.position_sigma_km is None else float(object_cfg.position_sigma_km)
        vel_sigma = 0.0 if object_cfg.velocity_sigma_km_s is None else float(object_cfg.velocity_sigma_km_s)
        matrix = np.diag([pos_sigma**2] * 3 + [vel_sigma**2] * 3)
    return _sym_psd(matrix)


def _process_noise_matrix(dt_s: float, accel_sigma_km_s2: float) -> np.ndarray:
    sigma2 = float(accel_sigma_km_s2) ** 2
    dt = float(dt_s)
    q = np.zeros((6, 6), dtype=float)
    q[:3, :3] = (dt**4 / 4.0) * sigma2 * np.eye(3)
    q[:3, 3:] = (dt**3 / 2.0) * sigma2 * np.eye(3)
    q[3:, :3] = q[:3, 3:].T
    q[3:, 3:] = (dt**2) * sigma2 * np.eye(3)
    return q


def _sym_psd(matrix: np.ndarray) -> np.ndarray:
    p = 0.5 * (np.asarray(matrix, dtype=float).reshape(6, 6) + np.asarray(matrix, dtype=float).reshape(6, 6).T)
    if not np.all(np.isfinite(p)):
        raise ValueError("Covariance matrix contains non-finite values.")
    eigvals = np.linalg.eigvalsh(p)
    if not np.all(np.isfinite(eigvals)) or float(np.min(eigvals)) < -_psd_eigenvalue_tolerance(eigvals):
        raise ValueError("Covariance matrix must be positive semidefinite.")
    return p


def _sym_square(matrix: np.ndarray) -> np.ndarray:
    p = np.asarray(matrix, dtype=float)
    if p.ndim != 2 or p.shape[0] != p.shape[1]:
        raise ValueError("Covariance matrix must be square.")
    p = 0.5 * (p + p.T)
    if not np.all(np.isfinite(p)):
        raise ValueError("Covariance matrix contains non-finite values.")
    eigvals = np.linalg.eigvalsh(p)
    if not np.all(np.isfinite(eigvals)) or float(np.min(eigvals)) < -_psd_eigenvalue_tolerance(eigvals):
        raise ValueError("Covariance matrix must be positive semidefinite.")
    return p


def _psd_eigenvalue_tolerance(eigvals: np.ndarray) -> float:
    """Return a PSD roundoff tolerance relative to this matrix's spectrum."""

    return 1e-12 * float(np.max(np.abs(eigvals)))


def _regularized_2d_covariance(matrix: np.ndarray) -> np.ndarray:
    p = _sym_square(np.asarray(matrix, dtype=float).reshape(2, 2))
    eigvals = np.linalg.eigvalsh(p)
    floor = max(float(np.max(eigvals)) * 1e-12, 1e-18)
    if float(np.min(eigvals)) < floor:
        p = p + (floor - float(np.min(eigvals))) * np.eye(2)
    return _sym_square(p)


def _object_covariance_rows(*, object_id: str, t_s: np.ndarray, cov_hist: np.ndarray) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i in range(int(min(t_s.size, cov_hist.shape[0]))):
        p = cov_hist[i]
        if not np.all(np.isfinite(p)):
            continue
        p = _sym_psd(p)
        pos = p[:3, :3]
        vel = p[3:, 3:]
        eig = np.linalg.eigvalsh(p)
        rows.append(
            {
                "sample_index": i,
                "time_s": float(t_s[i]),
                "object_id": object_id,
                "sigma_x_eci_km": _sqrt_nonnegative(pos[0, 0]),
                "sigma_y_eci_km": _sqrt_nonnegative(pos[1, 1]),
                "sigma_z_eci_km": _sqrt_nonnegative(pos[2, 2]),
                "sigma_vx_eci_km_s": _sqrt_nonnegative(vel[0, 0]),
                "sigma_vy_eci_km_s": _sqrt_nonnegative(vel[1, 1]),
                "sigma_vz_eci_km_s": _sqrt_nonnegative(vel[2, 2]),
                "sigma_pos_3d_km": _sqrt_nonnegative(float(np.trace(pos))),
                "sigma_vel_3d_km_s": _sqrt_nonnegative(float(np.trace(vel))),
                "condition_number": _condition_number(p),
                "min_eigenvalue": float(np.min(eig)),
                "max_eigenvalue": float(np.max(eig)),
            }
        )
    return rows


def _object_state_covariance_rows(
    *, object_id: str, t_s: np.ndarray, cov_hist: np.ndarray
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    component_order = json.dumps(["x", "y", "z", "vx", "vy", "vz"], separators=(",", ":"))
    units = json.dumps(["km", "km", "km", "km/s", "km/s", "km/s"], separators=(",", ":"))
    for i in range(int(min(t_s.size, cov_hist.shape[0]))):
        matrix = np.asarray(cov_hist[i], dtype=float)
        if matrix.shape != (6, 6) or not np.all(np.isfinite(matrix)):
            continue
        matrix = _sym_psd(matrix)
        rows.append(
            {
                "sample_index": i,
                "time_s": float(t_s[i]),
                "object_id": object_id,
                "frame": "ECI",
                "component_order_json": component_order,
                "units_json": units,
                "covariance_json": json.dumps(matrix.tolist(), separators=(",", ":"), allow_nan=False),
                "mathematically_valid": 1,
                "calibrated": 0,
                "calibration_scope": None,
                "source": "finite_difference_stm",
            }
        )
    return rows


def _pair_covariance_rows(
    *,
    deputy_id: str,
    chief_id: str,
    t_s: np.ndarray,
    deputy_hist: np.ndarray,
    chief_hist: np.ndarray,
    deputy_cov: np.ndarray,
    chief_cov: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    n = int(min(t_s.size, deputy_hist.shape[0], chief_hist.shape[0], deputy_cov.shape[0], chief_cov.shape[0]))
    for i in range(n):
        dep = np.asarray(deputy_hist[i, :6], dtype=float).reshape(6)
        chief = np.asarray(chief_hist[i, :6], dtype=float).reshape(6)
        rel = eci_relative_to_ric_rect(dep, chief)
        j = _relative_state_jacobian(dep, chief)
        p_abs = np.zeros((12, 12), dtype=float)
        p_abs[:6, :6] = deputy_cov[i]
        p_abs[6:, 6:] = chief_cov[i]
        p_rel = _sym_psd(j @ p_abs @ j.T)
        pos = p_rel[:3, :3]
        range_km = float(np.linalg.norm(rel[:3]))
        mahalanobis = _mahalanobis(rel[:3], pos)
        rows.append(
            {
                "sample_index": i,
                "time_s": float(t_s[i]),
                "deputy_id": deputy_id,
                "chief_id": chief_id,
                "r_radial_km": float(rel[0]),
                "i_intrack_km": float(rel[1]),
                "c_crosstrack_km": float(rel[2]),
                "range_km": range_km,
                "sigma_r_radial_km": _sqrt_nonnegative(pos[0, 0]),
                "sigma_i_intrack_km": _sqrt_nonnegative(pos[1, 1]),
                "sigma_c_crosstrack_km": _sqrt_nonnegative(pos[2, 2]),
                "sigma_rel_pos_3d_km": _sqrt_nonnegative(float(np.trace(pos))),
                "mahalanobis_distance_pos": mahalanobis,
                "range_to_sigma_pos_3d": range_km / _sqrt_nonnegative(float(np.trace(pos)))
                if float(np.trace(pos)) > 0.0
                else None,
            }
        )
    return rows


def _conjunction_screening_row(
    *,
    deputy_id: str,
    chief_id: str,
    t_s: np.ndarray,
    deputy_hist: np.ndarray,
    chief_hist: np.ndarray,
    deputy_cov: np.ndarray,
    chief_cov: np.ndarray,
    hard_body_radius_km: float,
    method: str,
    numeric_backend: str = "rust",
) -> dict[str, Any]:
    ca = _closest_approach(t_s=t_s, deputy_hist=deputy_hist, chief_hist=chief_hist)
    i0 = int(ca["sample_index"])
    alpha = float(ca["interval_fraction"])
    i1 = min(i0 + 1, int(deputy_cov.shape[0]) - 1, int(chief_cov.shape[0]) - 1)
    p_dep = (1.0 - alpha) * np.asarray(deputy_cov[i0, :3, :3], dtype=float) + alpha * np.asarray(
        deputy_cov[i1, :3, :3], dtype=float
    )
    p_chief = (1.0 - alpha) * np.asarray(chief_cov[i0, :3, :3], dtype=float) + alpha * np.asarray(
        chief_cov[i1, :3, :3], dtype=float
    )
    p_rel_eci = _sym_square(p_dep + p_chief)
    rel_pos = np.asarray(ca["relative_position_eci_km"], dtype=float).reshape(3)
    rel_vel = np.asarray(ca["relative_velocity_eci_km_s"], dtype=float).reshape(3)
    basis = _encounter_plane_basis(rel_pos=rel_pos, rel_vel=rel_vel)
    e = np.vstack((basis[0], basis[1]))
    mean_plane = e @ rel_pos
    cov_plane = _regularized_2d_covariance(e @ p_rel_eci @ e.T)
    pc = _small_object_collision_probability(
        mean_plane,
        cov_plane,
        float(hard_body_radius_km),
        numeric_backend=numeric_backend,
    )
    eig = np.linalg.eigvalsh(cov_plane)
    mahalanobis_plane = _mahalanobis(mean_plane, cov_plane)
    return {
        "deputy_id": deputy_id,
        "chief_id": chief_id,
        "method": method,
        "screening_label": "small-hard-body encounter-plane screening",
        "time_of_closest_approach_s": float(ca["time_s"]),
        "sample_index": i0,
        "interval_fraction": alpha,
        "miss_distance_km": float(ca["miss_distance_km"]),
        "relative_speed_km_s": float(ca["relative_speed_km_s"]),
        "hard_body_radius_km": float(hard_body_radius_km),
        "hard_body_radius_m": float(hard_body_radius_km) * 1000.0,
        "pc_screening": pc,
        "pc_log10": math.log10(pc) if pc is not None and pc > 0.0 else None,
        "mahalanobis_distance_plane": mahalanobis_plane,
        "plane_miss_x_km": float(mean_plane[0]),
        "plane_miss_y_km": float(mean_plane[1]),
        "plane_sigma_major_km": _sqrt_nonnegative(float(np.max(eig))),
        "plane_sigma_minor_km": _sqrt_nonnegative(float(np.min(eig))),
        "plane_cov_xx_km2": float(cov_plane[0, 0]),
        "plane_cov_xy_km2": float(cov_plane[0, 1]),
        "plane_cov_yy_km2": float(cov_plane[1, 1]),
        "range_to_hard_body_radius": float(ca["miss_distance_km"]) / float(hard_body_radius_km),
    }


def _closest_approach(
    *,
    t_s: np.ndarray,
    deputy_hist: np.ndarray,
    chief_hist: np.ndarray,
) -> dict[str, Any]:
    n = int(min(t_s.size, deputy_hist.shape[0], chief_hist.shape[0]))
    if n <= 0:
        raise ValueError("Cannot screen conjunction risk without trajectory samples.")
    best: dict[str, Any] | None = None
    for i in range(max(n - 1, 1)):
        i0 = min(i, n - 1)
        i1 = min(i + 1, n - 1)
        t0 = float(t_s[i0])
        dt = max(float(t_s[i1] - t_s[i0]), 0.0)
        r0 = np.asarray(deputy_hist[i0, :3], dtype=float) - np.asarray(chief_hist[i0, :3], dtype=float)
        if i1 != i0 and dt > 0.0:
            r1 = np.asarray(deputy_hist[i1, :3], dtype=float) - np.asarray(chief_hist[i1, :3], dtype=float)
            v_rel = (r1 - r0) / dt
        else:
            v_rel = np.asarray(deputy_hist[i0, 3:6], dtype=float) - np.asarray(chief_hist[i0, 3:6], dtype=float)
        speed2 = float(v_rel @ v_rel)
        tau = 0.0 if speed2 <= 0.0 or dt <= 0.0 else min(max(-float(r0 @ v_rel) / speed2, 0.0), dt)
        rel_pos = r0 + v_rel * tau
        miss = float(np.linalg.norm(rel_pos))
        candidate = {
            "sample_index": i0,
            "interval_fraction": 0.0 if dt <= 0.0 else tau / dt,
            "time_s": t0 + tau,
            "miss_distance_km": miss,
            "relative_position_eci_km": rel_pos,
            "relative_velocity_eci_km_s": v_rel,
            "relative_speed_km_s": float(math.sqrt(max(speed2, 0.0))),
        }
        if best is None or miss < float(best["miss_distance_km"]):
            best = candidate
    assert best is not None
    return best


def _encounter_plane_basis(*, rel_pos: np.ndarray, rel_vel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    speed = float(np.linalg.norm(rel_vel))
    if speed > 1e-12:
        normal = rel_vel / speed
    else:
        pos_norm = float(np.linalg.norm(rel_pos))
        normal = rel_pos / pos_norm if pos_norm > 1e-12 else np.array([0.0, 0.0, 1.0], dtype=float)
    e1 = rel_pos - float(rel_pos @ normal) * normal
    e1_norm = float(np.linalg.norm(e1))
    if e1_norm <= 1e-12:
        seed = np.array([1.0, 0.0, 0.0], dtype=float)
        if abs(float(seed @ normal)) > 0.9:
            seed = np.array([0.0, 1.0, 0.0], dtype=float)
        e1 = seed - float(seed @ normal) * normal
        e1_norm = float(np.linalg.norm(e1))
    e1 = e1 / max(e1_norm, 1e-12)
    e2 = np.cross(normal, e1)
    e2 = e2 / max(float(np.linalg.norm(e2)), 1e-12)
    return e1, e2


def _small_object_collision_probability(
    mean_plane_km: np.ndarray,
    covariance_plane_km2: np.ndarray,
    hard_body_radius_km: float,
    *,
    numeric_backend: str = "rust",
) -> float | None:
    if str(numeric_backend).strip().lower() == "rust":
        from sim.rust_relative_backend import small_object_collision_probability

        return small_object_collision_probability(
            mean_plane_km,
            covariance_plane_km2,
            float(hard_body_radius_km),
        )
    cov = _regularized_2d_covariance(covariance_plane_km2)
    try:
        det = float(np.linalg.det(cov))
        inv = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        return None
    if det <= 0.0 or not math.isfinite(det):
        return None
    exponent = -0.5 * float(mean_plane_km.T @ inv @ mean_plane_km)
    if exponent < -745.0:
        return 0.0
    density_at_origin = math.exp(exponent) / (2.0 * math.pi * math.sqrt(det))
    pc = math.pi * float(hard_body_radius_km) ** 2 * density_at_origin
    if not math.isfinite(pc):
        return None
    return float(min(max(pc, 0.0), 1.0))


def _relative_state_jacobian(deputy_state: np.ndarray, chief_state: np.ndarray) -> np.ndarray:
    steps = np.array([1e-3] * 3 + [1e-6] * 3 + [1e-3] * 3 + [1e-6] * 3, dtype=float)
    base = np.hstack((deputy_state, chief_state))
    jac = np.zeros((6, 12), dtype=float)
    for j, h in enumerate(steps):
        dx = np.zeros(12, dtype=float)
        dx[j] = float(h)
        plus = base + dx
        minus = base - dx
        y_plus = eci_relative_to_ric_rect(plus[:6], plus[6:])
        y_minus = eci_relative_to_ric_rect(minus[:6], minus[6:])
        jac[:, j] = (y_plus - y_minus) / (2.0 * float(h))
    return jac


def _write_covariance_artifacts(
    *,
    outdir: Path,
    scenario_name: str,
    object_rows: list[dict[str, Any]],
    pair_rows: list[dict[str, Any]],
    conjunction_rows: list[dict[str, Any]],
    assumptions: dict[str, Any],
    provenance: dict[str, Any],
    final_by_object: dict[str, dict[str, Any]],
    final_by_pair: dict[str, dict[str, Any]],
    final_by_conjunction_screen: dict[str, dict[str, Any]],
    style_name: str,
    dpi: int,
) -> dict[str, str]:
    outdir.mkdir(parents=True, exist_ok=True)
    summary_path = outdir / "covariance_summary.json"
    object_csv = outdir / "covariance_object_history.csv"
    pair_csv = outdir / "covariance_pair_history.csv"
    conjunction_csv = outdir / "conjunction_screening.csv"
    conjunction_json = outdir / "conjunction_screening.json"
    report_path = outdir / "covariance_report.md"
    generated_utc = _utc_stamp()
    plot_paths = _write_covariance_plots(
        outdir=outdir,
        scenario_name=scenario_name,
        generated_utc=generated_utc,
        object_rows=object_rows,
        pair_rows=pair_rows,
        conjunction_rows=conjunction_rows,
        style_name=style_name,
        dpi=dpi,
    )
    summary = {
        "scenario_name": scenario_name,
        "generated_utc": generated_utc,
        "assumptions": assumptions,
        "provenance": provenance,
        "final_by_object": final_by_object,
        "final_by_pair": final_by_pair,
        "final_by_conjunction_screen": final_by_conjunction_screen,
        "object_row_count": len(object_rows),
        "pair_row_count": len(pair_rows),
        "conjunction_screening_row_count": len(conjunction_rows),
        "plots": plot_paths,
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    conjunction_json.write_text(json.dumps({"rows": conjunction_rows}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(object_csv, object_rows)
    _write_csv(pair_csv, pair_rows)
    _write_csv(conjunction_csv, conjunction_rows)
    report_path.write_text(_covariance_report(summary), encoding="utf-8")
    artifacts = {
        "summary_json": str(summary_path),
        "object_history_csv": str(object_csv),
        "pair_history_csv": str(pair_csv),
        "report_md": str(report_path),
        **plot_paths,
    }
    if conjunction_rows:
        artifacts["conjunction_screening_csv"] = str(conjunction_csv)
        artifacts["conjunction_screening_json"] = str(conjunction_json)
    return artifacts


def _write_covariance_plots(
    *,
    outdir: Path,
    scenario_name: str,
    generated_utc: str,
    object_rows: list[dict[str, Any]],
    pair_rows: list[dict[str, Any]],
    conjunction_rows: list[dict[str, Any]],
    style_name: str,
    dpi: int,
) -> dict[str, str]:
    paths: dict[str, str] = {}
    if object_rows:
        path = outdir / "covariance_object_sigma.png"
        _plot_object_covariance(
            object_rows=object_rows,
            path=path,
            scenario_name=scenario_name,
            generated_utc=generated_utc,
            style_name=style_name,
            dpi=dpi,
        )
        paths["object_sigma_png"] = str(path)
    if pair_rows:
        path = outdir / "covariance_relative_screen.png"
        _plot_relative_covariance(
            pair_rows=pair_rows,
            path=path,
            scenario_name=scenario_name,
            generated_utc=generated_utc,
            style_name=style_name,
            dpi=dpi,
        )
        paths["relative_screen_png"] = str(path)
    if conjunction_rows:
        path = outdir / "conjunction_screening.png"
        _plot_conjunction_screening(
            pair_rows=pair_rows,
            conjunction_rows=conjunction_rows,
            path=path,
            scenario_name=scenario_name,
            generated_utc=generated_utc,
            style_name=style_name,
            dpi=dpi,
        )
        paths["conjunction_screening_png"] = str(path)
    return paths


def _plot_object_covariance(
    *,
    object_rows: list[dict[str, Any]],
    path: Path,
    scenario_name: str,
    generated_utc: str,
    style_name: str,
    dpi: int,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    by_object: dict[str, list[dict[str, Any]]] = {}
    for row in object_rows:
        by_object.setdefault(str(row.get("object_id", "")), []).append(row)
    for object_id, rows in sorted(by_object.items()):
        rows = sorted(rows, key=lambda item: float(item.get("time_s", 0.0)))
        t = [float(row.get("time_s", 0.0)) for row in rows]
        sigma_pos = [float(row.get("sigma_pos_3d_km", 0.0)) for row in rows]
        sigma_vel = [float(row.get("sigma_vel_3d_km_s", 0.0)) * 1000.0 for row in rows]
        color = role_color(object_id, style_name=style_name)
        ax.plot(t, sigma_pos, label=f"{object_id} position sigma RSS", color=color)
        ax.plot(t, sigma_vel, label=f"{object_id} velocity sigma (m/s)", linestyle="--", color=color, alpha=0.8)
    ax.set_title("Covariance Growth By Object")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Sigma magnitude")
    ax.legend(loc="best")
    metadata = OELArtifactMetadata(
        scenario_name=scenario_name,
        generated_utc=generated_utc,
        version=get_oel_version(),
        artifact_id="covariance_object_sigma",
    )
    show_save_close_oel(
        fig,
        mode="save",
        out_path=path,
        dpi=dpi,
        artifact_id="covariance_object_sigma",
        metadata=metadata,
        style_name=style_name,
        plt_module=plt,
        close=True,
        bbox_inches="tight",
    )


def _plot_relative_covariance(
    *,
    pair_rows: list[dict[str, Any]],
    path: Path,
    scenario_name: str,
    generated_utc: str,
    style_name: str,
    dpi: int,
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(9.5, 7.0), sharex=True)
    by_pair: dict[str, list[dict[str, Any]]] = {}
    for row in pair_rows:
        pair_id = f"{row.get('deputy_id', '')}:{row.get('chief_id', '')}"
        by_pair.setdefault(pair_id, []).append(row)
    for pair_id, rows in sorted(by_pair.items()):
        rows = sorted(rows, key=lambda item: float(item.get("time_s", 0.0)))
        t = [float(row.get("time_s", 0.0)) for row in rows]
        ranges = [float(row.get("range_km", 0.0)) for row in rows]
        sigma_rel = [float(row.get("sigma_rel_pos_3d_km", 0.0)) for row in rows]
        mahal = [float(row.get("mahalanobis_distance_pos", 0.0) or 0.0) for row in rows]
        axes[0].plot(t, ranges, label=f"{pair_id} range", color=role_color("actual", style_name=style_name))
        axes[0].plot(t, sigma_rel, label=f"{pair_id} relative position sigma", color=role_color("warning", style_name=style_name))
        axes[1].plot(t, mahal, label=f"{pair_id} Mahalanobis position distance", color=role_color("safety_zone", style_name=style_name))
    axes[0].set_title("Relative Encounter Covariance Screen")
    axes[0].set_ylabel("km")
    axes[0].legend(loc="best")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("sigma distance")
    axes[1].legend(loc="best")
    metadata = OELArtifactMetadata(
        scenario_name=scenario_name,
        generated_utc=generated_utc,
        version=get_oel_version(),
        artifact_id="covariance_relative_screen",
    )
    show_save_close_oel(
        fig,
        mode="save",
        out_path=path,
        dpi=dpi,
        artifact_id="covariance_relative_screen",
        metadata=metadata,
        style_name=style_name,
        plt_module=plt,
        close=True,
        bbox_inches="tight",
    )


def _plot_conjunction_screening(
    *,
    pair_rows: list[dict[str, Any]],
    conjunction_rows: list[dict[str, Any]],
    path: Path,
    scenario_name: str,
    generated_utc: str,
    style_name: str,
    dpi: int,
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(9.5, 7.2), constrained_layout=True)
    by_pair: dict[str, list[dict[str, Any]]] = {}
    for row in pair_rows:
        pair_id = f"{row.get('deputy_id', '')}:{row.get('chief_id', '')}"
        by_pair.setdefault(pair_id, []).append(row)
    for row in conjunction_rows:
        pair_id = f"{row.get('deputy_id', '')}:{row.get('chief_id', '')}"
        rows = sorted(by_pair.get(pair_id, []), key=lambda item: float(item.get("time_s", 0.0)))
        if rows:
            t = [float(item.get("time_s", 0.0)) for item in rows]
            ranges = [float(item.get("range_km", 0.0)) for item in rows]
            axes[0].plot(t, ranges, label=f"{pair_id} range", color=role_color("actual", style_name=style_name))
        tca = float(row.get("time_of_closest_approach_s", 0.0))
        miss = float(row.get("miss_distance_km", 0.0))
        hbr = float(row.get("hard_body_radius_km", 0.0))
        axes[0].axvline(tca, linestyle="--", color=role_color("warning", style_name=style_name), alpha=0.8)
        axes[0].scatter([tca], [miss], color=role_color("warning", style_name=style_name), zorder=5)
        if hbr > 0.0:
            axes[0].axhline(hbr, linestyle=":", color=role_color("safety_zone", style_name=style_name), alpha=0.7)
    axes[0].set_title("Conjunction Screening")
    axes[0].set_ylabel("Range / miss distance (km)")
    axes[0].legend(loc="best")

    labels = [f"{row.get('deputy_id', '')}:{row.get('chief_id', '')}" for row in conjunction_rows]
    pcs = [max(float(row.get("pc_screening") or 0.0), 1e-300) for row in conjunction_rows]
    axes[1].bar(labels, pcs, color=role_color("warning", style_name=style_name), alpha=0.85)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Screening Pc")
    axes[1].set_xlabel("Pair")
    axes[1].set_title("Small-Hard-Body Encounter-Plane Estimate")
    for tick in axes[1].get_xticklabels():
        tick.set_rotation(10)
        tick.set_ha("right")
    metadata = OELArtifactMetadata(
        scenario_name=scenario_name,
        generated_utc=generated_utc,
        version=get_oel_version(),
        artifact_id="conjunction_screening",
    )
    show_save_close_oel(
        fig,
        mode="save",
        out_path=path,
        dpi=dpi,
        artifact_id="conjunction_screening",
        metadata=metadata,
        style_name=style_name,
        plt_module=plt,
        close=True,
        bbox_inches="tight",
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _covariance_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Covariance Analysis",
        "",
        f"Scenario: `{summary.get('scenario_name', '')}`",
        f"Generated UTC: `{summary.get('generated_utc', '')}`",
        "",
        "This Pro analysis propagates configured 6-state ECI covariance with finite-difference STMs around the nominal OEL trajectory.",
        "It treats nominal applied acceleration as deterministic and does not include estimator-update or controller-feedback covariance sensitivity.",
        "",
        "## Source Binding",
        "",
    ]
    provenance = dict(summary.get("provenance", {}) or {})
    if provenance:
        lines.extend(
            [
                f"- Source product: `{provenance.get('source_product_id', '')}`",
                f"- Source product SHA-256: `{provenance.get('source_product_sha256', '')}`",
                f"- State/covariance binding: `{provenance.get('covariance_binding_id', '')}`",
                f"- Source report SHA-256: `{provenance.get('source_report_sha256', '')}`",
                f"- Covariance calibrated: `{provenance.get('calibrated', False)}`",
                f"- Calibration scope: `{provenance.get('calibration_scope')}`",
            ]
        )
        for artifact_id, digest in sorted(dict(provenance.get("source_hashes", {}) or {}).items()):
            lines.append(f"- Source hash `{artifact_id}`: `{digest}`")
    else:
        lines.append("- No interchange handoff provenance was supplied by the scenario.")
    lines.extend(
        [
            "",
            "## Final Object Covariance",
            "",
        ]
    )
    for object_id, row in dict(summary.get("final_by_object", {}) or {}).items():
        lines.append(
            f"- `{object_id}`: sigma_pos_3d={float(row.get('sigma_pos_3d_km', 0.0)):.6g} km, "
            f"sigma_vel_3d={float(row.get('sigma_vel_3d_km_s', 0.0)):.6g} km/s"
        )
    lines.extend(["", "## Final Pair Covariance", ""])
    for pair_id, row in dict(summary.get("final_by_pair", {}) or {}).items():
        sigma = row.get("sigma_rel_pos_3d_km")
        mahal = row.get("mahalanobis_distance_pos")
        lines.append(
            f"- `{pair_id}`: range={float(row.get('range_km', 0.0)):.6g} km, "
            f"sigma_rel_pos_3d={float(sigma or 0.0):.6g} km, "
            f"mahalanobis_pos={float(mahal or 0.0):.6g}"
        )
    screens = dict(summary.get("final_by_conjunction_screen", {}) or {})
    if screens:
        lines.extend(["", "## Collision-Risk Screening", ""])
        lines.append(
            "These are small-hard-body encounter-plane screening estimates. "
            "They are triage evidence, not catalog-grade conjunction assessment authority."
        )
        lines.append("")
        for pair_id, row in screens.items():
            pc = row.get("pc_screening")
            lines.append(
                f"- `{pair_id}`: tca={float(row.get('time_of_closest_approach_s', 0.0)):.6g} s, "
                f"miss={float(row.get('miss_distance_km', 0.0)):.6g} km, "
                f"hard_body_radius={float(row.get('hard_body_radius_m', 0.0)):.6g} m, "
                f"pc_screening={float(pc or 0.0):.6g}"
            )
    lines.extend(["", "## Assumptions", ""])
    for key, value in dict(summary.get("assumptions", {}) or {}).items():
        lines.append(f"- `{key}`: `{value}`")
    return "\n".join(lines) + "\n"


def _write_covariance_summary_patch(outdir: Path, summary: dict[str, Any]) -> None:
    summary_path = outdir / "master_run_summary.json"
    if summary_path.exists():
        current = json.loads(summary_path.read_text(encoding="utf-8"))
        current["covariance_analysis"] = summary.get("covariance_analysis")
        current["covariance_outputs"] = summary.get("covariance_outputs")
        summary_path.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_covariance_review_tables(db_path: Path, result: CovarianceAnalysisResult) -> None:
    if not db_path.exists():
        return
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS covariance_state (
                sample_index INTEGER,
                time_s REAL,
                object_id TEXT,
                sigma_x_eci_km REAL,
                sigma_y_eci_km REAL,
                sigma_z_eci_km REAL,
                sigma_vx_eci_km_s REAL,
                sigma_vy_eci_km_s REAL,
                sigma_vz_eci_km_s REAL,
                sigma_pos_3d_km REAL,
                sigma_vel_3d_km_s REAL,
                condition_number REAL,
                min_eigenvalue REAL,
                max_eigenvalue REAL
            );
            CREATE INDEX IF NOT EXISTS idx_covariance_state_object_time
            ON covariance_state(object_id, time_s);

            CREATE TABLE IF NOT EXISTS object_state_covariance (
                sample_index INTEGER,
                time_s REAL,
                object_id TEXT,
                frame TEXT,
                component_order_json TEXT,
                units_json TEXT,
                covariance_json TEXT,
                mathematically_valid INTEGER,
                calibrated INTEGER,
                calibration_scope TEXT,
                source TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_object_state_covariance_object_time
            ON object_state_covariance(object_id, time_s);

            CREATE TABLE IF NOT EXISTS relative_covariance (
                sample_index INTEGER,
                time_s REAL,
                deputy_id TEXT,
                chief_id TEXT,
                r_radial_km REAL,
                i_intrack_km REAL,
                c_crosstrack_km REAL,
                range_km REAL,
                sigma_r_radial_km REAL,
                sigma_i_intrack_km REAL,
                sigma_c_crosstrack_km REAL,
                sigma_rel_pos_3d_km REAL,
                mahalanobis_distance_pos REAL,
                range_to_sigma_pos_3d REAL
            );
            CREATE INDEX IF NOT EXISTS idx_relative_covariance_pair_time
            ON relative_covariance(deputy_id, chief_id, time_s);

            CREATE TABLE IF NOT EXISTS conjunction_screening (
                deputy_id TEXT,
                chief_id TEXT,
                method TEXT,
                screening_label TEXT,
                time_of_closest_approach_s REAL,
                sample_index INTEGER,
                interval_fraction REAL,
                miss_distance_km REAL,
                relative_speed_km_s REAL,
                hard_body_radius_km REAL,
                hard_body_radius_m REAL,
                pc_screening REAL,
                pc_log10 REAL,
                mahalanobis_distance_plane REAL,
                plane_miss_x_km REAL,
                plane_miss_y_km REAL,
                plane_sigma_major_km REAL,
                plane_sigma_minor_km REAL,
                plane_cov_xx_km2 REAL,
                plane_cov_xy_km2 REAL,
                plane_cov_yy_km2 REAL,
                range_to_hard_body_radius REAL
            );
            CREATE INDEX IF NOT EXISTS idx_conjunction_screening_pair_tca
            ON conjunction_screening(deputy_id, chief_id, time_of_closest_approach_s);
            """
        )
        conn.execute("DELETE FROM covariance_state")
        conn.execute("DELETE FROM object_state_covariance")
        conn.execute("DELETE FROM relative_covariance")
        conn.execute("DELETE FROM conjunction_screening")
        _insert_rows(conn, "covariance_state", result.object_rows)
        _insert_rows(conn, "object_state_covariance", result.object_covariance_rows)
        _insert_rows(conn, "relative_covariance", result.pair_rows)
        _insert_rows(conn, "conjunction_screening", result.conjunction_rows)
        conn.commit()


def _insert_rows(conn: sqlite3.Connection, table: str, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    columns = list(rows[0].keys())
    placeholders = ", ".join("?" for _ in columns)
    column_sql = ", ".join(columns)
    conn.executemany(
        f"INSERT INTO {table} ({column_sql}) VALUES ({placeholders})",
        [[_sql_value(row.get(column)) for column in columns] for row in rows],
    )


def _sql_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def _assumptions(cfg: SimulationScenarioConfig) -> dict[str, Any]:
    cov = cfg.analysis.covariance
    orbit = dict(dict(cfg.simulator.dynamics or {}).get("orbit", {}) or {})
    return {
        "method": "central finite-difference STM",
        "state_frame": "ECI Cartesian position/velocity",
        "initial_covariance_frame": "ECI",
        "position_step_km": float(cov.finite_difference.position_step_km),
        "velocity_step_km_s": float(cov.finite_difference.velocity_step_km_s),
        "process_noise_enabled": bool(cov.process_noise.enabled),
        "acceleration_sigma_km_s2": float(cov.process_noise.acceleration_sigma_km_s2),
        "integrator": str(orbit.get("integrator", "rk4")),
        "controller_feedback_sensitivity": "not modeled",
        "estimator_update_sensitivity": "not modeled",
        "cross_object_covariance": "assumed zero; object covariances are combined as independent",
    }


def _handoff_provenance(cfg: SimulationScenarioConfig) -> dict[str, Any]:
    metadata = dict(getattr(cfg, "metadata", {}) or {})
    handoff = dict(metadata.get("handoff", {}) or {})
    binding = dict(handoff.get("covariance_binding", {}) or {})
    if not handoff or not binding:
        return {}
    return {
        "source_product_id": str(handoff.get("source_product_id", "") or ""),
        "source_product_sha256": str(handoff.get("source_product_sha256", "") or ""),
        "adapter_id": str(handoff.get("adapter_id", "") or ""),
        "adapter_version": str(handoff.get("adapter_version", "") or ""),
        "source_hashes": dict(handoff.get("source_hashes", {}) or {}),
        "covariance_binding_id": str(binding.get("binding_id", "") or ""),
        "frame": binding.get("frame"),
        "epoch_jd_utc": binding.get("epoch_jd_utc"),
        "component_order": list(binding.get("component_order", []) or []),
        "units": list(binding.get("units", []) or []),
        "mathematically_valid": binding.get("mathematically_valid"),
        "calibrated": binding.get("calibrated"),
        "calibration_scope": binding.get("calibration_scope"),
        "source_report_sha256": binding.get("source_report_sha256"),
    }


def _mahalanobis(x: np.ndarray, covariance: np.ndarray) -> float | None:
    if not np.all(np.isfinite(covariance)):
        return None
    try:
        value = float(np.sqrt(max(float(x.T @ np.linalg.pinv(covariance) @ x), 0.0)))
    except np.linalg.LinAlgError:
        return None
    return value if math.isfinite(value) else None


def _condition_number(matrix: np.ndarray) -> float | None:
    try:
        value = float(np.linalg.cond(matrix))
    except np.linalg.LinAlgError:
        return None
    return value if math.isfinite(value) else None


def _sqrt_nonnegative(value: float) -> float:
    return float(math.sqrt(max(float(value), 0.0)))


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
