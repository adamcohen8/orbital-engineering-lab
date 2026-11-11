from __future__ import annotations

import csv
import hashlib
import json
import sqlite3
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from sim.dynamics.orbit.accelerations import OrbitContext
from sim.dynamics.orbit.environment import EARTH_MU_KM3_S2
from sim.dynamics.orbit.propagator import OrbitPropagator
from sim.estimation.batch_least_squares import solve_batch_least_squares
from sim.estimation.maneuver_detection import chi_square_threshold
from sim.estimation.parameters import EstimatedParameter, ParameterSet
from sim.observations import ObservationPacket, observation_packet_from_dict
from sim.security.output_writes import mkdir_output, staged_output_file, write_output_text
from sim.utils.io import write_json

SEQUENTIAL_OD_SCHEMA_VERSION = 1
SEQUENTIAL_OD_CHECKPOINT_VERSION = 2

SequentialMeasurementModel = Callable[[Mapping[str, Any], np.ndarray], Mapping[str, Any]]


@dataclass(frozen=True)
class SequentialProcessNoise:
    """Discrete covariance policy for a Cartesian ECI orbit state.

    ``white_acceleration`` interprets the three values as continuous white
    acceleration spectral densities in km^2/s^3. ``state_random_walk`` uses
    six per-second Cartesian covariance rates in native km/km-s units.
    """

    model: str = "white_acceleration"
    spectral_density: tuple[float, ...] = (1.0e-14, 1.0e-14, 1.0e-14)

    def __post_init__(self) -> None:
        if not isinstance(self.model, str):
            raise ValueError("process noise model must be a string.")
        model = str(self.model).strip().lower()
        expected = 3 if model == "white_acceleration" else 6 if model == "state_random_walk" else 0
        if not expected:
            raise ValueError("process noise model must be 'white_acceleration' or 'state_random_walk'.")
        values = np.asarray(self.spectral_density, dtype=float).reshape(-1)
        if values.size != expected or np.any(~np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError(f"{model} spectral_density must contain {expected} finite nonnegative values.")
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "spectral_density", tuple(float(value) for value in values))

    def covariance(self, dt_s: float) -> np.ndarray:
        if isinstance(dt_s, (bool, np.bool_)) or not isinstance(dt_s, Real):
            raise ValueError("process-noise interval must be a finite nonnegative number.")
        try:
            dt = float(dt_s)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("process-noise interval must be a finite nonnegative number.") from exc
        if not np.isfinite(dt) or dt < 0.0:
            raise ValueError("process-noise interval must be a finite nonnegative number.")
        values = np.asarray(self.spectral_density, dtype=float)
        if self.model == "state_random_walk":
            return np.diag(values * dt)
        q = np.zeros((6, 6), dtype=float)
        for axis, spectral_density in enumerate(values):
            q[axis, axis] = spectral_density * dt**3 / 3.0
            q[axis, axis + 3] = spectral_density * dt**2 / 2.0
            q[axis + 3, axis] = q[axis, axis + 3]
            q[axis + 3, axis + 3] = spectral_density * dt
        return q


@dataclass(frozen=True)
class SequentialODConfig:
    estimator: str = "ekf"
    process_noise: SequentialProcessNoise = field(default_factory=SequentialProcessNoise)
    innovation_gate_probability: float = 0.999
    maneuver_warning_probability: float = 0.99
    maneuver_confirmation_probability: float = 0.999
    maneuver_confirmation_count: int = 2
    maneuver_window: int = 3
    reinitialize_after_maneuver: bool = True
    reinitialization_covariance_inflation: float = 4.0
    covariance_eigenvalue_floor: float = 1.0e-18
    finite_difference_position_step_km: float = 1.0e-3
    finite_difference_velocity_step_km_s: float = 1.0e-6
    integration_substep_s: float = 10.0
    checkpoint_interval: int = 25
    ukf_alpha: float = 0.1
    ukf_beta: float = 2.0
    ukf_kappa: float = 0.0
    numeric_backend: str = "rust"

    def __post_init__(self) -> None:
        backend = str(self.numeric_backend).strip().lower()
        if backend not in {"python", "rust"}:
            raise ValueError("numeric_backend must be python or rust")
        object.__setattr__(self, "numeric_backend", backend)
        if self.estimator not in {"ekf", "ukf"}:
            raise ValueError("estimator must be 'ekf' or 'ukf'.")
        if not isinstance(self.process_noise, SequentialProcessNoise):
            raise ValueError("process_noise must be a SequentialProcessNoise instance.")
        def finite_float(name: str) -> float:
            raw = getattr(self, name)
            if isinstance(raw, (bool, np.bool_)):
                raise ValueError(f"{name} must be a finite number.")
            try:
                value = float(raw)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"{name} must be a finite number.") from exc
            if not np.isfinite(value):
                raise ValueError(f"{name} must be a finite number.")
            object.__setattr__(self, name, value)
            return value

        for name in (
            "innovation_gate_probability",
            "maneuver_warning_probability",
            "maneuver_confirmation_probability",
        ):
            value = finite_float(name)
            if not 0.0 < value < 1.0:
                raise ValueError(f"{name} must be between zero and one.")
        if self.maneuver_warning_probability > self.maneuver_confirmation_probability:
            raise ValueError("maneuver warning probability cannot exceed confirmation probability.")
        integer_fields = (
            "maneuver_confirmation_count",
            "maneuver_window",
            "checkpoint_interval",
        )
        for name in integer_fields:
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
                raise ValueError(f"{name} must be a positive integer.")
        if self.maneuver_confirmation_count <= 0 or self.maneuver_window <= 0:
            raise ValueError("maneuver confirmation count and window must be positive.")
        if self.maneuver_confirmation_count > self.maneuver_window:
            raise ValueError("maneuver confirmation count cannot exceed the maneuver window.")
        if not isinstance(self.reinitialize_after_maneuver, (bool, np.bool_)):
            raise ValueError("reinitialize_after_maneuver must be a boolean.")
        object.__setattr__(self, "reinitialize_after_maneuver", bool(self.reinitialize_after_maneuver))
        inflation = finite_float("reinitialization_covariance_inflation")
        if inflation < 1.0:
            raise ValueError("reinitialization covariance inflation must be at least one.")
        if self.checkpoint_interval <= 0:
            raise ValueError("checkpoint_interval must be positive.")
        covariance_floor = finite_float("covariance_eigenvalue_floor")
        if covariance_floor <= 0.0:
            raise ValueError("covariance_eigenvalue_floor must be finite and positive.")
        position_step = finite_float("finite_difference_position_step_km")
        velocity_step = finite_float("finite_difference_velocity_step_km_s")
        if position_step <= 0.0 or velocity_step <= 0.0:
            raise ValueError("finite-difference state steps must be finite and positive.")
        integration_step = finite_float("integration_substep_s")
        if integration_step <= 0.0:
            raise ValueError("integration_substep_s must be finite and positive.")
        ukf_alpha = finite_float("ukf_alpha")
        if ukf_alpha <= 0.0:
            raise ValueError("ukf_alpha must be finite and positive.")
        ukf_beta = finite_float("ukf_beta")
        if ukf_beta < 0.0:
            raise ValueError("ukf_beta must be finite and nonnegative.")
        ukf_kappa = finite_float("ukf_kappa")
        if 6.0 + ukf_kappa <= 0.0:
            raise ValueError("ukf_kappa must be finite and greater than -6.")


def run_sequential_orbit_od(
    packet: ObservationPacket | Mapping[str, Any],
    *,
    initial_state_eci_km_s: Sequence[float] | np.ndarray,
    initial_covariance: Sequence[Sequence[float]] | np.ndarray,
    config: SequentialODConfig | None = None,
    propagator: OrbitPropagator | None = None,
    orbit_context: OrbitContext | None = None,
    environment: Mapping[str, Any] | None = None,
    truth_states_eci_km_s: Sequence[Sequence[float]] | np.ndarray | None = None,
    output_dir: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    resume: bool = False,
    compare_batch: bool = False,
    measurement_model: SequentialMeasurementModel | None = None,
    measurement_model_id: str = "eci_cartesian_position_velocity_v1",
    dynamics_model: Any | None = None,
    allow_append: bool = False,
) -> dict[str, Any]:
    """Run replayable EKF/UKF orbit OD and a fixed-interval RTS smoother.

    The input is the common OEL observation packet. The initial state is at the
    first observation epoch. All state histories are canonical ECI Cartesian
    states in km and km/s. Propagation uses the supplied checked-in OEL
    ``OrbitPropagator`` and ``OrbitContext``.
    """

    packet_obj = packet if isinstance(packet, ObservationPacket) else observation_packet_from_dict(packet)
    rows = packet_obj.observations
    if len(rows) < 3:
        raise ValueError("sequential OD requires at least three ordered observations.")
    times = np.asarray([float(row["time_s"]) for row in rows], dtype=float)
    if not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0.0):
        raise ValueError("sequential OD observations must have finite and strictly increasing epochs.")
    if str(packet_obj.to_dict().get("frame", "")).strip().upper() != "ECI":
        raise ValueError("sequential OD requires the normalized ECI observation contract.")
    cfg = config or SequentialODConfig()
    model_id = str(measurement_model_id).strip()
    if not model_id:
        raise ValueError("measurement_model_id must be non-empty.")
    if measurement_model is not None and cfg.estimator != "ekf":
        raise ValueError("Custom nonlinear sequential measurement models currently require estimator='ekf'.")
    if measurement_model is not None and compare_batch:
        raise ValueError("compare_batch is available only for the canonical Cartesian measurement model.")
    if measurement_model is not None and cfg.reinitialize_after_maneuver:
        raise ValueError(
            "Custom nonlinear measurement models require reinitialize_after_maneuver=False; "
            "Cartesian measurement reinitialization is not valid for arbitrary measurement units."
        )
    prop = propagator or OrbitPropagator(integrator="rk4", numeric_backend=cfg.numeric_backend)
    context = orbit_context or OrbitContext(mu_km3_s2=EARTH_MU_KM3_S2, mass_kg=100.0)
    env = dict(environment or {})
    if dynamics_model is None:
        x0 = _state6(initial_state_eci_km_s, "initial_state_eci_km_s")
        p0 = _covariance6(initial_covariance, "initial_covariance")
    else:
        from sim.estimation.covariance import validate_covariance
        x0 = np.asarray(initial_state_eci_km_s, dtype=float).reshape(-1)
        if x0.size != len(dynamics_model.state_names) or x0.size < 6 or not np.all(np.isfinite(x0)):
            raise ValueError("augmented initial state must match the dynamics state contract")
        p0 = validate_covariance(initial_covariance, x0.size)
        if cfg.estimator != "ekf" or cfg.reinitialize_after_maneuver or compare_batch:
            raise ValueError("augmented dynamics requires EKF, no Cartesian reset, and no legacy batch comparison")
    n = x0.size
    truth = None if truth_states_eci_km_s is None else np.asarray(truth_states_eci_km_s, dtype=float)
    if truth is not None and truth.shape != (len(rows), n):
        raise ValueError("truth_states_eci_km_s must match observation count and state dimension.")

    identity = {"sequential_measurement_model_id": model_id}
    if dynamics_model is not None:
        identity["dynamics_model"] = dynamics_model.identity()
        identity["initial_state"] = x0.tolist()
        identity["initial_covariance"] = p0.tolist()
    def digest_at(count):
        active_identity = dict(identity)
        if dynamics_model is not None:
            model_identity = dict(dynamics_model.identity())
            model_identity["events"] = [e for e in model_identity.get("events", [])
                                       if e["time_s"] <= float(rows[count - 1]["time_s"])]
            active_identity["dynamics_model"] = model_identity
        return _observation_packet_digest({**packet_obj.to_dict(), "observations": rows[:count], **active_identity})

    packet_digest = digest_at(len(rows)) if allow_append else _observation_packet_digest({**packet_obj.to_dict(), **identity})
    checkpoint = Path(checkpoint_path).expanduser() if checkpoint_path is not None else None
    state = _empty_runtime(x0, p0)
    start_index = 0
    if resume:
        if checkpoint is None or not checkpoint.exists():
            raise ValueError("resume requires an existing checkpoint_path.")
        if allow_append:
            old = json.loads(checkpoint.read_text())
            prefix = int(old["next_index"])
            if prefix > len(rows):
                raise ValueError("appended packet cannot truncate processed observations")
            old_digest = digest_at(prefix)
            state, start_index = _load_checkpoint(checkpoint, packet_digest=old_digest, config=cfg)
        else:
            state, start_index = _load_checkpoint(checkpoint, packet_digest=packet_digest, config=cfg)

    for index in range(start_index, len(rows)):
        row = rows[index]
        t = float(times[index])
        previous_t = float(times[index - 1]) if index > 0 else t
        dt = max(t - previous_t, 0.0)
        if index == 0 and start_index == 0:
            x_pred = state["x"].copy()
            p_pred = state["p"].copy()
            transition = np.eye(n)
            sigma_bundle = None
        elif dynamics_model is not None:
            x_pred, transition, q = dynamics_model.predict(state["x"], previous_t, dt)
            p_pred = dynamics_model.stabilize(transition @ state["p"] @ transition.T + q)
            sigma_bundle = None
        elif cfg.estimator == "ekf":
            x_pred, transition = _predict_ekf(
                state["x"], previous_t, dt, prop, context, env, cfg
            )
            p_pred = transition @ state["p"] @ transition.T + cfg.process_noise.covariance(dt)
            p_pred = _project_covariance(p_pred, cfg.covariance_eigenvalue_floor)
            sigma_bundle = None
        else:
            x_pred, p_pred, sigma_bundle = _predict_ukf(
                state["x"], state["p"], previous_t, dt, prop, context, env, cfg
            )
            # The unscented forward pass already contains the exact
            # filtered-to-predicted cross covariance required by the RTS
            # backward pass.  Retaining it is both mathematically consistent
            # with the UKF and avoids a separate six-propagation EKF STM.
            transition = np.eye(6)

        if measurement_model is None:
            z, h, r = _measurement(row)
            if n > 6:
                h = np.pad(h, ((0, 0), (0, n - 6)))
            predicted_measurement = h @ x_pred
            innovation = z - predicted_measurement
        else:
            evaluation = _validate_measurement_evaluation(measurement_model(row, x_pred), x_pred)
            z = evaluation["observed"]
            predicted_measurement = evaluation["predicted"]
            h = evaluation["jacobian"]
            r = evaluation["covariance"]
            innovation = evaluation["innovation"]
        if cfg.estimator == "ukf" and sigma_bundle is not None:
            innovation, innovation_covariance, gain = _ukf_measurement_update(
                z, h, r, x_pred, p_pred, sigma_bundle
            )
        else:
            innovation_covariance = h @ p_pred @ h.T + r
            gain = _right_solve(p_pred @ h.T, innovation_covariance)
        nis = _quadratic(innovation, innovation_covariance)
        gate_threshold = chi_square_threshold(z.size, cfg.innovation_gate_probability)
        accepted = bool(np.isfinite(nis) and nis <= gate_threshold)

        maneuver = _maneuver_decision(state["maneuver_nis"], nis, z.size, cfg)
        state["maneuver_nis"].append(float(maneuver["confirmation_score"]))
        reinitialized = False
        estimated_delta_v_m_s: list[float] | None = None
        if maneuver["confirmed"] and cfg.reinitialize_after_maneuver:
            x_post, p_post, estimated_delta_v_m_s = _reinitialize_from_measurement(
                x_pred, p_pred, z, h, r, cfg
            )
            accepted = True
            reinitialized = True
            state["maneuvers"].append(
                {
                    "sample_index": index,
                    "time_s": t,
                    "nis": float(nis),
                    "estimated_delta_v_m_s": estimated_delta_v_m_s,
                    "action": "measurement_reinitialization",
                }
            )
            state["maneuver_nis"].clear()
        elif accepted:
            x_post = x_pred + gain @ innovation
            if cfg.estimator == "ekf":
                if dynamics_model is not None:
                    mask = dynamics_model.update_mask(t)
                    gain = gain * mask[:, None]
                    if cfg.numeric_backend == "rust" and np.all(mask):
                        from sim.rust_estimation_backend import ekf_update_innovation
                        x_post, p_post, _y, _s, _nis = ekf_update_innovation(x_pred, p_pred, innovation, h, r)
                    else:
                        x_post, p_post = joseph_update(x_pred, p_pred, innovation, h, r, gain=gain)
                else:
                    x_post, p_post = joseph_update(x_pred, p_pred, innovation, h, r, gain=gain)
            else:
                p_post = p_pred - gain @ innovation_covariance @ gain.T
            p_post = (dynamics_model.stabilize(p_post) if dynamics_model is not None
                      else _project_covariance(p_post, cfg.covariance_eigenvalue_floor))
        else:
            x_post = x_pred.copy()
            p_post = p_pred.copy()
            state["rejections"].append(
                {
                    "sample_index": index,
                    "time_s": t,
                    "nis": float(nis),
                    "threshold": float(gate_threshold),
                    "reason": "innovation_gate",
                }
            )

        truth_row = None if truth is None else truth[index]
        nees = None if truth_row is None else _quadratic(x_post - truth_row, p_post)
        state["records"].append(
            {
                "sample_index": index,
                "time_s": t,
                "measurement_dimension": int(z.size),
                "measurement_accepted": accepted,
                "innovation": innovation.tolist(),
                "observed_measurement": z.tolist(),
                "predicted_measurement": predicted_measurement.tolist(),
                "nis": float(nis),
                "nis_gate_threshold": float(gate_threshold),
                "maneuver_warning": bool(maneuver["warning"]),
                "maneuver_confirmed": bool(maneuver["confirmed"]),
                "reinitialized": reinitialized,
                "smoothing_boundary": reinitialized or (dynamics_model is not None and
                    dynamics_model.boundary_at(t, previous_t)),
                "estimated_delta_v_m_s": estimated_delta_v_m_s,
                "predicted_state": x_pred.tolist(),
                "filtered_state": x_post.tolist(),
                "predicted_covariance": p_pred.tolist(),
                "filtered_covariance": p_post.tolist(),
                "transition_from_previous": transition.tolist(),
                "smoother_cross_covariance_from_previous": (
                    None
                    if sigma_bundle is None
                    else np.asarray(sigma_bundle["cross_covariance"], dtype=float).tolist()
                ),
                "truth_state": None if truth_row is None else truth_row.tolist(),
                "filtered_nees": nees,
            }
        )
        state["x"] = x_post
        state["p"] = p_post
        if checkpoint is not None and ((index + 1) % cfg.checkpoint_interval == 0 or index + 1 == len(rows)):
            _write_checkpoint(
                checkpoint,
                packet_digest=digest_at(index + 1) if allow_append else packet_digest,
                config=cfg,
                next_index=index + 1,
                state=state,
            )

    smoothed_states, smoothed_covariances = _rts_smooth(
        state["records"], cfg.covariance_eigenvalue_floor,
        stabilize=None if dynamics_model is None else dynamics_model.stabilize)
    for index, record in enumerate(state["records"]):
        record["smoothed_state"] = smoothed_states[index].tolist()
        record["smoothed_covariance"] = smoothed_covariances[index].tolist()
        truth_row = record["truth_state"]
        record["smoothed_nees"] = (
            None
            if truth_row is None
            else _quadratic(smoothed_states[index] - np.asarray(truth_row, dtype=float), smoothed_covariances[index])
        )

    report = _build_report(
        packet_obj=packet_obj,
        packet_digest=packet_digest,
        cfg=cfg,
        prop=prop,
        context=context,
        state=state,
        measurement_model_id=model_id,
    )
    if dynamics_model is not None:
        report["state_contract"]["names"] = list(dynamics_model.state_names)
        report["state_contract"]["representation"] = "cartesian_with_augmented_parameters"
        report["state_contract"]["units"] = list(dynamics_model.state_units)
        report["dynamics"]["augmented_model"] = dynamics_model.identity()
        report["claim_posture"]["note"] = "Augmented workflow requires its own frozen qualification campaign."
    if compare_batch:
        report["batch_filter_smoother_comparison"] = _batch_filter_smoother_comparison(
            rows=rows,
            times=times,
            initial_state=x0,
            propagator=prop,
            context=context,
            environment=env,
            records=state["records"],
            integration_substep_s=cfg.integration_substep_s,
        )
    if output_dir is not None:
        output_root = Path(output_dir).expanduser()
        mkdir_output(output_root)
        packet_path = output_root / "observation_packet.json"
        write_output_text(
            str(packet_path),
            json.dumps(packet_obj.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n",
        )
        artifacts = _write_artifacts(output_root, report, state["records"])
        artifacts["observation_packet_json"] = str(packet_path)
        report["artifacts"] = artifacts
        write_json(str(output_root / "sequential_od_report.json"), report)
        emission_path = output_root / "state_estimate_products_emission.json"
        try:
            from sim.interchange.adapters.sequential_od import emit_sequential_od_state_products

            emission = emit_sequential_od_state_products(output_root)
        except ValueError as exc:
            emission = {
                "status": "not_emitted",
                "selection_required_for_materialization": True,
                "reason": str(exc),
            }
        write_json(str(emission_path), emission)
        report["state_estimate_products_emission"] = emission
        report["artifacts"]["state_estimate_products_emission_json"] = str(emission_path)
    return report


def joseph_update(state, covariance, innovation, jacobian, measurement_covariance, *, gain=None):
    """Joseph update with an optional Schmidt/consider-state gain mask."""
    h, r, p = jacobian, measurement_covariance, covariance
    if gain is None:
        gain = _right_solve(p @ h.T, h @ p @ h.T + r)
    ikh = np.eye(len(state)) - gain @ h
    return state + gain @ innovation, ikh @ p @ ikh.T + gain @ r @ gain.T


def _empty_runtime(x: np.ndarray, p: np.ndarray) -> dict[str, Any]:
    return {
        "x": x.copy(),
        "p": p.copy(),
        "records": [],
        "rejections": [],
        "maneuvers": [],
        "maneuver_nis": [],
    }


def _predict_ekf(
    state: np.ndarray,
    t_s: float,
    dt_s: float,
    propagator: OrbitPropagator,
    context: OrbitContext,
    environment: dict[str, Any],
    config: SequentialODConfig,
) -> tuple[np.ndarray, np.ndarray]:
    if dt_s > 0.0:
        steps = np.asarray([config.finite_difference_position_step_km] * 3
                           + [config.finite_difference_velocity_step_km_s] * 3)
        trials = np.vstack((state, state + np.diag(steps)))
        batch = _trial_batch(trials, dt_s, propagator, context, config)
        if batch is not None:
            return batch[0], ((batch[1:] - batch[0]) / steps[:, None]).T
    predicted = _propagate(
        state,
        t_s,
        dt_s,
        propagator,
        context,
        environment,
        config.integration_substep_s,
    )
    transition = _numerical_transition(
        state, predicted, t_s, dt_s, propagator, context, environment, config
    )
    return predicted, transition


def _predict_ukf(
    state: np.ndarray,
    covariance: np.ndarray,
    t_s: float,
    dt_s: float,
    propagator: OrbitPropagator,
    context: OrbitContext,
    environment: dict[str, Any],
    config: SequentialODConfig,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    n = 6
    lam = config.ukf_alpha**2 * (n + config.ukf_kappa) - n
    scale = n + lam
    root = _stable_cholesky(_project_covariance(covariance, config.covariance_eigenvalue_floor) * scale)
    sigma = [state]
    for axis in range(n):
        sigma.extend((state + root[:, axis], state - root[:, axis]))
    sigma_array = np.asarray(sigma, dtype=float)
    propagated = _trial_batch(sigma_array, dt_s, propagator, context, config)
    if propagated is None:
        propagated = np.asarray(
            [
                _propagate(
                    item,
                    t_s,
                    dt_s,
                    propagator,
                    context,
                    environment,
                    config.integration_substep_s,
                )
                for item in sigma_array
            ],
            dtype=float,
        )
    wm = np.full(2 * n + 1, 1.0 / (2.0 * scale), dtype=float)
    wc = wm.copy()
    wm[0] = lam / scale
    wc[0] = wm[0] + (1.0 - config.ukf_alpha**2 + config.ukf_beta)
    mean = np.sum(wm[:, None] * propagated, axis=0)
    predicted_covariance = config.process_noise.covariance(dt_s)
    cross_covariance = np.zeros((6, 6), dtype=float)
    for weight, prior_item, item in zip(wc, sigma_array, propagated, strict=True):
        prior_delta = prior_item - state
        predicted_delta = item - mean
        predicted_covariance += weight * np.outer(predicted_delta, predicted_delta)
        cross_covariance += weight * np.outer(prior_delta, predicted_delta)
    predicted_covariance = _project_covariance(predicted_covariance, config.covariance_eigenvalue_floor)
    return mean, predicted_covariance, {
        "sigma": propagated,
        "wm": wm,
        "wc": wc,
        "cross_covariance": cross_covariance,
    }


def _ukf_measurement_update(
    measurement: np.ndarray,
    measurement_matrix: np.ndarray,
    measurement_covariance: np.ndarray,
    predicted_state: np.ndarray,
    predicted_covariance: np.ndarray,
    bundle: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    del bundle
    z_mean = measurement_matrix @ predicted_state
    innovation_covariance = (
        measurement_matrix @ predicted_covariance @ measurement_matrix.T
        + measurement_covariance
    )
    cross_covariance = predicted_covariance @ measurement_matrix.T
    gain = _right_solve(cross_covariance, innovation_covariance)
    return measurement - z_mean, innovation_covariance, gain


def _trial_batch(states, dt_s, propagator, context, config):
    """Batch only the time-independent, identity-checked native orbit plan."""
    if config.numeric_backend != "rust":
        return None
    from sim.rust_relative_backend import try_orbit_trial_batch

    elapsed = 0.0
    widths = []
    while elapsed < float(dt_s) - 1.0e-12:
        step = min(float(config.integration_substep_s), float(dt_s) - elapsed)
        widths.append(step)
        elapsed += step
    if not widths:
        return np.asarray(states, dtype=float).copy()
    return try_orbit_trial_batch(propagator, states, widths, context)


def _propagate(
    state: np.ndarray,
    t_s: float,
    dt_s: float,
    propagator: OrbitPropagator,
    context: OrbitContext,
    environment: dict[str, Any],
    integration_substep_s: float,
) -> np.ndarray:
    current = np.asarray(state, dtype=float).copy()
    elapsed = 0.0
    while elapsed < float(dt_s) - 1.0e-12:
        step = min(float(integration_substep_s), float(dt_s) - elapsed)
        current = np.asarray(
            propagator.propagate(
                x_eci=current,
                dt_s=step,
                t_s=float(t_s) + elapsed,
                command_accel_eci_km_s2=np.zeros(3),
                env=dict(environment),
                ctx=context,
            ),
            dtype=float,
        )
        elapsed += step
    return current


def _numerical_transition(
    state: np.ndarray,
    base: np.ndarray,
    t_s: float,
    dt_s: float,
    propagator: OrbitPropagator,
    context: OrbitContext,
    environment: dict[str, Any],
    config: SequentialODConfig,
) -> np.ndarray:
    if dt_s <= 0.0:
        return np.eye(6)
    steps = np.asarray(
        [config.finite_difference_position_step_km] * 3
        + [config.finite_difference_velocity_step_km_s] * 3,
        dtype=float,
    )
    transition = np.zeros((6, 6), dtype=float)
    for axis, step in enumerate(steps):
        perturbed = np.asarray(state, dtype=float).copy()
        perturbed[axis] += step
        transition[:, axis] = (
            _propagate(
                perturbed,
                t_s,
                dt_s,
                propagator,
                context,
                environment,
                config.integration_substep_s,
            )
            - base
        ) / step
    return transition


def _measurement(row: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    position = np.asarray(row["position_eci_km"], dtype=float).reshape(3)
    velocity = row.get("velocity_eci_km_s")
    if velocity is None:
        z = position
        h = np.hstack((np.eye(3), np.zeros((3, 3))))
    else:
        z = np.hstack((position, np.asarray(velocity, dtype=float).reshape(3)))
        h = np.eye(6)
    uncertainty = dict(row.get("uncertainty", {}) or {})
    covariance = uncertainty.get("matrix")
    if covariance is None:
        position_sigma = float(row.get("position_sigma_km", 1.0) or 1.0)
        sigmas = [position_sigma] * 3
        if velocity is not None:
            sigmas.extend([float(row.get("velocity_sigma_km_s", 1.0e-3) or 1.0e-3)] * 3)
        r = np.diag(np.square(sigmas))
    else:
        r = np.asarray(covariance, dtype=float)
    if r.shape != (z.size, z.size):
        raise ValueError("observation covariance shape does not match its measurement dimension.")
    return z, h, _project_covariance(r, 1.0e-24)


def _validate_measurement_evaluation(
    value: Mapping[str, Any], predicted_state: np.ndarray
) -> dict[str, np.ndarray]:
    if not isinstance(value, Mapping):
        raise ValueError("sequential measurement model must return a mapping.")
    required = {"observed", "predicted", "jacobian", "covariance", "innovation"}
    if set(value) != required:
        raise ValueError("sequential measurement evaluation must contain exactly observed, predicted, jacobian, covariance, and innovation.")
    observed = np.asarray(value["observed"], dtype=float).reshape(-1)
    predicted = np.asarray(value["predicted"], dtype=float).reshape(-1)
    innovation = np.asarray(value["innovation"], dtype=float).reshape(-1)
    jacobian = np.asarray(value["jacobian"], dtype=float)
    covariance = np.asarray(value["covariance"], dtype=float)
    dimension = observed.size
    if dimension <= 0 or predicted.shape != observed.shape or innovation.shape != observed.shape:
        raise ValueError("sequential measurement vectors must have one common nonzero dimension.")
    if jacobian.shape != (dimension, predicted_state.size):
        raise ValueError("sequential measurement Jacobian shape does not match measurement and state dimensions.")
    if covariance.shape != (dimension, dimension):
        raise ValueError("sequential measurement covariance shape does not match its measurement dimension.")
    if any(np.any(~np.isfinite(item)) for item in (observed, predicted, innovation, jacobian, covariance)):
        raise ValueError("sequential measurement evaluation must contain only finite values.")
    return {
        "observed": observed,
        "predicted": predicted,
        "innovation": innovation,
        "jacobian": jacobian,
        "covariance": _project_covariance(covariance, 1.0e-24),
    }


def _maneuver_decision(
    history: Sequence[float],
    nis: float,
    dof: int,
    config: SequentialODConfig,
) -> dict[str, bool | float]:
    warning = bool(nis >= chi_square_threshold(dof, config.maneuver_warning_probability))
    detection_threshold = chi_square_threshold(dof, config.maneuver_confirmation_probability)
    confirmation_score = float(nis / detection_threshold) if detection_threshold > 0.0 else float("inf")
    recent = [*history[-(config.maneuver_window - 1) :], confirmation_score]
    detections = sum(value >= 1.0 for value in recent)
    return {
        "warning": warning,
        "confirmed": detections >= config.maneuver_confirmation_count,
        "confirmation_score": confirmation_score,
    }


def _reinitialize_from_measurement(
    predicted_state: np.ndarray,
    predicted_covariance: np.ndarray,
    measurement: np.ndarray,
    measurement_matrix: np.ndarray,
    measurement_covariance: np.ndarray,
    config: SequentialODConfig,
) -> tuple[np.ndarray, np.ndarray, list[float] | None]:
    state = predicted_state.copy()
    measured_indices = np.flatnonzero(np.any(np.abs(measurement_matrix) > 0.0, axis=0))
    state[measured_indices] = measurement
    covariance = predicted_covariance.copy()
    covariance[measured_indices, :] = 0.0
    covariance[:, measured_indices] = 0.0
    covariance[np.ix_(measured_indices, measured_indices)] = (
        config.reinitialization_covariance_inflation * measurement_covariance
    )
    covariance = _project_covariance(covariance, config.covariance_eigenvalue_floor)
    delta_v = None
    if measurement.size == 6:
        delta_v = (1000.0 * (measurement[3:] - predicted_state[3:])).tolist()
    return state, covariance, delta_v


def _rts_smooth(records: Sequence[Mapping[str, Any]], eigenvalue_floor: float, *, stabilize=None) -> tuple[np.ndarray, np.ndarray]:
    filtered = np.asarray([row["filtered_state"] for row in records], dtype=float)
    filtered_covariance = np.asarray([row["filtered_covariance"] for row in records], dtype=float)
    smoothed = filtered.copy()
    smoothed_covariance = filtered_covariance.copy()
    for index in range(len(records) - 2, -1, -1):
        if bool(records[index + 1].get("smoothing_boundary", False)):
            # A measurement reinitialization represents an unmodeled state
            # discontinuity.  Without a maneuver-state transition and its
            # covariance, post-event information must not be smoothed across
            # that boundary into the pre-event arc.
            continue
        transition = np.asarray(records[index + 1]["transition_from_previous"], dtype=float)
        predicted_next = np.asarray(records[index + 1]["predicted_state"], dtype=float)
        predicted_covariance_next = np.asarray(records[index + 1]["predicted_covariance"], dtype=float)
        cross_covariance = records[index + 1].get("smoother_cross_covariance_from_previous")
        if cross_covariance is None:
            smoother_numerator = filtered_covariance[index] @ transition.T
        else:
            smoother_numerator = np.asarray(cross_covariance, dtype=float)
        if stabilize is not None:
            # Solve in correlation coordinates for augmented mixed-unit states.
            # No diagonal jitter or variance floor is added to the model.
            scales = np.sqrt(np.maximum(np.diag(predicted_covariance_next), np.finfo(float).tiny))
            normalized = predicted_covariance_next / np.outer(scales, scales)
            gain = _right_solve(smoother_numerator / scales, normalized) / scales
        else:
            gain = _right_solve(smoother_numerator, predicted_covariance_next)
        smoothed[index] = filtered[index] + gain @ (smoothed[index + 1] - predicted_next)
        smoothed_covariance[index] = filtered_covariance[index] + gain @ (
            smoothed_covariance[index + 1] - predicted_covariance_next
        ) @ gain.T
        smoothed_covariance[index] = (stabilize(smoothed_covariance[index]) if stabilize else
                                      _project_covariance(smoothed_covariance[index], eigenvalue_floor))
    return smoothed, smoothed_covariance


def _build_report(
    *,
    packet_obj: ObservationPacket,
    packet_digest: str,
    cfg: SequentialODConfig,
    prop: OrbitPropagator,
    context: OrbitContext,
    state: Mapping[str, Any],
    measurement_model_id: str,
) -> dict[str, Any]:
    records = list(state["records"])
    covariance_checks = _covariance_checks(records)
    filtered_metrics = _truth_metrics(records, state_key="filtered_state", nees_key="filtered_nees")
    smoothed_metrics = _truth_metrics(records, state_key="smoothed_state", nees_key="smoothed_nees")
    finite_nis = np.asarray([row["nis"] for row in records if np.isfinite(row["nis"])], dtype=float)
    accepted_count = sum(bool(row["measurement_accepted"]) for row in records)
    smoothing_preserved = True
    if filtered_metrics["available"]:
        smoothing_preserved = bool(
            smoothed_metrics["position_rms_m"] <= filtered_metrics["position_rms_m"] * 1.000001
        )
    return {
        "schema_version": SEQUENTIAL_OD_SCHEMA_VERSION,
        "kind": "oel.sequential_orbit_determination_report",
        "packet_digest_sha256": packet_digest,
        "object_id": str(packet_obj.to_dict().get("object_id", "")),
        "measurement_model_id": measurement_model_id,
        "state_contract": {
            "frame": "ECI",
            "representation": "cartesian_position_velocity",
            "units": ["km", "km", "km", "km/s", "km/s", "km/s"],
        },
        "estimator": cfg.estimator,
        "filter_update": "Joseph form" if cfg.estimator == "ekf" else "unscented covariance update",
        "smoother": (
            "fixed_interval_unscented_RTS"
            if cfg.estimator == "ukf"
            else "fixed_interval_RTS"
        ),
        "dynamics": {
            "implementation": "sim.dynamics.orbit.propagator.OrbitPropagator",
            "model": prop.model,
            "integrator": prop.integrator,
            "plugins": [getattr(plugin, "__name__", str(plugin)) for plugin in prop.plugins],
            "mu_km3_s2": float(context.mu_km3_s2),
            "integration_substep_s": float(cfg.integration_substep_s),
            "observation_cadence_decoupled_from_integrator_step": True,
        },
        "process_noise": asdict(cfg.process_noise),
        "observation_count": len(records),
        "accepted_count": accepted_count,
        "rejected_count": len(state["rejections"]),
        "innovation_statistics": {
            "nis_mean": None if finite_nis.size == 0 else float(np.mean(finite_nis)),
            "nis_median": None if finite_nis.size == 0 else float(np.median(finite_nis)),
            "nis_max": None if finite_nis.size == 0 else float(np.max(finite_nis)),
        },
        "covariance_checks": covariance_checks,
        "filtered_truth_metrics": filtered_metrics,
        "smoothed_truth_metrics": smoothed_metrics,
        "smoothing_improves_or_preserves_position_rms": smoothing_preserved,
        "maneuver_recovery": {
            "confirmed_count": len(state["maneuvers"]),
            "events": list(state["maneuvers"]),
            "policy": "innovation change-point with controlled measurement reinitialization",
        },
        "rejections": list(state["rejections"]),
        "quality_gates": {
            "covariance_symmetric_psd": bool(covariance_checks["all_symmetric_psd"]),
            "smoothing_improves_or_preserves": smoothing_preserved,
            "evidence_status": (
                "ready_for_statistical_campaign"
                if accepted_count > 0
                and covariance_checks["all_symmetric_psd"]
                and smoothing_preserved
                else "review_required"
            ),
        },
        "claim_posture": {
            "single_arc": "diagnostic only",
            "calibrated_covariance": False,
            "note": "A covariance claim requires the separate 100-plus-member Phase 7 ensemble gate.",
        },
        "records": records,
    }


def _batch_filter_smoother_comparison(
    *,
    rows: Sequence[Mapping[str, Any]],
    times: np.ndarray,
    initial_state: np.ndarray,
    propagator: OrbitPropagator,
    context: OrbitContext,
    environment: dict[str, Any],
    records: Sequence[Mapping[str, Any]],
    integration_substep_s: float,
) -> dict[str, Any]:
    labels = ("x_km", "y_km", "z_km", "vx_km_s", "vy_km_s", "vz_km_s")
    scales = (1.0, 1.0, 1.0, 1.0e-3, 1.0e-3, 1.0e-3)
    parameters = ParameterSet(
        [
            EstimatedParameter(
                label,
                float(initial_state[index]),
                scale=scales[index],
                lower=float(initial_state[index] - 100.0 * scales[index]),
                upper=float(initial_state[index] + 100.0 * scales[index]),
            )
            for index, label in enumerate(labels)
        ]
    )

    def residual(candidate: np.ndarray) -> np.ndarray:
        states = _propagate_series(
            candidate,
            times,
            propagator=propagator,
            context=context,
            environment=environment,
            integration_substep_s=integration_substep_s,
        )
        chunks = []
        for row, state in zip(rows, states, strict=True):
            measurement, h, covariance = _measurement(row)
            chunks.append(np.linalg.solve(_stable_cholesky(covariance), measurement - h @ state))
        return np.concatenate(chunks)

    solve = solve_batch_least_squares(parameters, residual, max_nfev=60)
    batch_states = _propagate_series(
        solve.x_native,
        times,
        propagator=propagator,
        context=context,
        environment=environment,
        integration_substep_s=integration_substep_s,
    )
    truth_rows = [record.get("truth_state") for record in records]
    truth_available = all(item is not None for item in truth_rows)
    batch_position_rms_m = None
    if truth_available:
        truth = np.asarray(truth_rows, dtype=float)
        batch_position_rms_m = float(
            1000.0 * np.sqrt(np.mean(np.sum((batch_states[:, :3] - truth[:, :3]) ** 2, axis=1)))
        )
    filtered_position_rms_m = _series_position_rms(records, "filtered_state")
    smoothed_position_rms_m = _series_position_rms(records, "smoothed_state")
    return {
        "observation_identity": "same normalized rows and covariance blocks",
        "dynamics_identity": "same OrbitPropagator, OrbitContext, and environment",
        "batch_solver_success": bool(solve.success),
        "batch_initial_state_eci_km_s": solve.x_native.tolist(),
        "batch_position_rms_m": batch_position_rms_m,
        "filtered_position_rms_m": filtered_position_rms_m,
        "smoothed_position_rms_m": smoothed_position_rms_m,
        "selection_policy": "report all three; no winner selected from fit residual alone",
    }


def _series_position_rms(records: Sequence[Mapping[str, Any]], key: str) -> float | None:
    selected = [record for record in records if record.get("truth_state") is not None]
    if not selected:
        return None
    estimate = np.asarray([record[key] for record in selected], dtype=float)
    truth = np.asarray([record["truth_state"] for record in selected], dtype=float)
    return float(1000.0 * np.sqrt(np.mean(np.sum((estimate[:, :3] - truth[:, :3]) ** 2, axis=1))))


def _propagate_series(
    initial_state: np.ndarray,
    times: np.ndarray,
    *,
    propagator: OrbitPropagator,
    context: OrbitContext,
    environment: dict[str, Any],
    integration_substep_s: float,
) -> np.ndarray:
    states = [np.asarray(initial_state, dtype=float).copy()]
    for index in range(1, len(times)):
        states.append(
            _propagate(
                states[-1],
                float(times[index - 1]),
                float(times[index] - times[index - 1]),
                propagator,
                context,
                environment,
                integration_substep_s,
            )
        )
    return np.asarray(states)


def predict_sequential_state_covariance(
    initial_state_eci_km_s: Sequence[float] | np.ndarray,
    initial_covariance: Sequence[Sequence[float]] | np.ndarray,
    *,
    initial_time_s: float,
    prediction_times_s: Sequence[float] | np.ndarray,
    config: SequentialODConfig | None = None,
    propagator: OrbitPropagator | None = None,
    orbit_context: OrbitContext | None = None,
    environment: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Propagate one state and covariance to explicit future epochs.

    This uses the same authoritative propagation, numerical transition, and
    process-noise policy as the sequential EKF.  It performs no measurement
    updates and makes no covariance-calibration claim.
    """

    cfg = config or SequentialODConfig(estimator="ekf")
    state = _state6(initial_state_eci_km_s, "initial_state_eci_km_s")
    covariance = _covariance6(initial_covariance, "initial_covariance")
    prop = propagator or OrbitPropagator(integrator="rk4", numeric_backend=cfg.numeric_backend)
    context = orbit_context or OrbitContext(mu_km3_s2=EARTH_MU_KM3_S2, mass_kg=100.0)
    env = dict(environment or {})
    times = np.asarray(prediction_times_s, dtype=float).reshape(-1)
    if not np.isfinite(initial_time_s):
        raise ValueError("initial_time_s must be finite.")
    if np.any(~np.isfinite(times)) or np.any(times <= float(initial_time_s)) or np.any(np.diff(times) <= 0.0):
        raise ValueError("prediction_times_s must be finite, strictly increasing, and later than initial_time_s.")
    records: list[dict[str, Any]] = []
    current_time = float(initial_time_s)
    for target_time in times:
        dt = float(target_time - current_time)
        predicted, transition = _predict_ekf(state, current_time, dt, prop, context, env, cfg)
        covariance = _project_covariance(
            transition @ covariance @ transition.T + cfg.process_noise.covariance(dt),
            cfg.covariance_eigenvalue_floor,
        )
        state = predicted
        records.append(
            {
                "time_s": float(target_time),
                "state_eci_km_km_s": state.tolist(),
                "covariance_eci_km_km_s": covariance.tolist(),
                "transition_from_previous": transition.tolist(),
            }
        )
        current_time = float(target_time)
    return records


def _covariance_checks(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    minimum_eigenvalue = float("inf")
    maximum_asymmetry = 0.0
    passed = True
    for record in records:
        for key in ("predicted_covariance", "filtered_covariance", "smoothed_covariance"):
            covariance = np.asarray(record[key], dtype=float)
            maximum_asymmetry = max(maximum_asymmetry, float(np.max(np.abs(covariance - covariance.T))))
            eigenvalue = float(np.min(np.linalg.eigvalsh(0.5 * (covariance + covariance.T))))
            minimum_eigenvalue = min(minimum_eigenvalue, eigenvalue)
            passed = passed and maximum_asymmetry <= 1.0e-10 and eigenvalue >= -1.0e-14
    return {
        "all_symmetric_psd": bool(passed),
        "maximum_asymmetry": maximum_asymmetry,
        "minimum_eigenvalue": minimum_eigenvalue,
        "tolerances": {"symmetry_absolute": 1.0e-10, "minimum_eigenvalue": -1.0e-14},
    }


def _truth_metrics(
    records: Sequence[Mapping[str, Any]], *, state_key: str, nees_key: str
) -> dict[str, Any]:
    selected = [row for row in records if row.get("truth_state") is not None]
    if not selected:
        return {"available": False}
    estimates = np.asarray([row[state_key] for row in selected], dtype=float)
    truth = np.asarray([row["truth_state"] for row in selected], dtype=float)
    errors = estimates - truth
    nees = np.asarray([row[nees_key] for row in selected], dtype=float)
    containment = {}
    for label, probability in (("one_sigma", 0.6827), ("two_sigma", 0.9545), ("three_sigma", 0.9973)):
        threshold = chi_square_threshold(errors.shape[1], probability)
        containment[label] = float(np.mean(nees <= threshold))
    return {
        "available": True,
        "position_rms_m": float(1000.0 * np.sqrt(np.mean(np.sum(errors[:, :3] ** 2, axis=1)))),
        "velocity_rms_mm_s": float(1.0e6 * np.sqrt(np.mean(np.sum(errors[:, 3:6] ** 2, axis=1)))),
        "final_position_error_m": float(1000.0 * np.linalg.norm(errors[-1, :3])),
        "final_velocity_error_mm_s": float(1.0e6 * np.linalg.norm(errors[-1, 3:6])),
        "nees_mean": float(np.mean(nees)),
        "containment": containment,
    }


def _write_artifacts(output: Path, report: Mapping[str, Any], records: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    mkdir_output(output)
    report_path = output / "sequential_od_report.json"
    residual_path = output / "sequential_od_samples.csv"
    review_path = output / "review/run.sqlite"
    mkdir_output(residual_path.parent)
    with staged_output_file(residual_path) as residual_target:
        with residual_target.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=(
                    "sample_index",
                    "time_s",
                    "measurement_accepted",
                    "nis",
                    "filtered_nees",
                    "smoothed_nees",
                    "maneuver_confirmed",
                    "reinitialized",
                ),
            )
            writer.writeheader()
            for row in records:
                writer.writerow({key: row.get(key) for key in writer.fieldnames})
    mkdir_output(review_path.parent)
    with staged_output_file(review_path) as review_target:
        with sqlite3.connect(review_target) as connection:
            connection.executescript(
                """
                DROP TABLE IF EXISTS sequential_od_run;
                DROP TABLE IF EXISTS sequential_od_samples;
                DROP TABLE IF EXISTS sequential_od_rejections;
                DROP TABLE IF EXISTS sequential_od_maneuvers;
                CREATE TABLE sequential_od_run (
                    schema_version INTEGER NOT NULL, object_id TEXT NOT NULL,
                    estimator TEXT NOT NULL, observation_count INTEGER NOT NULL,
                    accepted_count INTEGER NOT NULL, rejected_count INTEGER NOT NULL,
                    evidence_status TEXT NOT NULL, report_json TEXT NOT NULL
                );
                CREATE TABLE sequential_od_samples (
                    sample_index INTEGER PRIMARY KEY, time_s REAL NOT NULL,
                    measurement_accepted INTEGER NOT NULL, nis REAL,
                    filtered_nees REAL, smoothed_nees REAL,
                    maneuver_confirmed INTEGER NOT NULL, reinitialized INTEGER NOT NULL,
                    filtered_state_json TEXT NOT NULL, smoothed_state_json TEXT NOT NULL
                );
                CREATE TABLE sequential_od_rejections (
                    sample_index INTEGER NOT NULL, time_s REAL NOT NULL,
                    nis REAL NOT NULL, threshold REAL NOT NULL, reason TEXT NOT NULL
                );
                CREATE TABLE sequential_od_maneuvers (
                    sample_index INTEGER NOT NULL, time_s REAL NOT NULL,
                    nis REAL NOT NULL, estimated_delta_v_json TEXT, action TEXT NOT NULL
                );
                """
            )
            connection.execute(
                "INSERT INTO sequential_od_run VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    int(report["schema_version"]),
                    str(report["object_id"]),
                    str(report["estimator"]),
                    int(report["observation_count"]),
                    int(report["accepted_count"]),
                    int(report["rejected_count"]),
                    str(report["quality_gates"]["evidence_status"]),
                    str(report_path),
                ),
            )
            connection.executemany(
                "INSERT INTO sequential_od_samples VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        int(row["sample_index"]),
                        float(row["time_s"]),
                        int(bool(row["measurement_accepted"])),
                        float(row["nis"]),
                        row.get("filtered_nees"),
                        row.get("smoothed_nees"),
                        int(bool(row["maneuver_confirmed"])),
                        int(bool(row["reinitialized"])),
                        json.dumps(row["filtered_state"]),
                        json.dumps(row["smoothed_state"]),
                    )
                    for row in records
                ],
            )
            connection.executemany(
                "INSERT INTO sequential_od_rejections VALUES (?, ?, ?, ?, ?)",
                [
                    (
                        int(row["sample_index"]),
                        float(row["time_s"]),
                        float(row["nis"]),
                        float(row["threshold"]),
                        str(row["reason"]),
                    )
                    for row in report["rejections"]
                ],
            )
            connection.executemany(
                "INSERT INTO sequential_od_maneuvers VALUES (?, ?, ?, ?, ?)",
                [
                    (
                        int(row["sample_index"]),
                        float(row["time_s"]),
                        float(row["nis"]),
                        json.dumps(row.get("estimated_delta_v_m_s")),
                        str(row["action"]),
                    )
                    for row in report["maneuver_recovery"]["events"]
                ],
            )
    write_json(str(report_path), dict(report))
    return {
        "report_json": str(report_path),
        "samples_csv": str(residual_path),
        "review_sqlite": str(review_path),
    }


def _write_checkpoint(
    path: Path,
    *,
    packet_digest: str,
    config: SequentialODConfig,
    next_index: int,
    state: Mapping[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint_version": SEQUENTIAL_OD_CHECKPOINT_VERSION,
        "packet_digest_sha256": packet_digest,
        "config": _config_payload(config),
        "next_index": int(next_index),
        "state": {
            "x": np.asarray(state["x"], dtype=float).tolist(),
            "p": np.asarray(state["p"], dtype=float).tolist(),
            "records": list(state["records"]),
            "rejections": list(state["rejections"]),
            "maneuvers": list(state["maneuvers"]),
            "maneuver_nis": list(state["maneuver_nis"]),
        },
    }
    write_json(str(path), payload)


def _load_checkpoint(
    path: Path, *, packet_digest: str, config: SequentialODConfig
) -> tuple[dict[str, Any], int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload.get("checkpoint_version", 0)) != SEQUENTIAL_OD_CHECKPOINT_VERSION:
        raise ValueError("unsupported sequential OD checkpoint version.")
    if str(payload.get("packet_digest_sha256", "")) != packet_digest:
        raise ValueError("checkpoint observation packet digest does not match.")
    if payload.get("config") != _config_payload(config):
        raise ValueError("checkpoint sequential OD configuration does not match.")
    raw = dict(payload["state"])
    state = {
        "x": np.asarray(raw["x"], dtype=float),
        "p": np.asarray(raw["p"], dtype=float),
        "records": list(raw["records"]),
        "rejections": list(raw["rejections"]),
        "maneuvers": list(raw["maneuvers"]),
        "maneuver_nis": list(raw["maneuver_nis"]),
    }
    return state, int(payload["next_index"])


def _config_payload(config: SequentialODConfig) -> dict[str, Any]:
    payload = asdict(config)
    # Preserve an explicit selector across checkpoint round trips now that the
    # omitted selector uses Rust.
    return json.loads(json.dumps(payload, sort_keys=True))


def _state6(value: Sequence[float] | np.ndarray, name: str) -> np.ndarray:
    state = np.asarray(value, dtype=float).reshape(-1)
    if state.size != 6 or np.any(~np.isfinite(state)):
        raise ValueError(f"{name} must contain six finite values.")
    return state


def _covariance6(value: Sequence[Sequence[float]] | np.ndarray, name: str) -> np.ndarray:
    covariance = np.asarray(value, dtype=float)
    if covariance.shape != (6, 6) or np.any(~np.isfinite(covariance)):
        raise ValueError(f"{name} must be a finite 6x6 matrix.")
    if not np.allclose(covariance, covariance.T, rtol=1.0e-10, atol=1.0e-14):
        raise ValueError(f"{name} must be symmetric.")
    if np.min(np.linalg.eigvalsh(covariance)) < -1.0e-14:
        raise ValueError(f"{name} must be positive semidefinite.")
    return covariance


def _project_covariance(covariance: np.ndarray, floor: float) -> np.ndarray:
    symmetric = 0.5 * (np.asarray(covariance, dtype=float) + np.asarray(covariance, dtype=float).T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    return (eigenvectors * np.maximum(eigenvalues, float(floor))) @ eigenvectors.T


def _right_solve(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.solve(right.T, left.T).T
    except np.linalg.LinAlgError:
        return left @ np.linalg.pinv(right)


def _quadratic(vector: np.ndarray, covariance: np.ndarray) -> float:
    value = np.asarray(vector, dtype=float)
    try:
        solved = np.linalg.solve(covariance, value)
    except np.linalg.LinAlgError:
        solved = np.linalg.pinv(covariance) @ value
    return float(value.T @ solved)


def _stable_cholesky(covariance: np.ndarray) -> np.ndarray:
    jitter = 0.0
    for _ in range(8):
        try:
            return np.linalg.cholesky(covariance + jitter * np.eye(covariance.shape[0]))
        except np.linalg.LinAlgError:
            jitter = 1.0e-18 if jitter == 0.0 else jitter * 10.0
    return np.linalg.cholesky(_project_covariance(covariance, 1.0e-12))


def _digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _observation_packet_digest(packet: Mapping[str, Any]) -> str:
    """Hash estimation semantics while ignoring packet packaging time."""

    semantic_packet = dict(packet)
    semantic_packet.pop("created_utc", None)
    return _digest(semantic_packet)
