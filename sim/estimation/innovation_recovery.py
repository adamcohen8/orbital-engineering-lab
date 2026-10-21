"""Opt-in causal full-state anomaly handling, shared by orbital EKFs.

This module orchestrates existing estimator predictions; it contains no dynamics.
Persistent discrepancies are evidence of a changed track, not proof of a maneuver.

Concept inspired by @sneed-and-feed, public OEL PR #97, head
20f8fd1cc977632b091caeed1101a1209bede1ef. This is a new causal implementation;
no code or tests from that patch were copied.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from numbers import Integral
from typing import Any

import numpy as np

from sim.core.models import Measurement, StateBelief
from sim.estimation.maneuver_detection import chi_square_threshold


@dataclass(frozen=True)
class InnovationRecoveryConfig:
    enabled: bool = False
    strategy: str = "predictive_consistency"
    nis_probability: float = 0.999
    confirmation_count: int = 2
    covariance_inflation: float = 4.0
    max_gap_s: float = 120.0

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("innovation recovery enabled must be a boolean.")
        if self.strategy not in {"predictive_consistency", "persistence"}:
            raise ValueError("innovation recovery strategy must be predictive_consistency or persistence.")
        if not np.isfinite(self.nis_probability) or not 0.0 < self.nis_probability < 1.0:
            raise ValueError("innovation recovery nis_probability must lie in (0, 1).")
        if (
            isinstance(self.confirmation_count, bool)
            or not isinstance(self.confirmation_count, Integral)
            or self.confirmation_count < 2
        ):
            raise ValueError("innovation recovery confirmation_count must be an integer >= 2.")
        if not np.isfinite(self.covariance_inflation) or self.covariance_inflation < 1.0:
            raise ValueError("innovation recovery covariance_inflation must be finite and >= 1.")
        if not np.isfinite(self.max_gap_s) or self.max_gap_s <= 0.0:
            raise ValueError("innovation recovery max_gap_s must be positive and finite.")


@dataclass(frozen=True)
class InnovationRecoveryDiagnostics:
    classification: str
    action: str
    strategy: str
    event_time_s: float
    decision_time_s: float
    nis: float
    candidate_nis: float | None = None
    consecutive_exceedances: int = 0


@dataclass
class _Pending:
    candidate: StateBelief
    expected: StateBelief
    first_t_s: float
    last_measurement_t_s: float
    count: int


def validate_recovery_config(config: InnovationRecoveryConfig | None, measurement_model: str) -> None:
    if config is not None and not isinstance(config, InnovationRecoveryConfig):
        raise TypeError("innovation_recovery must be InnovationRecoveryConfig or None.")
    if config is not None and config.enabled and measurement_model not in {"relative_state", "state"}:
        raise ValueError("innovation recovery requires a full-state measurement model.")


def _copy_belief(belief: StateBelief) -> StateBelief:
    return StateBelief(belief.state.copy(), belief.covariance.copy(), float(belief.last_update_t_s))


def _same_belief(left: StateBelief, right: StateBelief) -> bool:
    return (
        left.last_update_t_s == right.last_update_t_s
        and np.array_equal(left.state, right.state)
        and np.array_equal(left.covariance, right.covariance)
    )


def _predict(estimator: Any, belief: StateBelief, t_s: float) -> StateBelief:
    x, p = estimator._predict(belief.state, belief.covariance, from_t_s=belief.last_update_t_s, to_t_s=t_s)
    return StateBelief(x, p, t_s)


def _measurement_covariance(estimator: Any) -> np.ndarray:
    full = getattr(estimator, "meas_noise_covariance", None)
    r = np.asarray(full if full is not None else np.diag(estimator.meas_noise_diag), dtype=float)
    if r.shape != (6, 6) or not np.all(np.isfinite(r)) or not np.allclose(r, r.T, rtol=1e-10, atol=1e-14):
        raise ValueError("innovation recovery measurement covariance must be finite symmetric 6x6.")
    try:
        np.linalg.cholesky(r)
    except np.linalg.LinAlgError as exc:
        raise ValueError("innovation recovery requires positive-definite measurement covariance.") from exc
    return r.copy()


class FullStateInnovationRecovery:
    """Bounded two-hypothesis state, with no retrospective updates or lookahead.

    A candidate starts at the suspicious full-state observation. Its covariance
    and state propagate through the same engine as the held track. Confirmation
    requires a high NIS against the held track and (except for the simpler
    persistence baseline) a low NIS against the candidate. Independent epoch
    measurement errors are assumed. Sensor bias and model errors remain possible.
    """

    def __init__(self) -> None:
        self.pending: _Pending | None = None
        self._config: InnovationRecoveryConfig | None = None

    def reset(self) -> None:
        self.pending = None

    def update(self, estimator: Any, belief: StateBelief, measurement: Measurement | None, t_s: float) -> StateBelief:
        cfg = estimator.innovation_recovery
        validate_recovery_config(cfg, getattr(estimator, "measurement_model", "state"))
        # The original method validates all epoch and measurement contracts before
        # this controller mutates its state. Its raw diagnostics are retained.
        raw = estimator._update_standard(belief, measurement, t_s)
        diagnostics = estimator.last_update_diagnostics
        if cfg != self._config:
            self.reset()
            self._config = cfg
        pending = self.pending
        if pending is not None and not _same_belief(belief, pending.expected):
            pending = None  # External corrections, covariance edits and replaced tracks invalidate history.
        if pending is not None and float(t_s) - pending.last_measurement_t_s > cfg.max_gap_s:
            pending = None
        if measurement is None:
            if pending is not None:
                self.pending = _Pending(
                    _predict(estimator, pending.candidate, float(t_s)),
                    _copy_belief(raw),
                    pending.first_t_s,
                    pending.last_measurement_t_s,
                    pending.count,
                )
            else:
                self.reset()
            return raw
        if not diagnostics.update_applied:
            self.reset()
            return raw

        epoch = float(measurement.t_s)
        if pending is not None and epoch <= pending.last_measurement_t_s:
            raise ValueError("innovation recovery requires strictly increasing measurement epochs.")
        r = _measurement_covariance(estimator)
        threshold = chi_square_threshold(6, cfg.nis_probability)
        nis = float(diagnostics.nis)
        sign = -1.0 if getattr(estimator, "measurement_origin", "chief") == "deputy" else 1.0
        observed = sign * np.asarray(measurement.vector, dtype=float).reshape(-1)[:6]
        candidate_nis = None
        if pending is not None and cfg.strategy == "predictive_consistency":
            candidate = _predict(estimator, pending.candidate, epoch)
            residual = observed - candidate.state
            factor = np.linalg.cholesky(candidate.covariance + r)
            whitened = np.linalg.solve(factor, residual)
            candidate_nis = float(whitened @ whitened)

        count = 0
        event_t = epoch if pending is None else pending.first_t_s
        if np.isfinite(nis) and nis <= threshold:
            classification = "nominal" if pending is None else "transient_discrepancy"
            action, result = "accepted", raw
            posterior_trace = diagnostics.posterior_cov_trace
            self.reset()
        else:
            consistent = pending is not None and (
                cfg.strategy == "persistence" or (candidate_nis is not None and candidate_nis <= threshold)
            )
            count = pending.count + 1 if consistent else 1
            if count >= cfg.confirmation_count:
                classification, action = "persistent_discrepancy", "reinitialized"
                posterior_trace = float(np.trace(cfg.covariance_inflation * r))
                result = _predict(
                    estimator, StateBelief(observed.copy(), cfg.covariance_inflation * r, epoch), float(t_s)
                )
                self.reset()
            else:
                classification = "pending_discrepancy" if pending is None else "unresolved_discrepancy"
                action = "held"
                prediction = _predict(estimator, belief, epoch)
                posterior_trace = float(np.trace(prediction.covariance))
                result = _predict(estimator, prediction, float(t_s))
                if not consistent:
                    event_t = epoch
                # Rebase the candidate on the latest observation; never use truth.
                candidate = _predict(estimator, StateBelief(observed.copy(), r, epoch), float(t_s))
                self.pending = _Pending(candidate, _copy_belief(result), event_t, epoch, count)
        recovery = InnovationRecoveryDiagnostics(
            classification,
            action,
            cfg.strategy,
            event_t,
            epoch,
            nis,
            candidate_nis,
            count,
        )
        estimator.last_update_diagnostics = replace(
            diagnostics,
            update_applied=action != "held",
            posterior_cov_trace=posterior_trace,
            recovery_diagnostics=recovery,
        )
        return result
