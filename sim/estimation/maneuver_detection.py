from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import numpy as np


@lru_cache(maxsize=1)
def _scipy_inverse_regularized_gamma() -> Any | None:
    """Resolve SciPy only for a run that actually evaluates a NIS gate."""
    try:  # pragma: no cover - fallback exercised only when scipy is unavailable.
        from scipy.special import gammaincinv

        return gammaincinv
    except Exception:  # pragma: no cover
        return None


@dataclass(frozen=True)
class EKFManeuverDetectionConfig:
    enabled: bool = False
    warning_probability: float = 0.99
    detection_probability: float = 0.999
    window_size: int = 5
    warning_count: int = 3
    detection_count: int = 3
    min_updates: int = 3
    cooldown_updates: int = 0

    def __post_init__(self) -> None:
        if not 0.0 < float(self.warning_probability) < 1.0:
            raise ValueError("warning_probability must be between 0 and 1.")
        if not 0.0 < float(self.detection_probability) < 1.0:
            raise ValueError("detection_probability must be between 0 and 1.")
        if float(self.warning_probability) > float(self.detection_probability):
            raise ValueError("warning_probability must be <= detection_probability.")
        if int(self.window_size) <= 0:
            raise ValueError("window_size must be positive.")
        if int(self.warning_count) <= 0 or int(self.detection_count) <= 0:
            raise ValueError("warning_count and detection_count must be positive.")
        if int(self.warning_count) > int(self.window_size) or int(self.detection_count) > int(self.window_size):
            raise ValueError("warning_count and detection_count must be <= window_size.")
        if int(self.min_updates) <= 0:
            raise ValueError("min_updates must be positive.")
        if int(self.cooldown_updates) < 0:
            raise ValueError("cooldown_updates must be non-negative.")


@dataclass(frozen=True)
class EKFManeuverDetectionUpdate:
    evaluated: bool
    status: str
    nis: float | None = None
    dof: int | None = None
    warning_threshold: float | None = None
    detection_threshold: float | None = None
    window_warning_count: int = 0
    window_detection_count: int = 0
    new_suspect_event: bool = False
    new_confirmed_event: bool = False
    reason: str | None = None


@dataclass
class EKFManeuverDetector:
    config: EKFManeuverDetectionConfig = field(default_factory=EKFManeuverDetectionConfig)
    status: str = field(default="disabled", init=False)
    sample_count: int = field(default=0, init=False)
    warning_sample_count: int = field(default=0, init=False)
    detection_sample_count: int = field(default=0, init=False)
    suspect_event_count: int = field(default=0, init=False)
    confirmed_event_count: int = field(default=0, init=False)
    first_suspect_t_s: float | None = field(default=None, init=False)
    first_confirmed_t_s: float | None = field(default=None, init=False)
    last_event_t_s: float | None = field(default=None, init=False)
    max_nis: float | None = field(default=None, init=False)
    _history: deque[tuple[float, bool, bool]] = field(init=False, repr=False)
    _cooldown_remaining: int = field(default=0, init=False, repr=False)
    _recovery_seen: bool = field(default=False, init=False, repr=False)
    recovery_hold_count: int = field(default=0, init=False)
    recovery_reinitialization_count: int = field(default=0, init=False)
    recovery_transient_count: int = field(default=0, init=False)
    last_recovery_event_t_s: float | None = field(default=None, init=False)
    last_recovery_decision_t_s: float | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.status = "nominal" if self.config.enabled else "disabled"
        self._history = deque(maxlen=int(self.config.window_size))

    def update(self, diagnostics: Any, *, t_s: float) -> EKFManeuverDetectionUpdate:
        recovery = getattr(diagnostics, "recovery_diagnostics", None)
        if recovery is not None:
            self._recovery_seen = True
            if recovery.action == "held":
                self.recovery_hold_count += 1
            if recovery.action == "reinitialized":
                self.recovery_reinitialization_count += 1
                self.last_recovery_event_t_s = float(recovery.event_time_s)
                self.last_recovery_decision_t_s = float(recovery.decision_time_s)
            if recovery.classification == "transient_discrepancy":
                self.recovery_transient_count += 1
        if not self.config.enabled:
            return EKFManeuverDetectionUpdate(evaluated=False, status="disabled", reason="disabled")
        # Held measurements retain pre-recovery NIS evidence. Recovery events
        # are reported separately from physical maneuver hypotheses.
        if diagnostics is None or (not bool(getattr(diagnostics, "update_applied", False)) and recovery is None):
            return EKFManeuverDetectionUpdate(evaluated=False, status=self.status, reason="no_ekf_update")

        nis = float(getattr(diagnostics, "nis", float("nan")))
        if not np.isfinite(nis):
            return EKFManeuverDetectionUpdate(evaluated=False, status=self.status, reason="nonfinite_nis")
        dof = _diagnostic_dimension(diagnostics)
        if dof <= 0:
            return EKFManeuverDetectionUpdate(evaluated=False, status=self.status, reason="unknown_measurement_dimension")

        warn_threshold = chi_square_threshold(dof, float(self.config.warning_probability))
        detect_threshold = chi_square_threshold(dof, float(self.config.detection_probability))
        above_warning = bool(nis >= warn_threshold)
        above_detection = bool(nis >= detect_threshold)

        self.sample_count += 1
        if above_warning:
            self.warning_sample_count += 1
        if above_detection:
            self.detection_sample_count += 1
        self.max_nis = nis if self.max_nis is None else max(float(self.max_nis), nis)

        self._history.append((nis, above_warning, above_detection))
        window_warning_count = int(sum(1 for _nis, flag, _detect in self._history if flag))
        window_detection_count = int(sum(1 for _nis, _warn, flag in self._history if flag))

        enough_updates = self.sample_count >= int(self.config.min_updates)
        suspect = enough_updates and window_warning_count >= int(self.config.warning_count)
        confirmed = enough_updates and window_detection_count >= int(self.config.detection_count)
        new_suspect = False
        new_confirmed = False

        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1

        if confirmed:
            new_confirmed = self.status != "confirmed" and self._cooldown_remaining <= 0
            self.status = "confirmed"
            if new_confirmed:
                self.confirmed_event_count += 1
                self.last_event_t_s = float(t_s)
                if self.first_confirmed_t_s is None:
                    self.first_confirmed_t_s = float(t_s)
                self._cooldown_remaining = int(self.config.cooldown_updates)
        elif suspect:
            new_suspect = self.status == "nominal"
            self.status = "suspect"
            if new_suspect:
                self.suspect_event_count += 1
                self.last_event_t_s = float(t_s)
                if self.first_suspect_t_s is None:
                    self.first_suspect_t_s = float(t_s)
        else:
            self.status = "nominal"

        return EKFManeuverDetectionUpdate(
            evaluated=True,
            status=self.status,
            nis=nis,
            dof=dof,
            warning_threshold=warn_threshold,
            detection_threshold=detect_threshold,
            window_warning_count=window_warning_count,
            window_detection_count=window_detection_count,
            new_suspect_event=new_suspect,
            new_confirmed_event=new_confirmed,
        )

    def summary(self) -> dict[str, float | int | str | bool | None]:
        result = {
            "enabled": bool(self.config.enabled),
            "status": str(self.status),
            "sample_count": int(self.sample_count),
            "warning_sample_count": int(self.warning_sample_count),
            "detection_sample_count": int(self.detection_sample_count),
            "suspect_event_count": int(self.suspect_event_count),
            "confirmed_event_count": int(self.confirmed_event_count),
            "first_suspect_t_s": self.first_suspect_t_s,
            "first_confirmed_t_s": self.first_confirmed_t_s,
            "last_event_t_s": self.last_event_t_s,
            "max_nis": self.max_nis,
            "window_size": int(self.config.window_size),
            "warning_count": int(self.config.warning_count),
            "detection_count": int(self.config.detection_count),
            "warning_probability": float(self.config.warning_probability),
            "detection_probability": float(self.config.detection_probability),
        }

        if self._recovery_seen:
            result.update({
                "recovery_hold_count": self.recovery_hold_count,
                "recovery_reinitialization_count": self.recovery_reinitialization_count,
                "recovery_transient_count": self.recovery_transient_count,
                "last_recovery_event_t_s": self.last_recovery_event_t_s,
                "last_recovery_decision_t_s": self.last_recovery_decision_t_s,
            })
        return result


def chi_square_threshold(dof: int, probability: float) -> float:
    k = int(dof)
    if k <= 0:
        raise ValueError("dof must be positive.")
    p = float(probability)
    if not 0.0 < p < 1.0:
        raise ValueError("probability must be between 0 and 1.")
    return _chi_square_threshold_cached(k, p)


@lru_cache(maxsize=64)
def _chi_square_threshold_cached(k: int, p: float) -> float:
    inverse_regularized_gamma = _scipy_inverse_regularized_gamma()
    if inverse_regularized_gamma is not None:
        # scipy.stats.chi2.ppf(p, k) is implemented as this expression. Calling
        # the ufunc directly avoids importing the much larger scipy.stats stack.
        return float(2.0 * inverse_regularized_gamma(0.5 * k, p))
    # Wilson-Hilferty fallback using Acklam's inverse-normal approximation.
    z = _normal_ppf(p)
    return float(k * (1.0 - 2.0 / (9.0 * k) + z * np.sqrt(2.0 / (9.0 * k))) ** 3)


def _diagnostic_dimension(diagnostics: Any) -> int:
    innovation = np.asarray(getattr(diagnostics, "innovation", []), dtype=float).reshape(-1)
    finite = innovation[np.isfinite(innovation)]
    if finite.size:
        return int(finite.size)
    cov = np.asarray(getattr(diagnostics, "innovation_covariance", []), dtype=float)
    if cov.ndim == 2 and cov.shape[0] == cov.shape[1] and cov.shape[0] > 0:
        return int(cov.shape[0])
    return 0


def _normal_ppf(p: float) -> float:
    # Peter J. Acklam's rational approximation, sufficient for fallback gates.
    a = [
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.38357751867269e02,
        -3.066479806614716e01,
        2.506628277459239,
    ]
    b = [
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838,
        -2.549732539343734,
        4.374664141464968,
        2.938163982698783,
    ]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996, 3.754408661907416]
    plow = 0.02425
    phigh = 1.0 - plow
    if p < plow:
        q = np.sqrt(-2.0 * np.log(p))
        numerator = ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]
        denominator = (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        return float(numerator / denominator)
    if p <= phigh:
        q = p - 0.5
        r = q * q
        numerator = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
        denominator = ((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0
        return float(numerator / denominator)
    q = np.sqrt(-2.0 * np.log(1.0 - p))
    numerator = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
    denominator = (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
    return float(numerator / denominator)
