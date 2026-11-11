from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

# Scalar Julian dates lose a few tens of microseconds when converted to
# relative seconds through binary64 arithmetic at modern epochs.  Callers that
# derive their partition times from scalar JDs should use this accommodation;
# exact TAI-second workflows may continue to request zero tolerance.
JULIAN_DATE_BOUNDARY_TOLERANCE_S = 1.0e-4


@dataclass(frozen=True)
class ObservationPartition:
    fit_mask: np.ndarray
    holdout_mask: np.ndarray
    excluded_mask: np.ndarray
    fit_duration_s: float
    holdout_duration_s: float
    summary: dict[str, Any]


def partition_time_arc(
    times_s: np.ndarray,
    *,
    fit_duration_s: float | None,
    holdout_duration_s: float | None,
    minimum_fit_count: int = 2,
    allow_repeated_epochs: bool = False,
    boundary_tolerance_s: float = 1.0e-9,
) -> ObservationPartition:
    """Apply the common OD fit/holdout boundary contract to a relative time arc."""

    times = np.asarray(times_s, dtype=float).reshape(-1)
    if times.size == 0 or not np.all(np.isfinite(times)):
        raise ValueError("observation times must be a non-empty finite array.")
    differences = np.diff(times)
    if np.any(differences < 0.0) or (not allow_repeated_epochs and np.any(differences == 0.0)):
        ordering = "nondecreasing" if allow_repeated_epochs else "strictly increasing"
        raise ValueError(f"observation times must be {ordering}.")
    fit_duration = float(fit_duration_s) if fit_duration_s is not None else float(times[-1])
    holdout_duration = (
        float(holdout_duration_s) if holdout_duration_s is not None else max(0.0, float(times[-1]) - fit_duration)
    )
    if fit_duration <= 0.0:
        raise ValueError("fit_duration_s must be positive.")
    if holdout_duration < 0.0:
        raise ValueError("holdout_duration_s must be non-negative.")
    tolerance = float(boundary_tolerance_s)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("boundary_tolerance_s must be finite and non-negative.")
    holdout_end = fit_duration + holdout_duration
    fit_mask = times <= fit_duration + tolerance
    holdout_mask = (times > fit_duration + tolerance) & (times <= holdout_end + tolerance)
    excluded_mask = ~(fit_mask | holdout_mask)
    fit_count = int(np.count_nonzero(fit_mask))
    holdout_count = int(np.count_nonzero(holdout_mask))
    if fit_count < int(minimum_fit_count):
        raise ValueError(f"fit window must contain at least {int(minimum_fit_count)} observations.")
    summary = {
        "schema_version": 1,
        "strategy": "time_window",
        "time_origin": "first_observation",
        "fit_boundary": "time_s <= fit_duration_s",
        "holdout_boundary": "fit_duration_s < time_s <= fit_duration_s + holdout_duration_s",
        "fit_duration_s": fit_duration,
        "holdout_duration_s": holdout_duration,
        "boundary_tolerance_s": tolerance,
        "fit_observation_count": fit_count,
        "holdout_observation_count": holdout_count,
        "excluded_observation_count": int(np.count_nonzero(excluded_mask)),
        "holdout_status": "evaluated" if holdout_count else "not_evaluated",
    }
    return ObservationPartition(
        fit_mask=fit_mask,
        holdout_mask=holdout_mask,
        excluded_mask=excluded_mask,
        fit_duration_s=fit_duration,
        holdout_duration_s=holdout_duration,
        summary=summary,
    )
