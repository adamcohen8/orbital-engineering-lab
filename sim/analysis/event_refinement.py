"""Deterministic sampled-window construction and optional provider refinement."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from sim.numeric_backend import normalize_numeric_backend


@dataclass(frozen=True)
class RefinedTransition:
    transition_kind: str
    time_s: float
    bracket_start_s: float
    bracket_end_s: float
    disposition: str
    iterations: int
    reason_before: str
    reason_after: str


@dataclass(frozen=True)
class SampledAvailabilityInterval:
    interval_index: int
    start_s: float
    end_s: float
    duration_s: float
    start_censored: bool
    end_censored: bool
    acquisition_disposition: str
    loss_disposition: str
    acquisition_reason: str
    loss_reason: str


AvailabilityEvaluator = Callable[[float], tuple[bool, str]]


def _validated_samples(
    times_s: np.ndarray,
    available: np.ndarray,
    reasons: tuple[str, ...] | list[str] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    times = np.asarray(times_s, dtype=float)
    mask = np.asarray(available, dtype=bool)
    reason_tuple = tuple(str(value) for value in reasons)
    if times.ndim != 1 or times.size < 2 or not np.all(np.isfinite(times)):
        raise ValueError("times_s must contain at least two finite epochs.")
    if np.any(np.diff(times) <= 0.0):
        raise ValueError("times_s must be strictly increasing.")
    if mask.shape != times.shape or len(reason_tuple) != times.size:
        raise ValueError("available and reasons must match times_s.")
    if any(not value for value in reason_tuple):
        raise ValueError("reasons must contain non-empty strings.")
    return times, mask, reason_tuple


def refine_availability_transitions(
    times_s: np.ndarray,
    available: np.ndarray,
    reasons: tuple[str, ...] | list[str] | np.ndarray,
    *,
    evaluator_at_time: AvailabilityEvaluator | None = None,
    time_tolerance_s: float | None = None,
    max_iterations: int | None = None,
    numeric_backend: str = "rust",
) -> tuple[RefinedTransition, ...]:
    """Refine sampled boolean changes without inventing an interpolation model."""

    backend = normalize_numeric_backend(numeric_backend)
    times, mask, reason_tuple = _validated_samples(times_s, available, reasons)
    if evaluator_at_time is None:
        if time_tolerance_s is not None or max_iterations is not None:
            raise ValueError("Refinement tolerances require evaluator_at_time.")
    else:
        if time_tolerance_s is None or not np.isfinite(float(time_tolerance_s)) or float(
            time_tolerance_s
        ) <= 0.0:
            raise ValueError("time_tolerance_s must be positive and finite for refinement.")
        if (
            max_iterations is None
            or isinstance(max_iterations, (bool, np.bool_))
            or int(max_iterations) != max_iterations
            or int(max_iterations) <= 0
        ):
            raise ValueError("max_iterations must be a positive integer for refinement.")
    if backend == "rust":
        from sim.rust_coverage_backend import (
            refine_availability_transitions as rust_refine_availability_transitions,
        )

        raw = rust_refine_availability_transitions(
            times,
            mask,
            reason_tuple,
            evaluator_at_time=evaluator_at_time,
            time_tolerance_s=time_tolerance_s,
            max_iterations=max_iterations,
        )
        return tuple(RefinedTransition(*value) for value in raw)

    transitions: list[RefinedTransition] = []
    for index in np.flatnonzero(mask[1:] != mask[:-1]):
        left_index = int(index)
        right_index = left_index + 1
        left_time = float(times[left_index])
        right_time = float(times[right_index])
        kind = "acquisition" if mask[right_index] else "loss"
        if evaluator_at_time is None:
            transitions.append(
                RefinedTransition(
                    transition_kind=kind,
                    time_s=right_time,
                    bracket_start_s=left_time,
                    bracket_end_s=right_time,
                    disposition="sample_bounded",
                    iterations=0,
                    reason_before=reason_tuple[left_index],
                    reason_after=reason_tuple[right_index],
                )
            )
            continue

        left_available = bool(mask[left_index])
        right_available = bool(mask[right_index])
        left_reason = reason_tuple[left_index]
        right_reason = reason_tuple[right_index]
        iterations = 0
        provider_unavailable = False
        while right_time - left_time > float(time_tolerance_s) and iterations < int(
            max_iterations
        ):
            midpoint = 0.5 * (left_time + right_time)
            try:
                midpoint_available, midpoint_reason = evaluator_at_time(midpoint)
            except Exception:
                # Arbitrary-epoch evaluation is an optional precision upgrade.
                # A provider that cannot service the requested epoch must not
                # erase the conservative transition already present in the
                # retained samples.
                left_time = float(times[left_index])
                right_time = float(times[right_index])
                left_reason = reason_tuple[left_index]
                right_reason = reason_tuple[right_index]
                iterations = 0
                provider_unavailable = True
                break
            midpoint_available = bool(midpoint_available)
            midpoint_reason = str(midpoint_reason or "").strip()
            if not midpoint_reason:
                raise ValueError("evaluator_at_time must return a non-empty reason.")
            if midpoint_available == left_available:
                left_time = midpoint
                left_reason = midpoint_reason
            elif midpoint_available == right_available:
                right_time = midpoint
                right_reason = midpoint_reason
            else:
                raise ValueError("evaluator_at_time returned a state outside the transition bracket.")
            iterations += 1
        converged = (
            not provider_unavailable
            and right_time - left_time <= float(time_tolerance_s)
        )
        transitions.append(
            RefinedTransition(
                transition_kind=kind,
                time_s=0.5 * (left_time + right_time),
                bracket_start_s=left_time,
                bracket_end_s=right_time,
                disposition=(
                    "provider_refined"
                    if converged
                    else "sample_bounded"
                    if provider_unavailable
                    else "iteration_limited"
                ),
                iterations=iterations,
                reason_before=left_reason,
                reason_after=right_reason,
            )
        )
    return tuple(transitions)


def availability_intervals(
    times_s: np.ndarray,
    available: np.ndarray,
    reasons: tuple[str, ...] | list[str] | np.ndarray,
    *,
    transitions: tuple[RefinedTransition, ...] = (),
) -> tuple[SampledAvailabilityInterval, ...]:
    """Build left-closed/right-open intervals from sampled availability evidence."""

    times, mask, reason_tuple = _validated_samples(times_s, available, reasons)
    transition_by_right_index: dict[int, RefinedTransition] = {}
    sampled_changes = np.flatnonzero(mask[1:] != mask[:-1]) + 1
    if len(transitions) not in {0, sampled_changes.size}:
        raise ValueError("transitions must be empty or match every sampled availability change.")
    if transitions:
        for right_index, transition in zip(sampled_changes, transitions, strict=True):
            index = int(right_index)
            expected_kind = "acquisition" if mask[index] else "loss"
            if transition.transition_kind != expected_kind:
                raise ValueError("Transition kind does not match the sampled availability change.")
            if (
                not np.isfinite(transition.time_s)
                or not np.isfinite(transition.bracket_start_s)
                or not np.isfinite(transition.bracket_end_s)
                or transition.bracket_start_s < times[index - 1]
                or transition.bracket_end_s > times[index]
                or transition.bracket_start_s > transition.time_s
                or transition.time_s > transition.bracket_end_s
                or transition.bracket_start_s > transition.bracket_end_s
            ):
                raise ValueError("Transition timing must remain inside its sampled change bracket.")
            if transition.disposition not in {
                "sample_bounded",
                "provider_refined",
                "iteration_limited",
            }:
                raise ValueError("Transition disposition is unsupported.")
            if (
                isinstance(transition.iterations, (bool, np.bool_))
                or int(transition.iterations) != transition.iterations
                or int(transition.iterations) < 0
                or not str(transition.reason_before or "").strip()
                or not str(transition.reason_after or "").strip()
            ):
                raise ValueError("Transition evidence is malformed.")
            transition_by_right_index[int(right_index)] = transition

    intervals: list[SampledAvailabilityInterval] = []
    active_start: float | None = float(times[0]) if mask[0] else None
    start_censored = bool(mask[0])
    acquisition_disposition = "study_start_censored" if mask[0] else "not_applicable"
    acquisition_reason = "available" if mask[0] else "not_applicable"
    for right_index in sampled_changes:
        index = int(right_index)
        transition = transition_by_right_index.get(index)
        boundary_time = float(times[index]) if transition is None else transition.time_s
        disposition = "sample_bounded" if transition is None else transition.disposition
        if mask[index]:
            active_start = boundary_time
            start_censored = False
            acquisition_disposition = disposition
            acquisition_reason = (
                reason_tuple[index - 1]
                if transition is None
                else transition.reason_before
            )
            continue
        if active_start is None:
            raise RuntimeError("Loss transition encountered without an active interval.")
        intervals.append(
            SampledAvailabilityInterval(
                interval_index=len(intervals),
                start_s=active_start,
                end_s=boundary_time,
                duration_s=max(0.0, boundary_time - active_start),
                start_censored=start_censored,
                end_censored=False,
                acquisition_disposition=acquisition_disposition,
                loss_disposition=disposition,
                acquisition_reason=acquisition_reason,
                loss_reason=(
                    reason_tuple[index]
                    if transition is None
                    else transition.reason_after
                ),
            )
        )
        active_start = None

    if active_start is not None:
        intervals.append(
            SampledAvailabilityInterval(
                interval_index=len(intervals),
                start_s=active_start,
                end_s=float(times[-1]),
                duration_s=max(0.0, float(times[-1]) - active_start),
                start_censored=start_censored,
                end_censored=True,
                acquisition_disposition=acquisition_disposition,
                loss_disposition="study_end_censored",
                acquisition_reason=acquisition_reason,
                loss_reason="available",
            )
        )
    return tuple(intervals)


__all__ = [
    "AvailabilityEvaluator",
    "RefinedTransition",
    "SampledAvailabilityInterval",
    "availability_intervals",
    "refine_availability_transitions",
]
