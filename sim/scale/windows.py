from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class PropagationWindow:
    start_offset_s: float
    end_offset_s: float
    start_jd_utc: float
    end_jd_utc: float
    step_s: float
    sample_count: int


def build_propagation_windows(
    *,
    start_jd_utc: float,
    duration_s: float,
    step_s: float,
    chunk_duration_s: float,
) -> tuple[PropagationWindow, ...]:
    windows: list[PropagationWindow] = []
    start_offset = 0.0
    while start_offset < float(duration_s) - 1.0e-9:
        end_offset = min(float(duration_s), start_offset + float(chunk_duration_s))
        start_jd = float(start_jd_utc) + start_offset / 86400.0
        end_jd = float(start_jd_utc) + end_offset / 86400.0
        windows.append(
            PropagationWindow(
                start_offset_s=float(start_offset),
                end_offset_s=float(end_offset),
                start_jd_utc=start_jd,
                end_jd_utc=end_jd,
                step_s=float(step_s),
                sample_count=sample_count_for_duration(duration_s=end_offset - start_offset, step_s=step_s),
            )
        )
        start_offset = end_offset
    return tuple(windows)


def sample_count_for_duration(*, duration_s: float, step_s: float) -> int:
    step = float(step_s)
    if not math.isfinite(step) or step <= 0.0:
        raise ValueError("step_s must be positive and finite.")
    duration = float(duration_s)
    if not math.isfinite(duration) or duration < 0.0:
        raise ValueError("duration_s must be nonnegative and finite.")
    chunk_duration = round(duration, 9)
    # Keep the grid within the declared interval.  This mirrors the engine's
    # floor(duration / dt) + 1 contract and avoids emitting a sample beyond a
    # partial final Scale window.
    ratio = chunk_duration / step
    nearest_step_count = round(ratio)
    # Julian-date subtraction loses a few microseconds at OEL's supported
    # epoch magnitudes.  Recover an intended integer cadence without rounding
    # genuinely partial windows up to an extra sample.
    nearest_duration = float(nearest_step_count) * step
    if abs(chunk_duration - nearest_duration) <= 5.0e-5:
        ratio = float(nearest_step_count)
    return int(math.floor(ratio + 1.0e-12)) + 1


def sample_count_for_window(*, start_jd_utc: float, end_jd_utc: float, step_s: float) -> int:
    duration_s = max(0.0, round((float(end_jd_utc) - float(start_jd_utc)) * 86400.0, 9))
    return sample_count_for_duration(duration_s=duration_s, step_s=step_s)
