"""Trajectory sampling and named frame transformations; no orbit propagation."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from scipy.interpolate import CubicHermiteSpline

from sim.frame_time import FrameTransformContext, transform_cartesian_state, transform_covariance

from .contracts import DataIOError, Trajectory
from .time import at


class TrajectorySampler:
    """Piecewise cubic Hermite PV interpolation with no extrapolation.

    A segment is one continuous arc. Maneuver discontinuities must be separate
    segments; no implicit bridge across a user-declared break is permitted.
    """

    def __init__(self, trajectory: Trajectory, *, max_gap_s: float):
        if len(trajectory.times_s) < 2:
            raise DataIOError("Interpolation requires at least two PV samples.")
        if not np.isfinite(max_gap_s) or max_gap_s <= 0:
            raise DataIOError("Explicit finite positive max_gap_s is required.")
        self.trajectory = trajectory
        self.max_gap_s = max_gap_s
        self._spline = CubicHermiteSpline(
            trajectory.times_s, trajectory.states[:, :3], trajectory.states[:, 3:], extrapolate=False
        )

    def state_at(self, time_s: float) -> np.ndarray:
        t = float(time_s)
        times = self.trajectory.times_s
        if not np.isfinite(t) or t < times[0] or t > times[-1]:
            raise DataIOError("Requested epoch lies outside trajectory coverage; extrapolation is forbidden.")
        exact = np.flatnonzero(times == t)
        if exact.size:
            return self.trajectory.states[int(exact[0])].copy()
        upper = int(np.searchsorted(times, t))
        if upper in self.trajectory.metadata.get("discontinuity_sample_indices", []):
            raise DataIOError("Interpolation across a declared trajectory discontinuity is forbidden.")
        if times[upper] - times[upper - 1] > self.max_gap_s:
            raise DataIOError("Interpolation interval exceeds max_gap_s.")
        return np.concatenate((self._spline(t), self._spline(t, 1)))


def transform_trajectory(trajectory: Trajectory, target_frame: str, *, eop_series=None) -> Trajectory:
    if trajectory.center.upper() != "EARTH":
        raise DataIOError(
            "Earth frame transforms require Earth-centered data; use explicit SPICE center/frame conversion."
        )
    states, matrices = [], []
    for time, state in zip(trajectory.times_s, trajectory.states):
        epoch = at(trajectory.epoch, time)
        context = FrameTransformContext(
            epoch, earth_orientation=None if eop_series is None else eop_series.sample(epoch)
        )
        r, v = transform_cartesian_state(state[:3], state[3:], trajectory.frame, target_frame, context=context)
        states.append(np.concatenate((r, v)))
    if trajectory.covariances is not None:
        for time, matrix in zip(trajectory.covariance_times_s, trajectory.covariances):
            epoch = at(trajectory.epoch, time)
            context = FrameTransformContext(
                epoch, earth_orientation=None if eop_series is None else eop_series.sample(epoch)
            )
            matrices.append(transform_covariance(matrix, trajectory.frame, target_frame, context=context))
    return replace(
        trajectory,
        frame=target_frame,
        states=states,
        covariances=matrices or None,
        metadata={**trajectory.metadata, "frame_conversion": {"source": trajectory.frame, "target": target_frame}},
    )
