"""Validated interchange values, independent of file syntax and simulation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from sim.frame_time import Epoch

MAX_BYTES = 64 * 1024 * 1024
MAX_SAMPLES = 200_000
MAX_OBJECTS = 10_000


class DataIOError(ValueError):
    """Invalid data, unsupported profile, or an unsafe/lossy conversion."""


def reject_booleans(value):
    if isinstance(value, (bool, np.bool_)):
        raise DataIOError("Numeric data must not contain booleans.")
    if isinstance(value, (list, tuple)):
        for item in value:
            reject_booleans(item)
    elif isinstance(value, np.ndarray) and value.dtype.kind in {"b", "O"}:
        for item in value.flat:
            reject_booleans(item)


def finite_array(value: Any, shape: tuple[int, ...], label: str) -> np.ndarray:
    reject_booleans(value)
    try:
        result = np.array(value, dtype=float, copy=True)
    except (ValueError, TypeError) as exc:
        raise DataIOError(f"{label} must be numeric.") from exc
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise DataIOError(f"{label} requires finite values with shape {shape}.")
    result.flags.writeable = False
    return result


def text(value: str, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or any(ord(c) < 32 for c in value):
        raise DataIOError(f"{label} must be nonempty text without control characters.")
    return value.strip()


def validated_times(value: Any) -> np.ndarray:
    reject_booleans(value)
    result = np.array(value, dtype=float, copy=True)
    if result.ndim != 1 or not 1 <= len(result) <= MAX_SAMPLES:
        raise DataIOError(f"Expected 1..{MAX_SAMPLES} sample times.")
    if not np.all(np.isfinite(result)) or np.any(np.diff(result) <= 0):
        raise DataIOError("Sample times must be finite and strictly increasing.")
    result.flags.writeable = False
    return result


@dataclass(frozen=True)
class Trajectory:
    """One object's PV history; km, km/s, SI elapsed seconds from epoch.

    The frame is named explicitly. No generic ECI/ECEF aliases are inferred.
    Covariance samples can have their own epochs; they are never fabricated.
    """

    object_id: str
    center: str
    frame: str
    epoch: Epoch
    times_s: np.ndarray
    states: np.ndarray
    object_name: str = ""
    covariance_times_s: np.ndarray | None = None
    covariances: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("object_id", "center", "frame"):
            object.__setattr__(self, name, text(getattr(self, name), name))
        if self.frame.upper() in {"ECI", "ECEF", "J2000"}:
            raise DataIOError("Use a named frame (EME2000, GCRF, TEME, ITRF, or SPICE:<name>).")
        if not isinstance(self.epoch, Epoch):
            raise DataIOError("Trajectory epoch must be a physical Epoch.")
        times = validated_times(self.times_s)
        object.__setattr__(self, "times_s", times)
        object.__setattr__(self, "states", finite_array(self.states, (len(times), 6), "PV states"))
        if (self.covariances is None) != (self.covariance_times_s is None):
            raise DataIOError("Covariance requires both epochs and full matrices.")
        if self.covariances is not None:
            ct = validated_times(self.covariance_times_s)
            cv = finite_array(self.covariances, (len(ct), 6, 6), "Cartesian covariance")
            for matrix in cv:
                scale = max(float(np.max(np.abs(matrix))), 1e-30)
                if not np.allclose(matrix, matrix.T, atol=scale * 1e-12, rtol=1e-12):
                    raise DataIOError("Covariance must be symmetric.")
                if float(np.linalg.eigvalsh(matrix).min()) < -scale * 1e-12:
                    raise DataIOError("Covariance must be positive semidefinite.")
            if ct[0] < times[0] or ct[-1] > times[-1]:
                raise DataIOError("Covariance epochs must lie within trajectory coverage.")
            object.__setattr__(self, "covariance_times_s", ct)
            object.__setattr__(self, "covariances", cv)
        json.dumps(self.metadata, allow_nan=False)


@dataclass(frozen=True)
class AttitudeHistory:
    """Hamilton scalar-first quaternions rotating reference vectors to body.

    Angular rates, when present, are body angular velocity relative to the
    reference, expressed in body coordinates in rad/s.
    """

    object_id: str
    reference_frame: str
    body_frame: str
    epoch: Epoch
    times_s: np.ndarray
    quaternions: np.ndarray
    rates_rad_s: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("object_id", "reference_frame", "body_frame"):
            object.__setattr__(self, name, text(getattr(self, name), name))
        if not isinstance(self.epoch, Epoch):
            raise DataIOError("Attitude epoch must be a physical Epoch.")
        times = validated_times(self.times_s)
        q = finite_array(self.quaternions, (len(times), 4), "quaternions")
        if not np.allclose(np.linalg.norm(q, axis=1), 1.0, atol=1e-8, rtol=0):
            raise DataIOError("Attitude quaternions must be unit length; no silent normalization.")
        object.__setattr__(self, "times_s", times)
        object.__setattr__(self, "quaternions", q)
        if self.rates_rad_s is not None:
            object.__setattr__(self, "rates_rad_s", finite_array(self.rates_rad_s, (len(times), 3), "angular rates"))
        json.dumps(self.metadata, allow_nan=False)
