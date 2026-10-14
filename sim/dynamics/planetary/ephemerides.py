"""Geometric center-to-center Hermite tables; no geocentric fallback."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .resources import read_json


@dataclass(frozen=True)
class PreparedEphemeris:
    observer_center_id: int
    times_tdb_s: np.ndarray
    states: dict[int, np.ndarray]
    receipt: object
    source: str


def prepare_ephemeris(body, config, frame, start_s, end_s, targets):
    if not targets:
        return None
    if config.get("model", "table") != "table":
        raise ValueError("Planetary ephemeris model must be table")
    receipt, data = read_json(config["data_path"], config.get("sha256"))
    if (
        data.get("observer_center_id") != body.center_id
        or data.get("inertial_axes") != "J2000"
        or data.get("aberration") != "NONE"
    ):
        raise ValueError(
            "Planetary ephemeris requires matching physical origin, J2000 and simultaneous geometric NONE states"
        )
    if (
        data.get("position_units") != "km"
        or data.get("velocity_units") != "km/s"
        or data.get("time_scale") != "TDB"
    ):
        raise ValueError("Planetary ephemeris units/time-scale mismatch")
    if not data.get("source"):
        raise ValueError("Planetary ephemeris requires source provenance")
    times = np.asarray(data["tdb_seconds_past_j2000"], dtype=float)
    if (
        times.ndim != 1
        or times.size < 2
        or not np.all(np.isfinite(times))
        or np.any(np.diff(times) <= 0)
    ):
        raise ValueError("Invalid ephemeris time grid")
    states = {}
    for key, values in data["states_by_target_center"].items():
        target = int(key)
        if str(target) != key or target in states:
            raise ValueError("Ephemeris target IDs must be unique canonical integers")
        states[target] = np.asarray(values, dtype=float)
    for target in set(targets) | set(states):
        if (
            target not in states
            or states[target].shape != (len(times), 6)
            or not np.all(np.isfinite(states[target]))
        ):
            raise ValueError(f"Invalid ephemeris states for center {target}")
    if (
        not times[0]
        <= frame.elapsed_tdb_s(start_s)
        <= frame.elapsed_tdb_s(end_s)
        <= times[-1]
    ):
        raise ValueError("Planetary ephemeris does not cover the entire study interval")
    times.setflags(write=False)
    for s in states.values():
        s.setflags(write=False)
    return PreparedEphemeris(body.center_id, times, states, receipt, data["source"])
