from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sim.core.models import SimLog


@dataclass(frozen=True)
class EngagementMetrics:
    min_separation_km: float
    time_inside_keepout_s: float
    fuel_used_kg_by_object: dict[str, float]
    compute_overruns_by_object: dict[str, int]
    jitter_ms_by_object: dict[str, float]


def _sampled_dwell_time_s(mask: np.ndarray, time_s: np.ndarray) -> float:
    inside = np.array(mask, dtype=bool).reshape(-1)
    t = np.array(time_s, dtype=float).reshape(-1)
    n = min(inside.size, t.size)
    if n < 2:
        return 0.0
    dt = np.diff(t[:n])
    valid = np.isfinite(dt) & (dt > 0.0)
    if not np.any(valid):
        return 0.0
    return float(np.sum(dt[valid] * inside[: n - 1][valid]))


def compute_engagement_metrics(log: SimLog, keepout_radius_km: float | None = None) -> EngagementMetrics:
    object_ids = sorted(log.truth_by_object.keys())
    t = log.t_s

    min_sep = np.inf
    time_inside = 0.0
    for i, oid_i in enumerate(object_ids):
        ri = log.truth_by_object[oid_i][:, :3]
        for oid_j in object_ids[i + 1 :]:
            rj = log.truth_by_object[oid_j][:, :3]
            n = min(ri.shape[0], rj.shape[0])
            if n == 0:
                continue
            d = np.linalg.norm(ri[:n] - rj[:n], axis=1)
            finite_d = d[np.isfinite(d)]
            if finite_d.size:
                min_sep = min(min_sep, float(np.min(finite_d)))
            if keepout_radius_km is not None:
                time_inside += _sampled_dwell_time_s(d < keepout_radius_km, t)
    if np.isinf(min_sep):
        min_sep = 0.0

    fuel_used = {}
    overruns = {}
    jitter = {}
    for oid in object_ids:
        mass = np.asarray(log.truth_by_object[oid], dtype=float)
        mass = mass[:, 13] if mass.ndim == 2 and mass.shape[1] > 13 else np.empty(0, dtype=float)
        finite_mass = mass[np.isfinite(mass)]
        if finite_mass.size >= 2:
            fuel_used[oid] = float(max(0.0, finite_mass[0] - finite_mass[-1]))
        else:
            fuel_used[oid] = 0.0
        skip = np.asarray(log.controller_skipped_by_object[oid], dtype=bool).reshape(-1)
        rt = np.asarray(log.controller_runtime_ms_by_object[oid], dtype=float).reshape(-1)
        overruns[oid] = int(np.sum(skip))
        measured_rt = rt[1:] if rt.size > 1 else np.empty(0, dtype=float)
        measured_rt = measured_rt[np.isfinite(measured_rt)]
        jitter[oid] = float(np.std(measured_rt)) if measured_rt.size else 0.0

    return EngagementMetrics(
        min_separation_km=min_sep,
        time_inside_keepout_s=time_inside,
        fuel_used_kg_by_object=fuel_used,
        compute_overruns_by_object=overruns,
        jitter_ms_by_object=jitter,
    )
