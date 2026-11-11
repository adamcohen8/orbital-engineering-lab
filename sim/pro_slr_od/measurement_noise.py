"""Explicit normal-point mean noise and independent CRD-session random effects.

Noise parameters are trained/frozen before prediction. This owner constructs
covariance; it never estimates parameters from forecast observations.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class SlrSessionNoise:
    white_floor_m: float
    session_sigma_m: float
    session_drift_sigma_m_per_600s: float = 0.

    def __post_init__(self):
        if (not np.all(np.isfinite([self.white_floor_m, self.session_sigma_m,
                                   self.session_drift_sigma_m_per_600s]))
                or self.white_floor_m <= 0 or min(self.session_sigma_m, self.session_drift_sigma_m_per_600s) < 0):
            raise ValueError("SLR noise requires positive white floor and nonnegative session sigma.")

    def policy(self):
        values = asdict(self)
        if self.session_drift_sigma_m_per_600s == 0.:
            values.pop("session_drift_sigma_m_per_600s")
            return dict(schema="oel.slr_session_noise.v1", **values)
        return dict(schema="oel.slr_session_noise.v2", **values)

    @classmethod
    def from_policy(cls, value):
        if not isinstance(value, dict):
            raise ValueError("Invalid frozen SLR noise policy.")
        schema = value.get("schema")
        keys = {"schema", "white_floor_m", "session_sigma_m"}
        if schema == "oel.slr_session_noise.v2":
            keys.add("session_drift_sigma_m_per_600s")
        if schema not in {"oel.slr_session_noise.v1", "oel.slr_session_noise.v2"} or set(value) != keys:
            raise ValueError("Unsupported frozen SLR noise schema.")
        return cls(float(value["white_floor_m"]), float(value["session_sigma_m"]),
                   float(value.get("session_drift_sigma_m_per_600s", 0.)))

    def covariance_km2(self, rows):
        if not 1 <= len(rows) <= 1000:
            raise ValueError("SLR session covariance requires one to 1000 normal points.")
        sem, keys, drift_times = [], [], []
        references = {}
        for row in rows:
            rms, count = float(row["bin_rms_ps"]), float(row["raw_range_count"])
            identity = str(row["measurement_id"]).split(":")
            if (not np.all(np.isfinite([rms, count])) or rms < 0 or count < 1 or count != int(count)
                    or len(identity) != 3 or not str(row["source_sha256"])):
                raise ValueError("Invalid SLR normal-point scatter, count or session identity.")
            # CRD bin RMS is single-return scatter, not the error of its mean.
            # A zero supplied scatter supplies no information; the frozen floor
            # still applies. The floor includes effects that do not average down.
            sem.append(.5 * 299792.458e-12 * rms / np.sqrt(count))
            keys.append((str(row["source_sha256"]), identity[1], str(row["station_id"])))
            if self.session_drift_sigma_m_per_600s:
                reference = row.get("slr_noise_session_reference", {})
                if (reference.get("model") != "crd_h4_midpoint.v1"
                        or reference.get("source_sha256") != row["source_sha256"]):
                    raise ValueError("SLR session drift requires source-bound CRD H4 reference epochs.")
                epoch = float(reference["jd_utc"])
                if not np.isfinite(epoch) or references.setdefault(keys[-1], epoch) != epoch:
                    raise ValueError("Invalid or inconsistent SLR session reference epoch.")
                drift_times.append(((float(row["time_jd_utc"])-epoch)*86400.
                                    + float(row.get("epoch_offset_s", 0.)))/600.)
        same_session = np.array([[a == b for b in keys] for a in keys], dtype=float)
        result = np.diag(np.array(sem)**2 + (self.white_floor_m/1000.)**2) + (
            self.session_sigma_m/1000.)**2 * same_session
        if self.session_drift_sigma_m_per_600s:
            if not np.all(np.isfinite(drift_times)):
                raise ValueError("Invalid SLR session drift epochs.")
            result += (self.session_drift_sigma_m_per_600s/1000.)**2 * np.outer(drift_times, drift_times)*same_session
        return result


def bind_session_noise_epochs(rows, packets):
    """Bind a source-owned H4 midpoint, invariant under observation subsetting."""
    from copy import deepcopy

    sources = {str(p["source_sha256"]): p for p in packets}
    if len(sources) != len(packets):
        raise ValueError("Duplicate SLR session-noise source packet.")
    result = deepcopy(list(rows))
    for row in result:
        if "slr_noise_session_reference" in row:
            raise ValueError("SLR session-noise epoch is already bound.")
        try:
            _, session_index, point_index = str(row["measurement_id"]).split(":")
            session = sources[str(row["source_sha256"])]["sessions"][int(session_index)]
            point = session["normal_points"][int(point_index)]
        except (KeyError, IndexError, ValueError) as exc:
            raise ValueError("Invalid SLR session-noise source identity.") from exc
        if (session["station"]["code"].upper() != row["station_id"]
                or float(point["time_jd_utc"]) != float(row["time_jd_utc"])):
            raise ValueError("SLR session-noise observation identity mismatch.")
        start, end = float(session["start_jd_utc"]), float(session["end_jd_utc"])
        if not np.all(np.isfinite([start, end])) or end < start:
            raise ValueError("Invalid CRD H4 session interval.")
        row["slr_noise_session_reference"] = dict(model="crd_h4_midpoint.v1",
            source_sha256=row["source_sha256"], jd_utc=start+.5*(end-start))
    return result
