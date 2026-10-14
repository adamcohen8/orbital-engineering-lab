"""Content-bound rotation definitions; native stage evaluation uses TDB."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import numpy as np

from .resources import read_json


@dataclass(frozen=True)
class PreparedRotation:
    frame_id: str
    body_id: str
    definition: dict
    epoch_jd_utc: float
    receipt: object | None = None
    initial_tt_s: float = field(init=False)

    def __post_init__(self):
        from sim.dynamics.orbit.frame_time import parse_epoch, tai_minus_utc

        instant = datetime(2000, 1, 1, 12, tzinfo=timezone.utc) + timedelta(
            days=self.epoch_jd_utc - 2451545.0
        )
        dat = tai_minus_utc(
            parse_epoch(instant.isoformat().replace("+00:00", "Z"), "UTC")
        )
        object.__setattr__(
            self,
            "initial_tt_s",
            (self.epoch_jd_utc - 2451545.0) * 86400.0 + dat + 32.184,
        )

    def native_input(self):
        d = self.definition
        pole = d["kind"] == "pole_meridian"
        if pole:
            axes = [2, 0, 2]
            values = []
            for key, scale in (
                ("ra_deg", 36525.0),
                ("dec_deg", 36525.0),
                ("meridian_deg", 1.0),
            ):
                polynomial = d[key]
                if not isinstance(polynomial, list) or not 2 <= len(polynomial) <= 3:
                    raise ValueError(
                        "Planetary rotation polynomial requires two or three coefficients"
                    )
                values += [
                    polynomial[0],
                    polynomial[1] / scale,
                    (polynomial[2] if len(polynomial) == 3 else 0.0) / scale**2,
                ]
            terms = d.get("periodic_terms", [])
        elif d["kind"] == "euler_series":
            axes = d["axes"]
            values = []
            terms = []
            for i, angle in enumerate(d["angles"]):
                values += [
                    angle["offset_deg"],
                    angle["rate_deg_day"],
                    angle.get("quadratic_deg_day2", 0.0),
                ]
                terms += [{**term, "angle": i} for term in angle.get("terms", [])]
        else:
            raise ValueError("Unsupported planetary rotation kind")
        periodic = []
        for term in terms:
            if term["trig"] not in {"sin", "cos"}:
                raise ValueError("Planetary rotation term must use sin or cos")
            periodic.append(
                [
                    float(term["angle"]),
                    term["amplitude_deg"],
                    term["phase_deg"],
                    term["rate_deg_day"],
                    float(term["trig"] == "cos"),
                ]
            )
        return axes, values, periodic, d["valid_tdb_seconds"], pole

    def elapsed_tdb_s(self, t_s):
        from .native import module

        return module().planetary_tdb_seconds(self.initial_tt_s, float(t_s))

    def evaluate(self, t_s):
        from .native import module

        r, d = module().planetary_rotation(
            self.native_input(), self.initial_tt_s, float(t_s)
        )
        return np.asarray(r).reshape(3, 3), np.asarray(d).reshape(3, 3)

    def metadata(self):
        return {
            "frame_id": self.frame_id,
            "body_id": self.body_id,
            "inertial_axes": "J2000",
            "time_scale": "TDB",
            "time_conversion": "TAI elapsed + TT + low-order TDB",
            "time_conversion_limit": "Low-order TT-TDB approximation; does not claim SPICE ET accuracy",
            "definition": deepcopy(self.definition),
            "sha256": None if self.receipt is None else self.receipt.sha256,
        }


def prepare_rotation(body_id, selection, epoch_jd_utc):
    from .resources import DATA_ROOT

    model = selection.get("model", "iau2000")
    path = (
        selection["data_path"]
        if model == "resource"
        else DATA_ROOT / f"{body_id}_{model}_rotation.json"
    )
    receipt, data = read_json(path, selection.get("sha256"))
    if data.get("body_id") != body_id or data.get("inertial_axes") != "J2000":
        raise ValueError("Planetary rotation resource body/axes mismatch")
    if (
        not isinstance(data.get("frame_id"), str)
        or not data.get("frame_id")
        or not data.get("source")
    ):
        raise ValueError("Planetary rotation requires frame identity and source")
    result = PreparedRotation(data["frame_id"], body_id, data, epoch_jd_utc, receipt)
    r, d = result.evaluate(0.0)
    if (
        not np.all(np.isfinite(d))
        or not np.allclose(r @ r.T, np.eye(3), atol=2e-14, rtol=0.0)
        or np.linalg.det(r) < 0.999999999999
    ):
        raise ValueError("Invalid planetary rotation")
    return result
