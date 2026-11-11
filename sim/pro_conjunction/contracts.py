"""Closed, epoch-tagged Pro conjunction contracts (public v1 stays unchanged)."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping

import numpy as np

from sim.estimation.covariance import validate_covariance

PROBLEM_SCHEMA = "oel.pro_conjunction_problem.v1"
EVIDENCE_SCHEMA = "oel.pro_conjunction_evidence.v1"


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def closed(value, required, optional=(), *, name="record"):
    if not isinstance(value, Mapping) or set(value) - set(required) - set(optional) or set(required) - set(value):
        raise ValueError(f"{name}: missing or unknown fields; required={sorted(required)} optional={sorted(optional)}")
    return dict(value)


def number(value, *, low=None, high=None, integer=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("expected finite numeric value, not boolean")
    if (low is not None and value < low) or (high is not None and value > high) or (integer and int(value) != value):
        raise ValueError(f"numeric value outside [{low}, {high}] or not integral")
    return int(value) if integer else float(value)


def array(value, shape):
    def check(x):
        if isinstance(x, (list, tuple)):
            for child in x:
                check(child)
        else:
            number(x)

    check(value)
    x = np.asarray(value, dtype=float)
    if x.shape != shape:
        raise ValueError(f"expected shape {shape}")
    return x


def covariance(value, dimension):
    return validate_covariance(array(value, (dimension, dimension)), dimension).tolist()


def text(value):
    if not isinstance(value, str) or not value.strip() or len(value) > 512:
        raise ValueError("expected nonempty text of at most 512 characters")
    return value


def burn_record(value, duration):
    b = closed(value, {"time_s", "frame", "delta_v_m_s", "execution_covariance_m2_s2"}, name="burn")
    b["time_s"] = number(b["time_s"], low=0, high=duration)
    if b["frame"] not in {"eci", "ric"}:
        raise ValueError("burn frame must be eci or ric")
    b["delta_v_m_s"] = array(b["delta_v_m_s"], (3,)).tolist()
    b["execution_covariance_m2_s2"] = covariance(b["execution_covariance_m2_s2"], 3)
    return b


def validate_problem(value):
    """Inspection only; no entitlement, importing plugins, or execution."""
    p = closed(
        value,
        {
            "schema_version",
            "name",
            "epoch_jd_utc",
            "duration_s",
            "step_s",
            "force_model",
            "objects",
            "primary_id",
            "secondary_ids",
            "error_dependence",
            "applicability",
        },
        {"burn", "monte_carlo"},
        name="conjunction problem",
    )
    if p["schema_version"] != PROBLEM_SCHEMA or p["error_dependence"] != "independent_objects":
        raise ValueError("unsupported schema or object-error dependence; correlated objects are not supported")
    p["name"] = text(p["name"])
    p["epoch_jd_utc"] = number(p["epoch_jd_utc"], low=2440000, high=2500000)
    p["duration_s"] = number(p["duration_s"], low=0.001, high=86400)
    p["step_s"] = number(p["step_s"], low=0.001, high=p["duration_s"])
    if math.ceil(p["duration_s"] / p["step_s"]) > 2000:
        raise ValueError("at most 2000 propagation intervals per history")
    if p["force_model"] not in {"two_body", "j2"}:
        raise ValueError("supported force models: two_body, j2")
    if not isinstance(p["objects"], list) or not 2 <= len(p["objects"]) <= 32:
        raise ValueError("declare 2..32 objects")
    objects = []
    for value in p["objects"]:
        o = closed(
            value,
            {
                "object_id",
                "state_eci_km_km_s",
                "uncertainty",
                "hard_body_radius_m",
                "acceleration_spectral_density_km2_s3",
            },
            name="object",
        )
        o["object_id"] = text(o["object_id"])
        o["state_eci_km_km_s"] = array(o["state_eci_km_km_s"], (6,)).tolist()
        if np.linalg.norm(o["state_eci_km_km_s"][:3]) <= 6378.137:
            raise ValueError("Earth-orbit initial position must be above WGS84 equatorial radius")
        o["hard_body_radius_m"] = number(o["hard_body_radius_m"], low=1e-6, high=1000)
        q = array(o["acceleration_spectral_density_km2_s3"], (3,))
        if np.any(q < 0):
            raise ValueError("process spectral density must be nonnegative")
        o["acceleration_spectral_density_km2_s3"] = q.tolist()
        u = closed(
            o["uncertainty"],
            {"epoch_jd_utc", "frame", "units", "covariance", "source", "calibration"},
            name="uncertainty",
        )
        if u["epoch_jd_utc"] != p["epoch_jd_utc"] or u["frame"] != "EME2000" or u["units"] != "km,km/s":
            raise ValueError("state/covariance must share declared initial epoch, EME2000 frame and km,km/s units")
        number(u["epoch_jd_utc"])
        u["covariance"] = covariance(u["covariance"], 6)
        u["source"] = text(u["source"])
        if u["calibration"] not in {"assumed", "uncalibrated", "source_declared_calibrated"}:
            raise ValueError("declare covariance calibration disposition")
        o["uncertainty"] = u
        objects.append(o)
    p["objects"] = objects
    ids = [o["object_id"] for o in objects]
    if len(set(ids)) != len(ids) or p["primary_id"] not in ids:
        raise ValueError("unique object IDs and an existing primary are required")
    if not isinstance(p["secondary_ids"], list) or not p["secondary_ids"]:
        raise ValueError("declare a nonempty complete secondary inventory")
    if len(set(p["secondary_ids"])) != len(p["secondary_ids"]) or set(p["secondary_ids"]) != set(ids) - {
        p["primary_id"]
    }:
        raise ValueError("every non-primary object must be rescreened exactly once")
    a = closed(
        p["applicability"],
        {"sigma_extent", "max_curvature_sigma", "max_covariance_fraction", "max_velocity_sigma_fraction"},
    )
    a["sigma_extent"] = number(a["sigma_extent"], low=3, high=10)
    for key in set(a) - {"sigma_extent"}:
        a[key] = number(a[key], low=1e-8, high=1)
    p["applicability"] = a
    if "burn" in p:
        p["burn"] = burn_record(p["burn"], p["duration_s"])
    if "monte_carlo" in p:
        m = closed(p["monte_carlo"], {"trials", "seed", "confidence", "max_interval_width"})
        m["trials"] = number(m["trials"], low=64, high=10000, integer=True)
        m["seed"] = number(m["seed"], low=0, high=2**32 - 1, integer=True)
        m["confidence"] = number(m["confidence"], low=0.8, high=0.9999)
        m["max_interval_width"] = number(m["max_interval_width"], low=1e-6, high=0.5)
        if m["trials"] * len(ids) * math.ceil(p["duration_s"] / p["step_s"]) > 2_000_000:
            raise ValueError("Monte Carlo exceeds two million object-step budget")
        if any(np.any(o["acceleration_spectral_density_km2_s3"]) for o in objects):
            raise ValueError("nonlinear Monte Carlo v1 requires zero process noise; initial and burn uncertainty only")
        p["monte_carlo"] = m
    # Canonical clone prevents caller mutation after validation.
    return json.loads(json.dumps(p, allow_nan=False))
