"""Prepare optional extended external fields without altering Earth defaults."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import numpy as np

from .definitions import BodyDefinition, body_definition
from .ephemerides import prepare_ephemeris
from .frames import PreparedRotation, prepare_rotation
from .gravity import PreparedGravity, prepare_gravity
from .resources import ResourceReceipt, semantic_digest


def external_body(body_id):
    if body_id == "earth":
        from sim.dynamics.orbit.environment import EARTH_MU_KM3_S2, EARTH_RADIUS_KM

        return BodyDefinition("earth", 399, EARTH_MU_KM3_S2, EARTH_RADIUS_KM)
    return body_definition(body_id)


def _earth_gravity(body, selection):
    from sim.dynamics.orbit.spherical_harmonics import (
        SphericalHarmonicTerm,
        compile_spherical_harmonic_terms,
    )

    receipt, content = ResourceReceipt.read(selection["coeff_path"], selection.get("sha256"))
    seen = set()
    headers = {}
    terms = []
    n, m = selection["max_degree"], selection["max_order"]
    for line in content.decode("utf-8").splitlines():
        row = line.split("#", 1)[0].split()
        if not row:
            continue
        key = row[0].lower()
        if key in {"norm", "radius", "earth_gravity_constant"}:
            if len(row) != 2 or key in headers:
                raise ValueError(f"Invalid or duplicate ICGEM {key} header")
            headers[key] = row[1]
            continue
        if key in {"gfct", "trnd", "dot", "asin", "acos"}:
            raise ValueError("Additional Earth gravity requires a static ICGEM dataset")
        if key != "gfc":
            continue
        if len(row) not in {5, 7}:
            raise ValueError("Invalid ICGEM coefficient record")
        degree, order = int(row[1]), int(row[2])
        values = [float(v.replace("D", "E").replace("d", "e")) for v in row[3:]]
        if (
            not 0 <= order <= degree
            or (degree, order) in seen
            or not np.all(np.isfinite(values))
            or (order == 0 and values[1] != 0)
            or any(v < 0 for v in values[2:])
        ):
            raise ValueError("Invalid or duplicate ICGEM coefficient")
        if degree == 0 and (values[0] != 1 or values[1] != 0) or degree == 1 and any(values[:2]):
            raise ValueError("ICGEM degree zero/one must represent the center-of-mass monopole")
        seen.add((degree, order))
        if 2 <= degree <= n and order <= m:
            terms.append(SphericalHarmonicTerm(degree, order, *values[:2], normalized=True))
    if headers.get("norm", "").lower() != "fully_normalized":
        raise ValueError("Additional Earth gravity requires explicit fully_normalized ICGEM coefficients")
    if not {(i, j) for i in range(2, n + 1) for j in range(min(i, m) + 1)} <= seen:
        raise ValueError("ICGEM is missing requested coefficients")
    if not {"earth_gravity_constant", "radius"} <= headers.keys():
        raise ValueError("Additional Earth gravity requires dataset GM and reference radius")
    mu, radius = (
        float(headers[key].replace("D", "E").replace("d", "e")) / scale
        for key, scale in (("earth_gravity_constant", 1e9), ("radius", 1000.0))
    )
    if not np.all(np.isfinite([mu, radius])) or mu <= 0 or radius <= 0:
        raise ValueError("ICGEM dataset GM and reference radius must be finite and positive")
    # Compile the exact records validated above; a second parser must never
    # reinterpret an unvalidated record or a conflicting header.
    c = compile_spherical_harmonic_terms(terms)
    for v in vars(c).values():
        if isinstance(v, np.ndarray):
            v.setflags(write=False)
    return PreparedGravity(
        "earth_static",
        body.body_id,
        mu,
        radius,
        selection["frame_id"],
        c,
        (receipt,),
    )


def _clock(epoch, body_id="earth", frame_id="PREPARED_CLOCK"):
    return PreparedRotation(
        frame_id,
        body_id,
        {
            "kind": "euler_series",
            "axes": [2],
            "angles": [{"offset_deg": 0.0, "rate_deg_day": 0.0}],
            "valid_tdb_seconds": [-1.0e12, 1.0e12],
            "source": "time conversion only",
        },
        epoch,
    )


def _earth_frame(selection, gravity, epoch, duration):
    from sim.dynamics.orbit.accelerations import OrbitContext
    from sim.dynamics.orbit.propagator import OrbitPropagator, spherical_harmonics_plugin
    from sim.dynamics.orbit.rust_force_plan import make_plan

    receipt = ResourceReceipt.read(selection["eop_path"])[0]
    env = {
        "jd_utc_start": epoch,
        "spherical_harmonics_frame_model": "iau76_80_eop",
        "spherical_harmonics_eop_path": receipt.path,
        "eop_extrapolation": "error",
        "_compiled_spherical_harmonics_terms": gravity.compiled,
        "spherical_harmonics_reference_radius_km": gravity.radius_km,
    }
    adapter = OrbitPropagator(plugins=[spherical_harmonics_plugin], numeric_backend="rust")
    x = np.array([2.0 * gravity.radius_km, 0.0, 0.0, 0.0, 0.0, 0.0])
    *_, callback, _ = make_plan(adapter, x, 0.0, env, OrbitContext(gravity.mu_km3_s2, 1.0))
    if not getattr(callback, "native_preparation", False):
        raise ValueError("Additional Earth frame requires the native EOP stage context; rebuild rust/oel-orbit")
    # Coverage is checked using the same frame preparation used by ordinary Earth ONP.
    callback(0.0, x.tolist())
    callback(float(duration), x.tolist())
    receipt.verify()
    return _clock(epoch, "earth", selection["frame_id"]), callback, receipt


@dataclass
class PreparedAdditionalGravity:
    center_id: int
    clock: object
    duration_s: float
    ephemeris: object
    sources: list
    native: object
    digest: str
    receipts: tuple

    def verify_resources(self):
        for receipt in self.receipts:
            receipt.verify()

    def metadata(self):
        return {
            "origin_center_id": self.center_id,
            "inertial_axes": "J2000",
            "environment_sha256": self.digest,
            "force_convention": "full_field_at_spacecraft_minus_full_field_at_origin",
            "ephemeris": {
                "sha256": self.ephemeris.receipt.sha256,
                "source": self.ephemeris.source,
                "time_scale": "TDB",
                "aberration": "NONE",
                "interpolation": "cubic_hermite_position_and_velocity",
                "coverage_tdb_seconds": self.ephemeris.times_tdb_s[[0, -1]].tolist(),
            },
            "sources": deepcopy(self.sources),
        }

    def components(self, t_s, state):
        self.verify_resources()
        parts = self.native.stage(float(t_s), np.asarray(state, dtype=float).reshape(6).tolist())
        return {f"additional_gravity_{s['target_center_id']}": part for s, part in zip(self.sources, parts)}


def prepare_additional_gravity(cfg):
    from .native import module

    native_module = module()
    if not callable(getattr(native_module, "AdditionalGravityPlan", None)):
        raise ValueError(
            "Additional gravity requires oel-rust-orbit with AdditionalGravityPlan; rebuild rust/oel-orbit"
        )
    s = cfg.simulator
    origin = external_body(s.central_body)
    clock = _clock(s.initial_jd_utc)
    sources, inputs, frames, receipts = [], [], [], []
    for selected in s.additional_gravity["sources"]:
        body = external_body(selected["body_id"])
        gravity = (
            _earth_gravity(body, selected["gravity"])
            if selected["gravity"]["model"] == "earth_static"
            else prepare_gravity(body, selected["gravity"])
        )
        frame_config = selected["gravity_frame"]
        if frame_config["model"] == "earth":
            frame, callback, eop = _earth_frame(frame_config, gravity, s.initial_jd_utc, s.duration_s)
            frame_metadata = {
                "body_id": "earth",
                "frame_id": frame.frame_id,
                "model": "iau76_80_eop",
                "inertial_axes": "J2000",
                "eop_sha256": eop.sha256,
            }
            receipts.append(eop)
        else:
            frame = prepare_rotation(body.body_id, frame_config, s.initial_jd_utc)
            frame.evaluate(s.duration_s)
            callback = None
            frame_metadata = frame.metadata()
            receipts.append(frame.receipt)
        if gravity.compiled is None or frame.frame_id != gravity.frame_id:
            raise ValueError("Additional gravity requires harmonics and an exact matching coefficient/frame identity")
        c = gravity.compiled
        inputs.append(
            (
                body.center_id,
                gravity.mu_km3_s2,
                gravity.radius_km,
                (c.n_max, c.m_max),
                (c.c_nm.reshape(-1).tolist(), c.s_nm.reshape(-1).tolist()),
                frame.native_input(),
            )
        )
        frames.append(callback)
        receipts.extend(gravity.receipts)
        sources.append(
            {
                "body_id": body.body_id,
                "target_center_id": body.center_id,
                "gravity": gravity.metadata(),
                "gravity_frame": frame_metadata,
            }
        )
    ephem = prepare_ephemeris(
        origin, s.additional_gravity["ephemeris"], clock, 0.0, s.duration_s, {v["target_center_id"] for v in sources}
    )
    receipts.append(ephem.receipt)
    targets = sorted(ephem.states)
    table = (ephem.times_tdb_s.tolist(), targets, [ephem.states[t].reshape(-1).tolist() for t in targets])
    native = native_module.AdditionalGravityPlan(
        origin.center_id, clock.initial_tt_s, s.duration_s, table, inputs, frames
    )
    digest = semantic_digest(
        {
            "origin": origin.center_id,
            "epoch_jd_utc": s.initial_jd_utc,
            "duration_s": s.duration_s,
            "ephemeris": ephem.receipt.sha256,
            "sources": sources,
        }
    )
    return PreparedAdditionalGravity(
        origin.center_id, clock, s.duration_s, ephem, sources, native, digest, tuple(receipts)
    )
