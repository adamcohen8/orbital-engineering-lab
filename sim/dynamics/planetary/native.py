"""Declarative preparation for the source-public native planetary plan."""

from __future__ import annotations

import numpy as np


def module():
    try:
        import oel_rust_orbit as native
    except ImportError as exc:
        raise ValueError(
            "Planetary ONP requires the oel-rust-orbit native extension"
        ) from exc
    required = (
        "PlanetaryPlan",
        "PlanetaryGravityDomainError",
        "planetary_rotation",
        "planetary_atmosphere_sample",
        "planetary_tdb_seconds",
    )
    if any(not callable(getattr(native, name, None)) for name in required):
        raise ValueError(
            "Planetary ONP requires an updated oel-rust-orbit build with PlanetaryPlan; rebuild rust/oel-orbit"
        )
    return native


def atmosphere_input(atmosphere):
    d = atmosphere.definition
    datum = d["altitude_datum"]
    datum_kinds = {"radial_sphere": 0, "radial_ellipsoid": 1, "geodetic_ellipsoid": 2}
    try:
        kind = datum_kinds[datum["kind"]]
    except (KeyError, TypeError) as exc:
        raise ValueError("Unsupported planetary atmosphere altitude datum") from exc
    a = datum["radius_km"] if kind == 0 else datum["equatorial_radius_km"]
    b = a if kind == 0 else datum["polar_radius_km"]
    model = (
        0
        if atmosphere.model_id == "none"
        else 1 if atmosphere.model_id == "exponential" else 2
    )
    p = [
        a,
        b,
        *d["altitude_domain_km"],
        d.get("reference_density_kg_m3", 0.0),
        d.get("reference_altitude_km", 0.0),
        d.get("scale_height_km", 1.0),
    ]
    p += d.get("wind_body_m_s", [0.0, 0.0, 0.0]) if model == 1 else [0.0, 0.0, 0.0]
    h = [] if atmosphere.heights is None else atmosphere.heights.tolist()
    rho = [] if atmosphere.densities is None else np.log(atmosphere.densities).tolist()
    wind = [] if atmosphere.winds is None else atmosphere.winds.reshape(-1).tolist()
    return model, kind, p, h, rho, wind, atmosphere.policy == "vacuum_above"


def prepare_native_plan(plan):
    e = plan.environment
    c = e.gravity.compiled
    dims = (0, 0) if c is None else (c.n_max, c.m_max)
    coefficients = (
        ([1.0], [0.0])
        if c is None
        else (c.c_nm.reshape(-1).tolist(), c.s_nm.reshape(-1).tolist())
    )
    srp = e.selections["srp"]
    scalars = [
        e.gravity.mu_km3_s2,
        e.gravity.radius_km,
        plan.mass_kg,
        plan.cd * plan.drag_area_m2,
        plan.cr * plan.srp_area_m2,
        e.body.reference_radius_km,
        e.gravity_frame.initial_tt_s,
        e.duration_s,
        float(e.body.body_id == "sun"),
        float(e.selections["drag"]["enabled"]),
        float(srp["enabled"]),
        float(srp["shadow_model"] == "cylindrical"),
    ]
    ephem = e.ephemeris
    targets = [] if ephem is None else sorted(ephem.states)
    table = (
        ([], [], [])
        if ephem is None
        else (
            ephem.times_tdb_s.tolist(),
            targets,
            [ephem.states[target].reshape(-1).tolist() for target in targets],
        )
    )
    bodies = e.selections["third_bodies"]
    base = module().PlanetaryPlan(
        e.body.center_id,
        scalars,
        dims,
        coefficients,
        e.gravity_frame.native_input(),
        e.atmosphere_frame.native_input(),
        atmosphere_input(e.atmosphere),
        table,
        (
            [body["target_center_id"] for body in bodies],
            [body["mu_km3_s2"] for body in bodies],
        ),
    )
    if e.additional_gravity is not None:
        return _ExtendedPlanetaryPlan(base, e.additional_gravity)
    return base


class _ExtendedPlanetaryPlan:
    def __init__(self, base, additional):
        self.base, self.additional = base, additional

    def stage(self, t, x):
        self.additional.verify_resources()
        return self.additional.native.stage_planetary(self.base, t, x)

    def propagate(self, x, t, dt, method, atol, rtol):
        self.additional.verify_resources()
        return self.additional.native.propagate_planetary(self.base, x, t, dt, method, atol, rtol)
