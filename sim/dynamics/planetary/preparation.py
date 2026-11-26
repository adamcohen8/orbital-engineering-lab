"""Bind a non-Earth environment and its resources before propagation."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass

from .atmosphere import prepare_atmosphere
from .definitions import body_definition
from .ephemerides import prepare_ephemeris
from .frames import prepare_rotation
from .gravity import prepare_gravity
from .resources import DATA_ROOT, ResourceReceipt, read_json, semantic_digest


@dataclass(frozen=True)
class PreparedPlanetaryEnvironment:
    body: object
    gravity: object
    gravity_frame: object
    atmosphere: object
    atmosphere_frame: object
    ephemeris: object | None
    selections: dict
    initial_jd_utc: float
    duration_s: float
    digest: str
    constants_receipts: tuple
    additional_gravity: object | None = None

    def sample_atmosphere(self, t_s, r):
        R, D = self.atmosphere_frame.evaluate(t_s)
        return self.atmosphere.sample(R @ r, t_s=t_s), R, D

    def verify_resources(self):
        if self.additional_gravity is not None:
            self.additional_gravity.verify_resources()
        receipts = list(self.gravity.receipts)
        receipts.extend(self.constants_receipts)
        receipts += [
            frame.receipt
            for frame in (self.gravity_frame, self.atmosphere_frame)
            if frame.receipt is not None
        ]
        receipts.extend(self.atmosphere.source_receipts)
        if self.atmosphere.receipt is not None:
            receipts.append(self.atmosphere.receipt)
        if self.ephemeris is not None:
            receipts.append(self.ephemeris.receipt)
        for receipt in receipts:
            receipt.verify()

    def metadata(self):
        result = {
            "central_body": self.body.body_id,
            "origin_center_id": self.body.center_id,
            "inertial_axes": "J2000",
            "initial_jd_utc": self.initial_jd_utc,
            "elapsed_time_scale": "TAI_seconds",
            "environment_sha256": self.digest,
            "body": asdict(self.body),
            "constants_manifest_sha256": self.constants_receipts[0].sha256,
            "gravity": self.gravity.metadata(),
            "gravity_frame": self.gravity_frame.metadata(),
            "atmosphere_frame": self.atmosphere_frame.metadata(),
            "atmosphere": self.atmosphere.metadata(),
            "forces": deepcopy(self.selections),
            "ephemeris": (
                None
                if self.ephemeris is None
                else {
                    "source": self.ephemeris.source,
                    "observer_center_id": self.ephemeris.observer_center_id,
                    "sha256": self.ephemeris.receipt.sha256,
                    "inertial_axes": "J2000",
                    "aberration": "NONE",
                    "time_scale": "TDB",
                    "interpolation": "cubic_hermite_position_and_velocity",
                    "coverage_tdb_seconds": [
                        float(self.ephemeris.times_tdb_s[0]),
                        float(self.ephemeris.times_tdb_s[-1]),
                    ],
                    "target_center_ids": sorted(self.ephemeris.states),
                }
            ),
        }
        if self.additional_gravity is not None:
            result["additional_gravity"] = self.additional_gravity.metadata()
        return result


def prepare_environment(cfg):
    s = cfg.simulator
    p = deepcopy(s.planetary)
    if s.dynamics.get("orbit", {}).get("schwarzschild", {}).get("enabled"):
        p["schwarzschild"] = {"enabled": True}
    body = body_definition(s.central_body)
    manifest, constants = read_json(DATA_ROOT / "body_constants_manifest.json")
    constants_receipts = (manifest,) + tuple(
        ResourceReceipt.read(DATA_ROOT / resource["file"], resource["sha256"])[0]
        for resource in constants["resources"]
    )
    gravity = prepare_gravity(body, p["gravity"])
    gf = prepare_rotation(body.body_id, p["gravity_frame"], s.initial_jd_utc, duration_s=s.duration_s)
    af = prepare_rotation(body.body_id, p["atmosphere_frame"], s.initial_jd_utc, duration_s=s.duration_s)
    if gravity.compiled is not None and gf.frame_id != gravity.frame_id:
        raise ValueError(
            f"Gravity frame mismatch: coefficients require {gravity.frame_id}, selected {gf.frame_id}"
        )
    if "magnetic_field" in p and p["magnetic_field"]["frame_id"] != gf.frame_id:
        raise ValueError(f"Planetary magnetic_field.frame_id must match the selected gravity frame {gf.frame_id}")
    atmosphere = prepare_atmosphere(body, p["atmosphere"], af.frame_id)
    targets = {x["target_center_id"] for x in p["third_bodies"]}
    if p["srp"]["enabled"] and body.body_id != "sun":
        targets.add(10)
    ephem = prepare_ephemeris(
        body, p["ephemeris"], gf, 0.0, float(s.duration_s), targets
    )
    gf.evaluate(float(s.duration_s))
    af.evaluate(float(s.duration_s))
    raw = {
        "constants_manifest_sha256": ResourceReceipt.read(
            DATA_ROOT / "body_constants_manifest.json"
        )[0].sha256,
        "body": asdict(body),
        "gravity": gravity.metadata(),
        "gravity_frame": gf.metadata(),
        "atmosphere": atmosphere.metadata(),
        "atmosphere_frame": af.metadata(),
        "selections": p,
        "ephemeris": None if ephem is None else ephem.receipt.sha256,
        "initial_jd_utc": s.initial_jd_utc,
        "duration_s": s.duration_s,
    }
    additional = None
    if s.additional_gravity is not None:
        from .additional_gravity import prepare_additional_gravity
        additional = prepare_additional_gravity(cfg)
        raw["additional_gravity"] = additional.metadata()
    return PreparedPlanetaryEnvironment(
        body,
        gravity,
        gf,
        atmosphere,
        af,
        ephem,
        p,
        s.initial_jd_utc,
        s.duration_s,
        semantic_digest(raw),
        constants_receipts,
        additional,
    )


def planetary_model_catalog():
    """Pre-authoring settings for bundled, static-gravity planetary ONP profiles."""
    from pathlib import Path

    from sim.config.scenario.planetary import normalize_planetary

    from .definitions import BODIES
    from .frames import rotation_resource
    from .native import module
    from .resources import tdb_interval

    profiles = []
    for body in BODIES.values():
        models = ("point_mass", "gmm2b_static") if body.body_id == "mars" else ("point_mass",)
        for model in models:
            selection = normalize_planetary({"gravity": {"model": model}}, body.body_id)
            gravity = prepare_gravity(body, selection["gravity"])
            frames = {}
            for role in ("gravity_frame", "atmosphere_frame"):
                receipt, definition = rotation_resource(body.body_id, selection[role])
                name = Path(receipt.path).name
                frames[name] = definition["valid_tdb_seconds"]
            low = max(bounds[0] for bounds in frames.values())
            high = min(bounds[1] for bounds in frames.values())
            profiles.append({
                "central_body": body.body_id, "gravity_model": model,
                "mu_km3_s2": gravity.mu_km3_s2, "rotation_datasets": list(frames),
                "valid_tdb_seconds_past_j2000": [low, high],
                "valid_calendar_tdb": tdb_interval(low, high),
            })
    return {
        "schema": "oel.planetary_model_catalog.v1",
        "scope": "Bundled gravity and default rotation datasets. Custom frames, gravity and ephemerides require their own validation.",
        "gravity_model_path": "simulator.planetary.gravity.model",
        "schwarzschild": {
            "setting_path": "simulator.dynamics.orbit.schwarzschild.enabled",
            "native_supported": bool(getattr(module().PlanetaryPlan, "supports_schwarzschild", False)),
            "default": False,
        },
        "passive_attitude": {
            "setting_path": "simulator.dynamics.attitude.enabled",
            "disturbance_torques": ["gravity_gradient", "drag", "srp", "magnetic"],
            "inertia_path": "objects.<id>.specs.mass_properties.inertia_kg_m2",
            "magnetic_field": "explicit centered dipole in the selected primary gravity frame",
            "pressure_torques": "selected planetary force and body-frame center-of-pressure offset from mass center",
        },
        "profiles": profiles,
    }
