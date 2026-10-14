"""Declarative non-Earth configuration and explicit compatibility boundary."""

from __future__ import annotations

from copy import deepcopy

import numpy as np

from .primitives import _as_dict, _parse_bool, _parse_float, _reject_unknown_fields


def apply_planetary_defaults(root):
    root = deepcopy(root)
    objects = root.get("objects", {})
    if isinstance(objects, dict):
        for obj in objects.values():
            if isinstance(obj, dict):
                obj.setdefault("runtime_profile", "trajectory_only")
    outputs = root.setdefault("outputs", {})
    if isinstance(outputs, dict):
        for name, values in (
            ("plots", {"enabled": False}),
            ("animations", {"enabled": False}),
            ("review", {"enabled": True, "detail": "standard"}),
            ("stats", {"save_full_log": False}),
        ):
            section = outputs.setdefault(name, {})
            if isinstance(section, dict):
                for key, value in values.items():
                    section.setdefault(key, value)
    return root


def normalize_planetary(raw, body_id):
    from sim.dynamics.planetary.definitions import body_definition

    body_definition(body_id)
    p = deepcopy(_as_dict(raw, "simulator.planetary"))
    path = "simulator.planetary"
    _reject_unknown_fields(
        p,
        path,
        {
            "inertial_frame",
            "gravity",
            "gravity_frame",
            "atmosphere",
            "atmosphere_frame",
            "drag",
            "srp",
            "third_bodies",
            "ephemeris",
            "impact",
        },
    )
    p.setdefault("inertial_frame", "J2000")
    if p["inertial_frame"] != "J2000":
        raise ValueError(f"{path}.inertial_frame must be J2000")
    fields = {
        "gravity": {
            "model",
            "max_degree",
            "max_order",
            "coeff_path",
            "label_path",
            "frame_id",
            "sha256",
        },
        "gravity_frame": {"model", "data_path", "sha256"},
        "atmosphere_frame": {"model", "data_path", "sha256"},
        "atmosphere": {
            "model",
            "data_path",
            "sha256",
            "out_of_domain",
            "altitude_datum",
            "altitude_domain_km",
            "reference_density_kg_m3",
            "reference_altitude_km",
            "scale_height_km",
            "wind_body_m_s",
        },
        "drag": {"enabled"},
        "srp": {"enabled", "shadow_model"},
        "ephemeris": {"model", "data_path", "sha256"},
        "impact": {"enabled", "radius_km"},
    }
    for key, allowed in fields.items():
        d = _as_dict(p.get(key), f"{path}.{key}")
        _reject_unknown_fields(d, f"{path}.{key}", allowed)
        p[key] = d
    gravity = p["gravity"]
    gravity.setdefault("model", "point_mass")
    if gravity["model"] not in {
        "point_mass",
        "gmm2b_static",
        "gmm3_static",
        "pds_sha_static",
    }:
        raise ValueError(f"{path}.gravity.model unsupported")
    if gravity["model"] == "gmm3_static":
        raise ValueError(
            "GMM-3 is unavailable until its exact Konopliv rotation model is verified; use gmm2b_static with IAU1991_MARS"
        )
    if gravity["model"] == "gmm2b_static" and body_id != "mars":
        raise ValueError("gmm2b_static requires central_body=mars")
    unused_gravity = set(gravity) - {"model"}
    if gravity["model"] == "point_mass" and unused_gravity:
        raise ValueError("Point-mass gravity cannot select harmonic dataset options")
    if gravity["model"] == "gmm2b_static" and set(gravity) & {
        "coeff_path",
        "label_path",
        "frame_id",
        "sha256",
    }:
        raise ValueError(
            "Bundled GMM-2B uses its pinned dataset and frame; use pds_sha_static for custom data"
        )
    if gravity["model"] == "gmm3_static" and body_id != "mars":
        raise ValueError("gmm3_static requires central_body=mars")
    if gravity["model"] != "point_mass":
        for key in ("max_degree", "max_order"):
            value = gravity.get(key, gravity.get("max_degree", 20))
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{path}.gravity.{key} must be an integer")
            gravity[key] = value
        if (
            not 2 <= gravity["max_degree"] <= 20
            or not 0 <= gravity["max_order"] <= gravity["max_degree"]
        ):
            raise ValueError(
                "Planetary harmonics currently qualify degree 2..20 and order <= degree"
            )
        if gravity["model"] == "pds_sha_static":
            for key in ("coeff_path", "label_path", "frame_id"):
                if not isinstance(gravity.get(key), str) or not gravity[key]:
                    raise ValueError(f"{path}.gravity.{key} is required")
    for name in ("gravity_frame", "atmosphere_frame"):
        p[name].setdefault(
            "model",
            (
                "iau1991"
                if name == "gravity_frame" and gravity["model"] == "gmm2b_static"
                else "iau2000"
            ),
        )
        if p[name]["model"] not in {"iau2000", "iau1991", "resource"}:
            raise ValueError(f"{path}.{name}.model unsupported")
        if p[name]["model"] == "iau1991" and body_id != "mars":
            raise ValueError("The bundled IAU 1991 rotation is Mars-only")
        if p[name]["model"] == "resource" and not p[name].get("data_path"):
            raise ValueError(f"{path}.{name}.data_path is required")
        if p[name]["model"] != "resource" and set(p[name]) & {"data_path", "sha256"}:
            raise ValueError(f"{path}.{name} custom data requires model=resource")
    atm = p["atmosphere"]
    atm.setdefault("model", "none")
    atm.setdefault("out_of_domain", "error")
    if atm["model"] not in {"none", "exponential", "table", "mars_reference_table"}:
        raise ValueError(f"{path}.atmosphere.model unsupported")
    if atm["out_of_domain"] not in {"error", "vacuum_above"}:
        raise ValueError(f"{path}.atmosphere.out_of_domain unsupported")
    from sim.dynamics.planetary.atmosphere import validate_atmosphere_options

    validate_atmosphere_options(atm)
    if atm["model"] == "table" and not atm.get("data_path"):
        raise ValueError(f"{path}.atmosphere.data_path required")
    if atm["model"] == "exponential":
        for key in (
            "reference_density_kg_m3",
            "scale_height_km",
            "reference_altitude_km",
        ):
            atm[key] = _parse_float(atm.get(key), f"{path}.atmosphere.{key}")
        if atm["reference_density_kg_m3"] <= 0 or atm["scale_height_km"] <= 0:
            raise ValueError("Exponential atmosphere parameters must be positive")
        if (
            not isinstance(atm.get("altitude_domain_km"), list)
            or len(atm["altitude_domain_km"]) != 2
        ):
            raise ValueError("Exponential atmosphere requires altitude_domain_km")
        atm["altitude_domain_km"] = [
            _parse_float(v, "altitude_domain_km") for v in atm["altitude_domain_km"]
        ]
        if atm["altitude_domain_km"][0] >= atm["altitude_domain_km"][1]:
            raise ValueError("Atmosphere altitude domain must increase")
        if not isinstance(atm.get("altitude_datum"), dict):
            raise ValueError("Exponential atmosphere requires altitude_datum")
    for key in ("drag", "srp", "impact"):
        p[key]["enabled"] = _parse_bool(
            p[key].get(
                "enabled",
                key == "impact"
                and body_id not in {"sun", "jupiter", "saturn", "uranus", "neptune"},
            ),
            f"{path}.{key}.enabled",
        )
    if p["drag"]["enabled"] and atm["model"] == "none":
        raise ValueError("Planetary drag requires a selected atmosphere")
    if body_id == "sun" and (
        p["drag"]["enabled"]
        or atm["model"] != "none"
        or gravity["model"] != "point_mass"
    ):
        raise ValueError(
            "Sun-centered ONP supports point mass, SRP and external perturbers; neutral drag/solar harmonics unsupported"
        )
    p["srp"].setdefault("shadow_model", "none")
    if p["srp"]["shadow_model"] not in {"none", "cylindrical"}:
        raise ValueError("Planetary SRP shadow_model must be none or cylindrical")
    if body_id == "sun" and p["srp"]["shadow_model"] != "none":
        raise ValueError("The emitting Sun cannot eclipse itself")
    if "radius_km" in p["impact"]:
        p["impact"]["radius_km"] = _parse_float(
            p["impact"]["radius_km"], f"{path}.impact.radius_km"
        )
        if p["impact"]["radius_km"] <= 0:
            raise ValueError("Impact radius must be positive")
    bodies = p.setdefault("third_bodies", [])
    if not isinstance(bodies, list) or any(not isinstance(v, dict) for v in bodies):
        raise ValueError("Planetary third_bodies must be a list of mappings")
    targets = set()
    for i, d in enumerate(bodies):
        _reject_unknown_fields(
            d, f"{path}.third_bodies[{i}]", {"target_center_id", "mu_km3_s2", "name"}
        )
        target = d.get("target_center_id")
        if (
            isinstance(target, bool)
            or not isinstance(target, int)
            or target in targets
            or target == body_definition(body_id).center_id
        ):
            raise ValueError(
                "Planetary third bodies require unique external integer target_center_id"
            )
        targets.add(target)
        d["mu_km3_s2"] = _parse_float(
            d.get("mu_km3_s2"), f"{path}.third_bodies[{i}].mu_km3_s2"
        )
        if d["mu_km3_s2"] <= 0:
            raise ValueError("Third body GM must be positive")
    needs_ephem = bool(bodies) or (p["srp"]["enabled"] and body_id != "sun")
    if needs_ephem and not p["ephemeris"].get("data_path"):
        raise ValueError(
            "Planetary forces require an explicit geometric ephemeris table"
        )
    return p


def validate_planetary_scenario(cfg):
    """Reject components whose complete origin/frame contract is not audited."""
    s = cfg.simulator
    d = dict(s.dynamics)
    orbit = dict(d.get("orbit", {}))
    shared = {
        "model",
        "propagation_method",
        "integrator",
        "numeric_backend",
        "adaptive_atol",
        "adaptive_rtol",
        "orbit_substep_s",
    }
    extras = set(orbit) - shared
    if extras:
        raise ValueError(
            f"Planetary force selections belong in simulator.planetary; unsupported dynamics.orbit: {', '.join(sorted(extras))}"
        )
    if (
        orbit.get("model", "two_body") != "two_body"
        or orbit.get("propagation_method", "special") != "special"
    ):
        raise ValueError("Planetary scenarios require special ONP two_body dynamics")
    if orbit.get("integrator", "rk4") not in {"rk4", "rkf78", "dopri5", "adaptive"}:
        raise ValueError("Unsupported planetary integrator")
    if orbit.get("numeric_backend", "rust") != "rust":
        raise ValueError("Planetary ONP requires numeric_backend=rust")
    for key in ("adaptive_atol", "adaptive_rtol", "orbit_substep_s"):
        if key in orbit and orbit[key] is not None:
            value = _parse_float(orbit[key], f"simulator.dynamics.orbit.{key}")
            if value <= 0:
                raise ValueError(f"Planetary {key} must be positive")
    if bool(d.get("attitude", {}).get("enabled", False)):
        raise ValueError(
            "Planetary attitude coupling is not qualified; use attitude.enabled=false"
        )
    if any(key in d and d[key] for key in ("rocket", "reentry")):
        raise ValueError("Planetary rockets and reentry diagnostics are unsupported")
    if s.initial_jd_utc is None:
        raise ValueError("Planetary scenarios require simulator.initial_jd_utc")
    if s.environment:
        raise ValueError("Planetary environment belongs in simulator.planetary")
    if s.collisions or s.system_force_models:
        raise ValueError("Planetary collision/system-force plugins are not qualified")
    execution = dict(s.execution)
    objects_execution = execution.get("object_parallelism", {})
    if (
        objects_execution.get("enabled", False)
        or objects_execution.get("backend", "serial") != "serial"
    ):
        raise ValueError("Planetary scenarios require serial execution")
    if cfg.ground_stations or cfg.ground_segment.get("enabled", False):
        raise ValueError("Planetary ground access/ground segment is unsupported")
    if cfg.analysis.enabled or cfg.monte_carlo.enabled:
        raise ValueError("Planetary covariance/campaign workflows are not qualified")
    if cfg.outputs.orbital_analysis.enabled:
        raise ValueError("Earth coverage/link analysis cannot consume planetary runs")
    if cfg.outputs.ai_report.get("enabled", False) or cfg.outputs.ai_config.get(
        "enabled", False
    ):
        raise ValueError("Planetary AI output workflows are not qualified")
    if cfg.outputs.plots.enabled or cfg.outputs.animations.enabled:
        raise ValueError(
            "Planetary output plots/animations require planetary review recipes; Earth output recipes are unsupported"
        )
    for oid, obj in cfg.objects.items():
        if not obj.enabled:
            continue
        if obj.kind != "satellite" or obj.runtime_profile != "trajectory_only":
            raise ValueError(
                f"Planetary {oid} requires satellite runtime_profile=trajectory_only"
            )
        if obj.propagation_method not in {"", "special"} or obj.general:
            raise ValueError(
                f"Planetary objects.{oid} cannot select Earth general/TLE propagation"
            )
        for field in (
            "force_models",
            "flight_software",
            "knowledge",
            "orbit_control",
            "attitude_control",
            "mission_strategy",
            "mission_execution",
            "bridge",
            "reference_orbit",
            "guidance",
            "base_guidance",
            "guidance_modifiers",
            "mission_objectives",
        ):
            value = getattr(obj, field, None)
            if value:
                raise ValueError(f"Planetary objects.{oid}.{field} is unsupported")
        initial = obj.initial_state
        forms = set(initial) & {"position_inertial_km", "coes"}
        if len(forms) != 1:
            raise ValueError(
                f"Planetary objects.{oid} requires explicit inertial Cartesian state or COEs"
            )
        if "coes" in initial and (
            initial.get("reference_axes") != "J2000"
            or not ({"a_km", "semi_major_axis_km"} & set(initial["coes"]))
        ):
            raise ValueError(
                f"Planetary objects.{oid} COEs require reference_axes=J2000 and explicit semimajor axis"
            )
        specs = obj.specs
        if {"dry_mass_kg", "fuel_mass_kg"} & specs.keys():
            raise ValueError(
                f"Planetary objects.{oid} requires fixed total mass_kg; dry_mass_kg/fuel_mass_kg are unsupported"
            )
        for key in (
            "spacecraft_resources",
            "resources",
            "geometry",
            "geometry_profile_path",
            "mass_properties",
            "actuators",
            "thrusters",
            "thermal",
            "power",
            "specular",
            "radiation",
        ):
            if specs.get(key):
                raise ValueError(f"Planetary objects.{oid}.specs.{key} is unsupported")
        mass = _parse_float(specs.get("mass_kg"), f"objects.{oid}.specs.mass_kg")
        if mass <= 0:
            raise ValueError("Planetary mass_kg must be positive")
        if s.planetary["drag"]["enabled"]:
            for key in ("cd", "drag_area_m2"):
                value = _parse_float(specs.get(key), f"objects.{oid}.specs.{key}")
                if value <= 0:
                    raise ValueError(f"Planetary drag {key} must be positive")
        if s.planetary["srp"]["enabled"]:
            for key in ("cr", "srp_area_m2"):
                value = _parse_float(specs.get(key), f"objects.{oid}.specs.{key}")
                if value <= 0:
                    raise ValueError(f"Planetary SRP {key} must be positive")
    # Prepare declarative resources once during ordinary validation; no plugins.
    from sim.dynamics.planetary.preparation import prepare_environment

    environment = prepare_environment(cfg)
    from sim.dynamics.planetary.propagation import PlanetaryForcePlan
    from sim.dynamics.planetary.runtime import initial_truth

    for obj in cfg.objects.values():
        if obj.enabled:
            truth = initial_truth(obj, environment)
            plan = PlanetaryForcePlan(
                environment,
                mass_kg=obj.specs["mass_kg"],
                cd=obj.specs.get("cd", 0.0),
                drag_area_m2=obj.specs.get("drag_area_m2", 0.0),
                cr=obj.specs.get("cr", 0.0),
                srp_area_m2=obj.specs.get("srp_area_m2", 0.0),
            )
            plan.stage(0.0, np.r_[truth.position_eci_km, truth.velocity_eci_km_s])
