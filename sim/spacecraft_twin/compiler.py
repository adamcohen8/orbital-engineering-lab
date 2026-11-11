"""Deterministic compilation of engineering facts into existing OEL actuator contracts."""

from __future__ import annotations

import copy

import numpy as np

from sim.digital_twin.mass_properties import validate_mass_properties

from .contracts import SCHEMA, contained, file_digest, frames, identifier, physical_output_path, positive, vector

ADAPTERS = {
    "tracking_sensor": {
        "model": "knowledge.conditions",
        "required": [
            "max_range_km",
            "fov_half_angle_rad",
            "refresh_rate_s",
            "range_sigma_km",
            "angle_sigma_rad",
            "range_rate_sigma_km_s",
        ],
        "runtime": ["position_relative_to_com", "boresight", "access_limits", "measurement_noise"],
        "metadata_only": [],
    },
    "thruster": {
        "model": "hardware.rcs.v1",
        "required": ["max_thrust_n", "isp_s"],
        "runtime": ["position_relative_to_com", "force_axis", "max_thrust_n", "isp_s"],
        "metadata_only": [],
    },
    "reaction_wheel": {
        "model": "hardware.reaction_wheels.v1",
        "required": ["max_torque_nm", "max_momentum_nms", "inertia_kg_m2", "max_speed_rad_s"],
        "runtime": ["spin_axis", "max_torque_nm", "max_momentum_nms", "inertia_kg_m2", "max_speed_rad_s"],
        "metadata_only": ["position"],
    },
}


def _keys(row, allowed, label):
    extra = set(row) - set(allowed)
    if extra:
        raise ValueError(f"{label}: unknown fields {sorted(extra)}; preserve extra engineering data under metadata.")


def compile_model(model, root):
    """Compile pure data only. Raises for invalid facts; readiness is a separate gate."""
    from .schema import validate

    validate(model)
    _keys(
        model,
        [
            "schema",
            "id",
            "version",
            "mass_kg",
            "center_of_mass_body_m",
            "inertia_kg_m2",
            "frames",
            "components",
            "geometry",
            "sources",
            "evidence",
            "assumptions",
            "conflicts",
            "metadata",
        ],
        "twin",
    )
    if model.get("schema") != SCHEMA:
        raise ValueError(f"Expected {SCHEMA}; use migrate-v0 for existing packages.")
    identifier(model["id"])
    identifier("v" + str(model["version"]))
    mass = positive(model["mass_kg"])
    com = vector(model["center_of_mass_body_m"])
    specs = {
        "mass_kg": mass,
        "mass_properties": {
            "mass_kg": mass,
            "center_of_mass_body_m": com.tolist(),
            "inertia_kg_m2": model["inertia_kg_m2"],
            "frame": "body",
            "inertia_reference_point": "center_of_mass",
            "source": "user_supplied",
            "confidence": "unknown",
        },
    }
    errors = validate_mass_properties(specs).errors
    if errors:
        raise ValueError("; ".join(errors))
    transforms = frames(model)
    placements, required = [], ["mass_kg", "center_of_mass_body_m", "inertia_kg_m2"]
    required += [f"frames/{key}" for key in model.get("frames", {})]
    seen = set()
    components = model.get("components", [])
    if not isinstance(components, list) or len(components) > 256:
        raise ValueError("components must be a list of at most 256 entries.")
    for c in components:
        _keys(
            c,
            ["id", "kind", "frame", "position", "units", "axis", "axis_convention", "parameters", "metadata"],
            "component",
        )
        cid = identifier(c["id"])
        if cid in seen:
            raise ValueError(f"Duplicate component: {cid}")
        seen.add(cid)
        kind = c["kind"]
        if kind not in ADAPTERS:
            raise ValueError(f"Unsupported component adapter: {kind}; no runtime placement claim can be made.")
        r, t = transforms[c["frame"]]
        scale = {"m": 1.0, "cm": 0.01, "mm": 0.001}.get(c["units"])
        if scale is None:
            raise ValueError("Component position units must be m, cm, or mm.")
        position = t + r @ (vector(c["position"]) * scale)
        axis = r @ vector(c["axis"])
        if np.linalg.norm(axis) < 1e-12:
            raise ValueError("Component axis cannot be zero.")
        axis /= np.linalg.norm(axis)
        convention = c["axis_convention"]
        if kind == "thruster":
            if convention not in {"force", "plume"}:
                raise ValueError("Thruster axis_convention must be force or plume.")
            if convention == "plume":
                axis = -axis
        elif kind == "reaction_wheel" and convention != "positive_spin":
            raise ValueError("Reaction wheel axis_convention must be positive_spin.")
        elif kind == "tracking_sensor" and convention != "boresight":
            raise ValueError("Tracking sensor axis_convention must be boresight.")
        params = c["parameters"]
        _keys(params, ADAPTERS[kind]["required"], f"{cid} parameters")
        params = {key: positive(params[key]) for key in ADAPTERS[kind]["required"]}
        if kind == "tracking_sensor" and params["fov_half_angle_rad"] > np.pi:
            raise ValueError("Sensor FOV half angle cannot exceed pi.")
        required += [
            f"components/{cid}/{key}" for key in ("kind", "frame", "position", "units", "axis", "axis_convention")
        ]
        required += [f"components/{cid}/parameters/{key}" for key in params]
        placements.append(
            {
                "id": cid,
                "kind": kind,
                "position_body_m": position.tolist(),
                "axis_body": axis.tolist(),
                "axis_convention": "force" if kind == "thruster" else convention,
                "runtime_metadata_only": ADAPTERS[kind]["metadata_only"],
                "parameters": params,
            }
        )
    geometry = model.get("geometry")
    if geometry:
        _keys(
            geometry,
            ["profile_path", "sha256", "configuration", "qualification_path", "qualification_sha256", "cd", "cr"],
            "geometry",
        )
        from sim.dynamics.spacecraft_geometry import GeometryAreaProfile

        path = contained(root, geometry["profile_path"])
        if file_digest(path) != geometry["sha256"]:
            raise ValueError("Geometry profile identity changed; regenerate and review.")
        profile = GeometryAreaProfile.load(physical_output_path(path))
        if profile.metadata.get("configuration") != geometry["configuration"]:
            raise ValueError("Geometry configuration does not match the selected profile.")
        specs["geometry"] = {"profile_path": str(path.resolve())}
        specs["aero"] = {"cd": positive(geometry["cd"])}
        specs["cr"] = positive(geometry["cr"])
        required += ["geometry/configuration", "geometry/cd", "geometry/cr"]
    from .runtime_binding import hardware_parameters

    return {
        "hardware_parameters": hardware_parameters(placements, com.tolist()),
        "object": {"kind": "satellite", "enabled": True, "role": "agent", "specs": specs},
        "placements": placements,
        "required_evidence": required,
    }


def assess(model, root, purpose="attitude"):
    if purpose not in {"passive", "attitude", "environment"}:
        raise ValueError("purpose must be passive, attitude, or environment.")
    from .schema import validate

    try:
        validate(model)
    except ValueError as exc:
        return {
            "schema": "oel.twin_assessment.v1",
            "purpose": purpose,
            "package_valid": False,
            "runtime_compiled": False,
            "ready": False,
            "errors": [str(exc)],
            "missing": [],
            "warnings": [],
            "compiled": None,
            "claim": "Invalid engineering input.",
        }
    errors, missing, warnings = [], [], []
    result = None
    try:
        result = compile_model(model, root)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        errors.append(str(exc))
    sources = {}
    for source in model.get("sources", []):
        try:
            sid = identifier(source["id"])
            if sid in sources:
                raise ValueError(f"Duplicate source ID: {sid}")
            path = contained(root, source["path"])
            if file_digest(path) != source["sha256"]:
                raise ValueError(f"Source changed: {sid}")
            if not source.get("revision") or source["revision"] == "unreviewed":
                missing.append(f"source revision: {sid}")
            sources[sid] = source
        except (ValueError, KeyError, OSError) as exc:
            errors.append(str(exc))
    evidence = model.get("evidence", {})
    if result:
        for key in result["required_evidence"]:
            refs = evidence.get(key, [])
            if not isinstance(refs, list) or not refs:
                missing.append(f"evidence: {key}")
                continue
            for ref in refs:
                if ref.get("source_id") not in sources or not str(ref.get("locator", "")).strip():
                    errors.append(f"Invalid source reference: {key}")
                if ref.get("basis") not in {"extracted", "derived", "assumed"}:
                    errors.append(f"Evidence basis must be extracted, derived, or assumed: {key}")
                if ref.get("basis") in {"derived", "assumed"} and not ref.get("explanation"):
                    errors.append(f"Explain the transformation/assumption: {key}")
                if ref.get("basis") == "assumed" and not any(
                    a.get("field") == key and a.get("accepted") is True and a.get("reason")
                    for a in model.get("assumptions", [])
                ):
                    missing.append(f"unaccepted assumption: {key}")
    for conflict in model.get("conflicts", []):
        if not conflict.get("resolution") or not conflict.get("reason"):
            missing.append(f"unresolved conflict: {conflict.get('field', 'unknown')}")
    for assumption in model.get("assumptions", []):
        if assumption.get("accepted") is not True or not assumption.get("reason"):
            missing.append(f"unaccepted assumption: {assumption.get('field', 'unknown')}")
    kinds = {c.get("kind") for c in model.get("components", [])}
    if purpose == "attitude" and not kinds.intersection({"thruster", "reaction_wheel"}):
        missing.append("attitude study requires a supported actuator")
    if purpose == "environment":
        geometry = model.get("geometry", {})
        if not geometry.get("qualification_path"):
            missing.append("environment study requires qualified geometry")
        else:
            try:
                from .contracts import load

                path = contained(root, geometry["qualification_path"])
                if file_digest(path) != geometry["qualification_sha256"]:
                    raise ValueError("Geometry qualification changed.")
                report = load(path)
                from .qualification import verify_geometry_qualification

                verify_geometry_qualification(model, root, report, sources)
            except (ValueError, KeyError, TypeError, OSError) as exc:
                errors.append(str(exc))
    if model.get("assumptions"):
        warnings.append("Readiness includes explicitly accepted assumptions; it is not empirical accuracy evidence.")
    return {
        "schema": "oel.twin_assessment.v1",
        "purpose": purpose,
        "package_valid": not errors,
        "runtime_compiled": result is not None,
        "ready": not errors and not missing,
        "errors": errors,
        "missing": sorted(set(missing)),
        "warnings": warnings,
        "compiled": copy.deepcopy(result),
        "claim": "Engineering consistency and study-specific completeness, not flight qualification.",
    }
