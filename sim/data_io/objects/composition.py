"""Compose imported states through OEL scenario contracts; never propagate."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np

from sim.data_io.catalog import CatalogRecord
from sim.data_io.common import publish
from sim.data_io.contracts import AttitudeHistory, DataIOError
from sim.data_io.promotion import require_seed_eligible
from sim.data_io.properties import specs_at_sample
from sim.frame_time import epoch_julian_date
from sim.scenarios import ScenarioArtifact

from .contracts import ImportedObject


def orbital_entry(imported: ImportedObject):
    """One authoritative mapping from normalized data to an object seed."""
    imported.verify_sources()
    value, index = imported.value, imported.sample_index
    if isinstance(value, AttitudeHistory):
        raise DataIOError("Attitude requires an existing orbital object; use compose_object to attach it.")
    if isinstance(value, CatalogRecord):
        from dataclasses import asdict

        elements = {key: item for key, item in asdict(value.elements).items() if key not in {"line1", "line2"}}
        return {
            "kind": "satellite",
            "role": "catalog_object",
            "propagation_method": "general",
            "general": {"model": "sgp4", "output_frame": "teme", "frame_transform": "native"},
            "specs": {"mass_kg": 100.0},
            "initial_state": {"ogp_mean_elements": elements},
        }, {}
    decision = require_seed_eligible(value)
    if value.metadata.get("oel_product_binding"):
        if (
            value.center.upper() != "EARTH"
            or value.frame != "EME2000"
            or value.metadata.get("frame_conversion")
            or value.metadata.get("spice_transform")
        ):
            raise DataIOError(
                "OEL state-product seeding must retain its declared frame/center; use a compatible dedicated handoff for transformed products."
            )
        if index != 0:
            raise DataIOError("An OEL state product contains one selected state.")
        product = value.metadata["oel_product_binding"]["product"]
        original = product["payload"]["state"]
        if (
            value.object_id != product["payload"]["object"]["object_id"]
            or not np.allclose(value.states[index], original["values"], atol=1e-10, rtol=0)
            or abs(epoch_julian_date(imported.epoch, "UTC") - original["epoch"]["value"]) * 86400 > 5e-5
        ):
            raise DataIOError(
                "The selected state differs from its OEL product binding; use a new explicit product/handoff."
            )
        from sim.interchange.materialization import build_onp_scenario

        doc = build_onp_scenario(
            product,
            scenario_name="object_seed",
            output_dir="outputs/object_seed",
            duration_s=1,
            dt_s=1,
            manifest_path=Path("object_seed.receipt.json"),
        )
        return next(iter(doc["objects"].values())), doc["simulator"]
    specs = decision.get("object_specs") or specs_at_sample(value, index)
    if any(specs.get(key, {}).get("enabled", False) for key in ("thermal", "power")):
        raise DataIOError("Resource-state continuation requires the existing satellite checkpoint workflow.")
    pv = value.states[index]
    if value.center.upper() == "EARTH":
        if value.frame != "EME2000":
            raise DataIOError("Earth object seeds require EME2000; request an explicit target-frame conversion.")
        state = {"position_eci_km": pv[:3].tolist(), "velocity_eci_km_s": pv[3:].tolist()}
        simulator = deepcopy(value.metadata.get("seed_simulator", {}))
        entry = {"kind": "satellite", "role": "target", "specs": specs, "initial_state": state}
    else:
        from sim.dynamics.planetary.definitions import body_definition

        body = body_definition(value.center.lower())
        if value.frame != "SPICE:J2000":
            raise DataIOError("Planetary object seeds require explicit body-centered SPICE:J2000 axes.")
        entry = {
            "kind": "satellite",
            "runtime_profile": "trajectory_only",
            "role": "target",
            "specs": specs,
            "initial_state": {"position_inertial_km": pv[:3].tolist(), "velocity_inertial_km_s": pv[3:].tolist()},
        }
        simulator = {
            "central_body": body.body_id,
            "planetary": deepcopy(value.metadata.get("planetary_config", {"gravity": {"model": "point_mass"}})),
        }
    return entry, simulator


def compose_object(scenario, imported: ImportedObject, *, object_id=None, replace=False, allow_model_change=False):
    """Return a new scenario mapping; preserve unrelated objects/features.

    Replacement updates orbital state and supplied physical fields while retaining
    the target's controllers, sensors, attitude and other configuration.
    """
    imported.verify_sources()
    if not isinstance(replace, bool) or not isinstance(allow_model_change, bool):
        raise DataIOError("replace and allow_model_change must be booleans.")
    doc = deepcopy(scenario.to_dict() if hasattr(scenario, "to_dict") else scenario)
    objects = doc.setdefault("objects", {})
    target = object_id or imported.object_id
    if not isinstance(target, str) or not target.strip():
        raise DataIOError("A nonempty target object ID is required.")
    simulator = doc.setdefault("simulator", {})
    jd = epoch_julian_date(imported.epoch, "UTC")
    if simulator.get("initial_jd_utc") is None:
        if objects:
            raise DataIOError("An existing scenario needs an absolute epoch before adding external object data.")
        simulator["initial_jd_utc"] = jd
    if abs(float(simulator["initial_jd_utc"]) - jd) * 86400 > 5e-5:
        raise DataIOError(
            "Imported sample epoch differs from the scenario epoch; select an exact matching sample or create a new scenario."
        )
    if isinstance(imported.value, AttitudeHistory):
        if target not in objects:
            raise DataIOError("Attitude import requires an existing target orbital object.")
        if simulator.get("central_body", "earth") != "earth" or imported.value.reference_frame != "EME2000":
            raise DataIOError("Attitude attachment requires an Earth/EME2000 body-frame history.")
        if objects[target].get("propagation_method") == "general":
            raise DataIOError("OGP does not realize achieved attitude; attach attitude to an ONP object.")
        index, history = imported.sample_index, imported.value
        state = objects[target].setdefault("initial_state", {})
        state.update(
            attitude_quat_bn=history.quaternions[index].tolist(),
            angular_rate_body_rad_s=[0.0] * 3 if history.rates_rad_s is None else history.rates_rad_s[index].tolist(),
        )
        simulator.setdefault("dynamics", {}).setdefault("attitude", {})["enabled"] = True
        disposition = {"attitude": "selected initial sample; history is not replayed", "orbit": "preserved"}
    else:
        entry, context = orbital_entry(imported)
        center = context.get("central_body", "earth")
        current = simulator.get("central_body", "earth")
        if objects and center != current:
            raise DataIOError("Imported center differs from the scenario central body.")
        for key, value in context.items():
            if key == "dynamics":
                required = value.get("orbit", {})
                actual = simulator.get("dynamics", {}).get("orbit", {})
                mismatch = [
                    name
                    for name, expected in required.items()
                    if name != "orbit_substep_s"
                    and actual.get(name, False if isinstance(expected, bool) else None) != expected
                ]
                if objects and mismatch and not allow_model_change:
                    raise DataIOError(
                        "Imported model assumptions differ from the scenario: "
                        + ", ".join(mismatch)
                        + "; explicit allow_model_change is required."
                    )
                if not objects:
                    simulator.setdefault("dynamics", {}).update(deepcopy(value))
            elif key in {"central_body", "planetary", "environment", "frames", "plugin_validation"}:
                default = "earth" if key == "central_body" else {}
                if objects and simulator.get(key, default) != value and not allow_model_change:
                    raise DataIOError(f"Imported {key} assumptions differ; explicit allow_model_change is required.")
                if not objects:
                    simulator[key] = deepcopy(value)
        if target in objects:
            if not replace:
                raise DataIOError("Object ID already exists; explicitly select replace to update its orbit.")
            existing = objects[target]
            state = {
                key: value
                for key, value in existing.get("initial_state", {}).items()
                if key in {"attitude_quat_bn", "angular_rate_body_rad_s"}
            }
            state.update(entry["initial_state"])
            existing["initial_state"] = state
            supplied = dict(entry["specs"])
            source = imported.value
            if isinstance(source, CatalogRecord) or not any(
                source.metadata.get(key) for key in ("physical", "object_specs", "mass_kg_by_sample")
            ):
                supplied.pop("mass_kg", None)
            existing.setdefault("specs", {}).update(supplied)
            existing["propagation_method"] = entry.get("propagation_method", "special")
            existing["general"] = entry.get("general", {})
        else:
            objects[target] = entry
        disposition = {
            "orbit": "native mean elements"
            if isinstance(imported.value, CatalogRecord)
            else "selected Cartesian initial sample",
            "existing_features": "preserved",
            "model_change_explicit": bool(allow_model_change),
            "source_model_context": context,
            "destination_model_context": {key: simulator.get(key) for key in context},
            "omitted": ["history replay", "maneuver execution", "covariance assimilation", "runtime restoration"],
        }
    doc.setdefault("metadata", {}).setdefault("object_imports", []).append(
        {
            "object_id": target,
            "format": imported.format,
            "sample_index": imported.sample_index,
            "sources": list(imported.sources),
            "disposition": disposition,
        }
    )
    return doc


def write_composed_scenario(
    scenario,
    imported,
    output,
    *,
    scenario_name=None,
    output_dir=None,
    object_id=None,
    replace=False,
    allow_model_change=False,
    additional_sources=(),
):
    from sim.data_io.common import source_receipt

    identities = [source_receipt(path) for path in additional_sources]
    doc = compose_object(
        scenario, imported, object_id=object_id, replace=replace, allow_model_change=allow_model_change
    )
    if scenario_name is not None:
        doc["scenario_name"] = scenario_name
    if output_dir is not None:
        doc.setdefault("outputs", {})["output_dir"] = str(output_dir)
    if output_dir is None:
        raise DataIOError("Choose a new run output directory for the composed scenario.")
    artifact = ScenarioArtifact.from_dict(doc)
    from sim.api import SimulationWorkspace

    validation = SimulationWorkspace().validate_safe(artifact)
    if not validation.get("ok", validation.get("valid", False)):
        raise DataIOError(f"Composed scenario failed validation: {validation}.")
    imported.verify_sources()
    if identities != [source_receipt(path) for path in additional_sources]:
        raise DataIOError("The destination scenario source changed during composition.")
    return publish(
        output,
        artifact.to_yaml_text().encode(),
        receipt={
            "operation": "compose-object",
            "validation": validation,
            "validation_scope": "safe structural validation; plugin imports and execution remain separate",
            "sources": [*imported.sources, *identities],
            "object_imports": doc["metadata"]["object_imports"],
        },
        sources=tuple(row["path"] for row in imported.sources) + tuple(additional_sources),
    )


def create_scenario(imported, output, *, scenario_name, output_dir, duration_s, step_s, object_id=None):
    if not np.isfinite(duration_s) or not np.isfinite(step_s) or duration_s <= 0 or step_s <= 0:
        raise DataIOError("A new scenario needs positive finite duration_s and step_s.")
    value = imported.value
    earth = isinstance(value, CatalogRecord) or getattr(value, "center", None) == "EARTH"
    simulator = {
        "duration_s": duration_s,
        "dt_s": step_s,
        "dynamics": {"orbit": {"model": "two_body"}, "attitude": {"enabled": False}},
    }
    if earth:
        simulator["termination"] = {"earth_impact_enabled": False}
    doc = {
        "scenario_name": scenario_name,
        "objects": {},
        "simulator": simulator,
        "outputs": {
            "mode": "save",
            "review": {"enabled": True, "detail": "standard"},
            "stats": {"save_full_log": False},
            "plots": {"enabled": False},
            "animations": {"enabled": False},
        },
    }
    return write_composed_scenario(
        doc, imported, output, scenario_name=scenario_name, output_dir=output_dir, object_id=object_id
    )
