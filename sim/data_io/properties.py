"""Physical field mapping shared by object import and state export."""

from __future__ import annotations

from copy import deepcopy

import numpy as np

from .contracts import DataIOError

# These are declared values, not inferred spacecraft properties.
OPM_SPECS = {
    "MASS": "mass_kg",
    "DRAG_AREA": "drag_area_m2",
    "DRAG_COEFF": "cd",
    "SOLAR_RAD_AREA": "srp_area_m2",
    "SOLAR_RAD_COEFF": "cr",
}


def physical_at_sample(trajectory, index):
    values = {}
    specs = trajectory.metadata.get("object_specs", {})
    for key, field in OPM_SPECS.items():
        if field in specs:
            values[key] = str(float(specs[field]))
    values.update(trajectory.metadata.get("physical", {}))
    masses = trajectory.metadata.get("mass_kg_by_sample")
    if masses is not None:
        if len(masses) != len(trajectory.times_s):
            raise DataIOError("Mass history does not match trajectory samples.")
        values["MASS"] = str(float(masses[index]))
    for key, value in values.items():
        if key in OPM_SPECS and (isinstance(value, bool) or not np.isfinite(float(value)) or float(value) < 0):
            raise DataIOError(f"Invalid physical property {key}.")
    return values


def specs_at_sample(trajectory, index):
    specs = deepcopy(trajectory.metadata.get("object_specs", {}))
    for key, value in physical_at_sample(trajectory, index).items():
        if key in OPM_SPECS:
            specs[OPM_SPECS[key]] = float(value)
    mass = float(specs.setdefault("mass_kg", 100.0))
    if not np.isfinite(mass) or mass <= 0:
        raise DataIOError("Spacecraft mass must be finite and positive.")
    # Current total mass cannot safely be combined with stale configured fuel.
    if trajectory.metadata.get("mass_kg_by_sample") is not None or "MASS" in trajectory.metadata.get("physical", {}):
        specs.pop("dry_mass_kg", None)
        specs.pop("fuel_mass_kg", None)
    return specs
