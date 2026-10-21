"""Materialize imported state/catalog data through normal scenario validation."""

from __future__ import annotations

import numpy as np

from sim.frame_time import epoch_julian_date
from sim.scenarios import ScenarioArtifact

from .catalog import CatalogRecord
from .common import publish, source_receipt
from .contracts import DataIOError, Trajectory
from .time import at


def materialize_scenario(
    source: Trajectory | list[CatalogRecord],
    output,
    *,
    scenario_name: str,
    output_dir,
    duration_s: float,
    step_s: float,
    start_epoch=None,
    source_artifacts=(),
    expected_source_identities=None,
):
    identities = [source_receipt(path) for path in source_artifacts]
    if expected_source_identities is not None and identities != expected_source_identities:
        raise DataIOError("Source artifacts changed after input parsing.")
    if not np.isfinite(duration_s) or duration_s <= 0 or not np.isfinite(step_s) or step_s <= 0:
        raise DataIOError("Scenario duration and timestep must be positive finite numbers.")
    objects = {}
    warnings = []
    if isinstance(source, Trajectory):
        from .objects.composition import orbital_entry
        from .objects.contracts import ImportedObject

        entry, seed_context = orbital_entry(ImportedObject(source, "normalized"))
        epoch = at(source.epoch, source.times_s[0])
        objects[source.object_id] = entry
        warnings.append("First sample initializes a new ONP study; source trajectory is not replayed.")
        if source.covariances is not None:
            warnings.append(
                "Source covariance remains input evidence; this passive scenario does not estimate or propagate covariance."
            )
    else:
        if not source:
            raise DataIOError("Catalog must not be empty.")
        if start_epoch is None and len(source) == 1:
            from sim.dynamics.orbit.epoch import julian_date_to_datetime

            from .time import parse_time

            start_epoch = parse_time(
                julian_date_to_datetime(source[0].elements.epoch_jd_utc).isoformat().replace("+00:00", "Z")
            )
        if start_epoch is None:
            raise DataIOError("Multi-object catalog propagation requires an explicit common start epoch.")
        epoch = start_epoch
        for record in source:
            if record.object_id in objects:
                raise DataIOError("Select one mean-element epoch per catalog object before materialization.")
            from .objects.composition import orbital_entry
            from .objects.contracts import ImportedObject

            objects[record.object_id], _ = orbital_entry(ImportedObject(record, "catalog"))

    scenario = {
        "scenario_name": scenario_name,
        "objects": objects,
        "simulator": {
            "initial_jd_utc": epoch_julian_date(epoch, "UTC"),
            "duration_s": duration_s,
            "dt_s": step_s,
            "dynamics": {"orbit": {"model": "two_body"}, "attitude": {"enabled": False}},
            "termination": {"earth_impact_enabled": False},
        },
        "outputs": {
            "output_dir": str(output_dir),
            "mode": "save",
            "stats": {"save_full_log": False},
            "review": {"enabled": True, "detail": "standard"},
            "plots": {"enabled": False},
            "animations": {"enabled": False},
        },
    }
    if isinstance(source, Trajectory):
        for key in ("central_body", "planetary", "environment", "frames", "plugin_validation"):
            if key in seed_context:
                scenario["simulator"][key] = seed_context[key]
        if "dynamics" in seed_context:
            scenario["simulator"]["dynamics"].update(seed_context["dynamics"])
        if source.center.upper() != "EARTH":
            scenario["simulator"].pop("termination", None)
    artifact = ScenarioArtifact.from_dict(scenario)
    from sim.api import SimulationWorkspace

    report = SimulationWorkspace().validate_safe(artifact)
    if not report.get("ok", report.get("valid", False)):
        raise DataIOError(f"Generated scenario failed validation: {report}.")
    if [source_receipt(path) for path in source_artifacts] != identities:
        raise DataIOError("Input artifacts changed during scenario materialization.")
    return publish(
        output,
        artifact.to_yaml_text().encode(),
        receipt={
            "operation": "materialize-data-scenario",
            "validation": report,
            "validation_scope": "safe structural validation; ordinary validation and execution are separate",
            "sources": identities,
            "warnings": warnings,
            "defaults": {"fallback_mass_kg": 100.0, "dynamics": "two_body", "passive": True},
            "object_masses_kg": {key: obj["specs"].get("mass_kg") for key, obj in objects.items()},
        },
        sources=tuple(source_artifacts),
    )
