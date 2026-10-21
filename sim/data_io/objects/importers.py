"""Normalize explicit file/review inputs through existing format owners."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from sim.data_io.api import _eop, _option_sources, _options, as_trajectory, read_data
from sim.data_io.catalog import CatalogRecord, read_catalog
from sim.data_io.common import source_receipt
from sim.data_io.contracts import AttitudeHistory, DataIOError
from sim.data_io.trajectory import transform_trajectory

from .contracts import ImportedObject

CATALOG_FORMATS = {"tle", "gp-json", "gp-csv", "omm-kvn", "omm-xml"}


def import_object(source, *, format, object_id=None, sample_index=0, options=None):
    """Inspect and select data. Eligibility is enforced when it becomes a seed."""
    options = _options(options)
    if object_id is not None:
        options["object_id"] = object_id
    paths = [source, *_option_sources(options)]
    sidecar = Path(str(source) + ".receipt.json")
    if sidecar.is_file():
        paths.append(sidecar)
    sources = tuple(source_receipt(path) for path in paths) if format not in {"review", "tracking-od"} else ()
    if format == "review":
        from sim.data_io.review import read_review

        value = read_review(source, object_id=object_id).trajectory
        # SQLite/capsules are identity-bound through the review owner as well.
        from sim.review import ReviewWorkspace

        with ReviewWorkspace.open(source) as workspace:
            if workspace.logical_db_path.is_file():
                sources = (source_receipt(workspace.logical_db_path),)
        if sources:
            sources += tuple(source_receipt(path) for path in _option_sources(options))
    elif format == "tracking-od":
        from .od import read_tracking_od

        value = read_tracking_od(source)
        sources = tuple(value.metadata["tracking_od_binding"]["sources"])
        if object_id is not None and value.object_id != object_id:
            raise DataIOError("Tracking OD object ID does not match the selected object.")
    elif format in CATALOG_FORMATS:
        records = read_catalog(source, format=format, provider_profile=options.get("provider_profile", "celestrak-gp"))
        selected = [row for row in records if object_id is None or row.object_id == object_id]
        if len(selected) != 1:
            raise DataIOError("Select exactly one catalog object and epoch; object selection is ambiguous or missing.")
        value = selected[0]
    else:
        value = read_data(source, format=format, options=options)
        if not isinstance(value, AttitudeHistory):
            value = as_trajectory(value, options=options)
        from sim.data_io.annotations import inherited_annotations

        context = inherited_annotations(source)
        value = replace(value, metadata={**context, **value.metadata})
    if options.get("target_frame"):
        if isinstance(value, (CatalogRecord, AttitudeHistory)):
            raise DataIOError("Frame conversion here requires a Cartesian orbital trajectory.")
        value = transform_trajectory(value, options["target_frame"], eop_series=_eop(options))
    if options.get("target_center") or options.get("spice_target_frame"):
        from sim.data_io.spice import transform_spice

        value = transform_spice(
            value,
            target_center=options.get("target_center", value.center),
            target_frame=options.get("spice_target_frame", value.frame),
            kernels=options["kernels"],
        )
    result = ImportedObject(value, format, sample_index, sources, options)
    result.verify_sources()
    return result
