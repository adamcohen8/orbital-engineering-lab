"""One export router; domain owners retain their semantics and artifacts."""

from __future__ import annotations

from sim.data_io.api import _option_sources, _options, encode_data
from sim.data_io.common import json_bytes, publish, source_receipt
from sim.data_io.contracts import DataIOError


def export_estimate(source, output, *, format, options=None, object_id=None):
    from .od import read_tracking_od, verify_tracking_od_binding

    trajectory = read_tracking_od(source)
    if object_id is not None and trajectory.object_id != object_id:
        raise DataIOError("Object ID differs from the completed estimate.")
    options = _options(options)
    if options.get("sample_index", 0) != 0:
        raise DataIOError("A completed OD estimate has only sample index zero.")
    if format.startswith("opm-"):
        options.setdefault("sample_index", 0)
    resources = [source_receipt(path) for path in _option_sources(options)]
    data, extra = encode_data(trajectory, format=format, options=options)
    verify_tracking_od_binding(trajectory.metadata["tracking_od_binding"])
    if resources != [source_receipt(path) for path in _option_sources(options)]:
        raise DataIOError("Estimate export resources changed.")
    return publish(
        output,
        data,
        receipt={
            "operation": "export-object",
            "format": format,
            "state_semantics": "estimated",
            "input_metadata": trajectory.metadata,
            "resources": resources,
            **extra,
        },
        sources=tuple(row["path"] for row in trajectory.metadata["tracking_od_binding"]["sources"])
        + tuple(_option_sources(options)),
    )


def export_object(
    source,
    output,
    *,
    format,
    object_id=None,
    object_ids=None,
    sample_index=None,
    final=False,
    source_kind="auto",
    options=None,
):
    """Export orbital data, a state envelope, a checkpoint/snapshot, or power history."""
    from .od import locate_tracking_od

    if source_kind not in {"auto", "review", "tracking-od"}:
        raise DataIOError("source_kind must be auto, review, or tracking-od.")
    estimated = source_kind == "tracking-od" or (source_kind == "auto" and locate_tracking_od(source) is not None)
    if source_kind == "review" and locate_tracking_od(source) is not None:
        raise DataIOError("This source is completed tracking-OD evidence; select auto or tracking-od.")
    options = _options(options)
    if not isinstance(final, bool) or (final and sample_index is not None):
        raise DataIOError("Select final or sample_index, not both.")
    if format in {"satellite-checkpoint", "snapshot", "state-product", "power-history"} and options:
        raise DataIOError("Native continuation/power products do not accept orbital format adapter options.")
    if sample_index is not None:
        options["sample_index"] = sample_index
    if estimated:
        if object_ids or final or sample_index not in {None, 0}:
            raise DataIOError("A fitted OD state contains one exact epoch, not a simulated history.")
        return export_estimate(source, output, format=format, options=options, object_id=object_id)
    if format == "satellite-checkpoint":
        if sample_index is not None or not object_id:
            raise DataIOError("Satellite checkpoint export requires an object ID and selects final runtime state.")
        from sim.interchange.satellite_checkpoints import export_satellite_checkpoint

        return export_satellite_checkpoint(
            source, output_path=output, object_id=object_id, context_object_ids=object_ids or ()
        )
    if format == "snapshot":
        if object_id is not None:
            raise DataIOError("Snapshot uses object_ids, not a single object_id.")
        if not object_ids or (sample_index is None and not final):
            raise DataIOError("Snapshot export requires explicit object IDs and final or sample selection.")
        from sim.interchange.snapshots import export_completed_run_snapshot

        return export_completed_run_snapshot(
            source,
            output_path=output,
            object_ids=object_ids,
            selector="final" if final else "sample_index",
            sample_index=sample_index,
        )
    if format == "state-product":
        if object_ids:
            raise DataIOError("State-product selects one object; use snapshot for multiple objects.")
        if sample_index is None and not final:
            raise DataIOError("State-product export requires final or sample selection.")
        from sim.interchange.completed_runs import export_completed_run_state

        return export_completed_run_state(
            source,
            output_path=output,
            object_id=object_id,
            selector="final" if final else "sample_index",
            sample_index=sample_index,
        )
    if format == "power-history":
        if object_ids or sample_index is not None or final:
            raise DataIOError("Power history exports the complete selected object's recorded history.")
        if not object_id:
            raise DataIOError("Power-history export requires an explicit object ID.")
        from sim.analysis.history_adapters import history_from_review_store
        from sim.analysis.spacecraft_power import power_history_to_dict
        from sim.review import ReviewWorkspace

        with ReviewWorkspace.open(source) as workspace:
            identity = workspace.evidence_identity()
        history = power_history_to_dict(history_from_review_store(source, object_id=object_id))
        with ReviewWorkspace.open(source) as workspace:
            if workspace.evidence_identity()["sha256"] != identity["sha256"]:
                raise DataIOError("Power source changed during export.")
        return publish(
            output, json_bytes(history), receipt={"operation": "export-object", "format": format, "source": identity}
        )
    if object_ids:
        raise DataIOError("One orbital file selects one object; use snapshot for an atomic multi-object product.")
    from sim.data_io.api import export_review

    return export_review(
        source, output, format=format, object_id=object_id, sample_index=sample_index, final=final, options=options
    )
