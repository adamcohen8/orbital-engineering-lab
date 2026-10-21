"""Read-only, actionable object IO preflight and field disposition."""

from __future__ import annotations

from sim.data_io.catalog import CatalogRecord
from sim.data_io.contracts import AttitudeHistory, DataIOError
from sim.data_io.promotion import seed_decision
from sim.data_io.registry import formats
from sim.data_io.requirements import READ_OPTIONS, WRITE_OPTIONS
from sim.data_io.time import format_time


def describe_import(source, *, format, object_id=None, sample_index=0, options=None):
    options = dict(options or {})
    if object_id is not None:
        options["object_id"] = object_id
    missing = [key for key in READ_OPTIONS.get(format, ()) if options.get(key) is None]
    report = {
        "schema": "oel.object-io-preflight.v1",
        "operation": "import",
        "format": format,
        "simulation_executed": False,
        "required_options": list(READ_OPTIONS.get(format, ())),
        "missing_options": missing,
    }
    if missing:
        return {
            **report,
            "status": "input_required",
            "next_action": "Supply " + ", ".join("--" + key.replace("_", "-") for key in missing),
        }
    if format in {"satellite-checkpoint", "snapshot"}:
        from sim.interchange.inspection import inspect_path

        inspected = inspect_path(source)
        validation = inspected.get("validation", {})
        return {
            **report,
            "status": "inspected" if validation.get("valid") and validation.get("promotable") else "blocked",
            "route": "existing handoff continuation owner",
            "inspection": inspected,
            "preserved": ["native continuation contract"],
            "omitted": [],
        }
    from .importers import import_object

    try:
        imported = import_object(source, format=format, object_id=object_id, sample_index=sample_index, options=options)
        value = imported.value
        attitude = isinstance(value, AttitudeHistory)
        catalog = isinstance(value, CatalogRecord)
        if attitude and value.reference_frame != "EME2000":
            raise DataIOError("Attitude attachment requires an Earth/EME2000 body-frame history.")
        decision = {"promotable": True, "blockers": []} if attitude or catalog else seed_decision(value)
        if not attitude and decision["promotable"]:
            from .composition import orbital_entry

            orbital_entry(imported)
        physical = (
            []
            if attitude or catalog
            else sorted(set(value.metadata.get("physical", {})) | set(value.metadata.get("object_specs", {})))
        )
        return {
            **report,
            "status": "ready" if decision["promotable"] else "blocked",
            "object_id": imported.object_id,
            "epoch_utc": format_time(imported.epoch),
            "kind": "attitude" if attitude else "mean_elements" if catalog else "trajectory",
            "frame": value.reference_frame if attitude else "TEME" if catalog else value.frame,
            "center": None if attitude else "EARTH" if catalog else value.center,
            "physical_fields": physical,
            "promotion": decision,
            "sources": list(imported.sources),
            "preserved": [
                "explicit epoch/frame/units",
                "selected attitude" if attitude else "native mean elements" if catalog else "selected Cartesian PV",
                "available physical properties",
            ],
            "omitted": [
                "history replay",
                "maneuver execution",
                "controller/sensor/runtime restoration (use checkpoint)",
                "covariance propagation (covariance remains evidence)",
            ],
            "defaults": {"mass_kg_if_absent_on_new_object": 100.0} if not attitude else {},
            "next_action": "Attach the selected attitude sample to an existing ONP object."
            if attitude
            else "Create a new scenario or explicitly add/replace an object at a matching scenario epoch.",
        }
    except (ValueError, KeyError, OSError, TypeError) as exc:
        return {
            **report,
            "status": "blocked",
            "message": str(exc),
            "next_action": "Resolve the reported selector, compatibility or source-evidence requirement.",
        }


def describe_export(source, *, object_id=None, source_kind="auto"):
    from .od import locate_tracking_od, read_tracking_od

    if source_kind not in {"auto", "review", "tracking-od"}:
        raise DataIOError("source_kind must be auto, review, or tracking-od.")
    estimated = source_kind == "tracking-od" or (source_kind == "auto" and locate_tracking_od(source) is not None)
    if source_kind == "review" and locate_tracking_od(source) is not None:
        raise DataIOError("This source is completed tracking-OD evidence; select auto or tracking-od.")
    special = ["state-product", "satellite-checkpoint", "snapshot", "power-history"]
    attitude_blocker = "An estimated OD state does not contain achieved attitude."
    if estimated:
        value = read_tracking_od(source)
        summary = {
            "source_kind": "tracking-od",
            "state_semantics": "estimated",
            "objects": [value.object_id],
            "frame": value.frame,
            "samples": 1,
            "covariance_calibrated": False,
        }
        special = []
        if object_id is not None and value.object_id != object_id:
            raise DataIOError("Selected object differs from the completed estimate.")
    else:
        from sim.review import ReviewWorkspace

        with ReviewWorkspace.open(source) as workspace:
            tables = workspace.tables()
            table = "planetary_state" if "planetary_state" in tables else "object_state"
            if table not in tables:
                raise DataIOError(
                    "This review is not a simulation history; select the completed tracking-OD workflow when exporting an estimate."
                )
            ids = workspace.query(f"SELECT DISTINCT object_id FROM {table} ORDER BY object_id", max_rows=10001)
            summary = {
                "source_kind": "review",
                "state_semantics": "simulated truth",
                "objects": [row["object_id"] for row in ids.rows],
                "source": workspace.evidence_identity(),
            }
        if object_id is not None and object_id not in summary["objects"]:
            raise DataIOError("Selected object is absent from this review store.")
        value = None
        if object_id is not None or len(summary["objects"]) == 1:
            from sim.data_io.review import read_review

            value = read_review(source, object_id=object_id).trajectory
            try:
                read_review(source, object_id=object_id, include_attitude=True)
                attitude_blocker = None
            except DataIOError as exc:
                attitude_blocker = str(exc)
            summary.update(frame=value.frame, center=value.center, samples=len(value.times_s))
    selection_required = object_id is None and len(summary["objects"]) > 1
    products = [
        {
            "format": row["format"],
            "required_options": list(WRITE_OPTIONS.get(row["format"], ())),
            "limits": row["limits"],
            "dependency_available": row["dependency_available"],
            "frame_requirements": "GCRF or ITRF"
            if row["format"] == "czml"
            else "ITRF with epoch-covering EOP for Earth transforms"
            if row["format"] in {"sp3", "geojson", "kml"}
            else "explicit compatible frame",
        }
        for row in formats()
        if row["write"]
        and row["kind"]
        in (
            {"trajectory", "visualization", "navigation"}
            if estimated
            else {"trajectory", "visualization", "attitude", "navigation"}
        )
        and row["format"] not in {"omm-kvn", "omm-xml", "cdm-kvn", "cdm-xml"}
    ]
    for product in products:
        name = product["format"]
        blockers = []
        conditions = list(product["required_options"])
        if not product["dependency_available"]:
            blockers.append("Install the declared optional format dependency.")
        if selection_required:
            conditions.append("object_id")
        elif value is not None:
            if name in {"aem-kvn", "apm-kvn", "stk-attitude", "ck"}:
                if attitude_blocker:
                    blockers.append(attitude_blocker)
                if name == "apm-kvn":
                    conditions.append("sample_index")
            else:
                if value.frame.startswith("SPICE:") and not name.startswith("trajectory-") and name != "spk":
                    blockers.append(
                        "Preserve named SPICE axes through trajectory tables or SPK; this profile needs a supported explicit frame conversion."
                    )
                if value.covariances is not None and name in {"stk-ephemeris", "sp3", "spk"}:
                    blockers.append("This profile cannot carry the supplied covariance; use OEM or trajectory tables.")
                if name == "spk" and len(value.times_s) < 2:
                    blockers.append("SPK type 13 requires at least two PV samples.")
                if name in {"czml", "sp3", "geojson", "kml"} and value.center == "EARTH":
                    conditions.append("explicit compatible target_frame; ITRF conversion requires epoch-covering EOP")
        if estimated:
            conditions = [item for item in conditions if item != "sample_index"]
        product.update(
            available=not blockers and not selection_required,
            status="blocked"
            if blockers
            else "selection_required"
            if selection_required
            else "options_required"
            if conditions
            else "ready",
            blockers=blockers,
            conditions=conditions,
        )
    return {
        "schema": "oel.object-io-preflight.v1",
        "operation": "export",
        "status": "inspected",
        "simulation_executed": False,
        **summary,
        "object_selection_required": selection_required,
        "products": products,
        "continuation_products": special,
        "continuation_requirements": {
            "state-product": "Earth state, explicit final/sample selection, promotable native handoff contract",
            "satellite-checkpoint": "Earth ONP object with final compatible GNC runtime state; native owner validates compatibility",
            "snapshot": "at least two selected Earth objects and exact final/sample selection; native owner validates compatibility",
            "power-history": "explicit object with compatible recorded power history",
        }
        if not estimated
        else {},
        "semantics": {
            "orbital_formats": "orbital evidence with supported physical fields; not a complete runtime checkpoint",
            "satellite-checkpoint": "final complete GNC runtime continuation",
            "snapshot": "atomic state/context selection for multiple objects",
            "mean_elements": "fit-review is a separate Pro estimation action",
        },
    }
