"""Build external navigation products from already-computed OEL evidence."""

from __future__ import annotations

import json

import numpy as np

from sim.interchange.ccsds_odm import OdmCovariance, OdmHeader, OdmMetadata, OpmMessage, validate_odm

from .common import digest, read_text, source_receipt, strict_json
from .contracts import DataIOError, Trajectory
from .properties import physical_at_sample
from .time import at, creation_time, format_time, parse_time


def opm_from_trajectory(t: Trajectory, *, sample_index: int, maneuvers=()) -> OpmMessage:
    if isinstance(sample_index, bool) or not isinstance(sample_index, int) or not 0 <= sample_index < len(t.times_s):
        raise DataIOError("Select one valid integer trajectory sample index.")
    time, pv = float(t.times_s[sample_index]), t.states[sample_index]
    covariance = None
    if t.covariances is not None:
        matches = np.flatnonzero(t.covariance_times_s == time)
        if len(matches) != 1:
            raise DataIOError(
                "Selected state has no exact matching covariance; no covariance interpolation is implied."
            )
        covariance = OdmCovariance(tuple(map(tuple, t.covariances[int(matches[0])])), t.frame)
    state = dict(zip(("X", "Y", "Z", "X_DOT", "Y_DOT", "Z_DOT"), map(str, pv)))
    state = {"EPOCH": format_time(at(t.epoch, time)), **state}
    message = OpmMessage(
        OdmHeader("3.0", creation_time(), "OEL"),
        OdmMetadata(t.object_name or t.object_id, t.object_id, t.center, t.frame, "UTC"),
        state,
        {},
        physical_at_sample(t, sample_index),
        tuple(maneuvers),
        covariance,
        {},
        {},
    )
    validate_odm(message)
    return message


def trajectory_from_state_product(path) -> Trajectory:
    """Export a validated product without promoting or executing it."""
    from sim.dynamics.orbit.epoch import julian_date_to_datetime
    from sim.interchange.validation import load_interchange_document, validate_product

    product = load_interchange_document(path)
    report = validate_product(product, source_path=path)
    if not report.valid:
        raise DataIOError("Source state product failed contract/provenance validation.")
    if product["product_kind"] not in {"oel.state_estimate", "oel.completed_run_state"}:
        raise DataIOError("Expected a Cartesian OEL state product.")
    payload = product["payload"]
    state = payload["state"]
    if state["frame"] != "ECI" or state["units"] != ["km"] * 3 + ["km/s"] * 3:
        raise DataIOError("State product requires canonical ECI km/km/s semantics.")
    value = state["epoch"]
    if value["format"] != "jd" or value["time_system"] != "UTC":
        raise DataIOError("Unsupported state-product epoch representation.")
    epoch = parse_time(julian_date_to_datetime(value["value"]).isoformat().replace("+00:00", "Z"))
    covariance = payload.get("covariance", {})
    matrices = None
    if covariance.get("present"):
        if covariance.get("frame") != "ECI" or covariance.get("units") != ["km"] * 3 + ["km/s"] * 3:
            raise DataIOError("Unsupported state-product covariance semantics.")
        matrices = [covariance["matrix"]]
    return Trajectory(
        payload["object"]["object_id"],
        "EARTH",
        "EME2000",
        epoch,
        [0],
        [state["values"]],
        covariance_times_s=[0] if matrices else None,
        covariances=matrices,
        metadata={
            "product_id": product["product_id"],
            "quality": product["quality"],
            "freshness": product["freshness"],
            "covariance_disposition": covariance,
            "data_markings": product["data_markings"],
            "object_specs": payload.get("object_specs", {}),
            "model_assumptions": payload.get("model_assumptions", {}),
            "resource_state": payload.get("resource_state", {}),
            "oel_product_binding": {"product": product, "source": source_receipt(path)},
        },
    )


def cdm_from_assessment(evidence_path, problem_path, *, initial_epoch: str):
    """Serialize the saved baseline assessment, without rerunning the assessment."""
    from sim.analysis.conjunction_probability import ric_basis
    from sim.analysis.conjunction_workflow import ConjunctionAssessmentProblem
    from sim.interchange.ccsds_cdm import CdmHeader, CdmMessage, CdmObject, CdmRelativeMetadata, validate_cdm

    evidence = strict_json(read_text(evidence_path))
    problem = ConjunctionAssessmentProblem.from_mapping(strict_json(read_text(problem_path)))
    canonical = json.dumps(problem.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    if evidence.get("schema_version") != "oel.conjunction_assessment_evidence.v1" or evidence.get(
        "problem_sha256"
    ) != digest(canonical):
        raise DataIOError("Assessment evidence is not bound to the supplied problem.")
    baseline = evidence.get("baseline", {})
    if baseline.get("disposition") != "interior_tca_assessed":
        raise DataIOError("Only a completed interior-TCA assessment can produce this CDM profile.")
    closest = baseline["closest_approach"]
    epoch = parse_time(initial_epoch)
    tca = format_time(at(epoch, closest["time_s"]))
    primary = np.asarray(closest["primary_state_eci_km_km_s"], dtype=float)
    secondary = np.asarray(closest["secondary_state_eci_km_km_s"], dtype=float)
    relative = secondary - primary
    rotation = ric_basis(primary).T
    position, velocity = rotation @ relative[:3] * 1000, rotation @ relative[3:] * 1000
    pc = baseline["probability"]["collision_probability"]
    objects = []
    for index, (obj, state) in enumerate(((problem.primary, primary), (problem.secondary, secondary)), 1):
        basis = ric_basis(state).T
        transform = np.zeros((6, 6))
        transform[:3, :3] = transform[3:, 3:] = basis
        matrix = transform @ np.asarray(obj.covariance_eci_km_at_tca) @ transform.T * 1e6
        metadata = {
            "OBJECT_DESIGNATOR": obj.object_id,
            "CATALOG_NAME": "OEL",
            "OBJECT_NAME": obj.object_id,
            "INTERNATIONAL_DESIGNATOR": obj.object_id,
            "EPHEMERIS_NAME": "OEL SAVED ASSESSMENT",
            "COVARIANCE_METHOD": "CALCULATED",
            "MANEUVERABLE": "N/A",
            "ORBIT_CENTER": "EARTH",
            "REF_FRAME": "EME2000",
        }
        objects.append(CdmObject(f"OBJECT{index}", metadata, tuple(state), tuple(map(tuple, matrix)), {}, {}, {}))
    message = CdmMessage(
        CdmHeader("1.0", creation_time(), "OEL"),
        CdmRelativeMetadata(
            tca,
            float(np.linalg.norm(position)),
            float(np.linalg.norm(velocity)),
            tuple(position),
            tuple(velocity),
            float(pc),
            "OEL_EDUCATIONAL_2D",
        ),
        tuple(objects),
        comments=(
            "Derived from saved bounded assessment; covariance is declared at TCA, not calibrated or propagated.",
        ),
    )
    validate_cdm(message)
    return message


def tdm_from_measurements(rows, *, station_id: str, object_id: str):
    from sim.interchange.ccsds_tdm import parse_tdm_kvn

    if isinstance(rows, dict):
        if "measurement_rows" in rows:
            from sim.tracking_data import NORMALIZED_TRACKING_DATASET_SCHEMA

            if (
                rows.get("schema_version") != NORMALIZED_TRACKING_DATASET_SCHEMA
                or rows.get("measurement_semantics") != "reduced_geometric"
                or rows.get("object_id") != object_id
            ):
                raise DataIOError("Normalized tracking dataset identity/semantics do not match the requested object.")
            component_map = {"azimuth_deg": "ANGLE_1", "elevation_deg": "ANGLE_2", "range_km": "RANGE"}
            expanded = []
            for row in rows["measurement_rows"]:
                if row["station_id"] != station_id:
                    continue
                if len(row["components"]) != len(row["vector"]):
                    raise DataIOError("Measurement component/vector dimensions disagree.")
                for component, value in zip(row["components"], row["vector"]):
                    if component not in component_map:
                        raise DataIOError(f"Unsupported reduced-geometric TDM component: {component}.")
                    expanded.append(
                        {
                            "epoch_utc": row["epoch_utc"],
                            "observable": component_map[component],
                            "value": value,
                            "measurement_semantics": "reduced_geometric",
                        }
                    )
            rows = expanded
        else:
            rows = rows["rows"]

    if not rows:
        raise DataIOError("No synthetic measurement rows supplied.")
    ordered = sorted(rows, key=lambda row: (parse_time(row["epoch_utc"]).tai_seconds, row["observable"]))
    header = [
        "CCSDS_TDM_VERS = 2.0",
        "CREATION_DATE = " + ordered[0]["epoch_utc"],
        "ORIGINATOR = OEL",
        "META_START",
        "TIME_SYSTEM = UTC",
        "START_TIME = " + ordered[0]["epoch_utc"],
        "STOP_TIME = " + ordered[-1]["epoch_utc"],
        "PARTICIPANT_1 = " + station_id,
        "PARTICIPANT_2 = " + object_id,
        "MODE = SEQUENTIAL",
        "PATH = 2,1",
        "ANGLE_TYPE = AZEL",
        "RANGE_MODE = ONE_WAY",
        "RANGE_UNITS = km",
        "RANGE_MODULUS = 0",
        "TIMETAG_REF = RECEIVE",
        "DATA_QUALITY = VALIDATED",
        "CORRECTIONS_APPLIED = YES",
        "META_STOP",
        "DATA_START",
    ]
    for row in ordered:
        if (
            row["observable"] not in {"ANGLE_1", "ANGLE_2", "RANGE"}
            or row.get("measurement_semantics") != "reduced_geometric"
        ):
            raise DataIOError("Synthetic TDM export requires declared reduced-geometric AZEL degrees/range km rows.")
        header.append(f"{row['observable']} = {row['epoch_utc']} {float(row['value']):.17g}")
    return parse_tdm_kvn("\n".join([*header, "DATA_STOP"]) + "\n")


def maneuver_plan_from_opm(message):
    """Normalize OPM maneuver blocks into a non-executing, explicit plan."""
    from sim.interchange.ccsds_odm import OpmMessage

    if not isinstance(message, OpmMessage):
        raise DataIOError("A maneuver plan requires an OPM.")
    maneuvers = []
    for index, row in enumerate(message.maneuvers):
        duration = float(row["MAN_DURATION"])
        delta_mass = float(row["MAN_DELTA_MASS"])
        if duration < 0 or delta_mass > 0:
            raise DataIOError(
                "Maneuver duration must be nonnegative and consumed mass must not increase spacecraft mass."
            )
        frame = row["MAN_REF_FRAME"]
        if frame not in {"EME2000", "GCRF", "RTN", "RSW", "RIC"}:
            raise DataIOError("Unsupported maneuver reference frame; no axis relabeling is performed.")
        maneuvers.append(
            {
                "maneuver_id": f"opm:{index}",
                "ignition_epoch_utc": format_time(parse_time(row["MAN_EPOCH_IGNITION"], message.metadata.time_system)),
                "duration_s": duration,
                "frame": frame,
                "delta_v_m_s": [float(row[f"MAN_DV_{i}"]) * 1000 for i in range(1, 4)],
                "delta_mass_kg": delta_mass,
                "representation": "impulsive" if duration == 0 else "finite_burn_integrated_delta_v",
            }
        )
    return {
        "schema": "oel.maneuver-plan.v1",
        "object_id": message.metadata.object_id,
        "center": message.metadata.center_name,
        "maneuvers": maneuvers,
        "source_sha256": message.source_sha256,
        "execution_authorized": False,
        "interpretation": "OPM declares integrated delta-v; it does not specify an acceleration profile or authorize execution.",
    }


def opm_from_review_maneuvers(completed_run, *, object_id, sample_index=0):
    from sim.review import ReviewWorkspace

    from .review import read_review

    selection = read_review(completed_run, object_id=object_id)
    with ReviewWorkspace.open(completed_run) as workspace:
        before = workspace.evidence_identity()
        if before["sha256"] != selection.receipt["source_review_store"]["sha256"]:
            raise DataIOError("Review changed between state and maneuver selections.")
        result = workspace.query(
            "SELECT maneuver_id, time_s, frame, dv_x_eci_m_s, dv_y_eci_m_s, dv_z_eci_m_s FROM impulsive_maneuvers WHERE object_id = ? ORDER BY time_s, maneuver_id",
            (object_id,),
            max_rows=10000,
        )
        if result.truncated or not result.rows:
            raise DataIOError("No complete bounded impulsive-maneuver selection.")
        if workspace.evidence_identity()["sha256"] != before["sha256"]:
            raise DataIOError("Review changed during maneuver selection.")
    maneuvers = []
    for row in result.rows:
        if row["frame"] not in {"ECI", "eci"}:
            raise DataIOError("Impulsive review maneuver requires canonical ECI vectors.")
        maneuvers.append(
            {
                "MAN_EPOCH_IGNITION": format_time(at(selection.trajectory.epoch, row["time_s"])),
                "MAN_DURATION": "0",
                "MAN_DELTA_MASS": "0",
                "MAN_REF_FRAME": "EME2000",
                **{
                    f"MAN_DV_{i}": str(float(row[key]) / 1000)
                    for i, key in enumerate(("dv_x_eci_m_s", "dv_y_eci_m_s", "dv_z_eci_m_s"), 1)
                },
            }
        )
    return opm_from_trajectory(selection.trajectory, sample_index=sample_index, maneuvers=maneuvers), {
        **selection.receipt,
        "maneuver_ids": [row["maneuver_id"] for row in result.rows],
        "maneuver_interpretation": "historical ideal impulsive commands; zero mass change is the source ideal-impulse model, not measured propellant consumption",
    }


def opm_from_maneuver_plan(plan, trajectory, *, sample_index):
    if (
        plan.get("schema") != "oel.maneuver-plan.v1"
        or plan.get("object_id") != trajectory.object_id
        or plan.get("center") != trajectory.center
    ):
        raise DataIOError("Maneuver plan and reference trajectory object/center disagree.")
    if plan.get("execution_authorized") is not False:
        raise DataIOError("IO maneuver plans cannot carry execution authorization.")
    maneuvers = []
    for row in plan["maneuvers"]:
        vector = np.asarray(row["delta_v_m_s"], dtype=float)
        if vector.shape != (3,) or not np.all(np.isfinite(vector)):
            raise DataIOError("Plan delta-v must contain three finite components in m/s.")
        maneuvers.append(
            {
                "MAN_EPOCH_IGNITION": format_time(parse_time(row["ignition_epoch_utc"])),
                "MAN_DURATION": str(float(row["duration_s"])),
                "MAN_DELTA_MASS": str(float(row["delta_mass_kg"])),
                "MAN_REF_FRAME": row["frame"],
                **{f"MAN_DV_{i}": str(value / 1000) for i, value in enumerate(vector, 1)},
            }
        )
    message = opm_from_trajectory(trajectory, sample_index=sample_index, maneuvers=maneuvers)
    maneuver_plan_from_opm(message)
    return message
