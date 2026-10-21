"""Read-only, identity-bound extraction from completed OEL review stores."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from sim.dynamics.orbit.epoch import julian_date_to_datetime
from sim.review import ReviewWorkspace

from .common import digest
from .contracts import MAX_SAMPLES, AttitudeHistory, DataIOError, Trajectory
from .time import parse_time

PV_COLUMNS = ("pos_x_eci_km", "pos_y_eci_km", "pos_z_eci_km", "vel_x_eci_km_s", "vel_y_eci_km_s", "vel_z_eci_km_s")


@dataclass(frozen=True)
class ReviewSelection:
    trajectory: Trajectory
    receipt: dict[str, Any]
    attitude: AttitudeHistory | None = None


def select_review_sample(selection, *, sample_index=None, final=False):
    """Select one exact saved row, including its aligned mass/covariance/attitude."""
    trajectory = selection.trajectory
    index = len(trajectory.times_s) - 1 if final else sample_index
    if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(trajectory.times_s):
        raise DataIOError("Select one valid integer review sample index.")
    time = trajectory.times_s[index]
    cov_indices = (
        []
        if trajectory.covariances is None
        else np.flatnonzero(np.abs(trajectory.covariance_times_s - time) < 1e-9).tolist()
    )
    receipt = {
        **selection.receipt,
        "sample_indices": [selection.receipt["sample_indices"][index]],
        "selected_row_index": index,
    }
    metadata = {
        **trajectory.metadata,
        "review": receipt,
        "mass_kg_by_sample": [trajectory.metadata["mass_kg_by_sample"][index]],
    }
    trajectory = replace(
        trajectory,
        times_s=trajectory.times_s[index : index + 1],
        states=trajectory.states[index : index + 1],
        covariance_times_s=None if not cov_indices else trajectory.covariance_times_s[cov_indices],
        covariances=None if not cov_indices else trajectory.covariances[cov_indices],
        metadata=metadata,
    )
    attitude = selection.attitude
    if attitude is not None:
        attitude = replace(
            attitude,
            times_s=attitude.times_s[index : index + 1],
            quaternions=attitude.quaternions[index : index + 1],
            rates_rad_s=None if attitude.rates_rad_s is None else attitude.rates_rad_s[index : index + 1],
            metadata={**attitude.metadata, "review": receipt},
        )
    return ReviewSelection(trajectory, receipt, attitude)


def _rows(workspace, sql, parameters=(), *, limit=MAX_SAMPLES):
    result = workspace.query(sql, parameters, max_rows=limit, max_vm_steps=max(250_000, limit * 300))
    if result.truncated:
        raise DataIOError("Review selection exceeds the explicit row/resource limit; narrow the selection.")
    return list(result.rows)


def read_review(completed_run, *, object_id=None, start_s=None, stop_s=None, include_attitude=False) -> ReviewSelection:
    """Select saved truth, with no simulation calls and no log/config fallback."""
    with ReviewWorkspace.open(completed_run) as workspace:
        identity = workspace.evidence_identity()
        if "run_metadata" not in workspace.tables():
            raise DataIOError(
                "This review contains domain results rather than simulation state history; use object export on the completed tracking-OD workflow for fitted estimates."
            )
        metadata = _rows(workspace, "SELECT run_id, config_json, config_sha256 FROM run_metadata", limit=2)
        if len(metadata) != 1:
            raise DataIOError("Exactly one source run is required.")
        raw_config = metadata[0]["config_json"]
        if digest(raw_config.encode()) != metadata[0]["config_sha256"]:
            raise DataIOError("Review configuration identity mismatch.")
        config = json.loads(raw_config)
        if include_attitude and not config.get("simulator", {}).get("dynamics", {}).get("attitude", {}).get(
            "enabled", True
        ):
            raise DataIOError("The source run disabled attitude; placeholder quaternions are not an attitude product.")
        initial_jd = config.get("simulator", {}).get("initial_jd_utc")
        if initial_jd is None or not np.isfinite(float(initial_jd)):
            raise DataIOError("An absolute source initial_jd_utc is required.")
        epoch = parse_time(julian_date_to_datetime(float(initial_jd)).isoformat().replace("+00:00", "Z"))
        tables = workspace.tables()
        planetary = "planetary_state" in tables
        table = "planetary_state" if planetary else "object_state"
        objects = _rows(workspace, f"SELECT DISTINCT object_id FROM {table} ORDER BY object_id", limit=10_001)
        ids = [str(row["object_id"]) for row in objects]
        if object_id is None:
            if len(ids) != 1:
                raise DataIOError(f"Explicit object selection required: {ids}.")
            object_id = ids[0]
        if object_id not in ids:
            raise DataIOError(f"No history for object {object_id!r}.")
        if start_s is not None and not np.isfinite(float(start_s)):
            raise DataIOError("start_s must be finite.")
        if stop_s is not None and not np.isfinite(float(stop_s)):
            raise DataIOError("stop_s must be finite.")
        if start_s is not None and stop_s is not None and start_s > stop_s:
            raise DataIOError("start_s must not exceed stop_s.")
        if planetary:
            env = _rows(workspace, "SELECT central_body, inertial_axes FROM planetary_environment", limit=2)
            if len(env) != 1 or env[0]["inertial_axes"] != "J2000":
                raise DataIOError("Unsupported planetary review axes.")
            center, frame = str(env[0]["central_body"]).upper(), "SPICE:J2000"
            names = (
                "x_inertial_km",
                "y_inertial_km",
                "z_inertial_km",
                "vx_inertial_km_s",
                "vy_inertial_km_s",
                "vz_inertial_km_s",
            )
        else:
            frames = _rows(
                workspace, "SELECT state_frame FROM object_state_frame WHERE object_id = ?", (object_id,), limit=2
            )
            declared = str(frames[0]["state_frame"]).upper() if len(frames) == 1 else ""
            if declared not in {"ECI", "TEME"}:
                raise DataIOError("Named Earth ECI/TEME evidence required; rotating CR3BP/other frames are not ECI.")
            if include_attitude and declared != "ECI":
                raise DataIOError("Native TEME catalog states do not establish an attitude reference convention.")
            center, frame, names = "EARTH", {"ECI": "EME2000", "TEME": "TEME"}[declared], PV_COLUMNS
        conditions, parameters = ["object_id = ?"], [object_id]
        for operator, value in ((">=", start_s), ("<=", stop_s)):
            if value is not None:
                conditions.append(f"time_s {operator} ?")
                parameters.append(float(value))
        columns = "sample_index, time_s, mass_kg, " + ", ".join(names)
        if include_attitude:
            if planetary:
                raise DataIOError("Planetary review does not contain attitude histories.")
            columns += ", quat_w, quat_x, quat_y, quat_z, omega_x_rad_s, omega_y_rad_s, omega_z_rad_s"
        query = f"SELECT {columns} FROM {table} WHERE {' AND '.join(conditions)} ORDER BY sample_index"
        rows = _rows(workspace, query, tuple(parameters))
        if not rows:
            raise DataIOError("Selected review interval has no samples.")
        if any(row["mass_kg"] is None or not np.isfinite(row["mass_kg"]) or row["mass_kg"] <= 0 for row in rows):
            raise DataIOError("Review samples require positive finite saved mass.")
        if any(int(b["sample_index"]) <= int(a["sample_index"]) for a, b in zip(rows, rows[1:])):
            raise DataIOError("Review sample indices must be strictly increasing.")
        covariance_times, matrices = [], []
        covariance_dispositions = []
        if not planetary and "object_state_covariance" in tables:
            covs = _rows(
                workspace,
                "SELECT * FROM object_state_covariance WHERE object_id = ? ORDER BY sample_index",
                (object_id,),
            )
            selected = {int(row["sample_index"]): float(row["time_s"]) for row in rows}
            for row in covs:
                if int(row["sample_index"]) not in selected:
                    continue
                expected_covariance_frame = "ECI" if frame == "EME2000" else "TEME"
                if str(row["frame"]).upper() != expected_covariance_frame or not row["mathematically_valid"]:
                    raise DataIOError("Invalid/noncanonical review covariance.")
                if json.loads(row["component_order_json"]) != ["x", "y", "z", "vx", "vy", "vz"]:
                    raise DataIOError("Unsupported review covariance ordering.")
                if json.loads(row["units_json"]) != ["km", "km", "km", "km/s", "km/s", "km/s"]:
                    raise DataIOError("Unsupported review covariance units.")
                if abs(float(row["time_s"]) - selected[int(row["sample_index"])]) > 1e-9:
                    raise DataIOError("Covariance and state sample epochs disagree.")
                covariance_times.append(float(row["time_s"]))
                matrices.append(json.loads(row["covariance_json"]))
                covariance_dispositions.append(
                    {"calibrated": bool(row["calibrated"]), "scope": row["calibration_scope"]}
                )
        after = workspace.evidence_identity()
        if after["sha256"] != identity["sha256"]:
            raise DataIOError("Source review evidence changed while reading.")
        receipt = {
            "source_review_store": identity,
            "source_run_id": metadata[0]["run_id"],
            "source_config_sha256": metadata[0]["config_sha256"],
            "object_id": object_id,
            "sample_indices": [int(row["sample_index"]) for row in rows],
            "query": query,
            "parameters": parameters,
            "covariance_dispositions": covariance_dispositions,
            "interpretation": "completed simulated truth; not an independently measured orbit",
        }
        trajectory = Trajectory(
            object_id=object_id,
            center=center,
            frame=frame,
            epoch=epoch,
            times_s=[row["time_s"] for row in rows],
            states=[[row[n] for n in names] for row in rows],
            covariance_times_s=covariance_times or None,
            covariances=matrices or None,
            metadata={
                "review": receipt,
                "mass_kg_by_sample": [row["mass_kg"] for row in rows],
                "object_specs": config.get("objects", {}).get(object_id, {}).get("specs", {}),
                "seed_simulator": {
                    **{
                        key: config["simulator"][key]
                        for key in ("environment", "frames", "plugin_validation")
                        if key in config.get("simulator", {})
                    },
                    "dynamics": {"orbit": config.get("simulator", {}).get("dynamics", {}).get("orbit", {})},
                },
                "review_binding": {
                    "completed_run": str(Path(completed_run).expanduser().resolve()),
                    "sha256": identity["sha256"],
                },
                **({"planetary_config": config.get("simulator", {}).get("planetary", {})} if planetary else {}),
            },
        )
        attitude = None
        if include_attitude:
            attitude = AttitudeHistory(
                object_id,
                frame,
                "SC_BODY_1",
                epoch,
                trajectory.times_s,
                [[row[n] for n in ("quat_w", "quat_x", "quat_y", "quat_z")] for row in rows],
                [[row[n] for n in ("omega_x_rad_s", "omega_y_rad_s", "omega_z_rad_s")] for row in rows],
                {"review": receipt},
            )
        return ReviewSelection(trajectory, receipt, attitude)
