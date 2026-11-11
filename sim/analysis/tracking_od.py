"""Bounded CCSDS TDM to fit/holdout orbit-determination evidence workflow."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from sim.dynamics.orbit.frames import (
    FRAME_MODEL_SIMPLE_GMST,
    FrameContext,
    frame_context_from_mapping,
    normalize_frame_model,
)
from sim.estimation.ground_station_od import solve_ground_station_measurement_od
from sim.frame_time import TimeScale, format_epoch, parse_epoch
from sim.interchange.ccsds_tdm import CCSDS_TDM_PROFILE, TdmMessage, serialize_tdm_kvn, validate_tdm
from sim.tracking_data import NORMALIZED_TRACKING_DATASET_SCHEMA, normalize_tdm_tracking_dataset

TRACKING_OD_PROBLEM_SCHEMA = "oel.tracking_od_problem.v1"
TRACKING_OD_EVIDENCE_SCHEMA = "oel.tracking_od_evidence.v1"
MAX_TRACKING_OD_ARC_S = 7.0 * 86400.0


class TrackingOdError(ValueError):
    """Raised when a public tracking-OD problem is invalid."""


def _reject_unknown_keys(data: Mapping[str, Any], allowed: set[str], *, field: str) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise TrackingOdError(f"{field} contains unknown fields {unknown}.")


def _required_object(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TrackingOdError(f"{field} must be a JSON object.")
    return dict(value)


def _required_string(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TrackingOdError(f"{field} must be a non-empty string.")
    return value.strip()


def _required_number(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TrackingOdError(f"{field} must be a JSON number.")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise TrackingOdError(f"{field} must be finite.")
    return parsed


def _required_integer(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TrackingOdError(f"{field} must be a JSON integer.")
    return int(value)


def _required_boolean(value: Any, *, field: str) -> bool:
    if not isinstance(value, bool):
        raise TrackingOdError(f"{field} must be a JSON boolean.")
    return value


@dataclass(frozen=True)
class TrackingStation:
    station_id: str
    latitude_deg: float
    longitude_deg: float
    altitude_km: float

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> TrackingStation:
        raw = _required_object(data, field="stations[]")
        _reject_unknown_keys(
            raw,
            {"station_id", "latitude_deg", "longitude_deg", "altitude_km"},
            field="stations[]",
        )
        item = cls(
            station_id=_required_string(raw.get("station_id"), field="stations[].station_id"),
            latitude_deg=_required_number(raw.get("latitude_deg"), field="stations[].latitude_deg"),
            longitude_deg=_required_number(raw.get("longitude_deg"), field="stations[].longitude_deg"),
            altitude_km=_required_number(raw.get("altitude_km", 0.0), field="stations[].altitude_km"),
        )
        if not item.station_id:
            raise TrackingOdError("Every station requires station_id.")
        if not all(math.isfinite(value) for value in (item.latitude_deg, item.longitude_deg, item.altitude_km)):
            raise TrackingOdError(f"Station {item.station_id!r} coordinates must be finite.")
        if not -90.0 <= item.latitude_deg <= 90.0 or not -180.0 <= item.longitude_deg <= 180.0:
            raise TrackingOdError(f"Station {item.station_id!r} latitude/longitude is outside WGS84 bounds.")
        return item

    def native_mapping(self) -> dict[str, Any]:
        return {
            "id": self.station_id,
            "lat_deg": self.latitude_deg,
            "lon_deg": self.longitude_deg,
            "alt_km": self.altitude_km,
        }


@dataclass(frozen=True)
class TrackingOdProblem:
    name: str
    object_id: str
    measurement_semantics: str
    stations: tuple[TrackingStation, ...]
    initial_state_eci_km_km_s: tuple[float, float, float, float, float, float]
    initial_state_epoch_utc: str
    initial_state_frame: str
    frame_model: str
    angle_sigma_deg: float
    range_sigma_km: float
    fit_duration_s: float
    holdout_duration_s: float
    integration_step_s: float
    dynamics_model: str
    j2: bool
    max_nfev: int
    robust_loss: str
    robust_f_scale: float
    sigma_clip_threshold: float | None
    schema_version: str = TRACKING_OD_PROBLEM_SCHEMA

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> TrackingOdProblem:
        raw = _required_object(data, field="tracking-OD problem")
        _reject_unknown_keys(
            raw,
            {
                "schema_version",
                "name",
                "object_id",
                "measurement_semantics",
                "stations",
                "initial_state_eci_km_km_s",
                "initial_state_epoch_utc",
                "initial_state_frame",
                "frame_model",
                "uncertainties",
                "fit_duration_s",
                "holdout_duration_s",
                "propagation",
                "solver",
            },
            field="tracking-OD problem",
        )
        schema = _required_string(
            raw.get("schema_version", TRACKING_OD_PROBLEM_SCHEMA), field="schema_version"
        )
        if schema != TRACKING_OD_PROBLEM_SCHEMA:
            raise TrackingOdError(f"Unsupported tracking-OD schema {schema!r}.")
        name = _required_string(raw.get("name", "tracking_od"), field="name")
        object_id = _required_string(raw.get("object_id"), field="object_id")
        semantics = _required_string(raw.get("measurement_semantics"), field="measurement_semantics")
        if semantics != "reduced_geometric":
            raise TrackingOdError(
                "measurement_semantics must explicitly be 'reduced_geometric'; raw radiometric TDM is unsupported."
            )
        raw_stations = raw.get("stations")
        if not isinstance(raw_stations, list):
            raise TrackingOdError("stations must be a JSON array.")
        stations = tuple(TrackingStation.from_mapping(item) for item in raw_stations)
        if not stations:
            raise TrackingOdError("At least one station is required.")
        identifiers = [item.station_id for item in stations]
        if len(set(identifiers)) != len(identifiers):
            raise TrackingOdError("Station identifiers must be unique.")
        raw_state = raw.get("initial_state_eci_km_km_s")
        if not isinstance(raw_state, list) or any(
            isinstance(value, bool) or not isinstance(value, (int, float)) for value in raw_state
        ):
            raise TrackingOdError("initial_state_eci_km_km_s must be a JSON array of numbers.")
        state = np.asarray(raw_state, dtype=float)
        if state.shape != (6,) or not np.all(np.isfinite(state)) or float(np.linalg.norm(state[:3])) <= 0.0:
            raise TrackingOdError("initial_state_eci_km_km_s must contain six finite values and nonzero position.")
        initial_epoch_raw = _required_string(raw.get("initial_state_epoch_utc"), field="initial_state_epoch_utc")
        try:
            initial_epoch = parse_epoch(initial_epoch_raw, TimeScale.UTC)
        except ValueError as exc:
            raise TrackingOdError("initial_state_epoch_utc must be a valid UTC calendar or ordinal epoch.") from exc
        initial_epoch_utc = format_epoch(initial_epoch, TimeScale.UTC, include_z=True)
        initial_state_frame = _required_string(raw.get("initial_state_frame"), field="initial_state_frame").upper()
        if initial_state_frame != "ECI":
            raise TrackingOdError("The v1 tracking-OD problem requires initial_state_frame = 'ECI'.")
        try:
            frame_model = normalize_frame_model(
                _required_string(raw.get("frame_model"), field="frame_model")
            )
        except ValueError as exc:
            raise TrackingOdError("The v1 tracking-OD problem requires frame_model = 'simple_gmst'.") from exc
        if frame_model != FRAME_MODEL_SIMPLE_GMST:
            raise TrackingOdError("The v1 tracking-OD problem requires frame_model = 'simple_gmst'.")
        uncertainties = _required_object(raw.get("uncertainties"), field="uncertainties")
        solver = _required_object(raw.get("solver"), field="solver")
        propagation = _required_object(raw.get("propagation"), field="propagation")
        _reject_unknown_keys(uncertainties, {"angle_sigma_deg", "range_sigma_km"}, field="uncertainties")
        _reject_unknown_keys(
            solver,
            {"max_nfev", "robust_loss", "robust_f_scale", "sigma_clip_threshold"},
            field="solver",
        )
        _reject_unknown_keys(propagation, {"dynamics_model", "step_s", "j2"}, field="propagation")
        angle_sigma = _required_number(uncertainties.get("angle_sigma_deg"), field="uncertainties.angle_sigma_deg")
        range_sigma = _required_number(uncertainties.get("range_sigma_km"), field="uncertainties.range_sigma_km")
        fit_duration = _required_number(raw.get("fit_duration_s"), field="fit_duration_s")
        holdout_duration = _required_number(raw.get("holdout_duration_s"), field="holdout_duration_s")
        step = _required_number(propagation.get("step_s"), field="propagation.step_s")
        numeric_positive = {
            "uncertainties.angle_sigma_deg": angle_sigma,
            "uncertainties.range_sigma_km": range_sigma,
            "fit_duration_s": fit_duration,
            "holdout_duration_s": holdout_duration,
            "propagation.step_s": step,
        }
        for field, value in numeric_positive.items():
            if not math.isfinite(value) or value <= 0.0:
                raise TrackingOdError(f"{field} must be positive and finite.")
        if fit_duration + holdout_duration > MAX_TRACKING_OD_ARC_S:
            raise TrackingOdError(f"The public tracking-OD arc must not exceed {MAX_TRACKING_OD_ARC_S} seconds.")
        if step >= fit_duration:
            raise TrackingOdError("propagation.step_s must be smaller than fit_duration_s.")
        dynamics_model = _required_string(propagation.get("dynamics_model"), field="propagation.dynamics_model").lower()
        if dynamics_model != "two_body":
            raise TrackingOdError("The v1 tracking-OD problem supports dynamics_model = two_body only.")
        max_nfev = _required_integer(solver.get("max_nfev"), field="solver.max_nfev")
        robust_loss = _required_string(solver.get("robust_loss"), field="solver.robust_loss").lower()
        robust_f_scale = _required_number(solver.get("robust_f_scale"), field="solver.robust_f_scale")
        sigma_clip_raw = solver.get("sigma_clip_threshold")
        sigma_clip = (
            None
            if sigma_clip_raw is None
            else _required_number(sigma_clip_raw, field="solver.sigma_clip_threshold")
        )
        j2 = _required_boolean(propagation.get("j2"), field="propagation.j2")
        if not 1 <= max_nfev <= 200:
            raise TrackingOdError("solver.max_nfev must lie within [1, 200].")
        if robust_loss not in {"linear", "soft_l1", "huber", "cauchy", "arctan"}:
            raise TrackingOdError("solver.robust_loss is unsupported.")
        if not math.isfinite(robust_f_scale) or robust_f_scale <= 0.0:
            raise TrackingOdError("solver.robust_f_scale must be positive and finite.")
        if sigma_clip is not None and (not math.isfinite(sigma_clip) or sigma_clip <= 0.0):
            raise TrackingOdError("solver.sigma_clip_threshold must be positive and finite when supplied.")
        return cls(
            schema_version=schema,
            name=name,
            object_id=object_id,
            measurement_semantics=semantics,
            stations=stations,
            initial_state_eci_km_km_s=tuple(float(value) for value in state),
            initial_state_epoch_utc=initial_epoch_utc,
            initial_state_frame=initial_state_frame,
            frame_model=frame_model,
            angle_sigma_deg=angle_sigma,
            range_sigma_km=range_sigma,
            fit_duration_s=fit_duration,
            holdout_duration_s=holdout_duration,
            integration_step_s=step,
            dynamics_model=dynamics_model,
            j2=j2,
            max_nfev=max_nfev,
            robust_loss=robust_loss,
            robust_f_scale=robust_f_scale,
            sigma_clip_threshold=sigma_clip,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "object_id": self.object_id,
            "measurement_semantics": self.measurement_semantics,
            "stations": [asdict(item) for item in self.stations],
            "initial_state_eci_km_km_s": list(self.initial_state_eci_km_km_s),
            "initial_state_epoch_utc": self.initial_state_epoch_utc,
            "initial_state_frame": self.initial_state_frame,
            "frame_model": self.frame_model,
            "uncertainties": {
                "angle_sigma_deg": self.angle_sigma_deg,
                "range_sigma_km": self.range_sigma_km,
            },
            "fit_duration_s": self.fit_duration_s,
            "holdout_duration_s": self.holdout_duration_s,
            "propagation": {
                "dynamics_model": self.dynamics_model,
                "step_s": self.integration_step_s,
                "j2": self.j2,
            },
            "solver": {
                "max_nfev": self.max_nfev,
                "robust_loss": self.robust_loss,
                "robust_f_scale": self.robust_f_scale,
                "sigma_clip_threshold": self.sigma_clip_threshold,
            },
        }


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(dict(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_write_text(
        path,
        json.dumps(dict(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
    )


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        stream.write(text)
        stream.flush()
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _file_receipt(path: Path, *, relative_to: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {
        "path": str(path.relative_to(relative_to)),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def _read_prediction_ledger(path: Path) -> list[dict[str, Any]]:
    numeric = {
        "time_jd_utc",
        "time_s",
        "observed",
        "predicted",
        "residual",
        "sigma",
        "normalized_residual",
        "whitened_residual",
    }
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8", newline="") as stream:
        for raw in csv.DictReader(stream):
            if raw.get("partition") != "holdout" or raw.get("residual_kind") != "holdout":
                continue
            row: dict[str, Any] = {
                key: float(value) if key in numeric and value != "" else value
                for key, value in raw.items()
                if key
                in {
                    "measurement_id",
                    "station_id",
                    "partition",
                    "time_jd_utc",
                    "time_s",
                    "component",
                    "observed",
                    "predicted",
                    "residual",
                    "sigma",
                    "normalized_residual",
                    "whitened_residual",
                }
            }
            rows.append(row)
    return rows


def assess_tdm_orbit_determination(
    message: TdmMessage,
    problem: TrackingOdProblem | Mapping[str, Any],
    *,
    output_dir: str | Path,
) -> dict[str, Any]:
    parsed = TrackingOdProblem.from_mapping(problem.to_dict() if isinstance(problem, TrackingOdProblem) else problem)
    validate_tdm(message)
    normalized_problem = parsed.to_dict()
    problem_sha256 = _canonical_sha256(normalized_problem)
    canonical_tdm = serialize_tdm_kvn(message)
    dataset = normalize_tdm_tracking_dataset(
        message,
        stations=[item.native_mapping() for item in parsed.stations],
        measurement_semantics=parsed.measurement_semantics,
        angle_sigma_deg=parsed.angle_sigma_deg,
        range_sigma_km=parsed.range_sigma_km,
        expected_object_id=parsed.object_id,
    )
    dataset_span_s = float(dataset["last_epoch_tai_seconds"]) - float(dataset["first_epoch_tai_seconds"])
    requested_span_s = parsed.fit_duration_s + parsed.holdout_duration_s
    if requested_span_s > dataset_span_s:
        raise TrackingOdError(
            f"fit_duration_s + holdout_duration_s ({requested_span_s}) exceeds the TDM data span ({dataset_span_s})."
        )
    initial_state_epoch = parse_epoch(parsed.initial_state_epoch_utc, TimeScale.UTC)
    if float(initial_state_epoch.tai_seconds) != float(dataset["first_epoch_tai_seconds"]):
        raise TrackingOdError(
            "initial_state_epoch_utc must exactly match the first retained TDM measurement epoch."
        )
    frame_context = frame_context_from_mapping(
        {"model": parsed.frame_model},
        jd_utc_start=float(dataset["first_epoch_jd_utc"]),
        source="tracking_od_problem",
    )
    # This standalone public API is a trusted local/worker path-facing route,
    # not the MCP descriptor-bound writer boundary.  Keep symlinked workspace
    # aliases in parent components usable, but never treat a requested output
    # directory that is itself a symlink as a fresh evidence root.  Managed MCP
    # callers must wrap the producer in SecureWriteRoot/staged_output.
    requested_output = Path(output_dir).expanduser()
    if requested_output.is_symlink():
        raise TrackingOdError("output_dir must not be a symlink.")
    output_root = requested_output.resolve()
    output_existed = output_root.exists()
    if output_existed and not output_root.is_dir():
        raise TrackingOdError(f"output_dir must be a directory path: {output_root}.")
    if output_existed and any(output_root.iterdir()):
        raise TrackingOdError(f"output_dir must be absent or empty; refusing to mix evidence in {output_root}.")
    output_root.mkdir(parents=True, exist_ok=True)
    try:
        return _write_tracking_od_assessment(
            parsed=parsed,
            canonical_tdm=canonical_tdm,
            dataset=dataset,
            problem_sha256=problem_sha256,
            frame_context=frame_context,
            output_root=output_root,
        )
    except Exception:
        shutil.rmtree(output_root, ignore_errors=True)
        if output_existed:
            output_root.mkdir(parents=True, exist_ok=True)
        raise


def _write_tracking_od_assessment(
    *,
    parsed: TrackingOdProblem,
    canonical_tdm: str,
    dataset: Mapping[str, Any],
    problem_sha256: str,
    frame_context: FrameContext,
    output_root: Path,
) -> dict[str, Any]:
    canonical_tdm_path = output_root / "canonical_input.tdm"
    _atomic_write_text(canonical_tdm_path, canonical_tdm)
    normalized_dataset_path = output_root / "normalized_tracking_dataset.json"
    _write_json(normalized_dataset_path, dataset)
    estimator_dir = output_root / "estimator"
    report = solve_ground_station_measurement_od(
        dataset["measurement_rows"],
        object_id=parsed.object_id,
        output_dir=estimator_dir,
        initial_state_eci_km_s=parsed.initial_state_eci_km_km_s,
        initial_state_source=f"tracking_od_problem:{problem_sha256}",
        fit_duration_s=parsed.fit_duration_s,
        holdout_duration_s=parsed.holdout_duration_s,
        partition_boundary_tolerance_s=0.0,
        dt_s=parsed.integration_step_s,
        dynamics_model=parsed.dynamics_model,
        j2=parsed.j2,
        max_nfev=parsed.max_nfev,
        robust_loss=parsed.robust_loss,
        robust_f_scale=parsed.robust_f_scale,
        sigma_clip_threshold=parsed.sigma_clip_threshold,
        frame_context=frame_context,
        scenario_name=parsed.name,
    )
    residual_path = Path(str(report["residual_csv_path"])).resolve()
    holdout_prediction_ledger = _read_prediction_ledger(residual_path)
    if not holdout_prediction_ledger:
        raise TrackingOdError("The requested OD workflow produced no holdout prediction rows.")
    artifact_paths = {
        "canonical_tdm": canonical_tdm_path,
        "normalized_dataset": normalized_dataset_path,
        "estimator_report": Path(str(report["report_json_path"])).resolve(),
        "estimator_report_markdown": Path(str(report["report_md_path"])).resolve(),
        "residual_csv": residual_path,
        "residual_plot": Path(str(report["residual_plot_path"])).resolve(),
        "fitted_state_packet": Path(str(report["fitted_mission_input_packet_path"])).resolve(),
        "review_database": estimator_dir / "review" / "run.sqlite",
    }
    evidence = {
        "schema_version": TRACKING_OD_EVIDENCE_SCHEMA,
        "status": "completed",
        "problem_name": parsed.name,
        "problem_sha256": problem_sha256,
        "input": {
            "tdm_profile": CCSDS_TDM_PROFILE,
            "tdm_source_sha256": dataset["source_tdm_sha256"],
            "normalized_dataset_schema": NORMALIZED_TRACKING_DATASET_SCHEMA,
            "normalized_dataset_sha256": dataset["dataset_sha256"],
            "object_id": dataset["object_id"],
            "station_ids": dataset["station_ids"],
            "measurement_epoch_count": dataset["measurement_epoch_count"],
            "observable_record_count": dataset["observable_record_count"],
            "measurement_semantics": parsed.measurement_semantics,
            "initial_state_epoch_utc": parsed.initial_state_epoch_utc,
            "initial_state_frame": parsed.initial_state_frame,
            "frame_model": parsed.frame_model,
            "frame_provenance": report["frame_provenance"],
        },
        "partition": report["observation_partition"],
        "estimator": {
            "method": report["method"],
            "initial_state_source": report["initial_state_source"],
            "state_epoch_utc": parsed.initial_state_epoch_utc,
            "state_epoch_jd_utc": report["epoch_jd_utc"],
            "state_frame": parsed.initial_state_frame,
            "frame_provenance": report["frame_provenance"],
            "initial_state_eci_km_km_s": report["initial_state_eci_km_s"],
            "fitted_state_eci_km_km_s": report["fitted_state_eci_km_s"],
            "state_covariance_eci": report["state_covariance_eci_km_s"],
            "state_covariance_order": ["x_km", "y_km", "z_km", "vx_km_s", "vy_km_s", "vz_km_s"],
            "solver": report["solver"],
            "prefit_metrics": report["prefit_metrics"],
            "fit_metrics": report["fit_metrics"],
            "holdout_metrics": report["holdout_metrics"],
            "quality_gates": report["quality_gates"],
            "verdict": report["verdict"],
        },
        "authoritative_holdout_prediction": {
            "method": "fresh_oel_dynamics_repropagation_at_retained_holdout_epochs",
            "epoch_evaluation": report["epoch_evaluation"],
            "component_row_count": len(holdout_prediction_ledger),
            "components": sorted({str(row["component"]) for row in holdout_prediction_ledger}),
            "weighted_rms": report["holdout_metrics"].get("weighted_rms"),
            "prediction_ledger": holdout_prediction_ledger,
            "claim_boundary": (
                "Holdout residuals measure prediction against withheld observables; without independent truth they are not state-error truth."
            ),
        },
        "artifacts": {name: _file_receipt(path, relative_to=output_root) for name, path in artifact_paths.items()},
        "limitations": [
            "The v1 workflow supports one object, WGS84 stations, UTC, AZEL angles, and unambiguous one-way range in kilometers.",
            "Input observables must be explicitly declared reduced geometric; OEL does not apply TDM light-time, media, transponder, or calibration corrections.",
            "Doppler/frequency/phase, TDM XML, RADEC/XEYN/XSYE, ambiguous range, multi-way range, association, and custody are rejected.",
            "The bounded public estimator supports two-body dynamics with optional J2 and is not calibrated operational OD.",
            "Holdout prediction is primary evidence, but measurement-space residuals alone do not establish state truth or predicted orbit accuracy.",
        ],
    }
    evidence_path = output_root / "tracking_od_evidence.json"
    _write_json(evidence_path, evidence)
    return {**evidence, "evidence_path": str(evidence_path)}


__all__ = [
    "TRACKING_OD_EVIDENCE_SCHEMA",
    "TRACKING_OD_PROBLEM_SCHEMA",
    "TrackingOdError",
    "TrackingOdProblem",
    "TrackingStation",
    "assess_tdm_orbit_determination",
]
