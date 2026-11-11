"""Entitled sequential OD over reduced ground and optical measurements."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from sim.analysis.rust_numeric import numeric_backend as validate_numeric_backend
from sim.analysis.rust_numeric import require_native
from sim.dynamics.orbit.accelerations import OrbitContext
from sim.dynamics.orbit.environment import EARTH_MU_KM3_S2
from sim.dynamics.orbit.frames import frame_context_from_mapping
from sim.dynamics.orbit.propagator import OrbitPropagator, j2_plugin
from sim.estimation.ground_station_od import (
    normalize_ground_station_measurements,
    predict_ground_station_measurement,
)
from sim.estimation.partitioning import JULIAN_DATE_BOUNDARY_TOLERANCE_S
from sim.estimation.sequential_od import (
    SequentialODConfig,
    SequentialProcessNoise,
    predict_sequential_state_covariance,
    run_sequential_orbit_od,
)
from sim.observation_contracts import ObservationPacket
from sim.optical_tracking import adapt_optical_ra_dec_observations, los_to_ra_dec
from sim.utils.io import SafeReadError, canonical_evidence_artifact_bytes, read_regular_file_nofollow

PROBLEM_SCHEMA = "oel.pro_tracking_od_problem.v1"
EVIDENCE_SCHEMA = "oel.pro_tracking_od_evidence.v1"
MANIFEST_SCHEMA = "oel.pro_tracking_od_manifest.v1"
_PARTITION_BOUNDARY_TOLERANCE_S = JULIAN_DATE_BOUNDARY_TOLERANCE_S
MAX_PROBLEM_BYTES = 16 * 1024 * 1024
MAX_MEASUREMENTS = 20_000
MAX_ARC_S = 30.0 * 86400.0
_MODELS = {"ground_topocentric", "optical_ra_dec"}


class ProTrackingOdError(ValueError):
    """Raised when a Pro tracking-OD problem or evidence packet is invalid."""


def _finite(value: Any, field_name: str) -> float:
    if isinstance(value, bool):
        raise ProTrackingOdError(f"{field_name} must be a finite number.")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ProTrackingOdError(f"{field_name} must be a finite number.") from exc
    if not math.isfinite(result):
        raise ProTrackingOdError(f"{field_name} must be a finite number.")
    return result


def _exact(raw: Mapping[str, Any], allowed: set[str], field_name: str) -> None:
    extra = sorted(set(raw) - allowed)
    if extra:
        raise ProTrackingOdError(f"Unknown {field_name} fields: {extra}.")


def _state(value: Any, field_name: str) -> tuple[float, ...]:
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ProTrackingOdError(
            f"{field_name} must contain one finite nonzero-position Cartesian state."
        ) from exc
    if array.shape != (6,) or np.any(~np.isfinite(array)) or np.linalg.norm(array[:3]) <= 0.0:
        raise ProTrackingOdError(f"{field_name} must contain one finite nonzero-position Cartesian state.")
    return tuple(float(item) for item in array)


def _covariance(value: Any, field_name: str) -> tuple[tuple[float, ...], ...]:
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ProTrackingOdError(f"{field_name} must be a finite 6x6 matrix.") from exc
    if array.shape != (6, 6) or np.any(~np.isfinite(array)):
        raise ProTrackingOdError(f"{field_name} must be a finite 6x6 matrix.")
    if not np.allclose(array, array.T, rtol=1.0e-10, atol=1.0e-14):
        raise ProTrackingOdError(f"{field_name} must be symmetric.")
    if float(np.min(np.linalg.eigvalsh(array))) < -1.0e-14:
        raise ProTrackingOdError(f"{field_name} must be positive semidefinite.")
    return tuple(tuple(float(item) for item in row) for row in array)


@dataclass(frozen=True)
class ProTrackingOdProblem:
    name: str
    object_id: str
    measurement_model: str
    measurements: tuple[Mapping[str, Any], ...]
    initial_jd_utc: float
    initial_state_eci_km_km_s: tuple[float, ...]
    initial_covariance_eci: tuple[tuple[float, ...], ...]
    fit_duration_s: float
    holdout_duration_s: float
    propagation_step_s: float
    j2: bool
    process_noise: SequentialProcessNoise = field(default_factory=SequentialProcessNoise)
    schema_version: str = PROBLEM_SCHEMA
    numeric_backend: str = "rust"

    def __post_init__(self) -> None:
        try:
            validate_numeric_backend(self.numeric_backend)
        except ValueError as exc:
            raise ProTrackingOdError(str(exc)) from exc

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> ProTrackingOdProblem:
        if not isinstance(value, Mapping):
            raise ProTrackingOdError("Pro tracking-OD problem must be a JSON object.")
        raw = dict(value)
        _exact(
            raw,
            {
                "schema_version", "name", "object_id", "measurement_model", "measurements",
                "initial_jd_utc", "initial_state_eci_km_km_s", "initial_covariance_eci",
                "fit_duration_s", "holdout_duration_s", "propagation", "filter",
            },
            "problem",
        )
        if raw.get("schema_version") != PROBLEM_SCHEMA:
            raise ProTrackingOdError(f"schema_version must be {PROBLEM_SCHEMA!r}.")
        model = str(raw.get("measurement_model", "")).strip().lower()
        if model not in _MODELS:
            raise ProTrackingOdError(f"measurement_model must be one of {sorted(_MODELS)}.")
        measurements = raw.get("measurements")
        if not isinstance(measurements, list) or not 4 <= len(measurements) <= MAX_MEASUREMENTS:
            raise ProTrackingOdError(f"measurements must contain between 4 and {MAX_MEASUREMENTS} rows.")
        if any(not isinstance(item, Mapping) for item in measurements):
            raise ProTrackingOdError("Every measurement row must be a JSON object.")
        propagation_value = raw.get("propagation") or {}
        filter_value = raw.get("filter") or {}
        if not isinstance(propagation_value, Mapping) or not isinstance(filter_value, Mapping):
            raise ProTrackingOdError("propagation and filter must be JSON objects.")
        propagation = dict(propagation_value)
        filter_raw = dict(filter_value)
        _exact(propagation, {"step_s", "j2", "numeric_backend"}, "propagation")
        _exact(filter_raw, {"process_noise_model", "spectral_density"}, "filter")
        if "j2" in propagation and not isinstance(propagation["j2"], bool):
            raise ProTrackingOdError("propagation.j2 must be a boolean.")
        fit = _finite(raw.get("fit_duration_s"), "fit_duration_s")
        holdout = _finite(raw.get("holdout_duration_s"), "holdout_duration_s")
        step = _finite(propagation.get("step_s"), "propagation.step_s")
        if fit <= 0.0 or holdout <= 0.0 or step <= 0.0 or fit + holdout > MAX_ARC_S:
            raise ProTrackingOdError("fit/holdout durations and propagation step must be positive and the arc must be bounded.")
        if step >= fit:
            raise ProTrackingOdError("propagation.step_s must be smaller than fit_duration_s.")
        try:
            process_noise = SequentialProcessNoise(
                model=str(filter_raw.get("process_noise_model", "white_acceleration")),
                spectral_density=tuple(filter_raw.get("spectral_density", (1.0e-14,) * 3)),
            )
        except (TypeError, ValueError) as exc:
            raise ProTrackingOdError(f"filter process noise is invalid: {exc}") from exc
        name = str(raw.get("name", "")).strip()
        object_id = str(raw.get("object_id", "")).strip()
        if not name or not object_id:
            raise ProTrackingOdError("name and object_id must be non-empty strings.")
        return cls(
            schema_version=PROBLEM_SCHEMA,
            name=name,
            object_id=object_id,
            measurement_model=model,
            measurements=tuple(dict(item) for item in measurements),
            initial_jd_utc=_finite(raw.get("initial_jd_utc"), "initial_jd_utc"),
            initial_state_eci_km_km_s=_state(raw.get("initial_state_eci_km_km_s"), "initial_state_eci_km_km_s"),
            initial_covariance_eci=_covariance(raw.get("initial_covariance_eci"), "initial_covariance_eci"),
            fit_duration_s=fit,
            holdout_duration_s=holdout,
            propagation_step_s=step,
            j2=bool(propagation.get("j2", False)),
            process_noise=process_noise,
            numeric_backend=propagation.get("numeric_backend", "rust"),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> ProTrackingOdProblem:
        try:
            content = read_regular_file_nofollow(path, min_bytes=1, max_bytes=MAX_PROBLEM_BYTES)
            value = json.loads(content.decode("utf-8"), parse_constant=lambda item: (_ for _ in ()).throw(ValueError(item)))
        except (SafeReadError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise ProTrackingOdError(f"Could not safely parse Pro tracking-OD problem: {exc}") from exc
        return cls.from_mapping(value)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "object_id": self.object_id,
            "measurement_model": self.measurement_model,
            "measurements": [dict(item) for item in self.measurements],
            "initial_jd_utc": self.initial_jd_utc,
            "initial_state_eci_km_km_s": list(self.initial_state_eci_km_km_s),
            "initial_covariance_eci": [list(row) for row in self.initial_covariance_eci],
            "fit_duration_s": self.fit_duration_s,
            "holdout_duration_s": self.holdout_duration_s,
            "propagation": {"step_s": self.propagation_step_s, "j2": self.j2,
                            **{"numeric_backend": self.numeric_backend}},
            "filter": {
                "process_noise_model": self.process_noise.model,
                "spectral_density": list(self.process_noise.spectral_density),
            },
        }


def validate_problem(value: ProTrackingOdProblem | Mapping[str, Any]) -> dict[str, Any]:
    problem = ProTrackingOdProblem.from_mapping(value.to_dict() if isinstance(value, ProTrackingOdProblem) else value)
    rows, _context = _normalized_rows(problem)
    fit, holdout = _partition(rows, problem)
    return {
        "status": "valid",
        "schema_version": problem.schema_version,
        "measurement_model": problem.measurement_model,
        "measurement_count": len(rows),
        "fit_count": len(fit),
        "holdout_count": len(holdout),
        "problem_sha256": _digest(problem.to_dict()),
    }


def run_pro_tracking_od(
    value: ProTrackingOdProblem | Mapping[str, Any], output_dir: str | Path
) -> dict[str, Any]:
    problem = ProTrackingOdProblem.from_mapping(value.to_dict() if isinstance(value, ProTrackingOdProblem) else value)
    if problem.numeric_backend == "rust":
        require_native("orbit_trial_batch_bytes")
    destination = Path(output_dir).expanduser()
    if destination.is_symlink() or destination.exists():
        raise ProTrackingOdError("output_dir must be a new non-symlink path.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.building-", dir=destination.parent))
    try:
        summary = _materialize_evidence(problem, temporary)
        _write_json(temporary / "pro_tracking_od_manifest.json", _build_manifest(temporary, summary))
        destination_abs = destination.absolute()
        temporary.replace(destination_abs)
        return {**summary, "output_dir": str(destination_abs), "manifest": str(destination_abs / "pro_tracking_od_manifest.json")}
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def replay_evidence(output_dir: str | Path) -> dict[str, Any]:
    root = Path(output_dir).absolute()
    manifest = _read_json(root / "pro_tracking_od_manifest.json")
    manifest_fields = {"schema_version", "status", "problem_sha256", "result_sha256", "files"}
    if set(manifest) != manifest_fields or manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise ProTrackingOdError("Pro tracking-OD manifest schema is invalid.")
    listed = manifest.get("files")
    if not isinstance(listed, list):
        raise ProTrackingOdError("Pro tracking-OD manifest files must be a JSON array.")
    expected = {str(row.get("path", "")) for row in listed if isinstance(row, Mapping)}
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file() and path.name != "pro_tracking_od_manifest.json"}
    if expected != actual or len(expected) != len(listed):
        raise ProTrackingOdError("Pro tracking-OD evidence inventory is incomplete or unexpected.")
    retained_content = _verified_artifact_content(root, listed)
    retained_canonical = _canonical_artifact_content(root)
    problem_payload = _parse_json_bytes(retained_content["normalized_problem.json"], "normalized_problem.json")
    problem = ProTrackingOdProblem.from_mapping(problem_payload)
    with tempfile.TemporaryDirectory(
        prefix="oel-pro-tracking-od-replay-", dir=Path(tempfile.gettempdir()).resolve()
    ) as temp:
        reproduced_root = Path(temp)
        reproduced = _materialize_evidence(problem, reproduced_root)
        reproduced_canonical = _canonical_artifact_content(reproduced_root)
        reproduced_manifest = _build_manifest(reproduced_root, reproduced)
    if reproduced_canonical != retained_canonical:
        raise ProTrackingOdError("Authoritative Pro tracking-OD replay does not reproduce every retained artifact.")
    if _manifest_claims(manifest) != _manifest_claims(reproduced_manifest):
        raise ProTrackingOdError("Pro tracking-OD manifest claims differ from authoritative replay.")
    if [row["path"] for row in listed] != [row["path"] for row in reproduced_manifest["files"]]:
        raise ProTrackingOdError("Pro tracking-OD manifest artifact inventory differs from authoritative replay.")
    return {"status": "verified", "problem_sha256": reproduced["problem_sha256"], "result_sha256": _digest(reproduced)}


def _materialize_evidence(problem: ProTrackingOdProblem, root: Path) -> dict[str, Any]:
    summary = _solve(problem, root)
    _write_json(root / "normalized_problem.json", problem.to_dict())
    _write_json(root / "pro_tracking_od_summary.json", summary)
    _write_holdout_csv(root / "holdout_predictions.csv", summary["holdout_predictions"])
    return summary


def _artifact_content(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): read_regular_file_nofollow(path, max_bytes=128 * 1024 * 1024)
        for path in sorted(item for item in root.rglob("*") if item.is_file() and item.name != "pro_tracking_od_manifest.json")
    }


def _canonical_artifact_content(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): canonical_evidence_artifact_bytes(
            path,
            root=root,
            max_bytes=128 * 1024 * 1024,
        )
        for path in sorted(
            item for item in root.rglob("*") if item.is_file() and item.name != "pro_tracking_od_manifest.json"
        )
    }


def _manifest_claims(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {name: value for name, value in manifest.items() if name != "files"}


def _artifact_receipts(root: Path) -> list[dict[str, Any]]:
    return [
        {"path": name, "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
        for name, content in _artifact_content(root).items()
    ]


def _build_manifest(root: Path, summary: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": MANIFEST_SCHEMA,
        "status": "complete",
        "problem_sha256": summary["problem_sha256"],
        "result_sha256": _digest(summary),
        "files": _artifact_receipts(root),
    }


def _verified_artifact_content(root: Path, listed: Sequence[Any]) -> dict[str, bytes]:
    content_by_name: dict[str, bytes] = {}
    for value in listed:
        if not isinstance(value, Mapping) or set(value) != {"path", "bytes", "sha256"}:
            raise ProTrackingOdError("Pro tracking-OD artifact receipt contract is invalid.")
        name = value["path"]
        size = value["bytes"]
        digest = value["sha256"]
        if not isinstance(name, str) or name in content_by_name:
            raise ProTrackingOdError("Pro tracking-OD artifact receipt path is invalid.")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ProTrackingOdError(f"Evidence receipt size is invalid for {name}.")
        if not isinstance(digest, str) or len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ProTrackingOdError(f"Evidence receipt digest is invalid for {name}.")
        content = read_regular_file_nofollow(root / name, max_bytes=128 * 1024 * 1024)
        if len(content) != size or hashlib.sha256(content).hexdigest() != digest:
            raise ProTrackingOdError(f"Evidence receipt mismatch for {name}.")
        content_by_name[name] = content
    return content_by_name


def _solve(problem: ProTrackingOdProblem, output: Path) -> dict[str, Any]:
    if problem.numeric_backend == "rust":
        require_native("orbit_trial_batch_bytes")
    rows, context = _normalized_rows(problem)
    fit_rows, holdout_rows = _partition(rows, problem)
    config = SequentialODConfig(
        estimator="ekf",
        numeric_backend=problem.numeric_backend,
        process_noise=problem.process_noise,
        integration_substep_s=problem.propagation_step_s,
        checkpoint_interval=max(len(fit_rows), 1),
        # The shared Cartesian reset assigns measurement components directly
        # into state indices and is not valid for nonlinear ground/optical
        # measurements.  Retain change-point screening without pretending that
        # a confirmed innovation supplies a replacement Cartesian state.
        reinitialize_after_maneuver=False,
    )
    propagator = OrbitPropagator(integrator="rk4", plugins=[j2_plugin] if problem.j2 else [],
                                 numeric_backend=problem.numeric_backend)
    orbit_context = OrbitContext(mu_km3_s2=EARTH_MU_KM3_S2, mass_kg=100.0)
    model = _measurement_model(problem.measurement_model, context, numeric_backend=problem.numeric_backend)
    packet = ObservationPacket({
        "schema_version": 1,
        "object_id": problem.object_id,
        "frame": "ECI",
        "measurement_model": problem.measurement_model,
        "observations": fit_rows,
    })
    report = run_sequential_orbit_od(
        packet,
        initial_state_eci_km_s=problem.initial_state_eci_km_km_s,
        initial_covariance=problem.initial_covariance_eci,
        config=config,
        propagator=propagator,
        orbit_context=orbit_context,
        output_dir=output / "filter",
        measurement_model=model,
        measurement_model_id=f"{problem.measurement_model}_v1",
    )
    terminal = report["records"][-1]
    times = [float(row["time_s"]) for row in holdout_rows]
    predictions = predict_sequential_state_covariance(
        terminal["filtered_state"],
        terminal["filtered_covariance"],
        initial_time_s=float(terminal["time_s"]),
        prediction_times_s=times,
        config=config,
        propagator=propagator,
        orbit_context=orbit_context,
    )
    holdout = []
    for row, prediction in zip(holdout_rows, predictions, strict=True):
        evaluation = model(row, np.asarray(prediction["state_eci_km_km_s"], dtype=float))
        h = np.asarray(evaluation["jacobian"], dtype=float)
        covariance = np.asarray(evaluation["covariance"], dtype=float)
        state_covariance = np.asarray(prediction["covariance_eci_km_km_s"], dtype=float)
        innovation_covariance = h @ state_covariance @ h.T + covariance
        innovation = np.asarray(evaluation["innovation"], dtype=float)
        nis = float(innovation.T @ np.linalg.pinv(innovation_covariance) @ innovation)
        holdout.append({
            "measurement_id": str(row.get("measurement_id", "")),
            "time_s": float(row["time_s"]),
            "observed": list(evaluation["observed"]),
            "predicted": list(evaluation["predicted"]),
            "innovation": innovation.tolist(),
            "nis": nis,
            "state_eci_km_km_s": prediction["state_eci_km_km_s"],
            "covariance_eci_km_km_s": prediction["covariance_eci_km_km_s"],
        })
    return {
        "schema_version": EVIDENCE_SCHEMA,
        "status": "complete",
        "problem_sha256": _digest(problem.to_dict()),
        "measurement_model": problem.measurement_model,
        "fit_observation_count": len(fit_rows),
        "holdout_observation_count": len(holdout_rows),
        "filter_summary": {
            "accepted_count": report["accepted_count"],
            "rejected_count": report["rejected_count"],
            "covariance_checks": report["covariance_checks"],
            "innovation_statistics": report["innovation_statistics"],
            "maneuver_recovery": report["maneuver_recovery"],
            "quality_gates": report["quality_gates"],
        },
        "holdout_statistics": {
            "nis_mean": float(np.mean([row["nis"] for row in holdout])),
            "nis_max": float(np.max([row["nis"] for row in holdout])),
            "finite": bool(all(math.isfinite(row["nis"]) for row in holdout)),
        },
        "holdout_predictions": holdout,
        "claim_posture": {
            "calibrated_covariance": False,
            "predicted_accuracy": "holdout residual evidence only",
            "maneuver_estimation": "innovation change-point screening only; no nonlinear state reinitialization or attribution",
            "doppler_semantics": "reduced geometric range-rate; no raw count, frequency, media, or relativity reduction",
        },
    }


def _normalized_rows(problem: ProTrackingOdProblem) -> tuple[list[dict[str, Any]], Any]:
    context = frame_context_from_mapping({}, jd_utc_start=problem.initial_jd_utc, source="pro_tracking_od")
    if problem.measurement_model == "ground_topocentric":
        rows = normalize_ground_station_measurements(problem.measurements)
        first_jd_utc = float(rows[0]["time_jd_utc"])
        if abs((first_jd_utc - problem.initial_jd_utc) * 86400.0) > 1.0e-3:
            raise ProTrackingOdError(
                "initial_jd_utc must match the first ground measurement epoch because the initial state is defined there."
            )
    else:
        rows = adapt_optical_ra_dec_observations(
            problem.measurements,
            object_id=problem.object_id,
            initial_jd_utc=problem.initial_jd_utc,
            frame_context=context,
        )["observations"]
    times = np.asarray([float(row["time_s"]) for row in rows], dtype=float)
    if np.any(np.diff(times) <= 0.0):
        raise ProTrackingOdError("The v1 Pro sequential workflow requires strictly increasing measurement epochs.")
    if abs(float(times[0])) > 1.0e-3:
        raise ProTrackingOdError(
            "The first measurement must be at time_s=0 because the initial state is defined at the first observation epoch."
        )
    return rows, context


def _partition(rows: Sequence[Mapping[str, Any]], problem: ProTrackingOdProblem) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    tolerance_s = _PARTITION_BOUNDARY_TOLERANCE_S
    fit = [dict(row) for row in rows if float(row["time_s"]) <= problem.fit_duration_s + tolerance_s]
    holdout = [
        dict(row)
        for row in rows
        if float(row["time_s"]) > problem.fit_duration_s + tolerance_s
        and float(row["time_s"]) <= problem.fit_duration_s + problem.holdout_duration_s + tolerance_s
    ]
    if len(fit) < 3 or not holdout:
        raise ProTrackingOdError("The declared windows must contain at least three fit rows and one untouched holdout row.")
    if len(fit) + len(holdout) != len(rows):
        raise ProTrackingOdError("Every measurement must fall inside the declared fit and holdout windows.")
    return fit, holdout


def _measurement_model(model: str, context: Any, *, numeric_backend: str = "python"):
    def evaluate(row: Mapping[str, Any], state: np.ndarray) -> dict[str, Any]:
        observed, predicted, covariance = _measurement_vectors(model, row, state, context)
        angular_indices = (
            {0}
            if model == "optical_ra_dec"
            else {index for index, name in enumerate(row["components"]) if name == "azimuth_deg"}
        )
        steps = np.asarray([1.0e-3] * 3 + [1.0e-6] * 3)
        jacobian = np.zeros((observed.size, 6))
        native_predictions = None
        if numeric_backend == "rust" and model == "ground_topocentric":
            trials = np.vstack((state, *[trial for axis, step in enumerate(steps)
                for trial in (state + np.eye(6)[axis] * step, state - np.eye(6)[axis] * step)]))
            from sim.estimation.ground_station_od import _native_ground_station_predictions
            predictions = _native_ground_station_predictions([row] * len(trials), trials,
                epoch_jd_utc=float(row["time_jd_utc"]) - float(row["time_s"]) / 86400.,
                frame_context=context, geometry_cache={})
            if predictions is not None:
                native_predictions = np.asarray([[prediction[name] for name in row["components"]]
                                                 for prediction in predictions])
            if native_predictions is not None:
                predicted = native_predictions[0]
        for axis, step in enumerate(steps):
            plus = state.copy()
            plus[axis] += step
            minus = state.copy()
            minus[axis] -= step
            if native_predictions is None:
                _o, high, _c = _measurement_vectors(model, row, plus, context)
                _o, low, _c = _measurement_vectors(model, row, minus, context)
            else:
                high, low = native_predictions[1 + 2 * axis:3 + 2 * axis]
            jacobian[:, axis] = _angular_difference(high, low, angular_indices) / (2.0 * step)
        innovation = _angular_difference(observed, predicted, angular_indices)
        return {"observed": observed.tolist(), "predicted": predicted.tolist(), "jacobian": jacobian.tolist(), "covariance": covariance.tolist(), "innovation": innovation.tolist()}
    return evaluate


def _measurement_vectors(model: str, row: Mapping[str, Any], state: np.ndarray, context: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if model == "ground_topocentric":
        components = list(row["components"])
        prediction = predict_ground_station_measurement(
            target_state_eci=state,
            station=row["station"],
            t_s=float(row["time_s"]),
            jd_utc_start=float(row["time_jd_utc"]) - float(row["time_s"]) / 86400.0,
            frame_context=context,
        )
        observed = np.asarray([row["values"][name] for name in components], dtype=float)
        predicted = np.asarray([prediction[name] for name in components], dtype=float)
        covariance = np.asarray(row["uncertainty"]["matrix"], dtype=float)
        return observed, predicted, covariance
    observed = np.asarray([row["ra_deg"], row["dec_deg"]], dtype=float)
    predicted = np.asarray(los_to_ra_dec(state[:3] - np.asarray(row["observer_eci_km"], dtype=float)), dtype=float)
    covariance = np.asarray(row["angular_covariance_deg2"], dtype=float)
    return observed, predicted, covariance


def _angular_difference(left: np.ndarray, right: np.ndarray, angular_indices: set[int]) -> np.ndarray:
    result = np.asarray(left, dtype=float) - np.asarray(right, dtype=float)
    for index in angular_indices:
        result[index] = (result[index] + 180.0) % 360.0 - 180.0
    return result


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(value), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        content = read_regular_file_nofollow(path, min_bytes=1, max_bytes=128 * 1024 * 1024)
    except (SafeReadError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProTrackingOdError(f"Could not safely read {path.name}: {exc}") from exc
    return _parse_json_bytes(content, path.name)


def _parse_json_bytes(content: bytes, field: str) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ProTrackingOdError(f"{field} contains forbidden non-finite constant {value}.")

    def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in values:
            if key in result:
                raise ProTrackingOdError(f"{field} contains duplicate field {key!r}.")
            result[key] = value
        return result

    try:
        value = json.loads(content.decode("utf-8"), parse_constant=reject_constant, object_pairs_hook=pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProTrackingOdError(f"Could not safely parse {field}: {exc}") from exc
    if not isinstance(value, dict):
        raise ProTrackingOdError(f"{field} must contain a JSON object.")
    return value


def _write_holdout_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("measurement_id", "time_s", "nis", "innovation_json"))
        writer.writeheader()
        for row in rows:
            writer.writerow({"measurement_id": row["measurement_id"], "time_s": row["time_s"], "nis": row["nis"], "innovation_json": json.dumps(row["innovation"], separators=(",", ":"))})


__all__ = [
    "EVIDENCE_SCHEMA", "MANIFEST_SCHEMA", "PROBLEM_SCHEMA", "ProTrackingOdError",
    "ProTrackingOdProblem", "replay_evidence", "run_pro_tracking_od", "validate_problem",
]
