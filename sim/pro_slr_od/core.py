"""Product contract for bounded real-data ILRS/SLR orbit determination.

This module is orchestration only. CRD interpretation and two-way light time
remain owned by :mod:`sim.slr_tracking`; batch estimation remains owned by
:mod:`sim.estimation.slr_od`; and sequential filtering/smoothing remains owned
by :mod:`sim.estimation.sequential_od`.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from sim.dynamics.orbit.accelerations import OrbitContext
from sim.dynamics.orbit.environment import EARTH_MU_KM3_S2
from sim.dynamics.orbit.propagator import OrbitPropagator, j2_plugin
from sim.dynamics.orbit.tle import TLEElements, parse_tle_lines
from sim.estimation.partitioning import JULIAN_DATE_BOUNDARY_TOLERANCE_S
from sim.estimation.sequential_od import (
    SequentialODConfig,
    SequentialProcessNoise,
    predict_sequential_state_covariance,
    run_sequential_orbit_od,
)
from sim.estimation.slr_od import (
    _range_metrics,
    _slr_residual_records,
    fit_ogp_mean_elements_to_slr_normal_points,
)
from sim.observation_contracts import ObservationPacket
from sim.slr_tracking import (
    C_LIGHT_KM_S,
    SLR_TWO_WAY_CONTRACT,
    adapt_ilrs_crd_normal_points,
    parse_ilrs_crd_v2_normal_points,
    predict_slr_two_way_time_of_flight_s,
)
from sim.utils.io import SafeReadError, canonical_evidence_artifact_bytes, read_regular_file_nofollow

PROBLEM_SCHEMA = "oel.pro_slr_od_problem.v1"
EVIDENCE_SCHEMA = "oel.pro_slr_od_evidence.v1"
MANIFEST_SCHEMA = "oel.pro_slr_od_manifest.v1"
_PARTITION_BOUNDARY_TOLERANCE_S = JULIAN_DATE_BOUNDARY_TOLERANCE_S
MAX_PROBLEM_BYTES = 64 * 1024 * 1024
MAX_CRD_BYTES = 64 * 1024 * 1024
MAX_NORMAL_POINTS = 50_000
MAX_ARC_S = 30.0 * 86400.0


class ProSlrOdError(ValueError):
    """Raised when an SLR OD contract or evidence packet is invalid."""


def _exact(raw: Mapping[str, Any], allowed: set[str], field: str) -> None:
    extra = sorted(set(raw) - allowed)
    if extra:
        raise ProSlrOdError(f"Unknown {field} fields: {extra}.")


def _finite(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise ProSlrOdError(f"{field} must be a finite number.")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ProSlrOdError(f"{field} must be a finite number.") from exc
    if not math.isfinite(result):
        raise ProSlrOdError(f"{field} must be a finite number.")
    return result


def _covariance(value: Any) -> tuple[tuple[float, ...], ...]:
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ProSlrOdError("sequential.initial_covariance_eci must be a finite 6x6 matrix.") from exc
    if array.shape != (6, 6) or np.any(~np.isfinite(array)):
        raise ProSlrOdError("sequential.initial_covariance_eci must be a finite 6x6 matrix.")
    if not np.allclose(array, array.T, rtol=1.0e-10, atol=1.0e-14):
        raise ProSlrOdError("sequential.initial_covariance_eci must be symmetric.")
    if float(np.min(np.linalg.eigvalsh(array))) < -1.0e-14:
        raise ProSlrOdError("sequential.initial_covariance_eci must be positive semidefinite.")
    return tuple(tuple(float(item) for item in row) for row in array)


@dataclass(frozen=True)
class ProSlrOdProblem:
    name: str
    object_id: str
    crd_text: str
    crd_source_name: str
    crd_sha256: str
    station_catalog: Mapping[str, Mapping[str, Any]]
    sigma_floor_ps: float
    tle_line1: str
    tle_line2: str
    tle_require_checksum: bool
    fit_duration_s: float
    validation_duration_s: float
    holdout_duration_s: float
    estimate_station_biases: bool
    station_bias_prior_sigma_km: float
    batch_max_nfev: int
    robust_loss: str
    robust_f_scale: float
    sigma_clip_threshold: float | None
    initial_covariance_eci: tuple[tuple[float, ...], ...]
    propagation_step_s: float
    j2: bool
    process_noise: SequentialProcessNoise
    innovation_gate_probability: float
    schema_version: str = PROBLEM_SCHEMA

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], *, base_dir: Path | None = None) -> ProSlrOdProblem:
        if not isinstance(value, Mapping):
            raise ProSlrOdError("Pro SLR OD problem must be a JSON object.")
        raw = dict(value)
        _exact(raw, {"schema_version", "name", "object_id", "crd_path", "crd_text", "crd_source_name", "crd_sha256", "station_catalog", "measurement", "template_tle", "windows", "batch", "sequential"}, "problem")
        if raw.get("schema_version") != PROBLEM_SCHEMA:
            raise ProSlrOdError(f"schema_version must be {PROBLEM_SCHEMA!r}.")
        name = str(raw.get("name", "")).strip()
        object_id = str(raw.get("object_id", "")).strip()
        if not name or not object_id:
            raise ProSlrOdError("name and object_id must be non-empty strings.")

        has_path = "crd_path" in raw
        has_text = "crd_text" in raw
        if has_path == has_text:
            raise ProSlrOdError("Provide exactly one of crd_path or crd_text.")
        if has_path:
            source = Path(str(raw["crd_path"])).expanduser()
            if not source.is_absolute():
                source = (base_dir or Path.cwd()) / source
            try:
                content = read_regular_file_nofollow(source, min_bytes=1, max_bytes=MAX_CRD_BYTES)
                crd_text = content.decode("utf-8", errors="strict")
            except (SafeReadError, UnicodeDecodeError) as exc:
                raise ProSlrOdError(f"Could not safely read CRD source: {exc}") from exc
            source_name = source.name
        else:
            if not isinstance(raw["crd_text"], str):
                raise ProSlrOdError("crd_text must be a UTF-8 string.")
            crd_text = str(raw["crd_text"])
            content = crd_text.encode("utf-8")
            if not 1 <= len(content) <= MAX_CRD_BYTES:
                raise ProSlrOdError(f"crd_text must contain between 1 and {MAX_CRD_BYTES} UTF-8 bytes.")
            source_name = str(raw.get("crd_source_name", "normalized_input.crd")).strip()
            if not source_name:
                raise ProSlrOdError("crd_source_name must be non-empty.")
        crd_sha256 = hashlib.sha256(content).hexdigest()
        declared_sha = raw.get("crd_sha256")
        if declared_sha is not None and str(declared_sha) != crd_sha256:
            raise ProSlrOdError("crd_sha256 does not match the supplied CRD content.")

        station_catalog = raw.get("station_catalog")
        if not isinstance(station_catalog, Mapping) or not station_catalog:
            raise ProSlrOdError("station_catalog must be a non-empty object keyed by ILRS station code.")
        normalized_stations: dict[str, dict[str, Any]] = {}
        for code, metadata in station_catalog.items():
            if not isinstance(metadata, Mapping):
                raise ProSlrOdError(f"station_catalog.{code} must be an object.")
            normalized_code = str(code).strip().upper()
            if not normalized_code:
                raise ProSlrOdError("station_catalog keys must be non-empty station codes.")
            if normalized_code in normalized_stations:
                raise ProSlrOdError(f"station_catalog contains duplicate station code {normalized_code!r}.")
            station = dict(metadata)
            try:
                xyz = np.asarray(station.get("ecef_m"), dtype=float)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ProSlrOdError(f"station_catalog.{code}.ecef_m must contain three finite values.") from exc
            if xyz.shape != (3,) or np.any(~np.isfinite(xyz)):
                raise ProSlrOdError(f"station_catalog.{code}.ecef_m must contain three finite values.")
            if not str(station.get("source_url", "")).strip():
                raise ProSlrOdError(f"station_catalog.{code}.source_url must document coordinate provenance.")
            station["ecef_m"] = xyz.tolist()
            normalized_stations[normalized_code] = station

        measurement = raw.get("measurement") or {}
        tle = raw.get("template_tle") or {}
        windows = raw.get("windows") or {}
        batch = raw.get("batch") or {}
        sequential = raw.get("sequential") or {}
        if not all(isinstance(item, Mapping) for item in (measurement, tle, windows, batch, sequential)):
            raise ProSlrOdError("measurement, template_tle, windows, batch, and sequential must be objects.")
        measurement, tle, windows, batch, sequential = map(dict, (measurement, tle, windows, batch, sequential))
        _exact(measurement, {"sigma_floor_ps"}, "measurement")
        _exact(tle, {"line1", "line2", "require_checksum"}, "template_tle")
        _exact(windows, {"fit_duration_s", "validation_duration_s", "holdout_duration_s"}, "windows")
        _exact(batch, {"estimate_station_biases", "station_bias_prior_sigma_km", "max_nfev", "robust_loss", "robust_f_scale", "sigma_clip_threshold"}, "batch")
        _exact(sequential, {"initial_covariance_eci", "step_s", "j2", "process_noise_model", "spectral_density", "innovation_gate_probability"}, "sequential")
        for obj, key, default in ((tle, "require_checksum", False), (batch, "estimate_station_biases", True), (sequential, "j2", False)):
            if key in obj and not isinstance(obj[key], bool):
                raise ProSlrOdError(f"{key} must be a boolean.")
            obj.setdefault(key, default)
        line1, line2 = str(tle.get("line1", "")), str(tle.get("line2", ""))
        try:
            parse_tle_lines(line1, line2, require_checksum=bool(tle["require_checksum"]))
        except ValueError as exc:
            raise ProSlrOdError(f"template_tle is invalid: {exc}") from exc
        fit = _finite(windows.get("fit_duration_s"), "windows.fit_duration_s")
        validation = _finite(windows.get("validation_duration_s"), "windows.validation_duration_s")
        holdout = _finite(windows.get("holdout_duration_s"), "windows.holdout_duration_s")
        step = _finite(sequential.get("step_s"), "sequential.step_s")
        if min(fit, validation, holdout, step) <= 0.0 or fit + validation + holdout > MAX_ARC_S or step >= fit:
            raise ProSlrOdError("All windows and sequential.step_s must be positive, bounded, and step_s must be smaller than the fit window.")
        sigma_floor_ps = _finite(measurement.get("sigma_floor_ps", 50.0), "measurement.sigma_floor_ps")
        if sigma_floor_ps <= 0.0:
            raise ProSlrOdError("measurement.sigma_floor_ps must be positive.")
        prior_sigma = _finite(batch.get("station_bias_prior_sigma_km", 0.25), "batch.station_bias_prior_sigma_km")
        max_nfev_value = batch.get("max_nfev", 120)
        if isinstance(max_nfev_value, bool) or not isinstance(max_nfev_value, int):
            raise ProSlrOdError("batch.max_nfev must be a positive integer.")
        max_nfev = int(max_nfev_value)
        robust_f_scale = _finite(batch.get("robust_f_scale", 2.0), "batch.robust_f_scale")
        clip = batch.get("sigma_clip_threshold", 8.0)
        clip_value = None if clip is None else _finite(clip, "batch.sigma_clip_threshold")
        if prior_sigma <= 0.0 or max_nfev <= 0 or robust_f_scale <= 0.0 or (clip_value is not None and clip_value <= 0.0):
            raise ProSlrOdError("Batch solver limits and scales must be positive.")
        try:
            process_noise = SequentialProcessNoise(
                model=str(sequential.get("process_noise_model", "white_acceleration")),
                spectral_density=tuple(sequential.get("spectral_density", (1.0e-14,) * 3)),
            )
        except (TypeError, ValueError) as exc:
            raise ProSlrOdError(f"sequential process noise is invalid: {exc}") from exc
        gate = _finite(sequential.get("innovation_gate_probability", 0.999), "sequential.innovation_gate_probability")
        if not 0.0 < gate < 1.0:
            raise ProSlrOdError("sequential.innovation_gate_probability must be between zero and one.")
        return cls(
            name=name, object_id=object_id, crd_text=crd_text, crd_source_name=source_name,
            crd_sha256=crd_sha256, station_catalog=normalized_stations, sigma_floor_ps=sigma_floor_ps,
            tle_line1=line1, tle_line2=line2, tle_require_checksum=bool(tle["require_checksum"]),
            fit_duration_s=fit, validation_duration_s=validation, holdout_duration_s=holdout,
            estimate_station_biases=bool(batch["estimate_station_biases"]), station_bias_prior_sigma_km=prior_sigma,
            batch_max_nfev=max_nfev, robust_loss=str(batch.get("robust_loss", "huber")),
            robust_f_scale=robust_f_scale, sigma_clip_threshold=clip_value,
            initial_covariance_eci=_covariance(sequential.get("initial_covariance_eci")),
            propagation_step_s=step, j2=bool(sequential["j2"]), process_noise=process_noise,
            innovation_gate_probability=gate,
        )

    @classmethod
    def from_json(cls, path: str | Path) -> ProSlrOdProblem:
        source = Path(path).expanduser().absolute()
        try:
            value = json.loads(read_regular_file_nofollow(source, min_bytes=1, max_bytes=MAX_PROBLEM_BYTES).decode("utf-8"), parse_constant=lambda item: (_ for _ in ()).throw(ValueError(item)))
        except (SafeReadError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise ProSlrOdError(f"Could not safely parse Pro SLR OD problem: {exc}") from exc
        return cls.from_mapping(value, base_dir=source.parent)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version, "name": self.name, "object_id": self.object_id,
            "crd_text": self.crd_text, "crd_source_name": self.crd_source_name, "crd_sha256": self.crd_sha256,
            "station_catalog": {key: dict(value) for key, value in self.station_catalog.items()},
            "measurement": {"sigma_floor_ps": self.sigma_floor_ps},
            "template_tle": {"line1": self.tle_line1, "line2": self.tle_line2, "require_checksum": self.tle_require_checksum},
            "windows": {"fit_duration_s": self.fit_duration_s, "validation_duration_s": self.validation_duration_s, "holdout_duration_s": self.holdout_duration_s},
            "batch": {"estimate_station_biases": self.estimate_station_biases, "station_bias_prior_sigma_km": self.station_bias_prior_sigma_km, "max_nfev": self.batch_max_nfev, "robust_loss": self.robust_loss, "robust_f_scale": self.robust_f_scale, "sigma_clip_threshold": self.sigma_clip_threshold},
            "sequential": {"initial_covariance_eci": [list(row) for row in self.initial_covariance_eci], "step_s": self.propagation_step_s, "j2": self.j2, "process_noise_model": self.process_noise.model, "spectral_density": list(self.process_noise.spectral_density), "innovation_gate_probability": self.innovation_gate_probability},
        }


def validate_problem(value: ProSlrOdProblem | Mapping[str, Any]) -> dict[str, Any]:
    problem = value if isinstance(value, ProSlrOdProblem) else ProSlrOdProblem.from_mapping(value)
    rows = _measurement_rows(problem)
    fit, validation, holdout = _partition(rows, problem)
    return {"status": "valid", "schema_version": problem.schema_version, "measurement_contract": SLR_TWO_WAY_CONTRACT, "measurement_count": len(rows), "fit_count": len(fit), "validation_count": len(validation), "holdout_count": len(holdout), "station_ids": sorted({str(row["station_id"]) for row in rows}), "crd_sha256": problem.crd_sha256, "problem_sha256": _digest(problem.to_dict())}


def run_pro_slr_od(value: ProSlrOdProblem | Mapping[str, Any], output_dir: str | Path) -> dict[str, Any]:
    problem = value if isinstance(value, ProSlrOdProblem) else ProSlrOdProblem.from_mapping(value)
    # Artifact paths embedded by downstream writers must share the absolute
    # evidence root used by canonical replay, including for CLI-relative paths.
    destination = Path(output_dir).expanduser().absolute()
    if destination.is_symlink() or destination.exists():
        raise ProSlrOdError("output_dir must be a new non-symlink path.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.building-", dir=destination.parent))
    try:
        summary = _materialize_evidence(problem, temporary)
        _write_json(temporary / "pro_slr_od_manifest.json", _build_manifest(temporary, summary))
        destination_abs = destination.absolute()
        temporary.replace(destination_abs)
        return {**summary, "output_dir": str(destination_abs), "manifest": str(destination_abs / "pro_slr_od_manifest.json")}
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def replay_evidence(output_dir: str | Path) -> dict[str, Any]:
    root = Path(output_dir).absolute()
    manifest = _read_json(root / "pro_slr_od_manifest.json")
    manifest_fields = {"schema_version", "status", "problem_sha256", "result_sha256", "files"}
    if set(manifest) != manifest_fields or manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise ProSlrOdError("Pro SLR OD manifest schema is invalid.")
    listed = manifest.get("files")
    if not isinstance(listed, list):
        raise ProSlrOdError("Pro SLR OD manifest files must be a JSON array.")
    expected = {str(row.get("path", "")) for row in listed if isinstance(row, Mapping)}
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file() and path.name != "pro_slr_od_manifest.json"}
    if expected != actual or len(expected) != len(listed):
        raise ProSlrOdError("Pro SLR OD evidence inventory is incomplete or unexpected.")
    retained_content = _verified_artifact_content(root, listed)
    retained_canonical = _canonical_artifact_content(root)
    problem = ProSlrOdProblem.from_mapping(
        _parse_json_bytes(retained_content["normalized_problem.json"], "normalized_problem.json")
    )
    with tempfile.TemporaryDirectory(
        prefix="oel-pro-slr-od-replay-", dir=Path(tempfile.gettempdir()).resolve()
    ) as temp:
        reproduced_root = Path(temp)
        reproduced = _materialize_evidence(problem, reproduced_root)
        reproduced_canonical = _canonical_artifact_content(reproduced_root)
        reproduced_manifest = _build_manifest(reproduced_root, reproduced)
    if reproduced_canonical != retained_canonical:
        raise ProSlrOdError("Authoritative Pro SLR OD replay does not reproduce every retained artifact.")
    if _manifest_claims(manifest) != _manifest_claims(reproduced_manifest):
        raise ProSlrOdError("Pro SLR OD manifest claims differ from authoritative replay.")
    if [row["path"] for row in listed] != [row["path"] for row in reproduced_manifest["files"]]:
        raise ProSlrOdError("Pro SLR OD manifest artifact inventory differs from authoritative replay.")
    return {"status": "verified", "problem_sha256": reproduced["problem_sha256"], "result_sha256": _digest(reproduced), "crd_sha256": problem.crd_sha256}


def _materialize_evidence(problem: ProSlrOdProblem, root: Path) -> dict[str, Any]:
    summary = _solve(problem, root)
    _write_json(root / "normalized_problem.json", problem.to_dict())
    _write_json(root / "pro_slr_od_summary.json", summary)
    _write_prediction_csv(root / "validation_predictions.csv", summary["validation_predictions"])
    _write_prediction_csv(root / "holdout_predictions.csv", summary["holdout_predictions"])
    return summary


def _artifact_content(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): read_regular_file_nofollow(path, max_bytes=128 * 1024 * 1024)
        for path in sorted(item for item in root.rglob("*") if item.is_file() and item.name != "pro_slr_od_manifest.json")
    }


def _canonical_artifact_content(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): canonical_evidence_artifact_bytes(
            path,
            root=root,
            max_bytes=128 * 1024 * 1024,
        )
        for path in sorted(
            item for item in root.rglob("*") if item.is_file() and item.name != "pro_slr_od_manifest.json"
        )
    }


def _manifest_claims(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {name: value for name, value in manifest.items() if name != "files"}


def _build_manifest(root: Path, summary: Mapping[str, Any]) -> dict[str, Any]:
    files = [
        {"path": name, "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
        for name, content in _artifact_content(root).items()
    ]
    return {"schema_version": MANIFEST_SCHEMA, "status": "complete", "problem_sha256": summary["problem_sha256"], "result_sha256": _digest(summary), "files": files}


def _verified_artifact_content(root: Path, listed: Sequence[Any]) -> dict[str, bytes]:
    content_by_name: dict[str, bytes] = {}
    for value in listed:
        if not isinstance(value, Mapping) or set(value) != {"path", "bytes", "sha256"}:
            raise ProSlrOdError("Pro SLR OD artifact receipt contract is invalid.")
        name = value["path"]
        size = value["bytes"]
        digest = value["sha256"]
        if not isinstance(name, str) or name in content_by_name:
            raise ProSlrOdError("Pro SLR OD artifact receipt path is invalid.")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ProSlrOdError(f"Evidence receipt size is invalid for {name}.")
        if not isinstance(digest, str) or len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ProSlrOdError(f"Evidence receipt digest is invalid for {name}.")
        content = read_regular_file_nofollow(root / name, max_bytes=128 * 1024 * 1024)
        if len(content) != size or hashlib.sha256(content).hexdigest() != digest:
            raise ProSlrOdError(f"Evidence receipt mismatch for {name}.")
        content_by_name[name] = content
    return content_by_name


def _solve(problem: ProSlrOdProblem, output: Path) -> dict[str, Any]:
    rows = _measurement_rows(problem)
    fit_rows, validation_rows, holdout_rows = _partition(rows, problem)
    template = parse_tle_lines(problem.tle_line1, problem.tle_line2, require_checksum=problem.tle_require_checksum)
    batch = fit_ogp_mean_elements_to_slr_normal_points(
        rows, template_elements=template, object_id=problem.object_id, output_dir=output / "batch",
        fit_duration_s=problem.fit_duration_s, holdout_duration_s=problem.validation_duration_s + problem.holdout_duration_s,
        estimate_station_biases=problem.estimate_station_biases, station_bias_prior_sigma_km=problem.station_bias_prior_sigma_km,
        max_nfev=problem.batch_max_nfev, robust_loss=problem.robust_loss, robust_f_scale=problem.robust_f_scale,
        sigma_clip_threshold=problem.sigma_clip_threshold,
    )
    initial_state = np.asarray(batch["estimated_state_at_first_observation_eci_km_km_s"], dtype=float)
    bias_map = {str(item["name"]).split("__", 1)[1]: float(item["value"]) for item in batch["estimated_parameters"] if str(item["name"]).startswith("slr_range_bias_km__")}
    config = SequentialODConfig(
        estimator="ekf", process_noise=problem.process_noise,
        innovation_gate_probability=problem.innovation_gate_probability,
        integration_substep_s=problem.propagation_step_s, checkpoint_interval=max(len(fit_rows), 1),
        reinitialize_after_maneuver=False,
    )
    propagator = OrbitPropagator(integrator="rk4", plugins=[j2_plugin] if problem.j2 else [])
    context = OrbitContext(mu_km3_s2=EARTH_MU_KM3_S2, mass_kg=100.0)
    model = _measurement_model(bias_map=bias_map, propagator=propagator, context=context, config=config)
    packet = ObservationPacket({"schema_version": 1, "object_id": problem.object_id, "frame": "ECI", "measurement_model": "ilrs_crd_v2_two_way_range", "observations": fit_rows})
    sequential = run_sequential_orbit_od(
        packet, initial_state_eci_km_s=initial_state, initial_covariance=problem.initial_covariance_eci,
        config=config, propagator=propagator, orbit_context=context, output_dir=output / "sequential",
        measurement_model=model, measurement_model_id="ilrs_crd_v2_two_way_range_v1",
    )
    terminal = sequential["records"][-1]
    future_rows = [*validation_rows, *holdout_rows]
    predictions = predict_sequential_state_covariance(
        terminal["filtered_state"], terminal["filtered_covariance"], initial_time_s=float(terminal["time_s"]),
        prediction_times_s=[float(row["time_s"]) for row in future_rows], config=config,
        propagator=propagator, orbit_context=context,
    )
    evaluated = [_evaluate_prediction(row, prediction, model) for row, prediction in zip(future_rows, predictions, strict=True)]
    validation_predictions = evaluated[:len(validation_rows)]
    holdout_predictions = evaluated[len(validation_rows):]
    estimated_elements = _elements_from_payload(batch["estimated_elements"], template)
    validation_batch_records = _slr_residual_records(estimated_elements, validation_rows, parameter_values={f"slr_range_bias_km__{key}": value for key, value in bias_map.items()}, frame_transform="teme_to_eci_iau80", frame_model="simple", eop_path=None, partition="validation")
    holdout_batch_records = _slr_residual_records(estimated_elements, holdout_rows, parameter_values={f"slr_range_bias_km__{key}": value for key, value in bias_map.items()}, frame_transform="teme_to_eci_iau80", frame_model="simple", eop_path=None, partition="holdout")
    return {
        "schema_version": EVIDENCE_SCHEMA, "status": "complete", "problem_sha256": _digest(problem.to_dict()),
        "crd_receipt": {"source_name": problem.crd_source_name, "sha256": problem.crd_sha256, "bytes": len(problem.crd_text.encode("utf-8"))},
        "measurement_contract": SLR_TWO_WAY_CONTRACT, "station_ids": sorted({str(row["station_id"]) for row in rows}),
        "counts": {"total": len(rows), "fit": len(fit_rows), "validation": len(validation_rows), "holdout": len(holdout_rows)},
        "batch_summary": {"solver": batch["solver"], "fit_metrics": batch["fit_metrics"], "validation_metrics": _range_metrics(validation_batch_records), "holdout_metrics": _range_metrics(holdout_batch_records), "estimated_elements": batch["estimated_elements"], "estimated_parameters": batch["estimated_parameters"], "state_epoch_jd_utc": batch["state_epoch_jd_utc"], "estimated_state_at_first_observation_eci_km_km_s": batch["estimated_state_at_first_observation_eci_km_km_s"], "quality_gates": batch["quality_gates"]},
        "sequential_summary": {"accepted_count": sequential["accepted_count"], "rejected_count": sequential["rejected_count"], "covariance_checks": sequential["covariance_checks"], "innovation_statistics": sequential["innovation_statistics"], "smoother": sequential["smoother"], "quality_gates": sequential["quality_gates"]},
        "validation_statistics": _prediction_metrics(validation_predictions), "holdout_statistics": _prediction_metrics(holdout_predictions),
        "validation_predictions": validation_predictions, "holdout_predictions": holdout_predictions,
        "claim_posture": {"calibrated_covariance": False, "geodetic_orbit": False, "batch_covariance_use": "reported in native parameter coordinates only; sequential covariance is user-declared", "station_bias_semantics": "batch-estimated aggregate one-way terms held fixed during EKF/RTS and prediction", "media_and_target_corrections": "CRD flags preserved; no independent troposphere, center-of-mass, calibration, or target-signature model", "operational_use": False},
    }


def _measurement_rows(problem: ProSlrOdProblem) -> list[dict[str, Any]]:
    # ``tempfile.gettempdir()`` is commonly the /var compatibility symlink on
    # macOS. Resolve the directory itself so the deliberately no-follow CRD
    # reader receives a path with no symlink components.
    with tempfile.TemporaryDirectory(prefix="oel-pro-slr-crd-", dir=Path(tempfile.gettempdir()).resolve()) as temp:
        path = Path(temp) / "input.crd"
        path.write_text(problem.crd_text, encoding="utf-8")
        packet = parse_ilrs_crd_v2_normal_points(path)
    if packet["source_sha256"] != problem.crd_sha256:
        raise ProSlrOdError("Parsed CRD receipt does not match the normalized problem.")
    adapted = adapt_ilrs_crd_normal_points(
        packet,
        station_catalog=problem.station_catalog,
        sigma_floor_ps=problem.sigma_floor_ps,
    )
    rows = [dict(row) for row in adapted["measurement_rows"]]
    if len(rows) > MAX_NORMAL_POINTS:
        raise ProSlrOdError(f"CRD input exceeds the {MAX_NORMAL_POINTS}-normal-point bound.")
    times = np.asarray([float(row["time_s"]) for row in rows], dtype=float)
    if abs(float(times[0])) > 1.0e-3 or np.any(np.diff(times) <= 0.0):
        raise ProSlrOdError("The v1 Pro SLR workflow requires one strictly increasing stream beginning at time_s=0.")
    return rows


def _partition(rows: Sequence[Mapping[str, Any]], problem: ProSlrOdProblem) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    tolerance = _PARTITION_BOUNDARY_TOLERANCE_S
    fit_end = problem.fit_duration_s
    validation_end = fit_end + problem.validation_duration_s
    holdout_end = validation_end + problem.holdout_duration_s
    fit = [dict(row) for row in rows if float(row["time_s"]) <= fit_end + tolerance]
    validation = [dict(row) for row in rows if fit_end + tolerance < float(row["time_s"]) <= validation_end + tolerance]
    holdout = [dict(row) for row in rows if validation_end + tolerance < float(row["time_s"]) <= holdout_end + tolerance]
    if len(fit) < 8 or not validation or not holdout:
        raise ProSlrOdError("Declared windows require at least eight fit rows, one validation row, and one untouched holdout row.")
    if len(fit) + len(validation) + len(holdout) != len(rows):
        raise ProSlrOdError("Every normal point must fall inside the declared fit, validation, and holdout windows.")
    return fit, validation, holdout


def _measurement_model(*, bias_map: Mapping[str, float], propagator: OrbitPropagator, context: OrbitContext, config: SequentialODConfig):
    def vectors(row: Mapping[str, Any], state: np.ndarray) -> tuple[float, float, float]:
        epoch_jd = float(row["time_jd_utc"])
        epoch_time_s = float(row["time_s"])
        def state_at_jd(jd_utc: float) -> np.ndarray:
            dt = (float(jd_utc) - epoch_jd) * 86400.0
            if abs(dt) <= 1.0e-12:
                return np.asarray(state, dtype=float)
            return propagator.propagate(np.asarray(state, dtype=float), dt, epoch_time_s, np.zeros(3), {}, context)
        prediction = predict_slr_two_way_time_of_flight_s(
            epoch_jd_utc=epoch_jd, epoch_event=int(row["epoch_event"]), station_ecef_km=dict(row["station_metadata"])["ecef_km"],
            state_at_jd_utc=state_at_jd, aggregate_one_way_range_correction_km=float(bias_map.get(str(row["station_id"]), 0.0)),
        )
        observed = 0.5 * C_LIGHT_KM_S * float(row["observed_time_of_flight_s"])
        return observed, float(prediction["predicted_two_way_range_km"]), float(row["sigma_range_km"])
    def evaluate(row: Mapping[str, Any], state: np.ndarray) -> dict[str, Any]:
        observed, predicted, sigma = vectors(row, state)
        steps = np.asarray([config.finite_difference_position_step_km] * 3 + [config.finite_difference_velocity_step_km_s] * 3)
        jacobian = np.zeros((1, 6))
        for axis, step in enumerate(steps):
            plus, minus = np.asarray(state, dtype=float).copy(), np.asarray(state, dtype=float).copy()
            plus[axis] += step
            minus[axis] -= step
            jacobian[0, axis] = (vectors(row, plus)[1] - vectors(row, minus)[1]) / (2.0 * step)
        return {"observed": [observed], "predicted": [predicted], "jacobian": jacobian.tolist(), "covariance": [[sigma * sigma]], "innovation": [observed - predicted]}
    return evaluate


def _evaluate_prediction(row: Mapping[str, Any], prediction: Mapping[str, Any], model: Any) -> dict[str, Any]:
    evaluation = model(row, np.asarray(prediction["state_eci_km_km_s"], dtype=float))
    h = np.asarray(evaluation["jacobian"], dtype=float)
    covariance = np.asarray(evaluation["covariance"], dtype=float)
    state_covariance = np.asarray(prediction["covariance_eci_km_km_s"], dtype=float)
    innovation = np.asarray(evaluation["innovation"], dtype=float)
    innovation_covariance = h @ state_covariance @ h.T + covariance
    nis = float(innovation.T @ np.linalg.pinv(innovation_covariance) @ innovation)
    return {"measurement_id": str(row["measurement_id"]), "station_id": str(row["station_id"]), "time_s": float(row["time_s"]), "observed_range_km": float(evaluation["observed"][0]), "predicted_range_km": float(evaluation["predicted"][0]), "innovation_m": float(innovation[0] * 1000.0), "nis": nis, "state_eci_km_km_s": prediction["state_eci_km_km_s"], "covariance_eci_km_km_s": prediction["covariance_eci_km_km_s"]}


def _prediction_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    residuals = np.asarray([float(row["innovation_m"]) for row in rows], dtype=float)
    nis = np.asarray([float(row["nis"]) for row in rows], dtype=float)
    return {"count": len(rows), "range_rms_m": float(np.sqrt(np.mean(residuals * residuals))), "range_mean_m": float(np.mean(residuals)), "range_p95_abs_m": float(np.percentile(np.abs(residuals), 95.0)), "nis_mean": float(np.mean(nis)), "nis_max": float(np.max(nis)), "finite": bool(np.all(np.isfinite(residuals)) and np.all(np.isfinite(nis)))}


def _elements_from_payload(payload: Mapping[str, Any], template: TLEElements) -> TLEElements:
    from dataclasses import replace
    return replace(template, inclination_deg=float(payload["inclination_deg"]), raan_deg=float(payload["raan_deg"]), eccentricity=float(payload["eccentricity"]), argp_deg=float(payload["argp_deg"]), mean_anomaly_deg=float(payload["mean_anomaly_deg"]), mean_motion_rev_per_day=float(payload["mean_motion_rev_per_day"]))


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(value), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        content = read_regular_file_nofollow(path, min_bytes=1, max_bytes=128 * 1024 * 1024)
    except SafeReadError as exc:
        raise ProSlrOdError(f"Could not safely read {path.name}: {exc}") from exc
    return _parse_json_bytes(content, path.name)


def _parse_json_bytes(content: bytes, field: str) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ProSlrOdError(f"{field} contains forbidden non-finite constant {value}.")

    def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in values:
            if key in result:
                raise ProSlrOdError(f"{field} contains duplicate field {key!r}.")
            result[key] = value
        return result

    try:
        value = json.loads(content.decode("utf-8"), parse_constant=reject_constant, object_pairs_hook=pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProSlrOdError(f"Could not safely parse {field}: {exc}") from exc
    if not isinstance(value, dict):
        raise ProSlrOdError(f"{field} must contain a JSON object.")
    return value


def _write_prediction_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("measurement_id", "station_id", "time_s", "observed_range_km", "predicted_range_km", "innovation_m", "nis"))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in writer.fieldnames})


__all__ = ["EVIDENCE_SCHEMA", "MANIFEST_SCHEMA", "PROBLEM_SCHEMA", "ProSlrOdError", "ProSlrOdProblem", "replay_evidence", "run_pro_slr_od", "validate_problem"]
