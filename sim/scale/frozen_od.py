"""Frozen OGP solution assessment: exact fit identity, later data, read-only store.

Assessment never refits, promotes a catalog state, consumes observations or
claims covariance calibration. Published gates are part of the frozen packet.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from sim.dynamics.orbit.ogp import ogp_propagate_teme_batch_accelerated
from sim.dynamics.orbit.tle import ogp_mean_elements_from_mapping
from sim.scale.store_od import load_exact_od_assessment_inputs

SCHEMA = "oel.scale.frozen_ogp_od.v1"


def _bytes(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _digest(value):
    return hashlib.sha256(_bytes(value)).hexdigest()


def _model_identity():
    import oel_rust_orbit

    root = Path(__file__).resolve().parents[2]
    paths = [
        Path(__file__),
        Path(__file__).with_name("store_od.py"),
        Path(oel_rust_orbit.__file__),
        root / "sim/rust_ogp_backend.py",
        root / "sim/dynamics/orbit/ogp.py",
        root / "sim/dynamics/orbit/tle.py",
    ]
    paths += sorted((root / "rust/oel-orbit/src").rglob("*.rs"))
    return {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def _gates(position_rms_m_max, minimum_observations):
    if isinstance(position_rms_m_max, bool) or not math.isfinite(position_rms_m_max) or position_rms_m_max <= 0:
        raise ValueError("Position gate must be finite and positive")
    if isinstance(minimum_observations, bool) or not isinstance(minimum_observations, int) or minimum_observations < 2:
        raise ValueError("At least two independent-time holdout observations are required")
    return dict(position_rms_m_max=float(position_rms_m_max), minimum_observations=minimum_observations)


def freeze_ogp_od_solution(store_path, *, od_solution_id, output_path, position_rms_m_max=2.0, minimum_observations=6):
    """Freeze an exact stored fit, including review-required fits lacking a holdout."""
    path = Path(output_path)
    if path.exists() or path.with_suffix(path.suffix + ".sha256").exists():
        raise ValueError("Frozen solution output must be new")
    gates = _gates(position_rms_m_max, minimum_observations)
    bundle = load_exact_od_assessment_inputs(store_path, od_solution_id=od_solution_id)
    fit = bundle["solution"]
    if fit["method"] != "ogp_mean_elements_fit" or str(fit["native_frame"]).lower() != "teme":
        raise ValueError("Freeze requires a native TEME OGP mean-element fitted solution")
    elements = json.loads(fit["mean_elements_json"])
    ogp_mean_elements_from_mapping(elements)
    training = bundle["observations"]
    if any(row["object_id"] != fit["object_id"] or str(row["frame"]).lower() != "teme" for row in training):
        raise ValueError("Fit source object/frame mismatch")
    epochs = [float(row["time_jd_utc"]) for row in training]
    if not all(math.isfinite(t) for t in epochs):
        raise ValueError("Nonfinite source epoch")
    packet = dict(
        schema=SCHEMA,
        od_solution_id=od_solution_id,
        object_id=fit["object_id"],
        solution=fit,
        training_rows=training,
        training_end_jd_utc=max(epochs),
        fitted_elements=elements,
        gates=gates,
        model_identity=_model_identity(),
        fit_record_sha256=_digest(fit),
        source_rows_sha256=_digest(training),
        prediction_source="exact_frozen_od_solution",
        empirically_calibrated=False,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_bytes(packet))
    path.with_suffix(path.suffix + ".sha256").write_text(hashlib.sha256(path.read_bytes()).hexdigest() + "\n")
    return packet


def read_frozen_ogp_od_solution(store_path, frozen_solution_path):
    path = Path(frozen_solution_path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != path.with_suffix(path.suffix + ".sha256").read_text().strip():
        raise ValueError("Frozen solution digest mismatch")
    packet = json.loads(path.read_text())
    if packet["schema"] != SCHEMA or packet["model_identity"] != _model_identity():
        raise ValueError("Frozen model/source identity drift")
    _gates(**packet["gates"])
    bundle = load_exact_od_assessment_inputs(store_path, od_solution_id=packet["od_solution_id"])
    if bundle["solution"] != packet["solution"] or bundle["observations"] != packet["training_rows"]:
        raise ValueError("Frozen fitted solution or its source observations drifted")
    if packet["fitted_elements"] != json.loads(packet["solution"]["mean_elements_json"]):
        raise ValueError("Frozen parameters disagree with the exact stored fit")
    return packet


def assess_frozen_ogp_od_solution(store_path, *, frozen_solution_path, observation_ids, output_dir):
    """Assess exact later rows against a frozen fit; no optimizer or store writes."""
    out = Path(output_dir)
    if out.exists():
        raise ValueError("Use a new assessment output directory")
    frozen = read_frozen_ogp_od_solution(store_path, frozen_solution_path)
    observations = load_exact_od_assessment_inputs(store_path, observation_ids=observation_ids)["observations"]
    if len(observations) < frozen["gates"]["minimum_observations"]:
        raise ValueError("Insufficient held-out observations")
    source_ids = {row["observation_id"] for row in frozen["training_rows"]}
    if any(
        row["observation_id"] in source_ids
        or row["object_id"] != frozen["object_id"]
        or str(row["frame"]).lower() != "teme"
        for row in observations
    ):
        raise ValueError("Held-out source/object/frame overlap or mismatch")
    times = np.asarray([row["time_jd_utc"] for row in observations], dtype=float)
    positions = np.asarray([[row[f"position_{axis}_km"] for axis in "xyz"] for row in observations], dtype=float)
    if not np.all(np.isfinite(times)) or not np.all(np.isfinite(positions)) or np.any(np.diff(times) <= 0):
        raise ValueError("Assessment requires finite position observations at distinct increasing epochs")
    if np.any(times <= frozen["training_end_jd_utc"]):
        raise ValueError("Independent observations must follow every fitted/development source epoch")
    elements = ogp_mean_elements_from_mapping(frozen["fitted_elements"])
    batch = ogp_propagate_teme_batch_accelerated([elements], (times - elements.epoch_jd_utc) * 1440, backend="rust")
    if not np.all(batch.success):
        raise ValueError("Frozen OGP propagation failed")
    errors = (batch.position_teme_km[0] - positions) * 1000
    rms = float(np.sqrt(np.mean(np.sum(errors * errors, axis=1))))
    quality = json.loads(frozen["solution"]["quality_flags_json"])
    hard = {k: quality.get(k) is True for k in ["solver_success", "data_full_rank", "fit_improved", "regime_stable"]}
    checks = dict(
        fit_hard_gates=all(hard.values()),
        position_accuracy=rms <= frozen["gates"]["position_rms_m_max"],
        observations_disjoint=True,
        all_finite=True,
    )
    result = dict(
        schema="oel.scale.frozen_ogp_assessment.v1",
        passed=all(checks.values()),
        checks=checks,
        position_rms_m=rms,
        position_p95_m=float(np.percentile(np.linalg.norm(errors, axis=1), 95)),
        observation_count=len(observations),
        od_solution_id=frozen["od_solution_id"],
        object_id=frozen["object_id"],
        prediction_source="exact_frozen_od_solution",
        fitted_solution_sha256=frozen["fit_record_sha256"],
        frozen_packet_sha256=hashlib.sha256(Path(frozen_solution_path).read_bytes()).hexdigest(),
        heldout_rows_sha256=_digest(observations),
        heldout_observation_ids=[r["observation_id"] for r in observations],
        gates=frozen["gates"],
        fit_hard_gates=hard,
        current_state_updated=False,
        observations_consumed=False,
        empirically_calibrated=False,
        source_status=frozen["solution"]["status"],
        residuals=[dict(jd_utc=float(t), position_error_m=e.tolist()) for t, e in zip(times, errors, strict=True)],
    )
    read_frozen_ogp_od_solution(store_path, frozen_solution_path)
    if observations != load_exact_od_assessment_inputs(store_path, observation_ids=observation_ids)["observations"]:
        raise ValueError("Held-out observations changed during assessment")
    out.mkdir(parents=True)
    (out / "assessment.json").write_bytes(_bytes(result))
    return result
