"""Versioned full-force tracking-to-forecast workflow, with an append-only ledger.

The shared sequential owner performs estimation. This module owns contracts,
source identity, simultaneous-epoch assembly, revision transactions and replay.
"""

from __future__ import annotations

import copy
import importlib
import json
import os
import platform
import shutil
import sys
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from sim.estimation.augmented_dynamics import AugmentedONPDynamics, AugmentedParameter, canonical_digest
from sim.estimation.covariance import validate_covariance
from sim.estimation.maneuver_detection import chi_square_threshold
from sim.estimation.sequential_od import SequentialODConfig, run_sequential_orbit_od
from sim.estimation.tracking_measurements import TrackingMeasurementModel
from sim.observations import ObservationPacket
from sim.utils.io import sha256_regular_file_nofollow

SCHEMA = "oel.tracking_od_continuous.v1"
ROOT = Path(__file__).resolve().parents[1]


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def file_sha(path):
    return sha256_regular_file_nofollow(Path(path).resolve(), max_bytes=1024**3)[0]


def source_identity():
    files = [
        ROOT / "estimation" / name
        for name in ("augmented_dynamics.py", "tracking_measurements.py", "sequential_od.py", "covariance.py")
    ]
    files += [
        Path(__file__),
        ROOT / "slr_tracking.py",
        ROOT / "pro_slr_od/media.py",
        ROOT / "pro_tracking_od/core.py",
        ROOT / "pro_tracking_od/batch.py",
        ROOT / "pro_tracking_od/slr_inputs.py",
        ROOT / "pro_tracking_od/__main__.py",
        ROOT / "utils/io.py",
    ]
    files += sorted((ROOT / "estimation").glob("*.py"))
    # Bind the maintained numerical owners as well as the orchestration layer.
    files += sorted((ROOT / "dynamics/orbit").glob("*.py"))
    files += sorted((ROOT / "pro_perturbations").glob("*.py"))
    files += [ROOT / "runtime/satellite_factory.py"]
    result = {str(p.relative_to(ROOT.parent)): file_sha(p) for p in files}
    try:
        native = Path(importlib.import_module("oel_rust_orbit").__file__)
        result["native"] = file_sha(native)
    except ImportError:
        pass
    return {"files": result, "python": sys.version, "numpy": np.__version__, "platform": platform.platform()}


def verify_sources(receipts):
    for receipt in receipts:
        if set(receipt) != {"path", "sha256", "bytes", "role"}:
            raise ValueError("source receipts require path, sha256, bytes and role")
        digest, size = sha256_regular_file_nofollow(Path(receipt["path"]), max_bytes=1024**3)
        if size != receipt["bytes"] or digest != receipt["sha256"]:
            raise ValueError("tracking input source receipt mismatch")


def dynamics_from_scenario(raw, object_id, *, parameters=(), step_s=20.0, spectral_density=(0.0, 0.0, 0.0), events=()):
    from sim.api import SimulationSession
    from sim.dynamics.model import OrbitalAttitudeDynamics
    from sim.dynamics.orbit.accelerations import OrbitContext
    from sim.scenarios import ScenarioArtifact

    artifact = ScenarioArtifact.from_dict(copy.deepcopy(raw))
    session = SimulationSession.from_config(artifact, history_mode="dynamic")
    if session.reset() is None:
        raise ValueError("continuous OD requires a single-run validated passive scenario")
    engine = session._engine
    if (
        set(engine.agents) != {object_id}
        or engine.collision_stepper is not None
        or engine.system_force_stepper is not None
    ):
        raise ValueError("continuous OD requires one uncoupled object")
    agent = engine.agents[object_id]
    dynamics = agent.dynamics
    if (
        agent.bridge is not None
        or type(dynamics) is not OrbitalAttitudeDynamics
        or dynamics.resource_model is not None
        or dynamics.propagate_attitude
        and engine.attitude_enabled
        or dynamics.geometry_area_profile is not None
        or dynamics.use_rectangular_prism_for_aero_srp
    ):
        raise ValueError("continuous OD requires standard passive ONP without resource/attitude-dependent forces")
    if raw["objects"][object_id].get("flight_software", {}).get("stack") != "fsw.passive":
        raise ValueError("continuous OD requires the explicit fsw.passive stack")
    allowed_modules = {
        "sim.dynamics.orbit.propagator",
        "sim.pro_perturbations.schwarzschild",
        "sim.pro_perturbations.solid_earth_tides",
        "sim.pro_perturbations.ocean_tides",
        "sim.pro_perturbations.earth_radiation",
    }
    if any(plugin.__module__ not in allowed_modules for plugin in dynamics.orbit_propagator.plugins):
        raise ValueError("continuous OD supports only maintained built-in/precision force owners")
    context = OrbitContext(dynamics.mu_km3_s2, agent.truth.mass_kg, dynamics.area_m2, dynamics.cd, dynamics.cr)
    env = engine.satellite_stepper.passive_history._environment(agent)
    assets = {}

    def bind_paths(value, key=""):
        if isinstance(value, dict):
            for k, v in value.items():
                bind_paths(v, k)
        elif isinstance(value, list):
            for v in value:
                bind_paths(v, key)
        elif isinstance(value, str) and ("path" in key or key.endswith("_file")):
            candidate = Path(value)
            if candidate.is_file():
                assets[str(candidate.resolve())] = file_sha(candidate)

    bind_paths(raw)
    return AugmentedONPDynamics(
        propagator=dynamics.orbit_propagator,
        context=context,
        environment=env,
        parameters=parameters,
        step_s=step_s,
        acceleration_spectral_density=spectral_density,
        events=events,
        model_identity={"scenario_sha256": canonical_digest(raw), "assets": assets, "sources": source_identity()},
    )


def normalize_rows(rows):
    by_id = {}
    duplicates = 0
    for raw in rows:
        if not isinstance(raw, Mapping):
            raise ValueError("each tracking row must be a JSON object")
        row = copy.deepcopy(dict(raw))
        if not isinstance(row.get("measurement_id"), str) or not row["measurement_id"]:
            raise ValueError("each tracking row requires an explicit measurement_id")
        try:
            time_s = float(row["time_s"])
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError("tracking epochs must be finite and nonnegative") from exc
        if not np.isfinite(time_s) or time_s < 0:
            raise ValueError("tracking epochs must be finite and nonnegative")
        row["time_s"] = time_s
        key = row["measurement_id"]
        if key in by_id:
            if canonical_digest(row) != canonical_digest(by_id[key]):
                raise ValueError("conflicting duplicate measurement_id")
            duplicates += 1
        by_id[key] = row
    return sorted(by_id.values(), key=lambda r: (r["time_s"], r["measurement_id"])), duplicates


def epoch_rows(rows, *, independent, epoch_covariances):
    groups = []
    for row in rows:
        if not groups or groups[-1]["time_s"] != row["time_s"]:
            groups.append({"time_s": row["time_s"], "epoch_rows": [], "independent_measurements": independent})
        groups[-1]["epoch_rows"].append(row)
    for group in groups:
        key = str(float(group["time_s"]))
        if key in epoch_covariances:
            group["epoch_covariance"] = epoch_covariances[key]
    return groups


def validate_problem(problem):
    allowed = {
        "schema",
        "object_id",
        "scenario",
        "initial_state",
        "initial_covariance",
        "parameters",
        "events",
        "step_s",
        "acceleration_spectral_density",
        "gate_probability",
        "measurements",
        "closed_through_s",
        "fit_through_s",
        "independent_measurements",
        "epoch_covariances",
        "sources",
        "frame_model",
        "eop_path",
        "inertial_frame",
        "epoch_jd_utc",
        "analysis_mode",
    }
    if set(problem) - allowed or problem.get("schema") != SCHEMA:
        raise ValueError("unsupported continuous OD schema or unknown fields")
    if problem.get("inertial_frame") not in {"GCRF", "EME2000"}:
        raise ValueError("declare the exact GCRF or EME2000 inertial frame")
    if not np.isfinite(problem["epoch_jd_utc"]):
        raise ValueError("declare an absolute UTC state epoch")
    if not isinstance(problem.get("independent_measurements"), bool):
        raise ValueError("declare simultaneous measurement independence explicitly")
    if problem.get("analysis_mode", "frozen_holdout") not in {"frozen_holdout", "continuous"}:
        raise ValueError("analysis mode must be frozen_holdout or continuous")
    if problem.get("analysis_mode") == "continuous" and problem["fit_through_s"] != problem["closed_through_s"]:
        raise ValueError("continuous analysis assimilates every closed epoch; use frozen_holdout for qualification")
    if len(problem["measurements"]) > 20000 or len(problem.get("parameters", [])) > 32:
        raise ValueError("continuous OD exceeds the bounded observation/parameter envelope")
    if not np.isfinite(problem["closed_through_s"]) or problem["closed_through_s"] > 30 * 86400:
        raise ValueError("continuous OD arc must be bounded to 30 days")
    rows, duplicates = normalize_rows(problem["measurements"])
    if (
        not rows
        or rows[0]["time_s"] != 0
        or not np.isfinite(problem["closed_through_s"])
        or problem["closed_through_s"] < rows[-1]["time_s"]
    ):
        raise ValueError("ledger must begin at zero and close every supplied epoch")
    if (
        not np.isfinite(problem["fit_through_s"])
        or not 0 < problem["fit_through_s"] <= problem["scenario"]["simulator"]["duration_s"]
    ):
        raise ValueError("fit boundary must be positive and inside the declared scenario arc")
    if not 0 < problem.get("gate_probability", 0.999) < 1:
        raise ValueError("invalid fixed innovation gate probability")
    verify_sources(problem.get("sources", []))
    parameters = [AugmentedParameter(**p) for p in problem.get("parameters", [])]
    model = dynamics_from_scenario(
        problem["scenario"],
        problem["object_id"],
        parameters=parameters,
        step_s=problem["step_s"],
        spectral_density=problem["acceleration_spectral_density"],
        events=problem.get("events", []),
    )
    state = np.asarray(problem["initial_state"], dtype=float)
    if state.shape != (len(model.state_names),) or not np.all(np.isfinite(state)):
        raise ValueError("initial state must match declared augmented slots")
    validate_covariance(problem["initial_covariance"], len(state))
    scenario_epoch = problem["scenario"]["simulator"].get("initial_jd_utc")
    if scenario_epoch != problem["epoch_jd_utc"]:
        raise ValueError("scenario and estimation state epochs must agree exactly")
    from sim.dynamics.orbit.frames import frame_context_from_mapping

    frame_context = frame_context_from_mapping(
        problem["scenario"]["simulator"].get("frames", {}), jd_utc_start=problem["epoch_jd_utc"], source="continuous_od"
    )
    measurement = TrackingMeasurementModel(
        model,
        frame_context=frame_context,
        frame_model=problem.get("frame_model", "simple"),
        eop_path=problem.get("eop_path"),
    )
    source_digests = {r["sha256"] for r in problem.get("sources", [])}
    for row in rows:
        for entry in row.get("corrections", {}).values():
            if entry.get("status") in {"applied", "model"} and entry.get("source_sha256") not in source_digests:
                raise ValueError("correction provenance must bind a retained source file")
        if "time_jd_utc" in row and abs((row["time_jd_utc"] - problem["epoch_jd_utc"]) * 86400 - row["time_s"]) > 5e-5:
            raise ValueError("tracking UTC and relative epochs disagree")
        if row.get("inertial_frame", problem["inertial_frame"]) != problem["inertial_frame"]:
            raise ValueError("tracking row inertial frame differs from the declared contract")
    groups = epoch_rows(
        rows, independent=problem["independent_measurements"], epoch_covariances=problem.get("epoch_covariances", {})
    )
    if len([g for g in groups if g["time_s"] <= problem["fit_through_s"]]) < 3:
        raise ValueError("continuous OD requires at least three distinct fit epochs")
    return rows, groups, model, measurement, duplicates


def run_continuous_tracking_od(problem, output_dir, *, append=False):
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    # Exclusive writer lock; a stale lock requires explicit inspection, never automatic stealing.
    lock = root / ".writer.lock"
    fd = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(fd, str(os.getpid()).encode())
        return _run_locked(copy.deepcopy(problem), root, append=append)
    finally:
        os.close(fd)
        lock.unlink()


def _run_locked(problem, root, *, append):
    current = root / "current.json"
    previous = None
    if current.exists():
        if not append:
            raise ValueError("existing tracking ledger requires explicit append")
        previous = json.loads(current.read_text())
        prior_dir = root / previous["revision"]
        old = json.loads((prior_dir / "problem.json").read_text())
        if canonical_digest(old) != previous["problem_sha256"]:
            raise ValueError("prior tracking problem receipt mismatch")
        if source_identity() != previous["sources"]:
            raise ValueError("estimation source/runtime drift; freeze a new candidate")
        immutable = set(old) - {"measurements", "closed_through_s", "events", "epoch_covariances", "sources"}
        if old.get("analysis_mode") == "continuous":
            immutable.discard("fit_through_s")
            if problem["fit_through_s"] < old["fit_through_s"]:
                raise ValueError("continuous fit boundary cannot move backwards")
        if any(canonical_digest(problem.get(k)) != canonical_digest(old[k]) for k in immutable):
            raise ValueError("append cannot change initial conditions, physics, parameters, fit boundary or gates")
        old_rows, _ = normalize_rows(old["measurements"])
        new_rows, _ = normalize_rows(problem["measurements"])
        known = {r["measurement_id"]: r for r in old_rows}
        for row in new_rows:
            if row["measurement_id"] not in known and row["time_s"] <= old["closed_through_s"]:
                raise ValueError("cannot append a measurement to an already closed epoch")
        if problem["closed_through_s"] < old["closed_through_s"]:
            raise ValueError("closed watermark cannot move backwards")
        problem["measurements"] = old_rows + new_rows
        problem["sources"] = list(
            {canonical_digest(r): r for r in old.get("sources", []) + problem.get("sources", [])}.values()
        )
        # Epoch covariance for closed epochs cannot be rewritten or omitted.
        merged = dict(old.get("epoch_covariances", {}))
        for k, v in problem.get("epoch_covariances", {}).items():
            if k in merged and canonical_digest(v) != canonical_digest(merged[k]):
                raise ValueError("cannot change processed epoch covariance")
            merged[k] = v
        problem["epoch_covariances"] = merged
    rows, groups, dynamics, measurement, duplicates = validate_problem(problem)
    problem["measurements"] = rows
    revision = f"revision-{0 if previous is None else previous['index'] + 1:06d}"
    stage = root / (revision + ".building")
    build_identity = {"problem_sha256": canonical_digest(problem), "dynamics": dynamics.identity()}
    if (root / revision).exists():
        # Recover a completed revision whose atomic pointer was not yet published.
        completed = root / revision
        manifest = json.loads((completed / "manifest.json").read_text())
        if (
            manifest["problem_sha256"] != canonical_digest(problem)
            or manifest["sources"] != source_identity()
            or manifest["parent"] != (None if previous is None else canonical_digest(previous))
            or manifest["result_sha256"] != file_sha(completed / "result.json")
            or manifest["checkpoint_sha256"] != file_sha(completed / "filter_checkpoint.json")
        ):
            raise ValueError("unpublished revision identity mismatch; retain for inspection")
        write(root / "current.next.json", manifest)
        os.replace(root / "current.next.json", current)
        return json.loads((completed / "result.json").read_text())
    recovering = stage.exists()
    if recovering:
        if json.loads((stage / "build_identity.json").read_text()) != build_identity:
            raise ValueError("incomplete revision request/source drift; preserve and use a new output")
    else:
        stage.mkdir(exist_ok=False)
        write(stage / "build_identity.json", build_identity)
        write(stage / "problem.json", problem)
    checkpoint = stage / "filter_checkpoint.json"
    if previous and not recovering:
        prior_checkpoint = root / previous["revision"] / "filter_checkpoint.json"
        if file_sha(prior_checkpoint) != previous["checkpoint_sha256"]:
            raise ValueError("checkpoint source receipt mismatch")
        shutil.copyfile(prior_checkpoint, checkpoint)
    fit = [g for g in groups if g["time_s"] <= problem["fit_through_s"]]
    config = SequentialODConfig(
        numeric_backend=dynamics.propagator.numeric_backend,
        reinitialize_after_maneuver=False,
        integration_substep_s=problem["step_s"],
        innovation_gate_probability=problem.get("gate_probability", 0.999),
    )
    report = run_sequential_orbit_od(
        ObservationPacket({"object_id": problem["object_id"], "frame": "ECI", "observations": fit}),
        initial_state_eci_km_s=problem["initial_state"],
        initial_covariance=problem["initial_covariance"],
        config=config,
        propagator=dynamics.propagator,
        orbit_context=dynamics.context,
        measurement_model=measurement,
        measurement_model_id=SCHEMA,
        dynamics_model=dynamics,
        checkpoint_path=checkpoint,
        resume=checkpoint.exists(),
        allow_append=True,
    )
    terminal = report["records"][-1]
    x, p, t = np.asarray(terminal["filtered_state"]), np.asarray(terminal["filtered_covariance"]), terminal["time_s"]
    forecasts = []
    for row in groups:
        if row["time_s"] <= problem["fit_through_s"]:
            continue
        x, phi, q = dynamics.predict(x, t, row["time_s"] - t)
        p = dynamics.stabilize(phi @ p @ phi.T + q)
        t = row["time_s"]
        v = measurement(row, x)
        h, r, y = v["jacobian"], v["covariance"], v["innovation"]
        s = h @ p @ h.T + r
        np.linalg.cholesky(s)
        forecasts.append(
            {
                "time_s": t,
                "state": x.tolist(),
                "covariance": p.tolist(),
                "innovation": y.tolist(),
                "innovation_covariance": s.tolist(),
                "nis": float(y @ np.linalg.solve(s, y)),
                "nis_threshold": chi_square_threshold(len(y), problem.get("gate_probability", 0.999)),
                "assimilated": False,
            }
        )
    # Data-only observability, scaled by declared prior sigmas. Priors and
    # covariance inflation cannot manufacture force-parameter information.
    cumulative = np.eye(len(dynamics.state_names))
    design = []
    for group, record in zip(fit, report["records"], strict=True):
        cumulative = np.asarray(record["transition_from_previous"]) @ cumulative
        value = measurement(group, np.asarray(record["predicted_state"]))
        design.append(np.linalg.solve(np.linalg.cholesky(value["covariance"]), value["jacobian"] @ cumulative))
    scaled = np.vstack(design) * np.sqrt(np.diag(np.asarray(problem["initial_covariance"])))
    _u, singular, vt = np.linalg.svd(scaled, full_matrices=scaled.shape[0] < scaled.shape[1])
    threshold = max(float(singular[0]) * 1e-8, 1e-3)
    rank = int(np.count_nonzero(singular > threshold))
    null = vt[rank:]
    weak = [
        name
        for i, name in enumerate(dynamics.state_names[6:], 6)
        if dynamics.parameters[i - 6].estimate
        and (
            np.linalg.norm(null[:, i]) > 0.01
            or terminal["filtered_covariance"][i][i] > 0.9 * problem["initial_covariance"][i][i]
        )
    ]
    report["identifiability"] = {
        "data_rank": rank,
        "dimension": len(dynamics.state_names),
        "scaled_singular_values": singular.tolist(),
        "rank_threshold": threshold,
        "weak_estimated_parameters": weak,
        "parameter_claim_gate_passed": not weak,
    }
    report.update(
        workflow_schema=SCHEMA,
        holdout=forecasts,
        duplicate_rows_ignored=duplicates,
        qualification={"empirically_calibrated": False, "operationally_qualified": False},
        uncertainty_policy="full joint state/parameter covariance; inactive slots retained as consider states",
    )
    from sim.estimation.sequential_od import _write_artifacts

    _write_artifacts(stage / "filter", report, report["records"])
    report["review_store"] = "filter/review/run.sqlite"
    write(stage / "result.json", report)
    sources = source_identity()
    verify_sources(problem.get("sources", []))
    if sources != dynamics.model_identity["sources"] or any(
        file_sha(path) != digest for path, digest in dynamics.model_identity["assets"].items()
    ):
        raise ValueError("source drift during estimation; retain the incomplete revision")
    manifest = {
        "schema": SCHEMA,
        "index": int(revision.split("-")[-1]),
        "revision": revision,
        "problem_sha256": canonical_digest(problem),
        "sources": sources,
        "checkpoint_sha256": file_sha(checkpoint),
        "result_sha256": file_sha(stage / "result.json"),
        "review_sha256": file_sha(stage / "filter/review/run.sqlite"),
        "parent": None if previous is None else canonical_digest(previous),
    }
    write(stage / "manifest.json", manifest)
    stage.rename(root / revision)
    write(root / "current.next.json", manifest)
    os.replace(root / "current.next.json", current)
    return report


def replay_continuous_tracking_od(output_dir, replay_dir):
    root = Path(output_dir).resolve()
    current = json.loads((root / "current.json").read_text())
    revision = root / current["revision"]
    problem = json.loads((revision / "problem.json").read_text())
    if canonical_digest(problem) != current["problem_sha256"] or source_identity() != current["sources"]:
        raise ValueError("replay problem or source identity mismatch")
    result = revision / "result.json"
    if file_sha(result) != current["result_sha256"]:
        raise ValueError("replay result receipt mismatch")
    if file_sha(revision / "filter/review/run.sqlite") != current["review_sha256"]:
        raise ValueError("review-store receipt mismatch")
    fresh = run_continuous_tracking_od(problem, replay_dir)
    old = json.loads(result.read_text())
    # Duplicate delivery counts are ingestion evidence, not numerical semantics.
    fresh.pop("duplicate_rows_ignored", None)
    old.pop("duplicate_rows_ignored", None)
    if canonical_digest(fresh) != canonical_digest(old):
        raise ValueError("authoritative continuous tracking replay differs")
    return {"status": "replay_verified", "result_sha256": current["result_sha256"]}
