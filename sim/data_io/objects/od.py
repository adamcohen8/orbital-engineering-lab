"""Read a completed public tracking-OD packet without rerunning its estimator."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from sim.data_io.common import read_text, source_receipt, strict_json
from sim.data_io.contracts import DataIOError, Trajectory
from sim.data_io.time import parse_time
from sim.frame_time import epoch_julian_date


def locate_tracking_od(source):
    path = Path(source).expanduser().resolve()
    candidates = (
        [path]
        if path.is_file()
        else [
            path / "tracking_od_evidence.json",
            path.parent / "tracking_od_evidence.json" if path.name == "estimator" else path / "__no_parent_search__",
        ]
    )
    return next((p for p in candidates if p.name == "tracking_od_evidence.json" and p.is_file()), None)


def _load(source):
    path = locate_tracking_od(source)
    if path is None:
        raise DataIOError("Select a completed tracking-OD evidence file or its workflow/estimator directory.")
    original = source_receipt(path)
    evidence = strict_json(read_text(path))
    if evidence.get("schema_version") != "oel.tracking_od_evidence.v1" or evidence.get("status") != "completed":
        raise DataIOError("Expected completed oel.tracking_od_evidence.v1 evidence.")
    identities = [original]
    artifacts = {}
    for name, row in evidence["artifacts"].items():
        artifact = (path.parent / row["path"]).resolve()
        if not artifact.is_relative_to(path.parent):
            raise DataIOError("Tracking OD artifact paths must remain inside the completed packet.")
        identity = source_receipt(artifact)
        if identity["sha256"] != row["sha256"] or identity["size_bytes"] != row["bytes"]:
            raise DataIOError("Tracking OD artifact identity mismatch: " + name)
        identities.append(identity)
        artifacts[name] = artifact
    estimator = evidence["estimator"]
    report = strict_json(read_text(artifacts["estimator_report"]))
    packet = strict_json(read_text(artifacts["fitted_state_packet"]))
    object_id = evidence["input"]["object_id"]
    packet_state = packet["objects"][object_id]["initial_state"]
    vector = packet_state["position_eci_km"] + packet_state["velocity_eci_km_s"]
    if vector != estimator["fitted_state_eci_km_km_s"] or vector != report["fitted_state_eci_km_s"]:
        raise DataIOError("Tracking OD fitted-state artifacts disagree.")
    if (
        estimator["state_frame"] != "ECI"
        or packet_state["epoch_jd_utc"] != estimator["state_epoch_jd_utc"]
        or report["quality_gates"] != estimator["quality_gates"]
        or report["verdict"] != estimator["verdict"]
        or report["state_covariance_eci_km_s"] != estimator["state_covariance_eci"]
        or abs(epoch_julian_date(parse_time(estimator["state_epoch_utc"]), "UTC") - estimator["state_epoch_jd_utc"])
        * 86400
        > 5e-5
    ):
        raise DataIOError("Tracking OD epoch, frame, or quality artifacts disagree.")
    gates = estimator["quality_gates"]
    blockers = [
        {"code": "tracking_od." + name}
        for name in ("solver_success", "data_full_rank", "fit_improved_prefit_rms")
        if gates.get(name) is not True
    ]
    if (
        gates.get("holdout_evidence_status") != "evaluated"
        or estimator["verdict"].get("analyst_action") != "usable_for_propagation_study"
    ):
        blockers.append({"code": "tracking_od.analyst_review_required"})
    if source_receipt(path) != original:
        raise DataIOError("Tracking OD evidence changed while reading.")
    return evidence, report, identities, {"valid": True, "promotable": not blockers, "blockers": blockers}


def verify_tracking_od_binding(binding):
    _, _, identities, decision = _load(binding["sources"][0]["path"])
    if identities != binding["sources"]:
        raise DataIOError("Tracking OD source evidence changed after import.")
    return decision


def read_tracking_od(source):
    evidence, report, sources, decision = _load(source)
    estimator = evidence["estimator"]
    covariance = estimator["state_covariance_eci"]
    if covariance is not None and not np.all(np.isfinite(np.asarray(covariance, dtype=float))):
        raise DataIOError("Tracking OD covariance is not finite.")
    orbit = dict(report["orbit_force_model"])
    orbit.setdefault("model", report["dynamics_model"])
    return Trajectory(
        evidence["input"]["object_id"],
        "EARTH",
        "EME2000",
        parse_time(estimator["state_epoch_utc"]),
        [0],
        [estimator["fitted_state_eci_km_km_s"]],
        covariance_times_s=[0] if covariance is not None else None,
        covariances=[covariance] if covariance is not None else None,
        metadata={
            "state_semantics": "estimated",
            "quality": {
                "disposition": "accepted" if decision["promotable"] else "review_required",
                "gates": estimator["quality_gates"],
                "verdict": estimator["verdict"],
            },
            "covariance_disposition": {"calibrated": False},
            "object_specs": report.get("object_specs", {}),
            "seed_simulator": {"dynamics": {"orbit": orbit}, "environment": report.get("environment", {})},
            "tracking_od_binding": {"sources": sources},
            "limitations": evidence["limitations"],
        },
    )
