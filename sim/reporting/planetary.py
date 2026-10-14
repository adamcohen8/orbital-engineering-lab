"""Body-neutral output assembly, separate from Earth reports and products."""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone

import numpy as np

STATE_COLUMNS = [
    "x_inertial_km",
    "y_inertial_km",
    "z_inertial_km",
    "vx_inertial_km_s",
    "vy_inertial_km_s",
    "vz_inertial_km_s",
]


def build_payload(c):
    from sim.installation.provenance import execution_provenance

    environment = c.planetary_environment
    if environment is None:
        raise ValueError("Planetary reporting requires the prepared run environment")
    environment.verify_resources()
    model = environment.metadata()
    diagnostics = {}
    for oid, history in c.truth_hist.items():
        plan = c.planetary_force_plans[oid]
        diagnostics[oid] = [
            plan.stage(float(t), row[:6])[1] for t, row in zip(c.t_s, history)
        ]
    summary = {
        "scenario_name": c.cfg.scenario_name,
        "scenario_description": c.cfg.scenario_description,
        "objects": list(c.object_ids),
        "samples": int(c.t_s.size),
        "dt_s": float(c.dt_s),
        "duration_s": float(c.t_s[-1]) if c.t_s.size else 0.0,
        "terminated_early": c.terminated_early,
        "termination_reason": c.termination_reason,
        "termination_time_s": c.termination_time_s,
        "termination_object_id": c.termination_object_id,
        "central_body": environment.body.body_id,
        "origin_center_id": environment.body.center_id,
        "inertial_axes": "J2000",
        "planetary_environment": model,
        "object_initialization": {
            oid: {
                "state_form": (
                    "coes"
                    if "coes" in c.cfg.objects[oid].initial_state
                    else "cartesian"
                ),
                "origin_center_id": environment.body.center_id,
                "inertial_axes": "J2000",
                "mu_km3_s2": environment.gravity.mu_km3_s2,
                "initial_jd_utc": environment.initial_jd_utc,
            }
            for oid in c.object_ids
        },
        "object_propagation": dict(c.object_propagation),
        "runtime_profile": dict(c.runtime_profile),
        "object_runtime_profiles": {oid: "trajectory_only" for oid in c.object_ids},
        "installation_provenance": execution_provenance(),
        "thrust_stats": c.thrust_stats,
        "plot_outputs": {},
        "animation_outputs": {},
        "attitude_guardrail_stats": {},
    }
    return {
        "summary": summary,
        "time_s": c.t_s.tolist(),
        "planetary_environment": model,
        "object_state_frames": {oid: "planetary_inertial" for oid in c.object_ids},
        "state_columns": STATE_COLUMNS
        + [
            "q0",
            "q1",
            "q2",
            "q3",
            "wx_body_rad_s",
            "wy_body_rad_s",
            "wz_body_rad_s",
            "mass_kg",
        ],
        "truth_by_object": {oid: h.tolist() for oid, h in c.truth_hist.items()},
        "belief_by_object": {},
        "target_reference_orbit_truth": [],
        "applied_thrust_by_object": {
            oid: h.tolist() for oid, h in c.thrust_hist.items()
        },
        "applied_torque_by_object": {
            oid: h.tolist() for oid, h in c.torque_hist.items()
        },
        "object_initialization": {
            oid: {
                "state_form": (
                    "coes"
                    if "coes" in c.cfg.objects[oid].initial_state
                    else "cartesian"
                ),
                "origin_center_id": environment.body.center_id,
                "inertial_axes": "J2000",
                "mu_km3_s2": environment.gravity.mu_km3_s2,
                "initial_jd_utc": environment.initial_jd_utc,
            }
            for oid in c.object_ids
        },
        "object_propagation": dict(c.object_propagation),
        "object_runtime_profiles": summary["object_runtime_profiles"],
        "planetary_diagnostics_by_object": diagnostics,
    }


def write_artifacts(payload, c):
    from sim.reporting.output_index import write_output_index
    from sim.reporting.review_store import write_single_run_review_store
    from sim.utils.io import write_json

    summary = payload["summary"]
    outdir = c.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    data = c.cfg.to_dict()
    config_json = json.dumps(
        data, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    digest = hashlib.sha256(config_json.encode()).hexdigest()
    summary.update(
        config_sha256=digest,
        generated_utc=datetime.now(timezone.utc).isoformat(),
        review_detail=str(c.cfg.outputs.review.detail),
    )
    source = getattr(c.cfg, "source_path", None)
    if source is not None:
        summary["config_source_path"] = str(source)
    config_path = outdir / "effective_config.json"
    write_json(str(config_path), data)
    summary["effective_config_path"] = str(config_path)
    artifacts = {
        **dict(c.extra_artifacts or {}),
        "effective_config_json": str(config_path),
    }
    if c.cfg.outputs.stats.get("save_csv", False):
        path = outdir / "master_run_history.csv"
        with path.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(
                [
                    "object_id",
                    "sample_index",
                    "time_s",
                    "origin_center_id",
                    "inertial_axes",
                ]
                + STATE_COLUMNS
                + ["mass_kg"]
            )
            for oid, history in c.truth_hist.items():
                for i, (t, row) in enumerate(zip(c.t_s, history)):
                    writer.writerow(
                        [oid, i, float(t), summary["origin_center_id"], "J2000"]
                        + row[:6].tolist()
                        + [float(row[13])]
                    )
        artifacts["history_csv"] = str(path)
    if c.cfg.outputs.stats.get("save_history_npz", False):
        path = outdir / "master_run_history.npz"
        arrays = {
            "time_s": c.t_s,
            "origin_center_id": np.array(summary["origin_center_id"]),
            "inertial_axes": np.array("J2000"),
            "state_columns": np.array(payload["state_columns"]),
            "environment_sha256": np.array(
                summary["planetary_environment"]["environment_sha256"]
            ),
        }
        arrays.update({f"truth_{oid}": h for oid, h in c.truth_hist.items()})
        np.savez_compressed(path, **arrays)
        artifacts["history_npz"] = str(path)
    if c.cfg.outputs.stats.get("save_json", True):
        artifacts["summary_json"] = str(outdir / "master_run_summary.json")
    if c.cfg.outputs.stats.get("save_full_log", False):
        artifacts["run_log_json"] = str(outdir / "master_run_log.json")
    artifacts["output_index_md"] = str(outdir / "index.md")
    if c.cfg.outputs.review.enabled:
        artifacts["review_store"] = {
            "sqlite": str(outdir / "review/run.sqlite"),
            "schema_json": str(outdir / "review/schema.json"),
        }
    review = write_single_run_review_store(
        payload=payload,
        context=c,
        artifacts=artifacts,
        config_json=config_json,
        config_sha256=digest,
    )
    if review:
        artifacts["review_store"] = review
        summary.update(
            review_outputs=review,
            review_sqlite_path=review["sqlite"],
            review_store_status="available",
        )
        payload["review_outputs"] = review
    index = write_output_index(
        outdir=outdir,
        workflow="single_run",
        title=c.cfg.scenario_name,
        summary=summary,
        artifacts=artifacts,
    )
    summary["output_index_md"] = str(index)
    payload["output_index_md"] = str(index)
    if c.cfg.outputs.stats.get("save_json", True):
        write_json(str(outdir / "master_run_summary.json"), summary)
    if c.cfg.outputs.stats.get("save_full_log", False):
        write_json(str(outdir / "master_run_log.json"), payload)
    if c.cfg.outputs.stats.get("print_summary", True):
        from sim.reporting.single_run_artifacts import format_single_run_summary

        print(format_single_run_summary(summary))
    return payload
