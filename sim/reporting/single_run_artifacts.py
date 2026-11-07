from __future__ import annotations

import csv
import hashlib
import importlib
import json
import os
import re
import stat
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from sim.config import SimulationScenarioConfig
from sim.master_outputs import _expanded_figure_ids as _expanded_plot_figure_ids
from sim.master_outputs import animate_outputs as _animate_outputs_impl
from sim.master_outputs import plot_outputs as _plot_outputs_impl
from sim.reporting.ground_station_access_reports import write_ground_station_access_reports
from sim.reporting.output_index import write_output_index
from sim.reporting.review_store import write_single_run_review_store
from sim.runtime_support import _resolve_rocket_stack, _resolve_satellite_isp_s
from sim.security.output_writes import (
    bind_output_writer,
    current_output_writer,
    rewrite_staged_paths,
    staged_output_file,
)
from sim.utils.io import sha256_file, write_json


@dataclass(frozen=True)
class SingleRunArtifactContext:
    cfg: SimulationScenarioConfig
    outdir: Path
    t_s: np.ndarray
    truth_hist: dict[str, np.ndarray]
    target_reference_orbit_truth: np.ndarray | None
    belief_hist: dict[str, np.ndarray]
    thrust_hist: dict[str, np.ndarray]
    torque_hist: dict[str, np.ndarray]
    desired_attitude_hist: dict[str, np.ndarray]
    knowledge_hist: dict[str, dict[str, np.ndarray]]
    knowledge_measurement_hist: dict[str, dict[str, np.ndarray]]
    rocket_metrics: dict[str, np.ndarray]
    reentry_metrics: dict[str, dict[str, np.ndarray]]
    bridge_hist: dict[str, list[dict[str, Any]]]
    object_state_frames: dict[str, str]
    object_propagation: dict[str, Any] | None = None
    extra_artifacts: dict[str, Any] | None = None


def _writer_logical_path(path: str | Path, writer: Any | None = None) -> Path:
    """Map a producer path to the writer's retained logical namespace.

    A protected execution may hand the engine either its logical output root or
    the private path-facing staging root.  Both names describe the same output
    tree; this helper keeps artifact metadata on the logical name while the
    writer controls the physical destination.
    """

    active = writer or current_output_writer()
    raw = Path(path).expanduser()
    if not raw.is_absolute():
        raw = Path.cwd() / raw
    if active is None:
        return raw
    logical_root = Path(active.logical_root).expanduser()
    physical_root = Path(active.root).expanduser()
    for candidate, mapped_root in ((logical_root, logical_root), (physical_root, logical_root)):
        try:
            return mapped_root / raw.relative_to(candidate)
        except ValueError:
            continue
    raise ValueError(f"Output path {raw} is outside the protected writer roots.")


def _invoke_path_writer(function: Any, *, output_dir: Path, **kwargs: Any) -> Any:
    """Run a legacy path-facing producer in private staging when protected.

    Several optional report producers still accept only a filesystem path.  A
    private staging directory lets those producers retain their CLI behavior
    while the final publication uses the already-open descriptor writer.
    """

    active = current_output_writer()
    if active is None:
        return function(output_dir=output_dir, **kwargs)
    staging = active.create_staging(prefix=".oel-report-")
    try:
        with bind_output_writer(None):
            result = function(output_dir=staging.path, **kwargs)
        active.publish_staging(staging)
        return rewrite_staged_paths(result, staging.path, Path(active.logical_root))
    finally:
        if not staging._closed:
            staging.remove_tree()


def format_single_run_summary(summary: dict[str, Any]) -> str:
    lines: list[str] = []
    scenario_description = str(summary.get("scenario_description", "") or "").strip()
    lines.append("")
    lines.append("=" * 72)
    lines.append("MASTER SIMULATION SUMMARY")
    lines.append("=" * 72)
    lines.append(f"Scenario   : {summary.get('scenario_name', 'unknown')}")
    if scenario_description:
        lines.append(f"Desc       : {scenario_description}")
    objects = [str(item) for item in list(summary.get("objects", []) or [])]
    lines.append(f"Objects    : {', '.join(objects)}")
    lines.append(f"Samples    : {summary.get('samples', 0)}")
    lines.append(
        f"Timing     : dt={_fmt_float(float(summary.get('dt_s', 0.0)), 3)} s, "
        f"duration={_fmt_float(float(summary.get('duration_s', 0.0)), 1)} s"
    )
    lines.append("-" * 72)
    if bool(summary.get("terminated_early", False)):
        lines.append(
            "Termination: EARLY "
            f"(reason={summary.get('termination_reason')}, "
            f"t={summary.get('termination_time_s')}, "
            f"object={summary.get('termination_object_id')})"
        )
    else:
        lines.append("Termination: nominal (full duration reached)")
    review_status = str(summary.get("review_store_status", "") or "").strip()
    if review_status:
        lines.append(f"Evidence   : {review_status}")
    if "rocket" in objects and "rocket_insertion_achieved" in summary:
        ins_ok = bool(summary.get("rocket_insertion_achieved", False))
        ins_t = summary.get("rocket_insertion_time_s")
        lines.append(f"Insertion  : achieved at t={ins_t}" if ins_ok else "Insertion  : not achieved")
    reentry_summary = dict(summary.get("reentry_summary_by_object", {}) or {})
    if reentry_summary:
        entered = [oid for oid, item in reentry_summary.items() if bool(dict(item or {}).get("entered_reentry"))]
        lines.append(
            "Re-entry   : "
            + (", ".join(sorted(entered)) if entered else "configured, threshold not crossed")
        )
    thrust_stats = dict(summary.get("thrust_stats", {}) or {})
    if thrust_stats:
        lines.append("-" * 72)
        lines.append("Thrust Stats")
        lines.append(f"{'Object':<14}{'Burn Samples':>14}{'Max Accel (km/s^2)':>24}{'Total dV (m/s)':>18}")
        for oid in sorted(thrust_stats.keys()):
            stats = dict(thrust_stats.get(oid, {}) or {})
            lines.append(
                f"{oid:<14}"
                f"{int(stats.get('burn_samples', 0)):>14d}"
                f"{float(stats.get('max_accel_km_s2', 0.0)):>24.3e}"
                f"{float(stats.get('total_dv_m_s', 0.0)):>18.3f}"
            )
    mission_recovery = dict(summary.get("mission_recovery", {}) or {})
    if mission_recovery:
        estimate = dict(mission_recovery.get("recovery_estimate", {}) or {})
        lines.append("-" * 72)
        lines.append(
            f"{estimate.get('display_name', 'Original-Orbit Recovery Estimate')}: "
            f"{mission_recovery.get('object_id', '')} {mission_recovery.get('goal', '')} "
            f"dV={_fmt_optional_float(estimate.get('recovery_delta_v_m_s'), 3)} m/s "
            f"post-assessment time={_fmt_optional_float(estimate.get('recovery_time_s'), 1)} s "
            f"run-start completion={_fmt_optional_float(estimate.get('run_start_to_estimated_recovery_s'), 1)} s "
            f"method={estimate.get('method', '')}"
        )
        planner = dict(mission_recovery.get("planner", {}) or {})
        recommendations = dict(planner.get("recommended", {}) or {})
        candidates = {
            str(candidate.get("candidate_id")): dict(candidate or {})
            for candidate in list(planner.get("candidates", []) or [])
        }
        for mode in ("min_time", "min_delta_v", "constrained"):
            candidate_id = recommendations.get(mode)
            candidate = candidates.get(str(candidate_id))
            if not candidate:
                continue
            lines.append(
                f"  {mode:<13}: "
                f"{_fmt_optional_float(candidate.get('planned_delta_v_m_s'), 3)} m/s, "
                f"post-assessment {_fmt_optional_float(candidate.get('planned_time_s'), 1)} s "
                f"(run-start {_fmt_optional_float(candidate.get('run_start_to_planned_completion_s'), 1)} s), "
                f"verified={bool(candidate.get('verified', False))}, "
                f"source={candidate.get('source_family', candidate.get('source', ''))}, "
                f"target={candidate.get('target_basis', '')}"
            )
    orbital = dict(summary.get("orbital_analysis", {}) or {})
    for row in list(orbital.get("coverage", []) or []):
        item = dict(row or {})
        lines.append(
            "Coverage   : "
            f"{item.get('analysis_id', '')} mean={100.0 * float(item.get('time_weighted_mean_covered_fraction', 0.0)):.3f}% "
            f"ever={100.0 * float(item.get('ever_covered_fraction', 0.0)):.3f}%"
        )
    for row in list(orbital.get("directed_links", []) or []):
        item = dict(row or {})
        lines.append(
            "Link       : "
            f"{item.get('analysis_id', '')} available={100.0 * float(item.get('sampled_available_fraction', 0.0)):.3f}% "
            f"duration={float(item.get('available_duration_s', 0.0)):.3f} s"
        )
    plot_outputs = dict(summary.get("plot_outputs", {}) or {})
    anim_outputs = dict(summary.get("animation_outputs", {}) or {})
    guardrails = dict(summary.get("attitude_guardrail_stats", {}) or {})
    lines.append("-" * 72)
    lines.append(f"Artifacts  : plots={len(plot_outputs)}  animations={len(anim_outputs)}")
    lines.append(f"Guardrails : attitude_events={int(sum(int(v) for v in guardrails.values())) if guardrails else 0}")
    lines.append("=" * 72)
    return "\n".join(lines)


def write_single_run_artifacts(
    payload: dict[str, Any],
    context: SingleRunArtifactContext,
) -> dict[str, Any]:
    if context.cfg.simulator.central_body != "earth":
        from sim.reporting.planetary import write_artifacts
        return write_artifacts(payload, context)
    writer = current_output_writer()
    output_dir = _writer_logical_path(context.outdir, writer)
    summary = payload.setdefault("summary", {})
    orbital_analysis: dict[str, Any] = {}
    if bool(context.cfg.outputs.orbital_analysis.enabled):
        from sim.reporting.orbital_analysis import run_scenario_orbital_analysis

        orbital_analysis = run_scenario_orbital_analysis(context=context)
        payload["_orbital_analysis_review"] = orbital_analysis
        payload["orbital_analysis"] = {
            "schema_version": orbital_analysis.get("schema_version"),
            "coverage": [
                {
                    key: value
                    for key, value in item.items()
                    if key not in {"samples", "intervals", "transitions"}
                }
                for item in orbital_analysis.get("coverage", [])
            ],
            "directed_links": [
                {
                    key: value
                    for key, value in item.items()
                    if key not in {"samples", "windows", "transitions"}
                }
                for item in orbital_analysis.get("directed_links", [])
            ],
        }
        summary["orbital_analysis"] = {
            "coverage_analysis_count": len(orbital_analysis.get("coverage", [])),
            "directed_link_analysis_count": len(orbital_analysis.get("directed_links", [])),
            "schema_version": orbital_analysis.get("schema_version"),
            "coverage": [
                {
                    "analysis_id": item.get("analysis_id"),
                    "source_object_id": item.get("source_object_id"),
                    "sample_count": dict(item.get("summary", {}) or {}).get("sample_count"),
                    "instantaneous_covered_fraction_min": dict(item.get("summary", {}) or {}).get(
                        "instantaneous_covered_fraction_min"
                    ),
                    "instantaneous_covered_fraction_max": dict(item.get("summary", {}) or {}).get(
                        "instantaneous_covered_fraction_max"
                    ),
                    "time_weighted_mean_covered_fraction": dict(item.get("summary", {}) or {}).get(
                        "time_weighted_mean_covered_fraction"
                    ),
                    "ever_covered_fraction": dict(item.get("summary", {}) or {}).get("ever_covered_fraction"),
                }
                for item in orbital_analysis.get("coverage", [])
            ],
            "directed_links": [
                {
                    "analysis_id": item.get("analysis_id"),
                    "link_id": item.get("link_id"),
                    "tx_endpoint_id": item.get("tx_object_id"),
                    "rx_endpoint_id": item.get("rx_object_id"),
                    "tx_endpoint_kind": item.get("tx_endpoint_kind", "spacecraft"),
                    "rx_endpoint_kind": item.get("rx_endpoint_kind", "spacecraft"),
                    "sample_count": dict(item.get("summary", {}) or {}).get("sample_count"),
                    "available_sample_count": dict(item.get("summary", {}) or {}).get(
                        "available_sample_count"
                    ),
                    "interval_count": dict(item.get("summary", {}) or {}).get("interval_count"),
                    "sampled_available_fraction": dict(item.get("summary", {}) or {}).get(
                        "sampled_available_fraction"
                    ),
                    "available_duration_s": dict(item.get("summary", {}) or {}).get("available_duration_s"),
                    "minimum_margin_db": dict(dict(item.get("summary", {}) or {}).get("margin_db", {}) or {}).get(
                        "minimum"
                    ),
                    "maximum_margin_db": dict(dict(item.get("summary", {}) or {}).get("margin_db", {}) or {}).get(
                        "maximum"
                    ),
                    "estimated_delivered_data_bits": dict(item.get("summary", {}) or {}).get(
                        "estimated_delivered_data_bits"
                    ),
                    "required_eb_n0_db": dict(item.get("summary", {}) or {}).get("required_eb_n0_db"),
                    "minimum_fixed_site_elevation_deg": dict(item.get("summary", {}) or {}).get(
                        "minimum_fixed_site_elevation_deg"
                    ),
                    "maximum_range_km_threshold": dict(item.get("summary", {}) or {}).get(
                        "maximum_range_km_threshold"
                    ),
                    "primary_reason_sample_count": dict(item.get("summary", {}) or {}).get(
                        "primary_reason_sample_count"
                    ),
                }
                for item in orbital_analysis.get("directed_links", [])
            ],
        }
    previous_owned_artifacts = _previous_owned_artifacts(output_dir)
    source_path = getattr(context.cfg, "source_path", None)
    if source_path is not None:
        summary_source_path = Path(source_path).resolve()
        if writer is not None:
            summary_source_path = _writer_logical_path(source_path, writer)
        summary["config_source_path"] = str(summary_source_path)
    config_data = context.cfg.to_dict()
    if writer is not None:
        config_outputs = config_data.get("outputs")
        if isinstance(config_outputs, dict) and config_outputs.get("output_dir"):
            config_outputs["output_dir"] = str(_writer_logical_path(config_outputs["output_dir"], writer))
    config_json = json.dumps(config_data, sort_keys=True, separators=(",", ":"), allow_nan=False)
    config_sha256 = hashlib.sha256(config_json.encode("utf-8")).hexdigest()
    summary["config_sha256"] = config_sha256
    summary["generated_utc"] = os.environ.get("OEL_GENERATED_UTC", "").strip()
    summary["review_detail"] = str(context.cfg.outputs.review.detail)
    _add_relative_range_summary(summary=summary, context=context)
    trajectory_only = all(
        str(getattr(object_cfg, "runtime_profile", "") or "") == "trajectory_only"
        for object_cfg in context.cfg.objects.values()
    )
    if trajectory_only:
        mission_recovery = {}
    else:
        from sim.analysis.mission_recovery import build_mission_recovery_summary

        mission_recovery = build_mission_recovery_summary(
            cfg=context.cfg,
            t_s=context.t_s,
            truth_hist=context.truth_hist,
        )
    if mission_recovery:
        summary["mission_recovery"] = mission_recovery
        if source_path is not None and dict(mission_recovery.get("planner", {}) or {}).get("candidates"):
            try:
                from sim.interchange.adapters.planning import emit_mission_recovery_scenario_patches

                emission = _invoke_path_writer(
                    lambda *, output_dir, **_: emit_mission_recovery_scenario_patches(
                        mission_recovery,
                        source_scenario=source_path,
                        output_dir=output_dir,
                    ),
                    output_dir=output_dir / "scenario_patches",
                )
            except (OSError, ValueError) as exc:
                emission = {"status": "not_emitted", "reason": str(exc), "selection_required": True}
            mission_recovery["scenario_patch_emission"] = emission
    plot_outputs = _invoke_path_writer(
        lambda *, output_dir, **_: _plot_outputs(
            cfg=context.cfg,
            t_s=context.t_s,
            truth_hist=context.truth_hist,
            target_reference_orbit_truth=context.target_reference_orbit_truth,
            belief_hist=context.belief_hist,
            thrust_hist=context.thrust_hist,
            desired_attitude_hist=context.desired_attitude_hist,
            knowledge_hist=context.knowledge_hist,
            knowledge_measurement_hist=context.knowledge_measurement_hist,
            rocket_metrics=context.rocket_metrics if context.rocket_metrics else None,
            reentry_metrics=context.reentry_metrics,
            bridge_hist=context.bridge_hist,
            ground_station_access=dict(payload.get("ground_station_access", {}) or {}),
            outdir=output_dir,
        ),
        output_dir=output_dir,
    )
    if orbital_analysis:
        plot_outputs = dict(plot_outputs)
        for item in orbital_analysis.get("coverage", []):
            plot_path = str(dict(item.get("artifacts", {}) or {}).get("fraction_plot_png") or "")
            if plot_path:
                plot_outputs[f"coverage:{item.get('analysis_id')}"] = plot_path
        for item in orbital_analysis.get("directed_links", []):
            plot_path = str(dict(item.get("artifacts", {}) or {}).get("margin_plot_png") or "")
            if plot_path:
                plot_outputs[f"directed_link:{item.get('analysis_id')}"] = plot_path
    mission_recovery_plot = _invoke_path_writer(
        lambda *, output_dir, **_: _write_mission_recovery_trade_space_plot(
            cfg=context.cfg,
            mission_recovery=mission_recovery,
            outdir=output_dir,
        ),
        output_dir=output_dir,
    )
    if mission_recovery_plot:
        plot_outputs = dict(plot_outputs)
        plot_outputs["mission_recovery_trade_space"] = mission_recovery_plot
    animation_outputs = _invoke_path_writer(
        lambda *, output_dir, **_: _animate_outputs(
            cfg=context.cfg,
            t_s=context.t_s,
            truth_hist=context.truth_hist,
            thrust_hist=context.thrust_hist,
            target_reference_orbit_truth=context.target_reference_orbit_truth,
            outdir=output_dir,
        ),
        output_dir=output_dir,
    )
    summary["plot_outputs"] = plot_outputs
    summary["animation_outputs"] = animation_outputs
    access_report_paths, access_report_views = _invoke_path_writer(
        lambda *, output_dir, **_: write_ground_station_access_reports(
            outdir=output_dir,
            ground_station_access=dict(payload.get("ground_station_access", {}) or {}),
            ground_station_access_summary=dict(payload.get("ground_station_access_summary", {}) or {}),
            t_s=context.t_s,
            initial_jd_utc=context.cfg.simulator.initial_jd_utc,
        ),
        output_dir=output_dir,
    )
    if access_report_paths:
        summary["ground_station_access_report_epoch_utc"] = access_report_views.get("epoch_utc")
        summary["ground_station_access_report_epoch_jd_utc"] = access_report_views.get("epoch_jd_utc")
        summary["ground_station_access_report_outputs"] = access_report_paths
        payload["ground_station_access_report_views"] = access_report_views
    bridge_outputs, bridge_summary = _invoke_path_writer(
        lambda *, output_dir, **_: _write_private_bridge_artifacts(
            outdir=output_dir,
            bridge_hist=context.bridge_hist,
        ),
        output_dir=output_dir,
    )
    if bridge_outputs:
        summary["bridge_extension_outputs"] = bridge_outputs
        summary["bridge_extension_summary"] = bridge_summary

    effective_config_path = output_dir / "effective_config.json"
    write_json(str(effective_config_path), config_data)
    summary["effective_config_path"] = str(effective_config_path)
    artifacts: dict[str, Any] = {
        **dict(context.extra_artifacts or {}),
        "effective_config_json": str(effective_config_path),
    }
    if payload.get("orbital_analysis"):
        artifacts["orbital_analysis"] = {
            "coverage": [item.get("artifacts", {}) for item in payload["orbital_analysis"].get("coverage", [])],
            "directed_links": [item.get("artifacts", {}) for item in payload["orbital_analysis"].get("directed_links", [])],
        }
    if payload.get("ground_segment"):
        from sim.ground_segment.artifacts import write_ground_artifacts
        artifacts.update(
            _invoke_path_writer(
                lambda *, output_dir, **_: write_ground_artifacts(output_dir, payload["ground_segment"]),
                output_dir=output_dir,
            )
        )
    if payload.get("spacecraft_resources"):
        from sim.spacecraft_resources.artifacts import write_resource_artifacts

        artifacts.update(
            _invoke_path_writer(
                lambda *, output_dir, **_: write_resource_artifacts(output_dir, payload["spacecraft_resources"]),
                output_dir=output_dir,
            )
        )
    if bool(context.cfg.outputs.stats.get("save_csv", False)):
        artifacts["history_csv"] = str(_write_history_csv(context=context))
    if bool(context.cfg.outputs.stats.get("save_history_npz", False)):
        history_outputs = _write_history_npz(context=context)
        if history_outputs:
            summary["history_binary_outputs"] = history_outputs
            payload["history_binary_outputs"] = history_outputs
            artifacts["history_npz"] = str(history_outputs["npz"])
    else:
        stale_history = output_dir / "master_run_history.npz"
        if writer is None and stale_history.exists():
            stale_history.unlink()
    if bool(context.cfg.outputs.stats.get("save_json", True)):
        artifacts["summary_json"] = str(output_dir / "master_run_summary.json")
    if bool(context.cfg.outputs.stats.get("save_full_log", True)):
        artifacts["run_log_json"] = str(output_dir / "master_run_log.json")
    if plot_outputs:
        artifacts["plots"] = plot_outputs
    if animation_outputs:
        artifacts["animations"] = animation_outputs
    if access_report_paths:
        artifacts["ground_station_access_reports"] = access_report_paths
    if bridge_outputs:
        artifacts["bridge_extensions"] = bridge_outputs
    if writer is None:
        _prune_stale_owned_artifacts(
            output_dir=output_dir,
            previous=previous_owned_artifacts,
            current=_artifact_file_paths(artifacts),
        )
    artifacts["output_index_md"] = str(output_dir / "index.md")
    if bool(context.cfg.outputs.review.enabled):
        artifacts["review_store"] = {
            "sqlite": str(output_dir / "review" / "run.sqlite"),
            "schema_json": str(output_dir / "review" / "schema.json"),
        }
    review_outputs = _write_review_store(
        payload=payload,
        context=context,
        artifacts=artifacts,
        config_json=config_json,
        config_sha256=config_sha256,
    )
    payload.pop("_orbital_analysis_review", None)
    if review_outputs:
        summary["review_outputs"] = review_outputs
        summary["review_sqlite_path"] = str(review_outputs.get("sqlite") or "")
        payload["review_outputs"] = review_outputs
        artifacts["review_store"] = review_outputs
    else:
        artifacts.pop("review_store", None)
    index_path = write_output_index(
        outdir=output_dir,
        workflow="single_run",
        title=str(context.cfg.scenario_name or "single_run"),
        summary=summary,
        artifacts=artifacts,
    )
    summary["output_index_md"] = str(index_path)
    payload["output_index_md"] = str(index_path)
    artifacts["output_index_md"] = str(index_path)
    if bool(context.cfg.outputs.stats.get("save_json", True)):
        write_json(str(output_dir / "master_run_summary.json"), summary)
    if bool(context.cfg.outputs.stats.get("save_full_log", True)):
        write_json(str(output_dir / "master_run_log.json"), payload)
    if bool(context.cfg.outputs.stats.get("print_summary", True)):
        print(format_single_run_summary(summary))
    _write_owned_artifact_inventory(output_dir, _artifact_file_paths(artifacts))
    return payload


def _artifact_file_paths(value: Any) -> set[Path]:
    paths: set[Path] = set()
    if isinstance(value, dict):
        for child in value.values():
            paths.update(_artifact_file_paths(child))
    elif isinstance(value, (list, tuple)):
        for child in value:
            paths.update(_artifact_file_paths(child))
    elif isinstance(value, (str, Path)) and str(value).strip():
        paths.add(Path(value).expanduser().resolve())
    return paths


def _previous_owned_artifacts(output_dir: Path) -> set[Path]:
    # Fixed names are always owned. Dynamic names are eligible only when the
    # prior inventory's digest still matches a regular file under this output.
    relative_paths = (
        "index.md",
        "master_run_summary.json",
        "master_run_log.json",
        "master_run_history.npz",
        "master_run_history_manifest.json",
        "review/run.sqlite",
        "review/schema.json",
        "review/saved_views.json",
        "review/workflow_manifest.json",
        "review/generated_artifacts.json",
    )
    owned = {output_dir / relative for relative in relative_paths if (output_dir / relative).is_file()}
    fixed_owned = {path.resolve() for path in owned}
    inventory_path = output_dir / ".oel_run_artifacts.json"
    try:
        inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError, TypeError):
        return owned
    root = output_dir.resolve()
    for row in list(dict(inventory or {}).get("paths", []) or []):
        item = dict(row or {})
        relative = Path(str(item.get("path", "") or ""))
        expected = str(item.get("sha256", "") or "")
        if relative.is_absolute() or not relative.parts or ".." in relative.parts or not expected:
            continue
        candidate = root / relative
        if candidate.resolve() in fixed_owned:
            # Fixed artifacts are already unconditionally owned above; hashing
            # them again can reread hundreds of megabytes on every rerun.
            continue
        if candidate.is_file() and not candidate.is_symlink():
            actual = sha256_file(candidate)
            if actual == expected:
                owned.add(candidate)
    return owned


def _prune_stale_owned_artifacts(*, output_dir: Path, previous: set[Path], current: set[Path]) -> None:
    root = output_dir.resolve()
    for path in sorted(previous - current):
        try:
            relative = path.absolute().relative_to(root)
        except ValueError:
            continue
        if path.name != ".oel_run_artifacts.json":
            _unlink_regular_child_nofollow(root, relative)


def _unlink_regular_child_nofollow(root: Path, relative: Path) -> None:
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        return
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptors: list[int] = []
    try:
        descriptor = os.open(root, flags)
        descriptors.append(descriptor)
        for part in relative.parts[:-1]:
            descriptor = os.open(part, flags, dir_fd=descriptor)
            descriptors.append(descriptor)
        stat_result = os.stat(relative.name, dir_fd=descriptor, follow_symlinks=False)
        if stat.S_ISREG(stat_result.st_mode):
            os.unlink(relative.name, dir_fd=descriptor)
    except (FileNotFoundError, NotADirectoryError, OSError):
        return
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def _write_owned_artifact_inventory(output_dir: Path, paths: set[Path]) -> None:
    writer = current_output_writer()
    logical_output_dir = _writer_logical_path(output_dir, writer)
    inventory = logical_output_dir / ".oel_run_artifacts.json"
    root = logical_output_dir
    owned = []
    for path in sorted(paths):
        try:
            logical_path = _writer_logical_path(path, writer)
            relative = logical_path.relative_to(root)
        except ValueError:
            continue
        if writer is None:
            if logical_path.is_file() and not logical_path.is_symlink():
                owned.append({"path": relative.as_posix(), "sha256": sha256_file(logical_path)})
        else:
            try:
                writer_relative = logical_path.relative_to(Path(writer.logical_root))
                digest = writer.sha256(writer_relative)
            except (FileNotFoundError, OSError, ValueError):
                continue
            owned.append({"path": relative.as_posix(), "sha256": digest})
    serialized = json.dumps({"version": 2, "paths": owned}, indent=2) + "\n"
    if writer is None:
        inventory.parent.mkdir(parents=True, exist_ok=True)
        inventory.write_text(serialized, encoding="utf-8")
    else:
        with staged_output_file(inventory) as staged:
            staged.write_text(serialized, encoding="utf-8")


def _write_history_csv(*, context: SingleRunArtifactContext) -> Path:
    active_writer = current_output_writer()
    path = _writer_logical_path(context.outdir / "master_run_history.csv", active_writer)
    state_columns = (
        "x_eci_km", "y_eci_km", "z_eci_km", "vx_eci_km_s", "vy_eci_km_s", "vz_eci_km_s",
        "quat_w", "quat_x", "quat_y", "quat_z", "omega_x_body_rad_s", "omega_y_body_rad_s",
        "omega_z_body_rad_s", "mass_kg",
    )
    fieldnames = ["sample_index", "time_s", "object_id", *state_columns]
    stream_context = staged_output_file(path) if active_writer is not None else nullcontext(path)
    with stream_context as write_path:
        with write_path.open("w", encoding="utf-8", newline="") as stream:
            csv_writer = csv.DictWriter(stream, fieldnames=fieldnames)
            csv_writer.writeheader()
            times = np.asarray(context.t_s, dtype=float).reshape(-1)
            for object_id, raw in sorted(context.truth_hist.items()):
                history = np.asarray(raw, dtype=float)
                if history.ndim != 2:
                    continue
                for sample_index in range(min(times.size, history.shape[0])):
                    row: dict[str, Any] = {
                        "sample_index": sample_index,
                        "time_s": float(times[sample_index]),
                        "object_id": object_id,
                    }
                    row.update(
                        {
                            column: float(history[sample_index, index])
                            for index, column in enumerate(state_columns[: history.shape[1]])
                        }
                    )
                    csv_writer.writerow(row)
    return path


def _write_history_npz(*, context: SingleRunArtifactContext) -> dict[str, Any]:
    active_writer = current_output_writer()
    path = _writer_logical_path(context.outdir / "master_run_history.npz", active_writer)
    arrays: dict[str, np.ndarray] = {"time_s": np.asarray(context.t_s, dtype=float)}
    manifest: dict[str, Any] = {
        "format": "oel.single_run_history_npz",
        "version": 1,
        "arrays": {"time_s": {"kind": "time", "path": "time_s"}},
    }

    def add_array(key: str, arr: np.ndarray, *, kind: str, path_label: str) -> None:
        if key in arrays or key in manifest["arrays"]:
            raise ValueError(f"history NPZ key collision for {path_label!r}: {key!r}")
        arrays[key] = np.asarray(arr)
        manifest["arrays"][key] = {"kind": kind, "path": path_label}

    for object_id, arr in sorted(context.truth_hist.items()):
        add_array(
            _history_npz_key("truth", object_id),
            arr,
            kind="truth",
            path_label=f"truth_by_object.{object_id}",
        )
    if context.target_reference_orbit_truth is not None:
        add_array(
            "target_reference_orbit_truth",
            context.target_reference_orbit_truth,
            kind="target_reference_orbit_truth",
            path_label="target_reference_orbit_truth",
        )
    for object_id, arr in sorted(context.belief_hist.items()):
        add_array(
            _history_npz_key("belief", object_id),
            arr,
            kind="belief",
            path_label=f"belief_by_object.{object_id}",
        )
    for object_id, arr in sorted(context.thrust_hist.items()):
        add_array(
            _history_npz_key("applied_thrust", object_id),
            arr,
            kind="applied_thrust",
            path_label=f"applied_thrust_by_object.{object_id}",
        )
    for object_id, arr in sorted(context.torque_hist.items()):
        add_array(
            _history_npz_key("applied_torque", object_id),
            arr,
            kind="applied_torque",
            path_label=f"applied_torque_by_object.{object_id}",
        )
    for object_id, arr in sorted(context.desired_attitude_hist.items()):
        add_array(
            _history_npz_key("desired_attitude", object_id),
            arr,
            kind="desired_attitude",
            path_label=f"desired_attitude_by_object.{object_id}",
        )
    for observer_id, by_target in sorted(context.knowledge_hist.items()):
        for target_id, arr in sorted(by_target.items()):
            add_array(
                _history_npz_key("knowledge", observer_id, target_id),
                arr,
                kind="knowledge",
                path_label=f"knowledge_by_observer.{observer_id}.{target_id}",
            )
    for observer_id, by_target in sorted(context.knowledge_measurement_hist.items()):
        for target_id, arr in sorted(by_target.items()):
            add_array(
                _history_npz_key("knowledge_measurement", observer_id, target_id),
                arr,
                kind="knowledge_measurement",
                path_label=f"knowledge_measurements_by_observer.{observer_id}.{target_id}",
            )
    for metric_name, arr in sorted(context.rocket_metrics.items()):
        add_array(
            _history_npz_key("rocket_metric", metric_name),
            arr,
            kind="rocket_metric",
            path_label=f"rocket_metrics.{metric_name}",
        )
    for object_id, metrics in sorted(context.reentry_metrics.items()):
        for metric_name, arr in sorted(metrics.items()):
            add_array(
                _history_npz_key("reentry_metric", object_id, metric_name),
                arr,
                kind="reentry_metric",
                path_label=f"reentry_metrics_by_object.{object_id}.{metric_name}",
            )

    manifest_json = json.dumps(manifest, sort_keys=True)
    arrays["manifest_json"] = np.asarray(manifest_json)
    if active_writer is None:
        np.savez_compressed(path, **arrays)
    else:
        with staged_output_file(path) as staged:
            np.savez_compressed(staged, **arrays)
    return {
        "npz": str(path),
        "format": manifest["format"],
        "version": manifest["version"],
        "array_count": len(manifest["arrays"]),
    }


def _history_npz_key(prefix: str, *parts: str) -> str:
    raw = "__".join([prefix, *[str(part) for part in parts]])
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", raw)


def _write_review_store(
    *,
    payload: dict[str, Any],
    context: SingleRunArtifactContext,
    artifacts: dict[str, Any],
    config_json: str,
    config_sha256: str,
) -> dict[str, str]:
    review_cfg = context.cfg.outputs.review
    if not bool(review_cfg.enabled):
        review_dir = context.outdir / "review"
        if current_output_writer() is not None:
            return {}
        for name in ("run.sqlite", "run.sqlite.tmp", "schema.json"):
            path = review_dir / name
            if path.is_file() and not path.is_symlink():
                path.unlink()
        return {}
    try:
        outputs = write_single_run_review_store(
            payload=payload,
            context=context,
            artifacts=artifacts,
            config_json=config_json,
            config_sha256=config_sha256,
        )
        payload.setdefault("summary", {})["review_store_status"] = "complete" if outputs else "missing"
        return outputs
    except Exception as exc:
        if bool(review_cfg.strict):
            raise
        staged_db = context.outdir / "review" / "run.sqlite.tmp"
        if current_output_writer() is None and staged_db.is_file() and not staged_db.is_symlink():
            staged_db.unlink()
        status = f"failed:{type(exc).__name__}: {exc}"
        payload.setdefault("summary", {})["review_store_status"] = status
        return {}


def _write_mission_recovery_trade_space_plot(
    *,
    cfg: SimulationScenarioConfig,
    mission_recovery: dict[str, Any],
    outdir: Path,
) -> str | None:
    if not mission_recovery or not bool(cfg.outputs.plots.get("enabled", True)):
        return None
    figure_ids = _expanded_plot_figure_ids(dict(cfg.outputs.plots or {}))
    if "mission_recovery_trade_space" not in figure_ids:
        return None
    from sim.reporting.mission_recovery import write_mission_recovery_trade_space_plot

    return write_mission_recovery_trade_space_plot(
        mission_recovery=mission_recovery,
        outdir=outdir,
        mode=str(cfg.outputs.mode or "save"),
        dpi=int(cfg.outputs.plots.get("dpi", 150)),
    )


def _write_private_bridge_artifacts(
    *,
    outdir: Path,
    bridge_hist: dict[str, list[dict[str, Any]]],
) -> tuple[dict[str, str], dict[str, Any]]:
    if not bridge_hist:
        return {}, {}
    module_name = ".".join(("integrations", "c" + "f" + "s" + "_sil", "artifacts"))
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError:
        return {}, {}
    writer = getattr(module, "write_bridge_artifacts", None)
    if not callable(writer):
        return {}, {}
    return writer(outdir=outdir, bridge_hist=bridge_hist)


def _add_relative_range_summary(*, summary: dict[str, Any], context: SingleRunArtifactContext) -> None:
    pair = [str(item) for item in list(summary.get("primary_object_pair", []) or [])]
    if len(pair) != 2:
        return
    deputy_id, chief_id = pair
    deputy_frame = str(context.object_state_frames.get(deputy_id, "eci") or "eci").strip().lower()
    chief_frame = str(context.object_state_frames.get(chief_id, "eci") or "eci").strip().lower()
    if deputy_frame != chief_frame:
        return
    deputy = context.truth_hist.get(deputy_id)
    chief = context.truth_hist.get(chief_id)
    if deputy is None or chief is None or deputy.ndim != 2 or chief.ndim != 2:
        return
    n = int(min(deputy.shape[0], chief.shape[0], context.t_s.size))
    if n <= 0 or deputy.shape[1] < 3 or chief.shape[1] < 3:
        return
    rel = np.asarray(deputy[:n, :3], dtype=float) - np.asarray(chief[:n, :3], dtype=float)
    ranges = np.linalg.norm(rel, axis=1)
    finite = np.isfinite(ranges)
    if not bool(np.any(finite)):
        return
    finite_indices = np.flatnonzero(finite)
    closest_local = int(np.argmin(ranges[finite]))
    closest_idx = int(finite_indices[closest_local])
    final_idx = int(finite_indices[-1])
    initial_idx = int(finite_indices[0])
    summary["relative_range_summary"] = {
        "object_pair": [deputy_id, chief_id],
        "initial_range_km": float(ranges[initial_idx]),
        "final_range_km": float(ranges[final_idx]),
        "closest_approach_km": float(ranges[closest_idx]),
        "closest_approach_time_s": float(context.t_s[closest_idx]),
    }


def _plot_outputs(
    *,
    cfg: SimulationScenarioConfig,
    t_s: np.ndarray,
    truth_hist: dict[str, np.ndarray],
    target_reference_orbit_truth: np.ndarray | None,
    belief_hist: dict[str, np.ndarray] | None,
    thrust_hist: dict[str, np.ndarray],
    desired_attitude_hist: dict[str, np.ndarray] | None,
    knowledge_hist: dict[str, dict[str, np.ndarray]],
    knowledge_measurement_hist: dict[str, dict[str, np.ndarray]],
    rocket_metrics: dict[str, np.ndarray] | None,
    reentry_metrics: dict[str, dict[str, np.ndarray]] | None,
    bridge_hist: dict[str, list[dict[str, Any]]] | None,
    ground_station_access: dict[str, Any] | None,
    outdir: Path,
) -> dict[str, str]:
    return _plot_outputs_impl(
        cfg=cfg,
        t_s=t_s,
        truth_hist=truth_hist,
        target_reference_orbit_truth=target_reference_orbit_truth,
        thrust_hist=thrust_hist,
        belief_hist=belief_hist,
        desired_attitude_hist=desired_attitude_hist,
        knowledge_hist=knowledge_hist,
        knowledge_measurement_hist=knowledge_measurement_hist,
        rocket_metrics=rocket_metrics,
        reentry_metrics=reentry_metrics or {},
        bridge_hist=bridge_hist,
        ground_station_access=ground_station_access,
        outdir=outdir,
        resolve_rocket_stack=_resolve_rocket_stack,
        resolve_satellite_isp_s=_resolve_satellite_isp_s,
    )


def _animate_outputs(
    *,
    cfg: SimulationScenarioConfig,
    t_s: np.ndarray,
    truth_hist: dict[str, np.ndarray],
    thrust_hist: dict[str, np.ndarray],
    target_reference_orbit_truth: np.ndarray | None,
    outdir: Path,
) -> dict[str, str]:
    return _animate_outputs_impl(
        cfg=cfg,
        t_s=t_s,
        truth_hist=truth_hist,
        thrust_hist=thrust_hist,
        target_reference_orbit_truth=target_reference_orbit_truth,
        outdir=outdir,
        resolve_satellite_isp_s=_resolve_satellite_isp_s,
    )


def _fmt_float(x: float, digits: int = 3) -> str:
    return f"{float(x):.{digits}f}"


def _fmt_optional_float(x: Any, digits: int = 3) -> str:
    if x is None:
        return "n/a"
    value = float(x)
    if value != 0.0 and abs(value) < 10.0 ** (-digits):
        return f"{value:.3g}"
    return _fmt_float(value, digits)
